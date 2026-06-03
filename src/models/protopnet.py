"""ProtoPNet — "This Looks Like That" (Chen et al., NeurIPS 2019).

Member B's implementation. Conforms to the interfaces defined in
``src/models/base_model.py`` and ``src/trainer.py`` and to the output spec in
``docs/PROTOTYPE_METHODS_DETAILS.md``.

Layout::

    image -> backbone -> (B, 512, 7, 7) -> add-on convs -> (B, D, 7, 7)
          -> prototype layer (L2 distances) -> similarities -> FC -> logits

The prototype layer holds ``num_prototypes_per_class * num_classes`` prototype
vectors of dimension ``D``. Each prototype belongs to exactly one class. The FC
classifier is constrained so a class is driven positively by its own prototypes
and negatively by the others.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import PrototypeModel, build_backbone
from src.trainer import Trainer

# Numerical floor used when converting distances to similarities.
SIMILARITY_EPS = 1e-4
# Constrained-classifier initialisation strengths (Chen et al., 2019).
CORRECT_CLASS_CONNECTION = 1.0
INCORRECT_CLASS_CONNECTION = -0.5
# Default loss coefficients from the original paper.
DEFAULT_CLUSTER_COEF = 0.8
DEFAULT_SEPARATION_COEF = -0.08
# L1 sparsity coefficient on non-class FC connections (Chen et al. coefs['l1']).
DEFAULT_L1_COEF = 1e-4


class ProtoPNet(PrototypeModel):
    """Class-specific prototype network with a constrained linear head."""

    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 200,
        num_prototypes_per_class: int = 10,
        prototype_dim: int = 128,
        cluster_coef: float = DEFAULT_CLUSTER_COEF,
        separation_coef: float = DEFAULT_SEPARATION_COEF,
        l1_coef: float = DEFAULT_L1_COEF,
        backbone: nn.Module | None = None,
        feature_dim: int | None = None,
    ) -> None:
        if backbone is None:
            backbone, feature_dim = build_backbone(backbone_name)
        elif feature_dim is None:
            raise ValueError("feature_dim must be given when a backbone is injected")

        super().__init__(backbone, num_classes)

        self.feature_dim = feature_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes_per_class * num_classes
        self.cluster_coef = cluster_coef
        self.separation_coef = separation_coef
        # L1 penalty on non-class FC weights — applied in EVERY phase, matching
        # the original _train_or_test (coefs['l1'] * l1 is always in the loss).
        self.l1_coef = l1_coef

        # Add-on layers project the backbone feature map down to prototype_dim
        # and squash into (0, 1) — keeps L2 distances bounded by prototype_dim.
        # This is the original's 'regular' add_on_layers_type.
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(feature_dim, prototype_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(prototype_dim, prototype_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self._init_add_on_weights()

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.num_prototypes, prototype_dim, 1, 1)
        )

        # (P, num_classes) one-hot: which class each prototype belongs to.
        identity = torch.zeros(self.num_prototypes, num_classes)
        for proto_idx in range(self.num_prototypes):
            identity[proto_idx, proto_idx // num_prototypes_per_class] = 1.0
        self.register_buffer("prototype_class_identity", identity)
        # Retained for checkpoint compatibility; no longer used in the forward
        # path (the ||z||^2 term is now a direct channel-sum, see _distances).
        self.register_buffer("_ones", torch.ones_like(self.prototype_vectors))
        # Checkpointed source metadata for visualisation. ``-1`` means the
        # prototype has not been pushed yet. The Python list below is kept as a
        # convenient public representation for notebook utilities.
        self.register_buffer(
            "prototype_source_indices",
            torch.full((self.num_prototypes,), -1, dtype=torch.long),
        )
        self.register_buffer(
            "prototype_source_rows",
            torch.full((self.num_prototypes,), -1, dtype=torch.long),
        )
        self.register_buffer(
            "prototype_source_cols",
            torch.full((self.num_prototypes,), -1, dtype=torch.long),
        )

        self.classifier = nn.Linear(self.num_prototypes, num_classes, bias=False)
        self._init_classifier_weights()

        # Populated by push_prototypes(): one (image_index, row, col) per prototype.
        self.prototype_source_info: list[tuple[int, int, int] | None] = [
            None
        ] * self.num_prototypes

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights while accepting older checkpoints without source buffers."""
        source_keys = {
            "prototype_source_indices",
            "prototype_source_rows",
            "prototype_source_cols",
        }
        if strict and not source_keys <= set(state_dict):
            state_dict = dict(state_dict)
            state_dict.setdefault(
                "prototype_source_indices",
                torch.full((self.num_prototypes,), -1, dtype=torch.long),
            )
            state_dict.setdefault(
                "prototype_source_rows",
                torch.full((self.num_prototypes,), -1, dtype=torch.long),
            )
            state_dict.setdefault(
                "prototype_source_cols",
                torch.full((self.num_prototypes,), -1, dtype=torch.long),
            )
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._sync_source_info_from_buffers()
        return result

    def _sync_source_info_from_buffers(self) -> None:
        """Refresh ``prototype_source_info`` from checkpointed integer buffers."""
        indices = self.prototype_source_indices.detach().cpu().tolist()
        rows = self.prototype_source_rows.detach().cpu().tolist()
        cols = self.prototype_source_cols.detach().cpu().tolist()
        self.prototype_source_info = [
            None if index < 0 else (int(index), int(row), int(col))
            for index, row, col in zip(indices, rows, cols)
        ]

    def _store_source_info(self, sources: list[tuple[int, int, int] | None]) -> None:
        """Update both the public source list and checkpointed source buffers."""
        self.prototype_source_info = sources
        device = self.prototype_source_indices.device
        indices = torch.full((self.num_prototypes,), -1, dtype=torch.long, device=device)
        rows = torch.full((self.num_prototypes,), -1, dtype=torch.long, device=device)
        cols = torch.full((self.num_prototypes,), -1, dtype=torch.long, device=device)
        for proto_idx, source in enumerate(sources):
            if source is None:
                continue
            index, row, col = source
            indices[proto_idx] = int(index)
            rows[proto_idx] = int(row)
            cols[proto_idx] = int(col)
        self.prototype_source_indices.copy_(indices)
        self.prototype_source_rows.copy_(rows)
        self.prototype_source_cols.copy_(cols)

    # ----------------------------------------------------------------- setup
    def _init_add_on_weights(self) -> None:
        """Kaiming-normal init for the add-on convs (Chen et al. _initialize_weights).

        The original initialises every add-on ``Conv2d`` with
        ``kaiming_normal_(mode='fan_out', nonlinearity='relu')`` and zeroes the
        bias. PyTorch's default ``Conv2d`` init (kaiming-uniform) differs, so we
        set it explicitly to match the reference implementation.
        """
        for module in self.add_on_layers.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _init_classifier_weights(self) -> None:
        """+1 for a class's own prototypes, -0.5 for the rest."""
        positive = self.prototype_class_identity.t()  # (num_classes, P)
        weight = (
            CORRECT_CLASS_CONNECTION * positive
            + INCORRECT_CLASS_CONNECTION * (1.0 - positive)
        )
        self.classifier.weight.data.copy_(weight)

    # ------------------------------------------------------------- internals
    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + add-on layers -> (B, prototype_dim, H, W)."""
        return self.add_on_layers(self.backbone(x))

    def _distances(self, conv_features: torch.Tensor) -> torch.Tensor:
        """Squared L2 distance prototype<->patch -> (B, P, H, W).

        Uses ||z - p||^2 = ||z||^2 - 2 z.p + ||p||^2. The original computes the
        ||z||^2 term with a 1x1 conv against an all-ones ``(P, D, 1, 1)`` filter,
        which yields the same channel-sum **replicated P times** — a full
        ``(B, P, H, W)`` tensor whose P channels are identical. We instead sum
        over channels once into ``(B, 1, H, W)`` and let broadcasting fill the
        prototype dimension: mathematically identical, but it drops a P-way conv
        and a ``(B, P, H, W)`` allocation from the hot path (a real saving with
        P=2000 prototypes, on every forward pass and every push).
        """
        patch_sq = (conv_features**2).sum(dim=1, keepdim=True)  # (B, 1, H, W); sum_d z_d^2
        cross = F.conv2d(conv_features, self.prototype_vectors)  # (B, P, H, W); sum_d z_d p_d
        proto_sq = (self.prototype_vectors**2).sum(dim=(1, 2, 3)).view(1, -1, 1, 1)
        return F.relu(patch_sq - 2 * cross + proto_sq)  # broadcasts to (B, P, H, W)

    @staticmethod
    def _distance_to_similarity(distances: torch.Tensor) -> torch.Tensor:
        """log((d + 1) / (d + eps)) — monotonically decreasing in distance."""
        return torch.log((distances + 1) / (distances + SIMILARITY_EPS))

    def _logits_and_min_distances(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning both logits and per-prototype min L2.

        The trainer needs ``min_distances`` for the cluster/separation losses;
        exposing both here avoids recomputing the backbone pass.
        """
        conv_features = self._conv_features(x)
        distances = self._distances(conv_features)
        spatial = distances.shape[-2:]
        min_distances = -F.max_pool2d(-distances, kernel_size=spatial)
        min_distances = min_distances.flatten(1)  # (B, P)
        similarities = self._distance_to_similarity(min_distances)
        logits = self.classifier(similarities)
        # Cache for the shared Trainer, whose default training_step calls
        # compute_loss(logits, labels) without features: the cluster/separation
        # terms read this instead of forcing a second backbone pass.
        self._cached_min_distances = min_distances
        return logits, min_distances

    # --------------------------------------------------------------- BaseModel
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self._logits_and_min_distances(x)
        return logits

    def explain(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            conv_features = self._conv_features(x)
            distances = self._distances(conv_features)  # (B, P, H, W)
            width = distances.shape[-1]

            activation_maps = self._distance_to_similarity(distances)
            min_distances = distances.flatten(2).min(dim=2).values  # (B, P)
            similarities = self._distance_to_similarity(min_distances)

            argmin = distances.flatten(2).argmin(dim=2)  # (B, P)
            rows = torch.div(argmin, width, rounding_mode="floor")
            cols = argmin % width
            patch_locations = torch.stack([rows, cols], dim=-1)  # (B, P, 2)

        return {
            "prototype_similarities": similarities,
            "patch_locations": patch_locations,
            "activation_maps": activation_maps,
        }

    # ----------------------------------------------------------- PrototypeModel
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full ProtoPNet loss (Chen et al. ``_train_or_test``).

        ``total = cls + clst*cluster + sep*separation + l1*L1`` — the L1 sparsity
        term on non-class FC connections is part of the objective in **every**
        phase (warm/joint/last), exactly as the reference implementation always
        adds ``coefs['l1'] * l1``. ``avg_separation`` (mean distance to
        wrong-class prototypes) is reported for monitoring but not optimised.

        ``features`` carries the per-prototype ``min_distances`` tensor of shape
        ``(B, P)`` produced by :meth:`_logits_and_min_distances` — the positional
        third argument mandated by :class:`PrototypeModel`. When omitted (the
        shared Trainer's default ``training_step``) we fall back to the
        ``min_distances`` cached by the most recent forward pass, so the
        cluster/separation terms still apply. Only when no cache exists do those
        terms collapse to zero (the L1 + cross-entropy terms still apply).
        """
        cls_loss = F.cross_entropy(logits, labels)
        # raw L1 norm (logged); the scaled term enters the total below.
        l1 = self.last_layer_l1()

        if features is None:
            features = getattr(self, "_cached_min_distances", None)
        if features is None:
            zero = torch.zeros((), device=logits.device)
            return {
                "total": cls_loss + self.l1_coef * l1,
                "cls": cls_loss,
                "cluster": zero,
                "separation": zero,
                "avg_separation": zero,
                "l1": l1,
            }

        min_distances = features  # (B, P)
        max_dist = float(self.prototype_dim)
        # (B, P): 1 where the prototype belongs to the sample's true class.
        correct = self.prototype_class_identity[:, labels].t()
        wrong = 1.0 - correct

        # Cluster: pull each sample close to its nearest own-class prototype.
        inv_correct = torch.max((max_dist - min_distances) * correct, dim=1).values
        cluster = torch.mean(max_dist - inv_correct)

        # Separation: distance to the nearest wrong-class prototype (pushed up
        # via the negative coefficient).
        inv_wrong = torch.max((max_dist - min_distances) * wrong, dim=1).values
        separation = torch.mean(max_dist - inv_wrong)

        # Avg separation: mean distance to ALL wrong-class prototypes (logged
        # only) — matches the original avg_separation_cost.
        avg_separation = torch.mean(
            torch.sum(min_distances * wrong, dim=1) / torch.sum(wrong, dim=1)
        )

        total = (
            cls_loss
            + self.cluster_coef * cluster
            + self.separation_coef * separation
            + self.l1_coef * l1
        )
        return {
            "total": total,
            "cls": cls_loss,
            "cluster": cluster,
            "separation": separation,
            "avg_separation": avg_separation,
            "l1": l1,
        }

    @torch.no_grad()
    def push_prototypes(self, train_loader, device: str | torch.device | None = None) -> None:
        """Anchor every prototype to the nearest same-class training patch.

        Scans the whole loader; for each prototype keeps the training patch
        embedding with the smallest L2 distance and copies it into
        ``prototype_vectors``. Source coordinates are cached in
        ``prototype_source_info`` for visualisation.

        ``device`` is part of the :class:`PrototypeModel` contract (the shared
        Trainer passes ``self.device``); when omitted it defaults to the
        prototype tensor's own device.
        """
        self.eval()
        if device is None:
            device = self.prototype_vectors.device
        else:
            self.to(device)
        dim = self.prototype_dim

        global_min = torch.full((self.num_prototypes,), float("inf"), device=device)
        best_vectors = self.prototype_vectors.detach().clone()
        sources: list[tuple[int, int, int] | None] = list(self.prototype_source_info)

        # Precompute the prototype indices owned by each class.
        class_to_protos = [
            torch.nonzero(self.prototype_class_identity[:, c], as_tuple=True)[0]
            for c in range(self.num_classes)
        ]

        image_offset = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            conv = self._conv_features(images)  # (B, D, H, W)
            distances = self._distances(conv)  # (B, P, H, W)
            width = conv.shape[-1]

            for cls in range(self.num_classes):
                proto_idx = class_to_protos[cls]
                if proto_idx.numel() == 0:
                    continue
                mask = labels == cls
                if not bool(mask.any()):
                    continue

                conv_c = conv[mask]  # (n, D, H, W)
                dist_c = distances[mask][:, proto_idx]  # (n, ppc, H, W)
                n_imgs = conv_c.shape[0]
                dist_flat = dist_c.reshape(n_imgs, proto_idx.numel(), -1)

                min_spatial, argmin_spatial = dist_flat.min(dim=2)  # (n, ppc)
                min_img, argmin_img = min_spatial.min(dim=0)  # (ppc,)

                orig_positions = torch.nonzero(mask, as_tuple=True)[0]
                for k, proto in enumerate(proto_idx.tolist()):
                    if min_img[k] >= global_min[proto]:
                        continue
                    global_min[proto] = min_img[k]
                    img_in_subset = int(argmin_img[k])
                    location = int(argmin_spatial[img_in_subset, k])
                    row, col = divmod(location, width)
                    best_vectors[proto] = conv_c[img_in_subset, :, row, col].view(dim, 1, 1)
                    sources[proto] = (
                        image_offset + int(orig_positions[img_in_subset]),
                        row,
                        col,
                    )

            image_offset += images.shape[0]

        self.prototype_vectors.data.copy_(best_vectors)
        self._store_source_info(sources)

    def get_prototype_params(self):
        """Add-on layers + prototype vectors (trained in warm/joint phases)."""
        return list(self.add_on_layers.parameters()) + [self.prototype_vectors]

    def get_classifier_params(self):
        """FC head parameters (trained only in the last phase)."""
        return list(self.classifier.parameters())

    def last_layer_l1(self) -> torch.Tensor:
        """L1 of the non-class FC connections (paper's last-layer sparsity term).

        Penalises ``w(k, j)`` for prototypes ``j`` that do NOT belong to class
        ``k`` so the model relies less on negative reasoning ("not class k").
        """
        non_class = 1.0 - self.prototype_class_identity.t()  # (num_classes, P)
        return (self.classifier.weight * non_class).abs().sum()

    # --------------------------------------------------------------- phasing
    @staticmethod
    def _set_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def set_phase(self, phase: str) -> None:
        """Toggle ``requires_grad`` for the three training phases."""
        if phase == "warm":
            self._set_grad(self.backbone, False)
            self._set_grad(self.add_on_layers, True)
            self.prototype_vectors.requires_grad_(True)
            self._set_grad(self.classifier, False)
        elif phase == "joint":
            self._set_grad(self.backbone, True)
            self._set_grad(self.add_on_layers, True)
            self.prototype_vectors.requires_grad_(True)
            self._set_grad(self.classifier, False)
        elif phase == "last":
            self._set_grad(self.backbone, False)
            self._set_grad(self.add_on_layers, False)
            self.prototype_vectors.requires_grad_(False)
            self._set_grad(self.classifier, True)
        else:
            raise ValueError(f"Unknown phase {phase!r}. Use 'warm', 'joint', or 'last'.")


class ProtoPNetTrainer(Trainer):
    """Iterative ProtoPNet trainer following the paper's three cycled stages.

    Schedule (Chen et al. 2019, §2.2 — "cycle through these stages more than once"):

    1. **warm-up** — ``warm_epochs`` of joint SGD with the backbone frozen.
    2. **joint + periodic push** — ``joint_epochs`` of joint SGD; every
       ``push_interval`` epochs (and always at the end) the prototypes are
       **pushed** onto their nearest same-class patch and the **last layer is
       convex-optimised** (cross-entropy + L1 on non-class connections) for
       ``last_layer_iters`` epochs. Joint SGD then resumes, so the backbone
       re-adapts around the projected prototypes — this is what a single push
       cannot do.

    The model returned has **projected prototypes** (the last action is a push +
    last-layer optimisation) and is restored to the best post-push validation
    accuracy. Reuses the base :meth:`_train_epoch` / :meth:`validate`; the
    returned history is base-:class:`Trainer` compatible.
    """

    def __init__(
        self,
        model: ProtoPNet,
        device: str | torch.device,
        warm_epochs: int = 5,
        joint_epochs: int = 30,
        push_interval: int = 10,
        last_layer_iters: int = 20,
        warm_lr: float = 3e-3,
        joint_backbone_lr: float = 1e-4,
        joint_proto_lr: float = 3e-3,
        last_lr: float = 1e-4,
        joint_lr_step_size: int = 5,
        joint_lr_gamma: float = 0.1,
        l1_coef: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_fn: nn.Module | None = None,
    ) -> None:
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        # optimizer is rebuilt per stage inside train(); start with None.
        super().__init__(model, optimizer=None, loss_fn=loss_fn, device=device, scheduler=None)
        self.warm_epochs = warm_epochs
        self.joint_epochs = joint_epochs
        self.push_interval = push_interval
        self.last_layer_iters = last_layer_iters
        self.warm_lr = warm_lr
        self.joint_backbone_lr = joint_backbone_lr
        self.joint_proto_lr = joint_proto_lr
        self.last_lr = last_lr
        # StepLR on the joint optimizer (paper: step_size=5, gamma=0.1). Decaying
        # the joint LR is what keeps the iterative push stable — post-push joint
        # steps stay small instead of overshooting and diverging.
        self.joint_lr_step_size = joint_lr_step_size
        self.joint_lr_gamma = joint_lr_gamma
        self.l1_coef = l1_coef
        self.weight_decay = weight_decay
        # The loss (incl. its L1 term) lives on the model; keep the model's
        # coefficient in sync with this trainer's so both code paths agree.
        self.model.l1_coef = l1_coef

        self._current_phase = "warm"
        self.push_epochs: list[int] = []
        self.best_val_acc = 0.0
        self._best_state: dict | None = None
        # Set by train(); when given, the best model is also written here after
        # every improvement so a long run survives an early interruption.
        self._checkpoint_path: str | None = None

    # ----------------------------------------------------------- optimizers
    # Weight decay follows the original main.py: it regularises the backbone and
    # add-on convs but NOT the prototype vectors (free to move onto real patches)
    # nor the last layer (kept sparse by the L1 term instead).
    def _warm_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam([
            {"params": list(self.model.add_on_layers.parameters()),
             "lr": self.warm_lr, "weight_decay": self.weight_decay},
            {"params": [self.model.prototype_vectors],
             "lr": self.warm_lr, "weight_decay": 0.0},
        ])

    def _joint_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam([
            {"params": list(self.model.get_backbone_params()),
             "lr": self.joint_backbone_lr, "weight_decay": self.weight_decay},
            {"params": list(self.model.add_on_layers.parameters()),
             "lr": self.joint_proto_lr, "weight_decay": self.weight_decay},
            {"params": [self.model.prototype_vectors],
             "lr": self.joint_proto_lr, "weight_decay": 0.0},
        ])

    def _last_optimizer(self) -> torch.optim.Optimizer:
        # last layer kept sparse by L1, not weight decay
        return torch.optim.Adam(self.model.get_classifier_params(), lr=self.last_lr)

    # -------------------------------------------------------------- overrides
    def _set_phase(self, phase: str) -> None:
        self.model.set_phase(phase)
        self._current_phase = phase

    def training_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        # compute_loss already folds the L1 sparsity term into "total" in EVERY
        # phase (matching the original _train_or_test) and returns an "l1"
        # component for logging — no phase-specific re-add needed here.
        logits, min_distances = self.model._logits_and_min_distances(images)
        loss_dict = self.model.compute_loss(logits, labels, min_distances)
        loss_dict["logits"] = logits
        return loss_dict

    def push_prototypes(self, train_loader) -> None:
        self.model.push_prototypes(train_loader, self.device)

    # ------------------------------------------------------------------- loop
    def _train_one_epoch(self, label: str, train_loader, history: dict) -> None:
        """Run one epoch, appending its metrics to the base-format history."""
        metrics = self._train_epoch(train_loader)
        history["train_loss"].append(metrics.pop("total"))
        history["train_acc"].append(metrics.pop("acc"))
        for key, value in metrics.items():
            history.setdefault(key, []).append(value)

    def _validate_and_log(self, val_loader, history: dict, tag: str) -> float:
        # x-axis is the running count of recorded training epochs, so train and
        # val curves (and push markers) all share one coherent axis.
        step = len(history["train_loss"])
        val = self.validate(val_loader)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_epochs"].append(step)
        print(
            f"  [{tag}] step {step}: train_loss={history['train_loss'][-1]:.4f} "
            f"train_acc={history['train_acc'][-1]:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}"
        )
        return val["acc"]

    def _optimize_last_layer(self, train_loader, history: dict) -> None:
        self._set_phase("last")
        self.optimizer = self._last_optimizer()
        for _ in range(self.last_layer_iters):
            self._train_one_epoch("last", train_loader, history)

    def _snapshot_if_best(self, val_acc: float, history: dict) -> None:
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            # Persist immediately: the model is at its best right now (we just
            # validated it post-push), so the on-disk checkpoint is always the
            # best seen so far and an interrupted run stays recoverable.
            if self._checkpoint_path is not None:
                self.save_checkpoint(
                    self._checkpoint_path, len(history["train_loss"]), history
                )

    def _save_latest(self, history: dict) -> None:
        """Write a periodic recovery checkpoint next to the best one.

        Used during long joint stretches (e.g. single-cycle runs, where the only
        push — and therefore the only best-checkpoint — is at the very end). The
        file may hold **un-pushed** prototypes, so it is a recovery/resume artifact
        rather than the interpretable deliverable; run ``push_prototypes`` on it if
        you need the projected version. Path: ``<checkpoint>_latest<ext>``.
        """
        base, ext = os.path.splitext(self._checkpoint_path)
        self.save_checkpoint(
            f"{base}_latest{ext or '.pth'}", len(history["train_loss"]), history
        )

    def train(  # type: ignore[override]
        self,
        train_loader,
        val_loader,
        val_every: int = 1,
        push_loader=None,
        checkpoint_path: str | None = None,
        save_every: int | None = None,
        **_ignored,
    ) -> dict:
        """Run the iterative schedule. Returns a base-format history dict.

        ``push_loader`` should iterate the training images with **no augmentation**
        (deterministic resize + centre-crop) and **no shuffling**, so prototypes
        anchor to clean patches and the cached source locations reproduce for
        visualisation. Falls back to ``train_loader`` when not given.

        The phase schedule is governed by this trainer's own constructor
        (``warm_epochs`` / ``joint_epochs`` / ``push_interval``), so generic
        ``Trainer.train`` kwargs from the unified runner (``epochs``,
        ``push_epoch``, ``patience``) are accepted but ignored. When
        ``checkpoint_path`` is given the best post-push model is saved there
        **after every improvement during training** (not only at the end), so an
        interrupted long run leaves a usable best checkpoint on disk.

        ``save_every`` (joint epochs) additionally writes a ``*_latest`` recovery
        checkpoint at that cadence. This matters for single-cycle runs, where the
        only push — hence the only best-checkpoint — is at the very end: the
        ``*_latest`` file lets you recover mid-run (it may hold un-pushed
        prototypes; push it if you need the interpretable version).
        """
        push_loader = push_loader if push_loader is not None else train_loader
        self._checkpoint_path = checkpoint_path
        history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_epochs": [],
        }

        # --- Stage 1a: warm-up (backbone frozen) ---
        self._set_phase("warm")
        self.optimizer = self._warm_optimizer()
        for local in range(1, self.warm_epochs + 1):
            self._train_one_epoch("warm", train_loader, history)
            if local % val_every == 0:
                self._validate_and_log(val_loader, history, "warm")

        # epochs at which to push (always include the final joint epoch)
        push_at = set(range(self.push_interval, self.joint_epochs + 1, self.push_interval))
        if self.joint_epochs > 0:
            push_at.add(self.joint_epochs)

        # --- Stage 1b + 2 + 3: joint SGD with periodic push + last-layer opt ---
        joint_optimizer = self._joint_optimizer()
        joint_scheduler = torch.optim.lr_scheduler.StepLR(
            joint_optimizer, step_size=self.joint_lr_step_size, gamma=self.joint_lr_gamma
        )
        for local in range(1, self.joint_epochs + 1):
            self._set_phase("joint")
            self.optimizer = joint_optimizer
            self._train_one_epoch("joint", train_loader, history)
            joint_scheduler.step()  # decay joint LR -> stable post-push steps

            if local in push_at:
                print(f"\nPush + last-layer optimisation at joint epoch {local}...")
                self.push_prototypes(push_loader)
                self._optimize_last_layer(train_loader, history)
                self.push_epochs.append(len(history["train_loss"]))
                acc = self._validate_and_log(val_loader, history, "post-push")
                self._snapshot_if_best(acc, history)
            elif local % val_every == 0:
                self._validate_and_log(val_loader, history, "joint")

            # periodic recovery checkpoint (covers single-cycle runs, whose only
            # best-save is the final push)
            if save_every and self._checkpoint_path and local % save_every == 0:
                self._save_latest(history)

        # restore best post-push (projected) checkpoint
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            print(f"\nRestored best post-push model (val_acc={self.best_val_acc:.4f})")

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path, len(history["train_loss"]), history)

        return history


# --------------------------------------------------------------------- metrics
def count_trainable_params(model: nn.Module) -> int:
    """Number of parameters with ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def top_k_accuracy(
    model: nn.Module, loader, device: str | torch.device, k: int = 5
) -> float:
    """Top-k accuracy over a loader (default top-5, useful for fine-grained)."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        topk = logits.topk(k, dim=1).indices  # (B, k)
        correct += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


@torch.no_grad()
def mean_prototype_activation(
    model: ProtoPNet, loader, device: str | torch.device
) -> float:
    """Mean over the loader of each prototype's peak similarity per image."""
    model.eval()
    model.to(device)
    running = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        similarities = model.explain(images)["prototype_similarities"]  # (B, P)
        running += similarities.mean(dim=1).sum().item()
        total += images.size(0)
    return running / total if total else 0.0
