"""
TesNet — Transparent Embedding Space Network
Paper: "Interpretable Image Recognition by Constructing Transparent Embedding Space"
       Wang, Liu, Wang & Jing — ICCV 2021

Architecture:
  backbone  →  add_on_layers (bottleneck + sigmoid)  →  concept projection
           →  global max pool  →  linear classifier

Concept projection:
  - concept_vectors: (num_concepts_per_class × num_classes, concept_dim, 1, 1) explicit parameter
  - normalized before each conv2d so inner product = cosine similarity at each spatial location
  - global max pool picks the single best-matching location per concept across the feature map
  - each concept belongs to exactly one class (prototype_class_identity buffer); scales to any dataset

Loss (from settings_CUB.py / paper):
  total = CE(logits, y)
        + λ_clst  * cluster_loss          (pull features toward same-class concepts,    coef=0.8)
        − λ_sep   * sep_loss              (push features away from other-class concepts, coef=0.08)
        + λ_ortho * ||C_c C_c^T − I||²_F  (within-class orthogonality per class c,      coef=1e-4)
        + λ_l1    * ||W_cls||_1            (sparsity on classifier weights,              coef=1e-4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import PrototypeModel, build_backbone


class TesNet(PrototypeModel):
    """
    Wang et al. (ICCV 2021): "Interpretable Image Recognition by Constructing Transparent Embedding Space."

    Parameters
    ----------
    backbone_name : str
        "resnet34" or "vgg16".
    num_classes : int
        Number of output classes. Drives total concept count automatically.
    num_concepts_per_class : int
        Number of concept vectors per class.  Total = num_concepts_per_class × num_classes.
        Paper value: 10 (→ 2000 total for CUB-200).  Sweep: 5, 10, 20.
    concept_dim : int
        Dimension of concept space after bottleneck projection.  Paper value: 64.
    lambda_clst : float
        Cluster loss weight.  Paper value: 0.8.
    lambda_sep : float
        Separation loss weight (applied with negative sign).  Paper value: 0.08.
    lambda_ortho : float
        Within-class orthogonality regularization.  Paper value: 1e-4.
    lambda_l1 : float
        L1 sparsity on classifier weights.  Paper value: 1e-4.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 200,
        num_concepts_per_class: int = 10,
        concept_dim: int = 64,
        lambda_clst: float = 0.8,
        lambda_sep: float = 0.08,
        lambda_ortho: float = 1e-4,
        lambda_l1: float = 1e-4,
    ) -> None:
        backbone, feature_dim = build_backbone(backbone_name)
        super().__init__(backbone, num_classes)

        self.num_concepts_per_class = num_concepts_per_class
        self.num_concepts  = num_concepts_per_class * num_classes   # total concept count
        self.feature_dim   = feature_dim
        self.concept_dim   = concept_dim
        self.lambda_clst   = lambda_clst
        self.lambda_sep    = lambda_sep
        self.lambda_ortho  = lambda_ortho
        self.lambda_l1     = lambda_l1

        # prototype_class_identity[k, c] = 1  iff concept k belongs to class c.
        # Registered as a buffer so it moves with the model (to/from GPU) automatically
        class_identity = torch.zeros(self.num_concepts, num_classes)
        for k in range(self.num_concepts):
            class_identity[k, k // num_concepts_per_class] = 1
        self.register_buffer("prototype_class_identity", class_identity)

        # Bottleneck: compresses backbone features into bounded concept space
        # Sigmoid ensures activations are in [0, 1] before projection
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(feature_dim, concept_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(concept_dim, concept_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Concept vectors: shape (num_concepts, concept_dim, 1, 1)
        # Used as conv2d weights (after L2 normalization) to compute per-location
        # cosine similarities between feature maps and concept directions
        self.concept_vectors = nn.Parameter(
            torch.rand(self.num_concepts, concept_dim, 1, 1), requires_grad=True
        )

        # Linear classifier: concept scores -> class logits.  No bias so that all
        # class discrimination must flow through concept activations
        self.classifier = nn.Linear(self.num_concepts, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Per-class orthogonal init: within each class the K concepts start as an
        # orthogonal basis.  Different classes are initialized independently
        k = self.num_concepts_per_class
        for c in range(self.num_classes):
            s = c * k
            nn.init.orthogonal_(self.concept_vectors[s : s + k].view(k, -1))

        # Classifier init from paper: +1 for same-class concept connections,
        # -0.5 for cross-class, so the model starts with correct-class concepts contributing positively
        self._init_classifier(incorrect_strength=-0.5)

    def _init_classifier(self, incorrect_strength: float = -0.5) -> None:
        positive = self.prototype_class_identity.T          # (num_classes, num_concepts)
        negative = 1 - positive
        self.classifier.weight.data.copy_(
            1.0 * positive + incorrect_strength * negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_out = self.backbone(x)                          # (B, C, H, W)
        self._last_features = self.add_on_layers(backbone_out)   # (B, D, H, W)
        concept_scores = self._project(self._last_features)      # (B, num_concepts)
        return self.classifier(concept_scores)                   # (B, num_classes)

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        """Projection magnitude onto normalised concept basis → (B, num_concepts)."""
        norm_cv = F.normalize(self.concept_vectors, p=2, dim=1)
        proj    = F.conv2d(features, norm_cv)                    # (B, K, H, W)
        return F.adaptive_max_pool2d(proj, (1, 1)).flatten(1)   # (B, K)

    def _cosine_distances(self, features: torch.Tensor) -> torch.Tensor:
        """Cosine distance from every spatial location to each concept → (B, K, H, W)."""
        feat_norm = F.normalize(features, p=2, dim=1)
        conc_norm = F.normalize(self.concept_vectors, p=2, dim=1)
        return -F.conv2d(feat_norm, conc_norm)   # negate similarity → distance

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_loss   = F.cross_entropy(logits, labels)
        ortho_loss = self._orthogonality_loss()
        l1_loss    = self.classifier.weight.abs().mean()

        feats = features if features is not None else getattr(self, "_last_features", None)
        if feats is not None:
            cos_dists = self._cosine_distances(feats)            # (B, K, H, W)
            clst_loss, sep_loss = self._cluster_sep_loss(cos_dists, labels)
        else:
            clst_loss = sep_loss = torch.tensor(0.0, device=logits.device)

        total = (
            cls_loss
            + self.lambda_clst  * clst_loss
            - self.lambda_sep   * sep_loss    # negative: maximize distance to wrong-class concepts
            + self.lambda_ortho * ortho_loss
            + self.lambda_l1    * l1_loss
        )
        return {
            "total": total,
            "cls":   cls_loss,
            "clst":  clst_loss,
            "sep":   sep_loss,
            "ortho": ortho_loss,
        }

    def _cluster_sep_loss(
        self,
        cos_dists: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cos_dists : (B, K, H, W) — cosine distances to each concept at every location.
        labels    : (B,)         — ground-truth class indices.

        Returns
        -------
        clst_loss : mean of each image's min distance to its same-class concepts
                    (we want this small → pull toward correct-class concepts)
        sep_loss  : mean of each image's min distance to other-class concepts
                    (we want this large → applied with negative sign in total)
        """
        # Reduce spatial dimensions: min distance per concept over all locations -> (B, K)
        min_per_concept = cos_dists.flatten(2).min(dim=2).values

        # Boolean mask (B, K): True where concept k belongs to image i's class
        # prototype_class_identity is (K, num_classes); indexing by labels gives (K, B)
        same = self.prototype_class_identity[:, labels].T.bool()   # (B, K)

        large = torch.tensor(1e4, device=cos_dists.device)
        clst_loss = min_per_concept.where(same,  large).min(dim=1).values.mean()
        sep_loss  = min_per_concept.where(~same, large).min(dim=1).values.mean()
        return clst_loss, sep_loss

    def _orthogonality_loss(self) -> torch.Tensor:
        """
        Within-class orthogonality: for each class, its K concept vectors should
        form a mutually orthogonal set.  Computed via batched matrix multiply across
        all classes simultaneously — O(num_classes × K² × D).

        Cross-class orthogonality is intentionally NOT enforced: different classes
        can share low-level visual features (edges, colours) without harming
        interpretability.
        """
        k = self.num_concepts_per_class
        # Reshape to (num_classes, k, concept_dim) and row-normali`e
        C    = F.normalize(
            self.concept_vectors.view(self.num_classes, k, -1), p=2, dim=2
        )
        gram = torch.bmm(C, C.transpose(1, 2))                   # (num_classes, k, k)
        eye  = torch.eye(k, device=C.device).unsqueeze(0)         # (1, k, k)
        return (torch.norm(gram - eye, p="fro") ** 2) / self.num_classes

    def get_prototype_params(self):
        """add_on_layers + concept_vectors + classifier (no backbone)."""
        return (
            list(self.add_on_layers.parameters())
            + [self.concept_vectors]
            + list(self.classifier.parameters())
        )

    def class_concept_indices(self, class_idx: int) -> list[int]:
        """Returns the concept indices belonging to class_idx."""
        s = class_idx * self.num_concepts_per_class
        return list(range(s, s + self.num_concepts_per_class))

    # push_prototypes is intentionally a no-op for TesNet — concepts are
    # constrained via the orthogonality loss, not anchored to training patches

    def find_concept_exemplars(
        self,
        loader,
        device: str | torch.device,
        top_n: int = 5,
    ) -> dict[int, list[dict]]:
        """
        For each concept, find the top_n training images that activate it most.
        Run once after training to ground each concept in real images.
        Does NOT modify concept_vectors.

        Returns
        -------
        dict: concept_idx (int) → list of top_n dicts, each with:
            image       : (3, H, W) tensor
            score       : float   — concept activation for this image
            concept_map : (H', W') numpy array — spatial activation map
        """
        self.eval()
        self.to(device)

        all_scores: list[torch.Tensor] = []
        all_images: list[torch.Tensor] = []
        all_maps:   list[torch.Tensor] = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                out = self.explain(images)
                all_scores.append(out["concept_scores"].cpu())
                all_maps.append(out["concept_maps"].cpu())
                all_images.append(images.cpu())

        scores = torch.cat(all_scores)   # (N, K)
        maps   = torch.cat(all_maps)     # (N, K, H, W)
        images = torch.cat(all_images)   # (N, 3, H, W)

        exemplars: dict[int, list[dict]] = {}
        for k in range(self.num_concepts):
            top_idx = scores[:, k].topk(min(top_n, len(scores))).indices
            exemplars[k] = [
                {
                    "image":       images[i],
                    "score":       float(scores[i, k]),
                    "concept_map": maps[i, k].numpy(),
                }
                for i in top_idx
            ]
        return exemplars

    def explain(self, x: torch.Tensor) -> dict:
        """
        Returns concept activation information for visualization.

        Keys
        ----
        concept_scores    : (B, K)        global per-concept activation
        concept_maps      : (B, K, H', W') spatial activation before pooling —
                             upsample to input size and overlay on the image
        logits            : (B, num_classes)
        predicted_classes : (B,)          argmax of logits
        """
        with torch.no_grad():
            features     = self.add_on_layers(self.backbone(x))
            norm_cv      = F.normalize(self.concept_vectors, p=2, dim=1)
            concept_maps = F.conv2d(features, norm_cv)                      # (B, K, H, W)
            scores       = F.adaptive_max_pool2d(concept_maps, (1, 1)).flatten(1)
            logits       = self.classifier(scores)
        return {
            "concept_scores":    scores,
            "concept_maps":      concept_maps,
            "logits":            logits,
            "predicted_classes": logits.argmax(dim=1),
        }
