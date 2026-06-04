"""
TesNet: Transparent Embedding Space Network
Paper: "Interpretable Image Recognition by Constructing Transparent Embedding Space"
       Wang, Liu, Wang & Jing - ICCV 2021

Architecture:
  backbone  -> add_on_layers (bottleneck + sigmoid) -> concept projection
            -> global max pool -> linear classifier

Concept projection:
  - concept_vectors: (num_concepts_per_class × num_classes, concept_dim, 1, 1) explicit parameter
  - normalized before each conv2d so inner product = cosine similarity at each spatial location
  - global max pool picks the single best-matching location per concept across the feature map
  - each concept belongs to exactly one class (prototype_class_identity buffer); scales to any dataset

Loss:
  total = CE(logits, y)
        + λ_clst  * cluster_loss           (pull features toward same-class concepts)
        − λ_sep   * sep_loss               (push features away from other-class concepts)
        + λ_ortho * ||C_c C_c^T − I||²_F   (within-class orthogonality per class c)
        − λ_ss    * mean||P_c1 − P_c2||_F  (push class subspaces apart, Grassmann)
        + λ_l1    * ||W_cls||_1            (sparsity on classifier weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import PrototypeModel, build_backbone


class TesNet(PrototypeModel):
    """
    TesNet prototype model with a ResNet or VGG backbone.

    Parameters
    backbone_name : str
        "resnet34" or "vgg16".
    num_classes : int
        Number of output classes. Drives total concept count automatically.
    num_concepts_per_class : int
        Number of concept vectors per class.  Total = num_concepts_per_class × num_classes.
        Paper value: 10 (→ 2000 total for CUB-200). Sweep: 5, 10, 20.
    concept_dim : int
        Dimension of concept space after bottleneck projection.  Paper value: 64.
    lambda_clst : float
        Cluster loss weight. Paper value: 0.8.
    lambda_sep : float
        Separation loss weight (applied with negative sign). Paper value: 0.08.
    lambda_ortho : float
        Within-class orthogonality regularization. Paper value: 1e-4.
    lambda_l1 : float
        L1 sparsity on classifier weights. Paper value: 1e-4.
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
        lambda_ss: float = 0.08,
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
        self.lambda_ss     = lambda_ss
        self.lambda_l1     = lambda_l1

        # prototype_class_identity[k, c] = 1  iff concept k belongs to class c.
        class_identity = torch.zeros(self.num_concepts, num_classes)
        for k in range(self.num_concepts):
            class_identity[k, k // num_concepts_per_class] = 1
        self.register_buffer("prototype_class_identity", class_identity)

        # bottleneck: compresses backbone features into bounded concept space
        # Sigmoid ensures activations are in [0, 1] before projection
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(feature_dim, concept_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(concept_dim, concept_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Concept vectors: shape (num_concepts, concept_dim, 1, 1)
        # used as conv2d weights (after L2 normalization) to compute per-location
        # cosine similarities between feature maps and concept directions
        self.concept_vectors = nn.Parameter(
            torch.rand(self.num_concepts, concept_dim, 1, 1), requires_grad=True
        )

        # linear classifier: concept scores -> class logits
        # no bias so that all class discrimination must flow through concept activations
        self.classifier = nn.Linear(self.num_concepts, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # per-class orthogonal init: within each class the K concepts start as an
        # orthogonal basis; different classes are initialized independently
        k = self.num_concepts_per_class
        for c in range(self.num_classes):
            s = c * k
            nn.init.orthogonal_(self.concept_vectors[s : s + k].view(k, -1))

        # classifier init from paper: +1 for same-class concept connections,
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
        """Projection magnitude onto normalised concept basis -> (B, num_concepts)."""
        norm_cv = F.normalize(self.concept_vectors, p=2, dim=1, eps=1e-8)
        proj    = F.conv2d(features, norm_cv)                    # (B, K, H, W)
        return F.adaptive_max_pool2d(proj, (1, 1)).flatten(1)   # (B, K)

    def _cosine_distances(self, features: torch.Tensor) -> torch.Tensor:
        """Cosine distance from every spatial location to each concept -> (B, K, H, W)."""
        feat_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        conc_norm = F.normalize(self.concept_vectors, p=2, dim=1, eps=1e-8)
        return -F.conv2d(feat_norm, conc_norm)   # negate similarity → distance

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_loss   = F.cross_entropy(logits, labels)
        ortho_loss = self._orthogonality_loss()
        ss_loss    = self._subspace_separation_loss()
        l1_loss    = self._selective_l1_loss()

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
            - self.lambda_ss    * ss_loss     # negative: maximize distance between class subspaces
            + self.lambda_l1    * l1_loss
        )
        return {
            "total": total,
            "cls":   cls_loss,
            "clst":  clst_loss,
            "sep":   sep_loss,
            "ortho": ortho_loss,
            "ss":    ss_loss,
        }

    def _cluster_sep_loss(
        self,
        cos_dists: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cos_dists : (B, K, H, W) cosine distances to each concept at every location.
        labels    : (B,)         ground-truth class indices.

        Returns
        clst_loss : mean of each image's min distance to its same-class concepts
                    (we want this small -> pull toward correct-class concepts)
        sep_loss  : mean of each image's min distance to other-class concepts
                    (we want this large -> applied with negative sign in total)
        """
        # reduce spatial dimensions: min distance per concept over all locations -> (B, K)
        min_per_concept = cos_dists.flatten(2).min(dim=2).values

        # boolean mask (B, K): True where concept k belongs to image i's class
        # prototype_class_identity is (K, num_classes); indexing by labels gives (K, B)
        same = self.prototype_class_identity[:, labels].T.bool()   # (B, K)

        large = min_per_concept.new_tensor(1e4)   # matches input dtype/device
        clst_loss = min_per_concept.where(same,  large).min(dim=1).values.mean()
        sep_loss  = min_per_concept.where(~same, large).min(dim=1).values.mean()
        return clst_loss, sep_loss

    def _selective_l1_loss(self) -> torch.Tensor:
        """
        L1 penalty from paper Stage 2 (L_h): penalizes only cross-class classifier
        weights (W[c, k] where concept k does NOT belong to class c).
        Uniform L1 would fight the +1 same-class initialization; this does not.
        """
        cross_class = 1.0 - self.prototype_class_identity.T   # (num_classes, num_concepts)
        return (self.classifier.weight.abs() * cross_class).sum() / cross_class.sum()

    def _subspace_separation_loss(self) -> torch.Tensor:
        """
        Pushes every pair of class subspaces apart on the Grassmann manifold.
        Each class spans a k-dimensional subspace; we maximize the projection-metric
        distance ||P_c1 − P_c2||_F between their projection matrices P_c = B_c^T B_c.

        Returns the MEAN of pairwise distances (positive scalar); applied with a
        negative coefficient in compute_loss so that increasing distance reduces total.

        Memory: avoids the (num_classes, num_classes, D, D) tensor via the identity
            ||P_i − P_j||_F^2 = ||P_i||_F^2 + ||P_j||_F^2 − 2 <P_i, P_j>_F

        Only the upper-triangle pairs are sqrt'd (diagonal is exactly 0 → infinite
        gradient of sqrt -> NaN in backward), so we never compute sqrt(0).
        """
        k = self.num_concepts_per_class
        C = F.normalize(
            self.concept_vectors.view(self.num_classes, k, -1), p=2, dim=2, eps=1e-8
        )
        P        = torch.bmm(C.transpose(1, 2), C)       # (num_classes, D, D)
        norms_sq = (P * P).sum(dim=(1, 2))                          # (num_classes,)
        dots     = torch.einsum("nij,mij->nm", P, P)          # (num_classes, num_classes)
        dist_sq  = norms_sq[:, None] + norms_sq[None, :] - 2 * dots # (num_classes, num_classes)

        mask   = torch.triu(torch.ones_like(dist_sq, dtype=torch.bool), diagonal=1)
        upper  = dist_sq[mask].clamp(min=1e-8)
        return upper.sqrt().mean() / (2 ** 0.5)

    def _orthogonality_loss(self) -> torch.Tensor:
        """
        Within-class orthogonality: for each class, its K concept vectors should
        form a mutually orthogonal set.  Computed via batched matrix multiply across
        all classes simultaneously — O(num_classes × K² × D).

        Cross-class orthogonality is intentionally NOT enforced: different classes
        can share low-level visual features (edges, colors) without harming
        interpretability.
        """
        k = self.num_concepts_per_class
        # reshape to (num_classes, k, concept_dim) and row-normalize
        C    = F.normalize(
            self.concept_vectors.view(self.num_classes, k, -1), p=2, dim=2, eps=1e-8
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

    def push_prototypes(self, train_loader, device: str | torch.device) -> None:
        """
        Stage 2: Embedding space transparency.

        Replaces each concept vector with the feature embedding of its nearest
        training patch from its own class:
            b_j <- arg max_{p ∈ P_c} p^T · b_j

        After this call concept_vectors literally ARE image patch embeddings, so
        every explanation shown to a user is faithful by construction rather than
        post-hoc approximate. Call once near the end of training (push_epoch),
        then freeze backbone + concept_vectors and fine-tune only the classifier.
        """
        self.eval()
        self.to(device)
        k_per_c = self.num_concepts_per_class

        best_sim   = torch.full((self.num_concepts,), -float("inf"), device=device)
        best_patch = torch.zeros(self.num_concepts, self.concept_dim, device=device)

        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                features = self.add_on_layers(self.backbone(images))   # (B, D, H, W)
                B, D, H, W = features.shape

                # flatten spatial locations into one axis
                patches      = features.permute(0, 2, 3, 1).reshape(B * H * W, D)  # (N, D)
                patch_labels = labels.repeat_interleave(H * W)                      # (N,)
                concepts_n   = F.normalize(self.concept_vectors[:, :, 0, 0], p=2, dim=1)  # (K, D)
                patches_n    = F.normalize(patches, p=2, dim=1)
                sims         = patches_n @ concepts_n.T                             # (N, K)

                for c in range(self.num_classes):
                    mask = patch_labels == c
                    if not mask.any():
                        continue
                    ks = slice(c * k_per_c, (c + 1) * k_per_c)
                    c_sims    = sims[mask][:, ks]           # (n_c, k_per_c)
                    c_patches = patches[mask]               # (n_c, D)
                    max_sims, max_idx = c_sims.max(dim=0)   # (k_per_c,)

                    improved = max_sims > best_sim[ks]
                    best_sim[ks] = torch.where(improved, max_sims, best_sim[ks])
                    new_p = c_patches[max_idx]              # (k_per_c, D)
                    best_patch[ks] = torch.where(improved.unsqueeze(1), new_p, best_patch[ks])

        self.concept_vectors.data.copy_(best_patch.unsqueeze(-1).unsqueeze(-1))
        print(f"Push complete; mean best cosine similarity = {best_sim.mean():.4f}")

    def find_concept_exemplars(
        self,
        loader,
        device: str | torch.device,
        top_n: int = 5,
        concept_indices: list[int] | None = None,
    ) -> dict[int, list[dict]]:
        """
        For each concept, find the top_n training images that activate it most.
        Run once after training to ground each concept in real images.
        Does NOT modify concept_vectors.

        Parameters
        concept_indices : optional list of concept indices to compute exemplars for.
            Defaults to all concepts. Pass model.class_concept_indices(c) to limit
            to a single class — recommended when num_concepts is large (e.g. 2000).

        Returns
        dict: concept_idx (int) → list of top_n dicts, each with:
            image       : (3, H, W) tensor
            score       : float   — concept activation for this image
            concept_map : (H', W') numpy array — spatial activation map
        """
        self.eval()
        self.to(device)

        target = set(concept_indices) if concept_indices is not None else set(range(self.num_concepts))

        # Streaming top-n: keep only top_n candidates per concept at all times.
        # self.explain() still produces a (B, K, H, W) tensor per batch, but we
        # avoid accumulating all batches into a single (N, K, H, W) dataset tensor.
        top_scores: dict[int, list[tuple[float, int]]] = {k: [] for k in target}
        all_images: list[torch.Tensor] = []
        # Per-image concept maps stored sparsely: img_idx → {concept_idx: map}
        sparse_maps: list[dict[int, torch.Tensor]] = []

        offset = 0
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                out = self.explain(images)
                scores_b = out["concept_scores"].cpu()   # (B, K)
                maps_b   = out["concept_maps"].cpu()     # (B, K, H, W)
                images_b = images.cpu()

                for b in range(images_b.size(0)):
                    img_idx = offset + b
                    all_images.append(images_b[b])
                    batch_maps: dict[int, torch.Tensor] = {}

                    for k in target:
                        score = float(scores_b[b, k])
                        heap = top_scores[k]
                        if len(heap) < top_n or score > heap[0][0]:
                            batch_maps[k] = maps_b[b, k]
                            heap.append((score, img_idx))
                            heap.sort(key=lambda x: x[0])
                            if len(heap) > top_n:
                                heap.pop(0)

                    sparse_maps.append(batch_maps)
                offset += images_b.size(0)

        exemplars: dict[int, list[dict]] = {}
        for k in target:
            candidates = sorted(top_scores[k], key=lambda x: -x[0])
            exemplars[k] = [
                {
                    "image":       all_images[img_idx],
                    "score":       score,
                    "concept_map": sparse_maps[img_idx][k].numpy(),
                }
                for score, img_idx in candidates
            ]
        return exemplars

    def explain(self, x: torch.Tensor) -> dict:
        """
        Returns concept activation information for visualization.

        Keys
        concept_scores    : (B, K)          global per-concept activation
        concept_maps      : (B, K, H', W')  spatial activation before pooling; upsample to input size and overlay on the image
        logits            : (B, num_classes)
        predicted_classes : (B,)             argmax of logits
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
