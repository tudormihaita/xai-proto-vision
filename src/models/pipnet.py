import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base_model import PrototypeModel


# ============================================================
# Utilities
# ============================================================

def l2_convolution(
    x: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    Computes squared L2 distance between feature map patches
    and prototypes using convolution trick.

    Args:
        x:
            [B, C, H, W]

        prototypes:
            [P, C, 1, 1]

    Returns:
        distances:
            [B, P, H, W]
    """

    x2 = torch.sum(x ** 2, dim=1, keepdim=True)

    p2 = torch.sum(
        prototypes ** 2,
        dim=1,
        keepdim=True
    )

    xp = F.conv2d(x, prototypes)

    distances = x2 - 2 * xp + p2

    return distances


def distances_to_similarity(
    distances: torch.Tensor,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Converts distances to similarity scores.

    Same formulation as used in ProtoPNet/PIPNet family.

    Higher similarity == better match.
    """

    return torch.log((distances + 1.0) / (distances + eps))


# ============================================================
# Backbone Factory
# ============================================================

def build_backbone(
    backbone_name: str = "resnet50",
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:

    backbone_name = backbone_name.lower()

    if backbone_name == "resnet18":

        weights = (
            models.ResNet18_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        model = models.resnet18(weights=weights)

        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        out_channels = 512

    elif backbone_name == "resnet34":

        weights = (
            models.ResNet34_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        model = models.resnet34(weights=weights)

        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        out_channels = 512

    elif backbone_name == "resnet50":

        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2
            if pretrained else None
        )

        model = models.resnet50(weights=weights)

        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        out_channels = 2048

    elif backbone_name == "vgg16":

        weights = (
            models.VGG16_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        model = models.vgg16(weights=weights)

        features = model.features

        out_channels = 512

    elif backbone_name == "convnext_tiny":

        weights = (
            models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        model = models.convnext_tiny(weights=weights)

        features = model.features

        out_channels = 768

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}"
        )

    return features, out_channels


# ============================================================
# PIPNet
# ============================================================

class PIPNet(PrototypeModel):
    """
    PIP-Net implementation.

    Paper:
    "PIP-Net: Patch-Based Intuitive Prototypes for
    Interpretable Image Classification"

    Main ideas:
    - CNN backbone
    - projection head
    - learnable prototypes
    - sparse linear classifier
    - patch-level interpretability
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        num_prototypes: int = 256,
        prototype_dim: int = 512,
        image_size: int = 224,
        classifier_bias: bool = False,
        init_scale: float = 0.1,
        sparsity_threshold: float = 1e-3,
    ):
        # ====================================================
        # Backbone (build before super().__init__)
        # ====================================================

        feature_extractor, backbone_channels = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        # Call parent __init__ with backbone and num_classes
        super().__init__(backbone=feature_extractor, num_classes=num_classes)

        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.image_size = image_size
        self.sparsity_threshold = sparsity_threshold

        # Alias for compatibility
        self.feature_extractor = self.backbone

        # ====================================================
        # Projection Layer
        # ====================================================

        self.projector = nn.Sequential(
            nn.Conv2d(
                backbone_channels,
                prototype_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True),
        )

        # ====================================================
        # Prototype Layer
        # ====================================================

        self.prototype_vectors = nn.Parameter(
            torch.randn(
                num_prototypes,
                prototype_dim,
                1,
                1,
            ) * init_scale
        )

        # ====================================================
        # Sparse Classification Layer
        # ====================================================

        self.classifier = nn.Linear(
            num_prototypes,
            num_classes,
            bias=classifier_bias,
        )

        # ====================================================
        # Initialization
        # ====================================================

        self._initialize_weights()

    # ========================================================
    # Initialization
    # ========================================================

    def _initialize_weights(self):

        for m in self.projector.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

    # ========================================================
    # Feature Extraction
    # ========================================================

    def extract_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.feature_extractor(x)
        x = self.projector(x)

        return x

    # ========================================================
    # Prototype Matching
    # ========================================================

    def compute_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:

        return l2_convolution(
            features,
            self.prototype_vectors
        )

    def compute_similarities(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:

        distances = self.compute_distances(features)

        similarities = distances_to_similarity(
            distances
        )

        return similarities

    # ========================================================
    # Prototype Pooling
    # ========================================================

    def global_max_pool(
        self,
        similarities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, P, H, W = similarities.shape

        flattened = similarities.view(B, P, -1)

        pooled, indices = torch.max(
            flattened,
            dim=-1
        )

        return pooled, indices

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning logits (B, num_classes).
        Implements the abstract method from BaseModel.
        """
        # Feature maps
        features = self.extract_features(x)

        # Prototype similarities
        similarities = self.compute_similarities(features)

        # Global max pooling over spatial locations
        prototype_scores, _ = self.global_max_pool(similarities)

        # Classification
        logits = self.classifier(prototype_scores)

        return logits

    def forward_with_details(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning detailed outputs for training/analysis.
        """
        # Feature maps
        features = self.extract_features(x)

        # Prototype similarities
        similarities = self.compute_similarities(features)

        # Global max pooling over spatial locations
        prototype_scores, max_indices = self.global_max_pool(similarities)

        # Classification
        logits = self.classifier(prototype_scores)

        return {
            "logits": logits,
            "prototype_scores": prototype_scores,
            "max_indices": max_indices,
            "similarity_maps": similarities,
            "features": features,
        }

    # ========================================================
    # Prediction
    # ========================================================

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        outputs = self.forward(x)

        return torch.argmax(
            outputs["logits"],
            dim=1
        )

    # ========================================================
    # Explanation Utilities
    # ========================================================

    @torch.no_grad()
    def get_topk_prototypes(
        self,
        x: torch.Tensor,
        k: int = 10,
    ):

        outputs = self.forward_with_details(x)

        scores = outputs["prototype_scores"]

        values, indices = torch.topk(
            scores,
            k=k,
            dim=1
        )

        return {
            "scores": values,
            "prototype_indices": indices,
        }

    @torch.no_grad()
    def get_activation_maps(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        outputs = self.forward_with_details(x)

        return outputs["similarity_maps"]

    # ========================================================
    # Losses
    # ========================================================

    def classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        return F.cross_entropy(
            logits,
            labels
        )

    def sparsity_loss(
        self,
        lambda_l1: float = 1e-4
    ) -> torch.Tensor:

        return lambda_l1 * torch.norm(
            self.classifier.weight,
            p=1
        )

    def orthogonality_loss(
        self,
        lambda_orth: float = 1e-3
    ) -> torch.Tensor:
        """
        Encourages prototype diversity.
        """

        P = self.prototype_vectors.view(
            self.num_prototypes,
            -1
        )

        P = F.normalize(P, dim=1)

        similarity_matrix = torch.matmul(
            P,
            P.t()
        )

        identity = torch.eye(
            similarity_matrix.size(0),
            device=similarity_matrix.device
        )

        loss = (
            similarity_matrix - identity
        ).pow(2).mean()

        return lambda_orth * loss

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        lambda_l1: float = 1e-4,
        lambda_orth: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes total loss and individual loss components.
        Implements the abstract method from PrototypeModel.
        """
        cls_loss = self.classification_loss(
            logits,
            labels
        )

        sparse_loss = self.sparsity_loss(
            lambda_l1=lambda_l1
        )

        orth_loss = self.orthogonality_loss(
            lambda_orth=lambda_orth
        )

        total = (
            cls_loss
            + sparse_loss
            + orth_loss
        )

        return {
            "total": total,
            "cls": cls_loss,
            "sparsity": sparse_loss,
            "orthogonality": orth_loss,
        }

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        lambda_l1: float = 1e-4,
        lambda_orth: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        Legacy method for backward compatibility.
        Delegates to compute_loss().
        """
        return self.compute_loss(
            outputs["logits"],
            labels,
            lambda_l1=lambda_l1,
            lambda_orth=lambda_orth,
        )

    # ========================================================
    # Compatibility methods
    # ========================================================

    def push_prototypes(self, train_loader, device: str | torch.device) -> None:
        """
        Optional prototype anchoring (no-op for PIPNet).
        Implements method from PrototypeModel.
        """
        pass

    # ========================================================
    # Prototype Utilities
    # ========================================================

    @torch.no_grad()
    def prototype_similarity_matrix(
        self
    ) -> torch.Tensor:

        P = self.prototype_vectors.view(
            self.num_prototypes,
            -1
        )

        P = F.normalize(P, dim=1)

        return torch.matmul(P, P.t())

    @torch.no_grad()
    def prune_prototypes(
        self,
        threshold: float = 1e-3
    ) -> int:
        """
        Removes inactive prototypes based on
        classifier weights magnitude.

        Returns:
            Number of removed prototypes.
        """

        weights = torch.abs(
            self.classifier.weight
        ).sum(dim=0)

        keep_mask = weights > threshold

        keep_indices = torch.where(
            keep_mask
        )[0]

        removed = self.num_prototypes - len(keep_indices)

        if removed == 0:
            return 0

        # Update prototype vectors

        self.prototype_vectors = nn.Parameter(
            self.prototype_vectors.data[
                keep_indices
            ]
        )

        # Update classifier

        old_weights = self.classifier.weight.data[
            :,
            keep_indices
        ]

        new_classifier = nn.Linear(
            len(keep_indices),
            self.num_classes,
            bias=self.classifier.bias is not None
        )

        new_classifier.weight.data.copy_(
            old_weights
        )

        if self.classifier.bias is not None:
            new_classifier.bias.data.copy_(
                self.classifier.bias.data
            )

        self.classifier = new_classifier

        self.num_prototypes = len(keep_indices)

        return removed

    # ========================================================
    # Abstract Method Implementations
    # ========================================================

    def explain(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Returns explanation information for interpretability.
        Implements the abstract method from BaseModel.
        """
        outputs = self.forward_with_details(x)
        
        return {
            "logits": outputs["logits"],
            "prototype_scores": outputs["prototype_scores"],
            "similarity_maps": outputs["similarity_maps"],
            "features": outputs["features"],
        }

    def get_backbone_params(self):
        """
        Returns backbone parameters for selective optimization.
        Implements method from PrototypeModel.
        """
        return self.feature_extractor.parameters()

    def get_prototype_params(self):
        """
        Returns prototype layer parameters for selective optimization.
        Implements the abstract method from PrototypeModel.
        """
        return [self.prototype_vectors]

    # ========================================================
    # Freezing Utilities
    # ========================================================

    def freeze_backbone(self):

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def freeze_prototypes(self):

        self.prototype_vectors.requires_grad = False

    def unfreeze_prototypes(self):

        self.prototype_vectors.requires_grad = True

    # ========================================================
    # Model Info
    # ========================================================

    def extra_repr(self) -> str:

        return (
            f"num_classes={self.num_classes}, "
            f"num_prototypes={self.num_prototypes}, "
            f"prototype_dim={self.prototype_dim}"
        )