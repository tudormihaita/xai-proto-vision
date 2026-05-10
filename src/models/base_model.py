from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str) -> tuple[nn.Module, int]:
    """
    Returns (feature_extractor, feature_dim).

    Strips the classification head from the pretrained model, exposing the
    final convolutional feature map (B, C, H, W). For a 224x224 input both
    supported backbones output (B, 512, 7, 7).
    """
    if name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feature_dim = m.fc.in_features                         # 512
        backbone = nn.Sequential(*list(m.children())[:-2])     # strip avgpool + fc
        return backbone, feature_dim

    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        backbone = m.features                                   # final conv output: (B, 512, 7, 7)
        return backbone, 512

    raise ValueError(f"Unknown backbone {name!r}. Supported: 'resnet34', 'vgg16'")


class BaseModel(nn.Module, ABC):
    """
    Root interface shared by all models (baseline + four prototype methods).
    Every subclass must implement forward() and explain().
    """

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_classes) logits."""

    @abstractmethod
    def explain(self, x: torch.Tensor) -> dict:
        """
        Returns a method-specific explanation dict for a single batch.
        See docs/prototype_methods.md for the expected keys per method.
        """


class BaselineModel(BaseModel):
    """
    Standard classification baseline: pretrained backbone with a linear
    classifier on top. Serves as the reference accuracy for all comparisons.
    """
    def __init__(self, backbone_name: str = "resnet34", num_classes: int = 200) -> None:
        backbone, feature_dim = build_backbone(backbone_name)
        super().__init__(backbone, num_classes)
        self.feature_dim = feature_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)             # (B, C, H, W)
        pooled = self.pool(features).flatten(1) # (B, C)
        return self.classifier(pooled)          # (B, num_classes)

    def explain(self, x: torch.Tensor) -> dict:
        """Returns the spatial feature map — useful for CAM-style visualisation."""
        with torch.no_grad():
            features = self.backbone(x) # (B, C, H, W)
            logits = self.classifier(self.pool(features).flatten(1))
        return {"logits": logits, "features": features}


class PrototypeModel(BaseModel, ABC):
    """
    Shared interface for ProtoPNet, ProtoTree, TesNet, and PIPNet.

    Adds three responsibilities on top of BaseModel:
      - compute_loss(): returns a dict of loss components (total + per-term)
      - push_prototypes(): no-op by default; overridden by ProtoPNet & ProtoTree
      - parameter group accessors for selective optimisation during phased training
    """
    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns at minimum {"total": ..., "cls": ...}.
        Method-specific terms (cluster, separation, ortho, sparsity…) are
        added as additional keys and logged separately by the Trainer.
        """

    def push_prototypes(self, train_loader) -> None:
        """
        Scans the full training set and anchors each prototype to the nearest
        real training patch. No-op in the base class; overridden by ProtoPNet
        and ProtoTree. TesNet and PIPNet leave this as a no-op.
        """

    def get_backbone_params(self):
        """Returns backbone parameters for selective optimisation."""
        return self.backbone.parameters()

    @abstractmethod
    def get_prototype_params(self):
        """
        Returns prototype layer parameters.
        Used to freeze/unfreeze specific parameter groups during phased training
        (e.g. ProtoPNet Phase 1 freezes backbone, Phase 3 freezes prototypes).
        """
