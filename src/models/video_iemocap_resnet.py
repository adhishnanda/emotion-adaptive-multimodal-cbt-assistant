"""ResNet-based video emotion classifier for IEMOCAP multi-class emotion recognition."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os

import torch
import torch.nn as nn
import torchvision.models as models
import yaml

from config.config import load_config, PROJECT_ROOT


@dataclass
class IEMOCAPVideoModelConfig:
    """Configuration for IEMOCAP Video ResNet model."""

    backbone: str = "resnet18"  # "resnet18" or "resnet50"
    num_classes: int = 9  # IEMOCAP has 9 emotion classes
    pretrained: bool = True
    freeze_backbone: bool = False  # If True, only train the classifier head
    dropout: float = 0.3


class IEMOCAPVideoResNet(nn.Module):
    """ResNet-based model for IEMOCAP multi-class video emotion classification."""

    def __init__(self, cfg: IEMOCAPVideoModelConfig):
        """
        Initialize IEMOCAP Video ResNet model.

        Args:
            cfg: Model configuration
        """
        super().__init__()
        self.cfg = cfg

        # Load pretrained ResNet backbone
        if cfg.backbone == "resnet18":
            backbone = models.resnet18(pretrained=cfg.pretrained)
            feature_dim = backbone.fc.in_features
        elif cfg.backbone == "resnet50":
            backbone = models.resnet50(pretrained=cfg.pretrained)
            feature_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}. Use 'resnet18' or 'resnet50'")

        # Remove the original classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone if requested
        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(feature_dim, cfg.num_classes)

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, C, H, W) - ImageNet normalized
            labels: Ground truth labels (optional)

        Returns:
            Tuple of (loss, logits). Loss is None if labels are not provided.
        """
        # Extract features: (batch, C, H, W) -> (batch, feature_dim)
        features = self.backbone(x)
        # ResNet backbone outputs (batch, feature_dim, 1, 1), squeeze spatial dims
        features = features.squeeze(-1).squeeze(-1)

        # Apply dropout
        features = self.dropout(features)

        # Classifier
        logits = self.classifier(features)  # (batch, num_classes)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits


def build_iemocap_video_model(
    num_classes: Optional[int] = None,
    backbone: Optional[str] = None,
    freeze_backbone: Optional[bool] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build and return an IEMOCAP Video ResNet model.

    Args:
        num_classes: Number of emotion classes (if None, reads from config)
        backbone: Backbone architecture ("resnet18" or "resnet50", if None reads from config)
        freeze_backbone: Whether to freeze backbone weights (if None, reads from config)
        device: Device to move model to. If None, uses cfg.device.device

    Returns:
        Initialized model on the specified device

    Raises:
        RuntimeError: If model building fails
    """
    cfg = load_config()
    if device is None:
        device = cfg.device.device

    # Load raw config to access iemocap_video section
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Get config values
    video_cfg = raw_cfg.get("iemocap_video", {})
    if num_classes is None:
        num_classes = video_cfg.get("num_classes", 9)
    if backbone is None:
        backbone = video_cfg.get("backbone", "resnet18")
    if freeze_backbone is None:
        freeze_backbone = video_cfg.get("freeze_backbone", False)

    model_cfg = IEMOCAPVideoModelConfig(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )
    try:
        model = IEMOCAPVideoResNet(model_cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to build IEMOCAP Video ResNet model: {e}") from e

    return model.to(device)


__all__ = ["IEMOCAPVideoModelConfig", "IEMOCAPVideoResNet", "build_iemocap_video_model"]

