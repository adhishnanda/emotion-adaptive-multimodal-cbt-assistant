"""Video inference module for extracting face/video embeddings using ResNet18."""

from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.models as models

from src.utils.logging_utils import get_logger

logger = get_logger("video_inference")

# Global cache for backbone
_backbone_cache: Optional[nn.Module] = None


def get_device():
    """
    Get device for inference.
    
    Returns:
        torch.device: "cuda" if available, else "cpu"
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResNetFeatureExtractor(nn.Module):
    """ResNet18 feature extractor that outputs 512-dim embeddings."""
    
    def __init__(self, base_model: nn.Module):
        """
        Initialize ResNet feature extractor.
        
        Args:
            base_model: Base ResNet18 model from torchvision
        """
        super().__init__()
        # Everything except the final FC layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Feature tensor of shape (B, 512)
        """
        x = self.features(x)        # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)   # (B, 512)
        return x


def get_video_backbone() -> ResNetFeatureExtractor:
    """
    Get or create the video backbone model (lazy-loaded, cached).
    
    Returns:
        ResNetFeatureExtractor instance on the selected device in eval mode
    """
    global _backbone_cache
    
    if _backbone_cache is None:
        device = get_device()
        logger.info(f"Loading ResNet18 backbone on {device}")
        
        # Try to load with weights (newer torchvision API)
        try:
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            logger.info("Loaded ResNet18 with DEFAULT weights")
        except (AttributeError, TypeError):
            # Fallback to pretrained=True (older API)
            try:
                base_model = models.resnet18(pretrained=True)
                logger.info("Loaded ResNet18 with pretrained=True")
            except Exception as e:
                logger.error(f"Failed to load ResNet18: {e}")
                raise
        
        # Create feature extractor
        _backbone_cache = ResNetFeatureExtractor(base_model)
        _backbone_cache = _backbone_cache.to(device)
        _backbone_cache.eval()
        
        logger.info("ResNet18 backbone loaded and ready")
    
    return _backbone_cache


# Image preprocessing transforms
IMAGE_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def extract_video_embedding(
    image: Union[Image.Image, np.ndarray],
) -> torch.Tensor:
    """
    Extract a 512-dimensional embedding from an RGB image.
    
    Args:
        image: Input image as PIL.Image or numpy array (HxWxC, RGB)
    
    Returns:
        Embedding tensor of shape (512,)
    """
    device = get_device()
    backbone = get_video_backbone()
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        # Assume RGB HxWxC
        image = Image.fromarray(image)
    
    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply transforms
    x = IMAGE_TRANSFORM(image)      # (3, 224, 224)
    x = x.unsqueeze(0).to(device)   # (1, 3, 224, 224)
    
    # Extract embedding
    with torch.no_grad():
        emb = backbone(x)           # (1, 512)
    
    return emb.squeeze(0)           # (512,)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.video_inference <path_to_image>")
        raise SystemExit(1)
    
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        raise SystemExit(1)
    
    img = Image.open(img_path)
    emb = extract_video_embedding(img)
    print("Embedding shape:", emb.shape)
    print(f"Embedding stats: min={emb.min().item():.4f}, max={emb.max().item():.4f}, mean={emb.mean().item():.4f}")

