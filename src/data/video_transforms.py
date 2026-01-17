"""Video preprocessing and frame extraction utilities for IEMOCAP."""

from pathlib import Path
from typing import Optional

import cv2
import torch
from torchvision import transforms
from PIL import Image

from src.utils.logging_utils import get_logger


def extract_middle_frame(video_path: Path) -> Optional[Image.Image]:
    """
    Extract the middle frame from a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        PIL Image of the middle frame, or None if extraction fails
    """
    logger = get_logger(__name__)
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.warning(f"Video has no frames: {video_path}")
            cap.release()
            return None
        
        # Seek to middle frame
        middle_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback to first frame
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.warning(f"Could not read any frame from: {video_path}")
                return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        return pil_image
        
    except Exception as e:
        logger.warning(f"Error extracting frame from {video_path}: {e}")
        return None


def get_imagenet_transforms(image_size: int = 224, is_training: bool = False):
    """
    Get ImageNet-standard transforms for video frames.
    
    Args:
        image_size: Target image size (default 224 for ResNet)
        is_training: If True, applies data augmentation (random crop, flip)
                    If False, applies only normalization
        
    Returns:
        torchvision transforms.Compose
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def load_and_transform_video_frame(
    video_path: Path,
    image_size: int = 224,
    is_training: bool = False,
) -> Optional[torch.Tensor]:
    """
    Load a video frame and apply ImageNet transforms.
    
    Args:
        video_path: Path to video file
        image_size: Target image size
        is_training: Whether to apply training transforms (with augmentation)
        
    Returns:
        Transformed image tensor of shape (C, H, W), or None if loading fails
    """
    # Extract frame
    pil_image = extract_middle_frame(video_path)
    
    if pil_image is None:
        return None
    
    # Apply transforms
    transform = get_imagenet_transforms(image_size=image_size, is_training=is_training)
    tensor = transform(pil_image)
    
    return tensor


__all__ = [
    "extract_middle_frame",
    "get_imagenet_transforms",
    "load_and_transform_video_frame",
]

