"""Real-time video streaming and processing for live multimodal inference."""

from typing import Optional, Deque
from collections import deque
from PIL import Image
import torch
from torchvision import transforms

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LiveVideoStream:
    """Manages real-time video streaming and embedding extraction."""
    
    def __init__(
        self,
        image_size: int = 224,
        history_frames: int = 5,  # Number of recent frames to maintain
        device: Optional[torch.device] = None,
    ):
        """
        Initialize live video stream processor.
        
        Args:
            image_size: Target image size for processing
            history_frames: Number of recent frames to maintain
            device: Device to process video on
        """
        self.image_size = image_size
        self.history_frames = history_frames
        self.device = device if device is not None else torch.device("cpu")
        
        # Video history: store preprocessed frames
        self.frame_history: Deque[torch.Tensor] = deque(maxlen=history_frames)
        
        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def process_frame(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        Process a single video frame.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed frame tensor of shape (C, H, W) or None if processing fails
        """
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Apply transforms
            frame_tensor = self.transform(image)
            
            # Add to history
            self.frame_history.append(frame_tensor)
            
            return frame_tensor
            
        except Exception as e:
            logger.warning(f"Failed to process video frame: {e}")
            return None
    
    def get_recent_video_embedding(
        self,
        video_encoder,
    ) -> Optional[torch.Tensor]:
        """
        Get video embedding from recent frame history using the video encoder.
        
        Args:
            video_encoder: Video encoder model (IEMOCAP video model)
            
        Returns:
            Video embedding tensor or None if no video available
        """
        if not self.frame_history:
            return None
        
        try:
            # Use the most recent frame (or average recent frames)
            # For simplicity, use the most recent frame
            recent_frame = self.frame_history[-1]
            
            # Add batch dimension: (1, C, H, W)
            frame_batch = recent_frame.unsqueeze(0).to(self.device)
            
            # Extract embedding using video encoder
            with torch.no_grad():
                # Use the extract_video_embedding method from fusion model
                if hasattr(video_encoder, 'extract_video_embedding'):
                    video_emb = video_encoder.extract_video_embedding(frame_batch)
                else:
                    # Fallback: use forward pass and extract features before classifier
                    logger.warning("Video encoder does not have extract_video_embedding method")
                    return None
                
                # Remove batch dimension if needed
                if video_emb.dim() > 1 and video_emb.shape[0] == 1:
                    video_emb = video_emb.squeeze(0)
            
            return video_emb.cpu()
            
        except Exception as e:
            logger.warning(f"Failed to extract video embedding: {e}")
            return None
    
    def clear_history(self):
        """Clear video frame history."""
        self.frame_history.clear()
    
    def has_video(self) -> bool:
        """Check if there is any video history."""
        return len(self.frame_history) > 0


__all__ = ["LiveVideoStream"]

