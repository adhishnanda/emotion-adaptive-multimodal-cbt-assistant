"""Multimodal inference pipeline for IEMOCAP emotion classification."""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import os

import torch
import torch.nn as nn
import torchaudio.transforms as AT
import yaml
from PIL import Image
from transformers import AutoTokenizer

from config.config import load_config, PROJECT_ROOT
from src.models.iemocap_multimodal_fusion_model import (
    build_iemocap_fusion_model,
    IEMOCAPMultimodalFusionModel,
)
# Video preprocessing uses torchvision transforms directly
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def extract_mfcc_from_waveform(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
) -> torch.Tensor:
    """Extract MFCC features from audio waveform tensor."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    mfcc_transform = AT.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc_transform(waveform)
    
    if mfcc.shape[0] == 1:
        mfcc = mfcc.squeeze(0)
    
    return mfcc


class IEMOCAPMultimodalInference:
    """Inference pipeline for IEMOCAP multimodal emotion classification."""
    
    def __init__(
        self,
        fusion_model: Optional[IEMOCAPMultimodalFusionModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[torch.device] = None,
        emotion_labels: Optional[List[str]] = None,
    ):
        """
        Initialize IEMOCAP multimodal inference pipeline.
        
        Args:
            fusion_model: Trained fusion model (if None, loads from config)
            tokenizer: Text tokenizer (if None, loads from config)
            device: Device to run inference on (if None, uses config)
            emotion_labels: List of emotion labels (if None, uses IEMOCAP defaults)
        """
        cfg = load_config()
        # Explicitly set device: use cuda if available, else cpu
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load raw config
        cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
        cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
        with open(cfg_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        
        self.fusion_cfg = raw_cfg.get("iemocap_fusion", {})
        self.audio_cfg = raw_cfg.get("iemocap_audio", {})
        self.video_cfg = raw_cfg.get("iemocap_video", {})
        
        # Audio parameters
        self.sample_rate = self.audio_cfg.get("audio_sample_rate", 16000)
        self.n_mfcc = self.audio_cfg.get("n_mfcc", 40)
        
        # Video parameters
        self.image_size = self.video_cfg.get("image_size", 224)
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)
        else:
            self.tokenizer = tokenizer
        
        # Load fusion model
        if fusion_model is None:
            logger.info("Loading IEMOCAP fusion model...")
            num_classes = self.fusion_cfg.get("num_classes", 9)
            self.fusion_model = build_iemocap_fusion_model(
                num_classes=num_classes,
                device=self.device,
            )
            
            # Load trained fusion weights
            fusion_model_path = PROJECT_ROOT / self.fusion_cfg.get("fusion_model_path", "models/fusion/iemocap_multimodal_fusion_best.pt")
            if fusion_model_path.exists():
                try:
                    state_dict = torch.load(fusion_model_path, map_location=self.device)
                    self.fusion_model.load_state_dict(state_dict)
                    logger.info(f"Loaded fusion model weights from {fusion_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load fusion model weights: {e}")
            else:
                logger.warning(f"Fusion model weights not found at {fusion_model_path}, using untrained fusion head")
            
            # Explicitly move model to device and set to eval mode
            self.fusion_model.to(self.device)
            self.fusion_model.eval()
        else:
            self.fusion_model = fusion_model
            # Ensure model is on the correct device
            self.fusion_model.to(self.device)
            self.fusion_model.eval()
        
        # Emotion labels
        if emotion_labels is None:
            # IEMOCAP default emotion labels
            self.emotion_labels = [
                "anger",
                "happy",
                "sadness",
                "neutral",
                "excited",
                "frustration",
                "disgust",
                "fear",
                "surprise",
            ]
        else:
            self.emotion_labels = emotion_labels
        
        self.id_to_emotion = {i: label for i, label in enumerate(self.emotion_labels)}
        self.emotion_to_id = {label: i for i, label in enumerate(self.emotion_labels)}
    
    def preprocess_text(self, text: str, max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess text input.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoding["input_ids"], encoding["attention_mask"]
    
    def preprocess_audio(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio waveform to MFCC features.
        
        Args:
            audio_waveform: Audio waveform tensor
            
        Returns:
            MFCC features tensor of shape (n_mfcc, time)
        """
        # Ensure waveform is on CPU for MFCC extraction
        waveform_cpu = audio_waveform.cpu() if audio_waveform.is_cuda else audio_waveform
        
        # Extract MFCC
        mfcc = extract_mfcc_from_waveform(
            waveform_cpu,
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
        )
        
        return mfcc
    
    def preprocess_video(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess video/image input.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor of shape (C, H, W)
        """
        # Use torchvision transforms directly (same as in video_transforms)
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        # Convert PIL to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        frame_tensor = transform(image)
        return frame_tensor
    
    def predict(
        self,
        text: Optional[str] = None,
        audio_waveform: Optional[torch.Tensor] = None,
        video_image: Optional[Image.Image] = None,
    ) -> Dict[str, any]:
        """
        Predict emotion from multimodal inputs.
        
        Args:
            text: Input text (optional)
            audio_waveform: Audio waveform tensor (optional)
            video_image: PIL Image (optional)
            
        Returns:
            Dictionary with:
                - emotion: str (predicted emotion label)
                - confidence: float (confidence score)
                - logits: torch.Tensor (raw logits)
                - probs: torch.Tensor (probabilities)
                - per_modality: dict with per-modality predictions (optional)
        """
        # Check available modalities
        has_text = text is not None and text.strip()
        has_audio = audio_waveform is not None
        has_video = video_image is not None
        
        if not has_text and not has_audio and not has_video:
            raise ValueError("At least one modality must be provided")
        
        # Preprocess inputs and ensure all tensors are on the same device
        text_input_ids = None
        text_attention_mask = None
        if has_text:
            text_input_ids, text_attention_mask = self.preprocess_text(text)
            # Explicitly move to device
            text_input_ids = text_input_ids.to(self.device)
            text_attention_mask = text_attention_mask.to(self.device)
        
        audio_mfcc = None
        if has_audio:
            mfcc = self.preprocess_audio(audio_waveform)
            # Pad to a reasonable length (e.g., 200 frames)
            max_time = 200
            if mfcc.shape[1] < max_time:
                padding = max_time - mfcc.shape[1]
                mfcc = torch.nn.functional.pad(mfcc, (0, padding))
            elif mfcc.shape[1] > max_time:
                mfcc = mfcc[:, :max_time]
            # Add batch dimension and explicitly move to device
            audio_mfcc = mfcc.unsqueeze(0).to(self.device)
        
        video_frames = None
        if has_video:
            frame = self.preprocess_video(video_image)
            # Add batch dimension and explicitly move to device
            video_frames = frame.unsqueeze(0).to(self.device)
        
        # Forward pass through fusion model
        # All tensors are now guaranteed to be on self.device
        with torch.no_grad():
            loss, logits = self.fusion_model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_mfcc=audio_mfcc,
                video_frames=video_frames,
                labels=None,
            )
        
        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get prediction
        pred_id = logits.argmax(dim=-1).item()
        pred_emotion = self.id_to_emotion[pred_id]
        pred_confidence = probs[0, pred_id].item()
        
        # Build result
        result = {
            "emotion": pred_emotion,
            "confidence": pred_confidence,
            "logits": logits.cpu(),
            "probs": probs.cpu(),
            "available_modalities": {
                "text": has_text,
                "audio": has_audio,
                "video": has_video,
            },
        }
        
        return result


@torch.no_grad()
def load_iemocap_multimodal_inference(
    device: Optional[torch.device] = None,
) -> IEMOCAPMultimodalInference:
    """
    Load IEMOCAP multimodal inference pipeline.
    
    Args:
        device: Device to run inference on (if None, uses config)
        
    Returns:
        Initialized IEMOCAPMultimodalInference instance
    """
    return IEMOCAPMultimodalInference(device=device)


__all__ = [
    "IEMOCAPMultimodalInference",
    "load_iemocap_multimodal_inference",
    "extract_mfcc_from_waveform",
]

