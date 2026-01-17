"""Simple, modular late-fusion layer for multimodal emotion/depression signals."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from config.config import load_config
from src.utils import get_logger

logger = get_logger("late_fusion")


@dataclass
class FusionWeights:
    """Weights for late fusion of different modalities."""

    text: float = 1.0
    audio: float = 1.0
    video: float = 1.0


def normalize_weights(weights: FusionWeights, available: Dict[str, bool]) -> FusionWeights:
    """
    Normalize fusion weights based on available modalities.

    Args:
        weights: Current fusion weights
        available: Dictionary indicating which modalities are available
            e.g., {"text": True, "audio": False, "video": True}

    Returns:
        Normalized fusion weights with unavailable modalities set to 0
    """
    # Set weights to 0 for unavailable modalities
    text_weight = weights.text if available.get("text", False) else 0.0
    audio_weight = weights.audio if available.get("audio", False) else 0.0
    video_weight = weights.video if available.get("video", False) else 0.0

    total = text_weight + audio_weight + video_weight

    # Edge case: if no modality is available, return all zeros
    if total == 0.0:
        logger.warning("No modalities available for fusion. Returning zero weights.")
        return FusionWeights(text=0.0, audio=0.0, video=0.0)

    # Renormalize so they sum to 1.0
    return FusionWeights(
        text=text_weight / total,
        audio=audio_weight / total,
        video=video_weight / total,
    )


def fuse_emotion_logits(
    text_logits: Optional[torch.Tensor] = None,
    audio_logits: Optional[torch.Tensor] = None,
    video_logits: Optional[torch.Tensor] = None,
    weights: Optional[FusionWeights] = None,
) -> Optional[torch.Tensor]:
    """
    Perform late fusion by weighted summation of emotion logits from available modalities.

    Assumes that all non-None logits have the same shape: (batch_size, num_labels).

    Args:
        text_logits: Text modality logits
        audio_logits: Audio modality logits
        video_logits: Video modality logits
        weights: Fusion weights. If None, uses default FusionWeights()

    Returns:
        fused_logits: torch.Tensor of shape (batch_size, num_labels), or None if no logits provided
    """
    # Check if all logits are None
    if text_logits is None and audio_logits is None and video_logits is None:
        logger.warning("No logits provided for fusion. Returning None.")
        return None

    # Find first non-None tensor to infer shape
    first_non_none = None
    if text_logits is not None:
        first_non_none = text_logits
    elif audio_logits is not None:
        first_non_none = audio_logits
    elif video_logits is not None:
        first_non_none = video_logits

    if first_non_none is None:
        return None

    batch_size, num_labels = first_non_none.shape

    # Use default weights if not provided
    if weights is None:
        weights = FusionWeights()

    # Build available dict
    available = {
        "text": text_logits is not None,
        "audio": audio_logits is not None,
        "video": video_logits is not None,
    }

    # Normalize weights
    normalized_weights = normalize_weights(weights, available)

    # Check if all weights are zero (no available modalities)
    if (
        normalized_weights.text == 0.0
        and normalized_weights.audio == 0.0
        and normalized_weights.video == 0.0
    ):
        logger.warning("No available modalities after normalization. Returning None.")
        return None

    # Initialize fused logits
    fused = torch.zeros_like(first_non_none)

    # Add weighted contributions from each available modality
    if text_logits is not None:
        fused += normalized_weights.text * text_logits

    if audio_logits is not None:
        fused += normalized_weights.audio * audio_logits

    if video_logits is not None:
        fused += normalized_weights.video * video_logits

    return fused


def compute_depression_risk(logits: torch.Tensor, depressed_class_index: int = 1) -> float:
    """
    Compute depression risk as probability of the 'depressed' class from a binary classifier's logits.

    Args:
        logits: Tensor of shape (num_classes,) or (1, num_classes)
        depressed_class_index: Index of the 'depressed' class (default=1)

    Returns:
        risk: float in [0, 1]
    """
    # Squeeze to 1D if needed
    if logits.dim() > 1:
        logits = logits.squeeze()

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Clamp index if needed
    num_classes = probs.shape[0]
    if depressed_class_index < 0 or depressed_class_index >= num_classes:
        logger.warning(
            f"Depressed class index {depressed_class_index} out of range [0, {num_classes-1}]. "
            f"Using last index instead."
        )
        depressed_class_index = num_classes - 1

    return float(probs[depressed_class_index])


def build_emotional_state(
    fused_emotion_logits: Optional[torch.Tensor],
    emotion_id2label: Dict[int, str],
    depression_logits: Optional[torch.Tensor] = None,
    fusion_weights: Optional[FusionWeights] = None,
) -> Dict[str, Any]:
    """
    Build a structured emotional state dict from fused emotion logits and optional depression logits.

    Args:
        fused_emotion_logits: Fused emotion logits of shape (1, num_labels) or (num_labels,)
        emotion_id2label: Mapping from emotion ID to label string
        depression_logits: Optional depression classifier logits
        fusion_weights: Optional fusion weights for logging

    Returns:
        Dictionary with keys:
            - primary_emotion: str or None
            - primary_emotion_confidence: float or None
            - secondary_emotion: str or None
            - secondary_emotion_confidence: float or None
            - depression_risk: float or None
            - safety_flag: bool
            - raw_emotion_probs: Dict[str, float] or None
            - fusion_weights: Dict[str, float] or None
    """
    state: Dict[str, Any] = {}

    # Process emotion logits
    if fused_emotion_logits is not None:
        # Ensure shape is (1, num_labels) for consistency
        if fused_emotion_logits.dim() == 1:
            fused_emotion_logits = fused_emotion_logits.unsqueeze(0)

        # Apply softmax to get probabilities
        probs = F.softmax(fused_emotion_logits, dim=-1)
        probs_1d = probs.squeeze(0)  # (num_labels,)

        # Find top-2 indices by probability
        top2_probs, top2_indices = torch.topk(probs_1d, k=min(2, len(probs_1d)))

        # Map indices to labels
        primary_idx = int(top2_indices[0])
        primary_emotion = emotion_id2label.get(primary_idx, "unknown")
        primary_emotion_confidence = float(top2_probs[0])

        if len(top2_indices) > 1:
            secondary_idx = int(top2_indices[1])
            secondary_emotion = emotion_id2label.get(secondary_idx, "unknown")
            secondary_emotion_confidence = float(top2_probs[1])
        else:
            secondary_emotion = None
            secondary_emotion_confidence = None

        # Build raw emotion probabilities dict
        raw_emotion_probs = {}
        for idx in range(len(probs_1d)):
            label = emotion_id2label.get(idx, f"class_{idx}")
            raw_emotion_probs[label] = float(probs_1d[idx])

        state["primary_emotion"] = primary_emotion
        state["primary_emotion_confidence"] = primary_emotion_confidence
        state["secondary_emotion"] = secondary_emotion
        state["secondary_emotion_confidence"] = secondary_emotion_confidence
        state["raw_emotion_probs"] = raw_emotion_probs
    else:
        state["primary_emotion"] = None
        state["primary_emotion_confidence"] = None
        state["secondary_emotion"] = None
        state["secondary_emotion_confidence"] = None
        state["raw_emotion_probs"] = None

    # Process depression logits
    if depression_logits is not None:
        depression_risk = compute_depression_risk(depression_logits)
        state["depression_risk"] = depression_risk
    else:
        state["depression_risk"] = None

    # Safety flag (threshold = 0.7)
    safety_flag = state.get("depression_risk") is not None and state["depression_risk"] >= 0.7
    state["safety_flag"] = safety_flag

    # Include fusion weights for logging/debugging
    if fusion_weights is not None:
        state["fusion_weights"] = {
            "text": fusion_weights.text,
            "audio": fusion_weights.audio,
            "video": fusion_weights.video,
        }
    else:
        state["fusion_weights"] = None

    return state


class LateFusion(torch.nn.Module):
    """
    Neural network-based late fusion that projects different modality logits/embeddings to a common dimension.
    
    Args:
        text_dim: Dimension of text logits/embeddings (0 if not used)
        audio_dim: Dimension of audio logits/embeddings (0 if not used)
        video_dim: Dimension of video embeddings (0 if not used)
        fused_dim: Output dimension after fusion
    """
    
    def __init__(self, text_dim: int = 0, audio_dim: int = 0, video_dim: int = 0, fused_dim: int = 256):
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.fused_dim = fused_dim
        
        # Projection layers for each modality
        if text_dim > 0:
            self.text_proj = torch.nn.Linear(text_dim, fused_dim)
        else:
            self.text_proj = None
            
        if audio_dim > 0:
            self.audio_proj = torch.nn.Linear(audio_dim, fused_dim)
        else:
            self.audio_proj = None
        
        if video_dim > 0:
            self.video_proj = torch.nn.Linear(video_dim, fused_dim)
        else:
            self.video_proj = None
        
        # Count number of active modalities
        num_active = sum([
            text_dim > 0,
            audio_dim > 0,
            video_dim > 0,
        ])
        
        # Fusion network: concatenate all projections, then reduce to fused_dim
        total_dim = fused_dim * num_active
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(total_dim, fused_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )
    
    def forward(
        self,
        text_vec: Optional[torch.Tensor] = None,
        audio_vec: Optional[torch.Tensor] = None,
        video_vec: Optional[torch.Tensor] = None,
        # Backwards compatibility aliases
        text_logits: Optional[torch.Tensor] = None,
        audio_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse embeddings/logits from different modalities.
        
        Args:
            text_vec: Text logits/embeddings of shape (batch, text_dim) or None
            audio_vec: Audio logits/embeddings of shape (batch, audio_dim) or None
            video_vec: Video embeddings of shape (batch, video_dim) or None
            text_logits: Alias for text_vec (for backwards compatibility)
            audio_logits: Alias for audio_vec (for backwards compatibility)
        
        Returns:
            Fused representation of shape (batch, fused_dim)
        """
        # Support backwards compatibility: use text_logits/audio_logits if text_vec/audio_vec not provided
        if text_vec is None and text_logits is not None:
            text_vec = text_logits
        if audio_vec is None and audio_logits is not None:
            audio_vec = audio_logits
        
        feats = []
        
        if text_vec is not None and self.text_proj is not None:
            # Ensure batch dimension
            if text_vec.dim() == 1:
                text_vec = text_vec.unsqueeze(0)
            feats.append(self.text_proj(text_vec))
        
        if audio_vec is not None and self.audio_proj is not None:
            # Ensure batch dimension
            if audio_vec.dim() == 1:
                audio_vec = audio_vec.unsqueeze(0)
            feats.append(self.audio_proj(audio_vec))
        
        if video_vec is not None and self.video_proj is not None:
            # Ensure batch dimension
            if video_vec.dim() == 1:
                video_vec = video_vec.unsqueeze(0)
            feats.append(self.video_proj(video_vec))
        
        if not feats:
            raise ValueError("No modalities provided to LateFusion (all None).")
        
        # Concatenate all modality projections
        h = torch.cat(feats, dim=-1)   # (B, fused_dim * num_modalities)
        
        # Pass through fusion network
        fused = self.fusion(h)         # (B, fused_dim)
        
        return fused


__all__ = [
    "FusionWeights",
    "normalize_weights",
    "fuse_emotion_logits",
    "compute_depression_risk",
    "build_emotional_state",
    "LateFusion",
]
