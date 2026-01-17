"""DistilBERT-based emotion classifier for MELD text-only stage."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from config.config import load_config


@dataclass
class DistilBERTConfig:
    """Configuration for DistilBERT emotion classifier."""

    pretrained_name: str
    num_labels: int
    dropout: float = 0.3


class DistilBERTEmotionClassifier(nn.Module):
    """DistilBERT-based emotion classification model."""

    def __init__(self, model_cfg: DistilBERTConfig):
        """
        Initialize DistilBERT emotion classifier.

        Args:
            model_cfg: Model configuration
        """
        super().__init__()

        # Load transformer config
        self.config = AutoConfig.from_pretrained(
            model_cfg.pretrained_name, num_labels=model_cfg.num_labels
        )

        # Load base model
        self.model = AutoModel.from_pretrained(model_cfg.pretrained_name, config=self.config)

        # Classifier head
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.classifier = nn.Linear(self.config.hidden_size, model_cfg.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)

        Returns:
            Tuple of (loss, logits). Loss is None if labels are not provided.
        """
        # Pass inputs through transformer
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling over sequence length
        # attention_mask: (batch_size, seq_len)
        if attention_mask is not None:
            # Expand attention mask for broadcasting
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            # Sum over sequence dimension
            sum_hidden = (last_hidden_state * attention_mask_expanded).sum(dim=1)
            # Sum of attention mask (sequence lengths)
            sum_mask = attention_mask_expanded.sum(dim=1)
            # Avoid division by zero
            pooled = sum_hidden / (sum_mask + 1e-9)
        else:
            # Simple mean pooling if no attention mask
            pooled = last_hidden_state.mean(dim=1)

        # Apply dropout and classifier
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits


def build_text_model(
    num_labels: int, device: Optional[torch.device] = None
) -> nn.Module:
    """
    Build and return a DistilBERT emotion classifier.

    Args:
        num_labels: Number of emotion classes
        device: Device to move model to. If None, uses cfg.device.device

    Returns:
        Initialized model on the specified device

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        cfg = load_config()
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")

    # Create model configuration
    model_cfg = DistilBERTConfig(
        pretrained_name=cfg.text_model.pretrained_name,
        num_labels=num_labels,
    )

    # Determine device
    if device is None:
        device = cfg.device.device

    # Load model
    try:
        model = DistilBERTEmotionClassifier(model_cfg)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load transformer model '{model_cfg.pretrained_name}'. "
            f"Please check your internet connection and that the model name is valid. "
            f"Original error: {e}"
        )

    # Move to device
    try:
        model = model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to move model to device {device}: {e}")

    return model


__all__ = ["DistilBERTConfig", "DistilBERTEmotionClassifier", "build_text_model"]
