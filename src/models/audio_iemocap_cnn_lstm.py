"""CNN + BiLSTM audio classifier for IEMOCAP multi-class emotion recognition."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os

import torch
import torch.nn as nn

from config.config import load_config, PROJECT_ROOT
import yaml


@dataclass
class IEMOCAPAudioModelConfig:
    """Configuration for IEMOCAP AudioCNNLSTM model."""

    num_mfcc: int = 40
    num_classes: int = 9  # IEMOCAP has 9 emotion classes
    cnn_channels1: int = 64
    cnn_channels2: int = 128
    kernel_size: int = 5
    lstm_hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.3


class IEMOCAPAudioCNNLSTM(nn.Module):
    """CNN + BiLSTM model for IEMOCAP multi-class audio emotion classification from MFCC features."""

    def __init__(self, cfg: IEMOCAPAudioModelConfig):
        """
        Initialize IEMOCAP AudioCNNLSTM model.

        Args:
            cfg: Model configuration
        """
        super().__init__()
        self.cfg = cfg

        # Padding to maintain sequence length (before pooling)
        padding = cfg.kernel_size // 2

        # Conv1d block 1
        self.conv1 = nn.Conv1d(
            in_channels=cfg.num_mfcc,
            out_channels=cfg.cnn_channels1,
            kernel_size=cfg.kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(cfg.cnn_channels1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Conv1d block 2
        self.conv2 = nn.Conv1d(
            in_channels=cfg.cnn_channels1,
            out_channels=cfg.cnn_channels2,
            kernel_size=cfg.kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(cfg.cnn_channels2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cfg.cnn_channels2,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Dropout
        self.dropout = nn.Dropout(cfg.dropout)

        # Classifier (multi-class for IEMOCAP)
        # Bidirectional LSTM: 2 * hidden_size
        self.classifier = nn.Linear(2 * cfg.lstm_hidden, cfg.num_classes)

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_mfcc, time)
            labels: Ground truth labels (optional)

        Returns:
            Tuple of (loss, logits). Loss is None if labels are not provided.
        """
        # Conv block 1: (batch, n_mfcc, time) -> (batch, cnn_channels1, time)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch, cnn_channels1, time // 2)

        # Conv block 2: (batch, cnn_channels1, time // 2) -> (batch, cnn_channels2, time // 4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch, cnn_channels2, time // 4)

        # Permute for LSTM: (batch, channels, time) -> (batch, time, channels)
        x = x.permute(0, 2, 1)  # (batch, time // 4, cnn_channels2)

        # BiLSTM
        # outputs: (batch, time, 2 * hidden_size) - not used
        # h_n: (num_layers * 2, batch, hidden_size)
        # c_n: (num_layers * 2, batch, hidden_size)
        outputs, (h_n, c_n) = self.lstm(x)

        # Extract last layer's forward and backward hidden states
        # For bidirectional LSTM with 1 layer:
        # h_n[0] = forward hidden state
        # h_n[1] = backward hidden state
        # For multiple layers, last layer is at indices -2 and -1
        h_forward = h_n[-2, :, :]  # (batch, hidden_size)
        h_backward = h_n[-1, :, :]  # (batch, hidden_size)

        # Concatenate forward and backward
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (batch, 2 * hidden_size)

        # Apply dropout
        h_cat = self.dropout(h_cat)

        # Classifier
        logits = self.classifier(h_cat)  # (batch, num_classes)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits


def build_iemocap_audio_model(
    num_classes: Optional[int] = None,
    num_mfcc: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build and return an IEMOCAP AudioCNNLSTM model.

    Args:
        num_classes: Number of emotion classes (if None, reads from config)
        num_mfcc: Number of MFCC coefficients (if None, reads from config)
        device: Device to move model to. If None, uses cfg.device.device

    Returns:
        Initialized model on the specified device

    Raises:
        RuntimeError: If model building fails
    """
    cfg = load_config()
    if device is None:
        device = cfg.device.device

    # Load raw config to access iemocap_audio section
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    
    # Get config values
    iemocap_cfg = raw_cfg.get("iemocap_audio", {})
    if num_classes is None:
        num_classes = iemocap_cfg.get("num_classes", 9)
    if num_mfcc is None:
        num_mfcc = iemocap_cfg.get("n_mfcc", 40)

    model_cfg = IEMOCAPAudioModelConfig(num_mfcc=num_mfcc, num_classes=num_classes)
    try:
        model = IEMOCAPAudioCNNLSTM(model_cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to build IEMOCAP AudioCNNLSTM model: {e}") from e

    return model.to(device)


__all__ = ["IEMOCAPAudioModelConfig", "IEMOCAPAudioCNNLSTM", "build_iemocap_audio_model"]

