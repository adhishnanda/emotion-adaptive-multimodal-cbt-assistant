"""CNN + BiLSTM audio classifier for MFCC features."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from config.config import load_config


@dataclass
class AudioModelConfig:
    """Configuration for AudioCNNLSTM model."""

    num_mfcc: int = 40
    num_labels: int = 2
    cnn_channels1: int = 64
    cnn_channels2: int = 128
    kernel_size: int = 5
    lstm_hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.3


class AudioCNNLSTM(nn.Module):
    """CNN + BiLSTM model for audio emotion classification from MFCC features."""

    def __init__(self, cfg: AudioModelConfig):
        """
        Initialize AudioCNNLSTM model.

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

        # Classifier
        # Bidirectional LSTM: 2 * hidden_size
        self.classifier = nn.Linear(2 * cfg.lstm_hidden, cfg.num_labels)

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
        logits = self.classifier(h_cat)  # (batch, num_labels)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits


def build_audio_model(
    num_labels: int, num_mfcc: int = 40, device: Optional[torch.device] = None
) -> nn.Module:
    """
    Build and return an AudioCNNLSTM model.

    Args:
        num_labels: Number of emotion classes
        num_mfcc: Number of MFCC coefficients
        device: Device to move model to. If None, uses cfg.device.device

    Returns:
        Initialized model on the specified device

    Raises:
        RuntimeError: If model building fails
    """
    cfg = load_config()
    if device is None:
        device = cfg.device.device

    model_cfg = AudioModelConfig(num_mfcc=num_mfcc, num_labels=num_labels)
    try:
        model = AudioCNNLSTM(model_cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to build AudioCNNLSTM model: {e}") from e

    return model.to(device)


__all__ = ["AudioModelConfig", "AudioCNNLSTM", "build_audio_model"]

