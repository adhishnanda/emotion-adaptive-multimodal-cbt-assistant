"""Audio inference helper for DAIC-WOZ depression detection model."""

from pathlib import Path
from typing import Union, Optional, Dict, Any

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
import torchaudio.functional as AF
import yaml
import os

from config.config import load_config, PROJECT_ROOT
from src.utils.logging_utils import get_logger
from src.models.audio_cnn_lstm import AudioCNNLSTM, AudioModelConfig

logger = get_logger("audio_inference")


def get_device() -> str:
    """
    Get the device to use for inference.

    Returns:
        "cuda" if available and allowed by config, else "cpu"
    """
    cfg = load_config()
    
    # Check config for device preference
    prefer_gpu = cfg.device.prefer_gpu if hasattr(cfg, 'device') and hasattr(cfg.device, 'prefer_gpu') else True
    
    # Fallback: check raw config for training.device if needed
    if not hasattr(cfg, 'device'):
        cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
        cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
        with open(cfg_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        training_cfg = raw_cfg.get("training", {})
        if isinstance(training_cfg.get("device"), dict):
            prefer_gpu = training_cfg.get("device", {}).get("prefer_gpu", True)
        else:
            prefer_gpu = True
    
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_audio_model(num_classes: int, n_mfcc: int) -> AudioCNNLSTM:
    """
    Construct an AudioCNNLSTM with the same hyperparameters used in train_audio.py.

    Args:
        num_classes: Number of output classes (2 for binary classification)
        n_mfcc: Number of MFCC coefficients

    Returns:
        Initialized AudioCNNLSTM model on the selected device
    """
    device_str = get_device()
    device = torch.device(device_str)
    
    # Use default hyperparameters from AudioModelConfig
    model_cfg = AudioModelConfig(
        num_mfcc=n_mfcc,
        num_labels=num_classes,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.3,
    )
    
    model = AudioCNNLSTM(model_cfg)
    model = model.to(device)
    
    return model


def load_trained_audio_model(
    model_path: Union[str, Path],
    num_classes: int,
    n_mfcc: int,
    map_location: Optional[Union[str, torch.device]] = None,
) -> AudioCNNLSTM:
    """
    Build the audio model and load trained weights from `model_path`.

    Args:
        model_path: Path to the saved model state dict
        num_classes: Number of output classes
        n_mfcc: Number of MFCC coefficients
        map_location: Device to load the model on. If None, uses get_device()

    Returns:
        Loaded model in eval mode

    Raises:
        FileNotFoundError: If model_path does not exist
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if map_location is None:
        map_location = get_device()
    
    # Build model
    model = build_audio_model(num_classes, n_mfcc)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    
    # Set to eval mode
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def create_audio_transform(sample_rate: int, n_mfcc: int) -> torch.nn.Module:
    """
    Create a torchaudio-based transform that:
    - Converts to mono
    - Resamples to `sample_rate` if needed
    - Computes MFCCs with `n_mfcc` coefficients.

    Args:
        sample_rate: Target sample rate
        n_mfcc: Number of MFCC coefficients

    Returns:
        A callable nn.Module that takes (1, num_samples) and returns (n_mfcc, time)
    """
    class AudioTransform(nn.Module):
        def __init__(self, sample_rate: int, n_mfcc: int):
            super().__init__()
            self.sample_rate = sample_rate
            self.mfcc_transform = AT.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": 2048,
                    "hop_length": 512,
                    "n_mels": 64,
                }
            )
        
        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            """
            Args:
                waveform: Input waveform of shape (1, num_samples) or (channels, num_samples)
                          Already resampled to target sample_rate
            
            Returns:
                MFCC features of shape (n_mfcc, time)
            """
            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Ensure shape is (1, num_samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Compute MFCC: (1, n_mfcc, time)
            mfcc = self.mfcc_transform(waveform)
            
            # Remove channel dimension: (n_mfcc, time)
            mfcc = mfcc.squeeze(0)
            
            return mfcc
    
    return AudioTransform(sample_rate, n_mfcc)


def predict_depression_from_wav(
    wav_path: Union[str, Path],
    model: Optional[AudioCNNLSTM] = None,
    transform: Optional[torch.nn.Module] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run inference for a single WAV file.

    Args:
        wav_path: Path to the WAV file
        model: Pre-loaded model. If None, loads from config.
        transform: Pre-created transform. If None, creates default.
        threshold: Threshold for binary classification (only used if model outputs single value)

    Returns:
        Dictionary with:
        {
            "wav_path": str,
            "logits": <tensor on cpu>,
            "probs": <tensor on cpu>,
            "pred_class": int,
            "pred_label": "non_depressed" or "depressed",
        }
    """
    wav_path = Path(wav_path)
    
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    
    # Load config
    cfg = load_config()
    
    # Load raw YAML to access daic_woz section
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    daic_cfg = raw_cfg.get("daic_woz", {})
    
    sample_rate = daic_cfg.get("sample_rate", 16000)
    # Default n_mfcc to 40 (standard value used in training)
    n_mfcc = daic_cfg.get("n_mfcc", 40)
    
    # Get model path
    paths_cfg = raw_cfg.get("paths", {})
    audio_model_path = paths_cfg.get("audio_model_best")
    if audio_model_path is None:
        audio_model_path = cfg.paths.models_dir / "audio" / "audio_cnn_lstm_best.pt"
    else:
        audio_model_path = PROJECT_ROOT / audio_model_path
    
    # Load model if not provided
    if model is None:
        num_classes = 2  # Binary classification: non-depressed (0) vs depressed (1)
        model = load_trained_audio_model(audio_model_path, num_classes, n_mfcc)
    
    device = next(model.parameters()).device
    
    # Create transform if not provided
    if transform is None:
        transform = create_audio_transform(sample_rate, n_mfcc)
    
    # Load waveform
    waveform, orig_sr = torchaudio.load(str(wav_path))
    
    # Resample if needed
    if orig_sr != sample_rate:
        waveform = AF.resample(waveform, orig_freq=orig_sr, new_freq=sample_rate)
    
    # Apply transform
    with torch.no_grad():
        # Convert to mono and compute MFCC (transform handles this)
        features = transform(waveform)  # (n_mfcc, time)
        
        # Add batch dimension: (1, n_mfcc, time)
        features = features.unsqueeze(0)
        
        # Move to device
        features = features.to(device)
        
        # Forward pass
        _, logits = model(features, labels=None)  # (1, num_classes)
    
    # Move to CPU
    logits = logits.cpu()
    
    # Apply activation based on output shape
    if logits.shape[1] == 1:
        # Single output: use sigmoid for binary classification
        probs = torch.sigmoid(logits)  # (1, 1)
        pred_class = (probs > threshold).int().item()
    else:
        # Multiple classes: use softmax
        probs = torch.softmax(logits, dim=1)  # (1, num_classes)
        pred_class = torch.argmax(probs, dim=1).item()
    
    # Map class to label
    if pred_class == 0:
        pred_label = "non_depressed"
    else:
        pred_label = "depressed"
    
    return {
        "wav_path": str(wav_path),
        "logits": logits,
        "probs": probs,
        "pred_class": pred_class,
        "pred_label": pred_label,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.audio_inference <path_to_wav>")
        sys.exit(1)
    
    result = predict_depression_from_wav(sys.argv[1])
    print(result)

