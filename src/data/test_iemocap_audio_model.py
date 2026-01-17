"""Sanity check script for IEMOCAP audio model and dataset."""

from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torchaudio.transforms as AT

from config.config import load_config, PROJECT_ROOT
import os
import yaml
from src.data.iemocap_multimodal_dataset import load_iemocap_multimodal_dataset
from src.models.audio_iemocap_cnn_lstm import build_iemocap_audio_model
from src.utils.logging_utils import get_logger


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


def main():
    """Test IEMOCAP audio model and dataset."""
    logger = get_logger(__name__)
    cfg = load_config()
    
    logger.info("=" * 60)
    logger.info("IEMOCAP Audio Model Sanity Check")
    logger.info("=" * 60)
    
    # Load dataset
    logger.info("\n1. Loading IEMOCAP multimodal dataset (audio only)...")
    try:
        dataset = load_iemocap_multimodal_dataset(
            cfg=cfg,
            modalities=["audio"],
            split="train",
        )
        logger.info(f"   Dataset loaded: {len(dataset)} samples")
        logger.info(f"   Emotion labels: {dataset.emotion_labels}")
        logger.info(f"   Number of classes: {dataset.num_labels}")
    except Exception as e:
        logger.error(f"   Failed to load dataset: {e}")
        return
    
    # Get a sample
    logger.info("\n2. Fetching a sample from dataset...")
    try:
        sample = dataset[0]
        logger.info(f"   Sample keys: {list(sample.keys())}")
        logger.info(f"   Emotion: {sample['emotion']}")
        logger.info(f"   Utterance ID: {sample['utterance_id']}")
        
        audio = sample["audio"]
        logger.info(f"   Audio shape: {audio.shape}")
        logger.info(f"   Audio dtype: {audio.dtype}")
        logger.info(f"   Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    except Exception as e:
        logger.error(f"   Failed to get sample: {e}")
        return
    
    # Extract MFCC
    logger.info("\n3. Extracting MFCC features...")
    try:
        # Load raw config to access iemocap_audio section
        cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
        cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
        with open(cfg_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        iemocap_cfg = raw_cfg.get("iemocap_audio", {})
        sample_rate = iemocap_cfg.get("audio_sample_rate", 16000)
        n_mfcc = iemocap_cfg.get("n_mfcc", 40)
        
        mfcc = extract_mfcc_from_waveform(audio, sample_rate=sample_rate, n_mfcc=n_mfcc)
        logger.info(f"   MFCC shape: {mfcc.shape}")
        logger.info(f"   MFCC dtype: {mfcc.dtype}")
        logger.info(f"   MFCC range: [{mfcc.min():.4f}, {mfcc.max():.4f}]")
    except Exception as e:
        logger.error(f"   Failed to extract MFCC: {e}")
        return
    
    # Build model
    logger.info("\n4. Building IEMOCAP audio model...")
    try:
        num_classes = dataset.num_labels
        model = build_iemocap_audio_model(
            num_classes=num_classes,
            num_mfcc=n_mfcc,
            device=cfg.device.device,
        )
        logger.info(f"   Model built successfully")
        logger.info(f"   Model device: {next(model.parameters()).device}")
        logger.info(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        logger.error(f"   Failed to build model: {e}")
        return
    
    # Forward pass
    logger.info("\n5. Running forward pass...")
    try:
        # Prepare input: (batch=1, n_mfcc, time)
        mfcc_batch = mfcc.unsqueeze(0).to(cfg.device.device)
        logger.info(f"   Input shape: {mfcc_batch.shape}")
        
        # Create dummy label
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(dataset.emotion_labels)}
        label_id = emotion_to_id[sample["emotion"]]
        labels = torch.tensor([label_id], dtype=torch.long).to(cfg.device.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            loss, logits = model(mfcc_batch, labels)
        
        logger.info(f"   Output logits shape: {logits.shape}")
        logger.info(f"   Loss: {loss.item():.4f}")
        logger.info(f"   Logits: {logits[0].cpu().numpy()}")
        
        # Get predictions
        probs = torch.softmax(logits, dim=-1)
        pred_id = logits.argmax(dim=-1).item()
        pred_emotion = dataset.emotion_labels[pred_id]
        pred_prob = probs[0, pred_id].item()
        
        logger.info(f"   Predicted emotion: {pred_emotion} (ID: {pred_id}, prob: {pred_prob:.4f})")
        logger.info(f"   True emotion: {sample['emotion']} (ID: {label_id})")
        
    except Exception as e:
        logger.error(f"   Failed forward pass: {e}", exc_info=True)
        return
    
    # Test batch processing
    logger.info("\n6. Testing batch processing...")
    try:
        # Get a few samples
        batch_size = 3
        batch_samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
        
        # Extract MFCC for all samples
        batch_mfcc = []
        batch_labels = []
        for s in batch_samples:
            mfcc_s = extract_mfcc_from_waveform(s["audio"], sample_rate=sample_rate, n_mfcc=n_mfcc)
            batch_mfcc.append(mfcc_s)
            batch_labels.append(emotion_to_id[s["emotion"]])
        
        # Pad to same length
        max_time = max(mfcc.shape[1] for mfcc in batch_mfcc)
        padded_mfcc = []
        for mfcc_s in batch_mfcc:
            if mfcc_s.shape[1] < max_time:
                padding = max_time - mfcc_s.shape[1]
                mfcc_s = torch.nn.functional.pad(mfcc_s, (0, padding))
            padded_mfcc.append(mfcc_s)
        
        # Stack
        batch_features = torch.stack(padded_mfcc).to(cfg.device.device)
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(cfg.device.device)
        
        logger.info(f"   Batch features shape: {batch_features.shape}")
        logger.info(f"   Batch labels shape: {batch_labels_tensor.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            loss, logits = model(batch_features, batch_labels_tensor)
        
        logger.info(f"   Batch logits shape: {logits.shape}")
        logger.info(f"   Batch loss: {loss.item():.4f}")
        
        # Predictions
        predictions = logits.argmax(dim=-1)
        logger.info(f"   Predictions: {predictions.cpu().numpy()}")
        logger.info(f"   True labels: {batch_labels}")
        
    except Exception as e:
        logger.error(f"   Failed batch processing: {e}", exc_info=True)
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("Sanity check complete! All tests passed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

