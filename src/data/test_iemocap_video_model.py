"""Sanity check script for IEMOCAP video model and dataset."""

from pathlib import Path
import sys
import os

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import yaml

from config.config import load_config, PROJECT_ROOT
from src.data.iemocap_multimodal_dataset import load_iemocap_multimodal_dataset
from src.models.video_iemocap_resnet import build_iemocap_video_model
from src.utils.logging_utils import get_logger


def main():
    """Test IEMOCAP video model and dataset."""
    logger = get_logger(__name__)
    cfg = load_config()
    
    logger.info("=" * 60)
    logger.info("IEMOCAP Video Model Sanity Check")
    logger.info("=" * 60)
    
    # Load raw config
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    
    video_cfg = raw_cfg.get("iemocap_video", {})
    image_size = video_cfg.get("image_size", 224)
    
    # Load dataset
    logger.info("\n1. Loading IEMOCAP multimodal dataset (video only)...")
    try:
        dataset = load_iemocap_multimodal_dataset(
            cfg=cfg,
            modalities=["video"],
            split="train",
            image_size=image_size,
            is_training=False,  # No augmentation for testing
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
        
        video = sample["video"]
        logger.info(f"   Video shape: {video.shape}")
        logger.info(f"   Video dtype: {video.dtype}")
        logger.info(f"   Video range: [{video.min():.4f}, {video.max():.4f}]")
    except Exception as e:
        logger.error(f"   Failed to get sample: {e}")
        return
    
    # Build model
    logger.info("\n3. Building IEMOCAP video model...")
    try:
        num_classes = dataset.num_labels
        backbone = video_cfg.get("backbone", "resnet18")
        freeze_backbone = video_cfg.get("freeze_backbone", False)
        
        model = build_iemocap_video_model(
            num_classes=num_classes,
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            device=cfg.device.device,
        )
        logger.info(f"   Model built successfully")
        logger.info(f"   Backbone: {backbone}")
        logger.info(f"   Freeze backbone: {freeze_backbone}")
        logger.info(f"   Model device: {next(model.parameters()).device}")
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
    except Exception as e:
        logger.error(f"   Failed to build model: {e}")
        return
    
    # Forward pass
    logger.info("\n4. Running forward pass...")
    try:
        # Prepare input: (batch=1, C, H, W)
        video_batch = video.unsqueeze(0).to(cfg.device.device)
        logger.info(f"   Input shape: {video_batch.shape}")
        
        # Create dummy label
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(dataset.emotion_labels)}
        label_id = emotion_to_id[sample["emotion"]]
        labels = torch.tensor([label_id], dtype=torch.long).to(cfg.device.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            loss, logits = model(video_batch, labels)
        
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
    logger.info("\n5. Testing batch processing...")
    try:
        # Get a few samples
        batch_size = 3
        batch_samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
        
        # Extract video frames
        batch_videos = []
        batch_labels = []
        for s in batch_samples:
            batch_videos.append(s["video"])
            batch_labels.append(emotion_to_id[s["emotion"]])
        
        # Stack
        batch_features = torch.stack(batch_videos).to(cfg.device.device)
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

