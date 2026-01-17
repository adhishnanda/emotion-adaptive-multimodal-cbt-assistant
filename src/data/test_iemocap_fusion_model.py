"""Sanity check script for IEMOCAP multimodal fusion model and dataset."""

from pathlib import Path
import sys
import os

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torchaudio.transforms as AT
import yaml
from transformers import AutoTokenizer

from config.config import load_config, PROJECT_ROOT
from src.data.iemocap_multimodal_dataset import load_iemocap_multimodal_dataset
from src.models.iemocap_multimodal_fusion_model import build_iemocap_fusion_model
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
    """Test IEMOCAP fusion model and dataset."""
    logger = get_logger(__name__)
    cfg = load_config()
    
    logger.info("=" * 60)
    logger.info("IEMOCAP Multimodal Fusion Model Sanity Check")
    logger.info("=" * 60)
    
    # Load raw config
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    
    fusion_cfg = raw_cfg.get("iemocap_fusion", {})
    video_cfg = raw_cfg.get("iemocap_video", {})
    audio_cfg = raw_cfg.get("iemocap_audio", {})
    
    image_size = video_cfg.get("image_size", 224)
    sample_rate = audio_cfg.get("audio_sample_rate", 16000)
    n_mfcc = audio_cfg.get("n_mfcc", 40)
    
    # Load dataset
    logger.info("\n1. Loading IEMOCAP multimodal dataset (all modalities)...")
    try:
        dataset = load_iemocap_multimodal_dataset(
            cfg=cfg,
            modalities=["text", "audio", "video"],
            split="train",
            image_size=image_size,
            is_training=False,
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
        
        text = sample.get("text", "")
        audio = sample.get("audio")
        video = sample.get("video")
        
        logger.info(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
        if audio is not None:
            logger.info(f"   Audio shape: {audio.shape}")
        if video is not None:
            logger.info(f"   Video shape: {video.shape}")
    except Exception as e:
        logger.error(f"   Failed to get sample: {e}")
        return
    
    # Build model
    logger.info("\n3. Building IEMOCAP fusion model...")
    try:
        num_classes = dataset.num_labels
        model = build_iemocap_fusion_model(
            num_classes=num_classes,
            device=cfg.device.device,
        )
        logger.info(f"   Model built successfully")
        logger.info(f"   Model device: {next(model.parameters()).device}")
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
    except Exception as e:
        logger.error(f"   Failed to build model: {e}", exc_info=True)
        return
    
    # Forward pass
    logger.info("\n4. Running forward pass...")
    try:
        # Prepare inputs
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)
        text_encoding = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        text_input_ids = text_encoding["input_ids"].to(cfg.device.device)
        text_attention_mask = text_encoding["attention_mask"].to(cfg.device.device)
        
        logger.info(f"   Text input_ids shape: {text_input_ids.shape}")
        logger.info(f"   Text attention_mask shape: {text_attention_mask.shape}")
        
        # Extract MFCC from audio
        if audio is not None:
            audio_cpu = audio.cpu() if audio.is_cuda else audio
            audio_mfcc = extract_mfcc_from_waveform(audio_cpu, sample_rate=sample_rate, n_mfcc=n_mfcc)
            audio_mfcc = audio_mfcc.unsqueeze(0).to(cfg.device.device)  # Add batch dimension
            logger.info(f"   Audio MFCC shape: {audio_mfcc.shape}")
        else:
            audio_mfcc = None
        
        # Prepare video
        if video is not None:
            video_batch = video.unsqueeze(0).to(cfg.device.device)  # Add batch dimension
            logger.info(f"   Video batch shape: {video_batch.shape}")
        else:
            video_batch = None
        
        # Create label
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(dataset.emotion_labels)}
        label_id = emotion_to_id[sample["emotion"]]
        labels = torch.tensor([label_id], dtype=torch.long).to(cfg.device.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            loss, logits = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_mfcc=audio_mfcc,
                video_frames=video_batch,
                labels=labels,
            )
        
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
        batch_size = 2
        batch_samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
        
        # Prepare batch
        texts = [s.get("text", "") for s in batch_samples]
        text_encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        batch_text_ids = text_encodings["input_ids"].to(cfg.device.device)
        batch_text_mask = text_encodings["attention_mask"].to(cfg.device.device)
        
        # Extract MFCC for all samples
        batch_mfcc = []
        for s in batch_samples:
            audio_s = s.get("audio")
            if audio_s is not None:
                audio_cpu = audio_s.cpu() if audio_s.is_cuda else audio_s
                mfcc_s = extract_mfcc_from_waveform(audio_cpu, sample_rate=sample_rate, n_mfcc=n_mfcc)
                batch_mfcc.append(mfcc_s)
            else:
                batch_mfcc.append(torch.zeros(n_mfcc, 100, dtype=torch.float32))
        
        # Pad MFCC sequences
        max_time = max(mfcc.shape[1] for mfcc in batch_mfcc)
        padded_mfcc = []
        for mfcc_s in batch_mfcc:
            if mfcc_s.shape[1] < max_time:
                padding = max_time - mfcc_s.shape[1]
                mfcc_s = torch.nn.functional.pad(mfcc_s, (0, padding))
            padded_mfcc.append(mfcc_s)
        
        batch_audio_mfcc = torch.stack(padded_mfcc).to(cfg.device.device)
        
        # Stack video frames
        batch_videos = [s.get("video") for s in batch_samples]
        batch_video_frames = torch.stack(batch_videos).to(cfg.device.device)
        
        batch_labels = [emotion_to_id[s["emotion"]] for s in batch_samples]
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(cfg.device.device)
        
        logger.info(f"   Batch text_ids shape: {batch_text_ids.shape}")
        logger.info(f"   Batch audio MFCC shape: {batch_audio_mfcc.shape}")
        logger.info(f"   Batch video shape: {batch_video_frames.shape}")
        logger.info(f"   Batch labels shape: {batch_labels_tensor.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            loss, logits = model(
                text_input_ids=batch_text_ids,
                text_attention_mask=batch_text_mask,
                audio_mfcc=batch_audio_mfcc,
                video_frames=batch_video_frames,
                labels=batch_labels_tensor,
            )
        
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

