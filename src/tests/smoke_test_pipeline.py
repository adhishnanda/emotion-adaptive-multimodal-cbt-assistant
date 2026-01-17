"""Smoke test to verify the entire pipeline works end-to-end."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from config.config import load_config, PROJECT_ROOT, _load_yaml_config
from src.data.iemocap_multimodal_dataset import IEMOCAPMultimodalDataset
from src.models.text_distilbert import build_text_model
from src.models.audio_iemocap_cnn_lstm import build_iemocap_audio_model
from src.models.video_iemocap_resnet import build_iemocap_video_model
from src.models.iemocap_multimodal_fusion_model import build_iemocap_fusion_model
from src.utils.logging_utils import get_logger
from src.utils.seed_utils import set_global_seed

logger = get_logger(__name__)


def _extract_main_tensor(output, model_name: str):
    """
    Given a model output, return the main torch.Tensor to inspect.
    
    Handles:
      - single tensor
      - tuple/list where at least one element is a tensor
    
    Raises a clear error if no tensor is found.
    """
    if torch.is_tensor(output):
        return output
    
    if isinstance(output, (tuple, list)):
        # Prefer the first tensor element
        for item in output:
            if torch.is_tensor(item):
                return item
        
        raise TypeError(f"[{model_name}] Model output is tuple/list but contains no tensors: {type(output)}")
    
    raise TypeError(f"[{model_name}] Expected tensor or tuple/list of tensors, got {type(output)}")


def collate_multimodal_batch(batch, modalities):
    """Custom collate function for multimodal batches."""
    from transformers import AutoTokenizer
    
    texts = []
    audio_mfccs = []
    video_frames = []
    
    for item in batch:
        if "text" in modalities and item.get("text") is not None:
            texts.append(item["text"])
        if "audio" in modalities and item.get("audio") is not None:
            audio_mfccs.append(item["audio"])
        if "video" in modalities and item.get("video") is not None:
            video_frames.append(item["video"])
    
    # For smoke test, we only need dummy labels to verify shapes
    # Use zeros as placeholder labels since we're just testing forward pass
    batch_size = max(len(texts), len(audio_mfccs), len(video_frames), 1)
    result = {"labels": torch.zeros(batch_size, dtype=torch.long)}
    
    if texts:
        cfg = load_config()
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=cfg.text_model.max_seq_len,
            return_tensors="pt",
        )
        result["text_input_ids"] = encoded["input_ids"]
        result["text_attention_mask"] = encoded["attention_mask"]
    
    if audio_mfccs:
        # Pad audio MFCCs to same length
        max_time = max(mfcc.shape[1] for mfcc in audio_mfccs) if audio_mfccs else 200
        padded_audio = []
        for mfcc in audio_mfccs:
            if mfcc.shape[1] < max_time:
                padding = max_time - mfcc.shape[1]
                mfcc = torch.nn.functional.pad(mfcc, (0, padding))
            elif mfcc.shape[1] > max_time:
                mfcc = mfcc[:, :max_time]
            padded_audio.append(mfcc)
        result["audio_mfcc"] = torch.stack(padded_audio)
    
    if video_frames:
        result["video_frames"] = torch.stack(video_frames)
    
    return result


def test_text_model(device, cfg_dict):
    """Test text-only model."""
    logger.info("\n" + "="*60)
    logger.info("Testing Text Model")
    logger.info("="*60)
    
    try:
        # Load model
        num_labels = cfg_dict["iemocap_fusion"]["num_classes"]
        model = build_text_model(num_labels=num_labels, device=device)
        model.eval()
        logger.info("OK Text model loaded")
        
        # Load dataset
        index_path = PROJECT_ROOT / cfg_dict["iemocap_multimodal"]["index_path"]
        dataset = IEMOCAPMultimodalDataset(
            index_path=index_path,
            modalities=["text"],
            split="test",
        )
        
        if len(dataset) == 0:
            logger.warning("WARN No test samples found for text, trying train split...")
            dataset = IEMOCAPMultimodalDataset(
                index_path=index_path,
                modalities=["text"],
                split="train",
            )
        
        if len(dataset) == 0:
            logger.error("FAIL No samples found for text model test")
            return False
        
        logger.info(f"OK Loaded {len(dataset)} text samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda b: collate_multimodal_batch(b, ["text"]),
        )
        
        # Test forward pass
        batch = next(iter(dataloader))
        
        with torch.no_grad():
            outputs = model(
                input_ids=batch["text_input_ids"].to(device),
                attention_mask=batch["text_attention_mask"].to(device)
            )
            logits = _extract_main_tensor(outputs, "text")
        
        logger.info("OK Text model forward pass successful")
        logger.info("  Input shape: %s", tuple(batch["text_input_ids"].shape))
        logger.info("  Logits shape: %s", tuple(logits.shape))
        preds = logits.argmax(dim=1).cpu().numpy()
        logger.info("  Predictions: %s", preds)
        
        return "PASSED"
        
    except Exception as e:
        logger.error(f"FAIL Text model test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return "FAILED"


def test_audio_model(device, cfg_dict, cfg):
    """Test audio-only model."""
    logger.info("\n" + "="*60)
    logger.info("Testing Audio Model")
    logger.info("="*60)
    
    try:
        # Load model
        num_classes = cfg_dict["iemocap_fusion"]["num_classes"]
        model = build_iemocap_audio_model(num_classes=num_classes, device=device)
        model.eval()
        logger.info("OK Audio model loaded")
        
        # Load dataset
        iemocap_index_path = PROJECT_ROOT / cfg_dict["iemocap_multimodal"]["index_path"]
        audio_dataset = IEMOCAPMultimodalDataset(
            cfg=cfg,
            index_path=iemocap_index_path,
            modalities=["audio"],
            split="test",
        )
        logger.info("Loaded %d samples with modalities: ['audio']", len(audio_dataset))
        
        if len(audio_dataset) == 0:
            logger.warning("SKIP Audio model test: no audio samples in dataset")
            return "SKIPPED"
        
        audio_loader = DataLoader(audio_dataset, batch_size=2, shuffle=False)
        batch = next(iter(audio_loader))
        
        # Extract audio tensor depending on batch structure
        if isinstance(batch, dict):
            audio_batch = batch.get("audio")
        elif isinstance(batch, (tuple, list)):
            audio_batch = batch[0]
        else:
            raise TypeError(f"Unexpected audio batch type: {type(batch)}")
        
        if audio_batch is None:
            raise ValueError("Audio batch is None; expected tensor")
        
        audio_batch = audio_batch.to(device)
        
        with torch.no_grad():
            outputs = model(audio_batch)
            logits = _extract_main_tensor(outputs, "audio")
        
        logger.info("OK Audio model forward pass successful")
        logger.info("  Audio batch shape: %s", tuple(audio_batch.shape))
        logger.info("  Audio logits shape: %s", tuple(logits.shape))
        preds = logits.argmax(dim=1).cpu().numpy()
        logger.info("  Predictions (first 10): %s", preds[:10])
        
        return "PASSED"
        
    except Exception as e:
        logger.error(f"FAIL Audio model test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return "FAILED"


def test_video_model(device, cfg_dict):
    """Test video-only model."""
    logger.info("\n" + "="*60)
    logger.info("Testing Video Model")
    logger.info("="*60)
    
    try:
        # Load model
        num_classes = cfg_dict["iemocap_fusion"]["num_classes"]
        model = build_iemocap_video_model(num_classes=num_classes, device=device)
        model.eval()
        logger.info("OK Video model loaded")
        
        # Load dataset
        index_path = PROJECT_ROOT / cfg_dict["iemocap_multimodal"]["index_path"]
        dataset = IEMOCAPMultimodalDataset(
            index_path=index_path,
            modalities=["video"],
            split="test",
        )
        
        if len(dataset) == 0:
            logger.warning("WARN No test samples found for video, trying train split...")
            dataset = IEMOCAPMultimodalDataset(
                index_path=index_path,
                modalities=["video"],
                split="train",
            )
        
        if len(dataset) == 0:
            logger.error("FAIL No samples found for video model test")
            return False
        
        logger.info(f"OK Loaded {len(dataset)} video samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda b: collate_multimodal_batch(b, ["video"]),
        )
        
        # Test forward pass
        batch = next(iter(dataloader))
        video_batch = batch["video_frames"].to(device)
        
        with torch.no_grad():
            outputs = model(video_batch)
            logits = _extract_main_tensor(outputs, "video")
        
        logger.info("OK Video model forward pass successful")
        logger.info("  Input shape: %s", tuple(video_batch.shape))
        logger.info("  Output tensor shape: %s", tuple(logits.shape))
        
        return "PASSED"
        
    except Exception as e:
        logger.error(f"FAIL Video model test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return "FAILED"


def test_fusion_model(device, cfg_dict, cfg):
    """Test multimodal fusion model."""
    logger.info("\n" + "="*60)
    logger.info("Testing Fusion Model")
    logger.info("="*60)
    
    try:
        from transformers import AutoTokenizer
        
        # Load model
        num_classes = cfg_dict["iemocap_fusion"]["num_classes"]
        model = build_iemocap_fusion_model(num_classes=num_classes, device=device)
        model.to(device)
        model.eval()
        logger.info("OK Fusion model loaded")
        
        # Load dataset
        iemocap_index_path = PROJECT_ROOT / cfg_dict["iemocap_multimodal"]["index_path"]
        dataset = IEMOCAPMultimodalDataset(
            cfg=cfg,
            index_path=iemocap_index_path,
            modalities=["text", "audio", "video"],
            split="test",
        )
        logger.info("Loaded %d samples with modalities: ['text', 'audio', 'video']", len(dataset))
        
        if len(dataset) == 0:
            logger.warning("SKIP Fusion model test: no multimodal samples in dataset")
            return "SKIPPED"
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        if not isinstance(batch, dict):
            raise TypeError(f"Unexpected multimodal batch type: {type(batch)}")
        
        # Text: tokenize raw strings
        texts = batch.get("text")
        if texts is None:
            raise ValueError("Missing 'text' in multimodal batch")
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=cfg.text_model.max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        audio_batch = batch.get("audio")
        if audio_batch is None:
            raise ValueError("Missing 'audio' in multimodal batch")
        audio_batch = audio_batch.to(device)
        
        video_batch = batch.get("video")
        if video_batch is None:
            raise ValueError("Missing 'video' in multimodal batch")
        video_batch = video_batch.to(device)
        
        with torch.no_grad():
            outputs = model(
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                audio_mfcc=audio_batch,
                video_frames=video_batch,
            )
            fused = _extract_main_tensor(outputs, "fusion")
        
        logger.info("OK Fusion model forward pass successful")
        logger.info("  Fused output shape: %s", tuple(fused.shape))
        
        return "PASSED"
        
    except Exception as e:
        logger.error(f"FAIL Fusion model test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return "FAILED"


def main():
    """Run all smoke tests."""
    logger.info("="*60)
    logger.info("SMOKE TEST: Pipeline Verification")
    logger.info("="*60)
    
    # Load config
    cfg = load_config()
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg_dict = _load_yaml_config(cfg_path)
    
    # Set seed
    seed = cfg_dict.get("project", {}).get("seed", 42)
    set_global_seed(seed)
    
    # Device
    device = torch.device(cfg.device.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Run tests
    results = {}
    results["text"] = test_text_model(device, cfg_dict)
    results["audio"] = test_audio_model(device, cfg_dict, cfg)
    results["video"] = test_video_model(device, cfg_dict)
    results["fusion"] = test_fusion_model(device, cfg_dict, cfg)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, status in results.items():
        if status is True:
            status_str = "PASSED"
        elif status is False:
            status_str = "FAILED"
        else:
            status_str = str(status)
        logger.info(f"{test_name.upper()}: {status_str}")
        if status_str == "FAILED":
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("OK All smoke tests passed!")
        return 0
    else:
        logger.error("FAIL Some smoke tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

