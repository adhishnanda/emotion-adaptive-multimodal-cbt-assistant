"""Debug script to test IEMOCAP multimodal dataset loading."""

from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import load_config
from src.data.iemocap_multimodal_dataset import load_iemocap_multimodal_dataset
from src.utils.logging_utils import get_logger


def main():
    """Test loading IEMOCAP multimodal dataset."""
    logger = get_logger(__name__)
    cfg = load_config()
    
    logger.info("=" * 60)
    logger.info("Testing IEMOCAP Multimodal Dataset")
    logger.info("=" * 60)
    
    # Test different modality combinations
    test_configs = [
        {"modalities": ["text"], "name": "Text-only"},
        {"modalities": ["audio"], "name": "Audio-only"},
        {"modalities": ["video"], "name": "Video-only"},
        {"modalities": ["text", "audio"], "name": "Text + Audio"},
        {"modalities": ["text", "video"], "name": "Text + Video"},
        {"modalities": ["audio", "video"], "name": "Audio + Video"},
        {"modalities": ["text", "audio", "video"], "name": "All modalities"},
    ]
    
    for test_config in test_configs:
        modalities = test_config["modalities"]
        name = test_config["name"]
        
        logger.info(f"\n{'-' * 60}")
        logger.info(f"Testing: {name}")
        logger.info(f"Modalities: {modalities}")
        logger.info(f"{'-' * 60}")
        
        try:
            # Load dataset for train split
            dataset = load_iemocap_multimodal_dataset(
                cfg=cfg,
                modalities=modalities,
                split="train",
            )
            
            logger.info(f"Dataset size: {len(dataset)}")
            logger.info(f"Emotion labels: {dataset.emotion_labels}")
            logger.info(f"Number of labels: {dataset.num_labels}")
            
            # Try to load a few samples
            if len(dataset) > 0:
                logger.info("\nSample 0:")
                sample = dataset[0]
                for key, value in sample.items():
                    if isinstance(value, str):
                        logger.info(f"  {key}: {value[:100] if len(value) > 100 else value}")
                    elif hasattr(value, "shape"):
                        logger.info(f"  {key}: tensor with shape {value.shape}, dtype={value.dtype}")
                    else:
                        logger.info(f"  {key}: {type(value).__name__}")
                
                if len(dataset) > 1:
                    logger.info("\nSample 1:")
                    sample = dataset[1]
                    for key, value in sample.items():
                        if isinstance(value, str):
                            logger.info(f"  {key}: {value[:100] if len(value) > 100 else value}")
                        elif hasattr(value, "shape"):
                            logger.info(f"  {key}: tensor with shape {value.shape}, dtype={value.dtype}")
                        else:
                            logger.info(f"  {key}: {type(value).__name__}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
    
    # Test splits
    logger.info(f"\n{'-' * 60}")
    logger.info("Testing splits")
    logger.info(f"{'-' * 60}")
    
    for split in ["train", "val", "test"]:
        try:
            dataset = load_iemocap_multimodal_dataset(
                cfg=cfg,
                modalities=["text"],  # Use text-only for split testing
                split=split,
            )
            logger.info(f"{split}: {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Error loading {split} split: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Debug complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

