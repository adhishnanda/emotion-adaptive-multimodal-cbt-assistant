"""Evaluate trained text emotion model on MELD test split."""

from pathlib import Path
import json
import os
import yaml

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config.config import load_config, PROJECT_ROOT
from src.utils.logging_utils import get_logger
from src.data.meld_text_dataset import create_dataloaders
from src.models.text_distilbert import DistilBERTEmotionClassifier

# Alias for consistency with requirements
TextDistilBERT = DistilBERTEmotionClassifier
create_meld_dataloaders = create_dataloaders

logger = get_logger("text_eval")


def get_device():
    """
    Get device for evaluation.
    
    - If config["training"]["device"] is set, use that.
    - Else use "cuda" if available, else "cpu".
    
    Returns:
        torch.device object
    """
    cfg = load_config()
    
    # Check for training.device in raw config
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    
    training_cfg = raw_cfg.get("training", {})
    if isinstance(training_cfg, dict) and "device" in training_cfg:
        device_str = training_cfg["device"]
        return torch.device(device_str)
    
    # Fallback to config.device.device (ProjectConfig)
    if hasattr(cfg, 'device') and hasattr(cfg.device, 'device'):
        return cfg.device.device
    
    # Final fallback: cuda if available, else cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_text_model(num_labels: int, cfg) -> TextDistilBERT:
    """
    Build text model with same parameters as train_text.py.
    
    Args:
        num_labels: Number of emotion classes
        cfg: Project configuration
        
    Returns:
        Initialized model on the selected device
    """
    device = get_device()
    
    # Get model parameters from config
    pretrained_name = cfg.text_model.pretrained_name
    dropout = getattr(cfg.text_model, 'dropout', 0.3)
    
    # Use the existing build_text_model function but with custom dropout
    # Since build_text_model doesn't accept dropout, we'll build manually
    from src.models.text_distilbert import DistilBERTConfig, DistilBERTEmotionClassifier
    
    model_cfg = DistilBERTConfig(
        pretrained_name=pretrained_name,
        num_labels=num_labels,
        dropout=dropout,
    )
    
    model = DistilBERTEmotionClassifier(model_cfg)
    model = model.to(device)
    
    return model


def evaluate_text_model():
    """
    Main evaluation function for text emotion model.
    
    Evaluates on MELD test split and saves results to JSON.
    """
    # Load config
    cfg = load_config()
    logger.info("Loaded configuration")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_meld_dataloaders(cfg)
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise
    
    # Infer number of labels
    # Try from config first, then from dataset
    num_labels = None
    
    # Check config for num_emotions
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    
    data_cfg = raw_cfg.get("data", {})
    if "num_emotions" in data_cfg:
        num_labels = int(data_cfg["num_emotions"])
        logger.info(f"Number of labels from config: {num_labels}")
    
    # Fallback to dataset
    if num_labels is None:
        num_labels = test_loader.dataset.num_labels
        logger.info(f"Number of labels from dataset: {num_labels}")
    
    # Build model
    logger.info("Building model...")
    try:
        model = build_text_model(num_labels, cfg)
        logger.info("Model built successfully")
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        raise
    
    # Load best weights
    # Check config for best_text_model path
    paths_cfg = raw_cfg.get("paths", {})
    best_model_path = paths_cfg.get("best_text_model")
    
    if best_model_path is None:
        best_model_path = cfg.paths.models_dir / "text" / "distilbert_best.pt"
    else:
        best_model_path = PROJECT_ROOT / best_model_path
    
    logger.info(f"Loading model weights from: {best_model_path}")
    
    if not best_model_path.exists():
        logger.error(f"Model file not found: {best_model_path}")
        raise FileNotFoundError(f"Model file not found: {best_model_path}")
    
    try:
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise
    
    # Set model to eval mode
    model.eval()
    
    # Collect predictions and true labels
    all_true_labels = []
    all_pred_labels = []
    
    logger.info("Running evaluation on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Get predictions (multi-class: argmax)
            preds = logits.argmax(dim=-1)
            
            # Collect labels
            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")
    
    logger.info(f"Evaluation complete. Total samples: {len(all_true_labels)}")
    
    # Convert to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    # Get label names from dataset
    label_names = test_loader.dataset.EMOTION_LABELS
    
    # Classification report
    class_report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=label_names,
        output_dict=True,
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
    
    # Log summary
    overall_accuracy = class_report["accuracy"]
    macro_f1 = class_report["macro avg"]["f1-score"]
    
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    
    # Save report to JSON
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / "text_eval_report.json"
    
    report_dict = {
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Evaluation report saved to: {report_path}")
    
    # Print per-class metrics
    logger.info("\nPer-class metrics:")
    for label_name in label_names:
        if label_name in class_report:
            metrics = class_report[label_name]
            logger.info(
                f"  {label_name}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1-score']:.4f}"
            )


if __name__ == "__main__":
    evaluate_text_model()

