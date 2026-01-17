"""Evaluate trained text emotion model on IEMOCAP test split."""

from pathlib import Path
import json
import os
import yaml

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config.config import load_config, PROJECT_ROOT
from src.utils.logging_utils import get_logger
from src.data.iemocap_text_dataset import create_iemocap_text_dataloaders
from src.models.text_distilbert import DistilBERTEmotionClassifier

# Alias for consistency
TextDistilBERT = DistilBERTEmotionClassifier

logger = get_logger("text_eval_iemocap")


def get_device():
    """Get device for evaluation."""
    cfg = load_config()
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    
    training_cfg = raw_cfg.get("training", {})
    if isinstance(training_cfg, dict) and "device" in training_cfg:
        return torch.device(training_cfg["device"])
    
    if hasattr(cfg, 'device') and hasattr(cfg.device, 'device'):
        return cfg.device.device
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_text_model(num_labels: int, cfg) -> TextDistilBERT:
    """Build text model with same parameters as training."""
    device = get_device()
    from src.models.text_distilbert import DistilBERTConfig, DistilBERTEmotionClassifier
    
    model_cfg = DistilBERTConfig(
        pretrained_name=cfg.text_model.pretrained_name,
        num_labels=num_labels,
        dropout=getattr(cfg.text_model, 'dropout', 0.3),
    )
    
    model = DistilBERTEmotionClassifier(model_cfg)
    model = model.to(device)
    
    return model


def evaluate_iemocap_text_model():
    """Main evaluation function for IEMOCAP text emotion model."""
    cfg = load_config()
    logger.info("Loaded configuration")
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    logger.info("Creating IEMOCAP dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_iemocap_text_dataloaders(cfg)
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise
    
    num_labels = 9  # Hardcode to match the saved model checkpoint
    logger.info(f"Number of labels from dataset: {test_loader.dataset.num_labels}, but using hardcoded {num_labels} to match checkpoint.")
    
    logger.info("Building model...")
    model = build_text_model(num_labels, cfg)
    
    best_model_path = cfg.paths.models_dir / "text" / "distilbert_iemocap_best.pt"
    
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
    
    model.eval()
    
    all_true_labels = []
    all_pred_labels = []
    
    logger.info("Running evaluation on IEMOCAP test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            preds = logits.argmax(dim=-1)
            
            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
    
    logger.info(f"Evaluation complete. Total samples: {len(all_true_labels)}")
    
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    logger.info("Computing metrics...")
    
    # Hardcode label names to match the 9-label model, ensuring correct report generation
    label_names = ['anger', 'disgust', 'excited', 'fear', 'frustration', 'happy', 'neutral', 'sadness', 'surprise']
    
    class_report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=label_names,
        output_dict=True,
        labels=np.arange(len(label_names))
    )
    
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
    
    overall_accuracy = class_report["accuracy"]
    macro_f1 = class_report["macro avg"]["f1-score"]
    
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "text_eval_report_iemocap.json"
    
    report_dict = {
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Evaluation report saved to: {report_path}")
    
    logger.info("\nPer-class metrics:")
    for label_name in label_names:
        if label_name in class_report:
            metrics = class_report[label_name]
            logger.info(
                f"  {label_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}"
            )


if __name__ == "__main__":
    evaluate_iemocap_text_model()
