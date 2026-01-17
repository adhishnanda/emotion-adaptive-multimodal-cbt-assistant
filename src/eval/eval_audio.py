"""Evaluate trained audio emotion/depression model on DAIC-WOZ test split."""

from pathlib import Path
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config.config import load_config
from src.utils.logging_utils import get_logger
from src.data.daicwoz_audio_dataset import create_daicwoz_datasets
from src.models.audio_cnn_lstm import build_audio_model

logger = get_logger("audio_eval")


def evaluate_audio_model():
    """Main evaluation function for audio model on DAIC-WOZ."""
    cfg = load_config()
    logger.info("Loaded configuration")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating DAIC-WOZ dataloaders...")
    try:
        _, val_dataset, _ = create_daicwoz_datasets(cfg)
        if val_dataset is None:
            logger.error("Validation dataset for DAIC-WOZ not found. Aborting evaluation.")
            return
        
        # Create DataLoader for the validation set
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.text_model.batch_size, # Re-using batch size from text_model config
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(f"Evaluating on development (validation) set with {len(val_dataset)} samples.")

    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
        raise

    num_classes = val_dataset.num_labels
    num_mfcc = val_dataset.n_mfcc
    logger.info(f"Number of classes from dataset: {num_classes}")
    logger.info(f"Number of MFCCs from dataset: {num_mfcc}")

    logger.info("Building audio model...")
    model = build_audio_model(num_labels=num_classes, num_mfcc=num_mfcc, device=device)

    model_path = cfg.paths.models_dir / "audio" / "audio_cnn_lstm_best.pt"
    logger.info(f"Loading model weights from: {model_path}")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}", exc_info=True)
        raise

    model.eval()

    all_true_labels = []
    all_pred_labels = []

    logger.info("Running evaluation on DAIC-WOZ development (validation) set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Audio Eval]"):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            loss, logits = model(features, labels)
            preds = logits.argmax(dim=-1)

            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())

    logger.info(f"Evaluation complete. Total samples: {len(all_true_labels)}")

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    logger.info("Computing metrics...")
    label_names = ["non-depressed", "depressed"]

    class_report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=label_names,
        output_dict=True,
    )

    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)

    overall_accuracy = class_report["accuracy"]
    macro_f1 = class_report["macro avg"]["f1-score"]

    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "audio_eval_report.json"

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
    evaluate_audio_model()
