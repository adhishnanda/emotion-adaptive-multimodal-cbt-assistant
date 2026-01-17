"""Full video training pipeline for IEMOCAP multi-class emotion classifier."""

from pathlib import Path
import sys
import os
from typing import Dict, Tuple

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import yaml

from tqdm import tqdm

from config.config import load_config, PROJECT_ROOT
from src.utils.seed_utils import set_global_seed
from src.utils.logging_utils import get_logger

from src.data.iemocap_multimodal_dataset import load_iemocap_multimodal_dataset
from src.models.video_iemocap_resnet import build_iemocap_video_model


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.

    Args:
        labels: list of int labels
        num_classes: total number of classes

    Returns:
        torch.tensor of shape (num_classes,)
    """
    # Count how many times each class appears
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1.0

    # For class c, weight = 1.0 / max(count_c, 1)
    weights = 1.0 / torch.clamp(counts, min=1.0)

    return weights


def create_video_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create video dataloaders for IEMOCAP training, validation, and test.

    Args:
        cfg: Project configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = get_logger("iemocap_video_data")

    # Load raw config to access iemocap_video section
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    video_cfg = raw_cfg.get("iemocap_video", {})
    image_size = video_cfg.get("image_size", 224)
    batch_size = video_cfg.get("batch_size", 8)

    # Load datasets
    train_dataset = load_iemocap_multimodal_dataset(
        cfg=cfg,
        modalities=["video"],
        split="train",
        image_size=image_size,
        is_training=True,  # Apply training transforms with augmentation
    )

    val_dataset = load_iemocap_multimodal_dataset(
        cfg=cfg,
        modalities=["video"],
        split="val",
        image_size=image_size,
        is_training=False,  # No augmentation for validation
    )

    test_dataset = load_iemocap_multimodal_dataset(
        cfg=cfg,
        modalities=["video"],
        split="test",
        image_size=image_size,
        is_training=False,
    )

    if len(train_dataset) == 0:
        logger.warning("Train IEMOCAP video dataset is empty. Cannot train video model.")
        return None, None, None

    # Get emotion labels and create label mapping
    emotion_labels = train_dataset.emotion_labels
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    num_classes = len(emotion_labels)

    logger.info(f"Number of emotion classes: {num_classes}")
    logger.info(f"Emotion labels: {emotion_labels}")

    # Extract labels from train dataset for class weighting
    train_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        emotion = sample["emotion"]
        label_id = emotion_to_id[emotion]
        train_labels.append(label_id)

    # Compute class weights
    class_weights = compute_class_weights(train_labels, num_classes)

    # Compute sample weights
    sample_weights = [class_weights[label] for label in train_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
    )

    # Create val_loader
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
        )

    # Create test_loader
    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available() if torch.cuda.is_available() else False,
        )

    # Store label mapping in dataset for later use
    train_dataset.emotion_to_id = emotion_to_id
    train_dataset.id_to_emotion = {v: k for k, v in emotion_to_id.items()}

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    logger,
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        Tuple of (average training loss, training accuracy)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Video Train]")
    for batch in progress_bar:
        # Extract video frames and emotions
        video_frames = batch["video"]  # List of tensors
        emotions = batch["emotion"]  # List of strings

        # Convert emotions to label IDs
        emotion_to_id = train_loader.dataset.emotion_to_id
        labels = torch.tensor([emotion_to_id[emotion] for emotion in emotions], dtype=torch.long).to(device)

        # Stack video frames: (batch, C, H, W)
        frames = torch.stack(video_frames).to(device)

        # Forward pass
        loss, logits = model(frames, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track loss and accuracy
        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item(), "acc": correct / total_samples})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate model on validation/test set.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, accuracy, per-class metrics dict)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    # Per-class tracking
    class_correct = {}
    class_total = {}
    emotion_to_id = data_loader.dataset.emotion_to_id
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="[Val]")
        for batch in progress_bar:
            # Extract video frames and emotions
            video_frames = batch["video"]  # List of tensors
            emotions = batch["emotion"]  # List of strings

            # Convert emotions to label IDs
            labels = torch.tensor([emotion_to_id[emotion] for emotion in emotions], dtype=torch.long).to(device)

            # Stack video frames: (batch, C, H, W)
            frames = torch.stack(video_frames).to(device)

            # Forward pass
            loss, logits = model(frames, labels)

            # Track loss
            batch_size = frames.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()

            # Per-class metrics
            for label, pred in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                emotion = id_to_emotion[label]
                if emotion not in class_total:
                    class_total[emotion] = 0
                    class_correct[emotion] = 0
                class_total[emotion] += 1
                if label == pred:
                    class_correct[emotion] += 1

            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    # Compute per-class accuracies
    per_class_metrics = {}
    for emotion in class_total:
        per_class_metrics[emotion] = class_correct[emotion] / class_total[emotion] if class_total[emotion] > 0 else 0.0

    return avg_loss, accuracy, per_class_metrics


def train_video_iemocap():
    """
    Main training function for IEMOCAP video emotion classification.
    """
    cfg = load_config()
    logger = get_logger("video_training_iemocap")
    set_global_seed()

    logger.info("=" * 60)
    logger.info("IEMOCAP Video Emotion Classification Training")
    logger.info("=" * 60)

    # Load raw config to access iemocap_video section
    cfg_path_env = os.getenv("EMOTION_CBT_CONFIG")
    cfg_path = Path(cfg_path_env) if cfg_path_env else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Get IEMOCAP video config
    video_cfg = raw_cfg.get("iemocap_video", {})
    batch_size = video_cfg.get("batch_size", 8)
    learning_rate = video_cfg.get("learning_rate", 1e-4)
    num_epochs = video_cfg.get("num_epochs", 20)
    backbone = video_cfg.get("backbone", "resnet18")
    freeze_backbone = video_cfg.get("freeze_backbone", False)

    logger.info(f"Backbone: {backbone}")
    logger.info(f"Freeze backbone: {freeze_backbone}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")

    # Create dataloaders
    logger.info("Creating IEMOCAP video dataloaders...")
    train_loader, val_loader, test_loader = create_video_dataloaders(cfg)

    if train_loader is None:
        logger.error("Failed to create video dataloaders. Exiting.")
        return

    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.emotion_labels)
    logger.info(f"Number of emotion classes: {num_classes}")

    # Build model
    logger.info("Building IEMOCAP video model...")
    try:
        model = build_iemocap_video_model(
            num_classes=num_classes,
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            device=cfg.device.device,
        )
        logger.info(f"Model built and moved to {cfg.device.device}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        raise

    # Create optimizer
    # Only optimize parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)
    logger.info(f"Optimizer: AdamW with lr={learning_rate}")

    # Compute total steps
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    # Create LambdaLR scheduler
    def lr_lambda(step: int) -> float:
        """Learning rate schedule with warmup and linear decay."""
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(f"LR scheduler: Linear warmup ({warmup_steps} steps) + decay")

    # Training setup
    best_val_loss = float("inf")
    best_model_path = cfg.paths.models_dir / "video" / "video_iemocap_resnet_best.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Best model will be saved to: {best_model_path}")

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device.device,
            epoch=epoch,
            logger=logger,
        )
        logger.info(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} ({train_acc*100:.2f}%)")

        # Evaluate
        if val_loader is not None:
            val_loss, val_acc, per_class_metrics = evaluate_epoch(
                model=model,
                data_loader=val_loader,
                device=cfg.device.device,
            )
            logger.info(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            logger.info("Per-class validation accuracies:")
            for emotion, acc in per_class_metrics.items():
                logger.info(f"  {emotion}: {acc:.4f} ({acc*100:.2f}%)")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                try:
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"[SAVED] New best model (val_loss={val_loss:.4f}) to {best_model_path}")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

    # Final logging
    logger.info(f"\n{'='*50}")
    logger.info("IEMOCAP video training complete.")
    if val_loader is not None:
        logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    train_video_iemocap()

