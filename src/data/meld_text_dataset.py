"""MELD text-only dataset loader with caching and class balancing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from config.config import ProjectConfig, load_config
from src.utils import load_pickle, save_pickle


class MELDTextDataset(Dataset):
    """MELD text dataset with caching and tokenization."""

    # MELD emotion labels
    EMOTION_LABELS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]
    EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
    ID_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_ID.items()}

    # CSV file mapping
    SPLIT_FILES = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }

    def __init__(
        self,
        cfg: ProjectConfig,
        split: str = "train",
        subset_fraction: Optional[float] = None,
    ):
        """
        Initialize MELD text dataset.

        Args:
            cfg: Project configuration
            split: Dataset split (train, dev, test)
            subset_fraction: Fraction of data to use (only for train split)
        """
        self.cfg = cfg
        self.split = split
        self.subset_fraction = subset_fraction if split == "train" else None

        # Setup paths
        self.raw_dir = cfg.paths.raw_dir / "meld"
        self.cache_dir = cfg.paths.processed_dir / "text"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)
        except Exception as e:
            print(f"[WARN] Failed to load tokenizer: {e}")
            # Fallback to default
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Load data
        self.data = self._load_data()

    def _get_cache_path(self) -> Path:
        """Get cache file path for this split."""
        cache_name = f"meld_{self.split}"
        if self.subset_fraction is not None:
            cache_name += f"_subset{self.subset_fraction:.2f}"
        cache_name += f"_maxlen{self.cfg.text_model.max_seq_len}.pkl"
        return self.cache_dir / cache_name

    def _load_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load data from cache or process from CSV."""
        cache_path = self._get_cache_path()

        # Try to load from cache
        if cache_path.exists():
            try:
                print(f"[INFO] Loading cached data from {cache_path}")
                return load_pickle(cache_path)
            except Exception as e:
                print(f"[WARN] Failed to load cache: {e}. Reprocessing...")

        # Load from CSV
        try:
            raw_data = self._load_from_csv()
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            return []

        # Process and tokenize
        try:
            processed_data = self._process_data(raw_data)
        except Exception as e:
            print(f"[ERROR] Failed to process data: {e}")
            return []

        # Save to cache
        try:
            save_pickle(processed_data, cache_path)
            print(f"[INFO] Cached processed data to {cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

        return processed_data

    def _load_from_csv(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        filename = self.SPLIT_FILES.get(self.split)
        if filename is None:
            raise ValueError(f"Unknown split: {self.split}")

        csv_path = self.raw_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Apply subsetting for training
        if self.subset_fraction is not None and self.split == "train":
            n_samples = int(len(df) * self.subset_fraction)
            df = df.sample(n=n_samples, random_state=self.cfg.seed).reset_index(drop=True)
            print(f"[INFO] Using {n_samples}/{len(df)} samples ({self.subset_fraction:.1%}) for training")

        return df

    def _process_data(self, df: pd.DataFrame) -> List[Dict[str, torch.Tensor]]:
        """Process DataFrame into tokenized tensors."""
        processed = []

        for _, row in df.iterrows():
            # Extract text and emotion
            text = str(row.get("Utterance", "")).strip()
            emotion = str(row.get("Emotion", "neutral")).lower()

            # Map emotion to ID
            emotion_id = self.EMOTION_TO_ID.get(emotion, 0)  # Default to neutral

            # Skip empty text
            if not text:
                continue

            # Tokenize
            try:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.cfg.text_model.max_seq_len,
                    return_tensors="pt",
                )

                processed.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(emotion_id, dtype=torch.long),
                })
            except Exception as e:
                print(f"[WARN] Failed to tokenize text '{text[:50]}...': {e}")
                continue

        return processed

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        return self.data[idx]

    @property
    def num_labels(self) -> int:
        """Return number of emotion labels."""
        return len(self.EMOTION_LABELS)


def _compute_class_weights(dataset: MELDTextDataset) -> torch.Tensor:
    """Compute class weights for balanced sampling."""
    # Count labels
    label_counts = torch.zeros(dataset.num_labels)
    for sample in dataset.data:
        label = sample["labels"].item()
        label_counts[label] += 1

    # Compute weights (inverse frequency)
    total = label_counts.sum()
    weights = total / (label_counts + 1e-6)  # Add small epsilon to avoid division by zero
    weights = weights / weights.sum() * len(weights)  # Normalize

    return weights


def create_dataloaders(
    cfg: Optional[ProjectConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        cfg: Project configuration. If None, loads from default config.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if cfg is None:
        cfg = load_config()

    # Create datasets
    print("[INFO] Creating datasets...")
    train_dataset = MELDTextDataset(
        cfg=cfg,
        split="train",
        subset_fraction=cfg.text_model.train_subset_fraction,
    )
    val_dataset = MELDTextDataset(cfg=cfg, split="dev")
    test_dataset = MELDTextDataset(cfg=cfg, split="test")

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples: {len(val_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}")

    # Create samplers
    train_sampler = None
    if cfg.text_model.class_balance_strategy == "weighted_random_sampler":
        try:
            class_weights = _compute_class_weights(train_dataset)
            # Create sample weights for each sample
            sample_weights = torch.zeros(len(train_dataset))
            for idx, sample in enumerate(train_dataset.data):
                label = sample["labels"].item()
                sample_weights[idx] = class_weights[label]

            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print("[INFO] Using WeightedRandomSampler for class balancing")
        except Exception as e:
            print(f"[WARN] Failed to create WeightedRandomSampler: {e}. Using default sampler.")

    # Create dataloaders
    try:
        use_pin_memory = cfg.device.device.type == "cuda"
    except Exception:
        use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.text_model.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.text_model.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.text_model.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
