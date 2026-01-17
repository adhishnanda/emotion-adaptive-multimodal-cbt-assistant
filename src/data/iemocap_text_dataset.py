"""IEMOCAP text-only dataset loader reusing MELD text dataset utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from config.config import ProjectConfig, load_config
from src.data.iemocap_dataset import load_iemocap_dataset
from src.utils import get_logger, load_pickle, save_pickle


class TextEmotionDataset(Dataset):
    """
    Generic text emotion dataset that works with a DataFrame.
    
    Reusable class for text-based emotion classification tasks.
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        df: pd.DataFrame,
        split_name: str = "train",
        cache_name: Optional[str] = None,
    ):
        """
        Initialize text emotion dataset from DataFrame.

        Args:
            cfg: Project configuration
            df: DataFrame with columns 'Utterance' and 'Emotion'
            split_name: Name of the split (for caching)
            cache_name: Optional custom cache name (defaults to f"text_{split_name}")
        """
        self.cfg = cfg
        self.df = df.copy()
        self.split_name = split_name
        self.cache_name = cache_name or f"text_{split_name}"

        # Setup cache directory
        self.cache_dir = cfg.paths.processed_dir / "text"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)
        except Exception as e:
            get_logger(__name__).warning(f"Failed to load tokenizer: {e}. Using fallback.")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Get unique emotion labels from the dataset
        self.emotion_labels = sorted(self.df["Emotion"].unique().tolist())
        self.EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(self.emotion_labels)}
        self.ID_TO_EMOTION = {idx: emotion for emotion, idx in self.EMOTION_TO_ID.items()}

        # Load and process data
        self.data = self._load_data()

    def _get_cache_path(self) -> Path:
        """Get cache file path for this split."""
        cache_filename = f"{self.cache_name}_maxlen{self.cfg.text_model.max_seq_len}.pkl"
        return self.cache_dir / cache_filename

    def _load_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load data from cache or process from DataFrame."""
        cache_path = self._get_cache_path()

        # Try to load from cache
        if cache_path.exists():
            try:
                get_logger(__name__).info(f"Loading cached data from {cache_path}")
                return load_pickle(cache_path)
            except Exception as e:
                get_logger(__name__).warning(f"Failed to load cache: {e}. Reprocessing...")

        # Process DataFrame
        try:
            processed_data = self._process_dataframe()
        except Exception as e:
            get_logger(__name__).error(f"Failed to process data: {e}")
            return []

        # Save to cache
        try:
            save_pickle(processed_data, cache_path)
            get_logger(__name__).info(f"Cached processed data to {cache_path}")
        except Exception as e:
            get_logger(__name__).warning(f"Failed to save cache: {e}")

        return processed_data

    def _process_dataframe(self) -> List[Dict[str, torch.Tensor]]:
        """Process DataFrame into tokenized tensors."""
        processed = []

        for _, row in self.df.iterrows():
            # Extract text and emotion
            text = str(row.get("Utterance", "")).strip()
            emotion = str(row.get("Emotion", "")).strip()

            # Map emotion to ID
            emotion_id = self.EMOTION_TO_ID.get(emotion, 0)

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
                get_logger(__name__).warning(f"Failed to tokenize text '{text[:50]}...': {e}")
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
        return len(self.emotion_labels)


def _compute_class_weights(dataset: TextEmotionDataset) -> torch.Tensor:
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


def build_iemocap_text_dataframe(root_dir: Path) -> pd.DataFrame:
    """
    Build a pandas DataFrame with columns ['Utterance', 'Emotion']
    from IEMOCAPMultimodalDataset.

    Only keep samples where both text and emotion are non-empty.
    Strip the leading timestamp pattern like:
        [097.8900-102.9600]: Clearly.  You know...
    """
    logger = get_logger(__name__)
    
    # Load the base IEMOCAP dataset
    base_ds = load_iemocap_dataset(root_dir)
    
    rows = []
    for text, emo, audio, video in base_ds:
        if not text or not isinstance(text, str):
            continue
        if emo is None or not isinstance(emo, str):
            continue
        
        # Strip timestamps at the beginning
        cleaned = re.sub(r"^\[[0-9\.\-]+\]:\s*", "", text).strip()
        if not cleaned:
            continue
        
        rows.append({"Utterance": cleaned, "Emotion": emo.strip()})
    
    df = pd.DataFrame(rows)
    logger.info(f"Built DataFrame with {len(df)} samples from IEMOCAP dataset")
    return df


def create_iemocap_text_dataloaders(
    cfg: Optional[ProjectConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for IEMOCAP text-only emotion classification.

    Uses an 80/10/10 split stratified by emotion label.
    Reuses TextEmotionDataset from the MELD pipeline.
    """
    if cfg is None:
        cfg = load_config()
    
    logger = get_logger(__name__)
    
    # Compute root directory
    root_dir = cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"
    
    if not root_dir.exists():
        raise FileNotFoundError(f"IEMOCAP root directory not found: {root_dir}")
    
    # Build DataFrame
    logger.info("Building IEMOCAP text DataFrame...")
    df = build_iemocap_text_dataframe(root_dir)
    
    # Drop rows with NaN in Emotion or Utterance
    df = df.dropna(subset=["Emotion", "Utterance"])
    logger.info(f"After dropping NaN: {len(df)} samples")
    
    # Log label distribution for debugging and analysis
    label_col = "Emotion"
    if label_col in df.columns:
        logger.info("Label distribution in full IEMOCAP dataset:\n%s", df[label_col].value_counts())
    else:
        logger.warning("Label column '%s' not found in IEMOCAP DataFrame. Available columns: %s",
                       label_col, list(df.columns))
    
    # Split into train / temp (80/20)
    logger.info("Splitting dataset: 80% train, 20% temp...")
    train_df, temp_df = train_test_split(
        df,
        train_size=0.8,
        test_size=0.2,
        stratify=df["Emotion"],
        random_state=cfg.seed,
    )
    
    # Split temp into val / test (50/50 of temp = 10/10 of total)
    logger.info("Splitting temp: 50% val, 50% test...")
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        test_size=0.5,
        stratify=temp_df["Emotion"],
        random_state=cfg.seed,
    )
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Create datasets
    logger.info("Creating TextEmotionDataset instances...")
    train_dataset = TextEmotionDataset(
        cfg=cfg,
        df=train_df,
        split_name="train",
        cache_name="iemocap_train",
    )
    val_dataset = TextEmotionDataset(
        cfg=cfg,
        df=val_df,
        split_name="val",
        cache_name="iemocap_val",
    )
    test_dataset = TextEmotionDataset(
        cfg=cfg,
        df=test_df,
        split_name="test",
        cache_name="iemocap_test",
    )
    
    # Create samplers (reuse from MELD)
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
            logger.info("Using WeightedRandomSampler for class balancing")
        except Exception as e:
            logger.warning(f"Failed to create WeightedRandomSampler: {e}. Using default sampler.")
    
    # Create dataloaders (reuse pin_memory logic from MELD)
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


__all__ = ["TextEmotionDataset", "build_iemocap_text_dataframe", "create_iemocap_text_dataloaders"]

