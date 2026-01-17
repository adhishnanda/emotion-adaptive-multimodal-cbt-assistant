"""Base dataset class for emotion recognition datasets."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset, ABC):
    """Base class for emotion recognition datasets."""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_length: int = 512,
        emotion_labels: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset directory
            split: Dataset split (train, dev, test)
            max_length: Maximum sequence length for text
            emotion_labels: List of emotion labels
        """
        self.data_path = data_path
        self.split = split
        self.max_length = max_length
        
        if emotion_labels is None:
            self.emotion_labels = [
                "neutral", "joy", "sadness", "anger",
                "fear", "surprise", "disgust"
            ]
        else:
            self.emotion_labels = emotion_labels
        
        self.label_to_id = {label: idx for idx, label in enumerate(self.emotion_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        self.data = self._load_data()
    
    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from files.
        
        Returns:
            List of data samples
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with model inputs and labels
        """
        pass
    
    def get_label_id(self, label: str) -> int:
        """Convert emotion label to ID."""
        return self.label_to_id.get(label, 0)
    
    def get_label(self, label_id: int) -> str:
        """Convert label ID to emotion label."""
        return self.id_to_label.get(label_id, "neutral")
    
    def get_num_labels(self) -> int:
        """Get number of emotion labels."""
        return len(self.emotion_labels)


