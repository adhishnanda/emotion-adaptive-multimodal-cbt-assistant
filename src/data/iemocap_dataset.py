"""IEMOCAP multimodal dataset loader for emotion recognition."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from torch.utils.data import Dataset

from src.utils.logging_utils import get_logger


class IEMOCAPMultimodalDataset(Dataset):
    """IEMOCAP multimodal dataset with text, audio, and video support."""

    # Emotion label mapping from IEMOCAP codes to readable labels
    EMOTION_MAP = {
        "ang": "anger",
        "hap": "happy",
        "sad": "sadness",
        "neu": "neutral",
        "exc": "excited",
        "fru": "frustration",
        "dis": "disgust",
        "fea": "fear",
        "sur": "surprise",
    }

    # Valid emotion codes to keep
    VALID_EMOTIONS = set(EMOTION_MAP.keys())

    def __init__(self, root_dir: Path):
        """
        Initialize IEMOCAP multimodal dataset.

        Args:
            root_dir: Path to IEMOCAP dataset root directory (data/raw/iemocap/)
        """
        self.root_dir = Path(root_dir)
        self.logger = get_logger(__name__)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"IEMOCAP root directory not found: {self.root_dir}")

        # Load all utterances
        self.utterances = self._load_utterances()
        self.logger.info(f"Loaded {len(self.utterances)} utterances from IEMOCAP dataset")

    def _load_utterances(self) -> List[Dict]:
        """
        Load all utterances from all sessions.

        Returns:
            List of utterance dictionaries with keys:
                - utterance_id: str
                - text: str
                - emotion: str (normalized)
                - audio_path: Path | None
                - video_path: Path | None
        """
        all_utterances = []

        # Process each session (Session1 through Session5)
        for session_num in range(1, 6):
            session_name = f"Session{session_num}"
            session_dir = self.root_dir / session_name

            if not session_dir.exists():
                self.logger.warning(f"Session directory not found: {session_dir}")
                continue

            # Load emotion labels
            emotion_labels = self._load_emotion_labels(session_dir, session_name)

            # Load transcriptions
            transcriptions = self._load_transcriptions(session_dir, session_name)

            # Get audio and video paths
            audio_paths = self._get_media_paths(session_dir, "wav", session_name)
            video_paths = self._get_media_paths(session_dir, "avi", session_name)

            # Combine all information
            all_utterance_ids = set(emotion_labels.keys()) | set(transcriptions.keys())

            for utterance_id in all_utterance_ids:
                emotion = emotion_labels.get(utterance_id)
                # Only include utterances with valid emotion labels
                if emotion is None or emotion not in self.VALID_EMOTIONS:
                    continue

                # Normalize emotion label
                normalized_emotion = self.EMOTION_MAP[emotion]

                utterance = {
                    "utterance_id": utterance_id,
                    "text": transcriptions.get(utterance_id, ""),
                    "emotion": normalized_emotion,
                    "audio_path": audio_paths.get(utterance_id),
                    "video_path": video_paths.get(utterance_id),
                }

                all_utterances.append(utterance)

            self.logger.info(
                f"Loaded {len([u for u in all_utterances if u['utterance_id'].startswith(session_name)])} "
                f"utterances from {session_name}"
            )

        return all_utterances

    def _load_emotion_labels(self, session_dir: Path, session_name: str) -> Dict[str, str]:
        """
        Load emotion labels from EmoEvaluation files.

        Args:
            session_dir: Path to session directory
            session_name: Name of the session (for logging)

        Returns:
            Dictionary mapping utterance_id to emotion code
        """
        emotion_labels = {}

        emo_eval_dir = session_dir / "dialog" / "EmoEvaluation"
        if not emo_eval_dir.exists():
            self.logger.warning(f"EmoEvaluation directory not found: {emo_eval_dir}")
            return emotion_labels

        # Find all .txt files in EmoEvaluation directory
        txt_files = list(emo_eval_dir.glob("*.txt"))
        if not txt_files:
            self.logger.warning(f"No .txt files found in {emo_eval_dir}")
            return emotion_labels

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Skip section headers (lines that are just brackets)
                        if line.startswith("[") and line.endswith("]") and len(line.split()) == 1:
                            continue

                        # IEMOCAP format variations:
                        # 1. [timestamp] utterance_id emotion
                        # 2. utterance_id emotion
                        # 3. [timestamp] utterance_id [emotion]
                        
                        # Try to extract utterance_id and emotion
                        # Remove brackets from timestamps first
                        cleaned_line = re.sub(r"\[[^\]]*\]", "", line).strip()
                        
                        if not cleaned_line:
                            continue

                        parts = cleaned_line.split()
                        if len(parts) >= 2:
                            utterance_id = parts[0]
                            emotion = parts[1].lower()
                            
                            # Clean emotion (remove brackets if present)
                            emotion = re.sub(r"[\[\]]", "", emotion).strip()
                            
                            # Skip invalid emotions (including "xxx", "oth", "xxx", etc.)
                            if emotion in self.VALID_EMOTIONS and utterance_id:
                                # Ensure utterance_id includes session prefix for uniqueness
                                if not utterance_id.startswith(session_name):
                                    utterance_id = f"{session_name}_{utterance_id}"

                                emotion_labels[utterance_id] = emotion

            except Exception as e:
                self.logger.warning(f"Error parsing emotion file {txt_file}: {e}")
                continue

        self.logger.debug(f"Loaded {len(emotion_labels)} emotion labels from {session_name}")
        return emotion_labels

    def _load_transcriptions(self, session_dir: Path, session_name: str) -> Dict[str, str]:
        """
        Load transcriptions from transcription files.

        Args:
            session_dir: Path to session directory
            session_name: Name of the session (for logging)

        Returns:
            Dictionary mapping utterance_id to transcription text
        """
        transcriptions = {}

        trans_dir = session_dir / "dialog" / "transcriptions"
        if not trans_dir.exists():
            self.logger.warning(f"Transcriptions directory not found: {trans_dir}")
            return transcriptions

        # Find all .txt files in transcriptions directory
        txt_files = list(trans_dir.glob("*.txt"))
        if not txt_files:
            self.logger.warning(f"No .txt files found in {trans_dir}")
            return transcriptions

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # IEMOCAP format: utterance_id transcription text
                        # Split on first space or tab
                        parts = line.split(None, 1)

                        if len(parts) >= 1:
                            utterance_id = parts[0].strip()
                            text = parts[1].strip() if len(parts) >= 2 else ""

                            # Ensure utterance_id includes session prefix for uniqueness
                            if not utterance_id.startswith(session_name):
                                utterance_id = f"{session_name}_{utterance_id}"

                            transcriptions[utterance_id] = text

            except Exception as e:
                self.logger.warning(f"Error parsing transcription file {txt_file}: {e}")
                continue

        self.logger.debug(f"Loaded {len(transcriptions)} transcriptions from {session_name}")
        return transcriptions

    def _get_media_paths(
        self, session_dir: Path, media_type: str, session_name: str
    ) -> Dict[str, Path]:
        """
        Get paths to audio or video files.

        Args:
            session_dir: Path to session directory
            media_type: "wav" or "avi"
            session_name: Name of the session (for logging)

        Returns:
            Dictionary mapping utterance_id to media file path
        """
        media_paths = {}

        media_dir = session_dir / "dialog" / media_type
        if not media_dir.exists():
            self.logger.debug(f"{media_type.upper()} directory not found: {media_dir}")
            return media_paths

        # Find all media files
        if media_type == "wav":
            media_files = list(media_dir.rglob("*.wav"))
        elif media_type == "avi":
            media_files = list(media_dir.rglob("*.avi"))
        else:
            return media_paths

        for media_file in media_files:
            # Extract utterance_id from filename (without extension)
            utterance_id = media_file.stem

            # Ensure utterance_id includes session prefix for uniqueness
            if not utterance_id.startswith(session_name):
                utterance_id = f"{session_name}_{utterance_id}"

            media_paths[utterance_id] = media_file

        self.logger.debug(
            f"Found {len(media_paths)} {media_type.upper()} files for {session_name}"
        )
        return media_paths

    def __len__(self) -> int:
        """Return number of utterances in the dataset."""
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Tuple[str, str, Optional[Path], Optional[Path]]:
        """
        Get a single utterance.

        Args:
            idx: Index of the utterance

        Returns:
            Tuple of (text, emotion_label, audio_path, video_path)
            audio_path and video_path may be None if not available
        """
        utterance = self.utterances[idx]
        return (
            utterance["text"],
            utterance["emotion"],
            utterance["audio_path"],
            utterance["video_path"],
        )

    def get_utterance(self, idx: int) -> Dict:
        """
        Get full utterance dictionary.

        Args:
            idx: Index of the utterance

        Returns:
            Full utterance dictionary
        """
        return self.utterances[idx]

    @property
    def emotion_labels(self) -> List[str]:
        """Return list of unique emotion labels in the dataset."""
        unique_emotions = set(u["emotion"] for u in self.utterances)
        return sorted(list(unique_emotions))

    @property
    def num_labels(self) -> int:
        """Return number of unique emotion labels."""
        return len(self.emotion_labels)


def load_iemocap_dataset(root_dir: Path) -> IEMOCAPMultimodalDataset:
    """
    Load IEMOCAP multimodal dataset.

    Args:
        root_dir: Path to IEMOCAP dataset root directory (data/raw/iemocap/)

    Returns:
        IEMOCAPMultimodalDataset instance
    """
    return IEMOCAPMultimodalDataset(root_dir)


__all__ = ["IEMOCAPMultimodalDataset", "load_iemocap_dataset"]

