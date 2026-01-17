"""Build a CSV index for IEMOCAP multimodal dataset.

This script scans the IEMOCAP directory structure and creates a unified index
with all modalities (text, audio, video) aligned by utterance ID.
"""

from pathlib import Path
import sys
import re
from typing import Dict, List, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.config import load_config, PROJECT_ROOT
from src.utils.logging_utils import get_logger


# Emotion label mapping
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

VALID_EMOTIONS = set(EMOTION_MAP.keys())


def load_emotion_labels(session_dir: Path, session_name: str) -> Dict[str, str]:
    """Load emotion labels from EmoEvaluation files."""
    emotion_labels = {}
    emo_eval_dir = session_dir / "dialog" / "EmoEvaluation"
    
    if not emo_eval_dir.exists():
        return emotion_labels
    
    txt_files = list(emo_eval_dir.glob("*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip section headers
                    if line.startswith("[") and line.endswith("]") and len(line.split()) == 1:
                        continue
                    
                    # Remove timestamp brackets
                    cleaned_line = re.sub(r"\[[^\]]*\]", "", line).strip()
                    if not cleaned_line:
                        continue
                    
                    parts = cleaned_line.split()
                    if len(parts) >= 2:
                        utterance_id = parts[0]
                        emotion = parts[1].lower()
                        emotion = re.sub(r"[\[\]]", "", emotion).strip()
                        
                        if emotion in VALID_EMOTIONS and utterance_id:
                            if not utterance_id.startswith(session_name):
                                utterance_id = f"{session_name}_{utterance_id}"
                            emotion_labels[utterance_id] = emotion
        except Exception as e:
            get_logger(__name__).warning(f"Error parsing {txt_file}: {e}")
            continue
    
    return emotion_labels


def load_transcriptions(session_dir: Path, session_name: str) -> Dict[str, str]:
    """Load transcriptions from transcription files."""
    transcriptions = {}
    trans_dir = session_dir / "dialog" / "transcriptions"
    
    if not trans_dir.exists():
        return transcriptions
    
    txt_files = list(trans_dir.glob("*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(None, 1)
                    if len(parts) >= 1:
                        utterance_id = parts[0].strip()
                        text = parts[1].strip() if len(parts) >= 2 else ""
                        
                        if not utterance_id.startswith(session_name):
                            utterance_id = f"{session_name}_{utterance_id}"
                        
                        transcriptions[utterance_id] = text
        except Exception as e:
            get_logger(__name__).warning(f"Error parsing {txt_file}: {e}")
            continue
    
    return transcriptions


def find_media_files(session_dir: Path, session_name: str, media_type: str) -> Dict[str, Path]:
    """Find audio or video files for a session.
    
    Args:
        session_dir: Path to session directory
        session_name: Session name (e.g., "Session1")
        media_type: "wav" or "avi" or "avi_divx"
    
    Returns:
        Dictionary mapping utterance_id to media file path
    """
    media_paths = {}
    logger = get_logger(__name__)
    
    # Try different possible directory structures (dialog first, as per IEMOCAP structure)
    possible_dirs = [
        session_dir / "dialog" / media_type,  # Standard IEMOCAP structure
        session_dir / "dialog" / f"{media_type}_divx",  # For video
        session_dir / "sentences" / media_type,
        session_dir / "sentences" / f"{media_type}_divx",
        session_dir / "sentences" / media_type / session_name,  # Nested structure
        session_dir / "sentences" / f"{media_type}_divx" / session_name,  # Nested structure
    ]
    
    media_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists():
            media_dir = dir_path
            logger.debug(f"Found {media_type} directory: {media_dir}")
            break
    
    if media_dir is None:
        logger.warning(f"No {media_type} directory found for {session_name}")
        return media_paths
    
    # Find all media files recursively
    if media_type == "wav":
        media_files = list(media_dir.rglob("*.wav"))
    elif media_type in ["avi", "avi_divx"]:
        media_files = list(media_dir.rglob("*.avi"))
    else:
        return media_paths
    
    logger.debug(f"Found {len(media_files)} {media_type} files in {media_dir}")
    
    for media_file in media_files:
        # Extract utterance_id from filename (without extension)
        # IEMOCAP files are typically named like: "Ses01F_impro01_F000.wav"
        # But utterance IDs in transcriptions might be like: "Ses01F_impro01_F000" or "Session1_Ses01F_impro01_F000"
        file_stem = media_file.stem
        
        # Remove session prefix if present in filename
        base_stem = file_stem
        if file_stem.startswith(session_name + "_"):
            base_stem = file_stem[len(session_name) + 1:]
        
        # Store multiple ID variants for matching (both with and without session prefix)
        # 1. Base stem (without session prefix) - e.g., "Ses01F_impro01_F000"
        media_paths[base_stem] = media_file
        
        # 2. With session prefix - e.g., "Session1_Ses01F_impro01_F000"
        prefixed_id = f"{session_name}_{base_stem}"
        media_paths[prefixed_id] = media_file
        
        # 3. Original filename if different from base
        if file_stem != base_stem:
            media_paths[file_stem] = media_file
    
    logger.info(f"  Mapped {len(set(media_paths.values()))} unique {media_type} files to {len(media_paths)} ID variants")
    return media_paths


def assign_split(session_num: int) -> str:
    """Assign train/val/test split based on session number.
    
    Standard IEMOCAP split:
    - Sessions 1-3: train
    - Session 4: val
    - Session 5: test
    """
    if session_num <= 3:
        return "train"
    elif session_num == 4:
        return "val"
    else:  # session_num == 5
        return "test"


def build_media_file_index(session_dir: Path, extensions: list) -> Dict[str, Path]:
    """Build a dictionary mapping file basename (stem) to full path for a session.
    
    Args:
        session_dir: Path to session directory
        extensions: List of extensions to search for (e.g., [".wav", ".avi"])
    
    Returns:
        Dictionary mapping basename (stem) to Path
    """
    media_files = {}
    
    if not session_dir.exists():
        return media_files
    
    # Search for all files with given extensions recursively
    for ext in extensions:
        for media_file in session_dir.rglob(f"*{ext}"):
            # Use stem (filename without extension) as key
            basename = media_file.stem
            # Store first match (or overwrite if multiple matches exist)
            if basename not in media_files:
                media_files[basename] = media_file
    
    return media_files


def build_index(root_dir: Path) -> pd.DataFrame:
    """Build the multimodal index DataFrame."""
    logger = get_logger(__name__)
    logger.info(f"Building IEMOCAP multimodal index from {root_dir}")
    
    all_records = []
    
    # Process each session (Session1 through Session5)
    for session_num in range(1, 6):
        session_name = f"Session{session_num}"
        session_dir = root_dir / session_name
        
        if not session_dir.exists():
            logger.warning(f"Session directory not found: {session_dir}")
            continue
        
        logger.info(f"Processing {session_name}...")
        
        # Preload all audio files for this session (once per session)
        logger.info(f"  Scanning audio files in {session_name}...")
        audio_files = build_media_file_index(session_dir, [".wav"])
        logger.info(f"  Found {len(audio_files)} audio files")
        
        # Preload dialog-level video files (videos are stored per dialog, not per utterance)
        video_files = {}
        video_root = session_dir / "dialog" / "avi" / "DivX"
        if video_root.exists():
            for avi_file in video_root.rglob("*.avi"):
                stem = avi_file.stem  # e.g., "Ses04F_impro01", "Ses04F_impro01_F", etc.
                # Store first match (or overwrite if multiple matches exist)
                if stem not in video_files:
                    video_files[stem] = avi_file
        logger.info(f"  Found {len(video_files)} dialog-level video files in {video_root}")
        
        # Load all data for this session
        emotion_labels = load_emotion_labels(session_dir, session_name)
        transcriptions = load_transcriptions(session_dir, session_name)
        
        # Get all unique utterance IDs
        all_utterance_ids = set(emotion_labels.keys()) | set(transcriptions.keys())
        
        # Track matches for logging
        audio_matches = 0
        video_matches = 0
        both_matches = 0
        
        # Create records
        for utterance_id in all_utterance_ids:
            emotion = emotion_labels.get(utterance_id)
            
            # Only include utterances with valid emotion labels
            if emotion is None or emotion not in VALID_EMOTIONS:
                continue
            
            # Normalize emotion
            normalized_emotion = EMOTION_MAP[emotion]
            
            # Parse utterance_id to get base_name for audio lookup
            # Format: "Session1_Ses01F_impro04_M017" -> base_name="Ses01F_impro04_M017"
            if "_" in utterance_id:
                _, base_name = utterance_id.split("_", 1)
            else:
                # Fallback: assume utterance_id is the base_name
                base_name = utterance_id
            
            # Lookup audio file by basename (utterance-level)
            audio_path = ""
            audio_file = audio_files.get(base_name)
            if audio_file and audio_file.exists():
                # Store relative path from root_dir (IEMOCAP_full_release)
                audio_path = audio_file.relative_to(root_dir).as_posix()
                audio_matches += 1
            
            # Lookup video file by dialog_id (dialog-level)
            # Extract dialog_id from utterance_id: "Ses04F_impro01_M000" -> "Ses04F_impro01"
            video_path = ""
            parts = base_name.split("_")
            if len(parts) >= 2:
                dialog_id = "_".join(parts[:2])  # e.g., "Ses04F_impro01"
            else:
                dialog_id = base_name  # Fallback
            
            # Try to find video file matching this dialog
            video_file = None
            # Exact stem match first
            if dialog_id in video_files:
                video_file = video_files[dialog_id]
            else:
                # Fallback: any video whose stem starts with dialog_id
                candidates = [p for stem, p in video_files.items() if stem.startswith(dialog_id)]
                if candidates:
                    video_file = candidates[0]
            
            if video_file and video_file.exists():
                # Store relative path from root_dir (IEMOCAP_full_release)
                video_path = video_file.relative_to(root_dir).as_posix()
                video_matches += 1
            
            if audio_path and video_path:
                both_matches += 1
            
            record = {
                "utterance_id": utterance_id,
                "text": transcriptions.get(utterance_id, ""),
                "emotion": normalized_emotion,
                "audio_path": audio_path,  # Empty string if not found
                "video_path": video_path,  # Empty string if not found
                "session_id": session_name,
                "split": assign_split(session_num),
            }
            
            all_records.append(record)
        
        logger.info(f"  Found {len([r for r in all_records if r['session_id'] == session_name])} utterances")
        logger.info(f"  Matched {audio_matches} audio files, {video_matches} video files, {both_matches} with both")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Initialize audio_path and video_path as strings (not NaN)
    df["audio_path"] = df["audio_path"].fillna("").astype(str)
    df["video_path"] = df["video_path"].fillna("").astype(str)
    
    logger.info(f"Total utterances in index: {len(df)}")
    
    return df


def main():
    """Main function to build and save the index."""
    cfg = load_config()
    logger = get_logger(__name__)
    
    # Set up paths
    root_dir = PROJECT_ROOT / cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"
    output_path = PROJECT_ROOT / cfg.paths.processed_dir / "iemocap_multimodal_index.csv"
    
    if not root_dir.exists():
        raise FileNotFoundError(f"IEMOCAP root directory not found: {root_dir}")
    
    # Build index
    df = build_index(root_dir)
    
    # Ensure audio_path and video_path are strings (not NaN)
    df["audio_path"] = df["audio_path"].fillna("").astype(str)
    df["video_path"] = df["video_path"].fillna("").astype(str)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved index to {output_path}")
    
    # Print statistics using same logic as inspect script
    logger.info("\nIndex Statistics:")
    logger.info(f"  Total rows: {len(df)}")
    
    if "split" in df.columns:
        logger.info(f"  Rows in test split: {len(df[df['split'] == 'test'])}")
        logger.info(f"  Rows in train split: {len(df[df['split'] == 'train'])}")
        logger.info(f"  Rows in val split: {len(df[df['split'] == 'val'])}")
    
    # Use same "non-empty" definition as dataset and inspect script
    audio_col = df["audio_path"].fillna("").astype(str)
    video_col = df["video_path"].fillna("").astype(str)
    
    audio_nonempty = (audio_col.str.strip() != "").sum()
    video_nonempty = (video_col.str.strip() != "").sum()
    both_nonempty = ((audio_col.str.strip() != "") & (video_col.str.strip() != "")).sum()
    
    logger.info(f"\nRows with non-empty audio_path: {audio_nonempty}")
    logger.info(f"Rows with non-empty video_path: {video_nonempty}")
    logger.info(f"Rows with both audio_path and video_path: {both_nonempty}")
    
    # Detailed modality statistics
    has_text = df['text'].notna() & (df['text'] != '')
    has_audio = (audio_col.str.strip() != "")
    has_video = (video_col.str.strip() != "")
    
    logger.info(f"\nModality combinations:")
    logger.info(f"  Text only: {(has_text & ~has_audio & ~has_video).sum()}")
    logger.info(f"  Text + Audio: {(has_text & has_audio & ~has_video).sum()}")
    logger.info(f"  Text + Video: {(has_text & ~has_audio & has_video).sum()}")
    logger.info(f"  Text + Audio + Video: {(has_text & has_audio & has_video).sum()}")
    logger.info(f"  Audio only: {(~has_text & has_audio & ~has_video).sum()}")
    logger.info(f"  Video only: {(~has_text & ~has_audio & has_video).sum()}")
    logger.info(f"  Audio + Video: {(~has_text & has_audio & has_video).sum()}")
    
    logger.info(f"\nSplit distribution:")
    logger.info(f"  {df['split'].value_counts().to_dict()}")
    logger.info(f"\nEmotion distribution:")
    logger.info(f"  {df['emotion'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

