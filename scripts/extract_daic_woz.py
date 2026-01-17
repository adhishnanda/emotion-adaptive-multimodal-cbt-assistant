"""
Extract DAIC-WOZ participant archives into the project.

This script:
- Reads the official AVEC2017 train/dev/test split CSVs to determine which
  participant IDs to extract
- Extracts participant zip files (e.g., 300_P.zip) into organized session directories
- Is safe to re-run: skips zips that have already been extracted
- Copies the split CSVs into the project for reference

Usage:
    python scripts/extract_daic_woz.py

The script expects the DAIC-WOZ zip files to be located at:
    C:/Users/Adhish/dcapswoz.ict.usc.edu/wwwdaicwoz

Extracted data will be placed in:
    data/raw/daic_woz/sessions/
    data/raw/daic_woz/splits/
"""

from pathlib import Path
import zipfile
import csv
from typing import Set, List

from src.utils.logging_utils import get_logger

logger = get_logger("extract_daic_woz")


def load_split_ids(split_csv_path: Path) -> Set[str]:
    """
    Load participant IDs from an AVEC split CSV.

    Assumes there is a column that contains IDs (e.g. 'Participant_ID' or first column).
    Returns a set of ID strings like '300', '301', ...

    Args:
        split_csv_path: Path to the split CSV file

    Returns:
        Set of participant ID strings
    """
    if not split_csv_path.exists():
        logger.warning(f"Split CSV not found: {split_csv_path}")
        return set()

    ids = set()
    try:
        with open(split_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check if 'Participant_ID' column exists
            if 'Participant_ID' in reader.fieldnames:
                for row in reader:
                    participant_id = row.get('Participant_ID', '').strip()
                    if participant_id:
                        ids.add(participant_id)
            else:
                # Fall back to first column
                for row in reader:
                    first_key = list(row.keys())[0] if row.keys() else None
                    if first_key:
                        participant_id = row.get(first_key, '').strip()
                        if participant_id:
                            ids.add(participant_id)
    except Exception as e:
        logger.error(f"Error reading {split_csv_path}: {e}")
        return set()

    return ids


def extract_participant_zip(
    participant_id: str,
    download_root: Path,
    sessions_dir: Path,
) -> None:
    """
    Extract a single participant zip, e.g. 300_P.zip, into sessions_dir / '300'.

    If the target directory already exists and is non-empty, skip extraction.

    Args:
        participant_id: The participant ID (e.g., '300')
        download_root: Root directory containing the zip files
        sessions_dir: Directory where extracted sessions should be placed
    """
    zip_name = f"{participant_id}_P.zip"
    zip_path = download_root / zip_name

    if not zip_path.exists():
        logger.warning(f"Zip file not found: {zip_path}")
        return

    target_dir = sessions_dir / participant_id

    # Check if already extracted
    if target_dir.exists():
        # Check if directory has files
        if any(target_dir.iterdir()):
            logger.info(f"Skipping ID {participant_id} (already extracted)")
            return

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        logger.info(f"Successfully extracted {zip_name} to {target_dir}")
    except Exception as e:
        logger.error(f"Error extracting {zip_name}: {e}")


def extract_daic_woz() -> None:
    """
    Main function to extract DAIC-WOZ participant archives.

    Reads train/dev/test split CSVs, extracts corresponding participant zips,
    and copies the split CSVs into the project.
    """
    # Configurable path to downloaded DAIC-WOZ zips
    DAIC_DOWNLOAD_ROOT = Path(r"C:/Users/Adhish/dcapswoz.ict.usc.edu/wwwdaicwoz")

    # Infer project root
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "daic_woz"
    SESSIONS_DIR = OUTPUT_ROOT / "sessions"
    SPLITS_DIR = OUTPUT_ROOT / "splits"

    # Create directories if they don't exist
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Define split CSV paths
    train_csv = DAIC_DOWNLOAD_ROOT / "train_split_Depression_AVEC2017.csv"
    dev_csv = DAIC_DOWNLOAD_ROOT / "dev_split_Depression_AVEC2017.csv"
    test_csv = DAIC_DOWNLOAD_ROOT / "test_split_Depression_AVEC2017.csv"

    # Load IDs from split CSVs
    logger.info("Loading participant IDs from split CSVs...")
    train_ids = load_split_ids(train_csv)
    dev_ids = load_split_ids(dev_csv)
    test_ids = load_split_ids(test_csv)

    all_ids = sorted(train_ids | dev_ids | test_ids)
    logger.info(f"Found {len(all_ids)} total participant IDs "
                f"(train: {len(train_ids)}, dev: {len(dev_ids)}, test: {len(test_ids)})")

    # Extract participant zips
    logger.info(f"Extracting participant archives to {SESSIONS_DIR}...")
    for participant_id in all_ids:
        extract_participant_zip(participant_id, DAIC_DOWNLOAD_ROOT, SESSIONS_DIR)

    # Copy split CSVs to SPLITS_DIR
    logger.info(f"Copying split CSVs to {SPLITS_DIR}...")
    for src_csv in [train_csv, dev_csv, test_csv]:
        if src_csv.exists():
            dst_csv = SPLITS_DIR / src_csv.name
            dst_csv.parent.mkdir(parents=True, exist_ok=True)
            if not dst_csv.exists():
                dst_csv.write_bytes(src_csv.read_bytes())
                logger.info(f"Copied {src_csv.name} to {dst_csv}")
            else:
                logger.info(f"Split CSV already exists: {dst_csv}")

    logger.info("Extraction complete!")


if __name__ == "__main__":
    extract_daic_woz()

