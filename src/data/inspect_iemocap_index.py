"""Debug script to inspect IEMOCAP multimodal index CSV."""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.config import load_config, PROJECT_ROOT

def main():
    """Inspect the IEMOCAP multimodal index CSV."""
    cfg = load_config()
    
    # Load the same CSV used by IEMOCAPMultimodalDataset
    index_path = PROJECT_ROOT / cfg.paths.processed_dir / "iemocap_multimodal_index.csv"
    
    if not index_path.exists():
        print(f"ERROR: Index file not found: {index_path}")
        return
    
    print(f"Loading index from: {index_path}")
    df = pd.read_csv(index_path)
    
    print("\n" + "="*60)
    print("CSV STRUCTURE")
    print("="*60)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
    
    print("\n" + "="*60)
    print("SAMPLE ROWS (first 5)")
    print("="*60)
    if all(col in df.columns for col in ["utterance_id", "audio_path", "video_path", "split"]):
        print(df[["utterance_id", "audio_path", "video_path", "split"]].head(5).to_string())
    else:
        print("Warning: Some expected columns missing")
        print(df.head(5).to_string())
    
    # Normalize path columns (same as dataset)
    if "audio_path" in df.columns:
        audio_col = df["audio_path"].fillna("").astype(str)
    else:
        audio_col = pd.Series([""] * len(df), dtype=str)
        print("\nWARNING: audio_path column not found in CSV")
    
    if "video_path" in df.columns:
        video_col = df["video_path"].fillna("").astype(str)
    else:
        video_col = pd.Series([""] * len(df), dtype=str)
        print("\nWARNING: video_path column not found in CSV")
    
    # Compute statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    total_rows = len(df)
    print(f"\nTotal rows: {total_rows}")
    
    if "split" in df.columns:
        rows_test = len(df[df["split"] == "test"])
        rows_train = len(df[df["split"] == "train"])
        rows_val = len(df[df["split"] == "val"])
        print(f"Rows in test split: {rows_test}")
        print(f"Rows in train split: {rows_train}")
        print(f"Rows in val split: {rows_val}")
    else:
        print("WARNING: split column not found in CSV")
        rows_test = 0
        rows_train = 0
    
    # Count non-empty paths (same logic as dataset)
    audio_nonempty = (audio_col.str.strip() != "").sum()
    video_nonempty = (video_col.str.strip() != "").sum()
    both_nonempty = ((audio_col.str.strip() != "") & (video_col.str.strip() != "")).sum()
    
    print(f"\nRows with non-empty audio_path: {audio_nonempty}")
    print(f"Rows with non-empty video_path: {video_nonempty}")
    print(f"Rows with both audio_path and video_path: {both_nonempty}")
    
    # Check file existence
    print("\n" + "="*60)
    print("FILE EXISTENCE CHECKS")
    print("="*60)
    
    # Get data root (same as dataset)
    data_root = PROJECT_ROOT / cfg.paths.raw_dir / "iemocap" / "IEMOCAP_full_release"
    print(f"\nData root: {data_root}")
    print(f"Data root exists: {data_root.exists()}")
    
    # Check first 5 non-empty audio paths
    print("\n--- First 5 non-empty audio_path values ---")
    audio_nonempty_mask = audio_col.str.strip() != ""
    audio_nonempty_rows = df[audio_nonempty_mask].head(5)
    
    if len(audio_nonempty_rows) > 0:
        for idx, row in audio_nonempty_rows.iterrows():
            audio_path_str = str(row["audio_path"]).strip()
            print(f"\nRow {idx}:")
            print(f"  audio_path: {audio_path_str}")
            
            if audio_path_str:
                try:
                    audio_path = Path(audio_path_str)
                    # Handle relative paths
                    if not audio_path.is_absolute():
                        audio_path = PROJECT_ROOT / audio_path
                    exists = audio_path.exists()
                    print(f"  Path exists: {exists}")
                    if not exists:
                        print(f"  Full path checked: {audio_path}")
                except Exception as e:
                    print(f"  Error checking path: {e}")
            else:
                print(f"  Path exists: False (empty string)")
    else:
        print("No rows with non-empty audio_path found")
    
    # Check first 5 non-empty video paths
    print("\n--- First 5 non-empty video_path values ---")
    video_nonempty_mask = video_col.str.strip() != ""
    video_nonempty_rows = df[video_nonempty_mask].head(5)
    
    if len(video_nonempty_rows) > 0:
        for idx, row in video_nonempty_rows.iterrows():
            video_path_str = str(row["video_path"]).strip()
            print(f"\nRow {idx}:")
            print(f"  video_path: {video_path_str}")
            
            if video_path_str:
                try:
                    video_path = Path(video_path_str)
                    # Handle relative paths
                    if not video_path.is_absolute():
                        video_path = PROJECT_ROOT / video_path
                    exists = video_path.exists()
                    print(f"  Path exists: {exists}")
                    if not exists:
                        print(f"  Full path checked: {video_path}")
                except Exception as e:
                    print(f"  Error checking path: {e}")
            else:
                print(f"  Path exists: False (empty string)")
    else:
        print("No rows with non-empty video_path found")
    
    # Summary for test split
    if "split" in df.columns:
        print("\n" + "="*60)
        print("TEST SPLIT STATISTICS")
        print("="*60)
        df_test = df[df["split"] == "test"].copy()
        if len(df_test) > 0:
            audio_col_test = df_test["audio_path"].fillna("").astype(str)
            video_col_test = df_test["video_path"].fillna("").astype(str)
            
            audio_test_nonempty = (audio_col_test.str.strip() != "").sum()
            video_test_nonempty = (video_col_test.str.strip() != "").sum()
            both_test_nonempty = ((audio_col_test.str.strip() != "") & (video_col_test.str.strip() != "")).sum()
            
            print(f"\nTest split rows: {len(df_test)}")
            print(f"Test split rows with non-empty audio_path: {audio_test_nonempty}")
            print(f"Test split rows with non-empty video_path: {video_test_nonempty}")
            print(f"Test split rows with both: {both_test_nonempty}")
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

