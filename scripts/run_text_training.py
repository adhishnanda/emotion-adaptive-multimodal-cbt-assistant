from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import config and src.*
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import load_config
from src.utils.seed_utils import set_global_seed


def main() -> None:
    cfg = load_config()
    seed = set_global_seed()

    print("Project:", cfg.name)
    print("Using device:", cfg.device.device)
    print("Seed:", seed)
    print("Raw data dir:", cfg.paths.raw_dir)


if __name__ == "__main__":
    main()
