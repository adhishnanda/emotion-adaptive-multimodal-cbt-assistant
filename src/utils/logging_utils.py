import logging
from pathlib import Path

from config.config import load_config


def get_logger(name: str) -> logging.Logger:
    cfg = load_config()
    log_dir = cfg.paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers in notebooks
    if logger.handlers:
        return logger

    fh = logging.FileHandler(log_dir / f"{name}.log")
    ch = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
