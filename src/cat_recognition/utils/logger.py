from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(output_dir: str | Path | None = None, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("cat_recognition")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output_dir is not None and rank == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
