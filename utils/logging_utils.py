import logging
import os
from typing import Optional


def configure_logger(save_dir: str, name: str = "retrieval_forecasting", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(os.path.join(save_dir, "run.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_logger(name: str = "retrieval_forecasting") -> logging.Logger:
    return logging.getLogger(name)
