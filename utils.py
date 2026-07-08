"""
utils.py
--------
Shared utilities: reproducibility, logging, early stopping,
checkpoint save/load, and small helper functions used across the
project.
"""

import os
import random
import logging
import pickle
from datetime import datetime

import numpy as np
import torch

from config import Config


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int = Config.SEED) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if Config.CUDNN_BENCHMARK and torch.cuda.is_available():
        # Trade strict determinism for throughput: shapes are static in
        # this project (fixed feature/hidden dims), so benchmark mode is
        # a safe, meaningful speedup on GPU.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_device_info(logger: logging.Logger, device: torch.device = Config.DEVICE) -> None:
    """Print a clear, unambiguous summary of what compute device will be
    used for this run, including GPU name/memory when on CUDA."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info(f"Using GPU: {name} (cuda:{idx}, {total_mem_gb:.1f} GB total memory)")
        logger.info(
            f"CUDA available: {torch.cuda.is_available()} | "
            f"device count: {torch.cuda.device_count()} | "
            f"cudnn.benchmark: {torch.backends.cudnn.benchmark}"
        )
    else:
        logger.warning(
            "No CUDA GPU detected/selected — running on CPU. "
            "Training will be significantly slower."
        )


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def get_logger(name: str = "ddi", log_to_file: bool = True) -> logging.Logger:
    """Create (or fetch) a logger that writes to stdout and, optionally,
    to a timestamped file under Config.LOG_DIR."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_to_file:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(os.path.join(Config.LOG_DIR, f"{name}_{ts}.log"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------------------
# Early stopping
# ----------------------------------------------------------------------
class EarlyStopping:
    """Stops training when a monitored validation metric stops improving.

    Works for metrics that should be maximized (e.g. roc_auc, f1) or
    minimized (e.g. loss) via the `mode` argument.
    """

    def __init__(
        self,
        patience: int = Config.EARLY_STOPPING_PATIENCE,
        min_delta: float = Config.EARLY_STOPPING_MIN_DELTA,
        mode: str = "max",
    ):
        assert mode in ("max", "min")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update internal state with the latest validation score.

        Returns True if this score is the new best score.
        """
        if self.best_score is None:
            self.best_score = score
            return True

        improved = (
            (score > self.best_score + self.min_delta)
            if self.mode == "max"
            else (score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    if map_location is None:
        map_location = Config.DEVICE
    return torch.load(path, map_location=map_location, weights_only=False)


# ----------------------------------------------------------------------
# Pickle helpers
# ----------------------------------------------------------------------
def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# Simple running-average tracker (useful for training loss logging)
# ----------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0