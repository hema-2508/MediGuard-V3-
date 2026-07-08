"""
utils.py
--------
Shared utilities for Model-2: reproducibility, logging, early stopping,
checkpoint helpers, spaCy text preprocessing, and BIO tag utilities.
"""

import json
import logging
import os
import random
import re
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from spacy.language import Language

from config import Config


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int = Config.SEED) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if Config.CUDNN_BENCHMARK and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_device_info(logger: logging.Logger, device: torch.device = Config.DEVICE) -> None:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info(f"Using GPU: {name} (cuda:{idx}, {total_mem_gb:.1f} GB total memory)")
    else:
        logger.warning(
            "No CUDA GPU detected/selected — running on CPU. "
            "Training and inference will be slower."
        )


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def get_logger(name: str = "medicine_ner", log_to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

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
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
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


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------
def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ----------------------------------------------------------------------
# spaCy preprocessing
# ----------------------------------------------------------------------
_NLP: Optional["Language"] = None


def get_spacy_nlp(model_name: str = Config.SPACY_MODEL) -> "Language":
    """Load spaCy once and reuse across calls."""
    global _NLP
    if _NLP is not None:
        return _NLP

    import spacy

    try:
        _NLP = spacy.load(model_name, disable=["ner", "parser", "tagger", "lemmatizer"])
    except OSError:
        _NLP = spacy.blank("en")
        if "sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP


_OCR_FIXES = (
    (re.compile(r"\b0D\b"), "OD"),
    (re.compile(r"\bBID\b", re.I), "BD"),
    (re.compile(r"\bTID\b", re.I), "TDS"),
    (re.compile(r"\bQID\b", re.I), "QDS"),
    (re.compile(r"\s+"), " "),
)


def preprocess_prescription_text(text: str, nlp: Optional["Language"] = None) -> str:
    """Normalize OCR / prescription text before NER."""
    if not text or not text.strip():
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if Config.NORMALIZE_WHITESPACE:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    for pattern, replacement in _OCR_FIXES:
        cleaned = pattern.sub(replacement, cleaned)

    if Config.LOWERCASE_FOR_INFERENCE:
        cleaned = cleaned.lower()

    return cleaned


def tokenize_words(text: str, nlp: Optional["Language"] = None) -> List[str]:
    """Tokenize prescription text using whitespace splits.

    Training annotations use whitespace-delimited tokens (e.g. ``Tab.``,
    ``500``, ``mg``). Generic spaCy tokenization would split ``Tab.`` into
    ``Tab`` and ``.``, causing train/inference skew, so we keep the same
    convention here after OCR normalization.
    """
    cleaned = preprocess_prescription_text(text, nlp=nlp)
    if not cleaned:
        return []
    return [token for token in cleaned.split() if token.strip()]


# ----------------------------------------------------------------------
# BIO tag helpers
# ----------------------------------------------------------------------
def bio_to_entity_type(tag: str) -> Optional[str]:
    if tag == "O" or "-" not in tag:
        return None
    return tag.split("-", 1)[1]


def labels_to_spans(
    words: List[str],
    labels: List[str],
    confidences: Optional[List[float]] = None,
) -> List[Dict]:
    """Convert word-level BIO labels into entity span dicts."""
    spans: List[Dict] = []
    current_type: Optional[str] = None
    current_words: List[str] = []
    current_conf: List[float] = []

    def flush() -> None:
        nonlocal current_type, current_words, current_conf
        if current_type and current_words:
            conf = float(np.mean(current_conf)) if current_conf else 0.0
            spans.append(
                {
                    "label": current_type,
                    "text": " ".join(current_words),
                    "confidence": conf,
                }
            )
        current_type = None
        current_words = []
        current_conf = []

    for idx, label in enumerate(labels):
        entity_type = bio_to_entity_type(label)
        is_begin = label.startswith("B-")
        is_inside = label.startswith("I-")

        if entity_type is None:
            flush()
            continue

        if is_begin or (is_inside and entity_type != current_type):
            flush()
            current_type = entity_type
            current_words = [words[idx]]
            current_conf = [confidences[idx]] if confidences else []
        elif is_inside and entity_type == current_type:
            current_words.append(words[idx])
            if confidences:
                current_conf.append(confidences[idx])

    flush()
    return spans


def spans_to_medicines(spans: List[Dict]) -> List[Dict]:
    """Group extracted entity spans into per-medicine records."""
    medicines: List[Dict] = []
    current: Optional[Dict] = None

    def _is_valid_medicine_name(name: str) -> bool:
        stripped = name.strip()
        if len(stripped) < 2:
            return False
        if all(not char.isalnum() for char in stripped):
            return False
        return True

    def start_medicine(name_span: Dict) -> None:
        nonlocal current
        if not _is_valid_medicine_name(name_span["text"]):
            return
        if current is not None:
            medicines.append(current)
        current = {
            "name": name_span["text"],
            "strength": "",
            "dosage": "",
            "frequency": "",
            "duration": "",
            "route": "",
            "confidence": name_span["confidence"],
            "_field_confidences": {"name": name_span["confidence"]},
        }

    def attach_field(field: str, span: Dict) -> None:
        nonlocal current
        if current is None:
            current = {
                "name": "",
                "strength": "",
                "dosage": "",
                "frequency": "",
                "duration": "",
                "route": "",
                "confidence": span["confidence"],
                "_field_confidences": {},
            }
        if not current[field]:
            current[field] = span["text"]
            current["_field_confidences"][field] = span["confidence"]

    for span in spans:
        label = span["label"]
        if label == "MEDICINE":
            start_medicine(span)
        elif label in Config.ENTITY_TYPES:
            attach_field(label.lower(), span)

    if current is not None:
        medicines.append(current)

    cleaned: List[Dict] = []
    for record in medicines:
        field_confs = list(record.pop("_field_confidences", {}).values())
        if field_confs:
            record["confidence"] = round(float(np.mean(field_confs)), 4)
        else:
            record["confidence"] = 0.0
        if record["name"]:
            cleaned.append(record)

    return cleaned


def align_predictions_to_words(
    word_ids: List[Optional[int]],
    pred_ids: List[int],
    confidences: List[float],
    ignore_index: int = -100,
) -> Tuple[List[str], List[float], List[int]]:
    """Map subword predictions back to the first subword of each word."""
    word_to_label: Dict[int, int] = {}
    word_to_conf: Dict[int, float] = {}

    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == ignore_index:
            continue
        if word_id not in word_to_label:
            word_to_label[word_id] = pred_ids[idx]
            word_to_conf[word_id] = confidences[idx]

    if not word_to_label:
        return [], [], []

    max_word = max(word_to_label)
    aligned_labels = [word_to_label.get(i, 0) for i in range(max_word + 1)]
    aligned_confs = [word_to_conf.get(i, 0.0) for i in range(max_word + 1)]
    return aligned_labels, aligned_confs, list(range(max_word + 1))
