"""
config.py
---------
Central configuration for Model-2: Medicine Extraction from
prescription / OCR text. All paths and hyperparameters live here.
"""

import os

import torch


class Config:
    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MEDIGUARD_ROOT = os.path.dirname(PROJECT_ROOT)

    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prescription_ner")
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
    VAL_JSON = os.path.join(DATA_DIR, "val.json")
    TEST_JSON = os.path.join(DATA_DIR, "test.json")

    LABEL_MAP_PATH = os.path.join(CACHE_DIR, "label_map.json")
    TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model")

    # ------------------------------------------------------------------
    # Pretrained biomedical backbone
    # ------------------------------------------------------------------
    # BioBERT is used as the token-classification backbone and warm-start
    # for prescription NER fine-tuning.
    PRETRAINED_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

    # ------------------------------------------------------------------
    # Entity schema (BIO tagging)
    # ------------------------------------------------------------------
    ENTITY_TYPES = (
        "MEDICINE",
        "STRENGTH",
        "DOSAGE",
        "FREQUENCY",
        "DURATION",
        "ROUTE",
    )

    # ------------------------------------------------------------------
    # spaCy preprocessing
    # ------------------------------------------------------------------
    SPACY_MODEL = "en_core_web_sm"
    NORMALIZE_WHITESPACE = True
    LOWERCASE_FOR_INFERENCE = False

    # ------------------------------------------------------------------
    # Tokenization / model
    # ------------------------------------------------------------------
    MAX_SEQ_LENGTH = 256
    STRIDE = 64  # overlap for long prescriptions during inference

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    SEED = 42

    CUDA_DEVICE_INDEX = 0
    DEVICE = torch.device(
        f"cuda:{CUDA_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"
    )
    CUDNN_BENCHMARK = True
    PIN_MEMORY = torch.cuda.is_available()
    NON_BLOCKING = torch.cuda.is_available()

    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    GRAD_CLIP_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-4
    EARLY_STOPPING_METRIC = "f1"

    NUM_WORKERS = 0

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    API_HOST = "0.0.0.0"
    API_PORT = 8001
    API_MAX_BATCH_SIZE = 64

    @classmethod
    def ensure_dirs(cls) -> None:
        for directory in (
            cls.DATA_DIR,
            cls.CACHE_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR,
        ):
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def build_label_list(cls) -> list[str]:
        labels = ["O"]
        for entity in cls.ENTITY_TYPES:
            labels.append(f"B-{entity}")
            labels.append(f"I-{entity}")
        return labels


Config.ensure_dirs()
