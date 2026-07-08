"""
config.py
---------
Central configuration for the GraphSAGE Drug-Drug Interaction (DDI)
prediction project. All paths and hyperparameters live here so the
rest of the codebase never hard-codes a magic number or a path.
"""

import os
import torch


class Config:
    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    DRUGBANK_TRAIN_CSV = os.path.join(DATA_DIR, "drugbankddi", "train_clean.csv")
    DRUGBANK_TEST_CSV  = os.path.join(DATA_DIR, "drugbankddi", "test_clean.csv")

    BIOSNAP_TRAIN_CSV = os.path.join(DATA_DIR, "biosnapddi", "train_clean.csv")
    BIOSNAP_TEST_CSV  = os.path.join(DATA_DIR, "biosnapddi", "test_clean.csv")

    # Cached artifacts
    FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, "feature_cache.pkl")
    VOCAB_CACHE_PATH = os.path.join(CACHE_DIR, "drug_vocab.pkl")
    GRAPH_CACHE_PATH = os.path.join(CACHE_DIR, "ddi_graph.pt")
    SCALER_CACHE_PATH = os.path.join(CACHE_DIR, "descriptor_scaler.pkl")

    # Model checkpoints
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pt")

    # CSV schema
    CSV_LABEL_COL = "label"
    CSV_SMILES1_COL = "smile1"
    CSV_SMILES2_COL = "smile2"

    # ------------------------------------------------------------------
    # Molecular featurization
    # ------------------------------------------------------------------
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 1024
    USE_CHIRALITY = True

    # RDKit physicochemical descriptors used alongside the fingerprint.
    DESCRIPTOR_NAMES = [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "NumAromaticRings",
        "FractionCSP3",
        "HeavyAtomCount",
        "NumValenceElectrons",
        "MolMR",
        "NumSaturatedRings",
        "NumAliphaticRings",
        "BalabanJ",
    ]

    NODE_FEATURE_DIM = MORGAN_NBITS + len(DESCRIPTOR_NAMES)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    ADD_SELF_LOOPS = True
    MAKE_EDGES_BIDIRECTIONAL = True

    # ------------------------------------------------------------------
    # GraphSAGE model
    # ------------------------------------------------------------------
    SAGE_HIDDEN_DIM = 128
    SAGE_NUM_LAYERS = 2
    SAGE_DROPOUT = 0.3
    SAGE_AGGR = "mean"  # mean / max / lstm

    # Pair classifier head (operates on [A, B, |A-B|, A*B])
    PAIR_HIDDEN_DIM = 256
    PAIR_DROPOUT = 0.3

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    SEED = 42

    # ------------------------------------------------------------------
    # Device selection (GPU-first)
    # ------------------------------------------------------------------
    # Set to a specific index (e.g. 0) to pin to one GPU, or leave as
    # "cuda" to let PyTorch use the current default CUDA device.
    # Override at the CLI with `--device cuda:0` / `--device cpu`.
    CUDA_DEVICE_INDEX = 0
    DEVICE = torch.device(
        f"cuda:{CUDA_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"
    )
    # cudnn.benchmark speeds up training on GPU when input shapes are
    # stable (true here: fixed node-feature dim, fixed hidden dims).
    # Only takes effect when running on CUDA.
    CUDNN_BENCHMARK = True
    PIN_MEMORY = torch.cuda.is_available()
    NON_BLOCKING = torch.cuda.is_available()

    BATCH_SIZE = 512
    EPOCHS = 200
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 1e-4
    EARLY_STOPPING_METRIC = "roc_auc"  # metric on validation split to monitor
    GRAD_CLIP_NORM = 5.0

    # Train/validation split ratio carved out of DrugBank train_clean.csv
    VAL_SPLIT_RATIO = 0.1

    # Classification threshold applied to sigmoid probabilities
    DECISION_THRESHOLD = 0.5

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    NUM_WORKERS = 0  # PyG in-memory graph -> no need for multi-process loaders

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.CACHE_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)


Config.ensure_dirs()