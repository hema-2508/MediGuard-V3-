"""
dataset.py
----------
Loads the DrugBankDDI / BioSNAPDDI csv files, builds a global drug
vocabulary (canonical SMILES -> node index), computes/caches node
features for every unique drug, and exposes train/val/test/external
pair lists as (idx_a, idx_b, label) triples ready for graph_builder
and the trainer.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import Config
from feature_generator import FeatureGenerator, InvalidSMILESError, canonicalize_smiles
from utils import get_logger, save_pickle, load_pickle, set_seed

logger = get_logger("dataset")


@dataclass
class PairSplit:
    """A set of (drug_a_idx, drug_b_idx, label) interaction pairs."""

    idx_a: np.ndarray
    idx_b: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class DDIData:
    """Container holding everything downstream modules need."""

    vocab: Dict[str, int]  # canonical SMILES -> node index
    feature_matrix: np.ndarray  # (num_nodes, NODE_FEATURE_DIM)
    train: PairSplit
    val: PairSplit
    test: PairSplit
    external: PairSplit = field(default=None)


def _read_ddi_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {Config.CSV_LABEL_COL, Config.CSV_SMILES1_COL, Config.CSV_SMILES2_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.dropna(subset=[Config.CSV_LABEL_COL, Config.CSV_SMILES1_COL, Config.CSV_SMILES2_COL])
    df[Config.CSV_LABEL_COL] = df[Config.CSV_LABEL_COL].astype(int)
    return df.reset_index(drop=True)


def _canonicalize_column(df: pd.DataFrame, col: str) -> pd.Series:
    canon = []
    valid_mask = []
    for smi in df[col].astype(str):
        try:
            canon.append(canonicalize_smiles(smi))
            valid_mask.append(True)
        except InvalidSMILESError:
            canon.append(None)
            valid_mask.append(False)
    return pd.Series(canon, index=df.index), pd.Series(valid_mask, index=df.index)


def _clean_and_canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize both SMILES columns and drop rows RDKit can't parse."""
    c1, v1 = _canonicalize_column(df, Config.CSV_SMILES1_COL)
    c2, v2 = _canonicalize_column(df, Config.CSV_SMILES2_COL)
    df = df.copy()
    df["canon_smile1"] = c1
    df["canon_smile2"] = c2
    valid = v1 & v2
    n_dropped = (~valid).sum()
    if n_dropped:
        logger.warning(f"Dropping {n_dropped} rows with unparsable SMILES.")
    return df.loc[valid].reset_index(drop=True)


def build_vocab(smiles_lists: List[pd.Series]) -> Dict[str, int]:
    """Build a canonical-SMILES -> node-index vocabulary from any number
    of SMILES series (train + test + external, both columns each)."""
    vocab: Dict[str, int] = {}
    for series in smiles_lists:
        for smi in series:
            if smi not in vocab:
                vocab[smi] = len(vocab)
    return vocab


def build_feature_matrix(vocab: Dict[str, int], feature_gen: FeatureGenerator) -> np.ndarray:
    """Compute the (num_nodes, NODE_FEATURE_DIM) feature matrix, indexed
    exactly by `vocab`."""
    num_nodes = len(vocab)
    feat_dim = Config.NODE_FEATURE_DIM
    matrix = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    idx_to_smiles = {idx: smi for smi, idx in vocab.items()}
    for idx in range(num_nodes):
        smi = idx_to_smiles[idx]
        matrix[idx] = feature_gen.get_features(smi)
    return matrix


def _pairs_from_df(df: pd.DataFrame, vocab: Dict[str, int]) -> PairSplit:
    idx_a = df["canon_smile1"].map(vocab).to_numpy(dtype=np.int64)
    idx_b = df["canon_smile2"].map(vocab).to_numpy(dtype=np.int64)
    labels = df[Config.CSV_LABEL_COL].to_numpy(dtype=np.float32)
    return PairSplit(idx_a=idx_a, idx_b=idx_b, labels=labels)


# Artifacts required at inference time (no CSV reads).
INFERENCE_ARTIFACT_PATHS = {
    "vocab": Config.VOCAB_CACHE_PATH,
    "graph": Config.GRAPH_CACHE_PATH,
    "feature_cache": Config.FEATURE_CACHE_PATH,
    "descriptor_scaler": Config.SCALER_CACHE_PATH,
    "model_checkpoint": Config.BEST_MODEL_PATH,
}


def _missing_inference_artifacts() -> List[str]:
    return [name for name, path in INFERENCE_ARTIFACT_PATHS.items() if not os.path.exists(path)]


def load_inference_artifacts():
    """Load only the cached artifacts needed for prediction.

    Skips all DrugBank/BioSNAP CSV reads and feature-matrix rebuilds.
    Run ``python main.py train`` first to produce these caches.
    """
    missing = _missing_inference_artifacts()
    if missing:
        details = "\n".join(
            f"  - {name}: {INFERENCE_ARTIFACT_PATHS[name]}"
            for name in missing
        )
        raise FileNotFoundError(
            "Inference artifacts are missing. Train the model first so caches "
            "and checkpoints are created, then retry prediction.\n"
            f"Missing:\n{details}"
        )

    vocab = load_pickle(Config.VOCAB_CACHE_PATH)
    logger.info(f"Loaded cached vocab with {len(vocab)} drugs (inference-only path).")

    logger.info(f"Loading cached graph from {Config.GRAPH_CACHE_PATH}")
    graph = torch.load(Config.GRAPH_CACHE_PATH, weights_only=False)

    return vocab, graph


def load_ddi_data(
    train_csv: str = Config.DRUGBANK_TRAIN_CSV,
    test_csv: str = Config.DRUGBANK_TEST_CSV,
    biosnap_train_csv: str = Config.BIOSNAP_TRAIN_CSV,
    biosnap_test_csv: str = Config.BIOSNAP_TEST_CSV,
    val_split_ratio: float = Config.VAL_SPLIT_RATIO,
    use_cache: bool = True,
) -> DDIData:
    """Full data-loading pipeline:

    1. Read + canonicalize DrugBank train/test csvs (and BioSNAP external,
       if provided/available).
    2. Build a single global drug vocabulary spanning all three datasets
       so every drug maps to one persistent node index / feature row.
    3. Carve a stratified validation split out of the DrugBank training
       data (used for early stopping / model selection).
    4. Compute (and cache) Morgan fingerprint + descriptor features for
       every unique drug.
    """
    set_seed(Config.SEED)

    logger.info(f"Reading DrugBank train csv from {train_csv}")
    train_df = _clean_and_canonicalize(_read_ddi_csv(train_csv))

    logger.info(f"Reading DrugBank test csv from {test_csv}")
    test_df = _clean_and_canonicalize(_read_ddi_csv(test_csv))

    logger.info(f"Reading BioSNAP train csv from {biosnap_train_csv}")
    biosnap_train_df = _clean_and_canonicalize(_read_ddi_csv(biosnap_train_csv))

    logger.info(f"Reading BioSNAP test csv from {biosnap_test_csv}")
    biosnap_test_df = _clean_and_canonicalize(_read_ddi_csv(biosnap_test_csv))

    external_df = pd.concat(
        [biosnap_train_df, biosnap_test_df],
        ignore_index=True
    )

    # ---- stratified train/val split (validation used only for early
    # stopping / checkpoint selection, never for graph edges at test time
    # in a leaking way beyond what is standard for transductive setups)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=val_split_ratio,
        random_state=Config.SEED,
        stratify=train_df[Config.CSV_LABEL_COL].to_numpy(),
    )
    train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

    # ---- global vocabulary across all splits (features must exist for
    # every drug we might ever look up, even ones only seen at test time)
    smiles_series = [
        train_df["canon_smile1"], train_df["canon_smile2"],
        test_df["canon_smile1"], test_df["canon_smile2"],
    ]
    if external_df is not None:
        smiles_series += [external_df["canon_smile1"], external_df["canon_smile2"]]

    if use_cache:
        try:
            vocab = load_pickle(Config.VOCAB_CACHE_PATH)
            logger.info(f"Loaded cached vocab with {len(vocab)} drugs.")
        except (FileNotFoundError, EOFError):
            vocab = build_vocab(smiles_series)
            save_pickle(vocab, Config.VOCAB_CACHE_PATH)
    else:
        vocab = build_vocab(smiles_series)
        save_pickle(vocab, Config.VOCAB_CACHE_PATH)

    logger.info(f"Global drug vocabulary size: {len(vocab)}")

    # ---- features
    feature_gen = FeatureGenerator()
    # Fit descriptor scaler once on the union of unique training drugs only
    # (standard practice: no test/external leakage into the scaler stats).
    unique_train_smiles = pd.unique(
        pd.concat([train_df["canon_smile1"], train_df["canon_smile2"]])
    ).tolist()
    if feature_gen._desc_mean is None:
        logger.info("Fitting descriptor scaler on DrugBank training molecules...")
        feature_gen.fit_descriptor_scaler(unique_train_smiles)

    feature_matrix = build_feature_matrix(vocab, feature_gen)
    feature_gen.save_cache()
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")

    train_pairs = _pairs_from_df(train_split_df, vocab)
    val_pairs = _pairs_from_df(val_split_df, vocab)
    test_pairs = _pairs_from_df(test_df, vocab)
    external_pairs = _pairs_from_df(external_df, vocab) if external_df is not None else None

    logger.info(
        f"Split sizes -> train: {len(train_pairs)}, val: {len(val_pairs)}, "
        f"test: {len(test_pairs)}, external: {0 if external_pairs is None else len(external_pairs)}"
    )

    return DDIData(
        vocab=vocab,
        feature_matrix=feature_matrix,
        train=train_pairs,
        val=val_pairs,
        test=test_pairs,
        external=external_pairs,
    )