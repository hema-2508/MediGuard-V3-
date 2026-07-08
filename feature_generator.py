"""
feature_generator.py
---------------------
Turns a SMILES string into a fixed-length numerical feature vector:

    [ Morgan fingerprint (radius=2, 1024 bits) | RDKit physicochemical
      descriptors (normalized) ]

Features are cached to disk (keyed by canonical SMILES) so repeated
runs / repeated appearances of the same drug across train/test/
external datasets never recompute RDKit chemistry twice.
"""

from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors

from config import Config
from utils import get_logger, save_pickle, load_pickle

# Silence RDKit's verbose C++ warnings (invalid valence, etc.) - we handle
# invalid SMILES ourselves and log a clean message instead.
RDLogger.DisableLog("rdApp.*")

logger = get_logger("feature_generator")


class InvalidSMILESError(ValueError):
    """Raised when a SMILES string cannot be parsed by RDKit."""


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit's canonical form of a SMILES string.

    Raises InvalidSMILESError if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def _compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute the fixed set of physicochemical descriptors defined in
    Config.DESCRIPTOR_NAMES, in that exact order."""
    values = []
    for name in Config.DESCRIPTOR_NAMES:
        if name == "MolWt":
            values.append(Descriptors.MolWt(mol))
        elif name == "MolLogP":
            values.append(Crippen.MolLogP(mol))
        elif name == "TPSA":
            values.append(rdMolDescriptors.CalcTPSA(mol))
        elif name == "NumHDonors":
            values.append(rdMolDescriptors.CalcNumHBD(mol))
        elif name == "NumHAcceptors":
            values.append(rdMolDescriptors.CalcNumHBA(mol))
        elif name == "NumRotatableBonds":
            values.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        elif name == "RingCount":
            values.append(rdMolDescriptors.CalcNumRings(mol))
        elif name == "NumAromaticRings":
            values.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        elif name == "FractionCSP3":
            values.append(rdMolDescriptors.CalcFractionCSP3(mol))
        elif name == "HeavyAtomCount":
            values.append(mol.GetNumHeavyAtoms())
        elif name == "NumValenceElectrons":
            values.append(Descriptors.NumValenceElectrons(mol))
        elif name == "MolMR":
            values.append(Crippen.MolMR(mol))
        elif name == "NumSaturatedRings":
            values.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
        elif name == "NumAliphaticRings":
            values.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        elif name == "BalabanJ":
            values.append(Descriptors.BalabanJ(mol))
        else:
            raise ValueError(f"Unknown descriptor requested: {name}")
    arr = np.array(values, dtype=np.float64)
    # Guard against NaN/inf produced by degenerate molecules (e.g. BalabanJ
    # is undefined for acyclic / disconnected graphs in rare edge cases).
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _compute_morgan_fp(mol: Chem.Mol) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=Config.MORGAN_RADIUS,
        nBits=Config.MORGAN_NBITS,
        useChirality=Config.USE_CHIRALITY,
    )
    arr = np.zeros((Config.MORGAN_NBITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class FeatureGenerator:
    """Computes and caches molecular feature vectors for SMILES strings.

    Feature vector layout (Config.NODE_FEATURE_DIM long):
        [0 : MORGAN_NBITS)                          -> Morgan fingerprint bits
        [MORGAN_NBITS : MORGAN_NBITS + n_desc)       -> raw RDKit descriptors

    Descriptor standardization (z-score) is fit once on the union of all
    training molecules and cached, then reused at inference time for any
    new molecule so train/test/external features stay on the same scale.
    """

    def __init__(
        self,
        cache_path: str = Config.FEATURE_CACHE_PATH,
        scaler_path: str = Config.SCALER_CACHE_PATH,
    ):
        self.cache_path = cache_path
        self.scaler_path = scaler_path
        self._cache: Dict[str, np.ndarray] = self._load_cache()
        self._desc_mean: Optional[np.ndarray] = None
        self._desc_std: Optional[np.ndarray] = None
        self._load_scaler()

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------
    def _load_cache(self) -> Dict[str, np.ndarray]:
        try:
            cache = load_pickle(self.cache_path)
            logger.info(f"Loaded feature cache with {len(cache)} entries from {self.cache_path}")
            return cache
        except (FileNotFoundError, EOFError):
            logger.info("No existing feature cache found, starting fresh.")
            return {}

    def save_cache(self) -> None:
        save_pickle(self._cache, self.cache_path)
        logger.info(f"Saved feature cache with {len(self._cache)} entries to {self.cache_path}")

    def _load_scaler(self) -> None:
        try:
            scaler = load_pickle(self.scaler_path)
            self._desc_mean = scaler["mean"]
            self._desc_std = scaler["std"]
            logger.info(f"Loaded descriptor scaler from {self.scaler_path}")
        except (FileNotFoundError, EOFError):
            self._desc_mean = None
            self._desc_std = None

    def save_scaler(self) -> None:
        if self._desc_mean is None or self._desc_std is None:
            raise RuntimeError("Scaler has not been fit yet; call fit_descriptor_scaler first.")
        save_pickle({"mean": self._desc_mean, "std": self._desc_std}, self.scaler_path)
        logger.info(f"Saved descriptor scaler to {self.scaler_path}")

    # ------------------------------------------------------------------
    # Raw (pre-scaling) feature computation
    # ------------------------------------------------------------------
    def _compute_raw(self, smiles: str) -> np.ndarray:
        """Compute [morgan_fp | raw_descriptors] for a single SMILES,
        using the cache when available. Raises InvalidSMILESError for
        unparsable SMILES."""
        canon = canonicalize_smiles(smiles)
        if canon in self._cache:
            return self._cache[canon]

        mol = Chem.MolFromSmiles(canon)
        fp = _compute_morgan_fp(mol)
        desc = _compute_descriptors(mol)
        raw = np.concatenate([fp, desc]).astype(np.float64)
        self._cache[canon] = raw
        return raw

    # ------------------------------------------------------------------
    # Descriptor scaler fitting (call once, on the training set only)
    # ------------------------------------------------------------------
    def fit_descriptor_scaler(self, smiles_list: List[str]) -> None:
        n_fp = Config.MORGAN_NBITS
        desc_matrix = []
        for smi in smiles_list:
            try:
                raw = self._compute_raw(smi)
            except InvalidSMILESError as e:
                logger.warning(str(e))
                continue
            desc_matrix.append(raw[n_fp:])
        desc_matrix = np.vstack(desc_matrix)
        mean = desc_matrix.mean(axis=0)
        std = desc_matrix.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid divide-by-zero for constant descriptors
        self._desc_mean = mean
        self._desc_std = std
        self.save_scaler()
        self.save_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_features(self, smiles: str) -> np.ndarray:
        """Return the final (fingerprint | standardized-descriptors)
        feature vector for a SMILES string, as float32."""
        raw = self._compute_raw(smiles)
        n_fp = Config.MORGAN_NBITS
        fp = raw[:n_fp]
        desc = raw[n_fp:]
        if self._desc_mean is not None and self._desc_std is not None:
            desc = (desc - self._desc_mean) / self._desc_std
        return np.concatenate([fp, desc]).astype(np.float32)

    def get_features_batch(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Compute features for many SMILES at once, skipping (and
        logging) any that RDKit cannot parse."""
        out = {}
        n_bad = 0
        for smi in smiles_list:
            try:
                out[smi] = self.get_features(smi)
            except InvalidSMILESError as e:
                n_bad += 1
                logger.warning(str(e))
        if n_bad:
            logger.warning(f"Skipped {n_bad} unparsable SMILES out of {len(smiles_list)}.")
        return out

    def __len__(self) -> int:
        return len(self._cache)