"""
predict.py
----------
Inference-time utilities for scoring arbitrary drug pairs with a
trained DDI model, including drugs never seen during training
("cold-start"): such drugs get a feature vector computed on the fly
and are appended to the graph as an isolated node before encoding.
"""

import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from config import Config
from dataset import _missing_inference_artifacts, INFERENCE_ARTIFACT_PATHS
from feature_generator import FeatureGenerator, InvalidSMILESError, canonicalize_smiles
from graph_builder import extend_graph_with_new_nodes
from model import DDIModel
from utils import get_logger, load_checkpoint, load_pickle, log_device_info

logger = get_logger("predict")


class DDIPredictor:
    """Wraps a trained checkpoint + graph + vocab + feature generator
    into a single object that can score new SMILES pairs."""

    def __init__(
        self,
        checkpoint_path: str = Config.BEST_MODEL_PATH,
        graph=None,
        vocab: Dict[str, int] = None,
        device: torch.device = Config.DEVICE,
    ):
        self.device = device
        log_device_info(logger, device)

        missing = _missing_inference_artifacts()
        if missing:
            details = "\n".join(
                f"  - {name}: {INFERENCE_ARTIFACT_PATHS[name]}"
                for name in missing
            )
            raise FileNotFoundError(
                "Cannot run inference — required artifacts are missing. "
                "Run training first to build caches and checkpoints.\n"
                f"Missing:\n{details}"
            )

        # Loads feature_cache.pkl + descriptor_scaler.pkl from disk.
        self.feature_gen = FeatureGenerator()

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = load_pickle(Config.VOCAB_CACHE_PATH)
            logger.info(f"Loaded cached vocab with {len(self.vocab)} drugs.")

        if graph is not None:
            self.base_graph = graph.to(device)
        else:
            logger.info(f"Loading cached graph from {Config.GRAPH_CACHE_PATH}")
            graph = torch.load(Config.GRAPH_CACHE_PATH, weights_only=False)
            self.base_graph = graph.to(device)

        ckpt = load_checkpoint(checkpoint_path, map_location=device)
        model_cfg = ckpt.get("config", {})
        self.model = DDIModel(
            in_channels=model_cfg.get("in_channels", Config.NODE_FEATURE_DIM),
            hidden_channels=model_cfg.get("hidden_channels", Config.SAGE_HIDDEN_DIM),
            num_layers=model_cfg.get("num_layers", Config.SAGE_NUM_LAYERS),
            sage_dropout=model_cfg.get("sage_dropout", Config.SAGE_DROPOUT),
            pair_hidden_dim=model_cfg.get("pair_hidden_dim", Config.PAIR_HIDDEN_DIM),
            pair_dropout=model_cfg.get("pair_dropout", Config.PAIR_DROPOUT),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info(
            f"Loaded model checkpoint from {checkpoint_path} "
            f"(epoch {ckpt.get('epoch', '?')}, val_metrics={ckpt.get('val_metrics', {})})"
        )

    # ------------------------------------------------------------------
    def _resolve_indices(self, smiles_pairs: List[Tuple[str, str]]):
        """For every (smiles_a, smiles_b) pair, resolve node indices in
        the (possibly extended) graph, computing features for any new
        (cold-start) drug not already in self.vocab."""
        working_vocab = dict(self.vocab)
        new_smiles: List[str] = []

        canon_pairs = []
        for smi_a, smi_b in smiles_pairs:
            try:
                ca = canonicalize_smiles(smi_a)
                cb = canonicalize_smiles(smi_b)
            except InvalidSMILESError as e:
                raise InvalidSMILESError(f"Could not parse one of the pair ({smi_a!r}, {smi_b!r}): {e}")
            canon_pairs.append((ca, cb))
            for c in (ca, cb):
                if c not in working_vocab and c not in new_smiles:
                    new_smiles.append(c)

        graph = self.base_graph
        if new_smiles:
            logger.info(f"Found {len(new_smiles)} new (cold-start) drug(s) not in training vocab.")
            new_features = np.stack([self.feature_gen.get_features(s) for s in new_smiles])
            graph = extend_graph_with_new_nodes(self.base_graph.cpu(), new_features).to(self.device)
            start_idx = len(working_vocab)
            for offset, smi in enumerate(new_smiles):
                working_vocab[smi] = start_idx + offset

        idx_a = torch.tensor([working_vocab[a] for a, _ in canon_pairs], dtype=torch.long, device=self.device)
        idx_b = torch.tensor([working_vocab[b] for _, b in canon_pairs], dtype=torch.long, device=self.device)
        return graph, idx_a, idx_b

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_batch(
        self, smiles_pairs: List[Tuple[str, str]], threshold: float = Config.DECISION_THRESHOLD
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Score a list of (smiles_a, smiles_b) pairs.

        Returns a list of dicts: {smile1, smile2, probability, prediction}
        """
        graph, idx_a, idx_b = self._resolve_indices(smiles_pairs)
        node_embeddings = self.model.encode(graph.x, graph.edge_index)
        logits = self.model(graph.x, graph.edge_index, idx_a, idx_b, node_embeddings=node_embeddings)
        probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for (smi_a, smi_b), p in zip(smiles_pairs, probs):
            results.append(
                {
                    "smile1": smi_a,
                    "smile2": smi_b,
                    "probability": float(p),
                    "prediction": int(p >= threshold),
                }
            )
        return results

    def predict_pair(self, smiles_a: str, smiles_b: str, threshold: float = Config.DECISION_THRESHOLD) -> Dict:
        return self.predict_batch([(smiles_a, smiles_b)], threshold=threshold)[0]