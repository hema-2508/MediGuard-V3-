"""
evaluate.py
-----------
Computes the full evaluation suite required for the DDI task:
Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC and the
Confusion Matrix. Also provides helpers to run a trained model over
an arbitrary PairSplit (e.g. DrugBank test set, or the BioSNAP
external validation set) and to persist a human-readable report.

`compute_metrics` is intentionally free of any dependency on
model/trainer internals so it can be reused for offline metric
recomputation on saved prediction files too.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from config import Config
from dataset import PairSplit
from utils import get_logger, log_device_info

logger = get_logger("evaluate")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = Config.DECISION_THRESHOLD,
) -> Dict[str, float]:
    """Compute the full metric suite from ground-truth labels and
    predicted probabilities (post-sigmoid, in [0, 1])."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
    }

    # ROC-AUC / PR-AUC require both classes present; guard against the
    # degenerate single-class case (e.g. a tiny debug subset).
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        logger.warning("Only one class present in y_true; ROC-AUC/PR-AUC set to NaN.")
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]

    return metrics


@torch.no_grad()
def run_inference(
    model,
    graph,
    pairs: PairSplit,
    device: torch.device = Config.DEVICE,
    batch_size: int = 2048,
) -> np.ndarray:
    """Run the model over an entire PairSplit and return predicted
    probabilities (sigmoid applied)."""
    model.eval()
    model.to(device)
    graph = graph.to(device)
    node_embeddings = model.encode(graph.x, graph.edge_index)

    idx_a = torch.tensor(pairs.idx_a, dtype=torch.long, pin_memory=Config.PIN_MEMORY)
    idx_b = torch.tensor(pairs.idx_b, dtype=torch.long, pin_memory=Config.PIN_MEMORY)

    probs = []
    for start in range(0, len(pairs), batch_size):
        end = start + batch_size
        a_batch = idx_a[start:end].to(device, non_blocking=Config.NON_BLOCKING)
        b_batch = idx_b[start:end].to(device, non_blocking=Config.NON_BLOCKING)
        logits = model(graph.x, graph.edge_index, a_batch, b_batch, node_embeddings=node_embeddings)
        probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs)


def evaluate_split(
    model,
    graph,
    pairs: PairSplit,
    split_name: str,
    device: torch.device = Config.DEVICE,
    save_path: Optional[str] = None,
) -> Dict:
    """Run inference + full metrics for a named split, log a summary,
    and optionally persist a JSON report."""
    log_device_info(logger, device)
    probs = run_inference(model, graph, pairs, device=device)
    metrics = compute_metrics(pairs.labels, probs)

    logger.info(f"===== Evaluation results: {split_name} =====")
    logger.info(f"  N pairs        : {len(pairs)}")
    logger.info(f"  Accuracy       : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision      : {metrics['precision']:.4f}")
    logger.info(f"  Recall         : {metrics['recall']:.4f}")
    logger.info(f"  F1             : {metrics['f1']:.4f}")
    logger.info(f"  MCC            : {metrics['mcc']:.4f}")
    logger.info(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC         : {metrics['pr_auc']:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    logger.info(f"  Confusion Matrix [ [TN {cm[0,0]}, FP {cm[0,1]}], [FN {cm[1,0]}, TP {cm[1,1]}] ]")

    report = {"split": split_name, "n_pairs": len(pairs), **metrics}

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved evaluation report to {save_path}")

    return report