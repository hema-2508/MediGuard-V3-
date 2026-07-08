"""
evaluate.py
-----------
Entity-level evaluation for prescription NER using seqeval metrics.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from config import Config
from dataset import NERSplit, PrescriptionNERDataset
from utils import align_predictions_to_words, get_logger, log_device_info

logger = get_logger("evaluate")


def _json_safe(value: Any) -> Any:
    """Convert numpy/scalar types into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@torch.no_grad()
def evaluate_model(
    model,
    split: NERSplit,
    tokenizer,
    device: torch.device = Config.DEVICE,
    batch_size: int = Config.EVAL_BATCH_SIZE,
) -> Dict[str, float]:
    log_device_info(logger, device)
    model.eval()

    dataset = PrescriptionNERDataset(
        examples=split.examples,
        tokenizer=tokenizer,
        label2id=split.label2id,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []

    for batch in loader:
        batch = {key: value.to(device, non_blocking=Config.NON_BLOCKING) for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs["logits"]

        for idx in range(batch["input_ids"].size(0)):
            example_index = int(batch["example_index"][idx].item())
            example = split.examples[example_index]
            gold_labels = example["labels"]

            encoding = tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=Config.MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            pred_ids = logits[idx].argmax(dim=-1).cpu().numpy()
            probs = torch.softmax(logits[idx], dim=-1)
            pred_conf = probs.max(dim=-1).values.cpu().numpy()

            aligned_ids, _, _ = align_predictions_to_words(
                word_ids=word_ids,
                pred_ids=pred_ids.tolist(),
                confidences=pred_conf.tolist(),
            )
            pred_labels = [split.id2label[label_id] for label_id in aligned_ids]
            min_len = min(len(gold_labels), len(pred_labels))
            y_true.append(gold_labels[:min_len])
            y_pred.append(pred_labels[:min_len])

    metrics = {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = report
    return metrics


def save_evaluation_report(
    metrics: Dict,
    split_name: str,
    save_path: Optional[str] = None,
) -> str:
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, f"{split_name.lower().replace(' ', '_')}_report.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2)

    logger.info(
        f"{split_name}: precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f} "
        f"(saved to {save_path})"
    )
    return save_path
