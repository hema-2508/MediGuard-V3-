"""
inference.py
------------
Production inference for extracting structured medicine records from
prescription / OCR text. Supports single and batch prediction with
per-entity confidence scores.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import Config
from utils import (
    align_predictions_to_words,
    get_logger,
    get_spacy_nlp,
    labels_to_spans,
    load_json,
    log_device_info,
    preprocess_prescription_text,
    spans_to_medicines,
    tokenize_words,
)

logger = get_logger("inference")


class MedicineExtractionError(Exception):
    """Raised when inference cannot be completed."""


class MedicineExtractor:
    """Load a fine-tuned BioBERT NER checkpoint and extract medicines."""

    def __init__(
        self,
        checkpoint_path: str = Config.BEST_MODEL_PATH,
        device: torch.device = Config.DEVICE,
    ):
        self.device = device
        log_device_info(logger, device)

        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_path}. "
                "Train the model first with `python main.py train`."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint_path).to(device)
        self.model.eval()

        metadata_path = os.path.join(checkpoint_path, "training_metadata.json")
        label_map_path = os.path.join(checkpoint_path, "label_map.json")

        if os.path.exists(label_map_path):
            label_list = load_json(label_map_path)["label_list"]
        elif os.path.exists(metadata_path):
            label_list = load_json(metadata_path)["label_list"]
        else:
            label_list = Config.build_label_list()

        self.id2label = {idx: label for idx, label in enumerate(label_list)}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.nlp = get_spacy_nlp()

        logger.info(f"Loaded medicine extraction model from {checkpoint_path}")

    @torch.no_grad()
    def _predict_words(
        self,
        words: List[str],
    ) -> tuple[List[str], List[float]]:
        if not words:
            return [], []

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
            return_tensors="pt",
            return_offsets_mapping=False,
        )
        word_ids = encoding.word_ids(batch_index=0)
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        outputs = self.model(**encoding)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        pred_conf = probs.max(dim=-1).values.cpu().numpy()

        aligned_ids, aligned_confs, valid_word_indices = align_predictions_to_words(
            word_ids=word_ids,
            pred_ids=pred_ids.tolist(),
            confidences=pred_conf.tolist(),
        )
        pred_labels = [self.id2label[label_id] for label_id in aligned_ids]
        aligned_words = [words[idx] for idx in valid_word_indices if idx < len(words)]
        min_len = min(len(aligned_words), len(pred_labels))
        return pred_labels[:min_len], aligned_confs[:min_len]

    def _predict_long_text(self, text: str) -> tuple[List[str], List[float]]:
        """Sliding-window inference for prescriptions longer than MAX_SEQ_LENGTH."""
        words = tokenize_words(text, nlp=self.nlp)
        if not words:
            return [], []

        if len(words) <= Config.MAX_SEQ_LENGTH - 2:
            return self._predict_words(words)

        merged_labels = ["O"] * len(words)
        merged_conf = [0.0] * len(words)
        stride = Config.STRIDE

        for start in range(0, len(words), stride):
            chunk = words[start : start + Config.MAX_SEQ_LENGTH - 2]
            if not chunk:
                break

            chunk_labels, chunk_conf = self._predict_words(chunk)
            for offset, (label, conf) in enumerate(zip(chunk_labels, chunk_conf)):
                global_idx = start + offset
                if global_idx >= len(words):
                    break
                if label != "O" or merged_labels[global_idx] == "O":
                    merged_labels[global_idx] = label
                    merged_conf[global_idx] = conf

            if start + Config.MAX_SEQ_LENGTH - 2 >= len(words):
                break

        return merged_labels, merged_conf

    def extract(self, text: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Extract medicines from a single prescription text."""
        if not text or not str(text).strip():
            return {"medicines": []}

        try:
            cleaned = preprocess_prescription_text(str(text), nlp=self.nlp)
            labels, confidences = self._predict_long_text(cleaned)
            words = tokenize_words(cleaned, nlp=self.nlp)
            min_len = min(len(words), len(labels), len(confidences))
            spans = labels_to_spans(
                words=words[:min_len],
                labels=labels[:min_len],
                confidences=confidences[:min_len],
            )
            medicines = spans_to_medicines(spans)
            return {"medicines": medicines}
        except Exception as exc:
            raise MedicineExtractionError(f"Failed to extract medicines: {exc}") from exc

    def extract_batch(
        self,
        texts: List[str],
    ) -> List[Dict[str, List[Dict[str, Union[str, float]]]]]:
        """Extract medicines from multiple prescription texts."""
        if not texts:
            return []

        if len(texts) > Config.API_MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum allowed ({Config.API_MAX_BATCH_SIZE})."
            )

        return [self.extract(text) for text in texts]
