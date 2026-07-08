"""
model.py
--------
BioBERT-based token classification model for prescription entity
extraction (medicine, strength, dosage, frequency, duration, route).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForTokenClassification

from config import Config


class MedicineNERModel(nn.Module):
    """Thin wrapper around a biomedical pretrained token classifier."""

    def __init__(
        self,
        num_labels: int,
        pretrained_model_name: str = Config.PRETRAINED_MODEL_NAME,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        if dropout is not None:
            model_config.hidden_dropout_prob = dropout
            model_config.attention_probs_dropout_prob = dropout

        self.backbone = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name,
            config=model_config,
        )
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def save_pretrained(self, directory: str) -> None:
        self.backbone.save_pretrained(directory)
