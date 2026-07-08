"""
trainer.py
----------
Training loop for the BioBERT prescription NER model: AdamW optimizer,
linear warmup, gradient clipping, early stopping on validation F1, and
checkpoint saving.
"""

import math
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import Config
from dataset import NERSplit, build_dataloader, get_tokenizer, save_label_maps
from evaluate import evaluate_model, save_evaluation_report
from model import MedicineNERModel
from utils import AverageMeter, EarlyStopping, get_logger, log_device_info, save_json

logger = get_logger("trainer")


class Trainer:
    def __init__(
        self,
        model: MedicineNERModel,
        train_split: NERSplit,
        val_split: NERSplit,
        tokenizer: AutoTokenizer,
        device: torch.device = Config.DEVICE,
    ):
        log_device_info(logger, device)
        self.model = model.to(device)
        self.train_split = train_split
        self.val_split = val_split
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.early_stopper = EarlyStopping(mode="max")
        self.history = []

        train_loader = build_dataloader(train_split, tokenizer, Config.BATCH_SIZE, shuffle=True)
        total_steps = len(train_loader) * Config.EPOCHS
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.train_loader = train_loader

    def _train_one_epoch(self) -> float:
        self.model.train()
        loss_meter = AverageMeter()

        for batch in self.train_loader:
            batch = {key: value.to(self.device, non_blocking=Config.NON_BLOCKING) for key, value in batch.items()}
            labels = batch.pop("labels")
            batch.pop("example_index", None)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_NORM)
            self.optimizer.step()
            self.scheduler.step()

            loss_meter.update(loss.item(), n=batch["input_ids"].size(0))

        return loss_meter.avg

    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        save_label_maps(self.train_split.label_list, os.path.join(path, "label_map.json"))
        metadata = {
            "epoch": epoch,
            "val_metrics": {key: val_metrics[key] for key in ("precision", "recall", "f1")},
            "is_best": is_best,
            "pretrained_model_name": Config.PRETRAINED_MODEL_NAME,
            "label_list": self.train_split.label_list,
        }
        save_json(metadata, os.path.join(path, "training_metadata.json"))

    def fit(
        self,
        epochs: int = Config.EPOCHS,
        best_checkpoint_path: str = Config.BEST_MODEL_PATH,
        last_checkpoint_path: Optional[str] = Config.LAST_MODEL_PATH,
    ) -> Dict:
        logger.info(
            f"Starting training: {epochs} max epochs, "
            f"{len(self.train_split.examples)} train examples, "
            f"{len(self.val_split.examples)} val examples, "
            f"monitoring '{Config.EARLY_STOPPING_METRIC}' for early stopping."
        )

        best_f1 = -math.inf

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_metrics = evaluate_model(
                model=self.model,
                split=self.val_split,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            elapsed = time.time() - t0
            monitored = val_metrics[Config.EARLY_STOPPING_METRIC]
            is_best = self.early_stopper.step(monitored)

            logger.info(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_precision={val_metrics['precision']:.4f} | "
                f"val_recall={val_metrics['recall']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"time={elapsed:.1f}s" + (" | *new best*" if is_best else "")
            )

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                }
            )

            if last_checkpoint_path:
                self._save_checkpoint(last_checkpoint_path, epoch, val_metrics, is_best=False)

            if is_best:
                best_f1 = monitored
                self._save_checkpoint(best_checkpoint_path, epoch, val_metrics, is_best=True)

            if self.early_stopper.should_stop:
                logger.info(
                    f"Early stopping triggered after epoch {epoch} "
                    f"(best val_f1={self.early_stopper.best_score:.4f})."
                )
                break

        return {
            "best_val_f1": best_f1 if best_f1 > -math.inf else self.early_stopper.best_score,
            "history": self.history,
        }


def build_trainer(
    train_split: NERSplit,
    val_split: NERSplit,
    device: torch.device = Config.DEVICE,
) -> Trainer:
    tokenizer = get_tokenizer()
    save_label_maps(train_split.label_list)

    model = MedicineNERModel(
        num_labels=len(train_split.label_list),
        id2label=train_split.id2label,
        label2id=train_split.label2id,
    )
    return Trainer(
        model=model,
        train_split=train_split,
        val_split=val_split,
        tokenizer=tokenizer,
        device=device,
    )
