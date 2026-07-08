"""
trainer.py
----------
Training loop for the GraphSAGE DDI model: AdamW optimizer,
BCEWithLogitsLoss, gradient clipping, early stopping on a validation
metric, and checkpointing of the best model.
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from dataset import PairSplit
from evaluate import compute_metrics
from model import DDIModel
from utils import EarlyStopping, get_logger, save_checkpoint, AverageMeter, log_device_info

logger = get_logger("trainer")


def _make_loader(pairs: PairSplit, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(pairs.idx_a, dtype=torch.long),
        torch.tensor(pairs.idx_b, dtype=torch.long),
        torch.tensor(pairs.labels, dtype=torch.float32),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )


class Trainer:
    def __init__(
        self,
        model: DDIModel,
        graph,
        train_pairs: PairSplit,
        val_pairs: PairSplit,
        device: torch.device = Config.DEVICE,
    ):
        log_device_info(logger, device)
        self.model = model.to(device)
        self.graph = graph.to(device)
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.early_stopper = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            mode="max",
        )
        self.history = []

    # ------------------------------------------------------------------
    def _train_one_epoch(self) -> float:
        self.model.train()
        loader = _make_loader(self.train_pairs, Config.BATCH_SIZE, shuffle=True)
        loss_meter = AverageMeter()

        for idx_a, idx_b, labels in loader:
            idx_a = idx_a.to(self.device, non_blocking=Config.NON_BLOCKING)
            idx_b = idx_b.to(self.device, non_blocking=Config.NON_BLOCKING)
            labels = labels.to(self.device, non_blocking=Config.NON_BLOCKING)

            self.optimizer.zero_grad()
            logits = self.model(self.graph.x, self.graph.edge_index, idx_a, idx_b)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_NORM)
            self.optimizer.step()

            loss_meter.update(loss.item(), n=labels.size(0))

        return loss_meter.avg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, pairs: PairSplit) -> Dict[str, float]:
        self.model.eval()
        loader = _make_loader(pairs, batch_size=2048, shuffle=False)

        # Encode the graph once per evaluation pass (no dropout in eval mode,
        # so it's deterministic and safe to reuse across all pair batches).
        node_embeddings = self.model.encode(self.graph.x, self.graph.edge_index)

        all_logits = []
        all_labels = []
        for idx_a, idx_b, labels in loader:
            idx_a = idx_a.to(self.device, non_blocking=Config.NON_BLOCKING)
            idx_b = idx_b.to(self.device, non_blocking=Config.NON_BLOCKING)
            logits = self.model(
                self.graph.x, self.graph.edge_index, idx_a, idx_b,
                node_embeddings=node_embeddings,
            )
            all_logits.append(logits.cpu())
            all_labels.append(labels)

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        metrics = compute_metrics(labels, probs)
        return metrics

    # ------------------------------------------------------------------
    def fit(
        self,
        epochs: int = Config.EPOCHS,
        checkpoint_path: str = Config.BEST_MODEL_PATH,
        last_checkpoint_path: Optional[str] = Config.LAST_MODEL_PATH,
    ) -> Dict:
        logger.info(
            f"Starting training: {epochs} max epochs, "
            f"{len(self.train_pairs)} train pairs, {len(self.val_pairs)} val pairs, "
            f"monitoring '{Config.EARLY_STOPPING_METRIC}' for early stopping."
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_metrics = self._evaluate(self.val_pairs)
            elapsed = time.time() - t0

            monitored = val_metrics[Config.EARLY_STOPPING_METRIC]
            is_best = self.early_stopper.step(monitored)

            logger.info(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f} | "
                f"val_roc_auc={val_metrics['roc_auc']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f} | "
                f"time={elapsed:.1f}s" + (" | *new best*" if is_best else "")
            )

            self.history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": {
                            "in_channels": Config.NODE_FEATURE_DIM,
                            "hidden_channels": Config.SAGE_HIDDEN_DIM,
                            "num_layers": Config.SAGE_NUM_LAYERS,
                            "sage_dropout": Config.SAGE_DROPOUT,
                            "pair_hidden_dim": Config.PAIR_HIDDEN_DIM,
                            "pair_dropout": Config.PAIR_DROPOUT,
                        },
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved new best model checkpoint to {checkpoint_path}")

            if last_checkpoint_path:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    last_checkpoint_path,
                )

            if self.early_stopper.should_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement in {Config.EARLY_STOPPING_METRIC} for "
                    f"{Config.EARLY_STOPPING_PATIENCE} epochs)."
                )
                break

        logger.info(f"Training complete. Best {Config.EARLY_STOPPING_METRIC}: {self.early_stopper.best_score:.4f}")
        return {"history": self.history, "best_score": self.early_stopper.best_score}