"""
Training loop for deepfake detection model.

Implements a production-quality training pipeline with:
- Mixed-precision training (AMP)
- CosineAnnealingLR scheduler with optional warmup
- Early stopping based on validation AUC
- Checkpoint saving (best + periodic)
- Comprehensive logging with tqdm
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CombinedLoss
from ..evaluation.metrics import compute_auc, compute_ece

logger = logging.getLogger(__name__)


class Trainer:
    """
    Full training manager for the DeepfakeDetector model.

    Handles training and validation loops, scheduling, checkpointing,
    and early stopping.

    Args:
        model: The deepfake detection model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration dict with training hyperparameters.
        device: Torch device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or self._auto_device()

        # Training config
        train_cfg = config.get("training", {})
        self.lr_backbone = train_cfg.get("lr_backbone", 1e-5)
        self.lr_head = train_cfg.get("lr_head", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.n_epochs = train_cfg.get("epochs", 30)
        self.use_amp = train_cfg.get("use_amp", True) and self.device.type == "cuda"
        self.patience = train_cfg.get("early_stopping_patience", 5)
        self.es_metric = train_cfg.get("early_stopping_metric", "auc")
        self.warmup_epochs = train_cfg.get("warmup_epochs", 2)
        self.lr_min = train_cfg.get("lr_min", 1e-6)
        self.save_every = train_cfg.get("save_every_n_epochs", 5)

        # Checkpoint dir
        ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints/")
        self.checkpoint_dir = Path(ckpt_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        loss_cfg = {
            "use_focal": train_cfg.get("loss", "combined") in ("focal", "combined"),
            "focal_gamma": train_cfg.get("focal_gamma", 2.0),
            "focal_alpha": train_cfg.get("focal_alpha", 0.25),
            "ece_lambda": train_cfg.get("ece_lambda", 0.1),
        }
        self.criterion = CombinedLoss(**loss_cfg)

        # Optimizer
        if hasattr(model, "get_optimizer_param_groups"):
            param_groups = model.get_optimizer_param_groups(
                lr_backbone=self.lr_backbone,
                lr_head=self.lr_head,
                weight_decay=self.weight_decay,
            )
        else:
            param_groups = [{"params": model.parameters(), "lr": self.lr_head}]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # Scheduler with optional linear warmup
        total_steps = self.n_epochs
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - self.warmup_epochs),
            eta_min=self.lr_min,
        )

        if self.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            self.scheduler = cosine_scheduler

        # AMP scaler (CUDA only)
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # State
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_auc": [],
            "val_loss": [],
            "val_auc": [],
            "val_ece": [],
            "lr": [],
        }

        self.model.to(self.device)
        logger.info(
            f"Trainer initialized | device={self.device} | "
            f"epochs={self.n_epochs} | amp={self.use_amp}"
        )

    def train_epoch(self) -> Tuple[float, float]:
        """
        Run one epoch of training.

        Returns:
            Tuple of (average_loss, average_auc).
        """
        self.model.train()
        total_loss = 0.0
        all_probs: List[float] = []
        all_labels: List[int] = []
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.n_epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast("cuda"):
                    logits = self.model(images)
                    loss, loss_dict = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss, loss_dict = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss_dict["total"]
            n_batches += 1

            # Collect predictions for AUC
            with torch.no_grad():
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                cls=f"{loss_dict['cls']:.4f}",
                ece=f"{loss_dict['ece']:.4f}",
            )

        avg_loss = total_loss / max(n_batches, 1)
        avg_auc = compute_auc(
            labels=np.array(all_labels),
            probs=np.array(all_probs),
        )
        return avg_loss, avg_auc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """
        Run validation loop.

        Returns:
            Tuple of (avg_loss, auc, ece).
        """
        self.model.eval()
        total_loss = 0.0
        all_probs: List[float] = []
        all_labels: List[int] = []
        n_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.n_epochs} [Val]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).float()

            if self.use_amp:
                with autocast("cuda"):
                    logits = self.model(images)
                    loss, loss_dict = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss, loss_dict = self.criterion(logits, labels)

            total_loss += loss_dict["total"]
            n_batches += 1

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / max(n_batches, 1)
        labels_arr = np.array(all_labels)
        probs_arr = np.array(all_probs)
        auc = compute_auc(labels_arr, probs_arr)
        ece = compute_ece(labels_arr, probs_arr)

        return avg_loss, auc, ece

    def train(self, n_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            n_epochs: Override for number of epochs from config.

        Returns:
            Training history dictionary.
        """
        if n_epochs is not None:
            self.n_epochs = n_epochs

        logger.info(f"Starting training for {self.n_epochs} epochs.")
        start_time = time.time()

        for epoch in range(self.n_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training
            train_loss, train_auc = self.train_epoch()

            # Validation
            val_loss, val_auc, val_ece = self.validate()

            # Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["val_ece"].append(val_ece)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch + 1:3d}/{self.n_epochs}] "
                f"| train_loss: {train_loss:.4f} | train_auc: {train_auc:.4f} "
                f"| val_loss: {val_loss:.4f} | val_auc: {val_auc:.4f} "
                f"| val_ece: {val_ece:.4f} | lr: {current_lr:.2e} "
                f"| time: {epoch_time:.1f}s"
            )

            # Check improvement
            improved = self._check_improvement(val_auc, val_loss)

            if improved:
                self.epochs_without_improvement = 0
                self._save_checkpoint("best.pth", extra={
                    "epoch": epoch + 1,
                    "val_auc": val_auc,
                    "val_loss": val_loss,
                    "val_ece": val_ece,
                })
                logger.info(f"  → New best model saved (val_auc={val_auc:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1:03d}.pth", extra={
                    "epoch": epoch + 1,
                    "val_auc": val_auc,
                })

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"({self.patience} epochs without improvement)."
                )
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time / 60:.1f} minutes. "
            f"Best val_auc: {self.best_val_auc:.4f}"
        )
        return self.history

    def _check_improvement(self, val_auc: float, val_loss: float) -> bool:
        """Check if validation metric improved."""
        if self.es_metric == "auc":
            if val_auc > self.best_val_auc + 1e-5:
                self.best_val_auc = val_auc
                return True
        else:  # loss
            if val_loss < self.best_val_loss - 1e-5:
                self.best_val_loss = val_loss
                return True
        return False

    def _save_checkpoint(self, filename: str, extra: Optional[Dict] = None) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_auc": self.best_val_auc,
            "history": self.history,
            "config": self.config,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict:
        """
        Load a training checkpoint.

        Args:
            path: Path to checkpoint file.
            load_optimizer: Whether to restore optimizer/scheduler state.

        Returns:
            Full checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_auc = checkpoint.get("best_auc", 0.0)
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        logger.info(
            f"Checkpoint loaded from {path} "
            f"(epoch={self.current_epoch}, best_auc={self.best_val_auc:.4f})"
        )
        return checkpoint

    @staticmethod
    def _auto_device() -> torch.device:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
