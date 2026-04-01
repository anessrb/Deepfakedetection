#!/usr/bin/env python3
"""
Main training script for the DeepFake Detection model.

Loads configuration from YAML, builds datasets, constructs the model,
runs training with the Trainer, and saves final checkpoint with
temperature calibration.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --data_root data/ --epochs 30
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from src.datasets import (
    FaceForensicsPlusPlus,
    CelebDFv2,
    DF40Dataset,
    get_train_transforms,
    get_val_transforms,
)
from src.models import DeepfakeDetector, TemperatureScaling
from src.training import Trainer
from src.evaluation.metrics import compute_metrics, print_metrics_table
from src.visualization.plots import plot_training_history


def setup_logging(log_dir: str = "logs/") -> logging.Logger:
    """Set up file + console logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_dir / "training.log")),
        ],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def auto_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.getLogger(__name__).info(
            f"Using CUDA: {torch.cuda.get_device_name(0)}"
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.getLogger(__name__).info("Using Apple Silicon MPS.")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).warning("No GPU found. Training on CPU (slow).")
    return device


def build_datasets(config: Dict[str, Any]):
    """
    Build training and validation datasets from configuration.

    Returns train and val datasets (ConcatDataset if multiple datasets).
    """
    logger = logging.getLogger(__name__)

    aug_cfg = config.get("augmentation", {})
    img_size = aug_cfg.get("img_size", 224)
    ds_cfg = config.get("datasets", {})
    data_root = ds_cfg.get("data_root", "data/")
    max_samples = ds_cfg.get("max_samples_per_dataset", None)

    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)

    train_datasets = []
    val_datasets = []

    # FaceForensics++
    ff_cfg = ds_cfg.get("ff_plus_plus", {})
    ff_root = ff_cfg.get("root", str(Path(data_root) / "ff++"))
    ff_manips = ff_cfg.get("manipulations", ["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"])
    ff_compression = ff_cfg.get("compression", "c23")

    try:
        ff_train = FaceForensicsPlusPlus(
            root=ff_root, split="train", transform=train_transform,
            manipulations=ff_manips, compression=ff_compression, max_samples=max_samples,
        )
        ff_val = FaceForensicsPlusPlus(
            root=ff_root, split="val", transform=val_transform,
            manipulations=ff_manips, compression=ff_compression, max_samples=max_samples,
        )
        if len(ff_train) > 0:
            train_datasets.append(ff_train)
        if len(ff_val) > 0:
            val_datasets.append(ff_val)
        logger.info(f"FF++: {len(ff_train)} train, {len(ff_val)} val samples")
    except Exception as e:
        logger.warning(f"Could not load FF++ dataset: {e}")

    # DF40
    df40_cfg = ds_cfg.get("df40", {})
    df40_root = df40_cfg.get("root", str(Path(data_root) / "df40"))
    df40_csv = df40_cfg.get("csv_manifest", None)

    try:
        df40_train = DF40Dataset(
            root=df40_root, split="train", transform=train_transform,
            csv_manifest=df40_csv, max_samples=max_samples,
        )
        df40_val = DF40Dataset(
            root=df40_root, split="val", transform=val_transform,
            csv_manifest=df40_csv, max_samples=max_samples,
        )
        if len(df40_train) > 0:
            train_datasets.append(df40_train)
        if len(df40_val) > 0:
            val_datasets.append(df40_val)
        logger.info(f"DF40: {len(df40_train)} train, {len(df40_val)} val samples")
    except Exception as e:
        logger.warning(f"Could not load DF40 dataset: {e}")

    if not train_datasets:
        logger.error(
            "No training data found! Please check dataset paths in config."
        )
        raise RuntimeError("No training datasets loaded.")

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    return train_dataset, val_dataset


def build_loaders(
    train_dataset,
    val_dataset,
    config: Dict[str, Any],
    balance_classes: bool = True,
):
    """Build DataLoader objects with optional weighted sampling for balance."""
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True)

    # Weighted sampler for class balance
    if balance_classes and hasattr(train_dataset, "get_sample_weights"):
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_shuffle = False
    elif balance_classes and hasattr(train_dataset, "datasets"):
        # ConcatDataset: merge weights from sub-datasets
        import numpy as np
        all_weights = []
        for ds in train_dataset.datasets:
            if hasattr(ds, "get_sample_weights"):
                all_weights.append(ds.get_sample_weights())
        if all_weights:
            weights = torch.cat(all_weights)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_shuffle = False
        else:
            sampler = None
            train_shuffle = True
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train deepfake detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override dataset data root directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Override checkpoint save directory.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="Skip post-training temperature calibration.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/",
        help="Directory for saving plots and final model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (cpu, cuda, mps). If None, auto-selects.",
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    logger = setup_logging()

    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script location
        config_path = Path(__file__).parent.parent / args.config
    config = load_config(str(config_path))

    # CLI overrides
    if args.data_root:
        config.setdefault("datasets", {})["data_root"] = args.data_root
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.checkpoint_dir:
        config.setdefault("training", {})["checkpoint_dir"] = args.checkpoint_dir

    device = torch.device(args.device) if args.device else auto_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build Datasets ────────────────────────────────────────────────────
    logger.info("Building datasets...")
    train_dataset, val_dataset = build_datasets(config)
    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    train_loader, val_loader = build_loaders(train_dataset, val_dataset, config)
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    # ── Build Model ───────────────────────────────────────────────────────
    logger.info("Building model...")
    model_cfg = config.get("model", {})
    model = DeepfakeDetector(
        spatial_model_name=model_cfg.get("backbone", "vit_base_patch14_dinov2.lvd142m"),
        spatial_pretrained=True,
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 4),
        spatial_embed_dim=model_cfg.get("spatial_embed_dim", 768),
        freq_embed_dim=model_cfg.get("freq_embed_dim", 256),
        freq_conv_channels=model_cfg.get("freq_conv_channels", None),
        fusion_hidden_dims=model_cfg.get("fusion_hidden_dims", [512, 128]),
        fusion_dropout_rates=model_cfg.get("fusion_dropout", [0.3, 0.2]),
    )

    # ── Training ──────────────────────────────────────────────────────────
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training...")
    history = trainer.train()

    # ── Save Training History Plot ────────────────────────────────────────
    history_plot_path = str(output_dir / "training_history.png")
    plot_training_history(history, save_path=history_plot_path)
    logger.info(f"Training history plot saved: {history_plot_path}")

    # ── Load Best Checkpoint for Evaluation ──────────────────────────────
    best_ckpt = Path(config.get("training", {}).get("checkpoint_dir", "checkpoints/")) / "best.pth"
    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt), load_optimizer=False)
        logger.info(f"Loaded best checkpoint from {best_ckpt}")

    # ── Validation Metrics ────────────────────────────────────────────────
    logger.info("Computing final validation metrics...")
    import numpy as np

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].numpy()
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    final_metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    print_metrics_table({"validation": final_metrics}, title="Final Validation Metrics")

    # ── Temperature Calibration ───────────────────────────────────────────
    if not args.no_calibrate:
        logger.info("Running temperature scaling calibration...")
        calibrator = TemperatureScaling(
            initial_temperature=model_cfg.get("initial_temperature", 1.0)
        )
        final_temp = calibrator.calibrate(model, val_loader, device, verbose=True)
        logger.info(f"Optimal temperature: {final_temp:.4f}")

        # Save calibrated model
        calibrated_path = output_dir / "detector_calibrated.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "temperature": final_temp,
                "val_metrics": final_metrics,
                "config": config,
                "history": history,
            },
            str(calibrated_path),
        )
        logger.info(f"Calibrated model saved: {calibrated_path}")

        # Reliability diagram
        cal_plot_path = str(output_dir / "calibration_curve.png")
        calibrator.plot_reliability_diagram(
            np.array(all_probs), np.array(all_labels),
            title="Validation Calibration (Before Temperature Scaling)",
            save_path=cal_plot_path,
        )
    else:
        # Save uncalibrated final model
        final_path = output_dir / "detector_final.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_metrics": final_metrics,
                "config": config,
            },
            str(final_path),
        )
        logger.info(f"Final model saved: {final_path}")

    logger.info("Training pipeline complete!")
    logger.info(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
