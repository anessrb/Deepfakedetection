#!/usr/bin/env python3
"""
Post-hoc temperature scaling calibration script.

Loads a trained model checkpoint, fits optimal temperature on the
validation set, saves a calibrated checkpoint, and produces
reliability diagram comparisons (before/after calibration).

Usage:
    python scripts/calibrate.py \\
        --checkpoint checkpoints/best.pth \\
        --config configs/default.yaml \\
        --output_dir outputs/calibration/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.datasets import (
    FaceForensicsPlusPlus,
    DF40Dataset,
    get_val_transforms,
)
from src.models import DeepfakeDetector, TemperatureScaling
from src.evaluation.metrics import compute_metrics, print_metrics_table
from src.visualization.plots import plot_calibration_curve


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Collect all logits, probabilities, and labels from a DataLoader."""
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].numpy()

            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            logits_np = logits.squeeze(1).cpu().numpy()

            all_logits.extend(logits_np.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    return (
        np.array(all_logits),
        np.array(all_probs),
        np.array(all_labels),
    )


def build_val_loader(config: Dict[str, Any], batch_size: int = 64) -> DataLoader:
    """Build validation loader from config."""
    logger = logging.getLogger(__name__)
    aug_cfg = config.get("augmentation", {})
    img_size = aug_cfg.get("img_size", 224)
    ds_cfg = config.get("datasets", {})
    data_root = ds_cfg.get("data_root", "data/")

    transform = get_val_transforms(img_size=img_size)

    # Try FF++ first, then DF40
    ff_cfg = ds_cfg.get("ff_plus_plus", {})
    ff_root = ff_cfg.get("root", str(Path(data_root) / "ff++"))

    try:
        ds = FaceForensicsPlusPlus(root=ff_root, split="val", transform=transform)
        if len(ds) > 0:
            logger.info(f"Using FF++ val set: {len(ds)} samples")
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        logger.debug(f"FF++ not available: {e}")

    df40_cfg = ds_cfg.get("df40", {})
    df40_root = df40_cfg.get("root", str(Path(data_root) / "df40"))
    df40_csv = df40_cfg.get("csv_manifest", None)

    try:
        ds = DF40Dataset(root=df40_root, split="val", transform=transform,
                         csv_manifest=df40_csv)
        if len(ds) > 0:
            logger.info(f"Using DF40 val set: {len(ds)} samples")
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        logger.debug(f"DF40 not available: {e}")

    raise RuntimeError("No validation dataset found. Check data paths in config.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc temperature scaling calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/calibration/",
        help="Directory to save calibrated model and plots.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for calibration.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="LBFGS learning rate for temperature optimization.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="Maximum LBFGS iterations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Config ───────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    config = load_config(str(config_path))
    device = auto_device()

    # ── Load Model ────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_cfg = config.get("model", {})
    model = DeepfakeDetector(
        spatial_model_name=model_cfg.get("backbone", "vit_base_patch14_dinov2.lvd142m"),
        spatial_pretrained=False,
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 4),
        spatial_embed_dim=model_cfg.get("spatial_embed_dim", 768),
        freq_embed_dim=model_cfg.get("freq_embed_dim", 256),
        fusion_hidden_dims=model_cfg.get("fusion_hidden_dims", [512, 128]),
        fusion_dropout_rates=model_cfg.get("fusion_dropout", [0.3, 0.2]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded.")

    # ── Build Val Loader ──────────────────────────────────────────────────
    logger.info("Building validation loader...")
    val_loader = build_val_loader(config, batch_size=args.batch_size)

    # ── Pre-calibration predictions ───────────────────────────────────────
    logger.info("Collecting pre-calibration predictions...")
    _, probs_before, labels = collect_predictions(model, val_loader, device)

    metrics_before = compute_metrics(labels, probs_before)
    logger.info(
        f"Before calibration: AUC={metrics_before['auc']:.4f} | "
        f"ECE={metrics_before['ece']:.4f} | ACC={metrics_before['accuracy']:.4f}"
    )

    # Reliability diagram before
    cal_before_path = str(output_dir / "reliability_before.png")
    plot_calibration_curve(
        probs_before, labels,
        title="Reliability Diagram — Before Temperature Scaling",
        save_path=cal_before_path,
    )
    logger.info(f"Pre-calibration diagram saved: {cal_before_path}")

    # ── Temperature Scaling ───────────────────────────────────────────────
    logger.info("Fitting temperature scaling...")
    calibrator = TemperatureScaling(
        initial_temperature=model_cfg.get("initial_temperature", 1.0)
    )
    optimal_temp = calibrator.calibrate(
        model=model,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        max_iter=args.max_iter,
        verbose=True,
    )
    logger.info(f"Optimal temperature: {optimal_temp:.4f}")

    # ── Post-calibration predictions ──────────────────────────────────────
    logger.info("Collecting post-calibration predictions...")
    _, logits_raw, labels = collect_predictions(model, val_loader, device)

    # Apply temperature
    probs_after = 1.0 / (1.0 + np.exp(-logits_raw / optimal_temp))

    metrics_after = compute_metrics(labels, probs_after)
    logger.info(
        f"After calibration: AUC={metrics_after['auc']:.4f} | "
        f"ECE={metrics_after['ece']:.4f} | ACC={metrics_after['accuracy']:.4f}"
    )

    # Comparison table
    print_metrics_table(
        {"Before calibration": metrics_before, "After calibration": metrics_after},
        title="Calibration Impact",
    )

    # Reliability diagram after
    cal_after_path = str(output_dir / "reliability_after.png")
    plot_calibration_curve(
        probs_after, labels,
        title="Reliability Diagram — After Temperature Scaling",
        save_path=cal_after_path,
    )
    logger.info(f"Post-calibration diagram saved: {cal_after_path}")

    # ── Save Calibrated Checkpoint ────────────────────────────────────────
    calibrated_path = output_dir / "detector_calibrated.pth"
    calibrated_checkpoint = {
        "model_state_dict": model.state_dict(),
        "temperature": optimal_temp,
        "calibrator_state_dict": calibrator.state_dict(),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "config": config,
        "source_checkpoint": str(args.checkpoint),
    }
    torch.save(calibrated_checkpoint, str(calibrated_path))
    logger.info(f"Calibrated checkpoint saved: {calibrated_path}")

    logger.info("Calibration complete!")
    print(f"\nSummary:")
    print(f"  ECE before: {metrics_before['ece']:.4f}")
    print(f"  ECE after:  {metrics_after['ece']:.4f}")
    print(f"  Temperature: {optimal_temp:.4f}")
    print(f"  Saved to: {calibrated_path}")


if __name__ == "__main__":
    main()
