#!/usr/bin/env python3
"""
Cross-dataset evaluation script for the deepfake detection model.

Loads a trained checkpoint and evaluates on all configured datasets,
printing a comparison table and saving ROC curve plots.

Usage:
    python scripts/evaluate.py \\
        --checkpoint outputs/detector_calibrated.pth \\
        --config configs/default.yaml \\
        --output_dir outputs/evaluation/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.datasets import (
    FaceForensicsPlusPlus,
    CelebDFv2,
    DF40Dataset,
    WildDeepfake,
    get_val_transforms,
)
from src.models import DeepfakeDetector, TemperatureScaling
from src.evaluation import CrossDatasetEvaluator
from src.evaluation.metrics import print_metrics_table
from src.visualization.plots import plot_roc_curve


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


def build_eval_loaders(
    config: Dict[str, Any],
    datasets_to_eval: List[str],
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for all evaluation datasets."""
    logger = logging.getLogger(__name__)

    aug_cfg = config.get("augmentation", {})
    img_size = aug_cfg.get("img_size", 224)
    transform = get_val_transforms(img_size=img_size)

    ds_cfg = config.get("datasets", {})
    data_root = ds_cfg.get("data_root", "data/")

    loaders = {}

    if "ff_plus_plus" in datasets_to_eval:
        ff_cfg = ds_cfg.get("ff_plus_plus", {})
        ff_root = ff_cfg.get("root", str(Path(data_root) / "ff++"))
        try:
            ds = FaceForensicsPlusPlus(root=ff_root, split="test", transform=transform)
            if len(ds) > 0:
                loaders["FF++"] = DataLoader(
                    ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
                )
                logger.info(f"FF++ test: {len(ds)} samples")
        except Exception as e:
            logger.warning(f"Could not load FF++ for evaluation: {e}")

    if "celeb_df" in datasets_to_eval:
        celeb_cfg = ds_cfg.get("celeb_df", {})
        celeb_root = celeb_cfg.get("root", str(Path(data_root) / "celeb_df"))
        try:
            ds = CelebDFv2(root=celeb_root, split="test", transform=transform)
            if len(ds) > 0:
                loaders["Celeb-DF v2"] = DataLoader(
                    ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
                )
                logger.info(f"Celeb-DF v2 test: {len(ds)} samples")
        except Exception as e:
            logger.warning(f"Could not load Celeb-DF for evaluation: {e}")

    if "df40" in datasets_to_eval:
        df40_cfg = ds_cfg.get("df40", {})
        df40_root = df40_cfg.get("root", str(Path(data_root) / "df40"))
        df40_csv = df40_cfg.get("csv_manifest", None)
        try:
            ds = DF40Dataset(root=df40_root, split="test", transform=transform,
                             csv_manifest=df40_csv)
            if len(ds) > 0:
                loaders["DF40"] = DataLoader(
                    ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
                )
                logger.info(f"DF40 test: {len(ds)} samples")
        except Exception as e:
            logger.warning(f"Could not load DF40 for evaluation: {e}")

    if "wild_deepfake" in datasets_to_eval:
        wild_cfg = ds_cfg.get("wild_deepfake", {})
        wild_root = wild_cfg.get("root", str(Path(data_root) / "wild_deepfake"))
        try:
            ds = WildDeepfake(root=wild_root, split="test", transform=transform)
            if len(ds) > 0:
                loaders["WildDeepfake"] = DataLoader(
                    ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
                )
                logger.info(f"WildDeepfake test: {len(ds)} samples")
        except Exception as e:
            logger.warning(f"Could not load WildDeepfake for evaluation: {e}")

    return loaders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset deepfake detection evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/evaluation/",
        help="Directory to save plots and results.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate. Defaults to all in config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (cpu, cuda, mps). If None, auto-selects.",
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
    device = torch.device(args.device) if args.device else auto_device()

    # ── Load Model ────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_cfg = config.get("model", {})
    model = DeepfakeDetector(
        spatial_model_name=model_cfg.get("backbone", "vit_base_patch14_dinov2.lvd142m"),
        spatial_pretrained=False,  # Load from checkpoint
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 4),
        spatial_embed_dim=model_cfg.get("spatial_embed_dim", 768),
        freq_embed_dim=model_cfg.get("freq_embed_dim", 256),
        fusion_hidden_dims=model_cfg.get("fusion_hidden_dims", [512, 128]),
        fusion_dropout_rates=model_cfg.get("fusion_dropout", [0.3, 0.2]),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    # Apply temperature if calibrated
    temperature = checkpoint.get("temperature", None)
    if temperature is not None:
        logger.info(f"Applying temperature scaling: T={temperature:.4f}")
        calibrator = TemperatureScaling(initial_temperature=temperature)
        calibrator.to(device)

    # ── Build Loaders ─────────────────────────────────────────────────────
    eval_cfg = config.get("evaluation", {})
    datasets_to_eval = args.datasets or eval_cfg.get(
        "cross_dataset_eval", ["ff_plus_plus", "celeb_df", "df40", "wild_deepfake"]
    )

    logger.info(f"Evaluating on: {datasets_to_eval}")
    loaders = build_eval_loaders(
        config=config,
        datasets_to_eval=datasets_to_eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if not loaders:
        logger.error("No datasets found for evaluation. Check data paths in config.")
        sys.exit(1)

    # ── Cross-Dataset Evaluation ──────────────────────────────────────────
    evaluator = CrossDatasetEvaluator()
    collected = evaluator.evaluate_and_collect(model, loaders, device)

    # Build metrics dict for printing
    metrics_dict = {name: data["metrics"] for name, data in collected.items()}
    print_metrics_table(metrics_dict, title="Cross-Dataset Evaluation Results")

    # ── Save Results ──────────────────────────────────────────────────────
    # ROC curves
    roc_path = str(output_dir / "roc_curves.png")
    plot_roc_curve(collected, save_path=roc_path, title="Cross-Dataset ROC Curves")
    logger.info(f"ROC curves saved: {roc_path}")

    # Individual reliability diagrams
    for dataset_name, data in collected.items():
        probs = data["probs"]
        labels = data["labels"]

        cal_path = str(output_dir / f"calibration_{dataset_name.replace(' ', '_')}.png")
        from src.visualization.plots import plot_calibration_curve
        plot_calibration_curve(probs, labels, save_path=cal_path,
                                title=f"Calibration — {dataset_name}")

    # Save metrics as CSV
    try:
        import pandas as pd
        rows = []
        for name, metrics in metrics_dict.items():
            row = {"dataset": name}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = str(output_dir / "eval_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results CSV saved: {csv_path}")
    except ImportError:
        logger.warning("pandas not available. Skipping CSV export.")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
