"""
Robustness evaluation under image degradations.

Tests deepfake detector performance under common real-world degradations:
- JPEG compression artifacts (various quality levels)
- Gaussian blur (various sigma levels)
- Resize (downsample + upsample to simulate low-resolution capture)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..datasets.augmentations import get_robustness_transforms
from .metrics import compute_auc

logger = logging.getLogger(__name__)


class DegradedDataset(Dataset):
    """
    Wrapper dataset that applies a fixed degradation transform.

    Args:
        base_dataset: Original dataset to wrap.
        transform: Degradation + normalization transform to apply.
    """

    def __init__(self, base_dataset: Dataset, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        # Re-load raw image and apply degradation transform
        image = self.base_dataset._load_image(sample["path"])

        if self.transform is not None:
            augmented = self.transform(image=image)
            sample = dict(sample)
            sample["image"] = augmented["image"]

        return sample


class RobustnessEvaluator:
    """
    Evaluates model robustness under controlled image degradations.

    Systematically applies different levels of JPEG compression, Gaussian blur,
    and resolution reduction, reporting AUC-ROC at each degradation level.
    """

    def evaluate_jpeg(
        self,
        model: nn.Module,
        dataset: Dataset,
        qualities: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Dict[int, float]:
        """
        Evaluate AUC under different JPEG compression qualities.

        Args:
            model: Detection model.
            dataset: Base dataset (untransformed images needed via _load_image).
            qualities: List of JPEG quality levels to test.
            device: Computation device.
            batch_size: DataLoader batch size.
            num_workers: DataLoader workers.

        Returns:
            Dict mapping jpeg_quality → AUC.
        """
        if qualities is None:
            qualities = [30, 50, 70, 90, 100]
        if device is None:
            device = self._auto_device()

        results = {}
        for quality in qualities:
            logger.info(f"  JPEG quality={quality}")
            transform = get_robustness_transforms(jpeg_quality=quality)
            auc = self._run_evaluation(model, dataset, transform, device, batch_size, num_workers)
            results[quality] = auc
            logger.info(f"    AUC: {auc:.4f}")

        return results

    def evaluate_blur(
        self,
        model: nn.Module,
        dataset: Dataset,
        sigmas: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Dict[float, float]:
        """
        Evaluate AUC under different Gaussian blur sigmas.

        Args:
            model: Detection model.
            dataset: Base dataset.
            sigmas: List of blur sigma values. 0 = no blur.
            device: Computation device.
            batch_size: DataLoader batch size.
            num_workers: DataLoader workers.

        Returns:
            Dict mapping sigma → AUC.
        """
        if sigmas is None:
            sigmas = [0, 1, 2, 3, 5]
        if device is None:
            device = self._auto_device()

        results = {}
        for sigma in sigmas:
            logger.info(f"  Blur sigma={sigma}")
            transform = get_robustness_transforms(blur_sigma=sigma if sigma > 0 else None)
            auc = self._run_evaluation(model, dataset, transform, device, batch_size, num_workers)
            results[sigma] = auc
            logger.info(f"    AUC: {auc:.4f}")

        return results

    def evaluate_resize(
        self,
        model: nn.Module,
        dataset: Dataset,
        scales: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Dict[float, float]:
        """
        Evaluate AUC under different resize scale factors.

        Images are downsampled to scale × original_size, then upsampled back,
        simulating low-resolution capture or compression.

        Args:
            model: Detection model.
            dataset: Base dataset.
            scales: List of scale factors (0.25 = 25% resolution).
            device: Computation device.
            batch_size: DataLoader batch size.
            num_workers: DataLoader workers.

        Returns:
            Dict mapping scale → AUC.
        """
        if scales is None:
            scales = [0.25, 0.5, 0.75, 1.0]
        if device is None:
            device = self._auto_device()

        results = {}
        for scale in scales:
            logger.info(f"  Resize scale={scale:.2f}")
            transform = get_robustness_transforms(
                resize_scale=scale if scale < 1.0 else None
            )
            auc = self._run_evaluation(model, dataset, transform, device, batch_size, num_workers)
            results[scale] = auc
            logger.info(f"    AUC: {auc:.4f}")

        return results

    @torch.no_grad()
    def _run_evaluation(
        self,
        model: nn.Module,
        dataset: Dataset,
        transform,
        device: torch.device,
        batch_size: int,
        num_workers: int,
    ) -> float:
        """Run inference with a specific transform and compute AUC."""
        # Wrap dataset with degradation transform
        if hasattr(dataset, "_load_image"):
            degraded_dataset = DegradedDataset(dataset, transform)
        else:
            # Fallback: just use the dataset as-is
            degraded_dataset = dataset
            logger.warning("Dataset doesn't support raw image loading. Using existing transforms.")

        loader = DataLoader(
            degraded_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model.eval()
        all_probs: List[float] = []
        all_labels: List[int] = []

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy()

            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        if not all_labels:
            return 0.5

        return compute_auc(np.array(all_labels), np.array(all_probs))

    def plot_robustness_curves(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None,
        figsize=(15, 5),
    ):
        """
        Plot robustness AUC curves for all degradation types.

        Args:
            results: Dict with keys 'jpeg', 'blur', 'resize', each mapping
                     degradation level → AUC.
            save_path: Path to save figure. Display only if None.
            figsize: Figure size.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        n_plots = sum(1 for k in ("jpeg", "blur", "resize") if k in results)
        if n_plots == 0:
            logger.warning("No robustness results to plot.")
            return None

        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        ax_idx = 0
        colors = ["#2196F3", "#4CAF50", "#FF5722"]

        if "jpeg" in results:
            ax = axes[ax_idx]
            jpeg_data = results["jpeg"]
            qualities = sorted(jpeg_data.keys())
            aucs = [jpeg_data[q] for q in qualities]
            ax.plot(qualities, aucs, "o-", color=colors[0], linewidth=2, markersize=8)
            ax.fill_between(qualities, aucs, alpha=0.15, color=colors[0])
            ax.set_xlabel("JPEG Quality", fontsize=12)
            ax.set_ylabel("AUC-ROC", fontsize=12)
            ax.set_title("JPEG Compression Robustness", fontsize=13, fontweight="bold")
            ax.set_ylim(0.4, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax_idx += 1

        if "blur" in results:
            ax = axes[ax_idx]
            blur_data = results["blur"]
            sigmas = sorted(blur_data.keys())
            aucs = [blur_data[s] for s in sigmas]
            ax.plot(sigmas, aucs, "o-", color=colors[1], linewidth=2, markersize=8)
            ax.fill_between(sigmas, aucs, alpha=0.15, color=colors[1])
            ax.set_xlabel("Blur Sigma", fontsize=12)
            ax.set_ylabel("AUC-ROC", fontsize=12)
            ax.set_title("Gaussian Blur Robustness", fontsize=13, fontweight="bold")
            ax.set_ylim(0.4, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax_idx += 1

        if "resize" in results:
            ax = axes[ax_idx]
            resize_data = results["resize"]
            scales = sorted(resize_data.keys())
            aucs = [resize_data[s] for s in scales]
            ax.plot(scales, aucs, "o-", color=colors[2], linewidth=2, markersize=8)
            ax.fill_between(scales, aucs, alpha=0.15, color=colors[2])
            ax.set_xlabel("Resize Scale", fontsize=12)
            ax.set_ylabel("AUC-ROC", fontsize=12)
            ax.set_title("Resolution Robustness", fontsize=13, fontweight="bold")
            ax.set_ylim(0.4, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Robustness plots saved to {save_path}")

        return fig

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
