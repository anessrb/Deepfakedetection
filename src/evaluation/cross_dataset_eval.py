"""
Cross-dataset evaluation for deepfake detection generalization.

Evaluates a trained model on multiple datasets to measure how well
it generalizes beyond the training distribution.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, print_metrics_table

logger = logging.getLogger(__name__)


class CrossDatasetEvaluator:
    """
    Evaluator for cross-dataset generalization assessment.

    Runs inference on multiple dataset loaders and computes standardized
    metrics for each, enabling fair comparison of generalization ability.
    """

    def evaluate(
        self,
        model: nn.Module,
        dataset_loaders: Dict[str, DataLoader],
        device: torch.device,
        threshold: float = 0.5,
        use_amp: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on multiple datasets.

        Args:
            model: Trained deepfake detection model.
            dataset_loaders: Dict mapping dataset name → DataLoader.
            device: Computation device.
            threshold: Decision threshold for accuracy.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Dict mapping dataset_name → metrics_dict.
        """
        model.eval()
        results = {}

        for dataset_name, loader in dataset_loaders.items():
            logger.info(f"Evaluating on: {dataset_name}")
            metrics = self._evaluate_single(
                model=model,
                loader=loader,
                device=device,
                dataset_name=dataset_name,
                threshold=threshold,
                use_amp=use_amp,
            )
            results[dataset_name] = metrics
            logger.info(
                f"  {dataset_name}: AUC={metrics['auc']:.4f} | "
                f"ACC={metrics['accuracy']:.4f} | ECE={metrics['ece']:.4f}"
            )

        return results

    @torch.no_grad()
    def _evaluate_single(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        dataset_name: str,
        threshold: float = 0.5,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single dataset loader.

        Args:
            model: Detection model.
            loader: DataLoader for the dataset.
            device: Computation device.
            dataset_name: Name for logging purposes.
            threshold: Decision threshold.
            use_amp: Use AMP inference.

        Returns:
            Metrics dictionary.
        """
        all_probs: List[float] = []
        all_labels: List[int] = []

        pbar = tqdm(loader, desc=f"  [{dataset_name}]", leave=False, dynamic_ncols=True)

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy()

            if use_amp and device.type == "cuda":
                from torch.amp import autocast
                with autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        labels_arr = np.array(all_labels)
        probs_arr = np.array(all_probs)

        if len(labels_arr) == 0:
            logger.warning(f"No samples found for dataset: {dataset_name}")
            return {"auc": 0.0, "ece": 0.0, "accuracy": 0.0, "ap": 0.0}

        return compute_metrics(labels_arr, probs_arr, threshold=threshold)

    def print_results_table(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Cross-Dataset Evaluation",
    ) -> None:
        """
        Print formatted cross-dataset results table.

        Args:
            results: Dict mapping dataset_name → metrics_dict.
            title: Table title.
        """
        print_metrics_table(results, title=title)

    def plot_roc_curves(
        self,
        results: Dict[str, Dict],
        all_probs: Optional[Dict[str, np.ndarray]] = None,
        all_labels: Optional[Dict[str, np.ndarray]] = None,
        save_path: Optional[str] = None,
        figsize=(10, 7),
    ):
        """
        Plot ROC curves for all evaluated datasets.

        Note: This method requires raw predictions and labels to be passed
        separately (they are not stored in the metrics dict). Use the
        `evaluate_and_collect` method to get these, or call
        `src.visualization.plots.plot_roc_curve` directly.

        Args:
            results: Metrics results dict (for AUC labels).
            all_probs: Dict mapping dataset_name → probability arrays.
            all_labels: Dict mapping dataset_name → label arrays.
            save_path: Path to save figure. If None, display only.
            figsize: Figure size tuple.

        Returns:
            matplotlib Figure or None.
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        if all_probs is None or all_labels is None:
            logger.warning(
                "Raw predictions not provided. Cannot plot ROC curves. "
                "Use evaluate_and_collect() to obtain predictions."
            )
            return None

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_probs)))

        for (name, probs), (_, labels), color in zip(
            all_probs.items(), all_labels.items(), colors
        ):
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = results.get(name, {}).get("auc", 0.0)
            ax.plot(fpr, tpr, linewidth=2, color=color,
                    label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — Cross-Dataset Evaluation", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curves saved to {save_path}")

        return fig

    def evaluate_and_collect(
        self,
        model: nn.Module,
        dataset_loaders: Dict[str, DataLoader],
        device: torch.device,
        use_amp: bool = False,
    ) -> Dict[str, Dict]:
        """
        Evaluate model and collect raw predictions for plotting.

        Args:
            model: Detection model.
            dataset_loaders: Dict of dataset loaders.
            device: Computation device.
            use_amp: Use AMP.

        Returns:
            Dict with 'metrics', 'probs', 'labels' sub-dicts keyed by dataset name.
        """
        model.eval()
        collected = {}

        for name, loader in dataset_loaders.items():
            logger.info(f"Collecting predictions for: {name}")
            all_probs: List[float] = []
            all_labels: List[int] = []

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"  [{name}]", leave=False):
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].numpy()

                    if use_amp and device.type == "cuda":
                        from torch.amp import autocast
                        with autocast("cuda"):
                            logits = model(images)
                    else:
                        logits = model(images)

                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labels.tolist())

            labels_arr = np.array(all_labels)
            probs_arr = np.array(all_probs)
            metrics = compute_metrics(labels_arr, probs_arr)

            collected[name] = {
                "metrics": metrics,
                "probs": probs_arr,
                "labels": labels_arr,
            }

        return collected
