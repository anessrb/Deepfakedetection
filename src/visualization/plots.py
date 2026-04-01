"""
Plotting utilities for deepfake detection evaluation and analysis.

Provides production-quality matplotlib figures for:
- Calibration curves (reliability diagrams)
- Multi-dataset ROC curves
- Robustness bar charts
- Training history (loss + AUC)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def plot_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 7),
):
    """
    Plot reliability diagram (calibration curve).

    Args:
        probs: Predicted probabilities [N].
        labels: True binary labels [N].
        n_bins: Number of equal-width probability bins.
        title: Plot title.
        save_path: Save path. Display only if None.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    try:
        fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins)
    except Exception as e:
        logger.warning(f"calibration_curve failed: {e}. Computing manually.")
        bins = np.linspace(0, 1, n_bins + 1)
        mean_pred, fraction_pos = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                mean_pred.append(probs[mask].mean())
                fraction_pos.append(labels[mask].mean())
        mean_pred = np.array(mean_pred)
        fraction_pos = np.array(fraction_pos)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", zorder=1)

    # Model calibration
    ax1.plot(
        mean_pred, fraction_pos,
        "o-", color="#2196F3", linewidth=2, markersize=8,
        label="Model", zorder=2,
    )
    ax1.fill_between(
        mean_pred, fraction_pos, mean_pred,
        alpha=0.2, color="#2196F3",
    )

    ax1.set_xlabel("Mean predicted probability", fontsize=12)
    ax1.set_ylabel("Fraction of positives", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Histogram
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(probs, bins=bins)
    ax2.bar(bin_centers, counts / counts.sum(), width=1.0 / n_bins * 0.9,
            color="#4CAF50", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Fraction", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Calibration curve saved to {save_path}")

    return fig


def plot_roc_curve(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 7),
    title: str = "ROC Curves",
):
    """
    Plot multi-dataset ROC curves.

    Args:
        results_dict: Dict mapping dataset_name → dict with keys:
                      'probs' (np.ndarray), 'labels' (np.ndarray), 'metrics' (dict).
        save_path: Save path.
        figsize: Figure dimensions.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results_dict), 1)))

    for (name, data), color in zip(results_dict.items(), colors):
        probs = data.get("probs", np.array([]))
        labels = data.get("labels", np.array([]))

        if len(probs) == 0 or len(labels) == 0:
            continue

        try:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = data.get("metrics", {}).get("auc", 0.0)
            ax.plot(fpr, tpr, linewidth=2, color=color,
                    label=f"{name} (AUC={auc:.3f})")
        except Exception as e:
            logger.warning(f"ROC curve failed for {name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_robustness_bars(
    robustness_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    Plot robustness evaluation as bar charts.

    Args:
        robustness_results: Dict with keys 'jpeg', 'blur', 'resize',
                            each mapping level → AUC.
        save_path: Save path.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure or None.
    """
    import matplotlib.pyplot as plt

    available = {k: v for k, v in robustness_results.items()
                 if k in ("jpeg", "blur", "resize") and v}
    if not available:
        logger.warning("No robustness results to plot.")
        return None

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    palette = {"jpeg": "#2196F3", "blur": "#4CAF50", "resize": "#FF5722"}
    titles = {
        "jpeg": "JPEG Compression",
        "blur": "Gaussian Blur",
        "resize": "Resolution (Scale)",
    }
    xlabels = {
        "jpeg": "JPEG Quality",
        "blur": "Blur Sigma",
        "resize": "Scale Factor",
    }

    for ax, (key, data) in zip(axes, available.items()):
        levels = sorted(data.keys())
        aucs = [data[l] for l in levels]
        labels = [str(l) for l in levels]

        bars = ax.bar(labels, aucs, color=palette.get(key, "#9C27B0"),
                      alpha=0.85, edgecolor="white", linewidth=1.2)

        # Value labels on bars
        for bar, auc in zip(bars, aucs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{auc:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
        ax.set_xlabel(xlabels.get(key, key), fontsize=11)
        ax.set_ylabel("AUC-ROC", fontsize=11)
        ax.set_title(titles.get(key, key), fontsize=13, fontweight="bold")
        ax.set_ylim(0.0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Robustness Evaluation", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Robustness bars saved to {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot training and validation loss + AUC curves.

    Args:
        history: Dict with keys: train_loss, val_loss, train_auc, val_auc,
                 val_ece, lr.
        save_path: Save path.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        logger.warning("Empty training history.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Loss
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], "o-", color="#2196F3",
                linewidth=2, markersize=4, label="Train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "s-", color="#FF5722",
                linewidth=2, markersize=4, label="Val")
    ax.set_title("Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1]
    if "train_auc" in history:
        ax.plot(epochs, history["train_auc"], "o-", color="#2196F3",
                linewidth=2, markersize=4, label="Train")
    if "val_auc" in history:
        ax.plot(epochs, history["val_auc"], "s-", color="#FF5722",
                linewidth=2, markersize=4, label="Val")
    ax.set_title("AUC-ROC", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ECE
    ax = axes[2]
    if "val_ece" in history and history["val_ece"]:
        ax.plot(epochs, history["val_ece"], "^-", color="#9C27B0",
                linewidth=2, markersize=4, label="Val ECE")
        ax.set_title("Expected Calibration Error (Val)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("ECE", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    # Learning Rate
    ax = axes[3]
    if "lr" in history and history["lr"]:
        ax.semilogy(epochs, history["lr"], "D-", color="#4CAF50",
                    linewidth=2, markersize=4, label="LR")
        ax.set_title("Learning Rate", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("LR (log scale)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    plt.suptitle("Training History", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    return fig
