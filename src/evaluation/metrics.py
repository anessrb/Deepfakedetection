"""
Evaluation metrics for deepfake detection.

Implements AUC-ROC, ECE, accuracy, average precision, and
threshold-at-95%-TPR metrics with clean printing utilities.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC-ROC).

    Args:
        labels: True binary labels [N], values in {0, 1}.
        probs: Predicted fake probabilities [N], values in [0, 1].

    Returns:
        AUC-ROC score in [0, 1]. Returns 0.5 on failure.
    """
    try:
        if len(np.unique(labels)) < 2:
            logger.warning("Only one class present in labels. AUC is undefined, returning 0.5.")
            return 0.5
        return float(roc_auc_score(labels, probs))
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}. Returning 0.5.")
        return 0.5


def compute_ece(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures the average discrepancy between predicted confidence and
    actual accuracy across probability bins.

    Args:
        labels: True binary labels [N].
        probs: Predicted probabilities [N].
        n_bins: Number of equal-width bins. Defaults to 15.

    Returns:
        ECE score in [0, 1]. Lower is better (0 = perfect calibration).
    """
    if len(labels) == 0:
        return 0.0

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probs >= low) & (probs < high)
        if mask.sum() == 0:
            continue

        bin_probs = probs[mask]
        bin_labels = labels[mask]
        bin_size = mask.sum()

        avg_confidence = bin_probs.mean()
        accuracy = bin_labels.mean()
        ece += (bin_size / n) * abs(avg_confidence - accuracy)

    return float(ece)


def compute_threshold_at_tpr(
    labels: np.ndarray,
    probs: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """
    Find the decision threshold achieving a target True Positive Rate.

    Args:
        labels: True binary labels [N].
        probs: Predicted fake probabilities [N].
        target_tpr: Target TPR (e.g., 0.95 for 95% TPR).

    Returns:
        Threshold value in [0, 1]. Returns 0.5 on failure.
    """
    try:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        # Find threshold where TPR >= target_tpr
        idx = np.where(tpr >= target_tpr)[0]
        if len(idx) == 0:
            return float(thresholds[-1])
        # Take the threshold with highest TPR >= target (largest index = smallest threshold)
        return float(thresholds[idx[0]])
    except Exception as e:
        logger.warning(f"Threshold computation failed: {e}")
        return 0.5


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of evaluation metrics.

    Args:
        labels: True binary labels [N] in {0, 1}.
        probs: Predicted fake probabilities [N] in [0, 1].
        threshold: Decision threshold for accuracy computation.
        n_bins: Number of bins for ECE computation.

    Returns:
        Dictionary with keys: auc, ece, accuracy, ap, threshold_at_95_tpr.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    if len(labels) == 0:
        return {
            "auc": 0.0,
            "ece": 0.0,
            "accuracy": 0.0,
            "ap": 0.0,
            "threshold_at_95_tpr": 0.5,
        }

    predictions = (probs >= threshold).astype(int)

    # AUC
    auc = compute_auc(labels, probs)

    # ECE
    ece = compute_ece(labels, probs, n_bins=n_bins)

    # Accuracy
    acc = float(accuracy_score(labels, predictions))

    # Average Precision (area under PR curve)
    try:
        ap = float(average_precision_score(labels, probs))
    except Exception:
        ap = 0.0

    # Threshold at 95% TPR
    thresh_95 = compute_threshold_at_tpr(labels, probs, target_tpr=0.95)

    return {
        "auc": auc,
        "ece": ece,
        "accuracy": acc,
        "ap": ap,
        "threshold_at_95_tpr": thresh_95,
    }


def print_metrics_table(
    results_dict: Dict[str, Dict[str, float]],
    title: str = "Evaluation Results",
) -> None:
    """
    Pretty-print a comparison table of metrics across multiple datasets.

    Args:
        results_dict: Dictionary mapping dataset_name → metrics_dict.
                      Each metrics_dict should have: auc, ece, accuracy, ap.
        title: Table header title.
    """
    if not results_dict:
        print("No results to display.")
        return

    # Collect all metric keys
    all_keys = set()
    for metrics in results_dict.values():
        all_keys.update(metrics.keys())

    metric_order = ["auc", "accuracy", "ap", "ece", "threshold_at_95_tpr"]
    display_keys = [k for k in metric_order if k in all_keys]
    display_keys += [k for k in sorted(all_keys) if k not in metric_order]

    # Column formatting
    dataset_col_w = max(20, max(len(k) for k in results_dict.keys()) + 2)
    metric_col_w = 16
    headers = ["Dataset"] + [k.upper().replace("_", " ") for k in display_keys]
    col_widths = [dataset_col_w] + [metric_col_w] * len(display_keys)

    # Build separator line
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(
        f" {h:<{w}} " for h, w in zip(headers, col_widths)
    ) + "|"

    print()
    print(f"  {title}")
    print(sep)
    print(header_row)
    print(sep)

    for dataset_name, metrics in results_dict.items():
        row_values = [dataset_name]
        for key in display_keys:
            val = metrics.get(key, float("nan"))
            if isinstance(val, float):
                row_values.append(f"{val:.4f}")
            else:
                row_values.append(str(val))

        row = "|" + "|".join(
            f" {v:<{w}} " for v, w in zip(row_values, col_widths)
        ) + "|"
        print(row)

    print(sep)
    print()
