"""
Evaluation subpackage for deepfake detection.

Provides metrics computation, cross-dataset evaluation, and robustness
evaluation under image degradations.
"""

from .metrics import compute_auc, compute_ece, compute_metrics, print_metrics_table
from .cross_dataset_eval import CrossDatasetEvaluator
from .robustness_eval import RobustnessEvaluator

__all__ = [
    "compute_auc",
    "compute_ece",
    "compute_metrics",
    "print_metrics_table",
    "CrossDatasetEvaluator",
    "RobustnessEvaluator",
]
