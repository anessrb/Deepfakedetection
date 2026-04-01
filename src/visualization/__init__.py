"""
Visualization subpackage for deepfake detection.

Provides Grad-CAM visualization for ViT models and plotting utilities
for calibration curves, ROC plots, and training history.
"""

from .gradcam import VitGradCAM
from .plots import (
    plot_calibration_curve,
    plot_roc_curve,
    plot_robustness_bars,
    plot_training_history,
)

__all__ = [
    "VitGradCAM",
    "plot_calibration_curve",
    "plot_roc_curve",
    "plot_robustness_bars",
    "plot_training_history",
]
