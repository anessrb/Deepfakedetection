"""
Temperature Scaling for post-hoc probability calibration.

Temperature scaling is a simple but effective calibration method that
learns a single scalar 'temperature' T to divide logits by:
    calibrated_prob = sigmoid(logit / T)

A T > 1 makes the model less confident; T < 1 makes it more confident.
The optimal T is found by minimizing NLL on a held-out validation set.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.

    Wraps a trained model and divides its logits by a learned temperature
    parameter to produce well-calibrated probability estimates.

    Args:
        initial_temperature: Starting temperature value. Defaults to 1.0.
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([initial_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to raw logits.

        Args:
            logits: Raw model output logits [B, 1] or [B].

        Returns:
            Temperature-scaled logits, same shape as input.
        """
        return logits / self.temperature.clamp(min=1e-3)

    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50,
        verbose: bool = True,
    ) -> float:
        """
        Fit temperature parameter on a validation set using gradient descent.

        Collects all logits and labels from the validation set, then
        minimizes NLL with respect to temperature using LBFGS optimizer.

        Args:
            model: Trained model to calibrate (must have a forward() → logits).
            val_loader: DataLoader for the validation set.
            device: Computation device.
            lr: Learning rate for LBFGS optimizer.
            max_iter: Maximum optimization iterations.
            verbose: Whether to log calibration progress.

        Returns:
            Final optimal temperature value.
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).float()

                logits = model(images)  # [B, 1]
                if logits.ndim == 2:
                    logits = logits.squeeze(1)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)  # [N]
        all_labels = torch.cat(all_labels)  # [N]

        if verbose:
            logger.info(
                f"Calibrating temperature on {len(all_logits)} validation samples."
            )

        # Pre-calibration NLL
        with torch.no_grad():
            pre_nll = F.binary_cross_entropy_with_logits(all_logits, all_labels).item()

        # Optimize temperature via LBFGS
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature.clamp(min=1e-3)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Post-calibration NLL
        with torch.no_grad():
            scaled = all_logits / self.temperature.clamp(min=1e-3)
            post_nll = F.binary_cross_entropy_with_logits(scaled, all_labels).item()

        final_temp = self.temperature.item()
        if verbose:
            logger.info(
                f"Temperature calibration complete. "
                f"T = {final_temp:.4f} | "
                f"NLL: {pre_nll:.4f} → {post_nll:.4f}"
            )

        return final_temp

    def plot_reliability_diagram(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None,
    ):
        """
        Plot reliability diagram (calibration curve).

        Shows predicted probability bins vs actual fraction of positives.
        A perfectly calibrated model produces a diagonal line.

        Args:
            probs: Predicted probabilities [N].
            labels: True binary labels [N].
            n_bins: Number of equal-width probability bins.
            title: Plot title.
            save_path: If provided, save figure to this path.

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_predicted = []
        fraction_positive = []
        bin_counts = []

        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= low) & (probs < high)
            if mask.sum() > 0:
                mean_predicted.append(probs[mask].mean())
                fraction_positive.append(labels[mask].mean())
                bin_counts.append(mask.sum())
            else:
                mean_predicted.append((low + high) / 2)
                fraction_positive.append(np.nan)
                bin_counts.append(0)

        mean_predicted = np.array(mean_predicted)
        fraction_positive = np.array(fraction_positive)
        bin_counts = np.array(bin_counts)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})

        # Reliability diagram
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
        valid = ~np.isnan(fraction_positive)
        ax1.plot(
            mean_predicted[valid],
            fraction_positive[valid],
            "o-",
            color="#2196F3",
            linewidth=2,
            markersize=8,
            label="Model",
        )
        ax1.fill_between(
            mean_predicted[valid],
            fraction_positive[valid],
            mean_predicted[valid],
            alpha=0.2,
            color="#2196F3",
        )
        ax1.set_xlabel("Mean predicted probability", fontsize=12)
        ax1.set_ylabel("Fraction of positives", fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.bar(
            bin_centers,
            bin_counts / bin_counts.sum(),
            width=1.0 / n_bins * 0.9,
            color="#4CAF50",
            alpha=0.7,
            edgecolor="white",
        )
        ax2.set_xlabel("Predicted probability", fontsize=11)
        ax2.set_ylabel("Fraction of samples", fontsize=11)
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Reliability diagram saved to {save_path}")

        return fig

    def __repr__(self) -> str:
        return f"TemperatureScaling(T={self.temperature.item():.4f})"
