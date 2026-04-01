"""
Loss functions for deepfake detection training.

Provides:
- FocalLoss: Handles class imbalance by down-weighting easy examples.
- ECELoss: Expected Calibration Error as a soft differentiable proxy.
- CombinedLoss: Weighted BCE + ECE regularization.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Reduces the relative loss for well-classified examples, focusing
    training on hard, misclassified examples. Especially useful for
    imbalanced datasets.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
               gamma=0 reduces to standard BCE. Defaults to 2.0.
        alpha: Class weight balancing factor in [0, 1]. Values close to 1
               increase weight on fake class. Defaults to 0.25.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output [B, 1] or [B].
            targets: Binary targets [B] in {0, 1} as float.

        Returns:
            Scalar loss (if reduction='mean' or 'sum').
        """
        # Ensure consistent shapes
        if logits.ndim == 2:
            logits = logits.squeeze(1)
        targets = targets.float()

        # Binary cross-entropy (unreduced)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probability for the actual class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ECELoss(nn.Module):
    """
    Differentiable Expected Calibration Error (ECE) loss.

    ECE measures the difference between predicted confidence and actual
    accuracy. Using it as a regularization term encourages the model to
    produce well-calibrated probabilities during training.

    This is a soft/differentiable approximation using binning.

    Args:
        n_bins: Number of equal-width probability bins. Defaults to 15.
    """

    def __init__(self, n_bins: int = 15) -> None:
        super().__init__()
        self.n_bins = n_bins
        # Register bin boundaries as buffers
        self.register_buffer(
            "bin_lowers",
            torch.linspace(0, 1 - 1.0 / n_bins, n_bins),
        )
        self.register_buffer(
            "bin_uppers",
            torch.linspace(1.0 / n_bins, 1, n_bins),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute differentiable ECE.

        Args:
            logits: Raw model logits [B, 1] or [B].
            targets: Binary targets [B] in {0, 1}.

        Returns:
            Scalar ECE loss value.
        """
        if logits.ndim == 2:
            logits = logits.squeeze(1)
        targets = targets.float()

        probs = torch.sigmoid(logits)
        ece = torch.tensor(0.0, device=logits.device, requires_grad=False)
        total = probs.shape[0]

        if total == 0:
            return ece

        ece = torch.zeros(1, device=logits.device)

        for low, high in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (probs > low) & (probs <= high)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence = probs[in_bin].mean()
                ece += torch.abs(avg_confidence - accuracy_in_bin) * prop_in_bin

        return ece.squeeze()


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted BCE + ECE regularization.

    Optionally uses Focal Loss instead of standard BCE.

    Args:
        use_focal: If True, use FocalLoss instead of BCE. Defaults to True.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.
        ece_lambda: Weight for ECE regularization term. Defaults to 0.1.
        ece_n_bins: Number of bins for ECE computation.
        pos_weight: Optional positive class weight for BCE.
    """

    def __init__(
        self,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        ece_lambda: float = 0.1,
        ece_n_bins: int = 15,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.ece_lambda = ece_lambda

        if use_focal:
            self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            pw = torch.tensor([pos_weight]) if pos_weight else None
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.ece_loss = ECELoss(n_bins=ece_n_bins)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            logits: Raw model logits [B, 1] or [B].
            targets: Binary targets [B].

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        if logits.ndim == 2:
            logits_1d = logits.squeeze(1)
        else:
            logits_1d = logits

        targets = targets.float()
        cls = self.cls_loss(logits_1d, targets)
        ece = self.ece_loss(logits_1d, targets)
        total = cls + self.ece_lambda * ece

        return total, {
            "total": total.item(),
            "cls": cls.item(),
            "ece": ece.item(),
        }
