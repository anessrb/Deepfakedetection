"""
Full DeepfakeDetector model combining spatial and frequency branches.

The detector fuses DINOv2 spatial features (768-d) with frequency-domain
LightCNN features (256-d) through a multi-layer perceptron to produce
a single fake probability score.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch

logger = logging.getLogger(__name__)


class FusionMLP(nn.Module):
    """
    MLP for fusing spatial and frequency feature embeddings.

    Args:
        input_dim: Total input feature dimension (spatial + frequency).
        hidden_dims: List of hidden layer sizes.
        dropout_rates: Dropout probabilities after each hidden layer.
        output_dim: Output dimension. 1 for binary classification logit.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 128]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2]

        # Pad dropout_rates if shorter than hidden_dims
        while len(dropout_rates) < len(hidden_dims):
            dropout_rates.append(dropout_rates[-1] if dropout_rates else 0.1)

        layers = []
        in_dim = input_dim
        for out_dim, drop_p in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(drop_p),
            ])
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepfakeDetector(nn.Module):
    """
    Full deepfake detection model with dual-branch architecture.

    Combines:
    - SpatialBranch: DINOv2 ViT-B/14 → 768-d CLS token
    - FrequencyBranch: FFT + DCT + LightCNN → 256-d embedding
    - FusionMLP: (768 + 256) → 512 → 128 → 1 (logit)

    Args:
        spatial_model_name: timm model name for the spatial branch.
        spatial_pretrained: Whether to use pretrained spatial backbone.
        unfreeze_last_n_blocks: Number of ViT blocks to fine-tune.
        spatial_embed_dim: Spatial branch output dimension.
        freq_embed_dim: Frequency branch output dimension.
        freq_conv_channels: LightCNN channel sizes.
        fusion_hidden_dims: Fusion MLP hidden layer sizes.
        fusion_dropout_rates: Fusion MLP dropout rates.
        drop_path_rate: ViT stochastic depth rate.
    """

    def __init__(
        self,
        spatial_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        spatial_pretrained: bool = True,
        unfreeze_last_n_blocks: int = 4,
        spatial_embed_dim: int = 768,
        freq_embed_dim: int = 256,
        freq_conv_channels: Optional[List[int]] = None,
        fusion_hidden_dims: Optional[List[int]] = None,
        fusion_dropout_rates: Optional[List[float]] = None,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Spatial Branch ──────────────────────────────────────────────────
        self.spatial_branch = SpatialBranch(
            model_name=spatial_model_name,
            pretrained=spatial_pretrained,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            embed_dim=spatial_embed_dim,
            drop_path_rate=drop_path_rate,
        )
        actual_spatial_dim = self.spatial_branch.embed_dim

        # ── Frequency Branch ─────────────────────────────────────────────────
        self.freq_branch = FrequencyBranch(
            embed_dim=freq_embed_dim,
            conv_channels=freq_conv_channels,
        )

        # ── Fusion MLP ───────────────────────────────────────────────────────
        total_dim = actual_spatial_dim + freq_embed_dim
        self.fusion_mlp = FusionMLP(
            input_dim=total_dim,
            hidden_dims=fusion_hidden_dims or [512, 128],
            dropout_rates=fusion_dropout_rates or [0.3, 0.2],
            output_dim=1,
        )

        self._log_params()

    def _log_params(self) -> None:
        """Log total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"DeepfakeDetector | Total params: {total:,} | "
            f"Trainable: {trainable:,} ({100 * trainable / total:.1f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute raw logit for fake/real classification.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            Raw logit [B, 1]. Apply sigmoid for probability.
        """
        # Extract features from both branches
        spatial_feat = self.spatial_branch(x)   # [B, 768]
        freq_feat = self.freq_branch(x)           # [B, 256]

        # Concatenate and fuse
        fused = torch.cat([spatial_feat, freq_feat], dim=1)  # [B, 1024]
        logit = self.fusion_mlp(fused)                       # [B, 1]
        return logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute calibrated fake probability.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            Fake probability in [0, 1] with shape [B, 1].
        """
        logit = self.forward(x)
        return torch.sigmoid(logit)

    def get_optimizer_param_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> List[Dict]:
        """
        Build optimizer parameter groups with different learning rates.

        The frozen backbone parameters are excluded; trainable backbone
        blocks get a lower LR than the fusion head and frequency branch.

        Args:
            lr_backbone: Learning rate for unfrozen backbone parameters.
            lr_head: Learning rate for freq branch, fusion MLP, backbone norm.
            weight_decay: L2 regularization coefficient.

        Returns:
            List of parameter group dicts for torch.optim optimizers.
        """
        # Backbone trainable params (last N blocks + norm)
        backbone_params = [
            p for p in self.spatial_branch.parameters() if p.requires_grad
        ]

        # All other trainable params (freq branch + fusion MLP)
        head_params = (
            list(self.freq_branch.parameters()) +
            list(self.fusion_mlp.parameters())
        )

        param_groups = [
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        logger.info(
            f"Optimizer groups: "
            + " | ".join(
                f"{g['name']}: {sum(p.numel() for p in g['params']):,} params @ lr={g['lr']:.2e}"
                for g in param_groups
            )
        )
        return param_groups

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        """
        Save model state dict and optional metadata.

        Args:
            path: File path to save checkpoint.
            extra: Additional metadata to include in checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "spatial_embed_dim": self.spatial_branch.embed_dim,
                "freq_embed_dim": self.freq_branch.embed_dim,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, **kwargs) -> "DeepfakeDetector":
        """
        Load a model from a checkpoint file.

        Args:
            path: Path to checkpoint file.
            device: Target device. Auto-detected if None.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Loaded DeepfakeDetector instance.
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        checkpoint = torch.load(path, map_location=device)
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info(f"Model loaded from {path} on {device}")
        return model

    def __repr__(self) -> str:
        return (
            f"DeepfakeDetector(\n"
            f"  spatial={self.spatial_branch},\n"
            f"  freq={self.freq_branch},\n"
            f"  fusion={self.fusion_mlp}\n"
            f")"
        )
