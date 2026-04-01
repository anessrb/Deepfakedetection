"""
Spatial branch using DINOv2 ViT-B/14 as backbone.

Extracts rich semantic and structural features from face images using
a pretrained DINOv2 Vision Transformer. Only the last N transformer
blocks are fine-tuned; earlier layers are frozen for efficiency.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpatialBranch(nn.Module):
    """
    Spatial feature extractor using DINOv2 ViT-B/14.

    Uses the CLS token output of a DINOv2 Vision Transformer as a
    768-dimensional feature embedding representing spatial face features.

    The backbone is partially frozen: all layers except the last
    `unfreeze_last_n_blocks` transformer blocks and the norm layer
    are kept frozen for training efficiency.

    Args:
        model_name: timm model identifier. Defaults to DINOv2 ViT-B/14.
        pretrained: Whether to load pretrained ImageNet/DINOv2 weights.
        unfreeze_last_n_blocks: Number of final transformer blocks to fine-tune.
        embed_dim: Expected embedding dimension. Must match the model.
        drop_path_rate: Stochastic depth drop-path rate for fine-tuning blocks.
    """

    EMBED_DIM = 768  # ViT-B/14 CLS token dimension

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        unfreeze_last_n_blocks: int = 4,
        embed_dim: int = 768,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required. Install with: pip install timm>=0.9.12") from e

        logger.info(f"Loading backbone: {model_name} (pretrained={pretrained})")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # Remove classification head → returns CLS embedding
            drop_path_rate=drop_path_rate,
            img_size=224,           # Override default 518 to match our pipeline
        )

        actual_dim = self.backbone.num_features
        if actual_dim != embed_dim:
            logger.warning(
                f"Backbone embed dim ({actual_dim}) differs from configured embed_dim "
                f"({embed_dim}). Using actual: {actual_dim}."
            )
            self.embed_dim = actual_dim

        self._freeze_backbone()
        self._log_param_stats()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters, then selectively unfreeze last N blocks."""
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer blocks
        if hasattr(self.backbone, "blocks"):
            n_total = len(self.backbone.blocks)
            unfreeze_from = max(0, n_total - self.unfreeze_last_n_blocks)
            for i, block in enumerate(self.backbone.blocks):
                if i >= unfreeze_from:
                    for param in block.parameters():
                        param.requires_grad = True
            logger.info(
                f"Unfreezing blocks {unfreeze_from}..{n_total - 1} "
                f"({self.unfreeze_last_n_blocks}/{n_total} blocks)"
            )
        else:
            logger.warning(
                "Could not find 'blocks' attribute on backbone. "
                "No blocks will be unfrozen."
            )

        # Unfreeze final layer norm
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def _log_param_stats(self) -> None:
        """Log trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info(
            f"SpatialBranch params: {trainable:,} trainable / {total:,} total "
            f"({100 * trainable / total:.1f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from input images.

        Args:
            x: Input image tensor [B, 3, H, W]. H and W should be 224 for ViT-B/14.

        Returns:
            CLS token embedding [B, embed_dim] (768-d for ViT-B/14).
        """
        features = self.backbone(x)  # [B, embed_dim]
        return features

    def get_trainable_params(self):
        """Return list of trainable parameter groups with names."""
        return [(name, param) for name, param in self.backbone.named_parameters()
                if param.requires_grad]

    def __repr__(self) -> str:
        return (
            f"SpatialBranch(model={self.model_name}, "
            f"embed_dim={self.embed_dim}, "
            f"unfreeze_last_n={self.unfreeze_last_n_blocks})"
        )
