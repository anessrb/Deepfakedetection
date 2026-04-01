"""
Grad-CAM implementation for Vision Transformer (DINOv2 ViT).

Attention rollout and gradient-based class activation mapping for ViT
models. Produces spatial heatmaps highlighting face regions that most
contribute to the deepfake detection decision.

References:
    - Selvaraju et al., "Grad-CAM", ICCV 2017.
    - Chefer et al., "Transformer Interpretability Beyond Attention Visualization", CVPR 2021.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class VitGradCAM:
    """
    Gradient-based Class Activation Mapping for Vision Transformers.

    Computes a saliency heatmap by computing the gradient of the output
    with respect to the last attention layer's patch token features,
    then aggregating them spatially.

    Supports both the spatial branch backbone and the full detector model.

    Args:
        model: The full DeepfakeDetector model or a standalone ViT.
        target_layer_name: Name of the target transformer block. If None,
                            uses the last attention block automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.target_layer_name = target_layer_name

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: List = []

        self._register_hooks()

    def _get_target_layer(self) -> Optional[nn.Module]:
        """Find the target layer in the model hierarchy."""
        # Navigate into spatial_branch.backbone.blocks[-1] if detector model
        backbone = None

        if hasattr(self.model, "spatial_branch"):
            backbone = self.model.spatial_branch.backbone
        elif hasattr(self.model, "backbone"):
            backbone = self.model.backbone
        else:
            backbone = self.model

        if self.target_layer_name is not None:
            # Try to find by name
            for name, module in backbone.named_modules():
                if name == self.target_layer_name:
                    return module
            logger.warning(f"Target layer '{self.target_layer_name}' not found.")

        # Default: last transformer block's attention
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            last_block = backbone.blocks[-1]
            # Try common ViT attention attribute names
            for attr in ["attn", "attention", "self_attn"]:
                if hasattr(last_block, attr):
                    return getattr(last_block, attr)
            return last_block  # Fallback to full last block
        return None

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        target_layer = self._get_target_layer()
        if target_layer is None:
            logger.warning("Could not find target layer for Grad-CAM hooks.")
            return

        def forward_hook(module, input, output):
            # For attention layers, output might be a tuple (output, weights)
            if isinstance(output, tuple):
                self._activations = output[0].detach()
            else:
                self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self._gradients = grad_output[0].detach()
            else:
                self._gradients = grad_output.detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))
        logger.debug(f"Grad-CAM hooks registered on: {target_layer.__class__.__name__}")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor: Input image tensor [1, 3, H, W] or [3, H, W].
                          Will be expanded to batch size 1 if needed.
            target_class: Target output class (1 = fake, 0 = real).

        Returns:
            Heatmap as numpy array [H, W] in range [0, 1].
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_(False)

        self.model.eval()
        self._gradients = None
        self._activations = None

        # Forward pass with gradient tracking
        image_tensor_grad = image_tensor.clone().requires_grad_(True)

        output = self.model(image_tensor_grad)  # [1, 1]
        if output.ndim == 2:
            score = output[0, 0]
        else:
            score = output[0]

        # For fake class (1), maximize score; for real class (0), minimize it
        if target_class == 1:
            score.backward()
        else:
            (-score).backward()

        if self._gradients is None or self._activations is None:
            logger.warning(
                "Grad-CAM: gradients or activations not captured. "
                "Returning blank heatmap."
            )
            h, w = image_tensor.shape[-2], image_tensor.shape[-1]
            return np.zeros((h, w), dtype=np.float32)

        return self._compute_heatmap(image_tensor.shape[-2:])

    def _compute_heatmap(self, spatial_size: Tuple[int, int]) -> np.ndarray:
        """
        Compute spatial heatmap from captured gradients and activations.

        For ViT models, patch tokens are used to generate the spatial map.
        The CLS token (position 0) is excluded.

        Args:
            spatial_size: (H, W) of the original image.

        Returns:
            Normalized heatmap array [H, W] in [0, 1].
        """
        grads = self._gradients   # [B, N+1, D] or [B, D, H, W]
        acts = self._activations  # same shape

        if grads is None or acts is None:
            return np.zeros(spatial_size, dtype=np.float32)

        # Handle ViT token format: [B, N+1, D]
        if grads.ndim == 3:
            # Exclude CLS token at position 0
            grads_patches = grads[0, 1:, :]   # [N_patches, D]
            acts_patches = acts[0, 1:, :]     # [N_patches, D]

            # Global average pool over channels (gradient-weighted)
            weights = grads_patches.mean(dim=-1)  # [N_patches]
            cam = (weights * acts_patches.mean(dim=-1))  # [N_patches]

        elif grads.ndim == 4:
            # CNN-style: [B, C, H, W]
            weights = grads[0].mean(dim=(-2, -1), keepdim=True)  # [C, 1, 1]
            cam_map = (weights * acts[0]).sum(dim=0)  # [H, W]
            cam = cam_map.flatten()
        else:
            return np.zeros(spatial_size, dtype=np.float32)

        # Normalize cam values
        cam_np = cam.cpu().float().numpy()
        cam_np = np.maximum(cam_np, 0)  # ReLU

        # Infer patch grid size
        n_patches = len(cam_np)
        grid_size = int(np.sqrt(n_patches))
        if grid_size * grid_size != n_patches:
            # Non-square patch grid — use closest square
            grid_size = int(np.ceil(np.sqrt(n_patches)))
            # Pad to square
            padded = np.zeros(grid_size * grid_size)
            padded[:n_patches] = cam_np
            cam_np = padded

        cam_2d = cam_np.reshape(grid_size, grid_size)

        # Upsample to original image size
        h, w = spatial_size
        cam_resized = cv2.resize(cam_2d, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        if cam_resized.max() > cam_resized.min():
            cam_resized = (cam_resized - cam_resized.min()) / (
                cam_resized.max() - cam_resized.min()
            )
        else:
            cam_resized = np.zeros_like(cam_resized)

        return cam_resized.astype(np.float32)

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Overlay a heatmap on an image with a colormap.

        Args:
            image: RGB image [H, W, 3] as uint8 numpy array.
            heatmap: Heatmap [H, W] in range [0, 1].
            alpha: Blend factor (0 = image only, 1 = heatmap only).
            colormap: OpenCV colormap constant.

        Returns:
            PIL Image with heatmap overlay.
        """
        # Apply colormap (expects uint8)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        image_float = image.astype(np.float32) / 255.0
        heatmap_float = colored_heatmap.astype(np.float32) / 255.0
        overlay = (1 - alpha) * image_float + alpha * heatmap_float
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

        return Image.fromarray(overlay)

    def visualize_batch(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: List[int],
        probs: List[float],
        n_cols: int = 4,
        save_path: Optional[str] = None,
    ):
        """
        Create a visualization grid for a batch of images.

        Shows original image, Grad-CAM overlay, predicted probability,
        and true label for each sample in the batch.

        Args:
            model: Detection model.
            images: Batch of image tensors [B, 3, H, W].
            labels: True labels [B].
            probs: Predicted probabilities [B].
            n_cols: Number of columns in the grid.
            save_path: Path to save the figure.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        from ..datasets.augmentations import denormalize

        n = min(len(images), 16)  # Cap at 16 for visibility
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 4, n_rows * 3))
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        for i in range(n):
            row = i // n_cols
            col_start = (i % n_cols) * 2

            img_tensor = images[i:i+1]
            true_label = labels[i]
            prob = probs[i]

            # Denormalize for display
            img_display = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
            img_display = (img_display * 255).astype(np.uint8)

            # Generate heatmap
            try:
                heatmap = self.generate(img_tensor)
                overlay_img = self.overlay(img_display, heatmap)
                overlay_np = np.array(overlay_img)
            except Exception as e:
                logger.warning(f"Grad-CAM failed for sample {i}: {e}")
                overlay_np = img_display

            # Original image
            ax_orig = axes[row, col_start]
            ax_orig.imshow(img_display)
            label_str = "FAKE" if true_label == 1 else "REAL"
            color = "#F44336" if true_label == 1 else "#4CAF50"
            ax_orig.set_title(f"GT: {label_str}", fontsize=9, color=color, fontweight="bold")
            ax_orig.axis("off")

            # Overlay
            ax_cam = axes[row, col_start + 1]
            ax_cam.imshow(overlay_np)
            pred_str = "FAKE" if prob >= 0.5 else "REAL"
            pred_color = "#F44336" if prob >= 0.5 else "#4CAF50"
            ax_cam.set_title(f"P={prob:.2f} ({pred_str})", fontsize=9,
                              color=pred_color, fontweight="bold")
            ax_cam.axis("off")

        # Hide unused subplots
        total_slots = n_rows * n_cols
        for i in range(n, total_slots):
            row = i // n_cols
            col_start = (i % n_cols) * 2
            axes[row, col_start].axis("off")
            axes[row, col_start + 1].axis("off")

        plt.suptitle("Deepfake Detection — Grad-CAM Visualization",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Grad-CAM visualization saved to {save_path}")

        return fig

    def __del__(self):
        """Clean up hooks on destruction."""
        try:
            self.remove_hooks()
        except Exception:
            pass
