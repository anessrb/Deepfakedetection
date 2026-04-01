"""
Augmentation pipelines for deepfake detection training and evaluation.

Uses albumentations for efficient, composable image augmentations including
JPEG compression, Gaussian blur, and resize degradations for robustness testing.
"""

from typing import List, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    jpeg_quality_lower: int = 30,
    jpeg_quality_upper: int = 100,
    horizontal_flip_p: float = 0.5,
    jpeg_p: float = 0.5,
    blur_p: float = 0.3,
    color_jitter_p: float = 0.5,
) -> A.Compose:
    """
    Build training augmentation pipeline.

    Includes random horizontal flip, color jitter, JPEG compression artifacts,
    Gaussian blur, random resized crop, and normalization.

    Args:
        img_size: Target output image size (square).
        mean: Normalization mean per channel.
        std: Normalization std per channel.
        jpeg_quality_lower: Minimum JPEG quality (30-100).
        jpeg_quality_upper: Maximum JPEG quality (30-100).
        horizontal_flip_p: Probability of horizontal flip.
        jpeg_p: Probability of JPEG compression augmentation.
        blur_p: Probability of Gaussian blur.
        color_jitter_p: Probability of color jitter.

    Returns:
        albumentations.Compose transform pipeline.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=horizontal_flip_p),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=color_jitter_p,
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(0.1, 2.0),
                p=blur_p,
            ),
            A.ImageCompression(
                quality_range=(jpeg_quality_lower, jpeg_quality_upper),
                p=jpeg_p,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Build validation / inference augmentation pipeline.

    Minimal transforms: resize, center crop, normalize. No randomness.

    Args:
        img_size: Target output image size (square).
        mean: Normalization mean per channel.
        std: Normalization std per channel.

    Returns:
        albumentations.Compose transform pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=int(img_size * 1.143), width=int(img_size * 1.143)),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_robustness_transforms(
    jpeg_quality: Optional[int] = None,
    blur_sigma: Optional[float] = None,
    resize_scale: Optional[float] = None,
    img_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Build robustness evaluation transforms with specific degradation levels.

    Applies a single deterministic degradation (JPEG, blur, or resize)
    for systematic robustness evaluation.

    Args:
        jpeg_quality: Fixed JPEG quality level (1-100). None = no JPEG.
        blur_sigma: Fixed Gaussian blur sigma. None or 0 = no blur.
        resize_scale: Fraction to downsample then upsample (0-1).
                      None or 1.0 = no resize degradation.
        img_size: Output image size.
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        albumentations.Compose transform pipeline.
    """
    transforms = [
        A.Resize(height=int(img_size * 1.143), width=int(img_size * 1.143)),
        A.CenterCrop(height=img_size, width=img_size),
    ]

    # Apply resize degradation (downsample then upsample)
    if resize_scale is not None and resize_scale < 1.0:
        small_size = max(1, int(img_size * resize_scale))
        transforms += [
            A.Resize(height=small_size, width=small_size),
            A.Resize(height=img_size, width=img_size),
        ]

    # Apply JPEG compression
    if jpeg_quality is not None and jpeg_quality < 100:
        transforms.append(
            A.ImageCompression(
                quality_range=(jpeg_quality, jpeg_quality),
                p=1.0,
            )
        )

    # Apply Gaussian blur
    if blur_sigma is not None and blur_sigma > 0:
        blur_limit = max(3, int(blur_sigma * 3) * 2 + 1)  # Kernel size ≈ 3*sigma
        transforms.append(
            A.GaussianBlur(
                blur_limit=(blur_limit, blur_limit),
                sigma_limit=(blur_sigma, blur_sigma),
                p=1.0,
            )
        )

    transforms += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def denormalize(
    tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
):
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W].
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Denormalized tensor clipped to [0, 1].
    """
    import torch

    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 4:
        mean_t = mean_t[None, :, None, None]
        std_t = std_t[None, :, None, None]
    elif tensor.ndim == 3:
        mean_t = mean_t[:, None, None]
        std_t = std_t[:, None, None]

    return (tensor * std_t + mean_t).clamp(0, 1)
