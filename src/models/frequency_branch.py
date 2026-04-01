"""
Frequency branch combining FFT and DCT analysis with LightCNN.

Deepfakes often leave artifacts in the frequency domain that are invisible
to the naked eye but detectable via FFT/DCT analysis. This branch converts
images to their frequency representations and processes them with a lightweight
convolutional network.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def rgb_to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image tensor to grayscale using luminance weights.

    Args:
        x: Input tensor [B, 3, H, W] in range [0, 1] (after normalization, denorm first).

    Returns:
        Grayscale tensor [B, 1, H, W].
    """
    # ITU-R BT.601 luminance weights
    weights = torch.tensor([0.299, 0.587, 0.114],
                            device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def compute_fft_spectrum(gray: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-magnitude FFT spectrum of a grayscale image.

    Shifts the zero-frequency component to the center of the spectrum.

    Args:
        gray: Grayscale tensor [B, 1, H, W].

    Returns:
        Log-magnitude FFT spectrum [B, 1, H, W], normalized to [0, 1].
    """
    # Real FFT on 2D
    fft = torch.fft.rfft2(gray, norm="ortho")
    fft_full = torch.fft.fft2(gray, norm="ortho")  # Full spectrum for shift
    magnitude = torch.abs(fft_full)
    log_mag = torch.log1p(magnitude)  # log(1 + |F|) for numerical stability

    # Shift zero-frequency to center
    log_mag = torch.roll(log_mag, shifts=(log_mag.shape[2] // 2, log_mag.shape[3] // 2),
                         dims=(2, 3))

    # Normalize to [0, 1]
    b, c, h, w = log_mag.shape
    log_mag_flat = log_mag.view(b, c, -1)
    min_v = log_mag_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    max_v = log_mag_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    log_mag = (log_mag - min_v) / (max_v - min_v + 1e-8)

    return log_mag


def compute_dct(gray: torch.Tensor) -> torch.Tensor:
    """
    Compute the DCT-II of a grayscale image using FFT-based implementation.

    Based on the relationship: DCT-II can be computed using FFT on a
    symmetrically extended signal.

    Args:
        gray: Grayscale tensor [B, 1, H, W].

    Returns:
        DCT-II coefficients [B, 1, H, W], normalized to [0, 1].
    """
    B, C, H, W = gray.shape

    # DCT-II via FFT: extend signal symmetrically
    # For rows
    x = gray.squeeze(1)  # [B, H, W]

    # Apply DCT along rows
    x_flipped_h = torch.flip(x, dims=[1])
    x_ext_h = torch.cat([x, x_flipped_h], dim=1)  # [B, 2H, W]
    fft_h = torch.fft.rfft(x_ext_h, dim=1)
    k_h = torch.arange(H, device=gray.device, dtype=gray.dtype)
    phase_h = torch.exp(-1j * torch.tensor(torch.pi, dtype=gray.dtype) * k_h / (2 * H))
    # Use only real part after phase shift
    dct_h = (fft_h[:, :H, :] * phase_h.view(1, H, 1)).real

    # Apply DCT along columns
    dct_h_flipped = torch.flip(dct_h, dims=[2])
    dct_ext_w = torch.cat([dct_h, dct_h_flipped], dim=2)  # [B, H, 2W]
    fft_w = torch.fft.rfft(dct_ext_w, dim=2)
    k_w = torch.arange(W, device=gray.device, dtype=gray.dtype)
    phase_w = torch.exp(-1j * torch.tensor(torch.pi, dtype=gray.dtype) * k_w / (2 * W))
    dct_2d = (fft_w[:, :, :W] * phase_w.view(1, 1, W)).real  # [B, H, W]

    dct_2d = dct_2d.unsqueeze(1)  # [B, 1, H, W]

    # Normalize to [0, 1]
    b, c, h, w = dct_2d.shape
    flat = dct_2d.view(b, c, -1)
    min_v = flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    max_v = flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    dct_2d = (dct_2d - min_v) / (max_v - min_v + 1e-8)

    return dct_2d


class ConvBlock(nn.Module):
    """Single conv block: Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightCNN(nn.Module):
    """
    Lightweight CNN for frequency feature extraction.

    4 conv blocks followed by adaptive pooling and a projection head.

    Args:
        in_channels: Number of input channels (2 for FFT+DCT stack).
        conv_channels: Number of output channels per conv block.
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 2,
        conv_channels: Optional[List[int]] = None,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]

        # Build 4 conv blocks
        channels = [in_channels] + conv_channels
        blocks = []
        for i in range(len(conv_channels)):
            blocks.append(ConvBlock(channels[i], channels[i + 1]))
        self.conv_blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(conv_channels[-1] * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x


class FrequencyBranch(nn.Module):
    """
    Frequency analysis branch combining FFT and DCT features.

    Converts input images to frequency domain representations and
    processes them with a lightweight CNN to detect frequency-domain
    artifacts characteristic of deepfakes.

    Pipeline:
        1. RGB → grayscale
        2. 2D FFT → log-magnitude spectrum
        3. DCT-II → frequency coefficients
        4. Stack FFT + DCT → 2-channel input
        5. LightCNN → 256-d embedding

    Args:
        embed_dim: Output embedding dimension. Defaults to 256.
        conv_channels: Channel sizes for LightCNN blocks.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        conv_channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]

        self.lightcnn = LightCNN(
            in_channels=2,
            conv_channels=conv_channels,
            embed_dim=embed_dim,
        )

    def _extract_frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2-channel FFT+DCT frequency map from RGB input.

        Args:
            x: Normalized RGB tensor [B, 3, H, W]. Note: normalization shifts
               pixel values; we work with the tensor directly as a proxy for
               grayscale frequency content.

        Returns:
            Two-channel frequency tensor [B, 2, H, W].
        """
        # Convert to grayscale
        gray = rgb_to_grayscale(x)  # [B, 1, H, W]

        # Compute FFT spectrum
        fft_spectrum = compute_fft_spectrum(gray)  # [B, 1, H, W]

        # Compute DCT
        dct_coeffs = compute_dct(gray)  # [B, 1, H, W]

        # Stack as 2-channel input
        freq_input = torch.cat([fft_spectrum, dct_coeffs], dim=1)  # [B, 2, H, W]
        return freq_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-domain features from input images.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            Frequency feature embedding [B, embed_dim] (256-d by default).
        """
        freq_input = self._extract_frequency_features(x)
        features = self.lightcnn(freq_input)
        return features

    def __repr__(self) -> str:
        return f"FrequencyBranch(embed_dim={self.embed_dim})"
