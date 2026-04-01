"""
Abstract base class for deepfake detection datasets.

Provides a common interface for all dataset loaders with support for
class balancing, split management, and structured sample output.
"""

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DeepfakeDataset(ABC, Dataset):
    """
    Abstract base class for deepfake detection datasets.

    All dataset implementations must inherit from this class and implement
    the `_load_samples` method to populate self.samples.

    Args:
        root: Root directory of the dataset.
        split: Dataset split — 'train', 'val', or 'test'.
        transform: Albumentations or torchvision transform to apply.
        max_samples: Maximum number of samples to load (per class if balanced).
                     None loads all available samples.
        seed: Random seed for reproducible splits.
    """

    LABEL_REAL = 0
    LABEL_FAKE = 1

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.transform = transform
        self.max_samples = max_samples
        self.seed = seed

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {self.split}")

        if not self.root.exists():
            logger.warning(f"Dataset root does not exist: {self.root}")

        # Each sample: (image_path, label, video_id, manipulation_type)
        self.samples: List[Tuple[str, int, str, str]] = []

        self._load_samples()

        if not self.samples:
            logger.warning(
                f"{self.__class__.__name__}: No samples found in {self.root} "
                f"for split='{self.split}'"
            )
        else:
            n_real = sum(1 for _, lbl, _, _ in self.samples if lbl == self.LABEL_REAL)
            n_fake = sum(1 for _, lbl, _, _ in self.samples if lbl == self.LABEL_FAKE)
            logger.info(
                f"{self.__class__.__name__} [{self.split}]: "
                f"{len(self.samples)} samples | {n_real} real | {n_fake} fake"
            )

    @abstractmethod
    def _load_samples(self) -> None:
        """
        Populate self.samples with (image_path, label, video_id, manipulation_type) tuples.

        Must be implemented by each dataset subclass.
        """
        ...

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a sample dict with image tensor and metadata.

        Returns:
            Dictionary with keys:
                - image (torch.Tensor): Transformed image [C, H, W].
                - label (int): 0 = real, 1 = fake.
                - video_id (str): Identifier of source video.
                - manipulation_type (str): Type of manipulation or 'real'.
                - path (str): File path of the source image.
        """
        img_path, label, video_id, manipulation_type = self.samples[idx]

        try:
            image = self._load_image(img_path)
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Using blank image.")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform is not None:
            try:
                augmented = self.transform(image=image)
                image = augmented["image"]
            except Exception as e:
                logger.warning(f"Transform failed for {img_path}: {e}. Using raw tensor.")
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
            "manipulation_type": manipulation_type,
            "path": img_path,
        }

    def _load_image(self, path: str) -> np.ndarray:
        """
        Load an image from disk as an RGB numpy array.

        Args:
            path: Path to the image file.

        Returns:
            Image as numpy array [H, W, 3] with dtype uint8.
        """
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler (class balancing).

        Returns:
            1-D float tensor of weights, one per sample.
        """
        labels = np.array([lbl for _, lbl, _, _ in self.samples])
        class_counts = np.bincount(labels, minlength=2)
        class_counts = np.maximum(class_counts, 1)  # Avoid divide-by-zero

        # Weight = inverse class frequency
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return torch.tensor(sample_weights, dtype=torch.float32)

    def get_labels(self) -> List[int]:
        """Return list of all labels for this split."""
        return [lbl for _, lbl, _, _ in self.samples]

    def _split_paths(
        self,
        paths: List[str],
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> List[str]:
        """
        Select paths for the current split using deterministic shuffling.

        Args:
            paths: Full list of file paths.
            split_ratios: (train, val, test) ratios that must sum to 1.0.

        Returns:
            Subset of paths for self.split.
        """
        rng = random.Random(self.seed)
        shuffled = paths.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])

        if self.split == "train":
            return shuffled[:train_end]
        elif self.split == "val":
            return shuffled[train_end:val_end]
        else:  # test
            return shuffled[val_end:]

    def _scan_images(
        self,
        directory: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> List[Path]:
        """
        Recursively find all image files in a directory.

        Args:
            directory: Root directory to scan.
            extensions: Allowed file extensions (lowercase).

        Returns:
            Sorted list of matching Path objects.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        return sorted(
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        )

    def _limit_samples(self, samples: List, max_n: Optional[int]) -> List:
        """Randomly subsample if max_n is specified."""
        if max_n is None or len(samples) <= max_n:
            return samples
        rng = random.Random(self.seed)
        return rng.sample(samples, max_n)
