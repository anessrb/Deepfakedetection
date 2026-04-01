"""
Datasets subpackage for deepfake detection.

Provides dataset loaders for FaceForensics++, Celeb-DF v2, DF40,
and WildDeepfake datasets, plus augmentation pipelines.
"""

from .base_dataset import DeepfakeDataset
from .ff_plus_plus import FaceForensicsPlusPlus
from .celeb_df import CelebDFv2
from .df40 import DF40Dataset
from .wild_deepfake import WildDeepfake
from .augmentations import get_train_transforms, get_val_transforms, get_robustness_transforms

__all__ = [
    "DeepfakeDataset",
    "FaceForensicsPlusPlus",
    "CelebDFv2",
    "DF40Dataset",
    "WildDeepfake",
    "get_train_transforms",
    "get_val_transforms",
    "get_robustness_transforms",
]
