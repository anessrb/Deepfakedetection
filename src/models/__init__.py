"""
Models subpackage for deepfake detection.

Provides spatial branch (DINOv2), frequency branch (LightCNN),
fusion detector, and temperature scaling calibration.
"""

from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch
from .detector import DeepfakeDetector
from .calibration import TemperatureScaling

__all__ = [
    "SpatialBranch",
    "FrequencyBranch",
    "DeepfakeDetector",
    "TemperatureScaling",
]
