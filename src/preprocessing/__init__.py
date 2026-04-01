"""
Preprocessing subpackage for frame extraction and face detection.
"""

from .extract_frames import extract_frames
from .face_detector import FaceDetector

__all__ = ["extract_frames", "FaceDetector"]
