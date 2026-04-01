"""
FaceForensics++ dataset loader.

Supports all four manipulation types (Deepfakes, FaceSwap, Face2Face,
NeuralTextures) and both compression levels (c23 raw, c40 heavily compressed).

Expected directory structure:
    data/ff++/
    ├── real/           # Real face crops (frames)
    │   └── *.jpg
    └── fake/
        ├── Deepfakes/
        │   └── *.jpg
        ├── FaceSwap/
        │   └── *.jpg
        ├── Face2Face/
        │   └── *.jpg
        └── NeuralTextures/
            └── *.jpg
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .base_dataset import DeepfakeDataset

logger = logging.getLogger(__name__)


MANIPULATION_TYPES = ["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"]


class FaceForensicsPlusPlus(DeepfakeDataset):
    """
    FaceForensics++ dataset loader.

    Supports loading real and fake samples across multiple manipulation types
    with configurable compression level.

    Args:
        root: Root directory of the FF++ dataset.
        split: 'train', 'val', or 'test'.
        transform: Image augmentation pipeline.
        manipulations: List of manipulation types to include.
        compression: Compression level — 'c23' (high quality) or 'c40' (low quality).
        max_samples: Maximum samples per class (real/fake separately). None = all.
        seed: Random seed for split reproducibility.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        manipulations: Optional[List[str]] = None,
        compression: str = "c23",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.manipulations = manipulations if manipulations is not None else MANIPULATION_TYPES
        self.compression = compression

        # Validate manipulation types
        invalid = [m for m in self.manipulations if m not in MANIPULATION_TYPES]
        if invalid:
            raise ValueError(
                f"Invalid manipulation types: {invalid}. "
                f"Valid options: {MANIPULATION_TYPES}"
            )

        super().__init__(root=root, split=split, transform=transform,
                         max_samples=max_samples, seed=seed)

    def _load_samples(self) -> None:
        """
        Scan dataset directory and populate self.samples.

        Tries multiple directory layouts:
        1. data/ff++/real/ and data/ff++/fake/{manipulation}/
        2. data/ff++/{compression}/real/ and data/ff++/{compression}/fake/{manipulation}/
        """
        samples = []

        # Try layout 1: flat structure
        real_dir = self.root / "real"
        fake_base = self.root / "fake"

        # Try layout 2: compression-aware structure
        if not real_dir.exists():
            real_dir = self.root / self.compression / "real"
            fake_base = self.root / self.compression / "fake"

        # Load real samples
        real_paths = self._scan_images(real_dir)
        real_paths_str = [str(p) for p in real_paths]
        real_split = self._split_paths(real_paths_str)
        real_split = self._limit_samples(real_split, self.max_samples)

        for path in real_split:
            video_id = Path(path).stem.split("_frame")[0] if "_frame" in Path(path).stem else Path(path).stem
            samples.append((path, self.LABEL_REAL, video_id, "real"))

        # Load fake samples per manipulation type
        for manip in self.manipulations:
            manip_dir = fake_base / manip
            fake_paths = self._scan_images(manip_dir)
            fake_paths_str = [str(p) for p in fake_paths]
            fake_split = self._split_paths(fake_paths_str)
            max_per_manip = (
                self.max_samples // len(self.manipulations)
                if self.max_samples is not None
                else None
            )
            fake_split = self._limit_samples(fake_split, max_per_manip)

            for path in fake_split:
                video_id = Path(path).stem.split("_frame")[0] if "_frame" in Path(path).stem else Path(path).stem
                samples.append((path, self.LABEL_FAKE, video_id, manip))

        self.samples = samples

        if not samples:
            logger.warning(
                f"No samples found in {self.root}. "
                f"Expected structure: {self.root}/real/ and {self.root}/fake/{{manipulation}}/"
            )
