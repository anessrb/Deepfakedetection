"""
WildDeepfake dataset loader.

WildDeepfake is collected from the internet, capturing real-world
deepfakes with diverse quality and compression levels.

Expected directory structure:
    data/wild_deepfake/
    ├── real/
    │   └── *.jpg  (or nested)
    └── fake/
        └── *.jpg  (or nested)
"""

import logging
from pathlib import Path
from typing import Optional

from .base_dataset import DeepfakeDataset

logger = logging.getLogger(__name__)


class WildDeepfake(DeepfakeDataset):
    """
    WildDeepfake dataset loader.

    Contains deepfakes collected from the wild (internet), representing
    diverse real-world manipulation quality and styles.

    Args:
        root: Root directory of the WildDeepfake dataset.
        split: 'train', 'val', or 'test'.
        transform: Image augmentation pipeline.
        max_samples: Maximum samples per class. None = all.
        seed: Random seed.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(root=root, split=split, transform=transform,
                         max_samples=max_samples, seed=seed)

    def _load_samples(self) -> None:
        """Scan WildDeepfake directory structure and populate self.samples."""
        samples = []

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"

        # Load real samples
        real_paths = self._scan_images(real_dir)
        real_paths_str = [str(p) for p in real_paths]
        real_split = self._split_paths(real_paths_str)
        real_split = self._limit_samples(real_split, self.max_samples)

        for path in real_split:
            p = Path(path)
            video_id = p.parent.name if p.parent.name not in ("real", "fake") else p.stem
            samples.append((path, self.LABEL_REAL, video_id, "real"))

        # Load fake samples
        fake_paths = self._scan_images(fake_dir)
        fake_paths_str = [str(p) for p in fake_paths]
        fake_split = self._split_paths(fake_paths_str)
        fake_split = self._limit_samples(fake_split, self.max_samples)

        for path in fake_split:
            p = Path(path)
            video_id = p.parent.name if p.parent.name not in ("real", "fake") else p.stem
            samples.append((path, self.LABEL_FAKE, video_id, "wild_fake"))

        self.samples = samples

        if not samples:
            logger.warning(
                f"No samples found in {self.root}. "
                f"Expected: {self.root}/real/ and {self.root}/fake/"
            )
