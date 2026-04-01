"""
DF40 (2024) dataset loader.

The DF40 dataset contains 40 distinct manipulation/generation methods,
making it a comprehensive benchmark for generalization.

Expected directory structure (two layouts supported):
    Layout 1 - Manifest CSV:
        data/df40/
        ├── manifest.csv  (columns: path, label, manipulation_type, split)
        └── images/

    Layout 2 - Directory scan:
        data/df40/
        ├── real/
        │   └── *.jpg
        └── fake/
            ├── method_001/
            ├── method_002/
            └── ...
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .base_dataset import DeepfakeDataset

logger = logging.getLogger(__name__)


class DF40Dataset(DeepfakeDataset):
    """
    DF40 2024 dataset loader with 40 manipulation types.

    Supports loading from a CSV manifest file (preferred) or by scanning
    the directory structure directly.

    Args:
        root: Root directory of the DF40 dataset.
        split: 'train', 'val', or 'test'.
        transform: Image augmentation pipeline.
        csv_manifest: Path to CSV manifest file. If None, scans directories.
        max_samples: Maximum samples to load. None = all.
        seed: Random seed.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        csv_manifest: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.csv_manifest = csv_manifest
        super().__init__(root=root, split=split, transform=transform,
                         max_samples=max_samples, seed=seed)

    def _load_samples(self) -> None:
        """Load DF40 samples from CSV manifest or directory scan."""
        # Try CSV manifest first
        if self.csv_manifest is not None:
            manifest_path = Path(self.csv_manifest)
        else:
            manifest_path = self.root / "manifest.csv"

        if manifest_path.exists():
            self._load_from_csv(manifest_path)
        else:
            logger.info(
                f"No manifest CSV found at {manifest_path}. "
                "Falling back to directory scan."
            )
            self._load_from_directory()

    def _load_from_csv(self, manifest_path: Path) -> None:
        """Load samples from a CSV manifest file."""
        try:
            df = pd.read_csv(manifest_path)
        except Exception as e:
            logger.error(f"Failed to read manifest CSV {manifest_path}: {e}")
            self._load_from_directory()
            return

        required_cols = {"path", "label"}
        if not required_cols.issubset(df.columns):
            logger.warning(
                f"Manifest CSV missing required columns {required_cols}. "
                "Falling back to directory scan."
            )
            self._load_from_directory()
            return

        # Filter by split if column exists
        if "split" in df.columns:
            df = df[df["split"] == self.split]
        else:
            # Apply our own splitting
            all_paths = df["path"].tolist()
            split_paths = set(self._split_paths(all_paths))
            df = df[df["path"].isin(split_paths)]

        samples = []
        for _, row in df.iterrows():
            path = str(self.root / row["path"]) if not Path(row["path"]).is_absolute() else row["path"]
            label = int(row["label"])
            video_id = row.get("video_id", Path(path).stem)
            manip_type = row.get("manipulation_type", "unknown" if label == 1 else "real")
            samples.append((path, label, str(video_id), str(manip_type)))

        self.samples = self._limit_samples(samples, self.max_samples)
        logger.info(f"Loaded {len(self.samples)} samples from manifest CSV.")

    def _load_from_directory(self) -> None:
        """Load samples by scanning directory structure."""
        samples = []

        real_dir = self.root / "real"
        fake_base = self.root / "fake"

        # Real samples
        real_paths = self._scan_images(real_dir)
        real_paths_str = [str(p) for p in real_paths]
        real_split = self._split_paths(real_paths_str)
        real_split = self._limit_samples(real_split, self.max_samples)

        for path in real_split:
            p = Path(path)
            video_id = p.parent.name if p.parent != real_dir else p.stem
            samples.append((path, self.LABEL_REAL, video_id, "real"))

        # Fake samples — each subdirectory is a manipulation method
        if fake_base.exists():
            manip_dirs = sorted(
                d for d in fake_base.iterdir() if d.is_dir()
            )
            if not manip_dirs:
                # No subdirectories — flat fake directory
                manip_dirs = [fake_base]

            for manip_dir in manip_dirs:
                manip_name = manip_dir.name
                fake_paths = self._scan_images(manip_dir)
                fake_paths_str = [str(p) for p in fake_paths]
                fake_split = self._split_paths(fake_paths_str)
                max_per_manip = (
                    self.max_samples // len(manip_dirs)
                    if self.max_samples is not None
                    else None
                )
                fake_split = self._limit_samples(fake_split, max_per_manip)

                for path in fake_split:
                    p = Path(path)
                    video_id = p.parent.name if p.parent != manip_dir else p.stem
                    samples.append((path, self.LABEL_FAKE, video_id, manip_name))

        self.samples = samples

        if not samples:
            logger.warning(
                f"No samples found in {self.root}. "
                "Expected structure: real/ and fake/ directories."
            )
