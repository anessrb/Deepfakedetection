#!/usr/bin/env python3
"""
Dataset preprocessing pipeline: extract frames from videos and detect faces.

This script handles the full preprocessing pipeline for deepfake detection:
1. Extract frames from video files at a specified FPS
2. Detect and crop face regions from extracted frames
3. Save face crops in a standardized directory structure

Usage:
    python scripts/preprocess_dataset.py \\
        --input_dir data/raw/ff++ \\
        --output_dir data/ff++ \\
        --dataset ff++ \\
        --fps 1 \\
        --max_frames 300

Supported datasets: ff++, celeb_df, df40, wild_deepfake
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.extract_frames import extract_frames_from_directory
from src.preprocessing.face_detector import FaceDetector


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def preprocess_ff_plus_plus(
    input_dir: Path,
    output_dir: Path,
    fps: float,
    max_frames: int,
    face_detector: FaceDetector,
    manipulations: list,
) -> None:
    """
    Preprocess FaceForensics++ dataset.

    Expected input structure:
        input_dir/
        ├── original_sequences/   or   real/
        └── manipulated_sequences/
            ├── Deepfakes/
            ├── FaceSwap/
            └── ...
    """
    logger = logging.getLogger(__name__)

    # Process real videos
    # Try common FF++ directory layouts
    real_dirs = [
        input_dir / "original_sequences" / "youtube" / "c23" / "videos",
        input_dir / "original_sequences" / "actors" / "c23" / "videos",
        input_dir / "real",
        input_dir / "original",
    ]

    for real_video_dir in real_dirs:
        if real_video_dir.exists():
            logger.info(f"Extracting real frames from {real_video_dir}")
            frames_out = output_dir / "real_frames"
            extract_frames_from_directory(
                str(real_video_dir), str(frames_out), fps=fps, max_frames=max_frames
            )

            logger.info(f"Detecting faces in {frames_out}")
            face_out = output_dir / "real"
            face_detector.process_directory(str(frames_out), str(face_out))
            break

    # Process fake videos for each manipulation type
    for manip in manipulations:
        manip_dirs = [
            input_dir / "manipulated_sequences" / manip / "c23" / "videos",
            input_dir / "fake" / manip,
            input_dir / manip,
        ]

        for manip_dir in manip_dirs:
            if manip_dir.exists():
                logger.info(f"Extracting {manip} frames from {manip_dir}")
                frames_out = output_dir / "fake_frames" / manip
                extract_frames_from_directory(
                    str(manip_dir), str(frames_out), fps=fps, max_frames=max_frames
                )

                logger.info(f"Detecting faces in {frames_out}")
                face_out = output_dir / "fake" / manip
                face_detector.process_directory(str(frames_out), str(face_out))
                break


def preprocess_generic(
    input_dir: Path,
    output_dir: Path,
    fps: float,
    max_frames: int,
    face_detector: FaceDetector,
    dataset_name: str,
) -> None:
    """
    Generic preprocessing for datasets with real/ and fake/ directories.

    Works for Celeb-DF, DF40, WildDeepfake, and similar datasets.
    """
    logger = logging.getLogger(__name__)

    for label in ["real", "fake"]:
        label_dir = input_dir / label
        if not label_dir.exists():
            logger.warning(f"Directory not found: {label_dir}. Skipping.")
            continue

        logger.info(f"[{dataset_name}] Extracting {label} frames from {label_dir}")
        frames_out = output_dir / f"{label}_frames"
        extract_frames_from_directory(
            str(label_dir), str(frames_out), fps=fps, max_frames=max_frames
        )

        logger.info(f"[{dataset_name}] Detecting faces in {frames_out}")
        face_out = output_dir / label
        face_detector.process_directory(str(frames_out), str(face_out))

    logger.info(f"[{dataset_name}] Preprocessing complete → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess deepfake datasets: extract frames and detect faces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Raw dataset directory containing videos.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for processed face crops.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ff++", "celeb_df", "df40", "wild_deepfake"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Maximum frames per video.",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=224,
        help="Output face crop size.",
    )
    parser.add_argument(
        "--face_margin",
        type=float,
        default=0.2,
        help="Fractional margin around detected face.",
    )
    parser.add_argument(
        "--manipulations",
        nargs="+",
        default=["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"],
        help="FF++ manipulation types to process (only for --dataset ff++).",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip frame extraction, only run face detection on existing frames.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize face detector
    logger.info("Initializing face detector (MTCNN)...")
    face_detector = FaceDetector(
        image_size=args.face_size,
        margin=args.face_margin,
    )

    logger.info(
        f"Starting preprocessing: {args.dataset} | "
        f"fps={args.fps} | max_frames={args.max_frames}"
    )

    if args.dataset == "ff++":
        preprocess_ff_plus_plus(
            input_dir=input_dir,
            output_dir=output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            face_detector=face_detector,
            manipulations=args.manipulations,
        )
    else:
        preprocess_generic(
            input_dir=input_dir,
            output_dir=output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            face_detector=face_detector,
            dataset_name=args.dataset,
        )

    logger.info(f"Preprocessing complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
