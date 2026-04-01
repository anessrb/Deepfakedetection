"""
Frame extraction from video files using OpenCV.

Supports extracting frames at a specified FPS from videos, with optional
limiting of total frames extracted per video. Includes CLI support.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int = 300,
    quality: int = 95,
) -> List[str]:
    """
    Extract frames from a video file at a given FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames will be saved.
        fps: Target frames per second to extract. Defaults to 1.0.
        max_frames: Maximum number of frames to extract. Defaults to 300.
        quality: JPEG quality for saved frames (0-100). Defaults to 95.

    Returns:
        List of paths to extracted frame files.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        logger.warning(
            f"Invalid FPS ({video_fps}) detected for {video_path}. Defaulting to 25."
        )
        video_fps = 25.0

    # Calculate frame interval (how many source frames to skip between extractions)
    frame_interval = max(1, int(round(video_fps / fps)))

    logger.debug(
        f"Video: {video_path.name} | Source FPS: {video_fps:.1f} | "
        f"Total frames: {total_frames} | Extraction interval: {frame_interval}"
    )

    saved_paths: List[str] = []
    frame_idx = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = output_dir / f"frame_{saved_count:05d}.jpg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success = cv2.imwrite(str(frame_filename), frame, encode_params)
            if success:
                saved_paths.append(str(frame_filename))
                saved_count += 1
            else:
                logger.warning(f"Failed to save frame {frame_idx} from {video_path}")

        frame_idx += 1

    cap.release()
    logger.info(
        f"Extracted {saved_count} frames from {video_path.name} → {output_dir}"
    )
    return saved_paths


def extract_frames_from_directory(
    input_dir: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int = 300,
    video_extensions: Optional[List[str]] = None,
    quality: int = 95,
) -> dict:
    """
    Extract frames from all videos in a directory.

    Args:
        input_dir: Directory containing video files.
        output_dir: Root output directory; frames are saved in subdirectories
                    named after each video file (without extension).
        fps: Target frames per second. Defaults to 1.0.
        max_frames: Maximum frames per video. Defaults to 300.
        video_extensions: List of video file extensions to process.
        quality: JPEG quality. Defaults to 95.

    Returns:
        Dictionary mapping video filename → list of saved frame paths.
    """
    if video_extensions is None:
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    video_files = [
        f
        for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return {}

    logger.info(f"Found {len(video_files)} video files in {input_dir}")
    results = {}

    for video_file in video_files:
        # Preserve relative directory structure under output_dir
        relative_path = video_file.relative_to(input_dir)
        video_output_dir = output_dir / relative_path.parent / relative_path.stem

        try:
            frame_paths = extract_frames(
                video_path=str(video_file),
                output_dir=str(video_output_dir),
                fps=fps,
                max_frames=max_frames,
                quality=quality,
            )
            results[str(relative_path)] = frame_paths
        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
            results[str(relative_path)] = []

    return results


def main() -> None:
    """CLI entry point for frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a video file or directory containing videos.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for extracted frames.",
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
        help="Maximum number of frames to extract per video.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames (0-100).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)

    if input_path.is_file():
        paths = extract_frames(
            video_path=str(input_path),
            output_dir=args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            quality=args.quality,
        )
        print(f"Extracted {len(paths)} frames.")
    elif input_path.is_dir():
        results = extract_frames_from_directory(
            input_dir=str(input_path),
            output_dir=args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            quality=args.quality,
        )
        total = sum(len(v) for v in results.values())
        print(f"Extracted {total} frames from {len(results)} videos.")
    else:
        parser.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
