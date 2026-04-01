"""
Face detection and cropping pipeline using OpenCV's DNN face detector.

Provides reliable face detection with fallback to center crop when no face
is detected. Supports single image and batch directory processing.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using OpenCV's DNN-based face detector.

    Detects faces in images and returns cropped, resized face regions.
    Falls back to center crop if no face is detected.

    Args:
        device: Ignored (kept for API compatibility). Detection runs on CPU.
        image_size: Size of output face crops (square). Defaults to 224.
        margin: Fractional margin to add around detected bounding box.
        confidence_threshold: Minimum detection confidence (0-1).
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        image_size: int = 224,
        margin: float = 0.2,
        confidence_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        self.image_size = image_size
        self.margin = margin
        self.confidence_threshold = confidence_threshold

        # Use OpenCV's built-in Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from: {cascade_path}"
            )

        logger.info(f"FaceDetector initialized (OpenCV Haar Cascade)")

    def detect_and_crop(
        self,
        image: Union[np.ndarray, Image.Image, str],
        size: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Detect the largest face in an image and return a cropped region.

        Args:
            image: Input image as numpy array (BGR or RGB), PIL Image, or file path.
            size: Output size override. Uses self.image_size if None.

        Returns:
            Cropped face as numpy array (RGB, uint8) of shape (size, size, 3),
            or center crop if no face detected. Returns None on critical errors.
        """
        output_size = size if size is not None else self.image_size

        # Normalize to RGB PIL Image
        pil_image = self._to_pil(image)
        if pil_image is None:
            return None

        img_w, img_h = pil_image.size

        # Convert to grayscale for detection
        img_array = np.array(pil_image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except Exception as e:
            logger.warning(f"Face detection failed: {e}. Using center crop.")
            return self._center_crop(pil_image, output_size)

        if len(faces) == 0:
            logger.debug("No face detected. Falling back to center crop.")
            return self._center_crop(pil_image, output_size)

        # Select the largest face (by area)
        areas = [w * h for (x, y, w, h) in faces]
        best_idx = int(np.argmax(areas))
        x, y, w, h = faces[best_idx]

        # Apply margin
        mx = int(w * self.margin)
        my = int(h * self.margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img_w, x + w + mx)
        y2 = min(img_h, y + h + my)

        # Crop and resize
        face = pil_image.crop((x1, y1, x2, y2))
        face = face.resize((output_size, output_size), Image.BILINEAR)
        return np.array(face)

    def detect_and_crop_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        size: Optional[int] = None,
    ) -> List[Optional[np.ndarray]]:
        """
        Process a batch of images for face detection.

        Args:
            images: List of images (numpy arrays or PIL Images).
            size: Output size. Uses self.image_size if None.

        Returns:
            List of cropped face arrays (or center crops on failure).
        """
        return [self.detect_and_crop(img, size=size) for img in images]

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        size: Optional[int] = None,
        image_extensions: Optional[List[str]] = None,
        quality: int = 95,
        skip_existing: bool = True,
    ) -> dict:
        """
        Process all images in a directory, detecting and saving face crops.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory to save face-cropped images.
            size: Output face crop size. Uses self.image_size if None.
            image_extensions: Image file extensions to process.
            quality: JPEG save quality (0-100).
            skip_existing: Skip files that already exist in output_dir.

        Returns:
            Dictionary mapping relative input path -> output path (or None on fail).
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            f
            for f in input_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return {}

        logger.info(f"Processing {len(image_files)} images from {input_dir}")
        results = {}

        for img_file in image_files:
            relative = img_file.relative_to(input_dir)
            out_path = output_dir / relative.parent / (relative.stem + ".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if skip_existing and out_path.exists():
                results[str(relative)] = str(out_path)
                continue

            try:
                image = Image.open(img_file).convert("RGB")
                face_crop = self.detect_and_crop(image, size=size)

                if face_crop is not None:
                    face_pil = Image.fromarray(face_crop)
                    face_pil.save(str(out_path), "JPEG", quality=quality)
                    results[str(relative)] = str(out_path)
                else:
                    logger.warning(f"Failed to process {img_file}")
                    results[str(relative)] = None
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                results[str(relative)] = None

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(
            f"Processed {successful}/{len(image_files)} images successfully."
        )
        return results

    def _to_pil(self, image: Union[np.ndarray, Image.Image, str]) -> Optional[Image.Image]:
        """Convert various input formats to RGB PIL Image."""
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                logger.error(f"Cannot open image file {image}: {e}")
                return None
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            return Image.fromarray(image_rgb.astype(np.uint8))
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return None

    def _center_crop(self, image: Image.Image, size: int) -> np.ndarray:
        """Return a center-cropped and resized version of the image."""
        w, h = image.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        cropped = image.crop((left, top, right, bottom))
        resized = cropped.resize((size, size), Image.BILINEAR)
        return np.array(resized)

    def __repr__(self) -> str:
        return (
            f"FaceDetector(image_size={self.image_size}, "
            f"margin={self.margin})"
        )


def main() -> None:
    """CLI entry point for face detection on a directory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and crop faces from images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, help="Input directory with images.")
    parser.add_argument("--output_dir", required=True, help="Output directory for face crops.")
    parser.add_argument("--size", type=int, default=224, help="Output face crop size.")
    parser.add_argument(
        "--margin", type=float, default=0.2, help="Fractional margin around face."
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-process files even if output exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    detector = FaceDetector(image_size=args.size, margin=args.margin)
    results = detector.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size,
        skip_existing=not args.no_skip_existing,
    )
    successful = sum(1 for v in results.values() if v is not None)
    print(f"Processed {successful}/{len(results)} images.")


if __name__ == "__main__":
    main()
