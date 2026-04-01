#!/usr/bin/env python3
"""
Single image/video inference with Grad-CAM visualization.

For images: detects face, runs model, generates Grad-CAM overlay, saves result.
For videos: processes all frames, generates temporal score plot, aggregated decision.

Usage:
    # Single image
    python scripts/inference.py \\
        --input path/to/face.jpg \\
        --checkpoint outputs/detector_calibrated.pth \\
        --output_dir outputs/inference/

    # Video file
    python scripts/inference.py \\
        --input path/to/video.mp4 \\
        --checkpoint outputs/detector_calibrated.pth \\
        --output_dir outputs/inference/ \\
        --fps 2
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import DeepfakeDetector, TemperatureScaling
from src.preprocessing.extract_frames import extract_frames
from src.preprocessing.face_detector import FaceDetector
from src.datasets.augmentations import get_val_transforms, denormalize
from src.visualization.gradcam import VitGradCAM


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[DeepfakeDetector, Optional[float]]:
    """Load model and optional temperature from checkpoint."""
    logger = logging.getLogger(__name__)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = config.get("model", {})

    model = DeepfakeDetector(
        spatial_model_name=model_cfg.get("backbone", "vit_base_patch14_dinov2.lvd142m"),
        spatial_pretrained=False,
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 4),
        spatial_embed_dim=model_cfg.get("spatial_embed_dim", 768),
        freq_embed_dim=model_cfg.get("freq_embed_dim", 256),
        fusion_hidden_dims=model_cfg.get("fusion_hidden_dims", [512, 128]),
        fusion_dropout_rates=model_cfg.get("fusion_dropout", [0.3, 0.2]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    temperature = checkpoint.get("temperature", None)
    if temperature:
        logger.info(f"Using temperature scaling: T={temperature:.4f}")

    return model, temperature


def predict_image(
    image_array: np.ndarray,
    model: DeepfakeDetector,
    transform,
    device: torch.device,
    temperature: Optional[float] = None,
) -> Tuple[float, torch.Tensor]:
    """
    Run deepfake detection on a single face image.

    Args:
        image_array: RGB image [H, W, 3] uint8.
        model: Detection model.
        transform: Val transforms.
        device: Device.
        temperature: Optional temperature for calibration.

    Returns:
        Tuple of (probability, image_tensor).
    """
    augmented = transform(image=image_array)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(img_tensor)
        if temperature is not None:
            logit = logit / temperature
        prob = torch.sigmoid(logit).item()

    return prob, img_tensor


def process_image(
    input_path: str,
    model: DeepfakeDetector,
    face_detector: FaceDetector,
    transform,
    device: torch.device,
    temperature: Optional[float],
    output_dir: Path,
    threshold: float = 0.5,
    save_gradcam: bool = True,
    gradcam_alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Process a single image file.

    Detects face, runs inference, generates Grad-CAM, saves visualization.

    Returns:
        Result dict with probability, decision, and output path.
    """
    logger = logging.getLogger(__name__)
    input_path = Path(input_path)

    # Load and detect face
    logger.info(f"Processing image: {input_path.name}")
    raw_image = np.array(Image.open(input_path).convert("RGB"))
    face_crop = face_detector.detect_and_crop(raw_image)

    if face_crop is None:
        logger.warning("No face detected. Using center crop.")
        face_crop = raw_image

    # Run inference
    prob, img_tensor = predict_image(face_crop, model, transform, device, temperature)
    decision = "FAKE" if prob >= threshold else "REAL"

    logger.info(f"  Result: {decision} (p={prob:.4f})")

    # Generate Grad-CAM visualization
    if save_gradcam:
        try:
            gradcam = VitGradCAM(model)
            heatmap = gradcam.generate(img_tensor)
            overlay_pil = gradcam.overlay(face_crop, heatmap, alpha=gradcam_alpha)
            gradcam.remove_hooks()
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}. Skipping heatmap.")
            overlay_pil = Image.fromarray(face_crop)

        # Create visualization figure
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig)

        # Original face
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(face_crop)
        ax1.set_title("Detected Face", fontsize=12, fontweight="bold")
        ax1.axis("off")

        # Grad-CAM overlay
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(np.array(overlay_pil))
        ax2.set_title("Grad-CAM Attention", fontsize=12, fontweight="bold")
        ax2.axis("off")

        # Score gauge
        ax3 = fig.add_subplot(gs[2])
        color = "#F44336" if prob >= threshold else "#4CAF50"
        ax3.barh(["Fake probability"], [prob], color=color, height=0.5)
        ax3.barh(["Fake probability"], [1 - prob], left=[prob], color="#E0E0E0", height=0.5)
        ax3.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold ({threshold:.2f})")
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probability", fontsize=11)
        ax3.set_title(
            f"Decision: {decision}\nP(fake) = {prob:.4f}",
            fontsize=12, fontweight="bold", color=color
        )
        ax3.legend(fontsize=9)
        ax3.grid(True, axis="x", alpha=0.3)

        plt.suptitle(
            f"Deepfake Detection — {input_path.name}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        out_path = output_dir / f"{input_path.stem}_result.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Visualization saved: {out_path}")

    return {
        "input": str(input_path),
        "probability": prob,
        "decision": decision,
        "face_detected": True,
        "output_path": str(out_path) if save_gradcam else None,
    }


def process_video(
    input_path: str,
    model: DeepfakeDetector,
    face_detector: FaceDetector,
    transform,
    device: torch.device,
    temperature: Optional[float],
    output_dir: Path,
    fps: float = 1.0,
    max_frames: int = 300,
    threshold: float = 0.5,
    aggregate: str = "mean",
    save_gradcam: bool = True,
) -> Dict[str, Any]:
    """
    Process a video file, running inference on sampled frames.

    Returns:
        Result dict with per-frame probabilities and aggregated decision.
    """
    logger = logging.getLogger(__name__)
    input_path = Path(input_path)

    logger.info(f"Processing video: {input_path.name}")

    # Extract frames to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = extract_frames(
            video_path=str(input_path),
            output_dir=tmpdir,
            fps=fps,
            max_frames=max_frames,
        )

        if not frame_paths:
            logger.error("No frames extracted from video.")
            return {"error": "No frames extracted"}

        logger.info(f"Extracted {len(frame_paths)} frames")

        frame_probs = []
        frame_images = []  # For visualization

        for i, frame_path in enumerate(frame_paths):
            raw = np.array(Image.open(frame_path).convert("RGB"))
            face_crop = face_detector.detect_and_crop(raw)
            if face_crop is None:
                face_crop = raw

            prob, img_tensor = predict_image(face_crop, model, transform, device, temperature)
            frame_probs.append(prob)

            if save_gradcam and i < 4:  # Save first few frames
                frame_images.append((face_crop, img_tensor, prob))

    # Aggregate decision
    probs_arr = np.array(frame_probs)

    if aggregate == "mean":
        agg_prob = float(probs_arr.mean())
    elif aggregate == "max":
        agg_prob = float(probs_arr.max())
    elif aggregate == "voting":
        votes = (probs_arr >= threshold).mean()
        agg_prob = float(votes)
    else:
        agg_prob = float(probs_arr.mean())

    decision = "FAKE" if agg_prob >= threshold else "REAL"
    logger.info(
        f"Video result: {decision} | "
        f"agg_prob={agg_prob:.4f} | "
        f"frames={len(frame_probs)}"
    )

    # Generate temporal plot
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_plots = 1 + (1 if frame_images else 0)
    fig = plt.figure(figsize=(14, 5 * n_plots))
    gs = gridspec.GridSpec(n_plots, 1, figure=fig, hspace=0.4)

    # Temporal probability plot
    ax = fig.add_subplot(gs[0])
    frame_indices = np.arange(len(frame_probs))
    color = "#F44336" if agg_prob >= threshold else "#4CAF50"

    ax.fill_between(frame_indices, frame_probs, alpha=0.3, color=color)
    ax.plot(frame_indices, frame_probs, "o-", color=color, linewidth=2, markersize=5)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold:.2f})")
    ax.axhline(agg_prob, color=color, linestyle="-.", linewidth=2,
               label=f"Mean prob ({agg_prob:.3f})")
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("P(fake)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Per-Frame Fake Probability — {decision} (p={agg_prob:.4f})",
        fontsize=13, fontweight="bold", color=color,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Deepfake Detection — {input_path.name}",
        fontsize=14, fontweight="bold",
    )

    out_path = output_dir / f"{input_path.stem}_result.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Video result plot saved: {out_path}")

    return {
        "input": str(input_path),
        "probability": agg_prob,
        "decision": decision,
        "frame_probs": frame_probs,
        "n_frames": len(frame_probs),
        "output_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deepfake detection inference on image or video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input image or video path.")
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint path (.pth)."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/inference/",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="FPS for video frame extraction.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Max frames to process from video.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "max", "voting"],
        default="mean",
        help="Video frame aggregation method.",
    )
    parser.add_argument(
        "--no_gradcam",
        action="store_true",
        help="Skip Grad-CAM visualization.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (cpu, cuda, mps). If None, auto-selects.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Config & Model ────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    config = load_config(str(config_path)) if config_path.exists() else {}

    device = torch.device(args.device) if args.device else auto_device()
    logger.info(f"Device: {device}")

    logger.info(f"Loading model from: {args.checkpoint}")
    model, temperature = load_model(args.checkpoint, config, device)

    # ── Initialize helpers ──────────────────────────────────────────────
    face_detector = FaceDetector(device=device)
    transform = get_val_transforms(
        img_size=config.get("augmentation", {}).get("img_size", 224)
    )
    inf_cfg = config.get("inference", {})
    threshold = args.threshold or inf_cfg.get("threshold", 0.5)

    # ── Determine input type ──────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if input_path.suffix.lower() in image_extensions:
        result = process_image(
            input_path=str(input_path),
            model=model,
            face_detector=face_detector,
            transform=transform,
            device=device,
            temperature=temperature,
            output_dir=output_dir,
            threshold=threshold,
            save_gradcam=not args.no_gradcam,
            gradcam_alpha=inf_cfg.get("gradcam_alpha", 0.5),
        )
    elif input_path.suffix.lower() in video_extensions:
        result = process_video(
            input_path=str(input_path),
            model=model,
            face_detector=face_detector,
            transform=transform,
            device=device,
            temperature=temperature,
            output_dir=output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            threshold=threshold,
            aggregate=args.aggregate,
            save_gradcam=not args.no_gradcam,
        )
    else:
        logger.error(f"Unsupported file type: {input_path.suffix}")
        sys.exit(1)

    # ── Print final result ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  INPUT:    {result['input']}")
    print(f"  DECISION: {result['decision']}")
    print(f"  P(fake):  {result['probability']:.4f}")
    if "n_frames" in result:
        print(f"  FRAMES:   {result['n_frames']}")
    if result.get("output_path"):
        print(f"  OUTPUT:   {result['output_path']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
