import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference import load_model, predict_image
from src.models import DeepfakeDetector
from src.preprocessing.face_detector import FaceDetector
from src.datasets.augmentations import get_val_transforms
from src.visualization.gradcam import VitGradCAM

app = FastAPI(title="DeepFake Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for frontend and Grad-CAM results
STATIC_DIR = Path("app/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/", StaticFiles(directory="app", html=True), name="app")

# Global state
MODEL_PATH = "outputs_test/detector_calibrated.pth"  # Default to test model for now
CONFIG_PATH = "configs/default.yaml"

model = None
temperature = None
face_detector = None
transform = None
device = None

@app.on_event("startup")
async def startup_event():
    global model, temperature, face_detector, transform, device
    
    # Select device (force CPU for stability on Mac)
    device = torch.device("cpu")
    
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Load model
    if os.path.exists(MODEL_PATH):
        model, temperature = load_model(MODEL_PATH, config, device)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}. API will fail until model is provided.")
    
    # Initialize helpers
    face_detector = FaceDetector(device=device)
    transform = get_val_transforms(img_size=config.get("augmentation", {}).get("img_size", 224))

@app.get("/")
async def root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Determine file type
        suffix = Path(file.filename).suffix.lower()
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        
        # Save uploaded file temporarily
        temp_input = STATIC_DIR / f"input_{uuid.uuid4().hex}{suffix}"
        with open(temp_input, "wb") as f:
            f.write(await file.read())
            
        if suffix in image_extensions:
            # Process Image
            raw_image = np.array(Image.open(temp_input).convert("RGB"))
            face_crop = face_detector.detect_and_crop(raw_image)
            
            if face_crop is None:
                h, w, _ = raw_image.shape
                min_dim = min(h, w)
                start_h = (h - min_dim) // 2
                start_w = (w - min_dim) // 2
                face_crop = raw_image[start_h:start_h+min_dim, start_w:start_w+min_dim]
                face_crop = cv2.resize(face_crop, (224, 224))
            
            prob, img_tensor = predict_image(face_crop, model, transform, device, temperature)
            decision = "FAKE" if prob >= 0.5 else "REAL"
            
            # Grad-CAM
            gradcam_url = None
            try:
                gradcam = VitGradCAM(model)
                heatmap = gradcam.generate(img_tensor)
                overlay_pil = gradcam.overlay(face_crop, heatmap, alpha=0.5)
                gradcam.remove_hooks()
                
                filename = f"gradcam_{uuid.uuid4().hex}.jpg"
                save_path = STATIC_DIR / filename
                overlay_pil.save(str(save_path), "JPEG", quality=90)
                gradcam_url = f"/static/{filename}"
            except Exception as e:
                print(f"Grad-CAM error: {e}")
                
            return {
                "type": "image",
                "filename": file.filename,
                "probability": float(prob),
                "decision": decision,
                "gradcam_url": gradcam_url,
                "face_detected": True
            }
            
        elif suffix in video_extensions:
            # Process Video
            from scripts.inference import process_video
            
            result = process_video(
                input_path=str(temp_input),
                model=model,
                face_detector=face_detector,
                transform=transform,
                device=device,
                temperature=temperature,
                output_dir=STATIC_DIR,
                fps=1.0,
                max_frames=30,
                threshold=0.5
            )
            
            # The process_video function returns a dict with 'output_path' (the plot)
            # We need to convert the absolute path to a URL
            plot_url = f"/static/{Path(result['output_path']).name}"
            
            return {
                "type": "video",
                "filename": file.filename,
                "probability": float(result["probability"]),
                "decision": result["decision"],
                "frame_probs": result["frame_probs"],
                "plot_url": plot_url,
                "n_frames": result["n_frames"]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
            
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp input if needed (optional, keeping for now for debugging)
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
