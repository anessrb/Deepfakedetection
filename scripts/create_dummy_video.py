#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

def create_dummy_video(output_path="data/test_video.mp4", duration_sec=3, fps=10):
    """Create a dummy video with random noise and a moving circle."""
    width, height = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration_sec * fps
    
    print(f"Creating dummy video: {output_path} ({num_frames} frames)...")
    
    for i in range(num_frames):
        # Create random noise background
        frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        
        # Add a moving "face" (circle)
        center_x = int(width / 2 + 50 * np.sin(2 * np.pi * i / num_frames))
        center_y = int(height / 2 + 50 * np.cos(2 * np.pi * i / num_frames))
        cv2.circle(frame, (center_x, center_y), 40, (200, 200, 200), -1)
        
        out.write(frame)
        
    out.release()
    print("Successfully created dummy video.")

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    create_dummy_video()
