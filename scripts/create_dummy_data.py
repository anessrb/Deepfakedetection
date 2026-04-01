#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
from pathlib import Path

def create_dummy_dataset(root_dir="data/ff++", num_samples=10):
    """Create a dummy dataset with random images in the expected structure."""
    root = Path(root_dir)
    
    # Structure for FF++: data/ff++/real and data/ff++/fake/Deepfakes
    real_dir = root / "real"
    fake_dir = root / "fake" / "Deepfakes"
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy data in {root}...")
    
    for i in range(num_samples):
        # Create random 224x224 image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save real
        img.save(real_dir / f"real_{i:04d}.png")
        
        # Save fake
        img.save(fake_dir / f"fake_{i:04d}.png")
        
    print(f"Successfully created {num_samples} real and {num_samples} fake dummy samples.")

if __name__ == "__main__":
    create_dummy_dataset()
