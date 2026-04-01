#!/usr/bin/env python3
"""
Dataset Download Helper for DeepFake Detection.

This script provides instructions and automation for downloading the
required datasets for training the full model.
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def instructions_ff_plus_plus():
    print_header("FaceForensics++ (FF++)")
    print("Status: RESTRICTED (Registration Required)")
    print("1. Visit: https://github.com/ondyari/FaceForensics")
    print("2. Fill out the Google Form linked in the README.")
    print("3. Once approved, you will receive a download script via email.")
    print("4. Run the script to download the 'c23' (compressed) videos.")
    print("5. Place them in: data/raw/ff++/")

def instructions_celeb_df():
    print_header("Celeb-DF v2")
    print("Status: RESTRICTED (Registration Required)")
    print("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("2. Fill out the Google Form linked in the README.")
    print("3. Once approved, you will receive a download link.")
    print("4. Place the downloaded files in: data/raw/celeb_df/")

def instructions_df40():
    print_header("DF40")
    print("Status: PUBLIC (Direct Links)")
    print("1. Visit: https://github.com/YZY-stack/DF40")
    print("2. Use the Google Drive or Baidu links provided in the README.")
    print("3. Download the 'Pre-processed Data' for the fastest setup.")
    print("4. Place it in: data/df40/")

def download_wild_deepfake():
    print_header("WildDeepfake")
    print("Status: PUBLIC (Hugging Face)")
    print("You can download this dataset using the Hugging Face CLI or library.")
    print("\nTo download via Python:")
    print("  pip install huggingface_hub")
    print("  huggingface-cli download xingjunm/WildDeepfake --repo-type dataset --local-dir data/wild_deepfake")
    
    confirm = input("\nWould you like to try downloading a small subset of WildDeepfake now? (y/n): ")
    if confirm.lower() == 'y':
        try:
            from huggingface_hub import snapshot_download
            print("Downloading WildDeepfake subset...")
            snapshot_download(
                repo_id="xingjunm/WildDeepfake",
                repo_type="dataset",
                local_dir="data/wild_deepfake",
                allow_patterns=["*.jpg", "*.png", "metadata.csv"], # Adjust patterns as needed
                max_workers=4
            )
            print("Download complete!")
        except ImportError:
            print("Error: huggingface_hub not installed. Run 'pip install huggingface_hub' first.")
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    print("DeepFake Detection — Dataset Sourcing Helper")
    print("This script will guide you through obtaining the datasets needed for full training.")
    
    instructions_ff_plus_plus()
    instructions_celeb_df()
    instructions_df40()
    download_wild_deepfake()

if __name__ == "__main__":
    main()
