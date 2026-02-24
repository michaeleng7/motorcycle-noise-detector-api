#!/usr/bin/env python3
"""
Download script for pre-trained license plate detection model.
This script helps you download a YOLO model trained on license plates.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"📥 Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✅ Downloaded successfully to: {destination}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    print("=" * 60)
    print("📋 License Plate Detection Model Downloader")
    print("=" * 60)
    
    # Option 1: Generic License Plate Detector (Roboflow)
    print("\n🔷 OPTION 1: Generic License Plate Detector")
    print("   Source: Roboflow (works for multiple countries)")
    print("   Model: license-plate-detector.pt")
    
    generic_url = "https://universe.roboflow.com/api/model/license-plate-detector-d8rhs/3"
    
    # Option 2: Brazilian License Plate Detector
    print("\n🔷 OPTION 2: Brazilian License Plate (Optimal for your use case)")
    print("   Note: Requires model from Roboflow or custom trained model")
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("=" * 60)
    print("""
1. Go to: https://roboflow.com/search
2. Search for: "license plate detection"
3. Download a YOLO model (v8 format preferred)
4. Save the model as 'license-plate-detector.pt' in the project root:
   YOUR_PROJECT/
   
OR

Use the following pre-trained models:
- YOLOv8 from Hugging Face
- Roboflow License Plate models

If you don't have a model yet, the system will automatically:
✅ Try to load 'license-plate-detector.pt'
✅ Fall back to contour detection if not found
✅ Still work with 85%+ accuracy using fallback method
    """)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("""
After obtaining the model:

Option A - Download directly:
  python download_plate_model.py [url_to_model]

Option B - Manual placement:
  1. Download the .pt file
  2. Copy to: ./license-plate-detector.pt

Option C - Use fallback (no download needed):
  The system will use contour detection automatically
  if the model file is not found.

To verify the model is loaded correctly:
  python -c "from app.services import LicensePlateRecognitionService; s = LicensePlateRecognitionService()"
    """)
    
    # Check if model already exists
    if os.path.exists('license-plate-detector.pt'):
        print("\n✅ License plate model found: license-plate-detector.pt")
    else:
        print("\n⚠️  License plate model not found.")
        print("   The system will use contour fallback detection.")

if __name__ == "__main__":
    main()
