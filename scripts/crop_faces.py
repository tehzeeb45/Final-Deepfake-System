"""
Automatic Face Cropping & Dataset Preparation Script
=====================================================
Recursively scans full-body or raw photos in input directory (e.g., Desktop/finetune_data),
detects faces using MTCNN, crops the face with natural margin, and saves them
to a cropped output folder with a ready-to-upload ZIP file for Kaggle.

Usage:
    python scripts/crop_faces.py
    python scripts/crop_faces.py --input "C:\\Users\\User\\Desktop\\finetune_data"
"""

import os
import sys
import shutil
import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False


def get_face_detector(device):
    """Initialize MTCNN face detector."""
    if HAS_MTCNN:
        return MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)
    return None


def crop_face(image_np, detector, margin=0.25, target_size=456):
    """
    Detect largest face, crop with margin, and resize.
    Returns cropped image (RGB uint8) or None if no face found.
    """
    h, w, _ = image_np.shape
    
    # Try MTCNN first
    if detector is not None:
        try:
            boxes, probs = detector.detect(image_np)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]  # MTCNN selects largest when select_largest=True
                x1, y1, x2, y2 = box
                
                # Expand by margin
                bw = x2 - x1
                bh = y2 - y1
                dx = bw * margin
                dy = bh * margin
                
                nx1 = max(0, int(round(x1 - dx)))
                ny1 = max(0, int(round(y1 - dy)))
                nx2 = min(w, int(round(x2 + dx)))
                ny2 = min(h, int(round(y2 + dy)))
                
                crop = image_np[ny1:ny2, nx1:nx2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                    return crop_resized
        except Exception:
            pass

    # Fallback to OpenCV Haar Cascade if MTCNN missed
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 0:
            # Pick largest
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            dx = fw * margin
            dy = fh * margin
            nx1 = max(0, int(round(x - dx)))
            ny1 = max(0, int(round(y - dy)))
            nx2 = min(w, int(round(x + fw + dx)))
            ny2 = min(h, int(round(y + fh + dy)))
            crop = image_np[ny1:ny2, nx1:nx2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                return crop_resized
    except Exception:
        pass

    return None


def read_image_rgb(path: Path):
    """Safely read image handling unicode paths."""
    try:
        # PIL handles paths well
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            return np.array(rgb)
    except Exception:
        return None


def main():
    default_input = Path(r"C:\Users\User\Desktop\finetune_data")
    default_output = Path(r"C:\Users\User\Desktop\finetune_data_cropped")
    
    parser = argparse.ArgumentParser(description="Crop faces for DeepFake Fine-Tuning dataset")
    parser.add_argument("--input", "-i", type=str, default=str(default_input),
                        help="Input dataset folder containing full photos")
    parser.add_argument("--output", "-o", type=str, default=str(default_output),
                        help="Output folder where cropped faces will be saved")
    parser.add_argument("--margin", "-m", type=float, default=0.25,
                        help="Margin around face (default 0.25 = 25 percent)")
    parser.add_argument("--target-size", "-s", type=int, default=456,
                        help="Target image size (default 456 for EfficientNet-B5)")
    parser.add_argument("--no-zip", action="store_true",
                        help="Do not create a zip file at the end")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"\n❌ ERROR: Input folder not found at:\n   {input_path}")
        print("Please check the folder path and try again.\n")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  🚀 DeepFake Face Crop & Dataset Preparation Tool")
    print("=" * 60)
    print(f"  Input folder:  {input_path}")
    print(f"  Output folder: {output_path}")
    print(f"  Target size:   {args.target_size}x{args.target_size}")
    print(f"  Device:        {device}")
    print("-" * 60)
    
    detector = get_face_detector(device)
    if detector is None:
        print("  ⚠️ MTCNN not available, using OpenCV Haar Cascade.")
    else:
        print("  ✓ MTCNN face detector ready")
        
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts]
    
    if not all_files:
        print(f"\n⚠️ No images found in {input_path}!")
        return

    print(f"\nFound {len(all_files)} images to process...\n")
    
    success_count = 0
    fallback_count = 0
    fail_count = 0
    
    for idx, file_path in enumerate(all_files, 1):
        rel_path = file_path.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix(".jpg")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        img_np = read_image_rgb(file_path)
        if img_np is None:
            print(f"[{idx}/{len(all_files)}] ❌ Could not read: {rel_path}")
            fail_count += 1
            continue
            
        cropped = crop_face(img_np, detector, margin=args.margin, target_size=args.target_size)
        
        if cropped is not None:
            # Save cropped face
            cv2.imwrite(str(out_file), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[{idx}/{len(all_files)}] ✓ Cropped face: {rel_path}")
            success_count += 1
        else:
            # Fallback: if no face found, resize center square crop so image is not lost
            h, w, _ = img_np.shape
            min_dim = min(h, w)
            sy = (h - min_dim) // 2
            sx = (w - min_dim) // 2
            square = img_np[sy:sy + min_dim, sx:sx + min_dim]
            square_resized = cv2.resize(square, (args.target_size, args.target_size), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(out_file), cv2.cvtColor(square_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[{idx}/{len(all_files)}] ⚠️ No face detected, used center-crop: {rel_path}")
            fallback_count += 1

    print("\n" + "=" * 60)
    print("  Processing Complete!")
    print(f"  ✓ Faces detected & cropped: {success_count}")
    print(f"  ⚠️ Center-crop fallbacks:   {fallback_count}")
    print(f"  ❌ Failed:                  {fail_count}")
    print(f"  📁 Output saved at:         {output_path}")
    print("=" * 60)
    
    # Create ZIP file for Kaggle upload
    if not args.no_zip:
        zip_path = output_path.parent / "finetune-data.zip"
        print(f"\n📦 Creating Kaggle upload zip: {zip_path} ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_path.rglob("*"):
                if f.is_file():
                    arcname = f.relative_to(output_path)
                    zf.write(f, arcname)
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"  ✓ Ready! Zip file size: {size_mb:.1f} MB")
        print(f"  👉 Kaggle par yeh upload karein: {zip_path}\n")


if __name__ == "__main__":
    main()
