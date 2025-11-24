#!/usr/bin/env python3
"""
Generate Missing Files for Fauna Dataset

This script generates missing box.txt and metadata.json files
when you already have *_rgb.png and *_mask.png files.

Usage:
    python scripts/generate_missing_files.py \
        --data_dir data/fauna/Fauna_dataset/large_scale/mouse/train/seq_000
"""

import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate missing box.txt and metadata.json files")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing *_rgb.png and *_mask.png files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing box.txt and metadata.json files"
    )
    return parser.parse_args()


def compute_bbox_from_mask(mask):
    """Compute bounding box from binary mask"""
    coords = np.where(mask > 0)

    if len(coords[0]) == 0:
        # Return None if empty mask
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add 5% padding
    pad_y = int((y_max - y_min) * 0.05)
    pad_x = int((x_max - x_min) * 0.05)
    y_min = max(0, y_min - pad_y)
    y_max = min(mask.shape[0], y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(mask.shape[1], x_max + pad_x)

    return x_min, y_min, x_max, y_max


def create_metadata_and_box(frame_id, bbox, img_width, img_height, output_dir, prefix):
    """Create metadata.json and box.txt files"""
    x_min, y_min, x_max, y_max = bbox

    # Calculate width and height
    crop_w = x_max - x_min
    crop_h = y_max - y_min

    # Metadata JSON (4 required fields + 2 optional)
    metadata = {
        "video_frame_id": frame_id,
        "crop_box_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "video_frame_width": img_width,
        "video_frame_height": img_height,
        "crop_height": 256,  # Optional: standard crop size
        "crop_width": 256,   # Optional: standard crop size
    }

    meta_file = output_dir / f"{prefix}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Box.txt (9 values: frame_id x y width height full_w full_h sharpness label)
    box_line = f"{frame_id:07d} {x_min:.2f} {y_min:.2f} {crop_w:.2f} {crop_h:.2f} {img_width:.2f} {img_height:.2f} 0.00 0\n"

    box_file = output_dir / f"{prefix}_box.txt"
    with open(box_file, "w") as f:
        f.write(box_line)

    return meta_file, box_file


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    print("="*80)
    print("Generate Missing Files for Fauna Dataset")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Force overwrite: {args.force}")

    # Find all RGB files
    rgb_files = sorted(data_dir.glob("*_rgb.png"))

    if len(rgb_files) == 0:
        print(f"\nError: No *_rgb.png files found in {data_dir}")
        return

    print(f"\nFound {len(rgb_files)} RGB images")

    # Process each frame
    created_count = 0
    skipped_count = 0
    error_count = 0

    for rgb_file in tqdm(rgb_files, desc="Processing"):
        # Extract frame prefix (e.g., "0000000" from "0000000_rgb.png")
        prefix = rgb_file.stem.replace("_rgb", "")

        # Parse frame_id from prefix
        try:
            frame_id = int(prefix)
        except ValueError:
            print(f"\nWarning: Cannot parse frame_id from {prefix}, using 0")
            frame_id = 0

        # Check if mask exists
        mask_file = data_dir / f"{prefix}_mask.png"
        if not mask_file.exists():
            print(f"\nError: Mask not found for {prefix}")
            error_count += 1
            continue

        # Check if box.txt and metadata.json already exist
        box_file = data_dir / f"{prefix}_box.txt"
        meta_file = data_dir / f"{prefix}_metadata.json"

        if not args.force and box_file.exists() and meta_file.exists():
            skipped_count += 1
            continue

        # Load RGB and mask
        rgb = cv2.imread(str(rgb_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if rgb is None:
            print(f"\nError: Cannot load RGB: {rgb_file}")
            error_count += 1
            continue

        if mask is None:
            print(f"\nError: Cannot load mask: {mask_file}")
            error_count += 1
            continue

        # Get image dimensions
        img_h, img_w = rgb.shape[:2]

        # Compute bounding box from mask
        bbox = compute_bbox_from_mask(mask)

        if bbox is None:
            print(f"\nWarning: Empty mask for {prefix}, skipping")
            error_count += 1
            continue

        # Create metadata and box.txt
        try:
            meta_created, box_created = create_metadata_and_box(
                frame_id=frame_id,
                bbox=bbox,
                img_width=img_w,
                img_height=img_h,
                output_dir=data_dir,
                prefix=prefix
            )
            created_count += 1
        except Exception as e:
            print(f"\nError creating files for {prefix}: {e}")
            error_count += 1
            continue

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total RGB files: {len(rgb_files)}")
    print(f"Created: {created_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Errors: {error_count}")

    if error_count == 0 and created_count > 0:
        print("\n✅ All missing files generated successfully!")
    elif error_count > 0:
        print(f"\n⚠️  Completed with {error_count} errors")
    else:
        print("\n✅ No files needed to be created (all already exist)")

    print("\nNext steps:")
    print(f"  1. Verify files: ls -la {data_dir}/ | head -20")
    print(f"  2. Check one frame:")
    print(f"     cat {data_dir}/0000000_box.txt")
    print(f"     cat {data_dir}/0000000_metadata.json")


if __name__ == "__main__":
    main()
