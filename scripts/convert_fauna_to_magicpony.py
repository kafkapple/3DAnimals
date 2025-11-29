#!/usr/bin/env python3
"""
Convert SAM3D/Fauna format dataset to MagicPony format.

SAM3D/Fauna format:
    sequence/
    ├── 0000001_rgb.png
    ├── 0000001_mask.png
    ├── 0000001_metadata.json
    └── 0000001_feat16.png (optional)

MagicPony format:
    frame_000/
    ├── rgb.png
    ├── mask.png
    ├── metadata.json
    └── feat16.png

Usage:
    python scripts/convert_fauna_to_magicpony.py \
        --source data/fauna/large_scale/mouse \
        --target data/magicpony/mouse \
        --copy  # or --symlink
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SAM3D/Fauna to MagicPony format")
    parser.add_argument("--source", type=str, required=True,
                        help="Source Fauna dataset directory (with train/val/test splits)")
    parser.add_argument("--target", type=str, required=True,
                        help="Target MagicPony dataset directory")
    parser.add_argument("--copy", action="store_true", default=False,
                        help="Copy files instead of symlink")
    parser.add_argument("--extract-dino", action="store_true", default=False,
                        help="Extract DINO features if not present")
    return parser.parse_args()


def convert_split(source_split: Path, target_split: Path, use_copy: bool = False):
    """Convert a single split (train/val/test)."""
    if not source_split.exists():
        print(f"  [SKIP] {source_split} does not exist")
        return 0

    target_split.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    # Iterate through sequences
    for seq_dir in sorted(source_split.iterdir()):
        if not seq_dir.is_dir():
            continue

        # Find all RGB files in this sequence
        rgb_files = sorted(list(seq_dir.glob("*_rgb.png")) + list(seq_dir.glob("*_rgb.jpg")))

        for rgb_file in rgb_files:
            # Extract frame ID from filename (e.g., "0000001_rgb.png" -> "0000001")
            stem = rgb_file.stem.replace("_rgb", "")

            # Create target frame directory
            # Use global frame naming: seq_name + frame_id
            frame_name = f"{seq_dir.name}_{stem}"
            frame_dir = target_split / frame_name
            frame_dir.mkdir(parents=True, exist_ok=True)

            # Find corresponding files
            mask_file = seq_dir / f"{stem}_mask.png"
            metadata_file = seq_dir / f"{stem}_metadata.json"

            # Determine target paths
            target_rgb = frame_dir / f"rgb{rgb_file.suffix}"
            target_mask = frame_dir / "mask.png"
            target_metadata = frame_dir / "metadata.json"

            # Copy or symlink RGB
            if use_copy:
                shutil.copy2(rgb_file, target_rgb)
            else:
                if target_rgb.exists():
                    target_rgb.unlink()
                target_rgb.symlink_to(rgb_file.resolve())

            # Copy or symlink mask
            if mask_file.exists():
                if use_copy:
                    shutil.copy2(mask_file, target_mask)
                else:
                    if target_mask.exists():
                        target_mask.unlink()
                    target_mask.symlink_to(mask_file.resolve())

            # Handle metadata (need to ensure correct format)
            if metadata_file.exists():
                # Read and validate/fix metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Ensure required fields exist
                if "video_frame_id" not in metadata:
                    metadata["video_frame_id"] = int(stem)

                # Write to target
                with open(target_metadata, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                # Create minimal metadata if missing
                print(f"  [WARN] Missing metadata for {rgb_file}, creating minimal version")
                # Read image to get dimensions
                from PIL import Image
                img = Image.open(rgb_file)
                w, h = img.size

                metadata = {
                    "video_frame_id": int(stem) if stem.isdigit() else frame_count,
                    "video_frame_width": w,
                    "video_frame_height": h,
                    "crop_box_xyxy": [0, 0, w, h]
                }
                with open(target_metadata, 'w') as f:
                    json.dump(metadata, f, indent=2)

            # Handle DINO features (feat16.png)
            feat_file = seq_dir / f"{stem}_feat16.png"
            target_feat = frame_dir / "feat16.png"
            if feat_file.exists():
                if use_copy:
                    shutil.copy2(feat_file, target_feat)
                else:
                    if target_feat.exists():
                        target_feat.unlink()
                    target_feat.symlink_to(feat_file.resolve())

            frame_count += 1

    return frame_count


def main():
    args = parse_args()

    source = Path(args.source)
    target = Path(args.target)

    print(f"Converting Fauna → MagicPony format")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Mode: {'copy' if args.copy else 'symlink'}")
    print()

    if not source.exists():
        print(f"ERROR: Source directory does not exist: {source}")
        return 1

    target.mkdir(parents=True, exist_ok=True)

    total_frames = 0

    for split in ["train", "val", "test"]:
        source_split = source / split
        target_split = target / split

        print(f"Processing {split}...")
        count = convert_split(source_split, target_split, args.copy)
        print(f"  Converted {count} frames")
        total_frames += count

    print()
    print(f"✓ Conversion complete: {total_frames} total frames")
    print(f"  Output: {target}")

    return 0


if __name__ == "__main__":
    exit(main())
