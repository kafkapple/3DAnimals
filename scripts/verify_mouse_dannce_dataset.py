#!/usr/bin/env python3
"""
Verify Mouse DANNCE Dataset

This script verifies that the mouse_dannce_6view dataset is properly
formatted and ready for training.

Usage:
    python scripts/verify_mouse_dannce_dataset.py
"""

import os
import json
from pathlib import Path
import cv2


def verify_mouse_dannce_dataset():
    """Verify mouse_dannce_6view dataset"""

    print("=" * 80)
    print("Mouse DANNCE Dataset Verification")
    print("=" * 80)
    print()

    # Dataset path
    data_dir = Path("data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view")
    train_dir = data_dir / "train"

    if not data_dir.exists():
        print("❌ ERROR: Dataset symlink not found!")
        print(f"   Expected: {data_dir}")
        print()
        print("   Create symlink with:")
        print("   ln -sfn /home/joon/dev/data/mouse_dannce_6view \\")
        print("     data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view")
        return False

    print(f"✅ Dataset directory found: {data_dir}")
    print()

    # Check train directory
    if not train_dir.exists():
        print(f"❌ ERROR: Train directory not found: {train_dir}")
        return False

    print(f"✅ Train directory found: {train_dir}")
    print()

    # Find all sequences
    sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    print(f"📁 Found {len(sequences)} sequences:")
    for seq in sequences:
        print(f"   - {seq.name}")
    print()

    # Check each sequence
    total_frames = 0
    issues = []

    for seq_idx, seq_dir in enumerate(sequences):
        print(f"Checking sequence {seq_idx + 1}/{len(sequences)}: {seq_dir.name}")

        # Find all RGB images
        rgb_files = sorted(seq_dir.glob("*_rgb.png"))

        if len(rgb_files) == 0:
            issues.append(f"No RGB images in {seq_dir.name}")
            print(f"   ❌ No RGB images found!")
            continue

        print(f"   Found {len(rgb_files)} frames")
        total_frames += len(rgb_files)

        # Check first frame in detail
        frame = rgb_files[0]
        frame_id = frame.stem.replace("_rgb", "")

        # Required files
        mask_file = seq_dir / f"{frame_id}_mask.png"
        box_file = seq_dir / f"{frame_id}_box.txt"
        meta_file = seq_dir / f"{frame_id}_metadata.json"

        # Check all frames have required files
        missing_count = 0
        for rgb_file in rgb_files:
            fid = rgb_file.stem.replace("_rgb", "")

            if not (seq_dir / f"{fid}_mask.png").exists():
                missing_count += 1
                issues.append(f"Missing mask: {seq_dir.name}/{fid}_mask.png")

            if not (seq_dir / f"{fid}_box.txt").exists():
                missing_count += 1
                issues.append(f"Missing box: {seq_dir.name}/{fid}_box.txt")

            if not (seq_dir / f"{fid}_metadata.json").exists():
                missing_count += 1
                issues.append(f"Missing metadata: {seq_dir.name}/{fid}_metadata.json")

        if missing_count > 0:
            print(f"   ❌ {missing_count} missing files")
        else:
            print(f"   ✅ All required files present")

        # Check image can be loaded
        try:
            img = cv2.imread(str(frame))
            if img is None:
                issues.append(f"Cannot load RGB: {frame}")
                print(f"   ❌ Cannot load RGB image")
            else:
                h, w = img.shape[:2]
                print(f"   ✅ RGB image size: {w}×{h}")

                if (w, h) != (256, 256):
                    issues.append(f"RGB size not 256×256: {frame} ({w}×{h})")
                    print(f"   ⚠️  Expected 256×256, got {w}×{h}")
        except Exception as e:
            issues.append(f"Error loading RGB: {frame} - {e}")
            print(f"   ❌ Error loading RGB: {e}")

        # Check mask
        try:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues.append(f"Cannot load mask: {mask_file}")
                print(f"   ❌ Cannot load mask")
            else:
                # Check mask has foreground pixels
                if mask.max() == 0:
                    issues.append(f"Empty mask: {mask_file}")
                    print(f"   ❌ Mask is empty (all black)")
                else:
                    fg_pixels = (mask > 0).sum()
                    total_pixels = mask.size
                    fg_ratio = fg_pixels / total_pixels
                    print(f"   ✅ Mask foreground: {fg_ratio*100:.1f}%")
        except Exception as e:
            issues.append(f"Error loading mask: {mask_file} - {e}")
            print(f"   ❌ Error loading mask: {e}")

        # Check metadata
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            required_keys = ["video_frame_id", "video_frame_width", "video_frame_height", "crop_box_xyxy"]
            missing_keys = [k for k in required_keys if k not in meta]

            if missing_keys:
                issues.append(f"Missing metadata keys: {missing_keys} in {meta_file}")
                print(f"   ❌ Missing metadata keys: {missing_keys}")
            else:
                print(f"   ✅ Metadata valid")
        except Exception as e:
            issues.append(f"Error loading metadata: {meta_file} - {e}")
            print(f"   ❌ Error loading metadata: {e}")

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"Total sequences: {len(sequences)}")
    print(f"Total frames: {total_frames}")
    print()

    if len(issues) == 0:
        print("✅ ✅ ✅  ALL CHECKS PASSED!  ✅ ✅ ✅")
        print()
        print("Dataset is ready for training!")
        print()
        print("Next steps:")
        print("  1. Run debug training:")
        print("     python run.py --config-name train_fauna_mouse_dannce_debug")
        print()
        print("  2. If debug succeeds, run full training:")
        print("     python run.py --config-name train_fauna_mouse_dannce")
        print()
        return True
    else:
        print(f"❌ Found {len(issues)} issues:")
        print()
        for i, issue in enumerate(issues[:20], 1):  # Show first 20
            print(f"  {i}. {issue}")

        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")

        print()
        print("Please fix these issues before training.")
        return False


if __name__ == "__main__":
    import sys

    # Change to project root if running from scripts/
    if Path.cwd().name == "scripts":
        os.chdir("..")

    success = verify_mouse_dannce_dataset()
    sys.exit(0 if success else 1)
