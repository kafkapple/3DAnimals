#!/usr/bin/env python3
"""
Fix metadata.json format for Fauna compatibility

Reads box.txt and generates proper Fauna-format metadata.json
"""
import json
from pathlib import Path
import argparse


def fix_metadata_from_box(box_path: Path, metadata_path: Path) -> dict:
    """Generate Fauna-compatible metadata from box.txt"""

    # Read box.txt
    with open(box_path, 'r') as f:
        line = f.read().strip()
        parts = line.split()

        if len(parts) != 9:
            raise ValueError(f"Invalid box.txt format: {box_path}")

        frame_id_str, x0, y0, w, h, full_w, full_h, sharpness, label = parts

    # Convert to integers/floats
    frame_id = int(frame_id_str)
    x0, y0 = int(x0), int(y0)
    w, h = int(w), int(h)
    full_w, full_h = int(full_w), int(full_h)
    sharpness = float(sharpness)
    label = int(label)

    # Create Fauna-format metadata
    metadata = {
        "video_frame_id": frame_id,
        "crop_box_xyxy": [x0, y0, x0 + w, y0 + h],
        "video_frame_width": full_w,
        "video_frame_height": full_h,
        "sharpness": sharpness,
        "crop_height": h,
        "crop_width": w,
        "label": label
    }

    # Write metadata.json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Fix metadata.json format for Fauna')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Root directory containing train/val/test splits')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without modifying files')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"❌ Error: {data_dir} does not exist")
        return 1

    print("=" * 80)
    print("Fixing metadata.json format for Fauna compatibility")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print()

    # Find all box.txt files
    box_files = list(data_dir.rglob("*_box.txt"))

    if not box_files:
        print("❌ No box.txt files found!")
        return 1

    print(f"Found {len(box_files)} box.txt files")
    print()

    fixed_count = 0
    error_count = 0

    for box_path in sorted(box_files):
        # Get corresponding metadata.json path
        metadata_path = box_path.parent / box_path.name.replace('_box.txt', '_metadata.json')

        if not metadata_path.exists():
            print(f"⚠️  Warning: {metadata_path} does not exist, skipping")
            continue

        try:
            if args.dry_run:
                print(f"[DRY RUN] Would fix: {metadata_path.relative_to(data_dir)}")
            else:
                metadata = fix_metadata_from_box(box_path, metadata_path)
                fixed_count += 1

                if fixed_count % 50 == 0:
                    print(f"  Progress: {fixed_count}/{len(box_files)} files fixed...")

        except Exception as e:
            print(f"❌ Error processing {box_path}: {e}")
            error_count += 1

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if args.dry_run:
        print(f"[DRY RUN] Would fix {len(box_files)} files")
    else:
        print(f"✅ Fixed: {fixed_count} files")
        if error_count > 0:
            print(f"❌ Errors: {error_count} files")

    print()

    # Show sample of fixed metadata
    if not args.dry_run and fixed_count > 0:
        sample_box = box_files[0]
        sample_meta = sample_box.parent / sample_box.name.replace('_box.txt', '_metadata.json')

        print("Sample fixed metadata:")
        print(f"  File: {sample_meta.relative_to(data_dir)}")
        print()
        with open(sample_meta, 'r') as f:
            content = f.read()
            print("  " + content.replace("\n", "\n  "))

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
