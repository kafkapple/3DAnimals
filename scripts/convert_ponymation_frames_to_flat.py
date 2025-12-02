#!/usr/bin/env python3
"""
Convert frame_X/ structure to flat structure for Ponymation.

Current structure (wrong):
    sequence/
    ├── frame_0/
    │   ├── rgb.png
    │   ├── mask.png
    │   └── metadata.json
    ├── frame_1/
    │   └── ...

Required structure:
    sequence/
    ├── 000000rgb.png  (or with separator)
    ├── 000000mask.png
    ├── 000000metadata.json
    └── ...

Usage:
    python scripts/convert_ponymation_frames_to_flat.py \
        --data-dir data/ponymation/mouse
"""

import argparse
import shutil
from pathlib import Path
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Convert frame_X/ to flat structure")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Ponymation data directory (e.g., data/ponymation/mouse)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    return parser.parse_args()


def convert_sequence(seq_dir: Path, dry_run: bool = False) -> int:
    """Convert a single sequence from frame_X/ to flat structure."""

    # Find all frame_X directories
    frame_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")])

    if not frame_dirs:
        return 0

    converted = 0

    for frame_dir in frame_dirs:
        # Extract frame number
        match = re.match(r"frame_(\d+)", frame_dir.name)
        if not match:
            continue

        frame_num = int(match.group(1))
        frame_prefix = f"{frame_num:06d}"

        # Move files to parent directory
        for file in frame_dir.iterdir():
            if file.is_file():
                # New filename: 000000rgb.png, 000000mask.png, etc.
                new_name = f"{frame_prefix}{file.name}"
                new_path = seq_dir / new_name

                if dry_run:
                    print(f"  Would move: {file} -> {new_path}")
                else:
                    shutil.move(str(file), str(new_path))
                converted += 1

        # Remove empty frame directory
        if not dry_run:
            try:
                frame_dir.rmdir()
            except OSError:
                pass  # Directory not empty

    return converted


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)

    print(f"Converting Ponymation data structure")
    print(f"  Data dir: {data_dir}")
    print(f"  Mode: {'dry-run' if args.dry_run else 'actual'}")
    print()

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return 1

    total_converted = 0

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        print(f"Processing {split}...")

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            count = convert_sequence(seq_dir, args.dry_run)
            if count > 0:
                print(f"  {seq_dir.name}: {count} files")
                total_converted += count

    print()
    print(f"Total: {total_converted} files {'would be' if args.dry_run else ''} converted")

    if args.dry_run:
        print()
        print("Run without --dry-run to apply changes")

    return 0


if __name__ == "__main__":
    exit(main())
