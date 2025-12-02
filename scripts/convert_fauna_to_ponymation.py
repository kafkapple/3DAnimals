#!/usr/bin/env python3
"""
Convert Fauna format dataset to Ponymation format.

Fauna format:
    sequence/
    ├── 0000027_rgb.png
    ├── 0000027_mask.png
    ├── 0000027_metadata.json
    └── ...

Ponymation format (requires temporal sequences):
    sequence_000/
    ├── frame_0/
    │   ├── rgb.png
    │   ├── mask.png
    │   └── metadata.json
    ├── frame_1/
    │   └── ...
    └── frame_9/
        └── ...

Usage:
    python scripts/convert_fauna_to_ponymation.py \
        --source data/fauna_mouse/large_scale/mouse_dannce_6view \
        --target data/ponymation/mouse \
        --num-frames 10 \
        --copy
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Fauna to Ponymation format")
    parser.add_argument("--source", type=str, required=True,
                        help="Source Fauna dataset directory")
    parser.add_argument("--target", type=str, required=True,
                        help="Target Ponymation dataset directory")
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Number of frames per sequence (default: 10)")
    parser.add_argument("--copy", action="store_true", default=False,
                        help="Copy files instead of symlink")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Minimum frames to create a sequence (default: 5)")
    return parser.parse_args()


def get_frames_in_sequence(seq_dir: Path) -> List[Tuple[int, Path]]:
    """Get all frames in a sequence directory, sorted by frame ID."""
    frames = []
    for rgb_file in seq_dir.glob("*_rgb.png"):
        stem = rgb_file.stem.replace("_rgb", "")
        try:
            frame_id = int(stem)
            frames.append((frame_id, rgb_file))
        except ValueError:
            continue
    return sorted(frames, key=lambda x: x[0])


def convert_sequence(
    seq_dir: Path,
    target_dir: Path,
    seq_name: str,
    num_frames: int,
    use_copy: bool,
    min_frames: int
) -> int:
    """
    Convert a single Fauna sequence to Ponymation format.

    Creates multiple Ponymation sequences if there are enough frames.
    """
    frames = get_frames_in_sequence(seq_dir)

    if len(frames) < min_frames:
        print(f"  [SKIP] {seq_dir.name}: only {len(frames)} frames (need >= {min_frames})")
        return 0

    sequences_created = 0

    # Create sequences with sliding window or chunks
    for chunk_idx in range(0, len(frames), num_frames):
        chunk = frames[chunk_idx:chunk_idx + num_frames]

        if len(chunk) < min_frames:
            break

        # Create Ponymation sequence directory
        pony_seq_name = f"{seq_name}_{chunk_idx:04d}"
        pony_seq_dir = target_dir / pony_seq_name
        pony_seq_dir.mkdir(parents=True, exist_ok=True)

        # Convert each frame
        for frame_idx, (frame_id, rgb_file) in enumerate(chunk):
            frame_dir = pony_seq_dir / f"frame_{frame_idx}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            stem = rgb_file.stem.replace("_rgb", "")

            # Source files
            mask_file = seq_dir / f"{stem}_mask.png"
            metadata_file = seq_dir / f"{stem}_metadata.json"
            feat_file = seq_dir / f"{stem}_feat16.png"

            # Target files
            target_rgb = frame_dir / f"rgb{rgb_file.suffix}"
            target_mask = frame_dir / "mask.png"
            target_metadata = frame_dir / "metadata.json"
            target_feat = frame_dir / "feat16.png"

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

            # Handle metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if "video_frame_id" not in metadata:
                    metadata["video_frame_id"] = frame_id
                with open(target_metadata, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                # Create minimal metadata
                from PIL import Image
                img = Image.open(rgb_file)
                w, h = img.size
                metadata = {
                    "video_frame_id": frame_id,
                    "video_frame_width": w,
                    "video_frame_height": h,
                    "crop_box_xyxy": [0, 0, w, h]
                }
                with open(target_metadata, 'w') as f:
                    json.dump(metadata, f, indent=2)

            # Handle DINO features
            if feat_file.exists():
                if use_copy:
                    shutil.copy2(feat_file, target_feat)
                else:
                    if target_feat.exists():
                        target_feat.unlink()
                    target_feat.symlink_to(feat_file.resolve())

        sequences_created += 1

    return sequences_created


def convert_split(source_split: Path, target_split: Path, num_frames: int, use_copy: bool, min_frames: int) -> int:
    """Convert a single split (train/val/test)."""
    if not source_split.exists():
        print(f"  [SKIP] {source_split} does not exist")
        return 0

    target_split.mkdir(parents=True, exist_ok=True)

    total_sequences = 0

    # Iterate through source sequences
    for seq_dir in sorted(source_split.iterdir()):
        if not seq_dir.is_dir():
            continue

        count = convert_sequence(
            seq_dir,
            target_split,
            seq_dir.name,
            num_frames,
            use_copy,
            min_frames
        )
        total_sequences += count
        if count > 0:
            print(f"  {seq_dir.name}: {count} sequence(s)")

    return total_sequences


def main():
    args = parse_args()

    source = Path(args.source)
    target = Path(args.target)

    print(f"Converting Fauna → Ponymation format")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Mode: {'copy' if args.copy else 'symlink'}")
    print(f"  Frames per sequence: {args.num_frames}")
    print(f"  Min frames: {args.min_frames}")
    print()

    if not source.exists():
        print(f"ERROR: Source directory does not exist: {source}")
        return 1

    target.mkdir(parents=True, exist_ok=True)

    total_sequences = 0

    for split in ["train", "val", "test"]:
        source_split = source / split
        target_split = target / split

        print(f"Processing {split}...")
        count = convert_split(source_split, target_split, args.num_frames, args.copy, args.min_frames)
        print(f"  Total: {count} sequences")
        total_sequences += count

    print()
    print(f"Conversion complete: {total_sequences} total sequences")
    print(f"  Output: {target}")

    # Warning for Ponymation requirements
    if total_sequences > 0:
        print()
        print("Note: Ponymation requires:")
        print(f"  - At least {args.num_frames} frames per sequence for optimal training")
        print("  - Temporal consistency (consecutive frames from same video)")
        print("  - DINO features (feat16.png) for best results")

    return 0


if __name__ == "__main__":
    exit(main())
