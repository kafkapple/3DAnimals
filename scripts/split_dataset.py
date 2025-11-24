#!/usr/bin/env python3
"""
Dataset Train/Val/Test Split Script
====================================

Purpose:
  - Split DANNCE dataset into train/val/test directories
  - Support sequence-based split (keep sequences together)
  - Support frame-based split (split within sequences)
  - Handle symlink cleanup and real directory creation

Usage:
  python scripts/split_dataset.py --source <source_dir> --mode sequence --ratio 0.7,0.15,0.15
  python scripts/split_dataset.py --source <source_dir> --mode frame --ratio 0.8,0.1,0.1
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
import random


def find_sequences(source_dir: Path) -> List[Path]:
    """Find all sequence directories in source."""
    sequences = [d for d in source_dir.iterdir() if d.is_dir()]
    return sorted(sequences)


def find_frames(sequence_dir: Path) -> List[str]:
    """Find all frame IDs in a sequence."""
    rgb_files = sorted(sequence_dir.glob("*_rgb.png"))
    frame_ids = [f.stem.replace("_rgb", "") for f in rgb_files]
    return frame_ids


def copy_frame(source_seq: Path, target_seq: Path, frame_id: str):
    """Copy all files for a single frame (rgb, mask, box, metadata)."""
    target_seq.mkdir(parents=True, exist_ok=True)

    extensions = ["_rgb.png", "_mask.png", "_box.txt", "_metadata.json"]
    for ext in extensions:
        src_file = source_seq / f"{frame_id}{ext}"
        if src_file.exists():
            shutil.copy2(src_file, target_seq / f"{frame_id}{ext}")


def split_by_sequence(
    source_dir: Path,
    target_base: Path,
    ratios: Tuple[float, float, float],
    seed: int = 42
) -> dict:
    """
    Split dataset by sequences (entire sequences go to train/val/test).

    Best for: Multiple sequences with sufficient data
    Pros: No data leakage between splits
    Cons: Uneven split if few sequences
    """
    random.seed(seed)

    sequences = find_sequences(source_dir)
    num_sequences = len(sequences)

    if num_sequences < 3:
        raise ValueError(f"Need at least 3 sequences for splitting, found {num_sequences}")

    # Shuffle sequences
    random.shuffle(sequences)

    # Calculate split indices
    train_ratio, val_ratio, test_ratio = ratios
    train_end = int(num_sequences * train_ratio)
    val_end = train_end + int(num_sequences * val_ratio)

    # Ensure at least 1 sequence per split
    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)
    val_end = min(num_sequences - 1, val_end)

    splits = {
        'train': sequences[:train_end],
        'val': sequences[train_end:val_end],
        'test': sequences[val_end:]
    }

    # Copy sequences to splits
    stats = {}
    for split_name, split_seqs in splits.items():
        split_dir = target_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        total_frames = 0
        for seq in split_seqs:
            target_seq = split_dir / seq.name
            shutil.copytree(seq, target_seq, dirs_exist_ok=True)
            frames = len(find_frames(target_seq))
            total_frames += frames

        stats[split_name] = {
            'sequences': len(split_seqs),
            'frames': total_frames
        }

    return stats


def split_by_frame(
    source_dir: Path,
    target_base: Path,
    ratios: Tuple[float, float, float],
    seed: int = 42
) -> dict:
    """
    Split dataset by frames (frames from same sequence can go to different splits).

    Best for: Few sequences or very large sequences
    Pros: More even split
    Cons: Potential data leakage (similar frames in train/test)
    """
    random.seed(seed)

    sequences = find_sequences(source_dir)
    train_ratio, val_ratio, test_ratio = ratios

    stats = {'train': 0, 'val': 0, 'test': 0}

    for seq_idx, seq in enumerate(sequences):
        frames = find_frames(seq)
        num_frames = len(frames)

        # Shuffle frames within sequence
        random.shuffle(frames)

        # Calculate split indices
        train_end = int(num_frames * train_ratio)
        val_end = train_end + int(num_frames * val_ratio)

        frame_splits = {
            'train': frames[:train_end],
            'val': frames[train_end:val_end],
            'test': frames[val_end:]
        }

        # Copy frames to splits
        for split_name, split_frames in frame_splits.items():
            split_dir = target_base / split_name
            target_seq = split_dir / f"{seq_idx:06d}_00000"

            for frame_id in split_frames:
                copy_frame(seq, target_seq, frame_id)

            stats[split_name] += len(split_frames)

    return stats


def cleanup_symlinks(target_base: Path):
    """Remove existing symlinks (val -> train, test -> train)."""
    for split in ['val', 'test']:
        split_path = target_base / split
        if split_path.is_symlink():
            print(f"  Removing symlink: {split_path}")
            split_path.unlink()
        elif split_path.exists() and not any(split_path.iterdir()):
            print(f"  Removing empty directory: {split_path}")
            split_path.rmdir()


def main():
    parser = argparse.ArgumentParser(description="Split DANNCE dataset into train/val/test")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing sequences (e.g., .../mouse_dannce_6view/train)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target base directory (default: parent of source)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['sequence', 'frame'],
        default='sequence',
        help="Split mode: 'sequence' (entire sequences) or 'frame' (individual frames)"
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="0.7,0.15,0.15",
        help="Train/Val/Test ratio (comma-separated, default: 0.7,0.15,0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show split plan without copying files"
    )

    args = parser.parse_args()

    # Parse arguments
    source_dir = Path(args.source).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_base = Path(args.target).resolve() if args.target else source_dir.parent

    ratios = tuple(float(x) for x in args.ratio.split(','))
    if len(ratios) != 3:
        raise ValueError("Ratio must have 3 values (train,val,test)")
    if abs(sum(ratios) - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    # Print configuration
    print("=" * 80)
    print("Dataset Split Configuration")
    print("=" * 80)
    print(f"Source:      {source_dir}")
    print(f"Target:      {target_base}")
    print(f"Mode:        {args.mode}")
    print(f"Ratio:       train={ratios[0]:.2f}, val={ratios[1]:.2f}, test={ratios[2]:.2f}")
    print(f"Seed:        {args.seed}")
    print(f"Dry run:     {args.dry_run}")
    print()

    # Analyze source
    sequences = find_sequences(source_dir)
    total_frames = sum(len(find_frames(seq)) for seq in sequences)

    print(f"Source analysis:")
    print(f"  Sequences:   {len(sequences)}")
    print(f"  Total frames: {total_frames}")
    for seq in sequences:
        frames = len(find_frames(seq))
        print(f"    {seq.name}: {frames} frames")
    print()

    if args.dry_run:
        print("DRY RUN - No files will be copied")
        return

    # Clean up existing symlinks
    print("Cleaning up existing splits...")
    cleanup_symlinks(target_base)
    print()

    # Perform split
    print(f"Splitting dataset (mode: {args.mode})...")
    if args.mode == 'sequence':
        if len(sequences) < 3:
            print(f"WARNING: Only {len(sequences)} sequences found.")
            print("Recommendation: Use --mode frame for better split with few sequences")
            response = input("Continue with sequence mode? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        stats = split_by_sequence(source_dir, target_base, ratios, args.seed)
    else:
        stats = split_by_frame(source_dir, target_base, ratios, args.seed)

    # Print results
    print()
    print("=" * 80)
    print("Split Complete!")
    print("=" * 80)

    if args.mode == 'sequence':
        for split_name in ['train', 'val', 'test']:
            info = stats[split_name]
            print(f"{split_name.upper():5s}: {info['sequences']} sequences, {info['frames']} frames")
    else:
        for split_name in ['train', 'val', 'test']:
            frames = stats[split_name]
            percentage = (frames / total_frames) * 100
            print(f"{split_name.upper():5s}: {frames} frames ({percentage:.1f}%)")

    print()
    print(f"Output directories:")
    print(f"  {target_base}/train/")
    print(f"  {target_base}/val/")
    print(f"  {target_base}/test/")
    print()
    print("✅ Dataset split successful!")


if __name__ == "__main__":
    main()
