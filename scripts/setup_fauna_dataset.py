#!/usr/bin/env python3
"""
Setup Fauna dataset from existing rgb/mask images.

This script creates a Fauna-compatible dataset structure from an existing dataset
with rgb and mask images. It can either:
1. Create symlinks (fast, no disk usage)
2. Copy files (for cross-filesystem)
3. Preprocess (resize/crop if needed)

Usage:
    python scripts/setup_fauna_dataset.py \
        --source_dir /path/to/source/dataset \
        --output_dir data/fauna/mouse_large \
        --train_ratio 0.8 \
        --val_ratio 0.1

Supports multiple source formats:
- video_folders: video000/frame_0000_rgb.png (from sam3d_gui)
- flat: 0000_rgb.png, 0000_mask.png
- nested: seq001/0000_rgb.png
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import random
from PIL import Image
import numpy as np


def detect_source_format(source_dir: Path) -> str:
    """Detect the format of source dataset."""
    items = list(source_dir.iterdir())

    # Check for video folders (sam3d_gui format)
    if any(item.name.startswith('video') and item.is_dir() for item in items):
        return 'video_folders'

    # Check for nested sequence folders
    subdirs = [item for item in items if item.is_dir()]
    if subdirs:
        first_subdir = subdirs[0]
        if list(first_subdir.glob('*_rgb.png')) or list(first_subdir.glob('*rgb*.png')):
            return 'nested'

    # Check for flat structure
    if list(source_dir.glob('*_rgb.png')) or list(source_dir.glob('*rgb*.png')):
        return 'flat'

    raise ValueError(f"Cannot detect source format in {source_dir}")


def find_image_pairs(source_dir: Path, source_format: str) -> List[Tuple[Path, Path, str]]:
    """Find all (rgb, mask, sequence_id) pairs in source directory."""
    pairs = []

    if source_format == 'video_folders':
        # sam3d_gui format: video000/frame_0000_rgb.png
        for video_dir in sorted(source_dir.glob('video*')):
            if not video_dir.is_dir():
                continue
            for rgb_file in sorted(video_dir.glob('*_rgb.png')):
                mask_file = rgb_file.parent / rgb_file.name.replace('_rgb.png', '_mask.png')
                if mask_file.exists():
                    seq_id = video_dir.name
                    pairs.append((rgb_file, mask_file, seq_id))

    elif source_format == 'nested':
        # Nested: seq001/0000_rgb.png
        for seq_dir in sorted(source_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            for rgb_file in sorted(seq_dir.glob('*_rgb.png')):
                mask_file = rgb_file.parent / rgb_file.name.replace('_rgb.png', '_mask.png')
                if mask_file.exists():
                    seq_id = seq_dir.name
                    pairs.append((rgb_file, mask_file, seq_id))

    elif source_format == 'flat':
        # Flat: 0000_rgb.png
        for rgb_file in sorted(source_dir.glob('*_rgb.png')):
            mask_file = rgb_file.parent / rgb_file.name.replace('_rgb.png', '_mask.png')
            if mask_file.exists():
                seq_id = 'default'
                pairs.append((rgb_file, mask_file, seq_id))

    return pairs


def compute_bbox_from_mask(mask_path: Path, padding_ratio: float = 0.1) -> Tuple[int, int, int, int, int, int]:
    """Compute bounding box from mask image.

    Returns: (x0, y0, x1, y1, full_w, full_h)
    """
    mask = np.array(Image.open(mask_path).convert('L'))
    h, w = mask.shape

    # Find non-zero pixels
    rows = np.any(mask > 128, axis=1)
    cols = np.any(mask > 128, axis=0)

    if not rows.any() or not cols.any():
        # No foreground found, use full image
        return 0, 0, w, h, w, h

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    # Add padding
    pad_w = int((x1 - x0) * padding_ratio)
    pad_h = int((y1 - y0) * padding_ratio)

    x0 = max(0, x0 - pad_w)
    y0 = max(0, y0 - pad_h)
    x1 = min(w, x1 + pad_w)
    y1 = min(h, y1 + pad_h)

    return x0, y0, x1, y1, w, h


def create_fauna_files(
    rgb_path: Path,
    mask_path: Path,
    output_dir: Path,
    frame_id: int,
    use_symlink: bool = True,
    target_size: Optional[int] = None
) -> None:
    """Create Fauna-format files for a single image pair."""

    # File naming: {frame_id:07d}_rgb.png, etc.
    base_name = f"{frame_id:07d}"

    rgb_out = output_dir / f"{base_name}_rgb.png"
    mask_out = output_dir / f"{base_name}_mask.png"
    box_out = output_dir / f"{base_name}_box.txt"
    meta_out = output_dir / f"{base_name}_metadata.json"

    # Get bbox from mask
    x0, y0, x1, y1, full_w, full_h = compute_bbox_from_mask(mask_path)

    if target_size and target_size != full_w:
        # Need to resize - copy and resize
        rgb_img = Image.open(rgb_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')

        # Crop to bbox and resize
        rgb_crop = rgb_img.crop((x0, y0, x1, y1))
        mask_crop = mask_img.crop((x0, y0, x1, y1))

        # Resize to square
        rgb_crop = rgb_crop.resize((target_size, target_size), Image.BILINEAR)
        mask_crop = mask_crop.resize((target_size, target_size), Image.NEAREST)

        rgb_crop.save(rgb_out)
        mask_crop.save(mask_out)

        # Update bbox for cropped/resized image
        x0, y0, x1, y1 = 0, 0, target_size, target_size
        full_w, full_h = target_size, target_size
    else:
        # Use symlink or copy
        if use_symlink:
            if rgb_out.exists():
                rgb_out.unlink()
            if mask_out.exists():
                mask_out.unlink()
            rgb_out.symlink_to(rgb_path.resolve())
            mask_out.symlink_to(mask_path.resolve())
        else:
            shutil.copy2(rgb_path, rgb_out)
            shutil.copy2(mask_path, mask_out)

    # Write box.txt (Fauna format: x0, y0, w, h, full_w, full_h, sharpness, label)
    crop_w = x1 - x0
    crop_h = y1 - y0
    box_data = f"{x0} {y0} {crop_w} {crop_h} {full_w} {full_h} 1.0 0"
    with open(box_out, 'w') as f:
        f.write(box_data)

    # Write metadata.json
    metadata = {
        "video_frame_id": int(frame_id),
        "crop_box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "video_frame_width": int(full_w),
        "video_frame_height": int(full_h)
    }
    with open(meta_out, 'w') as f:
        json.dump(metadata, f, indent=2)


def setup_fauna_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    use_symlink: bool = True,
    target_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> None:
    """Setup Fauna dataset from source images."""

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Detect format
    source_format = detect_source_format(source_path)
    print(f"Detected source format: {source_format}")

    # Find all image pairs
    pairs = find_image_pairs(source_path, source_format)
    print(f"Found {len(pairs)} image pairs")

    if max_samples and len(pairs) > max_samples:
        random.seed(seed)
        pairs = random.sample(pairs, max_samples)
        print(f"Sampled {max_samples} pairs")

    # Split by sequence
    sequences = {}
    for rgb, mask, seq_id in pairs:
        if seq_id not in sequences:
            sequences[seq_id] = []
        sequences[seq_id].append((rgb, mask))

    seq_names = sorted(sequences.keys())
    random.seed(seed)
    random.shuffle(seq_names)

    n_train = int(len(seq_names) * train_ratio)
    n_val = int(len(seq_names) * val_ratio)

    train_seqs = seq_names[:n_train]
    val_seqs = seq_names[n_train:n_train + n_val]
    test_seqs = seq_names[n_train + n_val:]

    print(f"Split: {len(train_seqs)} train seqs, {len(val_seqs)} val seqs, {len(test_seqs)} test seqs")

    # Create Fauna directory structure
    # data/fauna/{name}/large_scale/mouse/train/{seq}/
    dataset_name = output_path.name
    fauna_base = output_path / "large_scale" / "mouse"

    for split, split_seqs in [('train', train_seqs), ('val', val_seqs), ('test', test_seqs)]:
        split_dir = fauna_base / split

        frame_id = 0
        for seq_id in split_seqs:
            seq_dir = split_dir / seq_id
            seq_dir.mkdir(parents=True, exist_ok=True)

            for rgb_path, mask_path in sequences[seq_id]:
                create_fauna_files(
                    rgb_path, mask_path, seq_dir, frame_id,
                    use_symlink=use_symlink,
                    target_size=target_size
                )
                frame_id += 1

        print(f"{split}: {frame_id} images in {len(split_seqs)} sequences")

    # Create placeholder directories for Fauna
    for placeholder in ['few_shot_animal3d', 'few_shot_web', 'few_shot_web_back']:
        (output_path / placeholder).mkdir(parents=True, exist_ok=True)

    # Save dataset info
    info = {
        "source_dir": str(source_path.resolve()),
        "source_format": source_format,
        "total_images": len(pairs),
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "test_sequences": len(test_seqs),
        "use_symlink": use_symlink,
        "target_size": target_size
    }
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset created at: {output_path}")
    print(f"Config path for Fauna: {output_path.relative_to(Path.cwd()) if output_path.is_relative_to(Path.cwd()) else output_path}")


def main():
    parser = argparse.ArgumentParser(description="Setup Fauna dataset from existing images")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Source directory with rgb/mask images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for Fauna dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of sequences for training (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of sequences for validation (default: 0.1)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinks")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resize images to this size (crop + resize)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")

    args = parser.parse_args()

    setup_fauna_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_symlink=not args.copy,
        target_size=args.resize,
        max_samples=args.max_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
