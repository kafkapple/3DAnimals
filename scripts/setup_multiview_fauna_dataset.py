#!/usr/bin/env python3
"""
Setup Multi-view Fauna dataset from sam3d_gui session data.

This script creates a Fauna-compatible dataset from sam3d_gui sessions,
properly organizing data by mouse, camera view, and sequence.

Data structure understanding:
- session_metadata.json contains video info
- video_path format: mouse_{1,2}/Camera{1-6}/{seq}.mp4
- saved_dir format: video_XXX_YYYY (XXX=idx, YYYY=sequence_start)
- Each video folder has frame_ZZZZ/mask.png, original.png

Pose Splatter paper setup:
- Training: 1 timestep × 6 camera views (same frame, different views)
- Testing: Different timestep × 6 camera views (unseen timestep)

Usage:
    # Debug mode (Pose Splatter reproduction)
    python scripts/setup_multiview_fauna_dataset.py \
        --session_dir /path/to/session \
        --output_dir data/fauna/mouse_6view_debug \
        --mode pose_splatter_debug \
        --mouse_id 1 \
        --train_seq 0 \
        --test_seq 3000

    # Full dataset
    python scripts/setup_multiview_fauna_dataset.py \
        --session_dir /path/to/session \
        --output_dir data/fauna/mouse_multiview \
        --mode full
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np


def parse_video_path(video_path: str) -> Tuple[int, int, int]:
    """Parse video_path to extract mouse_id, camera_id, sequence.

    Args:
        video_path: e.g., "mouse_1/Camera3/6000.mp4"

    Returns:
        (mouse_id, camera_id, sequence_start)
    """
    # Parse: mouse_{id}/Camera{id}/{seq}.mp4
    match = re.match(r'mouse_(\d+)/Camera(\d+)/(\d+)\.mp4', video_path)
    if not match:
        raise ValueError(f"Cannot parse video_path: {video_path}")

    mouse_id = int(match.group(1))
    camera_id = int(match.group(2))
    seq_start = int(match.group(3))

    return mouse_id, camera_id, seq_start


def load_session_metadata(session_dir: Path) -> Dict:
    """Load and parse session metadata."""
    metadata_path = session_dir / "session_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Parse video info
    videos_by_key = {}  # (mouse_id, camera_id, seq) -> video_info
    for video in metadata['videos']:
        mouse_id, camera_id, seq = parse_video_path(video['video_path'])
        video['mouse_id'] = mouse_id
        video['camera_id'] = camera_id
        video['sequence'] = seq
        videos_by_key[(mouse_id, camera_id, seq)] = video

    metadata['videos_by_key'] = videos_by_key
    return metadata


def compute_bbox_from_mask(mask_path: Path, padding_ratio: float = 0.1) -> Tuple[int, int, int, int, int, int]:
    """Compute bounding box from mask image."""
    mask = np.array(Image.open(mask_path).convert('L'))
    h, w = mask.shape

    rows = np.any(mask > 128, axis=1)
    cols = np.any(mask > 128, axis=0)

    if not rows.any() or not cols.any():
        return 0, 0, w, h, w, h

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    pad_w = int((x1 - x0) * padding_ratio)
    pad_h = int((y1 - y0) * padding_ratio)

    x0 = max(0, x0 - pad_w)
    y0 = max(0, y0 - pad_h)
    x1 = min(w, x1 + pad_w)
    y1 = min(h, y1 + pad_h)

    return int(x0), int(y0), int(x1), int(y1), int(w), int(h)


def create_fauna_files(
    rgb_path: Path,
    mask_path: Path,
    output_dir: Path,
    frame_id: int,
    use_symlink: bool = True
) -> None:
    """Create Fauna-format files for a single image pair."""
    base_name = f"{frame_id:07d}"

    rgb_out = output_dir / f"{base_name}_rgb.png"
    mask_out = output_dir / f"{base_name}_mask.png"
    box_out = output_dir / f"{base_name}_box.txt"
    meta_out = output_dir / f"{base_name}_metadata.json"

    # Get bbox from mask
    x0, y0, x1, y1, full_w, full_h = compute_bbox_from_mask(mask_path)

    # Symlink or copy
    if use_symlink:
        if rgb_out.exists():
            rgb_out.unlink()
        if mask_out.exists():
            mask_out.unlink()
        rgb_out.symlink_to(rgb_path.resolve())
        mask_out.symlink_to(mask_path.resolve())
    else:
        import shutil
        shutil.copy2(rgb_path, rgb_out)
        shutil.copy2(mask_path, mask_out)

    # Write box.txt
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


def get_6view_frames(
    session_dir: Path,
    metadata: Dict,
    mouse_id: int,
    sequence: int,
    frame_idx: int = 0
) -> List[Tuple[Path, Path, int]]:
    """Get 6-view frames for a specific timestep.

    Returns:
        List of (rgb_path, mask_path, camera_id)
    """
    frames = []
    for camera_id in range(1, 7):  # Camera 1-6
        key = (mouse_id, camera_id, sequence)
        if key not in metadata['videos_by_key']:
            print(f"Warning: Missing video for {key}")
            continue

        video_info = metadata['videos_by_key'][key]
        saved_dir = video_info['saved_dir']

        frame_dir = session_dir / saved_dir / f"frame_{frame_idx:04d}"
        rgb_path = frame_dir / "original.png"
        mask_path = frame_dir / "mask.png"

        if rgb_path.exists() and mask_path.exists():
            frames.append((rgb_path, mask_path, camera_id))
        else:
            print(f"Warning: Missing files in {frame_dir}")

    return frames


def setup_pose_splatter_debug(
    session_dir: Path,
    output_dir: Path,
    mouse_id: int = 1,
    train_seq: int = 0,
    test_seq: int = 3000,
    train_frame: int = 0,
    test_frame: int = 0,
    use_symlink: bool = True
) -> None:
    """Setup Pose Splatter paper debug dataset.

    Paper setup:
    - Training: 1 timestep × 6 views = 6 images
    - Testing: Different timestep × 6 views = 6 images
    """
    metadata = load_session_metadata(session_dir)
    output_path = Path(output_dir)

    # Create Fauna directory structure
    fauna_base = output_path / "large_scale" / "mouse"

    # Training data: 6 views from train_seq
    train_dir = fauna_base / "train" / f"seq{train_seq}"
    train_dir.mkdir(parents=True, exist_ok=True)

    train_frames = get_6view_frames(session_dir, metadata, mouse_id, train_seq, train_frame)
    print(f"Training: mouse_{mouse_id}, seq={train_seq}, frame={train_frame}")
    print(f"  Found {len(train_frames)} camera views")

    frame_id = 0
    for rgb_path, mask_path, camera_id in train_frames:
        # Use unique frame_id encoding camera info
        unique_id = train_seq * 10 + camera_id
        create_fauna_files(rgb_path, mask_path, train_dir, unique_id, use_symlink)
        print(f"  Camera {camera_id}: {rgb_path.parent.name}")
        frame_id += 1

    # Validation data: same as test (for small dataset)
    val_dir = fauna_base / "val" / f"seq{test_seq}"
    val_dir.mkdir(parents=True, exist_ok=True)

    test_frames = get_6view_frames(session_dir, metadata, mouse_id, test_seq, test_frame)
    print(f"\nTesting: mouse_{mouse_id}, seq={test_seq}, frame={test_frame}")
    print(f"  Found {len(test_frames)} camera views")

    for rgb_path, mask_path, camera_id in test_frames:
        unique_id = test_seq * 10 + camera_id
        create_fauna_files(rgb_path, mask_path, val_dir, unique_id, use_symlink)
        print(f"  Camera {camera_id}: {rgb_path.parent.name}")

    # Test data: same as val
    test_dir = fauna_base / "test" / f"seq{test_seq}"
    test_dir.mkdir(parents=True, exist_ok=True)
    for rgb_path, mask_path, camera_id in test_frames:
        unique_id = test_seq * 10 + camera_id
        create_fauna_files(rgb_path, mask_path, test_dir, unique_id, use_symlink)

    # Create placeholder directories
    for placeholder in ['few_shot_animal3d', 'few_shot_web', 'few_shot_web_back']:
        (output_path / placeholder).mkdir(parents=True, exist_ok=True)

    # Save dataset info
    info = {
        "mode": "pose_splatter_debug",
        "session_dir": str(session_dir.resolve()),
        "mouse_id": mouse_id,
        "train_sequence": train_seq,
        "test_sequence": test_seq,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "train_images": len(train_frames),
        "test_images": len(test_frames),
        "use_symlink": use_symlink
    }
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Dataset created at: {output_path}")
    print(f"   Train: {len(train_frames)} images (6 views × 1 timestep)")
    print(f"   Test: {len(test_frames)} images (6 views × 1 timestep)")


def setup_full_dataset(
    session_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    use_symlink: bool = True
) -> None:
    """Setup full multi-view dataset with all sequences and frames."""
    metadata = load_session_metadata(session_dir)
    output_path = Path(output_dir)

    # Get all unique (mouse, sequence) combinations
    all_keys = list(metadata['videos_by_key'].keys())
    sequences = sorted(set((m, s) for m, c, s in all_keys))

    print(f"Found {len(sequences)} unique (mouse, sequence) combinations")

    # Split sequences for train/val/test
    import random
    random.seed(42)
    random.shuffle(sequences)

    n_train = int(len(sequences) * train_ratio)
    n_val = int(len(sequences) * 0.1)

    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:n_train + n_val]
    test_seqs = sequences[n_train + n_val:]

    fauna_base = output_path / "large_scale" / "mouse"

    total_images = {'train': 0, 'val': 0, 'test': 0}

    for split, split_seqs in [('train', train_seqs), ('val', val_seqs), ('test', test_seqs)]:
        for mouse_id, seq in split_seqs:
            # Get first video to check num_frames
            first_key = (mouse_id, 1, seq)
            if first_key not in metadata['videos_by_key']:
                continue
            num_frames = metadata['videos_by_key'][first_key]['num_frames']

            seq_name = f"mouse{mouse_id}_seq{seq}"
            seq_dir = fauna_base / split / seq_name
            seq_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx in range(num_frames):
                frames_6view = get_6view_frames(session_dir, metadata, mouse_id, seq, frame_idx)
                for rgb_path, mask_path, camera_id in frames_6view:
                    unique_id = seq * 1000 + frame_idx * 10 + camera_id
                    create_fauna_files(rgb_path, mask_path, seq_dir, unique_id, use_symlink)
                    total_images[split] += 1

        print(f"{split}: {total_images[split]} images")

    # Create placeholder directories
    for placeholder in ['few_shot_animal3d', 'few_shot_web', 'few_shot_web_back']:
        (output_path / placeholder).mkdir(parents=True, exist_ok=True)

    # Save dataset info
    info = {
        "mode": "full",
        "session_dir": str(session_dir.resolve()),
        "train_images": total_images['train'],
        "val_images": total_images['val'],
        "test_images": total_images['test'],
        "use_symlink": use_symlink
    }
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Dataset created at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Setup Multi-view Fauna dataset")
    parser.add_argument("--session_dir", type=str, required=True,
                        help="Path to sam3d_gui session directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for Fauna dataset")
    parser.add_argument("--mode", type=str, default="pose_splatter_debug",
                        choices=["pose_splatter_debug", "full"],
                        help="Dataset mode")
    parser.add_argument("--mouse_id", type=int, default=1,
                        help="Mouse ID for debug mode (1 or 2)")
    parser.add_argument("--train_seq", type=int, default=0,
                        help="Training sequence for debug mode")
    parser.add_argument("--test_seq", type=int, default=3000,
                        help="Test sequence for debug mode")
    parser.add_argument("--train_frame", type=int, default=0,
                        help="Training frame index within sequence")
    parser.add_argument("--test_frame", type=int, default=0,
                        help="Test frame index within sequence")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train ratio for full mode")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinks")

    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    output_dir = Path(args.output_dir)

    if args.mode == "pose_splatter_debug":
        setup_pose_splatter_debug(
            session_dir=session_dir,
            output_dir=output_dir,
            mouse_id=args.mouse_id,
            train_seq=args.train_seq,
            test_seq=args.test_seq,
            train_frame=args.train_frame,
            test_frame=args.test_frame,
            use_symlink=not args.copy
        )
    else:
        setup_full_dataset(
            session_dir=session_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            use_symlink=not args.copy
        )


if __name__ == "__main__":
    main()
