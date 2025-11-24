#!/usr/bin/env python3
"""
Prepare Markerless Mouse Dataset for Fauna Training

This script converts the markerless_mouse_1_nerf dataset to Fauna-compatible format.

Input:
    - RGB videos: /home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf/videos_undist/{0-5}.mp4
    - Mask videos: /home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf/simpleclick_undist/{0-5}.mp4
    - Keypoints: /home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf/keypoints2d_undist/result_view_{0-5}.pkl

Output:
    - Fauna dataset: data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/train/

Usage:
    python scripts/prepare_markerless_mouse_dataset.py \
        --input_dir /home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf \
        --output_dir data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view \
        --sample_rate 10 \
        --start_frame 0 \
        --end_frame 1000
"""

import argparse
import os
import json
import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Markerless Mouse Dataset for Fauna")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/joon/dev/pose-splatter/data/markerless_mouse_1_nerf",
        help="Input directory containing videos and masks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view",
        help="Output directory for Fauna dataset"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=10,
        help="Sample every N frames (default: 10, i.e., 10fps from 100fps video)"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame index"
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=1000,
        help="End frame index (default: 1000, ~10 seconds at 100fps)"
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=6,
        help="Number of cameras (default: 6)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Output image size (default: 256)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing sequences"
    )
    return parser.parse_args()


def extract_frames_from_video(video_path, output_dir, start_frame, end_frame, sample_rate):
    """Extract frames from video using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    frame_indices = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Video: {video_path.name}")
    print(f"  Total frames: {total_frames}, FPS: {fps}")
    print(f"  Extracting frames {start_frame} to {end_frame} (every {sample_rate} frames)")

    for frame_idx in range(start_frame, min(end_frame, total_frames), sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"  Warning: Failed to read frame {frame_idx}")
            continue

        frames.append(frame)
        frame_indices.append(frame_idx)

    cap.release()

    print(f"  Extracted {len(frames)} frames")
    return frames, frame_indices


def compute_bbox_from_mask(mask):
    """Compute bounding box from binary mask"""
    coords = cv2.findNonZero(mask)

    if coords is None or len(coords) == 0:
        # Return full image bbox if no mask
        h, w = mask.shape
        return 0, 0, w, h

    x, y, w, h = cv2.boundingRect(coords)

    # Add margin
    margin = 20
    center_x = x + w // 2
    center_y = y + h // 2
    size = max(w, h) + margin * 2

    img_h, img_w = mask.shape

    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(img_w, center_x + size // 2)
    y2 = min(img_h, center_y + size // 2)

    return x1, y1, x2, y2


def create_metadata_and_box(frame_id, bbox, img_width, img_height, output_dir, prefix, crop_size=256):
    """Create metadata.json and box.txt files"""
    x1, y1, x2, y2 = bbox

    # Metadata JSON
    metadata = {
        "video_frame_id": frame_id,
        "crop_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "video_frame_width": img_width,
        "video_frame_height": img_height,
        "crop_height": crop_size,
        "crop_width": crop_size,
    }

    meta_file = output_dir / f"{prefix}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Box.txt
    crop_w = x2 - x1
    crop_h = y2 - y1
    box_line = f"{frame_id:07d} {x1:.2f} {y1:.2f} {crop_w:.2f} {crop_h:.2f} {img_width:.2f} {img_height:.2f} 0.00 0\n"

    box_file = output_dir / f"{prefix}_box.txt"
    with open(box_file, "w") as f:
        f.write(box_line)


def load_keypoints(keypoints_dir, camera_id):
    """Load 2D keypoints for a camera"""
    keypoint_file = keypoints_dir / f"result_view_{camera_id}.pkl"

    if not keypoint_file.exists():
        return None

    try:
        with open(keypoint_file, "rb") as f:
            keypoints = pickle.load(f)
        return keypoints
    except Exception as e:
        print(f"Warning: Failed to load keypoints from {keypoint_file}: {e}")
        return None


def save_keypoints(keypoints_2d, output_file):
    """Save 2D keypoints to text file"""
    # keypoints_2d shape: (num_keypoints, 2) - (x, y) coordinates
    np.savetxt(output_file, keypoints_2d, fmt="%.2f")


def process_camera(camera_id, args):
    """Process one camera's data"""
    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)

    # Input paths
    rgb_video = input_dir / "videos_undist" / f"{camera_id}.mp4"
    mask_video = input_dir / "simpleclick_undist" / f"{camera_id}.mp4"
    keypoints_dir = input_dir / "keypoints2d_undist"

    # Check if videos exist
    if not rgb_video.exists():
        print(f"Error: RGB video not found: {rgb_video}")
        return
    if not mask_video.exists():
        print(f"Error: Mask video not found: {mask_video}")
        return

    print(f"\n{'='*80}")
    print(f"Processing Camera {camera_id}")
    print(f"{'='*80}")

    # Extract RGB frames
    print("\n[1/4] Extracting RGB frames...")
    rgb_frames, frame_indices = extract_frames_from_video(
        rgb_video,
        None,
        args.start_frame,
        args.end_frame,
        args.sample_rate
    )

    # Extract mask frames
    print("\n[2/4] Extracting mask frames...")
    mask_frames, _ = extract_frames_from_video(
        mask_video,
        None,
        args.start_frame,
        args.end_frame,
        args.sample_rate
    )

    if len(rgb_frames) != len(mask_frames):
        print(f"Warning: Frame count mismatch! RGB: {len(rgb_frames)}, Mask: {len(mask_frames)}")
        min_len = min(len(rgb_frames), len(mask_frames))
        rgb_frames = rgb_frames[:min_len]
        mask_frames = mask_frames[:min_len]
        frame_indices = frame_indices[:min_len]

    # Load keypoints
    print("\n[3/4] Loading keypoints...")
    keypoints_data = load_keypoints(keypoints_dir, camera_id)

    # Create output directory for this camera
    # Using large_scale structure: camera_id as sequence_id
    sequence_name = f"cam{camera_id:02d}_seq_000"
    output_dir = output_base / "train" / sequence_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and len(list(output_dir.glob("*_rgb.png"))) > 0:
        print(f"  Skipping existing sequence: {sequence_name}")
        return

    print(f"\n[4/4] Saving frames to {output_dir}...")

    # Process each frame
    for idx, (rgb_frame, mask_frame, orig_frame_idx) in enumerate(tqdm(
        zip(rgb_frames, mask_frames, frame_indices),
        total=len(rgb_frames),
        desc=f"  Camera {camera_id}"
    )):
        # Convert mask to binary (grayscale)
        if len(mask_frame.shape) == 3:
            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_frame

        # Threshold to binary
        _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        # Get image dimensions
        img_h, img_w = rgb_frame.shape[:2]

        # Compute bounding box (already square from compute_bbox_from_mask)
        bbox = compute_bbox_from_mask(mask_binary)
        x1, y1, x2, y2 = bbox

        # Crop RGB and mask to square bounding box
        rgb_cropped = rgb_frame[y1:y2, x1:x2]
        mask_cropped = mask_binary[y1:y2, x1:x2]

        # Resize to target size (256x256 by default)
        target_size = (args.image_size, args.image_size)
        rgb_resized = cv2.resize(rgb_cropped, target_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_cropped, target_size, interpolation=cv2.INTER_NEAREST)

        # Frame naming: 7-digit frame ID
        frame_id = orig_frame_idx
        prefix = f"{frame_id:07d}"

        # Save cropped and resized RGB image
        rgb_file = output_dir / f"{prefix}_rgb.png"
        cv2.imwrite(str(rgb_file), rgb_resized)

        # Save cropped and resized mask image
        mask_file = output_dir / f"{prefix}_mask.png"
        cv2.imwrite(str(mask_file), mask_resized)

        # Create metadata and box.txt
        create_metadata_and_box(
            frame_id=frame_id,
            bbox=bbox,
            img_width=img_w,
            img_height=img_h,
            output_dir=output_dir,
            prefix=prefix,
            crop_size=args.image_size
        )

        # Save keypoints if available
        if keypoints_data is not None and orig_frame_idx < len(keypoints_data):
            kpts = keypoints_data[orig_frame_idx]
            if kpts is not None and len(kpts) > 0:
                keypoint_file = output_dir / f"{prefix}_keypoint.txt"
                save_keypoints(kpts, keypoint_file)

    print(f"\n✅ Camera {camera_id} complete: {len(rgb_frames)} frames saved to {output_dir}")


def create_validation_split(output_dir, val_ratio=0.1):
    """Create validation split by moving last N% of frames to val directory"""
    output_base = Path(output_dir)
    train_dir = output_base / "train"
    val_dir = output_base / "val"

    if not train_dir.exists():
        print("Warning: No train directory found, skipping validation split")
        return

    # Get all sequences
    sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    if len(sequences) == 0:
        print("Warning: No sequences found in train directory")
        return

    print(f"\n{'='*80}")
    print("Creating validation split...")
    print(f"{'='*80}")

    val_dir.mkdir(parents=True, exist_ok=True)

    for seq_dir in sequences:
        # Get all frames in this sequence
        rgb_files = sorted(seq_dir.glob("*_rgb.png"))

        if len(rgb_files) == 0:
            continue

        # Calculate split point
        total_frames = len(rgb_files)
        val_count = max(1, int(total_frames * val_ratio))
        train_count = total_frames - val_count

        print(f"\nSequence: {seq_dir.name}")
        print(f"  Total frames: {total_frames}")
        print(f"  Train: {train_count}, Val: {val_count}")

        # Create val sequence directory
        val_seq_dir = val_dir / seq_dir.name
        val_seq_dir.mkdir(parents=True, exist_ok=True)

        # Move last val_count frames to validation
        val_frames = rgb_files[-val_count:]

        for rgb_file in val_frames:
            prefix = rgb_file.stem.replace("_rgb", "")

            # Move all associated files
            for suffix in ["rgb.png", "mask.png", "metadata.json", "box.txt", "keypoint.txt"]:
                src_file = seq_dir / f"{prefix}_{suffix}"
                dst_file = val_seq_dir / f"{prefix}_{suffix}"

                if src_file.exists():
                    src_file.rename(dst_file)

        print(f"  ✅ Moved {val_count} frames to validation")

    print(f"\n✅ Validation split complete!")


def main():
    args = parse_args()

    print("="*80)
    print("Markerless Mouse Dataset Preparation for Fauna")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input dir: {args.input_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Frame range: {args.start_frame} to {args.end_frame}")
    print(f"  Sample rate: every {args.sample_rate} frames")
    print(f"  Number of cameras: {args.num_cameras}")
    print(f"  Output image size: {args.image_size}x{args.image_size}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each camera
    for camera_id in range(args.num_cameras):
        try:
            process_camera(camera_id, args)
        except Exception as e:
            print(f"\n❌ Error processing camera {camera_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create validation split
    create_validation_split(args.output_dir, val_ratio=0.1)

    print("\n" + "="*80)
    print("✅ Dataset preparation complete!")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Verify dataset: ls -la {args.output_dir}/train/")
    print(f"  2. Create config: config/model/fauna_mouse_markerless.yaml")
    print(f"  3. Run debug training: python run.py --config-name train_fauna_mouse_markerless_debug")
    print(f"  4. Run full training: python run.py --config-name train_fauna_mouse_markerless")


if __name__ == "__main__":
    main()
