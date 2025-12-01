#!/usr/bin/env python3
"""
Prepare 6-view mouse data for Fauna training.

This script follows the Pose Splatter paper methodology:
- Training: 1 timestep × 6 views = 6 images
- Testing: Different timestep × 6 views = 6 images

The 6 views from the same timestep provide multi-view supervision for
shape learning, which helps prevent mesh collapse.
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_frame_from_video(video_path: str, frame_idx: int) -> np.ndarray:
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def extract_mask_from_video(mask_video_path: str, frame_idx: int) -> np.ndarray:
    """Extract mask from mask video and binarize."""
    cap = cv2.VideoCapture(mask_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read frame {frame_idx} from {mask_video_path}")
    # Convert to grayscale and binarize
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return mask


def compute_bbox_from_mask(mask: np.ndarray, padding_ratio: float = 0.1) -> tuple:
    """Compute bounding box from mask with padding."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add padding
    h, w = mask.shape[:2]
    pad_y = int((y_max - y_min) * padding_ratio)
    pad_x = int((x_max - x_min) * padding_ratio)

    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)

    return x_min, y_min, x_max - x_min, y_max - y_min


def crop_and_resize(image: np.ndarray, bbox: tuple, target_size: int = 256) -> np.ndarray:
    """Crop image to bbox and resize to square."""
    x, y, w, h = bbox

    # Make square crop (use larger dimension)
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(image.shape[1], x1 + size)
    y2 = min(image.shape[0], y1 + size)

    # Crop
    cropped = image[y1:y2, x1:x2]

    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return resized


def prepare_6view_data(
    data_dir: str,
    output_dir: str,
    train_frame: int = 5000,
    test_frame: int = 10000,
    target_size: int = 256,
    num_cameras: int = 6
):
    """
    Prepare 6-view data following Pose Splatter methodology.

    Args:
        data_dir: Path to markerless_mouse_1_nerf data
        output_dir: Output directory for Fauna format data
        train_frame: Frame index for training (1 timestep)
        test_frame: Frame index for testing (different timestep)
        target_size: Output image size (256 for Fauna)
        num_cameras: Number of camera views
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    videos_dir = data_dir / "videos_undist"
    masks_dir = data_dir / "simpleclick_undist"

    # Create output directories
    train_dir = output_dir / "train" / "seq_000"
    test_dir = output_dir / "test" / "seq_000"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Preparing 6-view Fauna Dataset ===")
    print(f"Data source: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Train frame: {train_frame}")
    print(f"Test frame: {test_frame}")
    print()

    def process_frame(frame_idx: int, output_subdir: Path, split_name: str):
        """Process all views for a single frame."""
        print(f"\nProcessing {split_name} frame {frame_idx}...")

        for cam_idx in tqdm(range(num_cameras), desc=f"  Camera views"):
            video_path = videos_dir / f"{cam_idx}.mp4"
            mask_path = masks_dir / f"{cam_idx}.mp4"

            # Extract frame and mask
            frame = extract_frame_from_video(str(video_path), frame_idx)
            mask = extract_mask_from_video(str(mask_path), frame_idx)

            # Compute bbox from mask
            bbox = compute_bbox_from_mask(mask)
            if bbox is None:
                print(f"    Warning: Empty mask for camera {cam_idx}, frame {frame_idx}")
                continue

            # Crop and resize
            frame_cropped = crop_and_resize(frame, bbox, target_size)
            mask_cropped = crop_and_resize(mask, bbox, target_size)

            # Ensure mask is binary after resize
            _, mask_cropped = cv2.threshold(mask_cropped, 127, 255, cv2.THRESH_BINARY)

            # Generate frame ID (unique per camera and frame)
            # Format: CCCFFFF where CCC=camera, FFFF=frame
            file_id = f"{cam_idx:03d}{frame_idx:04d}"

            # Save RGB
            rgb_path = output_subdir / f"{file_id}_rgb.png"
            cv2.imwrite(str(rgb_path), frame_cropped)

            # Save mask
            mask_path_out = output_subdir / f"{file_id}_mask.png"
            cv2.imwrite(str(mask_path_out), mask_cropped)

            # Generate box.txt (9 values format)
            x, y, w, h = bbox
            orig_h, orig_w = frame.shape[:2]
            box_line = f"{file_id} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {orig_w:.2f} {orig_h:.2f} 0.00 0\n"
            box_path = output_subdir / f"{file_id}_box.txt"
            with open(box_path, "w") as f:
                f.write(box_line)

            # Generate metadata.json
            metadata = {
                "video_frame_id": frame_idx,
                "camera_id": cam_idx,
                "source": "markerless_mouse_6view",
                "split": split_name,
                "crop_box_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                "video_frame_width": orig_w,
                "video_frame_height": orig_h
            }
            metadata_path = output_subdir / f"{file_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"  Saved {num_cameras} views to {output_subdir}")

    # Process train and test frames
    process_frame(train_frame, train_dir, "train")
    process_frame(test_frame, test_dir, "test")

    # Summary
    train_count = len(list(train_dir.glob("*_rgb.png")))
    test_count = len(list(test_dir.glob("*_rgb.png")))

    print(f"\n=== Summary ===")
    print(f"Train images: {train_count} (frame {train_frame}, {num_cameras} views)")
    print(f"Test images: {test_count} (frame {test_frame}, {num_cameras} views)")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/seq_000/  ({train_count} images)")
    print(f"  └── test/seq_000/   ({test_count} images)")
    print(f"\nReady for Fauna training!")


def main():
    parser = argparse.ArgumentParser(description="Prepare 6-view mouse data for Fauna")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/joon/dev/project_splatter/data/markerless_mouse_1_nerf",
        help="Path to markerless_mouse_1_nerf data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/fauna/mouse_6view_posesplatter",
        help="Output directory for Fauna format"
    )
    parser.add_argument(
        "--train_frame",
        type=int,
        default=5000,
        help="Frame index for training"
    )
    parser.add_argument(
        "--test_frame",
        type=int,
        default=10000,
        help="Frame index for testing"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Output image size"
    )

    args = parser.parse_args()

    prepare_6view_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_frame=args.train_frame,
        test_frame=args.test_frame,
        target_size=args.target_size
    )


if __name__ == "__main__":
    main()
