#!/usr/bin/env python3
"""
SAM3D Dataset Preprocessing Pipeline
=====================================

Purpose:
  - Process SAM3D GUI output datasets for 3DAnimals training
  - Generate missing box.txt and metadata.json files
  - Validate dataset integrity
  - Prepare for Fauna dataset format

Input Format (SAM3D GUI output):
  source/
  └── {animal}/
      └── train/
          ├── seq_000/
          │   ├── {frame_id}_rgb.png
          │   └── {frame_id}_mask.png
          └── seq_001/
              └── ...

Output Format (Fauna compatible):
  target/
  └── {animal}/
      └── train/
          ├── seq_000/
          │   ├── {frame_id}_rgb.png
          │   ├── {frame_id}_mask.png
          │   ├── {frame_id}_box.txt
          │   └── {frame_id}_metadata.json
          └── seq_001/
              └── ...

Usage:
  # Interactive mode
  python scripts/preprocess_sam3d_dataset.py --interactive

  # Manual mode
  python scripts/preprocess_sam3d_dataset.py \\
    --source /path/to/sam3d_gui/outputs/fauna_datasets/mouse \\
    --animal mouse \\
    --output data/fauna_processed
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from tqdm import tqdm


class SAM3DDatasetPreprocessor:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, message: str, level: str = "INFO"):
        if self.verbose:
            colors = {"INFO": "\033[0;32m", "WARN": "\033[1;33m", "ERROR": "\033[0;31m"}
            color = colors.get(level, "")
            reset = "\033[0m"
            print(f"{color}[{level}]{reset} {message}")

    def analyze_dataset(self, source_dir: Path) -> Dict:
        """Analyze SAM3D dataset structure."""
        self.log("=" * 80)
        self.log("Analyzing Dataset Structure")
        self.log("=" * 80)

        info = {
            'sequences': [],
            'total_frames': 0,
            'has_rgb': True,
            'has_mask': True,
            'has_box': False,
            'has_metadata': False,
            'missing_files': []
        }

        train_dir = source_dir / "train"
        if not train_dir.exists():
            self.log(f"Train directory not found: {train_dir}", "ERROR")
            return info

        sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])

        for seq in sequences:
            rgb_files = sorted(seq.glob("*_rgb.png"))
            mask_files = sorted(seq.glob("*_mask.png"))
            box_files = sorted(seq.glob("*_box.txt"))
            meta_files = sorted(seq.glob("*_metadata.json"))

            seq_info = {
                'name': seq.name,
                'path': seq,
                'rgb_count': len(rgb_files),
                'mask_count': len(mask_files),
                'box_count': len(box_files),
                'meta_count': len(meta_files),
                'frame_ids': sorted([f.stem.replace("_rgb", "") for f in rgb_files])
            }

            info['sequences'].append(seq_info)
            info['total_frames'] += len(rgb_files)

            if len(box_files) == 0:
                info['has_box'] = False
            if len(meta_files) == 0:
                info['has_metadata'] = False

            # Check RGB/Mask pairs
            rgb_ids = set([f.stem.replace("_rgb", "") for f in rgb_files])
            mask_ids = set([f.stem.replace("_mask", "") for f in mask_files])
            missing = rgb_ids - mask_ids
            if missing:
                info['missing_files'].extend([f"{seq.name}/{fid}_mask.png" for fid in missing])

        # Print summary
        self.log(f"\n📁 Source: {source_dir}")
        self.log(f"📊 Sequences: {len(sequences)}")
        self.log(f"🖼️  Total frames: {info['total_frames']}\n")

        for seq_info in info['sequences']:
            self.log(f"  {seq_info['name']}:")
            self.log(f"    RGB:      {seq_info['rgb_count']} {'✓' if seq_info['rgb_count'] > 0 else '✗'}")
            self.log(f"    Mask:     {seq_info['mask_count']} {'✓' if seq_info['mask_count'] > 0 else '✗'}")
            self.log(f"    Box:      {seq_info['box_count']} {'✓' if seq_info['box_count'] > 0 else '✗'}")
            self.log(f"    Metadata: {seq_info['meta_count']} {'✓' if seq_info['meta_count'] > 0 else '✗'}")

        return info

    def extract_bbox_from_mask(self, mask_path: Path) -> Tuple[int, int, int, int]:
        """Extract bounding box from binary mask.

        Returns:
            (x0, y0, width, height)
        """
        try:
            mask = np.array(Image.open(mask_path).convert('L'))

            # Find non-zero pixels
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)

            if not rows.any() or not cols.any():
                # Empty mask, return image center box
                h, w = mask.shape
                return (w//4, h//4, w//2, h//2)

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Add small margin
            margin = 5
            h, w = mask.shape
            x0 = max(0, x_min - margin)
            y0 = max(0, y_min - margin)
            x1 = min(w, x_max + margin)
            y1 = min(h, y_max + margin)

            width = x1 - x0
            height = y1 - y0

            return (x0, y0, width, height)

        except Exception as e:
            self.log(f"Error extracting bbox from {mask_path}: {e}", "ERROR")
            return (0, 0, 256, 256)  # Default

    def generate_box_txt(self, mask_path: Path, output_path: Path, frame_id: str, image_size: Tuple[int, int] = (256, 256)):
        """Generate box.txt file from mask.

        Format: frame_id crop_x0 crop_y0 crop_w crop_h full_w full_h sharpness label
        """
        x0, y0, w, h = self.extract_bbox_from_mask(mask_path)
        full_w, full_h = image_size
        sharpness = 1.0  # Default sharpness
        label = 0  # Default label

        box_data = f"{frame_id} {x0} {y0} {w} {h} {full_w} {full_h} {sharpness} {label}"

        with open(output_path, 'w') as f:
            f.write(box_data)

    def generate_metadata_json(self, box_path: Path, output_path: Path, frame_id: str):
        """Generate Fauna-compatible metadata.json from box.txt.

        Reads box.txt to get crop box and image dimensions, then generates
        metadata in Fauna dataset format.
        """
        # Read box.txt to get all necessary information
        with open(box_path, 'r') as f:
            line = f.read().strip()
            parts = line.split()

            if len(parts) != 9:
                raise ValueError(f"Invalid box.txt format: {box_path}")

            frame_id_str, x0, y0, w, h, full_w, full_h, sharpness, label = parts

        # Convert to proper types
        x0, y0 = int(x0), int(y0)
        w, h = int(w), int(h)
        full_w, full_h = int(full_w), int(full_h)
        sharpness = float(sharpness)
        label = int(label)

        # Create Fauna-format metadata
        metadata = {
            "video_frame_id": int(frame_id),
            "crop_box_xyxy": [x0, y0, x0 + w, y0 + h],
            "video_frame_width": full_w,
            "video_frame_height": full_h,
            "sharpness": sharpness,
            "crop_height": h,
            "crop_width": w,
            "label": label
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_dataset(
        self,
        source_dir: Path,
        output_dir: Path,
        animal_name: str,
        copy_files: bool = True,
        overwrite: bool = False
    ) -> Dict:
        """Process complete dataset and generate missing files."""

        self.log("\n" + "=" * 80)
        self.log("Processing Dataset")
        self.log("=" * 80)

        # Analyze first
        info = self.analyze_dataset(source_dir)

        if info['total_frames'] == 0:
            self.log("No frames found to process", "ERROR")
            return {}

        # Create output structure
        output_animal_dir = output_dir / animal_name
        output_train_dir = output_animal_dir / "train"
        output_train_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'processed_frames': 0,
            'generated_boxes': 0,
            'generated_metadata': 0,
            'copied_files': 0,
            'skipped_frames': 0
        }

        # Process each sequence
        for seq_info in info['sequences']:
            seq_name = seq_info['name']
            source_seq = seq_info['path']
            target_seq = output_train_dir / seq_name
            target_seq.mkdir(parents=True, exist_ok=True)

            self.log(f"\nProcessing {seq_name}...")

            frame_ids = seq_info['frame_ids']

            for frame_id in tqdm(frame_ids, desc=f"  {seq_name}", disable=not self.verbose):
                # File paths
                source_rgb = source_seq / f"{frame_id}_rgb.png"
                source_mask = source_seq / f"{frame_id}_mask.png"
                source_box = source_seq / f"{frame_id}_box.txt"
                source_meta = source_seq / f"{frame_id}_metadata.json"

                target_rgb = target_seq / f"{frame_id}_rgb.png"
                target_mask = target_seq / f"{frame_id}_mask.png"
                target_box = target_seq / f"{frame_id}_box.txt"
                target_meta = target_seq / f"{frame_id}_metadata.json"

                # Skip if target exists and not overwriting
                if not overwrite and all([
                    target_rgb.exists(),
                    target_mask.exists(),
                    target_box.exists(),
                    target_meta.exists()
                ]):
                    stats['skipped_frames'] += 1
                    continue

                # Copy or symlink RGB and Mask
                if copy_files:
                    if source_rgb.exists():
                        shutil.copy2(source_rgb, target_rgb)
                        stats['copied_files'] += 1
                    if source_mask.exists():
                        shutil.copy2(source_mask, target_mask)
                        stats['copied_files'] += 1
                else:
                    if source_rgb.exists() and not target_rgb.exists():
                        target_rgb.symlink_to(source_rgb.resolve())
                    if source_mask.exists() and not target_mask.exists():
                        target_mask.symlink_to(source_mask.resolve())

                # Get image size from RGB
                if source_rgb.exists():
                    img = Image.open(source_rgb)
                    image_size = img.size
                else:
                    image_size = (256, 256)

                # Generate box.txt if missing
                if not source_box.exists() and source_mask.exists():
                    self.generate_box_txt(source_mask, target_box, frame_id, image_size)
                    stats['generated_boxes'] += 1
                elif source_box.exists():
                    if copy_files:
                        shutil.copy2(source_box, target_box)
                    else:
                        if not target_box.exists():
                            target_box.symlink_to(source_box.resolve())

                # Generate metadata.json if missing
                # Note: metadata.json must be generated after box.txt exists
                if not source_meta.exists():
                    # Generate from box.txt (must exist at this point)
                    if target_box.exists():
                        self.generate_metadata_json(target_box, target_meta, frame_id)
                        stats['generated_metadata'] += 1
                    else:
                        self.log(f"Warning: Cannot generate metadata without box.txt for {frame_id}", "WARNING")
                elif source_meta.exists():
                    if copy_files:
                        shutil.copy2(source_meta, target_meta)
                    else:
                        if not target_meta.exists():
                            target_meta.symlink_to(source_meta.resolve())

                stats['processed_frames'] += 1

        # Summary
        self.log("\n" + "=" * 80)
        self.log("Processing Complete!")
        self.log("=" * 80)
        self.log(f"Processed frames:    {stats['processed_frames']}")
        self.log(f"Generated boxes:     {stats['generated_boxes']}")
        self.log(f"Generated metadata:  {stats['generated_metadata']}")
        self.log(f"Copied files:        {stats['copied_files']}")
        self.log(f"Skipped frames:      {stats['skipped_frames']}")

        self.log(f"\n📁 Output directory: {output_animal_dir}")

        return stats

    def validate_output(self, output_dir: Path, animal_name: str) -> bool:
        """Validate processed dataset."""
        self.log("\n" + "=" * 80)
        self.log("Validating Output")
        self.log("=" * 80)

        train_dir = output_dir / animal_name / "train"

        if not train_dir.exists():
            self.log("Train directory not found", "ERROR")
            return False

        sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])

        all_valid = True

        for seq in sequences:
            rgb_files = sorted(seq.glob("*_rgb.png"))
            mask_files = sorted(seq.glob("*_mask.png"))
            box_files = sorted(seq.glob("*_box.txt"))
            meta_files = sorted(seq.glob("*_metadata.json"))

            counts = {
                'rgb': len(rgb_files),
                'mask': len(mask_files),
                'box': len(box_files),
                'meta': len(meta_files)
            }

            status = "✓" if all(c == counts['rgb'] for c in counts.values()) else "✗"

            self.log(f"  {seq.name}: RGB={counts['rgb']}, Mask={counts['mask']}, Box={counts['box']}, Meta={counts['meta']} {status}")

            if not all(c == counts['rgb'] for c in counts.values()):
                all_valid = False

        if all_valid:
            self.log("\n✅ All files validated successfully!")
        else:
            self.log("\n⚠️  Some files are missing", "WARN")

        return all_valid


def interactive_mode():
    """Interactive preprocessing wizard."""
    print("\n" + "=" * 80)
    print("SAM3D Dataset Preprocessing Wizard")
    print("=" * 80)

    # Step 1: Source directory
    print("\n[Step 1] Source Directory")
    default_source = "/home/joon/dev/sam3d_gui/outputs/fauna_datasets/mouse"
    source_input = input(f"Source directory [{default_source}]: ").strip()
    source_dir = Path(source_input if source_input else default_source)

    if not source_dir.exists():
        print(f"❌ Source not found: {source_dir}")
        return

    # Step 2: Animal name
    print("\n[Step 2] Animal Name")
    default_animal = source_dir.name
    animal_input = input(f"Animal name [{default_animal}]: ").strip()
    animal_name = animal_input if animal_input else default_animal

    # Step 3: Output directory
    print("\n[Step 3] Output Directory")
    default_output = "/home/joon/dev/3DAnimals/data/fauna_processed"
    output_input = input(f"Output directory [{default_output}]: ").strip()
    output_dir = Path(output_input if output_input else default_output)

    # Step 4: Copy or symlink
    print("\n[Step 4] File Handling")
    print("  [1] Copy files (standalone, uses disk space)")
    print("  [2] Symlink files (saves space, requires source)")
    method_input = input("Choose method [1]: ").strip()
    copy_files = method_input != "2"

    # Step 5: Overwrite
    print("\n[Step 5] Overwrite Existing")
    overwrite_input = input("Overwrite existing files? (y/n) [n]: ").strip().lower()
    overwrite = overwrite_input == "y"

    # Confirm
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Source:    {source_dir}")
    print(f"Animal:    {animal_name}")
    print(f"Output:    {output_dir}")
    print(f"Method:    {'Copy' if copy_files else 'Symlink'}")
    print(f"Overwrite: {overwrite}")
    print()

    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # Process
    preprocessor = SAM3DDatasetPreprocessor()
    stats = preprocessor.process_dataset(
        source_dir,
        output_dir,
        animal_name,
        copy_files=copy_files,
        overwrite=overwrite
    )

    # Validate
    preprocessor.validate_output(output_dir, animal_name)

    # Next steps
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print(f"\n1. Prepare for 3DAnimals training:")
    print(f"   python scripts/prepare_fauna_dataset.py \\")
    print(f"     --source {output_dir / animal_name / 'train'} \\")
    print(f"     --animal {animal_name}")
    print(f"\n2. Or manually integrate:")
    print(f"   mv {output_dir / animal_name} data/fauna/large_scale/")
    print(f"   python run.py --config-name train_{animal_name}_debug")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SAM3D dataset for 3DAnimals")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive wizard mode")
    parser.add_argument("--source", type=str, help="Source directory (SAM3D output)")
    parser.add_argument("--animal", type=str, help="Animal name")
    parser.add_argument("--output", type=str, default="/home/joon/dev/3DAnimals/data/fauna_processed", help="Output directory")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    if args.interactive or not args.source:
        interactive_mode()
    else:
        if not args.animal:
            print("❌ --animal is required in manual mode")
            sys.exit(1)

        source_dir = Path(args.source)
        output_dir = Path(args.output)

        preprocessor = SAM3DDatasetPreprocessor()
        stats = preprocessor.process_dataset(
            source_dir,
            output_dir,
            args.animal,
            copy_files=args.copy,
            overwrite=args.overwrite
        )

        preprocessor.validate_output(output_dir, args.animal)


if __name__ == "__main__":
    main()
