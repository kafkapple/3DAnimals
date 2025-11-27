#!/usr/bin/env python3
"""
SAM3D to Fauna Dataset Preprocessing Pipeline
==============================================

Converts SAM3D GUI output datasets to Fauna-compatible format for 3DAnimals training.

Input Format (SAM3D GUI output):
  source/
  ├── dataset_metadata.json
  ├── video000/
  │   ├── frame_0000_rgb.png
  │   ├── frame_0000_mask.png
  │   └── ...
  └── video001/
      └── ...

Output Format (Fauna compatible):
  target/large_scale/{animal_name}/
  └── train/
      ├── seq_000/
      │   ├── 0000000_rgb.png
      │   ├── 0000000_mask.png
      │   ├── 0000000_box.txt
      │   └── 0000000_metadata.json
      └── seq_001/
          └── ...

Usage:
  # With config file
  python scripts/preprocess_sam3d_dataset.py --config config/preprocess/sam3d_to_fauna.yaml

  # Command line arguments
  python scripts/preprocess_sam3d_dataset.py \\
    --source /path/to/sam3d_output \\
    --target /path/to/fauna/large_scale \\
    --animal mouse_sam3d

  # Interactive mode
  python scripts/preprocess_sam3d_dataset.py --interactive
"""

import os
import sys
import re
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class PreprocessConfig:
    """Configuration for SAM3D to Fauna preprocessing."""
    # Input
    source_dir: str = ""
    rgb_pattern: str = r"frame_(\d+)_rgb\.png"
    mask_pattern: str = r"frame_(\d+)_mask\.png"
    folder_pattern: str = r"video(\d+)"

    # Output
    target_dir: str = ""
    animal_name: str = "mouse_sam3d"
    frame_id_digits: int = 7
    seq_prefix: str = "seq_"

    # Split ratios (train:val:test)
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    shuffle_sequences: bool = True
    random_seed: int = 42

    # Processing
    copy_files: bool = True
    overwrite: bool = False
    generate_box: bool = True
    generate_metadata: bool = True
    bbox_margin: int = 5
    skip_empty_masks: bool = False
    num_workers: int = 4

    # Validation
    validate_output: bool = True
    check_integrity: bool = False

    # Logging
    verbose: bool = True
    log_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PreprocessConfig":
        """Load config from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()
        # Flatten nested structure
        if 'input' in data:
            config.source_dir = data['input'].get('source_dir', config.source_dir)
            config.rgb_pattern = data['input'].get('rgb_pattern', config.rgb_pattern)
            config.mask_pattern = data['input'].get('mask_pattern', config.mask_pattern)
            config.folder_pattern = data['input'].get('folder_pattern', config.folder_pattern)

        if 'output' in data:
            config.target_dir = data['output'].get('target_dir', config.target_dir)
            config.animal_name = data['output'].get('animal_name', config.animal_name)
            config.frame_id_digits = data['output'].get('frame_id_digits', config.frame_id_digits)
            config.seq_prefix = data['output'].get('seq_prefix', config.seq_prefix)

        if 'processing' in data:
            config.copy_files = data['processing'].get('copy_files', config.copy_files)
            config.overwrite = data['processing'].get('overwrite', config.overwrite)
            config.generate_box = data['processing'].get('generate_box', config.generate_box)
            config.generate_metadata = data['processing'].get('generate_metadata', config.generate_metadata)
            config.bbox_margin = data['processing'].get('bbox_margin', config.bbox_margin)
            config.skip_empty_masks = data['processing'].get('skip_empty_masks', config.skip_empty_masks)
            config.num_workers = data['processing'].get('num_workers', config.num_workers)

        if 'validation' in data:
            config.validate_output = data['validation'].get('validate_output', config.validate_output)
            config.check_integrity = data['validation'].get('check_integrity', config.check_integrity)

        if 'logging' in data:
            config.verbose = data['logging'].get('verbose', config.verbose)
            config.log_file = data['logging'].get('log_file', config.log_file)

        return config


class SAM3DToFaunaPreprocessor:
    """Preprocessor for converting SAM3D GUI output to Fauna format."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.log_file = None
        if config.log_file:
            self.log_file = open(config.log_file, 'w')

    def __del__(self):
        if self.log_file:
            self.log_file.close()

    def log(self, message: str, level: str = "INFO"):
        """Log message with color coding."""
        if not self.config.verbose and level == "INFO":
            return

        colors = {
            "INFO": "\033[0;32m",
            "WARN": "\033[1;33m",
            "ERROR": "\033[0;31m",
            "DEBUG": "\033[0;36m"
        }
        color = colors.get(level, "")
        reset = "\033[0m"
        formatted = f"{color}[{level}]{reset} {message}"
        print(formatted)

        if self.log_file:
            self.log_file.write(f"[{level}] {message}\n")

    def analyze_source(self) -> Dict:
        """Analyze source dataset structure."""
        self.log("=" * 80)
        self.log("Analyzing Source Dataset")
        self.log("=" * 80)

        source_path = Path(self.config.source_dir)
        if not source_path.exists():
            self.log(f"Source not found: {source_path}", "ERROR")
            return {}

        info = {
            'sequences': [],
            'total_frames': 0,
            'image_size': None,
            'has_metadata': False
        }

        # Check for dataset_metadata.json
        metadata_file = source_path / "dataset_metadata.json"
        if metadata_file.exists():
            info['has_metadata'] = True
            with open(metadata_file, 'r') as f:
                info['dataset_metadata'] = json.load(f)

        # Find sequence folders
        folder_regex = re.compile(self.config.folder_pattern)
        folders = sorted([
            d for d in source_path.iterdir()
            if d.is_dir() and folder_regex.match(d.name)
        ])

        self.log(f"\nSource: {source_path}")
        self.log(f"Found {len(folders)} sequence folders")

        rgb_regex = re.compile(self.config.rgb_pattern)
        mask_regex = re.compile(self.config.mask_pattern)

        for folder in folders:
            # Find RGB files
            rgb_files = sorted([
                f for f in folder.iterdir()
                if f.is_file() and rgb_regex.match(f.name)
            ])

            # Find mask files
            mask_files = sorted([
                f for f in folder.iterdir()
                if f.is_file() and mask_regex.match(f.name)
            ])

            # Extract frame IDs
            frame_ids = []
            for f in rgb_files:
                match = rgb_regex.match(f.name)
                if match:
                    frame_ids.append(int(match.group(1)))

            # Get image size from first RGB
            if rgb_files and info['image_size'] is None:
                try:
                    img = Image.open(rgb_files[0])
                    info['image_size'] = img.size
                except Exception as e:
                    self.log(f"Error reading image: {e}", "WARN")

            seq_info = {
                'name': folder.name,
                'path': folder,
                'rgb_count': len(rgb_files),
                'mask_count': len(mask_files),
                'frame_ids': frame_ids
            }
            info['sequences'].append(seq_info)
            info['total_frames'] += len(rgb_files)

        # Print summary
        self.log(f"\nTotal frames: {info['total_frames']}")
        if info['image_size']:
            self.log(f"Image size: {info['image_size'][0]} x {info['image_size'][1]}")

        for seq in info['sequences'][:5]:  # Show first 5
            self.log(f"  {seq['name']}: {seq['rgb_count']} RGB, {seq['mask_count']} Mask")
        if len(info['sequences']) > 5:
            self.log(f"  ... and {len(info['sequences']) - 5} more sequences")

        return info

    def extract_bbox_from_mask(self, mask_path: Path) -> Tuple[int, int, int, int, bool]:
        """
        Extract bounding box from binary mask.

        Returns:
            (x0, y0, width, height, is_valid)
        """
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            h, w = mask.shape

            # Find non-zero pixels
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)

            if not rows.any() or not cols.any():
                # Empty mask
                return (w // 4, h // 4, w // 2, h // 2, False)

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Add margin
            margin = self.config.bbox_margin
            x0 = max(0, x_min - margin)
            y0 = max(0, y_min - margin)
            x1 = min(w, x_max + margin + 1)
            y1 = min(h, y_max + margin + 1)

            return (x0, y0, x1 - x0, y1 - y0, True)

        except Exception as e:
            self.log(f"Error extracting bbox from {mask_path}: {e}", "ERROR")
            return (0, 0, 256, 256, False)

    def generate_box_txt(
        self,
        mask_path: Path,
        output_path: Path,
        frame_id: int,
        image_size: Tuple[int, int]
    ) -> bool:
        """
        Generate box.txt file from mask.

        Format: frame_id crop_x0 crop_y0 crop_w crop_h full_w full_h sharpness label
        """
        x0, y0, w, h, is_valid = self.extract_bbox_from_mask(mask_path)
        full_w, full_h = image_size
        sharpness = 1.0
        label = 0

        box_data = f"{frame_id} {x0} {y0} {w} {h} {full_w} {full_h} {sharpness} {label}"

        with open(output_path, 'w') as f:
            f.write(box_data)

        return is_valid

    def generate_metadata_json(
        self,
        box_path: Path,
        output_path: Path,
        frame_id: int
    ):
        """Generate Fauna-compatible metadata.json from box.txt."""
        with open(box_path, 'r') as f:
            parts = f.read().strip().split()

        if len(parts) != 9:
            raise ValueError(f"Invalid box.txt format: {box_path}")

        _, x0, y0, w, h, full_w, full_h, sharpness, label = parts
        x0, y0, w, h = int(x0), int(y0), int(w), int(h)
        full_w, full_h = int(full_w), int(full_h)

        metadata = {
            "video_frame_id": frame_id,
            "crop_box_xyxy": [x0, y0, x0 + w, y0 + h],
            "video_frame_width": full_w,
            "video_frame_height": full_h,
            "crop_height": h,
            "crop_width": w
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def process_frame(
        self,
        src_rgb: Path,
        src_mask: Path,
        dst_dir: Path,
        src_frame_id: int,
        seq_idx: int,
        image_size: Tuple[int, int]
    ) -> Dict:
        """Process a single frame."""
        stats = {
            'copied': 0,
            'box_generated': 0,
            'meta_generated': 0,
            'skipped': False
        }

        # Calculate global frame ID (sequence_idx * 1000 + frame_idx)
        global_frame_id = seq_idx * 10000 + src_frame_id

        # Format output frame ID
        frame_id_str = str(global_frame_id).zfill(self.config.frame_id_digits)

        # Output paths
        dst_rgb = dst_dir / f"{frame_id_str}_rgb.png"
        dst_mask = dst_dir / f"{frame_id_str}_mask.png"
        dst_box = dst_dir / f"{frame_id_str}_box.txt"
        dst_meta = dst_dir / f"{frame_id_str}_metadata.json"

        # Skip if exists and not overwriting
        if not self.config.overwrite and all([
            dst_rgb.exists(), dst_mask.exists(),
            dst_box.exists(), dst_meta.exists()
        ]):
            stats['skipped'] = True
            return stats

        # Copy/symlink RGB
        if self.config.copy_files:
            shutil.copy2(src_rgb, dst_rgb)
        else:
            if dst_rgb.exists():
                dst_rgb.unlink()
            dst_rgb.symlink_to(src_rgb.resolve())
        stats['copied'] += 1

        # Copy/symlink Mask
        if self.config.copy_files:
            shutil.copy2(src_mask, dst_mask)
        else:
            if dst_mask.exists():
                dst_mask.unlink()
            dst_mask.symlink_to(src_mask.resolve())
        stats['copied'] += 1

        # Generate box.txt
        if self.config.generate_box:
            is_valid = self.generate_box_txt(src_mask, dst_box, global_frame_id, image_size)
            stats['box_generated'] += 1

            if not is_valid and self.config.skip_empty_masks:
                # Clean up
                dst_rgb.unlink()
                dst_mask.unlink()
                dst_box.unlink()
                stats['skipped'] = True
                return stats

        # Generate metadata.json
        if self.config.generate_metadata and dst_box.exists():
            self.generate_metadata_json(dst_box, dst_meta, global_frame_id)
            stats['meta_generated'] += 1

        return stats

    def split_sequences(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """Split sequences into train/val/test sets."""
        n_seqs = len(sequences)
        train_ratio, val_ratio, test_ratio = self.config.split_ratio

        # Shuffle if requested
        seq_indices = list(range(n_seqs))
        if self.config.shuffle_sequences:
            random.seed(self.config.random_seed)
            random.shuffle(seq_indices)

        # Calculate split points
        n_train = int(n_seqs * train_ratio)
        n_val = int(n_seqs * val_ratio)

        # Ensure at least 1 sequence per split if we have enough
        if n_seqs >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)

        train_indices = seq_indices[:n_train]
        val_indices = seq_indices[n_train:n_train + n_val]
        test_indices = seq_indices[n_train + n_val:]

        splits = {
            'train': [sequences[i] for i in train_indices],
            'val': [sequences[i] for i in val_indices],
            'test': [sequences[i] for i in test_indices]
        }

        return splits

    def process(self) -> Dict:
        """Process the entire dataset."""
        self.log("\n" + "=" * 80)
        self.log("Starting Preprocessing")
        self.log("=" * 80)

        # Analyze source
        info = self.analyze_source()
        if not info or info['total_frames'] == 0:
            self.log("No frames found to process", "ERROR")
            return {}

        image_size = info.get('image_size', (256, 256))

        # Create output directory
        target_path = Path(self.config.target_dir)
        animal_dir = target_path / self.config.animal_name

        # Split sequences
        splits = self.split_sequences(info['sequences'])
        self.log(f"\nSplit: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        total_stats = {
            'processed': 0,
            'copied': 0,
            'box_generated': 0,
            'meta_generated': 0,
            'skipped': 0,
            'errors': 0,
            'splits': {'train': 0, 'val': 0, 'test': 0}
        }

        rgb_regex = re.compile(self.config.rgb_pattern)
        global_seq_idx = 0

        # Process each split
        for split_name, split_sequences in splits.items():
            if not split_sequences:
                continue

            split_dir = animal_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            self.log(f"\n--- Processing {split_name} split ({len(split_sequences)} sequences) ---")

            for local_seq_idx, seq_info in enumerate(split_sequences):
                seq_name = f"{self.config.seq_prefix}{local_seq_idx:03d}"
                dst_seq_dir = split_dir / seq_name
                dst_seq_dir.mkdir(parents=True, exist_ok=True)

                src_seq_path = seq_info['path']

                self.log(f"\n[{split_name}] {seq_info['name']} -> {seq_name}")

                # Get RGB files
                rgb_files = sorted([
                    f for f in src_seq_path.iterdir()
                    if f.is_file() and rgb_regex.match(f.name)
                ])

                for rgb_file in tqdm(rgb_files, desc=f"  {seq_name}", disable=not self.config.verbose):
                    # Extract frame ID
                    match = rgb_regex.match(rgb_file.name)
                    if not match:
                        continue

                    frame_id = int(match.group(1))

                    # Find corresponding mask
                    mask_file = src_seq_path / rgb_file.name.replace("_rgb.png", "_mask.png")
                    if not mask_file.exists():
                        self.log(f"Mask not found: {mask_file}", "WARN")
                        total_stats['errors'] += 1
                        continue

                    try:
                        stats = self.process_frame(
                            rgb_file, mask_file, dst_seq_dir,
                            frame_id, global_seq_idx, image_size
                        )

                        if stats['skipped']:
                            total_stats['skipped'] += 1
                        else:
                            total_stats['processed'] += 1
                            total_stats['copied'] += stats['copied']
                            total_stats['box_generated'] += stats['box_generated']
                            total_stats['meta_generated'] += stats['meta_generated']
                            total_stats['splits'][split_name] += 1

                    except Exception as e:
                        self.log(f"Error processing {rgb_file}: {e}", "ERROR")
                        total_stats['errors'] += 1

                global_seq_idx += 1

        # Summary
        self.log("\n" + "=" * 80)
        self.log("Preprocessing Complete!")
        self.log("=" * 80)
        self.log(f"Processed:     {total_stats['processed']}")
        self.log(f"  - train:     {total_stats['splits']['train']}")
        self.log(f"  - val:       {total_stats['splits']['val']}")
        self.log(f"  - test:      {total_stats['splits']['test']}")
        self.log(f"Files copied:  {total_stats['copied']}")
        self.log(f"Box generated: {total_stats['box_generated']}")
        self.log(f"Meta generated:{total_stats['meta_generated']}")
        self.log(f"Skipped:       {total_stats['skipped']}")
        self.log(f"Errors:        {total_stats['errors']}")
        self.log(f"\nOutput: {animal_dir}")

        # Validate if requested
        if self.config.validate_output:
            self.validate(animal_dir)

        return total_stats

    def validate(self, animal_dir: Path) -> bool:
        """Validate processed dataset."""
        self.log("\n" + "=" * 80)
        self.log("Validating Output")
        self.log("=" * 80)

        all_valid = True
        grand_total = 0

        for split_name in ['train', 'val', 'test']:
            split_dir = animal_dir / split_name
            if not split_dir.exists():
                continue

            self.log(f"\n[{split_name}]")
            sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            split_frames = 0

            for seq in sequences:
                rgb_count = len(list(seq.glob("*_rgb.png")))
                mask_count = len(list(seq.glob("*_mask.png")))
                box_count = len(list(seq.glob("*_box.txt")))
                meta_count = len(list(seq.glob("*_metadata.json")))

                is_valid = (rgb_count == mask_count == box_count == meta_count)
                status = "OK" if is_valid else "MISMATCH"

                self.log(f"  {seq.name}: RGB={rgb_count}, Mask={mask_count}, Box={box_count}, Meta={meta_count} [{status}]")

                if not is_valid:
                    all_valid = False
                split_frames += rgb_count

            self.log(f"  Sequences: {len(sequences)}, Frames: {split_frames}")
            grand_total += split_frames

        self.log(f"\nTotal frames across all splits: {grand_total}")

        if all_valid:
            self.log("\nValidation PASSED", "INFO")
        else:
            self.log("\nValidation FAILED - some files missing", "WARN")

        return all_valid


def interactive_mode():
    """Interactive preprocessing wizard."""
    print("\n" + "=" * 80)
    print("SAM3D to Fauna Dataset Preprocessing Wizard")
    print("=" * 80)

    config = PreprocessConfig()

    # Source
    print("\n[1/5] Source Directory")
    default_src = "/home/joon/dev/sam3d_gui/outputs/fauna_datasets/mouse_batch_20251125_manual"
    src_input = input(f"Source [{default_src}]: ").strip()
    config.source_dir = src_input if src_input else default_src

    if not Path(config.source_dir).exists():
        print(f"Source not found: {config.source_dir}")
        return

    # Animal name
    print("\n[2/5] Animal Name")
    default_animal = "mouse_sam3d_manual"
    animal_input = input(f"Animal name [{default_animal}]: ").strip()
    config.animal_name = animal_input if animal_input else default_animal

    # Target
    print("\n[3/5] Target Directory")
    default_target = "/home/joon/dev/3DAnimals/data/fauna/large_scale"
    target_input = input(f"Target [{default_target}]: ").strip()
    config.target_dir = target_input if target_input else default_target

    # Copy or symlink
    print("\n[4/5] File Handling")
    print("  [1] Copy files (standalone)")
    print("  [2] Symlink (saves space)")
    method = input("Choice [1]: ").strip()
    config.copy_files = (method != "2")

    # Overwrite
    print("\n[5/5] Overwrite existing?")
    overwrite = input("Overwrite? (y/n) [n]: ").strip().lower()
    config.overwrite = (overwrite == 'y')

    # Summary
    print("\n" + "=" * 80)
    print("Configuration Summary")
    print("=" * 80)
    print(f"Source:     {config.source_dir}")
    print(f"Target:     {config.target_dir}/{config.animal_name}")
    print(f"Copy files: {config.copy_files}")
    print(f"Overwrite:  {config.overwrite}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # Process
    preprocessor = SAM3DToFaunaPreprocessor(config)
    preprocessor.process()

    # Next steps
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print(f"\n1. Create training config:")
    print(f"   # Edit config/dataset/mouse.yaml to point to new dataset")
    print(f"\n2. Run debug training:")
    print(f"   python run.py --config-name train_mouse_debug")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SAM3D GUI output to Fauna format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With config file
  python scripts/preprocess_sam3d_dataset.py --config config/preprocess/sam3d_to_fauna.yaml

  # Command line
  python scripts/preprocess_sam3d_dataset.py \\
    --source /path/to/sam3d_output \\
    --target /path/to/fauna/large_scale \\
    --animal mouse_sam3d

  # Interactive
  python scripts/preprocess_sam3d_dataset.py --interactive
        """
    )

    parser.add_argument("--config", "-c", type=str, help="YAML config file path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--source", "-s", type=str, help="Source directory (SAM3D output)")
    parser.add_argument("--target", "-t", type=str, help="Target directory (fauna/large_scale)")
    parser.add_argument("--animal", "-a", type=str, help="Animal/category name")
    parser.add_argument("--split", type=str, default="0.8:0.1:0.1",
                        help="Train:val:test split ratio (default: 0.8:0.1:0.1)")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle sequences before splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument("--copy", action="store_true", help="Copy files (default: symlink)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return

    # Config file mode
    if args.config:
        if not HAS_YAML:
            print("PyYAML required for config files. Install: pip install pyyaml")
            sys.exit(1)

        config = PreprocessConfig.from_yaml(args.config)

        # Override with command line args
        if args.source:
            config.source_dir = args.source
        if args.target:
            config.target_dir = args.target
        if args.animal:
            config.animal_name = args.animal
        if args.copy:
            config.copy_files = True
        if args.overwrite:
            config.overwrite = True
        config.verbose = args.verbose

    # Command line mode
    elif args.source and args.target and args.animal:
        # Parse split ratio
        split_parts = args.split.split(':')
        if len(split_parts) == 3:
            split_ratio = tuple(float(x) for x in split_parts)
        else:
            split_ratio = (0.8, 0.1, 0.1)

        config = PreprocessConfig(
            source_dir=args.source,
            target_dir=args.target,
            animal_name=args.animal,
            split_ratio=split_ratio,
            shuffle_sequences=not args.no_shuffle,
            random_seed=args.seed,
            copy_files=args.copy,
            overwrite=args.overwrite,
            verbose=args.verbose
        )
    else:
        # Default: interactive
        interactive_mode()
        return

    # Validate config
    if not config.source_dir or not config.target_dir or not config.animal_name:
        print("Error: source, target, and animal name required")
        parser.print_help()
        sys.exit(1)

    # Process
    preprocessor = SAM3DToFaunaPreprocessor(config)
    stats = preprocessor.process()

    if stats.get('errors', 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
