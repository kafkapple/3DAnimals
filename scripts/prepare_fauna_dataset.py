#!/usr/bin/env python3
"""
Fauna Dataset Preparation Script
=================================

Purpose:
  - Prepare DANNCE or markerless dataset for 3DAnimals training
  - Auto-detect dataset format and structure
  - Split train/val/test with various strategies
  - Generate appropriate config files
  - Validate dataset integrity

Usage:
  # Interactive mode (recommended)
  python scripts/prepare_fauna_dataset.py --interactive

  # Manual mode
  python scripts/prepare_fauna_dataset.py \\
    --source /path/to/data/mouse_dannce_6view/train \\
    --animal mouse \\
    --split-mode frame \\
    --ratio 0.7,0.15,0.15
"""

import os
import sys
import shutil
import argparse
import json
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import random


class FaunaDatasetPreparator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.data_dir = project_root / "data" / "fauna" / "large_scale"

    def detect_dataset_structure(self, source_dir: Path) -> Dict:
        """Detect dataset format and structure."""
        print("\n" + "=" * 80)
        print("Detecting Dataset Structure")
        print("=" * 80)

        info = {
            'format': 'unknown',
            'sequences': [],
            'total_frames': 0,
            'has_rgb': False,
            'has_mask': False,
            'has_box': False,
            'has_metadata': False,
        }

        # Find sequences (subdirectories)
        sequences = sorted([d for d in source_dir.iterdir() if d.is_dir()])

        if not sequences:
            print(f"❌ No sequences found in {source_dir}")
            return info

        info['sequences'] = sequences

        # Check first sequence
        first_seq = sequences[0]
        files = list(first_seq.glob("*"))

        # Check file types
        has_rgb = any(f.name.endswith(('_rgb.png', '_rgb.jpg')) for f in files)
        has_mask = any(f.name.endswith('_mask.png') for f in files)
        has_box = any(f.name.endswith('_box.txt') for f in files)
        has_metadata = any(f.name.endswith('_metadata.json') for f in files)

        info['has_rgb'] = has_rgb
        info['has_mask'] = has_mask
        info['has_box'] = has_box
        info['has_metadata'] = has_metadata

        # Determine format
        if has_rgb and has_mask and has_box:
            info['format'] = 'dannce' if has_metadata else 'markerless'
        elif has_rgb and has_mask:
            info['format'] = 'minimal'

        # Count total frames
        for seq in sequences:
            rgb_files = list(seq.glob("*_rgb.*"))
            info['total_frames'] += len(rgb_files)

        # Print detection result
        print(f"\n📁 Source: {source_dir}")
        print(f"🔍 Format: {info['format'].upper()}")
        print(f"📊 Sequences: {len(info['sequences'])}")
        print(f"🖼️  Total frames: {info['total_frames']}")
        print(f"\n✅ Required files:")
        print(f"  - RGB:      {'✓' if has_rgb else '✗'}")
        print(f"  - Mask:     {'✓' if has_mask else '✗'}")
        print(f"  - Box:      {'✓' if has_box else '✗'}")
        print(f"  - Metadata: {'✓' if has_metadata else '✗'}")

        return info

    def split_dataset(
        self,
        source_dir: Path,
        target_animal_dir: Path,
        mode: str = 'frame',
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42
    ) -> Dict:
        """Split dataset into train/val/test."""
        print("\n" + "=" * 80)
        print(f"Splitting Dataset (mode: {mode})")
        print("=" * 80)

        random.seed(seed)
        sequences = sorted([d for d in source_dir.iterdir() if d.is_dir()])

        train_ratio, val_ratio, test_ratio = ratios
        print(f"Ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")

        # Create target directories
        for split in ['train', 'val', 'test']:
            (target_animal_dir / split).mkdir(parents=True, exist_ok=True)

        stats = {'train': 0, 'val': 0, 'test': 0}

        if mode == 'sequence':
            # Split by entire sequences
            num_seqs = len(sequences)
            random.shuffle(sequences)

            train_end = max(1, int(num_seqs * train_ratio))
            val_end = max(train_end + 1, train_end + int(num_seqs * val_ratio))
            val_end = min(num_seqs - 1, val_end)

            split_seqs = {
                'train': sequences[:train_end],
                'val': sequences[train_end:val_end],
                'test': sequences[val_end:]
            }

            for split_name, split_seqs_list in split_seqs.items():
                for seq in split_seqs_list:
                    target_seq = target_animal_dir / split_name / seq.name
                    shutil.copytree(seq, target_seq, dirs_exist_ok=True)
                    frames = len(list(target_seq.glob("*_rgb.*")))
                    stats[split_name] += frames
                    print(f"  {split_name}/{seq.name}: {frames} frames")

        else:  # frame mode
            # Split by frames within each sequence
            for seq_idx, seq in enumerate(sequences):
                # Get all frame IDs
                rgb_files = sorted(seq.glob("*_rgb.*"))
                frame_ids = [f.stem.replace("_rgb", "") for f in rgb_files]
                random.shuffle(frame_ids)

                num_frames = len(frame_ids)
                train_end = int(num_frames * train_ratio)
                val_end = train_end + int(num_frames * val_ratio)

                frame_splits = {
                    'train': frame_ids[:train_end],
                    'val': frame_ids[train_end:val_end],
                    'test': frame_ids[val_end:]
                }

                for split_name, split_frames in frame_splits.items():
                    if not split_frames:
                        continue

                    target_seq = target_animal_dir / split_name / f"{seq_idx:06d}_00000"
                    target_seq.mkdir(parents=True, exist_ok=True)

                    for frame_id in split_frames:
                        # Copy all files for this frame
                        for ext in ['_rgb.png', '_rgb.jpg', '_mask.png', '_box.txt', '_metadata.json']:
                            src_file = seq / f"{frame_id}{ext}"
                            if src_file.exists():
                                shutil.copy2(src_file, target_seq / f"{frame_id}{ext}")

                    stats[split_name] += len(split_frames)

        print(f"\n📊 Split complete:")
        for split_name in ['train', 'val', 'test']:
            print(f"  {split_name.upper():5s}: {stats[split_name]} frames")

        return stats

    def generate_configs(
        self,
        animal_name: str,
        dataset_info: Dict,
        spatial_scale: float = None,
        num_body_bones: int = None
    ):
        """Generate train/debug/dataset/model config files."""
        print("\n" + "=" * 80)
        print("Generating Config Files")
        print("=" * 80)

        # Auto-detect parameters if not provided
        if spatial_scale is None:
            # Estimate based on animal name
            scale_map = {
                'mouse': 4.5,
                'rat': 5.0,
                'cat': 6.0,
                'dog': 7.0,
                'horse': 10.0,
                'elephant': 15.0
            }
            spatial_scale = scale_map.get(animal_name.lower(), 6.0)

        if num_body_bones is None:
            # Default: small=6, medium=8, large=10
            num_body_bones = 6 if spatial_scale < 6 else 8

        configs = {}

        # 1. Dataset config
        dataset_config = {
            'data_type': 'fauna',
            'in_image_size': 256,
            'out_image_size': 256,
            'batch_size': 1,
            'num_workers': 2,
            'train_data_dir': 'data/fauna',
            'val_data_dir': 'data/fauna',
            'test_data_dir': 'data/fauna',
            'random_shuffle_samples_train': False,
            'random_xflip_train': False,
            'background_mode': 'none',
            'load_flow': False,
            'load_dino_feature': False,
            'load_dino_cluster': False,
            'dino_feature_dim': 16
        }

        dataset_path = self.config_dir / "dataset" / f"{animal_name}.yaml"
        with open(dataset_path, 'w') as f:
            f.write(f"# {animal_name.capitalize()} Dataset Configuration\n")
            f.write(f"# Auto-generated by prepare_fauna_dataset.py\n")
            f.write(f"# Source: {dataset_info.get('source', 'N/A')}\n")
            f.write(f"# Format: {dataset_info['format']}\n")
            f.write(f"# Frames: {dataset_info['total_frames']}\n\n")
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

        configs['dataset'] = str(dataset_path)
        print(f"✅ Dataset config: {dataset_path}")

        # 2. Model config
        model_config = {
            'defaults': ['fauna'],
            'name': 'Fauna',
            'cfg_predictor_base': {
                'cfg_shape': {
                    'grid_res': 64,
                    'grid_res_coarse_iter_range': [0, 30000],
                    'grid_res_coarse': 32,
                    'spatial_scale': spatial_scale,
                    'num_layers': 5,
                    'hidden_size': 128,
                    'embedder_freq': 8,
                    'embed_concat_pts': True,
                    'init_sdf': 'ellipsoid',
                    'pretrained_sdf': None,
                    'jitter_grid': 0.05,
                    'symmetrize': True
                }
            },
            'cfg_predictor_instance': {
                'enable_articulation': True,
                'cfg_articulation': {
                    'articulation_iter_range': [10000, 'inf'],
                    'num_body_bones': num_body_bones,
                    'num_legs': 4,
                    'num_leg_bones': 3,
                    'attach_legs_to_body_iter_range': [30000, 'inf']
                },
                'enable_deform': False
            },
            'cfg_render': {
                'spatial_scale': spatial_scale
            }
        }

        model_path = self.config_dir / "model" / f"{animal_name}.yaml"
        with open(model_path, 'w') as f:
            f.write(f"# {animal_name.capitalize()} Model Configuration\n")
            f.write(f"# Auto-generated by prepare_fauna_dataset.py\n")
            f.write(f"# Spatial scale: {spatial_scale}\n")
            f.write(f"# Body bones: {num_body_bones}\n\n")
            yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

        configs['model'] = str(model_path)
        print(f"✅ Model config: {model_path}")

        # 3. Train config (main)
        train_config = {
            'defaults': [
                {'dataset': animal_name},
                {'model': animal_name},
                '_self_'
            ],
            'exp_name': f'{animal_name}_50k',
            'run_name': '${exp_name}_${now:%Y%m%d_%H%M%S}',
            'num_iters': 50000,
            'save_checkpoint_freq': 5000,
            'log_image_freq': 500,
            'device': 'cuda',
            'gpu_ids': [0],
            'disable_tf32': True,
            'wandb': {
                'project': f'{animal_name}_fauna',
                'mode': 'online',
                'tags': [animal_name, 'fauna', '50k']
            },
            'output_dir': 'results/${exp_name}',
            'resume': None,
            'seed': 42,
            'run_train': True,
            'run_test': False,
            'keep_num_checkpoint': 5
        }

        train_path = self.config_dir / f"train_{animal_name}.yaml"
        with open(train_path, 'w') as f:
            f.write(f"# {animal_name.capitalize()} Training Configuration\n")
            f.write(f"# Auto-generated by prepare_fauna_dataset.py\n")
            f.write(f"# Duration: ~2-3 hours (50K iterations)\n\n")
            yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)

        configs['train'] = str(train_path)
        print(f"✅ Train config: {train_path}")

        # 4. Debug config
        debug_config = train_config.copy()
        debug_config['exp_name'] = f'{animal_name}_debug'
        debug_config['num_iters'] = 5000
        debug_config['save_checkpoint_freq'] = 1000
        debug_config['log_image_freq'] = 200
        debug_config['wandb']['mode'] = 'offline'
        debug_config['wandb']['tags'] = [animal_name, 'debug']

        debug_path = self.config_dir / f"train_{animal_name}_debug.yaml"
        with open(debug_path, 'w') as f:
            f.write(f"# {animal_name.capitalize()} Debug Training Configuration\n")
            f.write(f"# Auto-generated by prepare_fauna_dataset.py\n")
            f.write(f"# Duration: ~15-20 minutes (5K iterations)\n\n")
            yaml.dump(debug_config, f, default_flow_style=False, sort_keys=False)

        configs['debug'] = str(debug_path)
        print(f"✅ Debug config: {debug_path}")

        return configs

    def validate_dataset(self, animal_dir: Path) -> bool:
        """Validate dataset integrity."""
        print("\n" + "=" * 80)
        print("Validating Dataset")
        print("=" * 80)

        all_good = True

        for split in ['train', 'val', 'test']:
            split_dir = animal_dir / split
            if not split_dir.exists():
                print(f"❌ {split}: directory not found")
                all_good = False
                continue

            sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            total_frames = 0

            for seq in sequences:
                rgb_files = list(seq.glob("*_rgb.*"))
                mask_files = list(seq.glob("*_mask.png"))

                if len(rgb_files) != len(mask_files):
                    print(f"⚠️  {split}/{seq.name}: RGB/Mask mismatch ({len(rgb_files)} vs {len(mask_files)})")
                    all_good = False

                total_frames += len(rgb_files)

            print(f"✅ {split.upper():5s}: {len(sequences)} sequences, {total_frames} frames")

        return all_good


def interactive_mode(preparator: FaunaDatasetPreparator):
    """Interactive setup wizard."""
    print("\n" + "=" * 80)
    print("Fauna Dataset Preparation Wizard")
    print("=" * 80)

    # Step 1: Source directory
    print("\n[Step 1] Source Directory")
    default_source = "/home/joon/dev/project_splatter/data/fauna_mouse/large_scale/mouse_dannce_6view/train"
    source_input = input(f"Source directory [{default_source}]: ").strip()
    source_dir = Path(source_input if source_input else default_source)

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return

    # Detect dataset
    dataset_info = preparator.detect_dataset_structure(source_dir)
    dataset_info['source'] = str(source_dir)

    if dataset_info['format'] == 'unknown':
        print("❌ Could not detect dataset format. Please check your data.")
        return

    # Step 2: Animal name
    print("\n[Step 2] Animal Name")
    animal_name = input("Animal name (e.g., mouse, cat, dog): ").strip().lower()

    if not animal_name:
        print("❌ Animal name is required")
        return

    # Step 3: Split mode
    print("\n[Step 3] Split Mode")
    print("  [1] frame - Split by frames (recommended for few sequences)")
    print("  [2] sequence - Split by sequences (recommended for many sequences)")
    split_mode_input = input("Choose mode [1]: ").strip()
    split_mode = 'sequence' if split_mode_input == '2' else 'frame'

    # Step 4: Split ratio
    print("\n[Step 4] Split Ratio")
    ratio_input = input("Train/Val/Test ratio [0.7,0.15,0.15]: ").strip()
    ratios = tuple(float(x) for x in ratio_input.split(',')) if ratio_input else (0.7, 0.15, 0.15)

    # Step 5: Animal parameters
    print("\n[Step 5] Model Parameters (optional, press Enter to auto-detect)")
    spatial_scale_input = input(f"Spatial scale [auto]: ").strip()
    spatial_scale = float(spatial_scale_input) if spatial_scale_input else None

    num_bones_input = input(f"Number of body bones [auto]: ").strip()
    num_body_bones = int(num_bones_input) if num_bones_input else None

    # Confirm
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Source:       {source_dir}")
    print(f"Animal:       {animal_name}")
    print(f"Format:       {dataset_info['format']}")
    print(f"Frames:       {dataset_info['total_frames']}")
    print(f"Split mode:   {split_mode}")
    print(f"Split ratio:  {ratios}")
    print(f"Spatial scale: {spatial_scale or 'auto'}")
    print(f"Body bones:   {num_body_bones or 'auto'}")
    print()

    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # Execute
    target_animal_dir = preparator.data_dir / animal_name
    stats = preparator.split_dataset(source_dir, target_animal_dir, split_mode, ratios)
    configs = preparator.generate_configs(animal_name, dataset_info, spatial_scale, num_body_bones)
    valid = preparator.validate_dataset(target_animal_dir)

    # Summary
    print("\n" + "=" * 80)
    print("✅ Dataset Preparation Complete!")
    print("=" * 80)
    print(f"\nDataset location:")
    print(f"  {target_animal_dir}")
    print(f"\nGenerated configs:")
    for config_type, config_path in configs.items():
        print(f"  {config_type:8s}: {config_path}")
    print(f"\nValidation: {'✅ PASSED' if valid else '⚠️  WARNINGS'}")
    print(f"\nNext steps:")
    print(f"  1. Review configs: config/train_{animal_name}*.yaml")
    print(f"  2. Run debug mode: ./scripts/train_{animal_name}.sh debug")
    print(f"  3. Run full training: ./scripts/train_{animal_name}.sh full")


def main():
    parser = argparse.ArgumentParser(description="Prepare Fauna dataset for 3DAnimals training")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive wizard mode")
    parser.add_argument("--source", type=str, help="Source directory containing sequences")
    parser.add_argument("--animal", type=str, help="Animal name (e.g., mouse, cat, dog)")
    parser.add_argument("--split-mode", type=str, choices=['sequence', 'frame'], default='frame', help="Split mode")
    parser.add_argument("--ratio", type=str, default="0.7,0.15,0.15", help="Train/Val/Test ratio")
    parser.add_argument("--spatial-scale", type=float, help="Model spatial scale (auto-detect if not specified)")
    parser.add_argument("--num-body-bones", type=int, help="Number of body bones (auto-detect if not specified)")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    preparator = FaunaDatasetPreparator(project_root)

    if args.interactive or not args.source:
        interactive_mode(preparator)
    else:
        # Manual mode
        if not args.animal:
            print("❌ --animal is required in manual mode")
            sys.exit(1)

        source_dir = Path(args.source)
        if not source_dir.exists():
            print(f"❌ Source directory not found: {source_dir}")
            sys.exit(1)

        dataset_info = preparator.detect_dataset_structure(source_dir)
        dataset_info['source'] = str(source_dir)

        target_animal_dir = preparator.data_dir / args.animal
        ratios = tuple(float(x) for x in args.ratio.split(','))

        stats = preparator.split_dataset(source_dir, target_animal_dir, args.split_mode, ratios)
        configs = preparator.generate_configs(args.animal, dataset_info, args.spatial_scale, args.num_body_bones)
        valid = preparator.validate_dataset(target_animal_dir)

        print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
