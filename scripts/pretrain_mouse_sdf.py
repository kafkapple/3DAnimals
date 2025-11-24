#!/usr/bin/env python3
"""
Pre-train SDF network from MAMMAL mouse mesh

Usage:
    python scripts/pretrain_mouse_sdf.py

Requirements:
    - MAMMAL mouse model at /home/joon/dev/MAMMAL_mouse/mouse_model/mouse.pkl
    - Tetrahedral grids at data/tets/64_tets.npz
    - GPU with CUDA support
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from model.geometry.dmtet import DMTetGeometry
from model.geometry.sdf_pretraining import SDFPretrainer


def main():
    print("="*70)
    print(" "*15 + "SDF Pre-training from MAMMAL Mouse Mesh")
    print("="*70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires GPU.")
        return 1

    print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Configuration
    config = {
        'grid_res': 64,
        'spatial_scale': 5.0,
        'num_layers': 5,
        'hidden_size': 64,
        'embedder_freq': 8,
        'embed_concat_pts': True,
        'init_sdf': None,  # No default initialization
        'jitter_grid': 0.0,
        'symmetrize': False,
    }

    # Paths
    mesh_path = Path('/home/joon/dev/MAMMAL_mouse/mouse_model/mouse.pkl')
    tet_path = Path('data/tets/64_tets.npz')
    output_dir = Path('checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify paths
    print(f"\n📁 Checking paths...")
    if not mesh_path.exists():
        print(f"❌ Mouse mesh not found: {mesh_path}")
        print("   Please ensure MAMMAL_mouse project is available")
        return 1
    print(f"   ✅ Mouse mesh: {mesh_path}")

    if not tet_path.exists():
        print(f"❌ Tet grid not found: {tet_path}")
        print("   Please download tetrahedral grids:")
        print("   cd data/tets && sh download_tets.sh")
        return 1
    print(f"   ✅ Tet grid: {tet_path}")

    # Initialize DMTet
    print(f"\n1️⃣ Initializing DMTet geometry...")
    print(f"   Grid resolution: {config['grid_res']}")
    print(f"   Spatial scale: {config['spatial_scale']}")
    print(f"   MLP layers: {config['num_layers']} × {config['hidden_size']}")

    try:
        dmtet = DMTetGeometry(**config).cuda()
        print(f"   ✅ DMTet initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize DMTet: {e}")
        return 1

    # Create pre-trainer
    print(f"\n2️⃣ Creating SDF pre-trainer...")
    try:
        pretrainer = SDFPretrainer(
            dmtet,
            mesh_path=str(mesh_path),
            device='cuda'
        )
    except Exception as e:
        print(f"   ❌ Failed to create pre-trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Pre-training configuration
    pretrain_config = {
        'num_iters': 10000,
        'lr': 1e-3,
        'batch_size': 2048,
        'log_interval': 100,
        'save_path': output_dir / 'mouse_sdf_pretrained.pth'
    }

    print(f"\n3️⃣ Pre-training SDF network...")
    print(f"   Iterations: {pretrain_config['num_iters']}")
    print(f"   Learning rate: {pretrain_config['lr']}")
    print(f"   Batch size: {pretrain_config['batch_size']}")

    try:
        best_loss = pretrainer.pretrain(**pretrain_config)
    except Exception as e:
        print(f"   ❌ Pre-training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Evaluate
    print(f"\n4️⃣ Evaluating pre-trained SDF...")
    try:
        metrics = pretrainer.evaluate(grid_res=config['grid_res'])
    except Exception as e:
        print(f"   ⚠️ Evaluation failed: {e}")
        metrics = {'mae': float('nan'), 'rmse': float('nan')}

    # Extract mesh
    print(f"\n5️⃣ Extracting mesh for visualization...")
    try:
        pretrainer.visualize_mesh(
            output_path=output_dir / 'mouse_sdf_extracted.obj'
        )
    except Exception as e:
        print(f"   ⚠️ Mesh extraction failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("✅ Pre-training pipeline completed successfully!")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   Training loss: {best_loss:.6f}")
    print(f"   MAE:  {metrics['mae']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")

    print(f"\n💾 Outputs:")
    print(f"   Weights: {output_dir / 'mouse_sdf_pretrained.pth'}")
    print(f"   Mesh:    {output_dir / 'mouse_sdf_extracted.obj'}")

    print(f"\n📝 Next steps:")
    print(f"   1. Review extracted mesh: {output_dir / 'mouse_sdf_extracted.obj'}")
    print(f"   2. Update training config to load pre-trained weights:")
    print(f"      model.cfg_predictor_base.cfg_shape.pretrained_sdf: 'checkpoints/mouse_sdf_pretrained.pth'")
    print(f"   3. Start training: python run.py --config-name train_fauna_mouse")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
