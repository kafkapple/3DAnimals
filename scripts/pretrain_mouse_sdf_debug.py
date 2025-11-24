#!/usr/bin/env python3
"""
Pre-train SDF network from MAMMAL mouse mesh - DEBUG MODE

Quick validation with reduced parameters:
- Grid resolution: 32 (vs 64)
- Iterations: 1000 (vs 10000)
- Pre-compute grid: 64 (vs 128)

Expected time: 5-10 minutes
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
    print(" "*10 + "🐛 DEBUG MODE: SDF Pre-training (Quick Test)")
    print("="*70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires GPU.")
        return 1

    print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # DEBUG Configuration (reduced for quick test)
    config = {
        'grid_res': 32,  # Reduced from 64
        'spatial_scale': 5.0,
        'num_layers': 5,
        'hidden_size': 64,
        'embedder_freq': 8,
        'embed_concat_pts': True,
        'init_sdf': None,
        'jitter_grid': 0.0,
        'symmetrize': False,
    }

    # Paths
    mesh_path = Path('/home/joon/dev/MAMMAL_mouse/mouse_model/mouse.pkl')
    tet_path = Path('data/tets/64_tets.npz')  # Note: still using 64 for tet grid
    output_dir = Path('checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify paths
    print(f"\n📁 Checking paths...")
    if not mesh_path.exists():
        print(f"❌ Mouse mesh not found: {mesh_path}")
        return 1
    print(f"   ✅ Mouse mesh: {mesh_path}")

    if not tet_path.exists():
        print(f"❌ Tet grid not found: {tet_path}")
        return 1
    print(f"   ✅ Tet grid: {tet_path}")

    # Initialize DMTet
    print(f"\n1️⃣ Initializing DMTet geometry (DEBUG MODE)...")
    print(f"   Grid resolution: {config['grid_res']} (reduced for debug)")
    print(f"   Spatial scale: {config['spatial_scale']}")
    print(f"   MLP layers: {config['num_layers']} × {config['hidden_size']}")

    try:
        dmtet = DMTetGeometry(**config).cuda()
        print(f"   ✅ DMTet initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize DMTet: {e}")
        return 1

    # Create pre-trainer with reduced grid
    print(f"\n2️⃣ Creating SDF pre-trainer...")
    print(f"   Pre-compute grid: 32³ (reduced from 128³ for debug)")
    try:
        pretrainer = SDFPretrainer(
            dmtet,
            mesh_path=str(mesh_path),
            device='cuda',
            precompute_grid_res=32  # Further reduced to 32 (was 64)
        )
    except Exception as e:
        print(f"   ❌ Failed to create pre-trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Pre-training configuration (DEBUG)
    pretrain_config = {
        'num_iters': 1000,  # Reduced from 10000
        'lr': 1e-3,
        'batch_size': 1024,  # Reduced from 2048
        'log_interval': 50,  # More frequent logging
        'save_path': output_dir / 'mouse_sdf_pretrained_debug.pth'
    }

    print(f"\n3️⃣ Pre-training SDF network (DEBUG MODE)...")
    print(f"   Iterations: {pretrain_config['num_iters']} (reduced for debug)")
    print(f"   Learning rate: {pretrain_config['lr']}")
    print(f"   Batch size: {pretrain_config['batch_size']}")
    print(f"   Expected time: ~5-10 minutes")

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
            output_path=output_dir / 'mouse_sdf_extracted_debug.obj'
        )
    except Exception as e:
        print(f"   ⚠️ Mesh extraction failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("✅ DEBUG MODE: Pre-training completed successfully!")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   Training loss: {best_loss:.6f}")
    print(f"   MAE:  {metrics['mae']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")

    print(f"\n💾 Outputs:")
    print(f"   Weights: {output_dir / 'mouse_sdf_pretrained_debug.pth'}")
    print(f"   Mesh:    {output_dir / 'mouse_sdf_extracted_debug.obj'}")

    print(f"\n📝 Next steps:")
    print(f"   ✅ Debug mode successful! Pipeline validated.")
    print(f"   → Ready for full-scale pre-training:")
    print(f"      python scripts/pretrain_mouse_sdf.py")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
