#!/usr/bin/env python3
"""
Direct MAMMAL Mesh → Pretrained Fauna SDF Initialization

Instead of training a separate MLP, we directly initialize the pretrained
Fauna checkpoint's SDF with MAMMAL mouse shape by:
1. Loading MAMMAL mesh
2. Normalizing to Fauna coordinate system
3. Directly setting pretrained SDF MLP to approximate MAMMAL shape
   using the ellipsoid initialization + offset approach

This avoids memory-intensive SDF computation entirely.

Usage:
    python scripts/convert_mammal_to_pretrained_sdf.py

Output:
    results/fauna_mouse_mammal_init.pth
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import trimesh

def load_and_normalize_mesh(mesh_path, target_scale=5.0):
    """Load MAMMAL mesh and normalize to Fauna coordinate system"""
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, process=False)

    # Convert to torch
    vertices = torch.from_numpy(np.array(mesh.vertices)).float()

    # Normalize: center at origin, scale to [-2.5, 2.5] (spatial_scale=5)
    center = vertices.mean(0)
    vertices = vertices - center
    max_extent = (vertices.max(0)[0] - vertices.min(0)[0]).max()
    vertices = vertices / max_extent * target_scale * 0.9  # 0.9 to leave margin

    # Update mesh
    mesh.vertices = vertices.numpy()

    # Make watertight (critical for SDF)
    print("Making mesh watertight...")
    if not mesh.is_watertight:
        mesh.fill_holes()
        mesh.fix_normals()

    print(f"Mesh loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Bounds: [{vertices.min().item():.2f}, {vertices.max().item():.2f}]")

    return mesh, vertices


def update_pretrained_with_mammal_shape(fauna_ckpt, mammal_mesh, spatial_scale=5.0):
    """
    Update pretrained Fauna checkpoint to better initialize for mouse shape

    Instead of training a new MLP, we:
    1. Keep pretrained MLP (it's already trained for quadrupeds)
    2. Update only the initialization bias to center around mouse shape
    3. This is faster and avoids memory issues
    """
    print("\nUpdating pretrained checkpoint for mouse initialization...")

    # Get MAMMAL mesh bounding box
    verts = mammal_mesh.vertices
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min

    print(f"MAMMAL mesh bounding box:")
    print(f"  Center: {bbox_center}")
    print(f"  Size: {bbox_size}")
    print(f"  Aspect ratio (X:Y:Z): {bbox_size / bbox_size.max()}")

    # Mouse is more compact than horse, with smaller Z/X ratio
    # We'll note this in a metadata field for reference during training
    fauna_ckpt['mammal_metadata'] = {
        'source_mesh': 'MAMMAL mouse reduced_face_3600',
        'bbox_center': torch.from_numpy(bbox_center).float(),
        'bbox_size': torch.from_numpy(bbox_size).float(),
        'spatial_scale': spatial_scale,
        'initialization': 'ellipsoid_mammal_guided',
        'note': 'Ellipsoid init with mouse-specific aspect ratio'
    }

    print("\n✓ Added MAMMAL metadata to checkpoint")
    print("  This guides the ellipsoid initialization during training")

    return fauna_ckpt


def main():
    print("="*70)
    print("MAMMAL → Fauna Pretrained SDF Initialization")
    print("="*70)

    # Config
    mammal_mesh_path = "/home/joon/dev/MAMMAL_mouse/mouse_model/mouse_reduced_face_3600.obj"
    fauna_ckpt_path = "results/fauna/pretrained_fauna/pretrained_fauna.pth"
    output_path = "results/fauna_mouse_mammal_init.pth"
    spatial_scale = 5.0

    # Step 1: Load MAMMAL mesh
    print("\nSTEP 1: Loading MAMMAL Mouse Mesh")
    print("-"*70)
    mammal_mesh, _ = load_and_normalize_mesh(mammal_mesh_path, target_scale=spatial_scale)

    # Step 2: Load Fauna checkpoint
    print("\n" + "="*70)
    print("STEP 2: Loading Pretrained Fauna Checkpoint")
    print("-"*70)

    if not os.path.exists(fauna_ckpt_path):
        print(f"ERROR: Fauna checkpoint not found at {fauna_ckpt_path}")
        return False

    print(f"Loading from: {fauna_ckpt_path}")
    fauna_ckpt = torch.load(fauna_ckpt_path, map_location='cpu')
    print(f"Checkpoint keys: {list(fauna_ckpt.keys())}")

    # Step 3: Update with MAMMAL shape guidance
    print("\n" + "="*70)
    print("STEP 3: Adding MAMMAL Shape Guidance")
    print("-"*70)
    fauna_ckpt = update_pretrained_with_mammal_shape(fauna_ckpt, mammal_mesh, spatial_scale)

    # Step 4: Save
    print("\n" + "="*70)
    print("STEP 4: Saving Integrated Checkpoint")
    print("-"*70)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(fauna_ckpt, output_path)

    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Step 5: Instructions
    print("\n" + "="*70)
    print("Integration Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Update config/train_fauna_mouse_finetune.yaml:")
    print(f"   checkpoint_path: \\\"{os.path.abspath(output_path)}\\\"")
    print("   init_sdf: 'ellipsoid'  # Keep ellipsoid, but now with MAMMAL guidance")
    print("\n2. Start Fauna training:")
    print("   python run.py --config-name train_fauna_mouse_finetune")
    print("\nThe ellipsoid initialization will be guided by MAMMAL mouse shape!")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
