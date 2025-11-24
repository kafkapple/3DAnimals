#!/usr/bin/env python3
"""
MAMMAL Mouse Mesh → SDF MLP Training Script

Converts MAMMAL parametric mouse model to SDF representation
and trains Fauna's MLP to represent it.

Usage:
    python scripts/train_mammal_sdf_mlp.py

Output:
    results/mammal_mouse_sdf_mlp.pth  (trained MLP weights)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Set up EGL for headless rendering (before importing pyrender)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import torch.nn.functional as F
import numpy as np
import trimesh
from tqdm import tqdm
import time

# Fauna's CoordMLP
from model.networks import CoordMLP


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


def compute_sdf_samples_sparse(mesh, n_surface=80000, n_random=40000,
                                spatial_scale=5.0, device='cuda'):
    """
    Sparse SDF sampling (memory efficient)

    Args:
        n_surface: Number of points to sample on surface (with perturbation)
        n_random: Number of random points in volume
        spatial_scale: Coordinate system scale
        device: torch device
    """
    print(f"Sparse SDF sampling: {n_surface} surface + {n_random} random")
    start_time = time.time()

    # 1. Surface points (trimesh.sample provides this)
    print("  Sampling points on surface...")
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
    surface_pts = torch.from_numpy(surface_pts).float()

    # 2. Near-surface perturbation (SDF detail around surface)
    print("  Adding near-surface perturbations...")
    noise = torch.randn_like(surface_pts) * (spatial_scale * 0.05)  # ±5% noise
    near_surface_pts = surface_pts + noise

    # 3. Random volume points (global coverage)
    print("  Sampling random volume points...")
    half_scale = spatial_scale / 2
    random_pts = torch.rand(n_random, 3) * spatial_scale - half_scale

    # 4. Combine all points
    all_pts = torch.cat([near_surface_pts, random_pts], dim=0)  # (120K, 3)
    print(f"Total points: {len(all_pts):,}")

    # 5. Compute SDF (batched to avoid memory issues)
    print("  Computing signed distances (batched)...")
    batch_size = 10000  # 10K points per batch (ultra-aggressive for RAM)
    sdf_values = []

    for i in tqdm(range(0, len(all_pts), batch_size), desc="SDF batches"):
        batch = all_pts[i:i+batch_size].cpu().numpy()

        # Nearest distance (unsigned)
        _, distances, _ = mesh.nearest.on_surface(batch)

        # Sign (inside/outside) via ray casting
        is_inside = mesh.contains(batch)

        # Signed distance
        sdf_batch = torch.from_numpy(distances).float()
        sdf_batch[torch.from_numpy(is_inside)] *= -1

        sdf_values.append(sdf_batch)

    sdf_values = torch.cat(sdf_values, dim=0).unsqueeze(-1)

    elapsed = time.time() - start_time
    print(f"SDF computed in {elapsed/60:.1f} minutes")

    # Statistics
    print(f"SDF stats:")
    print(f"  Min: {sdf_values.min().item():.4f}")
    print(f"  Max: {sdf_values.max().item():.4f}")
    print(f"  Mean: {sdf_values.mean().item():.4f}")
    print(f"  Median: {sdf_values.median().item():.4f}")

    return all_pts.to(device), sdf_values.to(device)


def train_sdf_mlp(grid_points, sdf_values, spatial_scale=5.0, device='cuda'):
    """Train MLP to fit SDF samples"""
    print("\nInitializing MLP...")

    # Match Fauna's architecture exactly
    embedder_scalar = 2 * np.pi / spatial_scale * 0.9

    mlp = CoordMLP(
        cin=3,  # input channels (x, y, z coordinates)
        cout=1,  # output channels (SDF value)
        num_layers=5,
        nf=64,  # hidden_size
        dropout=0,
        activation=None,
        min_max=None,
        n_harmonic_functions=8,
        embedder_scalar=embedder_scalar,
        embed_concat_pts=True
    ).to(device)

    print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    # Training config
    num_iters = 50000
    batch_size = 8192
    log_freq = 500
    save_freq = 5000

    print(f"\nTraining for {num_iters:,} iterations...")
    print(f"Batch size: {batch_size:,}")
    print(f"Estimated time: 2-3 hours\n")

    best_loss = float('inf')

    for iter in tqdm(range(num_iters), desc="Training MLP"):
        # Random batch sampling
        idx = torch.randperm(len(grid_points), device=device)[:batch_size]
        pts = grid_points[idx]
        gt_sdf = sdf_values[idx]

        # Forward
        pred_sdf = mlp(pts)
        loss = F.mse_loss(pred_sdf, gt_sdf)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if (iter + 1) % log_freq == 0:
            tqdm.write(f"Iter {iter+1:6d}: Loss {loss.item():.6f} | LR {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (iter + 1) % save_freq == 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                os.makedirs('results', exist_ok=True)
                torch.save({
                    'iter': iter + 1,
                    'model_state_dict': mlp.state_dict(),
                    'loss': loss.item(),
                }, f'results/mammal_mouse_sdf_mlp_iter{iter+1}.pth')
                tqdm.write(f"  → Saved checkpoint (loss: {loss.item():.6f})")

    # Final save
    print(f"\nTraining complete! Final loss: {loss.item():.6f}")
    torch.save(mlp.state_dict(), 'results/mammal_mouse_sdf_mlp.pth')
    print("Saved final weights to: results/mammal_mouse_sdf_mlp.pth")

    return mlp


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        print("WARNING: Running on CPU will be VERY slow (20-30x slower)")
        print("Please ensure CUDA is available!")
        return

    # Config
    mesh_path = "/home/joon/dev/MAMMAL_mouse/mouse_model/mouse_reduced_face_3600.obj"
    spatial_scale = 5.0  # Match Fauna config
    n_surface = 30000    # Surface sampling (ultra-aggressive for RAM constraint)
    n_random = 15000     # Volume sampling (ultra-aggressive)

    # Step 1: Load mesh
    print("="*70)
    print("STEP 1/3: Loading MAMMAL Mouse Mesh")
    print("="*70)
    mesh, vertices = load_and_normalize_mesh(mesh_path, target_scale=spatial_scale)

    # Step 2: Compute SDF (Sparse sampling to avoid memory issues)
    print("\n" + "="*70)
    print("STEP 2/3: Computing SDF Samples (Sparse)")
    print("="*70)
    grid_points, sdf_values = compute_sdf_samples_sparse(
        mesh,
        n_surface=n_surface,
        n_random=n_random,
        spatial_scale=spatial_scale,
        device=device
    )

    # Step 3: Train MLP
    print("\n" + "="*70)
    print("STEP 3/3: Training SDF MLP")
    print("="*70)
    mlp = train_sdf_mlp(grid_points, sdf_values, spatial_scale=spatial_scale, device=device)

    print("\n" + "="*70)
    print("MAMMAL → SDF MLP Training Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python scripts/integrate_mammal_checkpoint.py")
    print("2. Update config: checkpoint_path = 'results/fauna_mouse_mammal_init.pth'")
    print("3. Start Fauna training with MAMMAL prior!")


if __name__ == '__main__':
    main()
