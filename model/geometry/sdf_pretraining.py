"""
SDF Pre-training from Explicit Mesh

Pre-train DMTet SDF network from explicit articulated mesh (e.g., MAMMAL mouse model)
to provide better initialization than random or simple geometric shapes (sphere/ellipsoid).

Author: Claude Code
Date: 2025-11-10
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from tqdm import tqdm
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️ trimesh not installed. Install with: pip install trimesh")


class SDFPretrainer:
    """Pre-train DMTet SDF network from explicit mesh"""

    def __init__(self, dmtet_geometry, mesh_path, device='cuda', precompute_grid_res=128):
        """
        Args:
            dmtet_geometry: DMTetGeometry instance to pre-train
            mesh_path: Path to mesh file (.pkl or .obj)
            device: Device for training
            precompute_grid_res: Resolution for pre-computed SDF grid (default: 128)
        """
        self.geometry = dmtet_geometry
        self.device = device

        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for SDF pre-training. Install with: pip install trimesh")

        # Load mesh
        self.mesh = self._load_mesh(mesh_path)

        # Get mesh bounds for sampling
        self.bbox_min = torch.FloatTensor(self.mesh.bounds[0]).to(device)
        self.bbox_max = torch.FloatTensor(self.mesh.bounds[1]).to(device)
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2
        self.bbox_size = (self.bbox_max - self.bbox_min).max()

        print(f"✅ Loaded mesh: {self.mesh.vertices.shape[0]} vertices, "
              f"{self.mesh.faces.shape[0]} faces")
        print(f"   Bounding box: [{self.bbox_min.cpu().numpy()}] to [{self.bbox_max.cpu().numpy()}]")
        print(f"   Center: {self.bbox_center.cpu().numpy()}, Size: {self.bbox_size.item():.3f}")

        # Pre-compute SDF on grid (MEMORY OPTIMIZATION)
        print(f"\n🔄 Pre-computing SDF grid (one-time, resolution: {precompute_grid_res}³)...")
        self._precompute_sdf_grid(grid_res=precompute_grid_res)

    def _load_mesh(self, mesh_path):
        """Load mesh from file"""
        mesh_path = Path(mesh_path)

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        if mesh_path.suffix == '.pkl':
            # MAMMAL format
            print(f"Loading MAMMAL format mesh from {mesh_path}")
            with open(mesh_path, 'rb') as f:
                data = pickle.load(f)

            if 'vertices' not in data or 'faces_vert' not in data:
                raise ValueError("Invalid MAMMAL mesh file. Must contain 'vertices' and 'faces_vert'")

            mesh = trimesh.Trimesh(
                vertices=data['vertices'],
                faces=data['faces_vert']
            )
        elif mesh_path.suffix in ['.obj', '.ply', '.stl', '.off']:
            # Standard mesh formats
            print(f"Loading mesh from {mesh_path}")
            mesh = trimesh.load(mesh_path)
        else:
            raise ValueError(f"Unsupported mesh format: {mesh_path.suffix}")

        # Ensure watertight for accurate SDF
        if not mesh.is_watertight:
            print("⚠️ Mesh is not watertight. SDF may be inaccurate.")
            print("   Attempting to fix...")
            mesh.fill_holes()
            if mesh.is_watertight:
                print("   ✅ Mesh fixed")
            else:
                print("   ⚠️ Could not fix mesh. Proceeding anyway...")

        return mesh

    def _precompute_sdf_grid(self, grid_res):
        """
        Pre-compute SDF values on a regular 3D grid (ONE-TIME)

        This dramatically reduces memory usage and speeds up training by avoiding
        repeated trimesh.proximity.signed_distance() calls during training loop.

        Args:
            grid_res: Grid resolution (e.g., 128 → 128³ = 2M points)
        """
        bbox_min = self.bbox_min.cpu().numpy()
        bbox_max = self.bbox_max.cpu().numpy()

        # Create 3D grid
        x = np.linspace(bbox_min[0], bbox_max[0], grid_res)
        y = np.linspace(bbox_min[1], bbox_max[1], grid_res)
        z = np.linspace(bbox_min[2], bbox_max[2], grid_res)

        grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        grid_points = grid_points.reshape(-1, 3)  # (grid_res³, 3)

        total_points = len(grid_points)
        print(f"   Total grid points: {total_points:,} ({grid_res}³)")

        # Batch processing to avoid OOM during pre-computation
        batch_size = 10000  # Reduced to 10K per batch (was 100K)
        sdf_values = []

        num_batches = (total_points + batch_size - 1) // batch_size
        print(f"   Processing in {num_batches} batches of {batch_size:,} points...")

        for i in tqdm(range(0, total_points, batch_size), desc="Computing SDF grid"):
            batch = grid_points[i:i+batch_size]
            sdf_batch = trimesh.proximity.signed_distance(self.mesh, batch)
            sdf_values.append(sdf_batch)

        # Combine and convert to tensors
        self.sdf_grid = np.concatenate(sdf_values).astype(np.float32)
        self.grid_points = grid_points.astype(np.float32)

        # Store on GPU for fast sampling
        self.sdf_grid_tensor = torch.FloatTensor(self.sdf_grid).to(self.device)
        self.grid_points_tensor = torch.FloatTensor(self.grid_points).to(self.device)

        print(f"   ✅ SDF grid pre-computed: {len(self.sdf_grid):,} points")
        print(f"      Memory: SDF {self.sdf_grid.nbytes / 1e6:.1f} MB + Points {self.grid_points.nbytes / 1e6:.1f} MB")
        print(f"      SDF range: [{self.sdf_grid.min():.3f}, {self.sdf_grid.max():.3f}]")

    def pretrain(self, num_iters=10000, lr=1e-3, batch_size=2048,
                 log_interval=100, save_path=None):
        """
        Pre-train SDF network to match input mesh

        Args:
            num_iters: Number of training iterations
            lr: Learning rate
            batch_size: Batch size for training
            log_interval: Logging interval
            save_path: Path to save best weights (optional)
        """
        print(f"\n🚀 Starting SDF pre-training...")
        print(f"   Iterations: {num_iters}")
        print(f"   Learning rate: {lr}")
        print(f"   Batch size: {batch_size}")

        # Optimizer
        optimizer = torch.optim.Adam(self.geometry.mlp.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

        best_loss = float('inf')
        best_state = None

        # Training loop
        pbar = tqdm(range(num_iters), desc="Pre-training SDF")
        for iter in pbar:
            # 1. Sample training points
            points, target_sdf = self.sample_training_batch(batch_size)

            # 2. Predict SDF
            pred_sdf = self.geometry.get_sdf(points)

            # 3. Compute loss
            loss = F.l1_loss(pred_sdf.squeeze(-1), target_sdf)

            # 4. Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.geometry.mlp.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # 5. Logging
            if iter % log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # 6. Save best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.geometry.state_dict().items()}

        # Load best weights
        self.geometry.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        print(f"\n✅ Pre-training complete. Best loss: {best_loss:.6f}")

        # Save if path provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_path)
            print(f"💾 Saved pre-trained weights to {save_path}")

        return best_loss

    def sample_training_batch(self, batch_size):
        """
        Sample training points from pre-computed SDF grid (FAST & MEMORY EFFICIENT)

        Strategy:
        - 25% on surface (SDF ≈ 0) - sampled from mesh
        - 75% from pre-computed grid - random sampling

        NO trimesh.proximity.signed_distance() calls during training!
        """
        n_surface = batch_size // 4
        n_grid = batch_size - n_surface

        # 1. On-surface points (SDF ≈ 0) - still use mesh sampling for accuracy
        surface_points, _ = trimesh.sample.sample_surface(self.mesh, n_surface)
        surface_sdf = np.zeros(n_surface)

        # 2. Sample from pre-computed grid (NO SDF computation!)
        total_grid_points = len(self.grid_points_tensor)
        indices = torch.randint(0, total_grid_points, (n_grid,), device=self.device)

        grid_sampled_points = self.grid_points_tensor[indices]
        grid_sampled_sdf = self.sdf_grid_tensor[indices]

        # 3. Combine (convert surface points to tensor)
        surface_points_tensor = torch.FloatTensor(surface_points).to(self.device)
        surface_sdf_tensor = torch.FloatTensor(surface_sdf).to(self.device)

        all_points = torch.cat([surface_points_tensor, grid_sampled_points], dim=0)
        all_sdf = torch.cat([surface_sdf_tensor, grid_sampled_sdf], dim=0)

        return all_points, all_sdf

    def evaluate(self, grid_res=64, verbose=True):
        """
        Evaluate SDF accuracy on grid

        Args:
            grid_res: Grid resolution for evaluation
            verbose: Print results

        Returns:
            dict: Evaluation metrics (mae, rmse)
        """
        print(f"\n📊 Evaluating pre-trained SDF (grid_res={grid_res})...")

        # Get grid points
        tets_path = f'data/tets/{grid_res}_tets.npz'
        if not os.path.exists(tets_path):
            print(f"⚠️ Tet grid not found: {tets_path}")
            print("   Skipping evaluation")
            return {'mae': float('nan'), 'rmse': float('nan')}

        tets = np.load(tets_path)
        grid_points = tets['vertices'] * self.geometry.grid_scale

        # Ground truth SDF
        print("   Computing ground truth SDF...")
        gt_sdf = trimesh.proximity.signed_distance(self.mesh, grid_points)

        # Predicted SDF
        print("   Computing predicted SDF...")
        with torch.no_grad():
            grid_points_tensor = torch.FloatTensor(grid_points).to(self.device)
            pred_sdf = self.geometry.get_sdf(grid_points_tensor)
            pred_sdf = pred_sdf.cpu().numpy().squeeze()

        # Metrics
        mae = np.abs(gt_sdf - pred_sdf).mean()
        rmse = np.sqrt(((gt_sdf - pred_sdf) ** 2).mean())

        if verbose:
            print(f"\n   Results:")
            print(f"   MAE:  {mae:.6f}")
            print(f"   RMSE: {rmse:.6f}")
            print(f"   Max error: {np.abs(gt_sdf - pred_sdf).max():.6f}")

        return {'mae': mae, 'rmse': rmse}

    def visualize_mesh(self, output_path='output_mesh.obj'):
        """
        Extract and save mesh from pre-trained SDF

        Args:
            output_path: Path to save extracted mesh
        """
        print(f"\n🎨 Extracting mesh from pre-trained SDF...")

        with torch.no_grad():
            mesh = self.geometry.getMesh()

        # Save mesh
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy
        vertices = mesh.v_pos.cpu().numpy()
        faces = mesh.t_pos_idx.cpu().numpy()

        # Save as OBJ
        extracted_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        extracted_mesh.export(output_path)

        print(f"💾 Saved extracted mesh to {output_path}")
        print(f"   Vertices: {len(vertices)}, Faces: {len(faces)}")

        return extracted_mesh


def main():
    """Example usage"""
    import sys
    sys.path.append('/home/joon/dev/3DAnimals')

    from model.geometry.dmtet import DMTetGeometry

    print("="*60)
    print("SDF Pre-training from MAMMAL Mouse Mesh")
    print("="*60)

    # Configuration
    config = {
        'grid_res': 64,
        'spatial_scale': 5.0,
        'num_layers': 5,
        'hidden_size': 64,
        'embedder_freq': 8,
        'embed_concat_pts': True,
        'init_sdf': None,  # No default initialization
    }

    mesh_path = '/home/joon/dev/MAMMAL_mouse/mouse_model/mouse.pkl'
    output_dir = Path('/home/joon/dev/3DAnimals/checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DMTet
    print("\n1️⃣ Initializing DMTet geometry...")
    dmtet = DMTetGeometry(**config).cuda()

    # Create pre-trainer
    print("\n2️⃣ Creating SDF pre-trainer...")
    pretrainer = SDFPretrainer(
        dmtet,
        mesh_path=mesh_path,
        device='cuda'
    )

    # Pre-train
    print("\n3️⃣ Pre-training SDF network...")
    best_loss = pretrainer.pretrain(
        num_iters=10000,
        lr=1e-3,
        batch_size=2048,
        log_interval=100,
        save_path=output_dir / 'mouse_sdf_pretrained.pth'
    )

    # Evaluate
    print("\n4️⃣ Evaluating pre-trained SDF...")
    metrics = pretrainer.evaluate(grid_res=64)

    # Visualize
    print("\n5️⃣ Extracting mesh...")
    pretrainer.visualize_mesh(
        output_path=output_dir / 'mouse_sdf_extracted.obj'
    )

    print("\n" + "="*60)
    print("✅ Pre-training pipeline completed successfully!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Weights: {output_dir / 'mouse_sdf_pretrained.pth'}")
    print(f"  - Mesh:    {output_dir / 'mouse_sdf_extracted.obj'}")
    print(f"\nMetrics:")
    print(f"  - Training loss: {best_loss:.6f}")
    print(f"  - MAE:  {metrics['mae']:.6f}")
    print(f"  - RMSE: {metrics['rmse']:.6f}")


if __name__ == "__main__":
    main()
