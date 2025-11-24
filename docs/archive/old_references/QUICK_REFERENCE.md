# 3DAnimals Codebase - Quick Reference

## Project Overview

Three related projects in one codebase:
1. **MagicPony** (CVPR 2023) - Category-specific 3D animal from single image
2. **3D-Fauna** (CVPR 2024) - Pan-category quadruped 3D reconstruction
3. **Ponymation** (ECCV 2024) - 3D animal motion generation

All use the same core: **Image → Shape + Articulation + Texture → Mesh**

---

## Data Flow (Core Pipeline)

```
RGB Image (B,F,C,H,W)
    ↓
[ViTEncoder] → Features (global + patches)
    ↓
[BasePredictorBank] → Prior shape (SDF → Mesh via DMTet)
    ↓
[InstancePredictorFauna]
    ├→ Pose: Image features → Camera extrinsics
    ├→ Shape: Prior shape + learned deformation
    ├→ Texture: Image features → Albedo, Normal
    ├→ Articulation: Image → Bone rotations
    └→ Skinning: Vertex deformation via LBS
    ↓
[NVDiffRast Renderer] → Rendered image
    ↓
Loss Computation & Backprop
```

---

## Key Classes

### Models (`model/models/`)
- **AnimalModel**: Base PyTorch Lightning module
- **FaunaModel**: Main training model for 3D-Fauna
- **Fauna4DModel**: 4D video reconstruction variant

### Predictors (`model/predictors/`)
- **BasePredictorBase**: Category-level shape prior
  - `netShape`: DMTetGeometry (SDF → mesh)
  - `netDINO`: Semantic feature field

- **InstancePredictorBase**: Per-image instance parameters
  - `netEncoder`: ViT feature extraction
  - `netPose`: Camera pose prediction
  - `netTexture`: Color prediction
  - `netDeform`: Optional shape deformation
  - `netArticulation`: Bone rotation prediction
  - `netLight`: Optional lighting

### Geometry (`model/geometry/`)
- **DMTetGeometry**: Differentiable SDF → mesh conversion
  - Key method: `getMesh()`
- **estimate_bones()**: Extract skeleton from shape
- **skinning()**: Apply bone deformations via LBS

### Networks (`model/networks/`)
- **ViTEncoder**: DINO feature extraction
- **CoordMLP**: MLPs with positional encoding
- **ArticulationNetwork**: Predicts rotation parameters

---

## Key Configurations

Located in `config/` (Hydra):

```yaml
# Main config determines model type
model:
  type: 'fauna'  # or 'magicpony', 'ponymation'

# Shape representation
cfg_shape:
  grid_res: 64  # Tetrahedral grid resolution
  spatial_scale: 5.0

# Instance prediction
cfg_articulation:
  num_body_bones: 8
  num_legs: 4
  num_leg_bones: 3
  max_arti_angle: 60  # Degree constraint

# Training
learning_rate: 0.001
batch_size: 4
num_epochs: 200
```

---

## Important File Locations

### Core Model Files
- Model definition: `model/models/Fauna.py`
- Instance predictor: `model/predictors/InstancePredictorFauna.py`
- Shape representation: `model/geometry/dmtet.py`

### Articulation System
- Bone estimation: `model/geometry/skinning.py::estimate_bones()`
- Articulation network: `model/networks/ArticulationNetwork.py`
- Skinning/LBS: `model/geometry/skinning.py::skinning()`

### Rendering
- Rasterizer: `model/render/render.py`
- Material system: `model/render/material.py`
- Lighting: `model/render/light.py`

### Entry Points
- Training: `run.py --config-name train_fauna`
- Testing: `run.py --config-name test_fauna`
- Visualization: `visualization/visualize_results_fauna.py`

---

## Articulation Parameters (Key for Pose Splatter)

**Structure**: 
```python
arti_params: Tensor(B, F, num_bones, 3)  # Euler angles per bone
```

**Extraction**:
```python
# In InstancePredictorFauna.forward_articulation():
1. estimate_bones(shape) → bones, kinematic_tree
2. Sample image features at bone locations
3. Pass through netArticulation → Euler angles
4. Apply skinning to deform shape
```

**Fauna Constraints**:
- Leg bones: Limited to specific axes
- Magnitude: Clamped to ±max_arti_angle
- Temporal smoothness: Regularized

---

## Shape Representation Details

### SDF to Mesh (DMTet)

```python
# Step 1: Sample SDF at tet grid vertices
sdf = netShape.get_sdf(vertices, feats=class_vector)

# Step 2: Marching tetrahedrons
verts, faces, uvs, uv_idx = marching_tets(vertices, sdf, tet_indices)

# Step 3: Mesh object
mesh = Mesh(verts, faces, uvs, uv_idx)
```

### Key Parameters
- `grid_res`: Resolution of tetrahedral grid (64, 128, etc.)
- `spatial_scale`: World coordinate range (±5.0)
- `jitter_grid`: Optional noise for regularization

---

## Loss Functions

### Reconstruction
- `rgb_loss`: L1/L2 pixel-level
- `mask_loss`: Binary CE for silhouettes
- `dino_loss`: Feature space consistency
- `flow_loss`: Optical flow for sequences

### Regularization
- `arti_reg_loss`: L2(rotation angles)
- `deform_reg_loss`: L2(shape deformation)
- `arti_smooth_loss`: Temporal consistency
- `sdf_bce_reg_loss`: SDF zero-crossing constraint
- `sdf_gradient_reg_loss`: Eikonal constraint

---

## Training Tips

### Progressive Training Stages
1. **Shape prior** (BasePredictorBank): Category-level learning
2. **Instance deformation**: Shape adaptation to individual animals
3. **Articulation**: Skeleton-based pose/motion
4. **Texture refinement**: Color/material tuning

### Iteration Ranges
- `articulation_iter_range`: When to enable articulation training
- `texture_iter_range`: When to optimize appearance
- `coarse_iter_range`: When to use coarse shape grid
- `attach_legs_to_body_iter_range`: When legs connect to body

### Hyperparameters
- `skinning_temperature`: Sharpness of LBS weights (1.0 = moderate)
- `max_arti_angle`: Rotation constraint (60 degrees typical)
- `embedder_freq`: Positional encoding frequency (8-10)

---

## Testing/Inference

### Quick Test
```bash
python run.py --config-name test_fauna
```

### Visualization Modes
- `input_view`: Mesh from input camera angle
- `other_views`: 12 rotating views
- `rotation`: 360° rotation video
- `animation`: Articulation animation (quadrupeds)

### Output
- Mesh: OBJ format with texture
- Images: Shaded renders, normal maps
- Keypoints: 3D joint positions

---

## For Pose Splatter Integration

### Directly Reusable (100%)
- Image encoder (ViTEncoder)
- Pose estimation (netPose)
- Articulation prediction (ArticulationNetwork)
- Bone estimation (estimate_bones)

### Needs Adaptation (20-30%)
- DMTet → Gaussian SDF field
- Mesh rendering → Gaussian splatting
- Per-vertex skinning → Per-Gaussian skinning

### Key Functions to Understand
1. `estimate_bones()` - Shape → skeleton
2. `skinning()` - Skeleton → deformation
3. `ArticulationNetwork.forward()` - Features → rotations
4. `render()` - Mesh → image

---

## Quick Debug Commands

```bash
# Check config
python run.py --config-name test_fauna --dry-run

# Single GPU training
python run.py --config-name train_fauna

# Multi-GPU training
accelerate launch --multi_gpu run.py --config-name train_fauna

# Validation during training
python run.py --config-name test_fauna +test=true

# Visualize results
python visualization/visualize_results_fauna.py --config-name test_fauna
```

---

## File Statistics

- **Total Python files**: ~50
- **Lines of code**: ~15,000
- **Model parameters**: ~2-5M (without backbone)
- **GPU memory**: ~8-12GB for training batch_size=4
- **Typical training time**: 50-100 hours on 1 GPU

---

## Important Notes

1. **DMTet**: Uses pre-computed tetrahedral grids from `data/tets/` - must be downloaded
2. **ViT Backbone**: Uses frozen DINO v1 - no gradient through encoder
3. **Category Bank**: For multi-category (Fauna) - stores category embeddings
4. **Per-Frame Parameters**: 4D models (Fauna4D) store per-frame pose & articulation dicts
5. **Skinning**: Uses softmax distance weights - differentiable but can be unstable

---

## References

- **MagicPony**: https://3dmagicpony.github.io/ (CVPR 2023)
- **3D-Fauna**: https://kyleleey.github.io/3DFauna/ (CVPR 2024)
- **Ponymation**: https://keqiangsun.github.io/projects/ponymation/ (ECCV 2024)
- **DMTet**: https://research.nvidia.com/labs/toronto-ai/DMTet/
- **DINO**: https://github.com/facebookresearch/dino

