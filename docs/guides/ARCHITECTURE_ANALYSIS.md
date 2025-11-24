# 3DAnimals Repository Architecture Analysis

## Overview
The 3DAnimals repository is a unified codebase for multiple projects on articulated 3D animal reconstruction and animation:
- **MagicPony** (CVPR 2023): Category-specific single-image 3D animal reconstruction
- **3D-Fauna** (CVPR 2024): Pan-category single-image quadruped reconstruction
- **Ponymation** (ECCV 2024): Articulated 3D animal motion generative model

## 1. Model Architecture Overview

### High-Level Pipeline

```
Input Image 
    ↓
[Encoder] (ViT-based DINO features)
    ↓
[Base Predictor] → Prior Shape (SDF-based)
    ↓
[Instance Predictor]
    ├→ Shape Deformation (DMTet)
    ├→ Pose Estimation (Camera extrinsics)
    ├→ Texture Prediction (Albedo, Roughness, Normal)
    ├→ Articulation (Bone rotations)
    └→ Lighting (if enabled)
    ↓
[Renderer] (NVDiffRast)
    ↓
Output: Mesh + Rendered Images
```

## 2. Key Model Classes and Responsibilities

### 2.1 Base Models (`model/models/`)

#### **AnimalModel** (`AnimalModel.py`)
- **Responsibility**: Base model class inheriting from PyTorch Lightning
- **Key Methods**:
  - `forward()`: Main training loop
  - `render()`: Handles rendering pipeline
  - `compute_reconstruction_losses()`: Loss computation
  - `log_visuals()`: Visualization logging

#### **FaunaModel** (`Fauna.py`)
- **Extends**: `AnimalModel`
- **Key Features**:
  - `netBase`: BasePredictorBank (category-wise shape)
  - `netInstance`: InstancePredictorFauna (per-instance deformation, articulation)
  - `netDisc`: Mask discriminator for adversarial training
  - **Additional Methods**:
    - `compute_mask_disc_loss_gen()`: Generator loss for discriminator
    - `discriminator_step()`: Discriminator training step
    - `get_random_view_mask()`: Generates masks for random viewpoints

#### **Fauna4DModel** (`Fauna4D.py`)
- **Extends**: `FaunaFinetune`
- **Key Features**:
  - Extends Fauna with 4D video reconstruction
  - `netBase`: BasePredictorFauna4D (time-aware base predictor)
  - `netInstance`: InstancePredictorFauna4D (temporal consistency)
  - Methods:
    - `compute_mean_feature()`: Computes temporal mean features
    - `set_finetune_arti()`: Fine-tunes articulation parameters
    - `set_finetune_texture()`: Fine-tunes texture
    - `inference()`: Full 4D inference pipeline

### 2.2 Predictors (`model/predictors/`)

#### **BasePredictorBase** (`BasePredictorBase.py`)
- **Responsibility**: Predicts category-level prior shape
- **Components**:
  ```python
  netShape: DMTetGeometry  # SDF-based shape representation
  netDINO: CoordMLP       # DINO feature field (category descriptor)
  ```
- **Forward Pass**:
  1. Predicts SDF values at tetrahedral grid vertices
  2. Marching tetrahedrons → mesh extraction
  3. Predicts DINO features for semantic consistency

#### **BasePredictorBank** (`BasePredictorBank.py`)
- **Extends**: `BasePredictorBase`
- **Key Addition**: Memory bank for multi-category support
  - Stores category embeddings
  - Weighted retrieval for category-specific shapes

#### **InstancePredictorBase** (`InstancePredictorBase.py`)
- **Responsibility**: Per-image instance-specific parameters
- **Config**: `InstancePredictorConfig` with sub-configs:
  ```python
  cfg_encoder: ViTEncoderConfig
  cfg_texture: TextureConfig
  cfg_pose: PoseConfig
  cfg_deform: DeformConfig
  cfg_articulation: ArticulationConfig
  cfg_light: LightingConfig
  ```
- **Core Networks**:
  ```python
  netEncoder: ViTEncoder          # DINO patches
  netPose: Encoder32              # Camera pose (6 or 19 dims)
  netTexture: CoordMLP            # Albedo, Roughness, Normal
  netDeform: CoordMLP             # Per-vertex deformation (opt)
  netArticulation: ArticulationNetwork  # Bone rotations (opt)
  netLight: DirectionalLight      # Lighting (opt)
  ```

#### **InstancePredictorFauna** (`InstancePredictorFauna.py`)
- **Extends**: `InstancePredictorBase`
- **Fauna-Specific Features**:
  - Automatic bone estimation from shape
  - Quadruped-specific articulation constraints
  - Pose hypothesis sampling (4-8 hypotheses for viewpoint ambiguity)
  - Regularization for leg articulation

#### **InstancePredictorFauna4D** (`InstancePredictorFauna4D.py`)
- **Extends**: `InstancePredictorFauna`
- **4D Extensions**:
  - Temporal parameter dictionaries (per-frame articulation & pose)
  - Motion VAE for smooth motion generation
  - Frame-specific parameter optimization

## 3. Shape Representation: DMTet (SDF to Mesh)

### 3.1 DMTetGeometry (`model/geometry/dmtet.py`)

**Purpose**: Differentiable mesh extraction from SDF representation

**Key Components**:

```python
class DMTetGeometry:
    __init__():
        grid_res: int          # 64, 128, etc. tetrahedral grid resolution
        spatial_scale: float   # World coordinate scale (~5.0)
        mlp: CoordMLP         # SDF prediction network
        
    getMesh():
        1. Sample SDF at tetrahedral vertices
        2. Run marching tetrahedrons
        3. Extract vertex positions and faces
        4. Compute UV mappings
        
    get_sdf():
        - Positional encoding with harmonics
        - Optional symmetry enforcement
        - Optional initialization (sphere/ellipsoid)
```

**Marching Tetrahedrons Logic** (`DMTet` class):
- Lookup table based on vertex occupancy (16 cases)
- Edge intersection interpolation using SDF values
- Bilinear interpolation for vertex positions
- Returns: `(verts, faces, uvs, uv_idx)`

### 3.2 SDF Network Architecture

```python
mlp = CoordMLP(
    in_dim=3,           # x, y, z coordinates
    out_dim=1,          # SDF value
    num_layers=5,       # Configurable
    hidden_size=64,     # Configurable
    n_harmonic_functions=8,  # Positional encoding
    embedder_scalar=embedder_scalar  # Frequency scaling
)
```

**Features**:
- Harmonic embedding of coordinates (sin/cos features)
- Optional feature conditioning (category embedding)
- Range constraints using min/max tensors

## 4. Articulation & Deformation System

### 4.1 Bone Structure (`model/geometry/skinning.py`)

**Bone Estimation**:
```python
estimate_bones(
    seq_shape,           # Vertex positions (B, F, V, 3)
    n_body_bones=8,      # Number of body segments
    n_legs=4,
    n_leg_bones=3,
    body_bones_mode='z_minmax',  # Find extrema along Z axis
    compute_kinematic_chain=True
) → (bones, kinematic_chain, aux)
```

**Kinematic Chain Structure**:
- List of tuples: `[(bone_id, [dependent_child_bones])]`
- Parent-child relationships for hierarchical transformation

**Bone Representation**:
```
bones: (B, F, num_bones, 2, 3)
        ↓     ↓  ↓           ↓  ↓
      batch frame bone_idx start/end position
```

### 4.2 Articulation Network (`model/networks/ArticulationNetwork.py`)

```python
class ArticulationNetwork(nn.Module):
    def __init__(
        self,
        net_type='mlp',         # or 'attention'
        feat_dim=256,           # Global image feature
        posenc_dim=10,          # Encoded bone position info
        num_layers=4,
        nf=64,                  # Hidden size
        n_harmonic_functions=8  # Positional encoding
    ):
        self.network: MLP or Attention
        
    def forward(self, x, pos):
        # x: [B*F*K, feat_dim] - features
        # pos: [B*F*K, posenc_dim] - bone positional info
        # Returns: [B*F*K, 3] - Euler angle rotations (x, y, z)
```

**Input Position Encoding** (posenc_dim=10):
- 1 dim: bone index (normalized)
- 2 dims: 2D bone midpoint position (in image space)
- 6 dims: 3D bone endpoint positions (2 endpoints × 3 coords)
- Harmonic positional encoding on these

### 4.3 Skinning / LBS (`model/geometry/skinning.py`)

```python
def skinning(
    v_pos,              # Rest mesh vertices (B, F, V, 3)
    bones_pred,         # Bone positions (B, F, K, 2, 3)
    kinematic_tree,     # Bone hierarchy
    deform_params,      # Articulation angles (B, F, K, 3)
    temperature=1       # Softmax temperature for weighting
) → deformed_verts, aux
```

**Process**:
1. **Compute LBS Weights**: Softmax distance from vertices to bones
2. **Kinematic Chain Traversal**: Root → leaf for each bone
3. **Transform Composition**: Accumulate transforms through chain
4. **Per-Vertex Blending**: Weighted sum of transformations

## 5. Data Flow: 2D Image to 3D Mesh

### Complete Pipeline

```
1. INPUT
   image: (B, F, 3, H, W) - RGB input images
   ↓

2. ENCODER (ViT-based)
   images → feat_out (B*F, feat_dim)
           → feat_key (B*F, vit_feat_dim)
           → patch_out (B*F, num_patches, patch_feat_dim)
           → patch_key (B*F, num_patches, patch_key_dim)
   ↓

3. BASE PREDICTOR (Shape Prior)
   feat_out + batch_info
        ↓
   netShape.getMesh(feats=class_vector)
        → sample SDF at tet vertices
        → marching tetrahedrons
        → prior_shape: (1, V, 3) vertex positions
   
   netDINO(vertices, feats=class_vector)
        → dino_features: (1, V, dino_dim)
   ↓

4. INSTANCE PREDICTOR (Per-image Parameters)
   
   a) POSE ESTIMATION
      patch_out → netPose → poses_raw (B*F, 6 or 19)
      ├─ 3 Euler angles (or 8×4 hypotheses for quadrupeds)
      └─ 3 Translation parameters
      → compute MVP, w2c, campos matrices
      
   b) SHAPE DEFORMATION
      If enabled:
      mesh_vertices → netDeform(with feat_out)
      → delta_verts (B*F, V, 3)
      → deformed_shape = prior_shape + delta_verts
      
   c) TEXTURE PREDICTION
      mesh_vertices → netTexture(with feat_out)
      → albedo (B*F, V, 3)
      → roughness (B*F, V, 3)
      → normal_perturbation (B*F, V, 3)
      
   d) ARTICULATION (Fauna-specific)
      estimate_bones(deformed_shape)
      → bones (B*F, K, 2, 3), kinematic_tree
      
      bones_feat = sample features at bone locations
      → netArticulation(bones_feat, bone_pos)
      → articulation_angles (B*F, K, 3) Euler angles
      
      skinning(deformed_shape, bones, kinematic_tree, angles)
      → articulated_shape (B*F, V, 3)
      
   e) LIGHTING (Optional)
      feat_out → netLight
      → direction, ambient, diffuse intensity
   ↓

5. RENDERER (NVDiffRast)
   render(
       shape=articulated_shape,
       mvp=mvp, w2c=w2c,
       material={albedo, roughness, normal},
       light=light_params,
       render_modes=['shaded', 'geo_normal', 'dino_pred']
   )
   → rendered_images (B*F, H, W, 3/4)
   ↓

6. OUTPUT
   reconstruction: (B, F, H, W, 3)
   mesh: OBJ format with vertices + textures
```

## 6. Articulation Parameters Structure

### Configuration
```python
@dataclass
class ArticulationConfig:
    num_body_bones: int = 8         # E.g., 4 for Fauna
    num_legs: int = 4
    num_leg_bones: int = 3
    articulation_iter_range: List[int] = [-1, -1]  # Enable from iteration X
    architecture: str = 'mlp'       # or 'attention'
    num_layers: int = 4
    hidden_size: int = 64
    embedder_freq: int = 8
    bone_feature_mode: str = 'global'  # 'sample', 'sample+global'
    max_arti_angle: float = 60.     # Constraint in degrees
    skinning_temperature: float = 1.0  # LBS weight sharpness
    use_fauna_constraints: bool = False
```

### Parameters at Runtime
```python
arti_params: Tensor
    shape: (B, F, num_bones, 3)  # Euler angles (X, Y, Z rotations)
    range: [-max_arti_angle, +max_arti_angle]
    
# For Fauna4D temporal:
articulation_dict: Dict[frame_id: int, Tensor(K, 3)]
```

### Fauna-Specific Constraints
```python
# From FaunaInstancePredictorConfig
apply_fauna_articulation_regularizer():
    - Bottom 2 bones of each leg: only X-axis rotation
    - Small angle constraints for leg joints
    - Body rotation regularization (reg_body_rotate_mult=0.1)
```

## 7. Potential Gaussian Splatting Integration

### Current Mesh-Based Pipeline
- **Representation**: Triangle mesh (vertices + faces)
- **Rendering**: Rasterization (NVDiffRast)
- **Advantages**: Topology explicit, controllable
- **Limitations**: No transparency/soft boundaries, discrete topology

### Gaussian Splatting Alternative ("Pose Splatter")

#### Integration Points

**1. Replace DMTet Shape**
```
Current: SDF → Marching Tets → Mesh → Rasterization
Proposed: SDF/GS → Gaussian Cloud → Splatting
```

**Option A: SDF to Gaussians**
- Extract Gaussian centers from mesh vertices
- Optimize covariance matrices from SDF gradients
- Spherical harmonics coefficients from texture/features

**Option B: Direct Gaussian SDF**
- Each Gaussian represents local SDF region
- Implicit SDF from Gaussian field
- Differentiable extraction

**2. Articulation with Gaussians**

Current skinning works on mesh vertices:
```python
skinning(vertices, bones, angles) → deformed_vertices
```

For Gaussians:
```python
# Each Gaussian: (position, covariance, spherical_harmonics, opacity)
deformed_gaussians = apply_skinning_to_gaussians(
    gaussians,           # List of (mu, sigma, sh, opacity)
    bones,
    kinematic_tree,
    arti_angles
)

# Skinning: applies LBS transform to Gaussian centers & covariances
# Covariance: Sigma' = R @ Sigma @ R^T
```

**3. Benefit Areas**

| Aspect | Mesh (Current) | Gaussians (Proposed) |
|--------|---|---|
| **Rendering** | Rasterization | Splatting |
| **Soft edges** | No | Yes (natural) |
| **Occlusion** | Hard | Soft (opacity) |
| **Articulation** | Vertex LBS | Gaussian LBS |
| **Topology** | Fixed | Implicit |
| **Real-time** | Slower | Faster |
| **Finetuning** | Slower convergence | Faster convergence |

**4. Implementation Plan**

```python
# New module: model/geometry/gaussian_sdf.py
class GaussianSDFField(nn.Module):
    def __init__(self, num_gaussians, spatial_scale):
        self.gaussians = nn.Parameter(...)  # (N, 3 + 3 + 16 + 1)
        
    def get_sdf(self, pts):
        # Compute SDF from Gaussian field
        distances = ||pts - mu_i|| for each Gaussian
        sdf = sum(w_i * exp(-0.5 * dist / sigma_i^2))
        
    def extract_gaussians(self, articulated=False):
        # Return {positions, covariances, colors, opacities}
        
    def apply_skinning(self, bones, angles):
        # Transform Gaussian positions and covariances

# In model/predictors/InstancePredictorBase.py
if use_gaussians:
    deformed_gaussians = self.apply_skinning_to_gaussians(
        gaussians, bones, angles
    )
    rendered = gaussian_splat_renderer(deformed_gaussians, mvp, resolution)
```

## 8. Module Reusability for Pose Splatter

### Can Be Directly Reused

1. **Articulation Estimation**
   - `estimate_bones()`: Shape → bone structure
   - `ArticulationNetwork`: Feature → Euler angles
   - `Kinematic chains`: Hierarchy representation
   - **Status**: 100% reusable

2. **Encoder**
   - `ViTEncoder`: DINO patch extraction
   - `netEncoder`: Image → features
   - **Status**: 100% reusable

3. **Pose Estimation**
   - `forward_pose()`: Features → camera extrinsics
   - **Status**: 100% reusable

4. **Lighting** (if needed)
   - `netLight`: Features → lighting parameters
   - **Status**: Reusable with minor adaptations

### Requires Adaptation

1. **Shape Representation**
   - `DMTetGeometry` → `GaussianSDFField`
   - Different SDF query mechanism
   - Different mesh extraction
   - **Effort**: Medium (1-2 weeks)

2. **Skinning**
   - Current: Vertex LBS
   - Needed: Gaussian covariance transformation
   - `R @ Sigma @ R^T` for rotated Gaussians
   - **Effort**: Low (few days, mostly linear algebra)

3. **Rendering**
   - Current: `nvdiffrast` (rasterization)
   - Needed: Gaussian splatting renderer
   - Option: Use external library (diff-gaussian-rasterization)
   - **Effort**: Low (integration) if using library

4. **Texture/Features**
   - Current: Per-vertex texture from mesh UVs
   - For Gaussians: Per-Gaussian color (RGB + spherical harmonics)
   - Can adapt existing `netTexture` → `netGaussianColor`
   - **Effort**: Medium (rewrite texture prediction)

### Module Replacement Summary

```
Current → Proposed Replacements

❌ model/geometry/dmtet.py
✅ model/geometry/gaussian_sdf.py (NEW)

✅ model/predictors/InstancePredictorBase.py
   (keep encoder, pose, articulation)
❌ render() method
✅ gaussian_render() (NEW)

✅ model/geometry/skinning.py
✅ skinning() (keep, reuse for Gaussians)
✅ estimate_bones() (keep, reuse)

❌ model/render/render.py (NVDiffRast rasterizer)
✅ model/render/gaussian_render.py (NEW, Gaussian splatting)
```

## 9. Loss Functions

### Reconstruction Losses
```python
rgb_loss: L1 or L2 loss between rendered and GT images
mask_loss: Binary cross-entropy for object mask
dino_loss: L2 distance in DINO feature space
flow_loss: L2 loss for optical flow consistency
```

### Regularization Losses
```python
arti_reg_loss: L2(arti_params) - encourage small rotations
deform_reg_loss: L2(deformation) - smooth deformation
arti_smooth_loss: Temporal smoothness of articulation
bone_smooth_loss: Temporal smoothness of bone positions
sdf_bce_reg_loss: Consistency at SDF zero-level set
sdf_gradient_reg_loss: Eikonal constraint ||∇SDF|| = 1
```

## 10. Configuration System

Uses **Hydra** (dataclass-based):

```
config/
├── config.yaml          # Base config
├── models/
│   ├── fauna.yaml
│   ├── magicpony.yaml
│   └── ponymation.yaml
├── datasets/
│   ├── fauna.yaml
│   └── magicpony.yaml
└── test_*.yaml         # Test configurations
```

Each config is structured as nested dataclasses matching the code structure.

---

## Summary Table: Key Integration Points for Pose Splatter

| Component | Current | Adaptation Needed | Reusable Code % |
|-----------|---------|------------------|-----------------|
| Image Encoder | ViTEncoder | None | 100% |
| Pose Estimation | netPose | None | 100% |
| Articulation | netArticulation + estimate_bones | None | 100% |
| Shape Repr. | DMTet (SDF→mesh) | Gaussian field | 0% |
| Skinning | LBS on vertices | LBS on Gaussians | 80% |
| Rendering | Rasterization | Splatting | 0% |
| Texture | UV-mapped | Spherical harmonics | 30% |
| **Overall** | **19 modules** | **3-4 major** | **~75%** |

