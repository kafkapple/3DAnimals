# 3DAnimals System Comprehensive Guide

**Date**: 2025-11-21
**Author**: Joon + Claude Code
**Version**: 1.0

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Dataset Structure](#dataset-structure)
3. [Training Workflows](#training-workflows)
4. [Inference & Visualization](#inference--visualization)
5. [Adding New Animal Datasets](#adding-new-animal-datasets)
6. [Configuration System](#configuration-system)
7. [Quick Start Examples](#quick-start-examples)
8. [Troubleshooting](#troubleshooting)

---

## 1. System Overview

### 1.1 Architecture

3DAnimals is a monocular 3D animal reconstruction system that learns:
- **Canonical 3D Shape**: SDF (Signed Distance Function) representation
- **Articulation**: Skeletal motion with automatic bone estimation
- **Appearance**: Texture and lighting
- **Pose**: Camera viewpoint estimation

### 1.2 Key Components

```
3DAnimals/
├── config/              # Hydra configuration files
│   ├── dataset/        # Dataset configs (fauna_mouse.yaml, etc.)
│   ├── model/          # Model configs (architecture, hyperparams)
│   └── train_*.yaml    # Training configs (entry point)
├── model/              # Model implementations
│   ├── Trainer.py      # Training loop
│   ├── predictors/     # Base/Instance predictors
│   └── geometry/       # SDF, DMTet, skinning
├── visualization/      # Result visualization scripts
└── data/              # Dataset storage
    └── fauna/
        └── Fauna_dataset/
            ├── few_shot_animal3d/    # Small datasets (10-50 images)
            └── large_scale/          # Large datasets (200+ images)
```

### 1.3 Supported Datasets

**Few-shot (10-50 images per animal)**:
- 40+ animal categories from Animal3D
- Examples: american_black_bear, arctic_wolf, golden_retriever_dog, etc.

**Large-scale (200+ images per animal)**:
- bear, cow, elephant, giraffe, horse, sheep, zebra
- **Mouse** (multi-view): dannce_6view, markerless_6view

---

## 2. Dataset Structure

### 2.1 Fauna Dataset Format

Each animal dataset follows this structure:

```
{animal_name}/
├── train/
│   ├── {sequence_id}/
│   │   ├── {frame_id}_rgb.png          # Input image
│   │   ├── {frame_id}_mask.png         # Segmentation mask
│   │   ├── {frame_id}_keypoint.txt     # Optional: 2D keypoints
│   │   ├── {frame_id}_box.txt          # Bounding box
│   │   └── {frame_id}_metadata.json    # Camera parameters
│   └── ...
├── val/   (optional)
└── test/  (optional)
```

### 2.2 Multi-view Dataset (Mouse Example)

```
mouse_markerless_6view/
├── train/
│   ├── cam00_seq_000/  # Camera 0
│   ├── cam01_seq_000/  # Camera 1
│   ├── ...
│   └── cam05_seq_000/  # Camera 5
```

Each sequence contains:
- `{frame_id}_rgb.png`: 256×256 image (resized from 1152×1024)
- `{frame_id}_mask.png`: Binary segmentation mask
- `{frame_id}_keypoint.txt`: 2D keypoints (optional)
- `{frame_id}_metadata.json`: Camera intrinsics/extrinsics

### 2.3 Required Data Files

**Minimum Requirements**:
- RGB images (`*_rgb.png`)
- Segmentation masks (`*_mask.png`)
- Bounding boxes (`*_box.txt`)

**Optional but Recommended**:
- 2D keypoints (`*_keypoint.txt`) - for evaluation
- DINO features (pre-extracted) - speeds up training
- Camera parameters (`*_metadata.json`) - for multi-view

---

## 3. Training Workflows

### 3.1 Training Strategies

| Strategy | Use Case | Dataset Size | Iterations | Time (RTX 3060) |
|----------|----------|--------------|------------|-----------------|
| **Few-shot from scratch** | New animal, limited data | 10-50 images | 50K-100K | 2-3 hours |
| **Large-scale from scratch** | New animal, rich data | 200+ images | 200K-500K | 10-12 hours |
| **Fine-tuning** | Similar animal, adaptation | Any | 20K-50K | 1-2 hours |
| **Debug mode** | **ALWAYS RUN FIRST** | Any | 5K | 15-30 min |

### 3.2 Configuration Files

**Three-level Configuration System**:

1. **Dataset Config** (`config/dataset/fauna_{animal}.yaml`):
   - Image resolution
   - Batch size
   - Data augmentation
   - Data paths

2. **Model Config** (`config/model/fauna_{animal}.yaml`):
   - Network architecture
   - Shape parameters (grid_res, spatial_scale)
   - Articulation (num_bones, skeleton)
   - Loss weights

3. **Training Config** (`config/train_fauna_{animal}.yaml`):
   - Training duration (num_iters)
   - Checkpointing frequency
   - Dataset paths
   - Logging (WandB)

### 3.3 Training Commands

#### Debug Mode (MANDATORY FIRST STEP)

```bash
# 1. Run debug mode first (15-30 min)
conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_debug

# 2. Check results
tail -f results/fauna_mouse_debug/tensorboard_logs/events.out.*

# 3. Verify checkpoints
ls results/fauna_mouse_debug/checkpoint*.pth
```

#### Full Training

```bash
# After debug succeeds, run full training
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_from_scratch \
  > /tmp/fauna_mouse_train.log 2>&1 &

# Monitor progress
tail -f /tmp/fauna_mouse_train.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### 3.4 Training Schedule (Progressive Learning)

```yaml
# Iteration ranges for different features
articulation_iter_range: [20000, inf]     # Enable at 20K
attach_legs_to_body: [60000, inf]         # Attach at 60K
deform_iter_range: [800000, inf]          # Deformation at 800K (if needed)

# Grid resolution (multi-resolution training)
grid_res_coarse: 64                       # 0-300K iterations
grid_res: 128                             # 300K+ iterations
```

### 3.5 Key Hyperparameters by Animal Size

| Animal Size | Examples | spatial_scale | grid_res | num_body_bones | GPU Memory |
|-------------|----------|---------------|----------|----------------|------------|
| **Small** | mouse, rat, squirrel | 4-5 | 64-128 | 4-6 | 4-8 GB |
| **Medium** | cat, dog, rabbit | 6-7 | 128-256 | 6-8 | 8-12 GB |
| **Large** | horse, cow, elephant | 7-10 | 256-512 | 8-12 | 16-24 GB |

**RTX 3060 12GB Limits**:
- `grid_res: 64` → ~4GB (✅ Safe)
- `grid_res: 128` → ~14GB (❌ OOM!)
- **Recommendation**: Use `grid_res: 64` for mouse

---

## 4. Inference & Visualization

### 4.1 Inference Command

```bash
# Run inference on test set
python visualization/visualize_results_fauna.py \
  --config-name test_fauna \
  checkpoint_dir=results/fauna_mouse_from_scratch \
  checkpoint_name=checkpoint200000.pth \
  dataset.test_data_dir=data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/test \
  output_dir=results/fauna_mouse_from_scratch/visualization
```

### 4.2 Visualization Modes

**Render Modes** (`render_modes` in test config):

1. **input_view**: Reconstruct input viewpoint
   - Outputs: RGB, mask, normal, shading

2. **other_views**: Multi-view rendering
   - Generate novel viewpoints
   - 360° rotation around object

3. **rotation**: Animated rotation
   - Creates video sequence
   - Shows 3D shape quality

### 4.3 Output Files

```
visualization/
├── {sample_id}_input_rgb.png           # Input image
├── {sample_id}_input_rgb_pred.png      # Reconstructed RGB
├── {sample_id}_input_mask_pred.png     # Predicted mask
├── {sample_id}_rotation_000.png        # Rotation frames
├── {sample_id}_rotation_001.png
├── ...
└── {sample_id}_rotation_video.mp4      # Rotation video
```

### 4.4 Evaluation Metrics

**Quantitative Metrics**:
- **Mask IoU**: Segmentation accuracy
- **RGB PSNR**: Appearance quality
- **Keypoint Error**: Pose accuracy (if keypoints available)

**Qualitative Metrics**:
- Shape consistency across views
- Articulation quality
- Texture detail

---

## 5. Adding New Animal Datasets

### 5.1 Complete Workflow

#### Step 1: Prepare Your Data

**Required Format**:
```python
# Minimum required files per frame:
{frame_id}_rgb.png      # Image (any resolution → will be resized to 256×256)
{frame_id}_mask.png     # Binary mask (same size as RGB)
{frame_id}_box.txt      # Bounding box [x_min, y_min, x_max, y_max]
```

**Optional Files**:
```python
{frame_id}_keypoint.txt      # 2D keypoints [x1, y1, x2, y2, ..., visibility]
{frame_id}_metadata.json     # Camera parameters
```

#### Step 2: Organize Dataset Structure

```bash
# Create dataset directory
mkdir -p data/fauna/Fauna_dataset/large_scale/{your_animal}_dataset

# Organize data
data/fauna/Fauna_dataset/large_scale/{your_animal}_dataset/
├── train/
│   ├── seq_000/
│   │   ├── 0000000_rgb.png
│   │   ├── 0000000_mask.png
│   │   ├── 0000000_box.txt
│   │   └── ...
│   └── seq_001/
└── val/  (optional)
```

#### Step 3: Create Configuration Files

**3.1 Dataset Config** (`config/dataset/fauna_{your_animal}.yaml`):

```bash
# Use template
cp config/dataset/fauna_new_animal_template.yaml \
   config/dataset/fauna_rabbit.yaml

# Edit parameters:
# - batch_size: based on GPU memory
# - train_data_dir: path to your dataset
```

**3.2 Model Config** (`config/model/fauna_{your_animal}.yaml`):

```bash
# Use template
cp config/model/fauna_new_animal_template.yaml \
   config/model/fauna_rabbit.yaml

# Adjust based on animal size:
# Small (mouse):  spatial_scale: 4-5,  grid_res: 64,  num_body_bones: 4-6
# Medium (rabbit): spatial_scale: 6-7,  grid_res: 128, num_body_bones: 6-8
# Large (horse):  spatial_scale: 7-10, grid_res: 256, num_body_bones: 8-12
```

**3.3 Training Config** (`config/train_fauna_{your_animal}.yaml`):

```bash
# Use template
cp config/train_fauna_new_animal_template.yaml \
   config/train_fauna_rabbit.yaml

# Update:
# - exp_name: fauna_rabbit_from_scratch
# - dataset.train_data_dir: (your data path)
# - num_iters: based on dataset size
# - wandb.project: fauna_rabbit
```

#### Step 4: Create Debug Config

```bash
# Create debug version
cp config/train_fauna_rabbit.yaml \
   config/train_fauna_rabbit_debug.yaml

# Modify for quick testing:
# - num_iters: 5000
# - save_checkpoint_freq: 1000
# - log_image_freq: 100
# - wandb.mode: offline
```

#### Step 5: Run Debug Training

```bash
# ALWAYS run debug first (15-30 min)
conda run -n 3danimals python run.py \
  --config-name train_fauna_rabbit_debug

# Verify:
# 1. Data loads correctly
# 2. Model initializes (no shape errors)
# 3. Training loop runs (1 epoch completes)
# 4. Checkpoints save successfully
# 5. No CUDA OOM
```

#### Step 6: Full Training

```bash
# After debug succeeds
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_rabbit \
  > /tmp/fauna_rabbit_train.log 2>&1 &

# Monitor
tail -f /tmp/fauna_rabbit_train.log
```

### 5.2 Data Preparation Scripts

#### Convert Your Images to Fauna Format

```python
# Example: Convert custom dataset to Fauna format
import os
import cv2
import numpy as np
from pathlib import Path

def convert_to_fauna_format(
    input_dir: str,
    output_dir: str,
    animal_name: str
):
    """
    Convert custom image dataset to Fauna format.

    Args:
        input_dir: Directory with raw images
        output_dir: Output directory (will create train/val/test)
        animal_name: Name of the animal
    """
    # Create output structure
    train_dir = Path(output_dir) / animal_name / "train" / "seq_000"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    image_files = sorted(Path(input_dir).glob("*.jpg"))

    for idx, img_path in enumerate(image_files):
        frame_id = f"{idx:07d}"

        # 1. Load and resize image
        img = cv2.imread(str(img_path))
        img_resized = cv2.resize(img, (256, 256))

        # 2. Generate mask (simple background subtraction)
        # TODO: Use proper segmentation (SAM, Segment Anything, etc.)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # 3. Compute bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            bbox = [x_min, y_min, x_max, y_max]
        else:
            bbox = [0, 0, 256, 256]

        # 4. Save files
        cv2.imwrite(str(train_dir / f"{frame_id}_rgb.png"), img_resized)
        cv2.imwrite(str(train_dir / f"{frame_id}_mask.png"), mask)
        np.savetxt(train_dir / f"{frame_id}_box.txt", bbox)

    print(f"Converted {len(image_files)} images to {train_dir}")

# Usage
convert_to_fauna_format(
    input_dir="path/to/your/images",
    output_dir="data/fauna/Fauna_dataset/large_scale",
    animal_name="rabbit_dataset"
)
```

### 5.3 Mask Generation Options

**Option 1: Segment Anything Model (SAM)**

```python
from segment_anything import sam_model_registry, SamPredictor

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Generate mask
predictor.set_image(image)
masks, _, _ = predictor.predict(
    point_coords=center_point,  # Click on animal
    point_labels=[1]
)
```

**Option 2: Background Subtraction**

```python
# For static camera
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
mask = bg_subtractor.apply(image)
```

**Option 3: Pre-trained Segmentation Models**

```python
# DeepLabV3+, Mask R-CNN, etc.
from torchvision.models.segmentation import deeplabv3_resnet101
model = deeplabv3_resnet101(pretrained=True)
```

### 5.4 Configuration Parameter Guide

#### Small Animals (Mouse, Rat, Squirrel)

```yaml
# config/model/fauna_mouse.yaml
cfg_predictor_base:
  cfg_shape:
    grid_res: 64              # Low memory footprint
    grid_res_coarse: 32
    spatial_scale: 4.5        # Small scene size
    num_layers: 5
    hidden_size: 128          # Smaller network

  cfg_articulation:
    num_body_bones: 5         # Fewer spine bones
    num_legs: 4
    num_leg_bones: 3
    articulation_iter_range: [20000, inf]

# Training
num_iters: 100000             # Shorter training
batch_size: 6                 # Can fit more
```

#### Medium Animals (Cat, Dog, Rabbit)

```yaml
# config/model/fauna_dog.yaml
cfg_predictor_base:
  cfg_shape:
    grid_res: 128
    grid_res_coarse: 64
    spatial_scale: 6.5
    hidden_size: 256

  cfg_articulation:
    num_body_bones: 7
    num_legs: 4
    num_leg_bones: 3

# Training
num_iters: 200000
batch_size: 4
```

#### Large Animals (Horse, Cow, Elephant)

```yaml
# config/model/fauna_horse.yaml
cfg_predictor_base:
  cfg_shape:
    grid_res: 256             # High resolution (needs 24GB GPU)
    grid_res_coarse: 128
    spatial_scale: 8.0
    hidden_size: 256

  cfg_articulation:
    num_body_bones: 10        # More spine articulation
    num_legs: 4
    num_leg_bones: 3

# Training
num_iters: 300000             # Longer training
batch_size: 2                 # Limited by memory
```

---

## 6. Configuration System

### 6.1 Hydra Configuration Hierarchy

```yaml
# Training config (top-level)
config/train_fauna_mouse.yaml
  ├── defaults:
  │   ├── model: fauna_mouse      # Model config
  │   └── dataset: fauna_mouse    # Dataset config
  ├── exp_name: fauna_mouse_from_scratch
  ├── num_iters: 200000
  ├── wandb: {...}
  └── ...

# Model config (architecture)
config/model/fauna_mouse.yaml
  ├── defaults:
  │   └── fauna                   # Base fauna config
  ├── cfg_predictor_base:
  │   ├── cfg_shape: {...}        # SDF parameters
  │   ├── cfg_dino: {...}         # Feature parameters
  │   └── cfg_bank: {...}         # Memory bank
  └── cfg_predictor_instance:
      ├── cfg_encoder: {...}      # Image encoder
      ├── cfg_texture: {...}      # Appearance
      ├── cfg_pose: {...}         # Viewpoint
      ├── cfg_articulation: {...} # Skeleton
      ├── cfg_deform: {...}       # Deformation
      └── cfg_light: {...}        # Lighting

# Dataset config (data loading)
config/dataset/fauna_mouse.yaml
  ├── data_type: fauna
  ├── in_image_size: 256
  ├── batch_size: 6
  ├── train_data_dir: (auto-filled by hydra)
  └── ...
```

### 6.2 Command-line Overrides

```bash
# Override specific parameters
python run.py \
  --config-name train_fauna_mouse \
  num_iters=50000 \
  dataset.batch_size=4 \
  model.cfg_predictor_base.cfg_shape.grid_res=64 \
  wandb.mode=offline

# Override dataset path
python run.py \
  --config-name train_fauna_mouse \
  dataset.train_data_dir=data/fauna/Fauna_dataset/large_scale/my_custom_mouse
```

### 6.3 Configuration Templates

**Dataset Template**: `config/dataset/fauna_new_animal_template.yaml`
- Fully documented with inline comments
- Parameter ranges and recommendations
- GPU memory guidelines

**Model Template**: `config/model/fauna_new_animal_template.yaml`
- Architecture explanations
- Hyperparameter tuning guide
- Animal size-specific settings

**Training Template**: `config/train_fauna_new_animal_template.yaml`
- Training strategy recommendations
- Debug-first workflow
- Time estimation tables

---

## 7. Quick Start Examples

### 7.1 Example 1: Train on Existing Mouse Dataset

```bash
# Step 1: Verify data
ls data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/train/

# Step 2: Debug mode (15-30 min)
conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_markerless_debug

# Step 3: Check results
tensorboard --logdir results/fauna_mouse_markerless_debug/tensorboard_logs

# Step 4: Full training (10-12 hours)
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_markerless \
  > /tmp/fauna_mouse.log 2>&1 &

# Step 5: Monitor
tail -f /tmp/fauna_mouse.log
```

### 7.2 Example 2: Add Custom Cat Dataset

```bash
# Step 1: Prepare data (assume you have images in ~/cat_images/)
python scripts/convert_to_fauna.py \
  --input_dir ~/cat_images \
  --output_dir data/fauna/Fauna_dataset/large_scale \
  --animal_name cat_custom \
  --use_sam_masks  # Use Segment Anything for masks

# Step 2: Create configs
cp config/dataset/fauna_new_animal_template.yaml \
   config/dataset/fauna_cat.yaml

cp config/model/fauna_new_animal_template.yaml \
   config/model/fauna_cat.yaml

cp config/train_fauna_new_animal_template.yaml \
   config/train_fauna_cat.yaml

# Step 3: Edit configs
# - fauna_cat.yaml (dataset): set batch_size=4, train_data_dir
# - fauna_cat.yaml (model): spatial_scale=6.5, grid_res=128, num_body_bones=7
# - train_fauna_cat.yaml: exp_name, dataset paths, wandb project

# Step 4: Debug
cp config/train_fauna_cat.yaml config/train_fauna_cat_debug.yaml
# Edit: num_iters=5000, log_image_freq=100

conda run -n 3danimals python run.py --config-name train_fauna_cat_debug

# Step 5: Full training
conda run -n 3danimals python run.py --config-name train_fauna_cat
```

### 7.3 Example 3: Fine-tune Pretrained Model

```bash
# Assume you have a pretrained horse model, want to adapt to zebra

# Step 1: Create config
cp config/train_fauna_horse.yaml config/train_fauna_zebra_finetune.yaml

# Step 2: Edit config
# - exp_name: fauna_zebra_finetune
# - num_iters: 50000  (shorter for fine-tuning)
# - resume: results/fauna_horse/checkpoint200000.pth  (pretrained)
# - dataset.train_data_dir: data/fauna/.../zebra_dataset

# Step 3: Train
conda run -n 3danimals python run.py \
  --config-name train_fauna_zebra_finetune
```

### 7.4 Example 4: Inference on Trained Model

```bash
# Generate visualizations
python visualization/visualize_results_fauna.py \
  --config-name test_fauna \
  checkpoint_dir=results/fauna_mouse_markerless \
  checkpoint_name=checkpoint200000.pth \
  dataset.test_data_dir=data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/test \
  output_dir=results/fauna_mouse_markerless/viz \
  render_modes=[input_view,rotation]

# Check outputs
ls results/fauna_mouse_markerless/viz/
# - *_input_rgb_pred.png (reconstructions)
# - *_rotation_*.png (rotation frames)
# - *_rotation_video.mp4 (video)
```

---

## 8. Troubleshooting

### 8.1 Common Errors

#### Error 1: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions**:
1. Reduce `grid_res` in model config:
   ```yaml
   grid_res: 128 → 64 → 32
   ```

2. Reduce `batch_size` in dataset config:
   ```yaml
   batch_size: 6 → 4 → 2
   ```

3. Reduce network size:
   ```yaml
   hidden_size: 256 → 128 → 64
   ```

4. Use gradient checkpointing (if implemented)

#### Error 2: Data Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/fauna/...'
```

**Solutions**:
1. Check dataset path in config:
   ```yaml
   dataset:
     train_data_dir: data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/train
   ```

2. Verify data exists:
   ```bash
   ls data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/train/
   ```

3. Use absolute path if relative path fails

#### Error 3: Shape Mismatch in Encoder

```
RuntimeError: Given input size: (X, Y, Z). Calculated output size: ...
```

**Solution**: Minimum image size is 128×128. Check:
```yaml
dataset:
  in_image_size: 256  # Must be ≥ 128
  out_image_size: 256
```

#### Error 4: None Reference Error (arti_params, etc.)

```
TypeError: rearrange(None, ...) received None
```

**Root Cause**: Progressive training - feature not enabled yet

**Solution**: Add None check in save/log functions (already fixed):
```python
if log.arti_params is not None:
    misc.save_txt(..., rearrange(log.arti_params, ...), ...)
```

**Prevention**: Always use debug mode first to catch early

### 8.2 Performance Issues

#### Issue 1: Training Too Slow

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Check CPU bottleneck
htop
```

**Solutions**:
- Reduce `num_workers` if CPU bottleneck
- Use smaller `grid_res` for faster iterations
- Check disk I/O speed (slow storage?)

#### Issue 2: Loss Not Decreasing

**Diagnosis**:
- Check WandB dashboard for loss curves
- Verify data quality (masks, images)
- Check for NaN values

**Solutions**:
1. Verify data loading in debug mode
2. Reduce learning rate:
   ```python
   # In model config (if exposed)
   learning_rate: 0.0001 → 0.00005
   ```
3. Check SDF initialization (ellipsoid vs sphere)
4. Increase SDF regularization

### 8.3 Result Quality Issues

#### Issue 1: Mesh Collapse

**Symptoms**: Degenerate geometry, flat shapes

**Solutions**:
- Increase SDF regularization:
  ```yaml
  cfg_loss:
    sdf_gradient_reg_loss_weight: 0.1 → 0.5
  ```
- Use ellipsoid initialization:
  ```yaml
  cfg_shape:
    init_sdf: ellipsoid
  ```
- Check spatial_scale matches animal size

#### Issue 2: Poor Articulation

**Symptoms**: Limbs don't move naturally

**Solutions**:
- Adjust skeleton parameters:
  ```yaml
  num_body_bones: 8  # Try 6, 8, 10
  num_leg_bones: 3   # Usually 3 is best
  ```
- Enable leg constraints (for quadrupeds):
  ```yaml
  constrain_legs: true
  ```
- Check attachment iteration range:
  ```yaml
  attach_legs_to_body_iter_range: [60000, inf]
  ```

#### Issue 3: Texture Artifacts

**Symptoms**: Blurry or incorrect colors

**Solutions**:
- Train longer (200K → 300K iterations)
- Check lighting parameters:
  ```yaml
  cfg_light:
    amb_diff_minmax: [[0.0, 1.0], [0.5, 1.0]]
  ```
- Verify DINO feature quality
- Increase texture resolution (if exposed)

### 8.4 Debugging Workflow

**Standard Debugging Process**:

1. **Verify Data**:
   ```bash
   # Check file counts
   find data/.../train -name "*_rgb.png" | wc -l
   find data/.../train -name "*_mask.png" | wc -l

   # Visualize samples
   python scripts/visualize_dataset.py --data_dir data/.../train
   ```

2. **Run Debug Mode**:
   ```bash
   # Short training (5K iters, 15-30 min)
   python run.py --config-name train_{animal}_debug
   ```

3. **Check Logs**:
   ```bash
   # Training logs
   tail -100 /tmp/train.log

   # TensorBoard
   tensorboard --logdir results/{exp_name}/tensorboard_logs

   # WandB
   # Check online dashboard
   ```

4. **Inspect Checkpoints**:
   ```bash
   # List checkpoints
   ls results/{exp_name}/checkpoint*.pth

   # Load and inspect
   python scripts/inspect_checkpoint.py --path results/.../checkpoint5000.pth
   ```

5. **Test Inference**:
   ```bash
   # Quick inference test
   python visualization/visualize_results_fauna.py \
     --config-name test_fauna \
     checkpoint_dir=results/{exp_name} \
     checkpoint_name=checkpoint5000.pth \
     dataset.test_data_dir=data/.../test \
     render_modes=[input_view]
   ```

---

## Appendix A: Parameter Reference

### A.1 Dataset Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_type` | str | `fauna` | Dataset type (fauna, magicpony, ponymation) |
| `in_image_size` | int | 256 | Input image size (min: 128) |
| `out_image_size` | int | 256 | Output render size |
| `batch_size` | int | 6 | Batch size (adjust for GPU) |
| `num_workers` | int | 4 | Data loading workers |
| `random_xflip_train` | bool | false | Random horizontal flip augmentation |
| `background_mode` | str | none | Background: none, white, checkerboard |
| `load_dino_feature` | bool | false | Load pre-extracted DINO features |

### A.2 Model Config Parameters (Key Subset)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spatial_scale` | float | 7.0 | Scene size (4-5: small, 6-7: medium, 7-10: large) |
| `grid_res` | int | 128 | SDF grid resolution (64, 128, 256, 512) |
| `grid_res_coarse` | int | 64 | Initial coarse resolution |
| `num_layers` | int | 5 | MLP depth (3-7) |
| `hidden_size` | int | 256 | MLP width (64, 128, 256, 512) |
| `num_body_bones` | int | 8 | Spine bones (4-6: small, 6-8: med, 8-12: large) |
| `num_legs` | int | 4 | Number of legs (4: quadruped, 2: biped) |
| `num_leg_bones` | int | 3 | Bones per leg (typically 3) |

### A.3 Training Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iters` | int | 200000 | Total training iterations |
| `save_checkpoint_freq` | int | 10000 | Checkpoint save frequency |
| `log_image_freq` | int | 1000 | Image logging frequency |
| `eval_freq` | int | 5000 | Validation frequency (-1: disable) |
| `resume` | str | null | Resume checkpoint path |
| `seed` | int | 42 | Random seed |

---

## Appendix B: GPU Memory Requirements

### B.1 Memory Usage by Configuration

| Config | grid_res | batch_size | hidden_size | Memory | GPU |
|--------|----------|------------|-------------|--------|-----|
| **Minimal** | 32 | 2 | 64 | ~2 GB | GTX 1660 |
| **Small (Mouse)** | 64 | 4 | 128 | ~4 GB | RTX 3060 |
| **Medium (Dog)** | 128 | 4 | 256 | ~8 GB | RTX 3070 |
| **Large (Horse)** | 256 | 2 | 256 | ~16 GB | RTX 3090 |
| **Huge** | 512 | 1 | 512 | ~32 GB | A100 40GB |

### B.2 OOM Prevention Checklist

- [ ] `grid_res` fits GPU memory
- [ ] `batch_size` reduced if needed
- [ ] `hidden_size` reasonable for task
- [ ] No memory leaks (check with `nvidia-smi`)
- [ ] Use debug mode first to verify

---

## Appendix C: Dataset Preparation Checklist

### C.1 Data Quality Checklist

- [ ] Images are clear and well-lit
- [ ] Animal is visible in all frames
- [ ] Masks are accurate (cover entire animal)
- [ ] Bounding boxes are tight
- [ ] Consistent camera intrinsics (if multi-view)
- [ ] Diverse poses (different viewpoints/articulations)
- [ ] No severe occlusions or truncations

### C.2 Data Quantity Guidelines

| Dataset Type | Num Images | Viewpoint Coverage | Training Strategy |
|--------------|------------|-------------------|-------------------|
| **Few-shot** | 10-50 | Limited | From scratch (50-100K iters) |
| **Medium** | 50-200 | Moderate | From scratch (100-200K iters) |
| **Large-scale** | 200+ | Rich | From scratch (200-500K iters) |
| **Multi-view** | 50+ (6 views) | Excellent | Specialized (100-200K iters) |

---

## Summary

This guide provides a comprehensive overview of the 3DAnimals system:

1. **System Architecture**: SDF-based 3D reconstruction with articulation
2. **Dataset Structure**: Fauna format with RGB, mask, bbox, keypoints
3. **Training Workflows**: Debug-first, progressive learning, checkpointing
4. **Inference**: Multi-mode visualization (input, novel views, rotation)
5. **Adding New Animals**: Step-by-step workflow with templates
6. **Configuration**: Hydra-based three-level config system
7. **Troubleshooting**: Common errors and solutions

**Key Takeaways**:
- **Always run debug mode first** (15-30 min) before full training (10+ hours)
- **Use templates** for quick setup of new animals
- **Adjust hyperparameters** based on animal size (small/medium/large)
- **Monitor GPU memory** - reduce grid_res/batch_size if OOM
- **Check data quality** - good masks are critical

**Next Steps**:
1. Verify your dataset structure matches Fauna format
2. Copy and customize config templates
3. Run debug mode to validate setup
4. Launch full training
5. Visualize results and iterate

For questions or issues, refer to:
- This guide's troubleshooting section
- Template configs (fully documented)
- Original 3DAnimals paper/code

---

**Document Version**: 1.0
**Last Updated**: 2025-11-21
**Maintainer**: Joon
