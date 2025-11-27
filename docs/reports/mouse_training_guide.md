# Mouse Training & Inference Guide

## 1. Data Preprocessing

```bash
# SAM3D output -> Fauna format (with train/val/test split)
python scripts/preprocess_sam3d_dataset.py \
    --source /path/to/sam3d_output \
    --target data/fauna/large_scale \
    --animal mouse_custom \
    --split 0.8:0.1:0.1 \
    --copy
```

**Output structure:**
```
data/fauna/large_scale/mouse_custom/
├── train/seq_000/, seq_001/, ...
├── val/seq_000/, ...
└── test/seq_000/, ...
```

---

## 2. Training

### Option A: From Scratch (Mouse Only)
```bash
# Debug first (~20 min)
python run.py --config-name train_mouse_debug

# Full training (~10-11 hours)
nohup python run.py --config-name train_mouse_scratch > /tmp/mouse_train.log 2>&1 &
```

### Option B: Finetune from Pretrained
```bash
# Resume from pretrained checkpoint
python run.py --config-name train_mouse \
    resume=results/fauna/pretrained_fauna/pretrained_fauna.pth
```

### Key Config Parameters
```yaml
# config/train_mouse.yaml
defaults:
  - dataset: mouse
  - model: mouse

exp_name: mouse_custom
num_iters: 200000        # Full: 200K, Debug: 5K
save_checkpoint_freq: 5000
log_image_freq: 500
device: cuda
run_train: true
run_test: false
```

---

## 3. Inference & Visualization

### Single Image Test
```bash
python run.py --config-name test_fauna \
    checkpoint_dir=results/mouse_custom/ \
    checkpoint_name=latest.pth \
    dataset.test_data_dir=data/fauna/large_scale/mouse_custom/test
```

### Render Modes
```yaml
# Available modes in test config
render_modes: [input_view, other_views, rotation]

# Options:
# - input_view: Original viewpoint reconstruction
# - other_views: Novel view synthesis
# - rotation: 360° turntable video
```

### Output Location
```
results/mouse_custom/visualization/
├── {seq_name}_input_view.png
├── {seq_name}_other_views.png
└── {seq_name}_rotation.mp4
```

---

## 4. Custom Visualization Script

```python
# scripts/visualize_mouse.py (example)
import torch
from model.models.Fauna import Fauna
from visualization.visualize_results_fauna import visualize_reconstruction

# Load model
model = Fauna.from_pretrained("results/mouse_custom/latest.pth")
model.eval()

# Run inference
with torch.no_grad():
    result = model.forward(input_image)

# Save visualization
visualize_reconstruction(
    result,
    output_dir="results/visualization",
    render_modes=["input_view", "rotation"]
)
```

---

## 5. Training Strategies Summary

| Strategy | Command | Duration | Use Case |
|----------|---------|----------|----------|
| Debug | `train_mouse_debug` | ~20 min | Config validation |
| From scratch | `train_mouse_scratch` | ~11 hours | Large dataset |
| Finetune | `train_mouse + resume` | ~3-5 hours | Small dataset |

---

## 6. Troubleshooting

**CUDA OOM:**
```yaml
# Reduce grid resolution
model:
  grid_res: 64  # default 128 -> 64
```

**Empty masks:**
```bash
# Skip empty masks during preprocessing
python scripts/preprocess_sam3d_dataset.py ... --skip-empty-masks
```

**Mesh collapse:**
```yaml
# Increase SDF regularization
sdf_reg_decay_start_iter: 10000
sdf_bce_reg_loss_weight: 0.05
```
