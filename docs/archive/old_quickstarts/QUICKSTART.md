# Fauna Mouse Training - Quick Start Guide

> **Last Updated**: 2025-11-12
> **Hardware Tested**: NVIDIA RTX 3060 12GB
> **Status**: From-scratch training ready (50 images, 1-view)

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Setup (5 minutes)](#quick-setup-5-minutes)
- [Training](#training)
- [Known Issues & Solutions](#known-issues--solutions)
- [Dataset Information](#dataset-information)
- [Research Notes](#research-notes)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 12GB or better (CUDA capable)
- **RAM**: 16GB+ recommended
- **Storage**: ~50GB free space
- **OS**: Linux (tested on Ubuntu 22.04)

### Software Requirements
- **Conda/Miniconda**: Required for environment management
- **CUDA**: 11.8 (for PyTorch 2.0.0)
- **Git**: For cloning repository

---

## Quick Setup (5 minutes)

### 1. Clone Repository
```bash
cd ~/dev  # or your preferred directory
git clone <repository-url> 3DAnimals
cd 3DAnimals
```

### 2. Create Conda Environment
```bash
# Create environment with Python 3.9
conda create -n 3danimals python=3.9 -y
conda activate 3danimals

# Install PyTorch with CUDA 11.8 (CRITICAL: Must match)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install Dependencies
```bash
# Install remaining dependencies
pip install hydra-core omegaconf trimesh scipy pillow matplotlib tensorboard
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
```

### 4. Verify CUDA Setup
```python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA: True, Device: NVIDIA GeForce RTX 3060
```

### 5. Prepare Data

#### Step 5a: Check if Data Already Exists
```bash
cd /home/joon/dev/3DAnimals

# Check if training data exists
ls -la data/fauna_mouse/large_scale/mouse_dannce_6view/train/

# Should show 5 directories (sequences):
# 000000_00000/, 000001_00000/, 000002_00000/, 000003_00000/, 000004_00000/

# Count total images (should be 50)
find data/fauna_mouse/large_scale/mouse_dannce_6view/train -name "*_rgb.png" | wc -l
```

#### Step 5b: If Data is Missing - Obtain DANNCE Dataset

**Option 1: Use Existing Data** (If you have DANNCE mouse data)
```bash
# Locate your DANNCE data (example paths)
ls /media/joon/kafka/data/3DAnimals/fauna_mouse/
# or
ls /path/to/your/dannce_mouse_6view/
```

**Option 2: Download Public Dataset**
```bash
# DANNCE mouse dataset sources:
# - Original DANNCE paper: https://github.com/spoonsso/dannce
# - Request access from authors if not public
# - Alternative: Use your own multi-camera mouse recordings
```

#### Step 5c: Convert DANNCE to Fauna Format

If you have raw DANNCE data, convert it:

```bash
cd /home/joon/dev/3DAnimals

# Create conversion script if not exists
# (This script should already be in the repo)

# Run conversion
python scripts/convert_dannce_to_fauna.py \
  --dannce_root /path/to/dannce_mouse_6view \
  --output_root data/fauna_mouse \
  --extract_views best \
  --num_workers 4

# Monitor conversion
tail -f /tmp/fauna_conversion.log
```

**Expected Output Structure**:
```
data/fauna_mouse/
└── large_scale/
    └── mouse_dannce_6view/
        ├── train/
        │   ├── 000000_00000/  (10 frames)
        │   │   ├── 0000027_rgb.png
        │   │   ├── 0000027_mask.png
        │   │   ├── 0000027_metadata.json
        │   │   ├── 0000027_box.txt
        │   │   └── ... (10 frames total)
        │   ├── 000001_00000/  (10 frames)
        │   ├── 000002_00000/  (10 frames)
        │   ├── 000003_00000/  (10 frames)
        │   └── 000004_00000/  (10 frames)
        ├── val/ → train (symlink)
        └── test/ → train (symlink)

Total: 50 images (5 sequences × 10 frames, 256×256 resolution)
```

#### Step 5d: Verify Data
```bash
# Count images
find data/fauna_mouse -name "*_rgb.png" | wc -l
# Expected: 50

# Check sample image
ls data/fauna_mouse/large_scale/mouse_dannce_6view/train/000000_00000/

# Verify metadata
cat data/fauna_mouse/large_scale/mouse_dannce_6view/train/000000_00000/0000027_metadata.json
```

**If verification succeeds**: ✅ Ready to train!
**If verification fails**: See [Dataset Information](#dataset-information) section for troubleshooting

---

## Training

### Option 1: From-Scratch Training (Recommended for Mouse)

**Why from-scratch?** Mouse shape is completely different from pretrained horse/cow models. Fine-tuning from horse/cow leads to mesh collapse (tested extensively).

```bash
cd /home/joon/dev/3DAnimals

# Start training (200K iterations, ~11 hours)
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_from_scratch \
  > /tmp/fauna_from_scratch.log 2>&1 &

# Monitor progress
tail -f /tmp/fauna_from_scratch.log
```

**Training Configuration**:
- **Dataset**: 50 images (5 sequences × 10 frames, 1-view)
- **Iterations**: 200,000 (~11 hours on RTX 3060)
- **Batch size**: 4
- **Grid resolution**: 64 (reduced from 128 for 12GB GPU)
- **SDF regularization**: Ultra-strong (20.0 BCE, 5.0 gradient)
- **Initialization**: Ellipsoid (no pretrained weights)

**Monitoring**:
```bash
# Check last 50 lines
tail -50 /tmp/fauna_from_scratch.log

# Watch training progress
watch -n 10 tail -20 /tmp/fauna_from_scratch.log

# Check GPU usage
nvidia-smi
```

### Option 2: Fine-Tuning (Not Recommended - For Reference Only)

**Warning**: Fine-tuning from horse/cow pretrained weights **will fail** for mouse due to shape manifold mismatch. This is documented for research purposes only.

```bash
# DO NOT RUN - This will cause mesh collapse around iteration 100-200
# See research notes for detailed failure analysis
# nohup conda run -n 3danimals python run.py \
#   --config-name train_fauna_mouse_finetune \
#   > /tmp/fauna_finetune.log 2>&1 &
```

---

## Known Issues & Solutions

### Issue 1: CUDA Out of Memory (OOM)

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate 140.00 MiB
```

**Solution**: Reduce `grid_res` in config file
```yaml
# config/train_fauna_mouse_from_scratch.yaml
model:
  cfg_predictor_base:
    cfg_shape:
      grid_res: 64  # Change from 128 to 64 (or even 32)
```

**Grid Resolution Trade-offs**:
| grid_res | Memory | Quality | Speed |
|----------|--------|---------|-------|
| 128 | ~14GB | Highest | Slowest |
| 64 | ~3-4GB | Good | Faster |
| 32 | ~1GB | Lower | Fastest |

### Issue 2: Circular Symlink in Dataset

**Problem**: `data/fauna_mouse/fauna_mouse -> data/fauna_mouse` (infinite loop)

**Solution**:
```bash
cd /home/joon/dev/3DAnimals
rm data/fauna_mouse/fauna_mouse  # Remove circular symlink
```

### Issue 3: Mesh Collapse During Fine-Tuning

**Symptom**: Training crashes with "Got empty training triangle mesh"

**Cause**: Shape manifold mismatch (horse/cow ≠ mouse)

**Solution**: **Use from-scratch training** instead of fine-tuning

---

## Dataset Information

### Current Dataset Status

```
Location: data/fauna_mouse/large_scale/mouse_dannce_6view/

Total Images: 50
Resolution: 256×256
Format: RGB + Mask + Metadata + BBox
Source: DANNCE 6-view mouse dataset (best view selected)

Structure:
├── train/
│   ├── 000000_00000/ (10 frames)
│   ├── 000001_00000/ (10 frames)
│   ├── 000002_00000/ (10 frames)
│   ├── 000003_00000/ (10 frames)
│   └── 000004_00000/ (10 frames)
├── val/ → train (symlink)
└── test/ → train (symlink)
```

### Dataset Sufficiency Analysis

| Metric | Current | Minimum | Recommended | Status |
|--------|---------|---------|-------------|--------|
| **Images** | 50 | 20-50 | 500+ | ⚠️ Minimum |
| **Views** | 1 | 1+ | 3-6 | ⚠️ Expandable |
| **Potential** | 300 (6-view) | - | - | ✅ Available |
| **Resolution** | 256×256 | 256×256 | 256×256 | ✅ Perfect |

### Expanding Dataset (Optional)

**Option A: Extract All 6 Views** (50 → 300 images)
```bash
# Modify conversion script to extract all 6 camera views
# instead of just best view
# → 6× more data, multi-view consistency
```

**Option B: Extract More Frames** (50 → 500-3000 images)
```bash
# Extract 50-100 frames per sequence instead of 10
# → More diverse poses and behaviors
```

---

## Research Notes

### Key Findings from Experiments

#### 1. Fine-Tuning Failures (Documented)

**Attempt 1**: Baseline regularization
- **Config**: `sdf_bce: 5.0, gradient: 1.0`
- **Result**: Mesh collapse at iteration 100
- **Conclusion**: Insufficient regularization

**Attempt 2**: Ultra-strong regularization
- **Config**: `sdf_bce: 10.0, gradient: 2.0`
- **Result**: Mesh collapse at iteration 182 (delayed but still failed)
- **Conclusion**: Regularization delays but doesn't solve root cause

**Root Cause**: Shape manifold mismatch
- Horse/Cow: 1.5-2m height, long legs, barrel chest
- Mouse: 5-10cm height (30-40× smaller!), short legs, compact body
- **Conclusion**: Transfer learning impossible between these species

#### 2. From-Scratch is Necessary

**Evidence from Fauna Paper** (Li et al., CVPR 2022):
> "We train from scratch on each species for 1M iterations..."

**Our Approach**:
- Start from ellipsoid initialization (generic prior)
- No pretrained weights from horse/cow
- Ultra-strong regularization for stability
- 200K iterations (1/5 of paper's 1M)

### Detailed Documentation

All research notes and failure analyses are documented in:
```
/home/joon/Documents/Obsidian/40_Areas/2_Research/_Notes/
├── 251112_research_fauna_mouse_complete_journey.md
├── 251112_fauna_mouse_mesh_collapse_analysis.md
├── 251112_research_mammal_sdf_memory_solution.md
└── 251112_research_fauna_dataset_specification.md
```

---

## Expected Results

### Training Timeline (200K iterations, ~11 hours)

| Phase | Iterations | Duration | Expected Progress |
|-------|-----------|----------|-------------------|
| **Phase 1** | 0-50K | ~2.5h | SDF field stabilization (ellipsoid → mouse-like) |
| **Phase 2** | 50-100K | ~2.5h | Basic shape learning + articulation |
| **Phase 3** | 100-150K | ~2.5h | Texture & details |
| **Phase 4** | 150-200K | ~2.5h | Refinement |

### Success Criteria

**Minimum (50 images)**:
- ✅ No mesh collapse (validated by ultra-strong regularization)
- ✅ Recognizable mouse shape
- ✅ Mask IoU: 0.6-0.7
- ✅ RGB PSNR: 18-22 dB

**Good (300+ images, 6-view)**:
- ✅ Detailed mouse anatomy
- ✅ Multi-view consistency
- ✅ Mask IoU: 0.75-0.85
- ✅ RGB PSNR: 22-26 dB

---

## Troubleshooting

### Training Not Starting

1. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```

2. **Verify conda environment**:
   ```bash
   conda activate 3danimals
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Check data path**:
   ```bash
   ls data/fauna_mouse/large_scale/mouse_dannce_6view/train/
   ```

### Training Crashes

1. **Check log file**:
   ```bash
   tail -100 /tmp/fauna_from_scratch.log
   ```

2. **Common errors**:
   - **OOM**: Reduce `grid_res` to 64 or 32
   - **Data not found**: Check dataset paths in config
   - **CUDA error**: Restart training, check GPU

---

## Next Steps After Training

### 1. Evaluate Results
```bash
# Check final checkpoint
ls results/fauna_mouse_from_scratch/

# Visualize reconstructions (TODO: create script)
python scripts/visualize_fauna_results.py
```

### 2. Improve with More Data
```bash
# Extract all 6 views from DANNCE
# → 50 → 300 images
# → Better quality, multi-view consistency
```

### 3. Longer Training
```bash
# If results are promising, continue training
# 200K → 500K → 1M iterations
# → Higher quality, more details
```

---

## Citation & Acknowledgments

**Fauna Paper**:
```bibtex
@inproceedings{li2022fauna,
  title={FAUNA: Learning Quadrupedal Locomotion from Single Videos},
  author={Li, Sheng and others},
  booktitle={CVPR},
  year={2022}
}
```

**MAMMAL Paper**:
```bibtex
@inproceedings{staps2021mammal,
  title={MAMMAL: Morphology-Aware Mesh Alignment},
  author={Staps, Joeri and others},
  booktitle={ICCV},
  year={2021}
}
```

---

## Contact & Support

For issues or questions:
1. Check research notes in `/home/joon/Documents/Obsidian/40_Areas/2_Research/_Notes/`
2. Review training logs in `/tmp/fauna_*.log`
3. Check GPU memory with `nvidia-smi`

**Key Config Files**:
- From-scratch: `config/train_fauna_mouse_from_scratch.yaml`
- Fine-tuning (not recommended): `config/train_fauna_mouse_finetune.yaml`

---

**Status**: Ready for from-scratch training once GPU is available ✅
