# SAM3D Dataset Processing Pipeline Guide

**Last Updated**: 2025-11-25
**Purpose**: Process SAM3D GUI output datasets for 3DAnimals training

---

## 📋 Overview

This guide covers the complete pipeline for processing mouse (and other animal) datasets generated from SAM3D GUI into the Fauna dataset format required by 3DAnimals.

### What This Pipeline Does

1. ✅ **Generates missing files**: box.txt, metadata.json from masks
2. ✅ **Validates dataset**: Ensures all required files exist
3. ✅ **Splits train/val/test**: Automatic 70/15/15 split
4. ✅ **Integrates with 3DAnimals**: Ready-to-train format

---

## 🎯 Quick Start

### One-Command Update

```bash
# Replace old mouse data with new SAM3D data
./scripts/update_mouse_dataset.sh
```

**What it does:**
1. Preprocesses SAM3D data (generates box.txt, metadata.json)
2. Backs up old mouse data
3. Splits train/val/test (70/15/15)
4. Validates and tests loading
5. Ready to train!

**Duration**: 2-3 minutes for 200 frames

---

## 📁 Input Format (SAM3D GUI Output)

**Expected structure:**
```
/home/joon/dev/sam3d_gui/outputs/fauna_datasets/
└── mouse/
    └── train/
        ├── seq_000/
        │   ├── 0000000_rgb.png
        │   ├── 0000000_mask.png
        │   ├── 0000001_rgb.png
        │   ├── 0000001_mask.png
        │   └── ...
        └── seq_001/
            └── ...
```

**Required files per frame:**
- `{frame_id}_rgb.png` - RGB image ✅
- `{frame_id}_mask.png` - Binary mask ✅

**Generated automatically:**
- `{frame_id}_box.txt` - Bounding box (from mask)
- `{frame_id}_metadata.json` - Camera parameters

---

## 🔧 Step-by-Step Pipeline

### Step 1: Preprocess SAM3D Data

**Purpose**: Generate missing box.txt and metadata.json files

```bash
# Interactive mode (recommended)
conda run -n 3danimals python scripts/preprocess_sam3d_dataset.py --interactive

# Manual mode
conda run -n 3danimals python scripts/preprocess_sam3d_dataset.py \
  --source /home/joon/dev/sam3d_gui/outputs/fauna_datasets/mouse \
  --animal mouse \
  --output data/fauna_processed \
  --copy
```

**Options:**
- `--copy`: Copy files (standalone, uses disk space)
- No `--copy`: Symlink files (saves space, requires source)
- `--overwrite`: Overwrite existing files

**Output:**
```
data/fauna_processed/mouse/train/
├── seq_000/
│   ├── 0000000_rgb.png
│   ├── 0000000_mask.png
│   ├── 0000000_box.txt       ← Generated
│   └── 0000000_metadata.json ← Generated
└── seq_001/
    └── ...
```

**Example output:**
```
[INFO] Processed frames:    200
[INFO] Generated boxes:     200
[INFO] Generated metadata:  200
[INFO] Copied files:        400
✅ All files validated successfully!
```

---

### Step 2: Prepare for 3DAnimals

**Purpose**: Split train/val/test and create configs

```bash
python scripts/prepare_fauna_dataset.py \
  --source data/fauna_processed/mouse/train \
  --animal mouse \
  --split-mode frame \
  --ratio 0.7,0.15,0.15
```

**This creates:**
```
data/fauna/large_scale/mouse/
├── train/ (140 frames, 70%)
├── val/   (30 frames, 15%)
└── test/  (30 frames, 15%)
```

**Plus configs:**
- `config/train_mouse.yaml`
- `config/train_mouse_debug.yaml`
- `config/dataset/mouse.yaml`
- `config/model/mouse.yaml`

---

### Step 3: Verify and Train

```bash
# Verify data loads correctly
conda run -n 3danimals python run.py --config-name train_mouse_debug 2>&1 | \
  grep "using.*categories"

# Expected: using 1 categories, contains: ['large_scale_mouse']

# Run debug training (15-20 min)
conda run -n 3danimals python run.py --config-name train_mouse_debug

# Run full training (3-5 hours for 200 frames)
conda run -n 3danimals python run.py --config-name train_mouse
```

---

## 🔄 Complete Workflow Example

### Scenario: New mouse data from SAM3D GUI

```bash
# 1. Check new data
ls /home/joon/dev/sam3d_gui/outputs/fauna_datasets/mouse/train/
# → seq_000/ seq_001/ (200 frames total)

# 2. Run complete pipeline (one command)
./scripts/update_mouse_dataset.sh

# Pipeline steps:
# ✓ Preprocessing... (30 sec)
# ✓ Backing up old data...
# ✓ Splitting train/val/test... (10 sec)
# ✓ Verifying... (5 sec)
# ✓ Testing loading... (20 sec)

# 3. Start training
conda run -n 3danimals python run.py --config-name train_mouse_debug
```

**Total time**: 2-3 minutes setup + training time

---

## 📊 Generated Files

### box.txt Format

```
frame_id crop_x0 crop_y0 crop_w crop_h full_w full_h sharpness label
```

**Example:**
```
0000000 45 67 166 122 256 256 1.0 0
```

**How it's generated:**
1. Load binary mask
2. Find bounding box of non-zero pixels
3. Add 5-pixel margin
4. Save in Fauna format

### metadata.json Format

```json
{
  "frame_id": "0000000",
  "image_width": 256,
  "image_height": 256,
  "camera": {
    "focal_length": 525.0,
    "principal_point": [128.0, 128.0],
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "pose": {
    "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "translation": [0.0, 0.0, 10.0]
  }
}
```

**Default values** (monocular setup):
- Focal length: 525.0 (for 256×256 images)
- Principal point: Image center
- No distortion
- Identity rotation, 10.0 depth

---

## 🔧 Advanced Usage

### Custom Split Ratios

```bash
# 80/10/10 split
python scripts/prepare_fauna_dataset.py \
  --source data/fauna_processed/mouse/train \
  --animal mouse \
  --ratio 0.8,0.1,0.1
```

### Sequence-Based Split

```bash
# Split entire sequences (not frames within sequences)
python scripts/prepare_fauna_dataset.py \
  --source data/fauna_processed/mouse/train \
  --animal mouse \
  --split-mode sequence \
  --ratio 0.6,0.2,0.2
```

**Use when:**
- Have many sequences (5+)
- Want no frame leakage between splits

### Process Multiple Animals

```bash
# Cat dataset
conda run -n 3danimals python scripts/preprocess_sam3d_dataset.py \
  --source /path/to/sam3d_output/cat \
  --animal cat \
  --output data/fauna_processed

# Then prepare
python scripts/prepare_fauna_dataset.py \
  --source data/fauna_processed/cat/train \
  --animal cat
```

---

## 🗂️ Data Management

### Backup Strategy

**Automatic backup:**
```bash
# Old data backed up before replacement
data/fauna/large_scale/mouse_backup_20251125_123456/
```

**Manual backup:**
```bash
# Before processing new data
cp -r data/fauna/large_scale/mouse \
      data/fauna/large_scale/mouse_backup_manual
```

### Disk Space

**Estimates for 200 frames:**
- Raw RGB+Mask: ~50 MB
- After preprocessing (copy): ~100 MB
- After split: ~100 MB (same files, organized)
- Total: ~250 MB

**Space-saving option:**
```bash
# Use symlinks instead of copy
python scripts/preprocess_sam3d_dataset.py \
  --source /path/to/sam3d \
  --animal mouse \
  --output data/fauna_processed
# (no --copy flag)

# Space used: ~50 MB (only generated files)
```

---

## 🔍 Troubleshooting

### Issue: "No frames found"

**Cause**: Wrong source directory

**Fix:**
```bash
# Ensure source points to animal/train directory
# ✗ Wrong: /path/to/fauna_datasets
# ✓ Correct: /path/to/fauna_datasets/mouse
```

### Issue: RGB/Mask mismatch

**Symptoms:**
```
⚠️  seq_000: RGB=100, Mask=95
```

**Fix:**
```bash
# Check for missing masks
ls /path/to/seq_000/*_mask.png | wc -l

# Manually create missing masks or remove unpaired RGBs
```

### Issue: Generated boxes are wrong

**Symptoms**: All boxes are (0, 0, 256, 256)

**Cause**: Empty or invalid masks

**Fix:**
```bash
# Check mask quality
conda run -n 3danimals python << EOF
from PIL import Image
import numpy as np

mask = Image.open("seq_000/0000000_mask.png")
mask_array = np.array(mask)
print(f"Mask shape: {mask_array.shape}")
print(f"Non-zero pixels: {np.sum(mask_array > 0)}")
print(f"Unique values: {np.unique(mask_array)}")
EOF

# Mask should have non-zero pixels
# If all zero: regenerate masks in SAM3D GUI
```

### Issue: Training fails to load

**Error:**
```
FileNotFoundError: .../mouse/train
```

**Fix:**
```bash
# Verify final structure
ls data/fauna/large_scale/mouse/
# Should show: train/ val/ test/

# Check symlink
ls -la data/fauna/large_scale/mouse
# Should NOT be symlink if using copied files
```

---

## 📈 Performance

### Processing Speed

| Frames | Preprocess | Split | Total |
|--------|-----------|-------|-------|
| 100 | 30 sec | 5 sec | 35 sec |
| 200 | 60 sec | 10 sec | 70 sec |
| 500 | 2 min | 20 sec | 2.5 min |
| 1000 | 4 min | 40 sec | 5 min |

**Factors:**
- Mask complexity (bbox extraction)
- Copy vs symlink
- Disk I/O speed

### Training Impact

**Dataset size vs training time:**

| Frames | Split (70%) | Iterations | Time (RTX 3060) |
|--------|-------------|-----------|-----------------|
| 50 | 35 | 50K | 2-3 hours |
| 200 | 140 | 50K | 3-5 hours |
| 500 | 350 | 100K | 10-15 hours |
| 1000 | 700 | 150K | 20-30 hours |

---

## 🎯 Best Practices

### Data Quality

✅ **DO:**
- Check mask quality before processing
- Use diverse poses and angles
- Keep sequence structure (temporal coherence)
- Validate after each step

❌ **DON'T:**
- Mix different camera setups in one dataset
- Use corrupted or low-quality masks
- Skip validation steps
- Delete backups immediately

### File Organization

**Recommended structure:**
```
/home/joon/dev/
├── sam3d_gui/
│   └── outputs/
│       └── fauna_datasets/      ← SAM3D raw output
│           ├── mouse/
│           ├── cat/
│           └── dog/
└── 3DAnimals/
    ├── data/
    │   ├── fauna_processed/     ← Intermediate (with box/meta)
    │   │   ├── mouse/
    │   │   └── cat/
    │   └── fauna/large_scale/   ← Final (with train/val/test)
    │       ├── mouse/
    │       └── cat/
    └── scripts/
```

### Workflow Tips

1. **Always test with debug mode first**
   ```bash
   conda run -n 3danimals python run.py --config-name train_mouse_debug
   ```

2. **Monitor first 100 iterations**
   - Loss should decrease
   - Check GPU utilization
   - Verify batch loading

3. **Keep processing logs**
   ```bash
   ./scripts/update_mouse_dataset.sh 2>&1 | tee logs/update_mouse_$(date +%Y%m%d).log
   ```

---

## 🚀 Future Enhancements

### Planned Features

- [ ] **Multi-view support**: Process DANNCE-style multi-camera data
- [ ] **Automatic mask refinement**: SAM-based mask improvement
- [ ] **Quality filtering**: Auto-detect and remove bad frames
- [ ] **Temporal consistency**: Frame interpolation for smooth sequences
- [ ] **Batch processing**: Handle multiple animals simultaneously

### Integration Ideas

- **SAM3D GUI plugin**: Direct export to 3DAnimals format
- **Continuous pipeline**: Auto-process new SAM3D outputs
- **Cloud processing**: Distributed preprocessing for large datasets

---

## 📚 Related Documentation

- [Fauna Dataset Complete Guide](FAUNA_DATASET_COMPLETE_GUIDE.md)
- [Multi-Animal Training Guide](MULTI_ANIMAL_TRAINING_GUIDE.md)
- [Mouse Training Guide](MOUSE_TRAINING_GUIDE.md)

---

## 🆘 Support

**Common questions:**

Q: Can I process other animals besides mouse?
A: Yes! Use `--animal cat`, `--animal dog`, etc.

Q: What if I have more sequences coming?
A: Run preprocessing again, it will merge with existing data.

Q: Can I change the split ratio later?
A: Yes, re-run `prepare_fauna_dataset.py` with new ratio.

Q: How do I delete old backups?
A: `rm -rf data/fauna/large_scale/mouse_backup_*`

---

**Questions? Issues?** Check [GitHub Issues](https://github.com/3DAnimals/3DAnimals/issues)
