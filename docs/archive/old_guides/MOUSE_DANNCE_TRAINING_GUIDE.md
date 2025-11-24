# Mouse DANNCE Training Guide

**Dataset**: `/home/joon/dev/data/mouse_dannce_6view`
**Date**: 2025-11-21
**Status**: Ready for Training ✅

---

## 📊 Dataset Summary

### Dataset Information

| Property | Value |
|----------|-------|
| **Dataset Name** | mouse_dannce_6view |
| **Animal** | Mouse (small quadruped) |
| **Total Frames** | 50 frames |
| **Sequences** | 5 sequences × 10 frames each |
| **Resolution** | 256×256 (already processed) |
| **Format** | Fauna format (ready to use) |
| **Multi-view** | Yes (synchronized cameras) |

### Dataset Structure

```
/home/joon/dev/data/mouse_dannce_6view/
├── train/
│   ├── 000000_00000/  (10 frames)
│   ├── 000001_00000/  (10 frames)
│   ├── 000002_00000/  (10 frames)
│   ├── 000003_00000/  (10 frames)
│   └── 000004_00000/  (10 frames)
├── val -> train (symlink)
└── test -> train (symlink)
```

### Files per Frame

Each frame has:
- ✅ `{frame_id}_rgb.png` - RGB image (256×256)
- ✅ `{frame_id}_mask.png` - Segmentation mask
- ✅ `{frame_id}_box.txt` - Bounding box
- ✅ `{frame_id}_metadata.json` - Metadata

**Status**: All required files present ✅

---

## ⚙️ Configuration Files Created

### 1. Dataset Config
`config/dataset/fauna_mouse_dannce.yaml`

**Key Settings**:
- Image size: 256×256 (no resizing needed)
- Batch size: 2 (small for few-shot)
- No data augmentation (preserve multi-view geometry)

### 2. Model Config
`config/model/fauna_mouse_dannce.yaml`

**Key Settings** (Optimized for Mouse + RTX 3060):
- Spatial scale: 4.5 (small animal)
- Grid resolution: 64 (GPU friendly, ~4GB VRAM)
- Hidden size: 128 (smaller network)
- Body bones: 5 (small spine)
- Articulation: Enabled at 10K iters
- Deformation: Disabled (not enough data)

### 3. Training Configs

**Debug Config**: `config/train_fauna_mouse_dannce_debug.yaml`
- Iterations: 3K (~10-15 min)
- Purpose: Validation before full training

**Full Training Config**: `config/train_fauna_mouse_dannce.yaml`
- Iterations: 50K (~2-3 hours)
- Strategy: From scratch, few-shot optimized

---

## 🚀 Training Commands

### Step 1: Debug Mode (MANDATORY - 10-15 minutes)

**Always run debug mode first!**

```bash
# Navigate to project directory
cd /home/joon/dev/3DAnimals

# Activate conda environment
conda activate 3danimals

# Run debug training
python run.py --config-name train_fauna_mouse_dannce_debug
```

**What to Check**:
- ✅ Data loads correctly (should see "50 frames")
- ✅ Model initializes (no shape errors)
- ✅ No CUDA OOM (grid_res=64 should fit in 12GB)
- ✅ Training loop runs (~10-15 min)
- ✅ Checkpoints save to `results/fauna_mouse_dannce_debug/`

**Monitor Progress**:
```bash
# Watch training logs
tail -f nohup.out

# Or if using tensorboard
tensorboard --logdir results/fauna_mouse_dannce_debug/tensorboard_logs --port 6006

# Check GPU usage
watch -n 1 nvidia-smi
```

### Step 2: Full Training (2-3 hours)

**Only proceed if debug mode succeeds!**

```bash
# Option 1: Foreground (watch progress)
python run.py --config-name train_fauna_mouse_dannce

# Option 2: Background (recommended for long training)
nohup python run.py --config-name train_fauna_mouse_dannce \
  > /tmp/fauna_mouse_dannce_train.log 2>&1 &

# Monitor background training
tail -f /tmp/fauna_mouse_dannce_train.log

# Check GPU
watch -n 1 nvidia-smi
```

**Training Progress**:
```
Iteration 0-5K:    SDF initialization (ellipsoid → mouse)
Iteration 5K-10K:  Shape refinement, pose learning
Iteration 10K-30K: Articulation enabled, skeleton learning
Iteration 30K-50K: Legs attach, full articulation refinement

Total time: ~2-3 hours (RTX 3060 12GB)
```

---

## 📈 Monitoring Training

### WandB (Online Dashboard)

Training logs to WandB project: `fauna_mouse_dannce`

**Metrics to Monitor**:
- `loss/mask_loss` - Should decrease
- `loss/rgb_loss` - Should decrease
- `loss/total_loss` - Should decrease steadily
- `images/pred_rgb` - Visual quality
- `images/pred_mask` - Mask accuracy

### Tensorboard (Local)

```bash
tensorboard --logdir results/fauna_mouse_dannce_from_scratch/tensorboard_logs \
  --port 6006 --bind_all

# Open in browser: http://localhost:6006
```

### Checkpoints

Checkpoints saved every 5K iterations:
```
results/fauna_mouse_dannce_from_scratch/
├── checkpoint5000.pth   (30 min)
├── checkpoint10000.pth  (1 hour)
├── checkpoint15000.pth
├── ...
└── checkpoint50000.pth  (final, ~2.5 hours)
```

---

## 🎨 Visualization & Testing

### After Training Completes

```bash
# Generate visualizations
python visualization/visualize_results_fauna.py \
  --config-name test_fauna \
  checkpoint_dir=results/fauna_mouse_dannce_from_scratch \
  checkpoint_name=checkpoint50000.pth \
  dataset.test_data_dir=data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view/test \
  output_dir=results/fauna_mouse_dannce_from_scratch/visualization \
  render_modes=[input_view,rotation]
```

### Output Files

```
results/fauna_mouse_dannce_from_scratch/visualization/
├── {frame_id}_input_rgb.png          # Input image
├── {frame_id}_input_rgb_pred.png     # Reconstructed RGB
├── {frame_id}_input_mask_pred.png    # Predicted mask
├── {frame_id}_rotation_000.png       # Rotation frames
├── {frame_id}_rotation_001.png
├── ...
└── {frame_id}_rotation_video.mp4     # 360° rotation video
```

### Quality Metrics

**Expected Performance** (Few-shot dataset):
- **Mask IoU**: 0.75-0.85 (good overlap)
- **RGB PSNR**: 18-22 dB (reasonable quality)
- **Visual Quality**: Recognizable mouse shape

**Note**: Performance limited by small dataset (50 frames). For better quality, collect 100+ frames.

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```yaml
# Edit config/model/fauna_mouse_dannce.yaml
cfg_predictor_base:
  cfg_shape:
    grid_res: 64 → 32  # Further reduce resolution

# Or edit config/dataset/fauna_mouse_dannce.yaml
batch_size: 2 → 1  # Reduce batch size
```

### Issue 2: Data Not Found

**Error**: `FileNotFoundError: data/fauna/...`

**Solution**:
```bash
# Check symlink exists
ls -la /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view

# If missing, recreate symlink
ln -sfn /home/joon/dev/data/mouse_dannce_6view \
  /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view
```

### Issue 3: Training Too Slow

**Diagnosis**:
```bash
# Check GPU utilization (should be >80%)
nvidia-smi -l 1

# Check if CPU bottleneck
htop
```

**Solutions**:
- Reduce `num_workers` if CPU bottleneck
- Check data is on fast storage (SSD)
- Ensure CUDA is enabled (not CPU mode)

### Issue 4: Poor Results

**Symptoms**: Blurry reconstructions, wrong shape

**Possible Causes**:
1. **Limited data**: 50 frames is minimal
   - Solution: Collect more data (target: 100+ frames)

2. **Hyperparameters not tuned**:
   - Try different `spatial_scale`: 4.0, 4.5, 5.0
   - Adjust `num_body_bones`: 4, 5, 6

3. **Insufficient training**:
   - Train longer: 50K → 75K or 100K iterations

---

## 📋 Complete Training Workflow

### Full Checklist

```bash
# 1. Verify dataset
cd /home/joon/dev/3DAnimals
ls data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view/train/

# 2. Activate environment
conda activate 3danimals

# 3. Debug mode (10-15 min) - MANDATORY
python run.py --config-name train_fauna_mouse_dannce_debug

# 4. Verify debug results
ls results/fauna_mouse_dannce_debug/checkpoint*.pth

# 5. Full training (2-3 hours)
nohup python run.py --config-name train_fauna_mouse_dannce \
  > /tmp/fauna_mouse_dannce_train.log 2>&1 &

# 6. Monitor training
tail -f /tmp/fauna_mouse_dannce_train.log

# 7. Check GPU usage
watch -n 1 nvidia-smi

# 8. Visualize results (after training)
python visualization/visualize_results_fauna.py \
  --config-name test_fauna \
  checkpoint_dir=results/fauna_mouse_dannce_from_scratch \
  checkpoint_name=checkpoint50000.pth \
  dataset.test_data_dir=data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view/test \
  render_modes=[input_view,rotation]

# 9. Review outputs
ls results/fauna_mouse_dannce_from_scratch/visualization/
```

---

## 📊 Expected Timeline

| Stage | Duration | Checkpoint | What to Expect |
|-------|----------|------------|----------------|
| **Debug** | 10-15 min | 3K iters | Validation, no OOM |
| **Early Training** | 30 min | 5K iters | Shape emerges |
| **Mid Training** | 1 hour | 10K iters | Articulation starts |
| **Late Training** | 2 hours | 30K iters | Legs attach |
| **Final** | 2.5 hours | 50K iters | Full quality |
| **Visualization** | 5-10 min | - | Generate outputs |

**Total**: ~3 hours (debug + training + visualization)

---

## 🎯 Success Criteria

### Training Success
- ✅ No CUDA OOM errors
- ✅ Loss decreases steadily
- ✅ Checkpoints save successfully
- ✅ GPU utilization >80%
- ✅ Completes 50K iterations

### Result Quality
- ✅ Mask IoU > 0.75
- ✅ Recognizable mouse shape in rotation video
- ✅ Consistent multi-view reconstruction
- ✅ Reasonable articulation (limited by data)

### Known Limitations (Few-shot)
- ⚠️ May not generalize to new poses
- ⚠️ Articulation quality limited by data diversity
- ⚠️ Texture details may be limited

**Recommendation**: For production use, collect 100+ frames with diverse poses.

---

## 🔄 Next Steps

### After Successful Training

1. **Evaluate Quality**
   - Check reconstruction videos
   - Measure quantitative metrics
   - Compare with input images

2. **Collect More Data** (if quality insufficient)
   - Target: 100-200 frames
   - Diverse poses and viewpoints
   - Follow [Dataset Preparation Guide](docs/FAUNA_DATASET_PREPARATION_GUIDE.md)

3. **Fine-tune** (if more data collected)
   ```bash
   # Resume from checkpoint with new data
   python run.py --config-name train_fauna_mouse_dannce \
     resume=results/fauna_mouse_dannce_from_scratch/checkpoint50000.pth \
     num_iters=100000
   ```

4. **Deploy for Inference**
   - Use trained model for new mouse images
   - Generate 3D reconstructions
   - Extract articulation parameters

---

## 📝 Quick Reference

### Key Files
```
Config Files:
- config/dataset/fauna_mouse_dannce.yaml
- config/model/fauna_mouse_dannce.yaml
- config/train_fauna_mouse_dannce_debug.yaml
- config/train_fauna_mouse_dannce.yaml

Dataset:
- /home/joon/dev/data/mouse_dannce_6view/
- Symlink: data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view

Outputs:
- results/fauna_mouse_dannce_debug/ (debug)
- results/fauna_mouse_dannce_from_scratch/ (full training)
```

### Key Commands
```bash
# Debug (always first!)
python run.py --config-name train_fauna_mouse_dannce_debug

# Full training
python run.py --config-name train_fauna_mouse_dannce

# Background training
nohup python run.py --config-name train_fauna_mouse_dannce > /tmp/train.log 2>&1 &

# Visualization
python visualization/visualize_results_fauna.py \
  --config-name test_fauna \
  checkpoint_dir=results/fauna_mouse_dannce_from_scratch \
  checkpoint_name=checkpoint50000.pth
```

---

## 🎓 Additional Resources

- **[Dataset Preparation Guide](docs/FAUNA_DATASET_PREPARATION_GUIDE.md)** - How to prepare more data
- **[System Comprehensive Guide](docs/reports/251121_3danimals_system_comprehensive_guide.md)** - Full system documentation
- **[README.md](README.md)** - Project overview and FAQ

---

**Ready to train!** 🚀

Start with: `python run.py --config-name train_fauna_mouse_dannce_debug`
