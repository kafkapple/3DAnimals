# Mouse DANNCE Training - Fixed Commands

**Status**: ✅ Configuration Fixed and Ready

---

## ⚠️ Important Fixes Applied

1. **Model name**: Changed to `Fauna` (matches registry)
2. **Dataset path**: Set to Fauna dataset root
3. **Loading**: System loads all Fauna animals (141 categories total including mouse_dannce)

---

## 🚀 Training Commands

### Step 1: Debug Mode (10-15 minutes)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals
python run.py --config-name train_fauna_mouse_dannce_debug
```

**Expected Output**:
```
Loading training data from /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset
using 141 categories, contains: [...'large_scale_mouse'...]
Archiving code to results/archived_code.zip
Resetting optimizers...
```

### Step 2: Monitor Progress

```bash
# Check tensorboard logs
tensorboard --logdir results/fauna_mouse_dannce_debug/tensorboard_logs --port 6006

# Or check GPU
watch -n 1 nvidia-smi
```

### Step 3: Full Training (2-3 hours)

**After debug succeeds:**

```bash
# Background training
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_dannce \
  > /tmp/fauna_mouse_dannce_train.log 2>&1 &

# Monitor
tail -f /tmp/fauna_mouse_dannce_train.log
```

---

## 📊 What's Happening

The system loads **all Fauna datasets** (141 categories) including:
- `large_scale_mouse` (your mouse_dannce data)
- Other animals (bear, cow, elephant, etc.)

This is **normal behavior** for Fauna - it trains on multiple animals simultaneously for better generalization.

**Your mouse_dannce data is included** as `large_scale_mouse` in the training mix.

---

## 🎯 Training Progress

### Debug Mode Milestones

| Iteration | Time | Expected |
|-----------|------|----------|
| 0-500 | 2-3 min | Initial setup, SDF init |
| 500-1000 | 5 min | First checkpoint saved |
| 1000-2000 | 8 min | Shape learning |
| 2000-3000 | 12 min | Pose learning |
| **3000** | **15 min** | **Debug complete** |

### Full Training Milestones

| Iteration | Time | Expected |
|-----------|------|----------|
| 0-5K | 30 min | SDF init → mouse shape |
| 5K-10K | 1 hour | Articulation starts |
| 10K-30K | 2 hours | Skeleton learning |
| 30K-50K | 3 hours | Full refinement |
| **50K** | **~3 hours** | **Training complete** |

---

## 🔍 Verification

### Check Dataset Loading

The mouse data should appear in the category list:
```bash
python run.py --config-name train_fauna_mouse_dannce_debug 2>&1 | grep "large_scale_mouse"
```

Expected output:
```
...'large_scale_mouse'...
```

### Check Checkpoints

```bash
ls -lh results/fauna_mouse_dannce_debug/checkpoint*.pth
```

Expected:
```
checkpoint500.pth
checkpoint1000.pth
...
checkpoint3000.pth
```

---

## 📈 Monitor Training

### TensorBoard

```bash
tensorboard --logdir results/fauna_mouse_dannce_debug/tensorboard_logs \
  --port 6006 --bind_all

# Open: http://localhost:6006
```

### GPU Usage

```bash
watch -n 1 nvidia-smi
```

Expected GPU usage: ~4-6GB VRAM

---

## ✅ Success Criteria

### Debug Mode Success
- [ ] Completes 3000 iterations (15 min)
- [ ] No CUDA OOM errors
- [ ] Checkpoints save successfully
- [ ] Loss decreases
- [ ] GPU utilization >80%

### Ready for Full Training
If all above checks pass:
```bash
nohup python run.py --config-name train_fauna_mouse_dannce \
  > /tmp/train.log 2>&1 &
```

---

## 🔧 Troubleshooting

### Issue: CUDA OOM

**Solution 1**: Reduce grid resolution
```bash
# Edit: config/model/fauna_mouse_dannce.yaml
# Change: grid_res: 64 → 32
```

**Solution 2**: Reduce batch size
```bash
# Edit: config/dataset/fauna_mouse_dannce.yaml
# Change: batch_size: 2 → 1
```

### Issue: Training Too Slow

**Check GPU utilization**:
```bash
nvidia-smi -l 1
```

If <80%, possible causes:
- CPU bottleneck
- Disk I/O slow
- Data loading issues

---

## 📝 Configuration Summary

### Files Created
- `config/dataset/fauna_mouse_dannce.yaml` ✅
- `config/model/fauna_mouse_dannce.yaml` ✅ (name: Fauna)
- `config/train_fauna_mouse_dannce_debug.yaml` ✅
- `config/train_fauna_mouse_dannce.yaml` ✅

### Key Settings
```yaml
# Model
name: Fauna  # Fixed!
spatial_scale: 4.5
grid_res: 64
num_body_bones: 5

# Dataset
data_type: fauna
batch_size: 2
train_data_dir: /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset  # Fixed!

# Training
num_iters: 3000 (debug) / 50000 (full)
save_checkpoint_freq: 500 (debug) / 5000 (full)
```

---

## 🎉 Ready to Train!

**Start now**:
```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals
python run.py --config-name train_fauna_mouse_dannce_debug
```

Good luck! 🚀
