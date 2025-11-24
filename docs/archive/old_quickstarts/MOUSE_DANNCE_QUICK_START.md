# Mouse DANNCE Quick Start

**Ready to train in 3 commands!** 🚀

---

## ✅ Dataset Verified

```
Dataset: mouse_dannce_6view
Location: /home/joon/dev/data/mouse_dannce_6view
Status: ✅ ALL CHECKS PASSED

Sequences: 5
Frames: 50 (10 per sequence)
Resolution: 256×256
Format: Fauna (ready to use)
```

---

## 🚀 Training in 3 Steps

### Step 1: Verify Dataset (30 seconds)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals
python scripts/verify_mouse_dannce_dataset.py
```

**Expected output**: ✅ ALL CHECKS PASSED

---

### Step 2: Debug Training (10-15 minutes)

```bash
# Activate environment
conda activate 3danimals

# Run debug mode
python run.py --config-name train_fauna_mouse_dannce_debug

# Monitor (optional)
# tensorboard --logdir results/fauna_mouse_dannce_debug/tensorboard_logs --port 6006
```

**What happens**:
- Validates configuration
- Checks GPU memory (~4GB VRAM)
- Runs 3000 iterations (~10-15 min)
- Saves checkpoints to `results/fauna_mouse_dannce_debug/`

**Success criteria**:
- ✅ No CUDA OOM errors
- ✅ Loss decreases
- ✅ Completes in 10-15 minutes

---

### Step 3: Full Training (2-3 hours)

**Only if debug succeeds!**

```bash
# Background training (recommended)
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_dannce \
  > /tmp/fauna_mouse_dannce_train.log 2>&1 &

# Monitor progress
tail -f /tmp/fauna_mouse_dannce_train.log

# Check GPU
watch -n 1 nvidia-smi
```

**What happens**:
- Trains for 50,000 iterations (~2-3 hours)
- Saves checkpoints every 5K iterations
- Logs to WandB: `fauna_mouse_dannce` project
- Final checkpoint: `results/fauna_mouse_dannce_from_scratch/checkpoint50000.pth`

---

## 📊 Monitor Training

### Option 1: Log File

```bash
tail -f /tmp/fauna_mouse_dannce_train.log
```

### Option 2: TensorBoard

```bash
tensorboard --logdir results/fauna_mouse_dannce_from_scratch/tensorboard_logs \
  --port 6006 --bind_all

# Open: http://localhost:6006
```

### Option 3: WandB (Online)

Training automatically logs to WandB project: `fauna_mouse_dannce`

---

## 🎨 Visualize Results

### After training completes:

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

### Check outputs:

```bash
ls results/fauna_mouse_dannce_from_scratch/visualization/

# You should see:
# - *_input_rgb_pred.png (reconstructions)
# - *_rotation_*.png (rotation frames)
# - *_rotation_video.mp4 (360° video)
```

---

## 📋 Complete Command Sequence

**Copy-paste ready!**

```bash
# 1. Navigate and activate
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# 2. Verify dataset
python scripts/verify_mouse_dannce_dataset.py

# 3. Debug training (wait for completion, ~15 min)
python run.py --config-name train_fauna_mouse_dannce_debug

# 4. Full training (background, ~3 hours)
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_dannce \
  > /tmp/fauna_mouse_dannce_train.log 2>&1 &

# 5. Monitor
tail -f /tmp/fauna_mouse_dannce_train.log

# 6. Visualize (after training completes)
conda run -n 3danimals python visualization/visualize_results_fauna.py \
  --config-name test_fauna \
  checkpoint_dir=results/fauna_mouse_dannce_from_scratch \
  checkpoint_name=checkpoint50000.pth \
  dataset.test_data_dir=data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view/test \
  render_modes=[input_view,rotation]
```

---

## ⏱️ Timeline

| Stage | Duration | What Happens |
|-------|----------|--------------|
| Verify | 30 sec | Dataset validation |
| Debug | 10-15 min | Config validation, test run |
| Training | 2-3 hours | Full 50K iterations |
| Visualization | 5-10 min | Generate outputs |
| **Total** | **~3 hours** | End-to-end |

---

## 🎯 Expected Results

### Training Metrics
- **Mask IoU**: 0.75-0.85
- **RGB PSNR**: 18-22 dB
- **Loss**: Steadily decreasing

### Visual Quality
- ✅ Recognizable mouse shape
- ✅ Smooth 360° rotation
- ✅ Consistent multi-view reconstruction
- ⚠️ Limited articulation (few-shot data)

---

## 🔧 Quick Troubleshooting

### CUDA Out of Memory
```bash
# Reduce grid resolution
# Edit: config/model/fauna_mouse_dannce.yaml
# Change: grid_res: 64 → 32
```

### Training Too Slow
```bash
# Check GPU utilization (should be >80%)
nvidia-smi -l 1
```

### Poor Results
- **Limited data**: 50 frames is minimal
- **Solution**: Collect 100+ frames for better quality

---

## 📚 Full Documentation

- **[Mouse DANNCE Training Guide](MOUSE_DANNCE_TRAINING_GUIDE.md)** - Complete guide
- **[Dataset Preparation Guide](docs/FAUNA_DATASET_PREPARATION_GUIDE.md)** - Add more data
- **[System Guide](docs/reports/251121_3danimals_system_comprehensive_guide.md)** - Full docs

---

## 🎉 You're Ready!

**Start now**: `python run.py --config-name train_fauna_mouse_dannce_debug`

Good luck! 🚀
