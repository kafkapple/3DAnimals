# Multi-Animal Training Guide

**Last Updated**: 2025-11-25

---

## 📊 Training Modes Comparison

| Aspect | Mouse Only | Multi-Animal (All 8) |
|--------|-----------|---------------------|
| **Animals** | 🐭 Mouse | 🐭🐻🐄🐘🦒🐴🐑🦓 (8 animals) |
| **Total Frames** | 50 | ~63,000 |
| **Training Time** | 2-3 hours | 30-50 hours |
| **Iterations** | 50K | 200K |
| **GPU Memory** | 4-6 GB | 8-12 GB |
| **Generalization** | ⭐⭐ Low | ⭐⭐⭐⭐⭐ Excellent |
| **Use Case** | Quick prototype | Production model |

---

## 🎯 Method 1: Single Animal (Current)

### Current Status

```bash
# Check current animals
ls data/fauna/large_scale/
# Output: mouse
```

**Active**: Mouse only (50 frames)

### Keep Mouse-Only Training

**No action needed!** Already configured.

```bash
# Debug mode (15-20 min)
conda run -n 3danimals python run.py --config-name train_mouse_debug

# Full training (2-3 hours)
conda run -n 3danimals python run.py --config-name train_mouse
```

---

## 🚀 Method 2: Multi-Animal Training

### Available Animals

From `data/fauna/Fauna_dataset/large_scale/`:

| Animal | Sequences | Frames | Size |
|--------|-----------|--------|------|
| 🐻 Bear | 136 | 5,884 | Large |
| 🐄 Cow | 236 | 6,952 | Large |
| 🐘 Elephant | 393 | 15,215 | Large |
| 🦒 Giraffe | 149 | 4,775 | Large |
| 🐴 Horse | 347 | 13,718 | Large |
| 🐑 Sheep | 438 | 12,499 | Medium |
| 🦓 Zebra | 105 | 4,151 | Large |
| 🐭 Mouse (markerless) | 6 | 540 | Small |
| **Total** | **1,810** | **63,734** | - |

### Setup Multi-Animal Training

#### Option A: Interactive Setup (Recommended)

```bash
./scripts/setup_multi_animal_training.sh
```

**Wizard prompts:**
```
Setup options:
  [1] All animals (8 animals, ~63K frames)
  [2] Select specific animals
  [3] Mouse + Large animals (mouse, elephant, horse, giraffe)
  [4] Small animals only (mouse, sheep)

Choose option [1]: 1

Link method:
  [1] Symlink (fast, saves space)
  [2] Copy (slow, uses space, standalone)

Choose method [1]: 1
```

#### Option B: Manual Setup

```bash
# Symlink all animals
cd data/fauna/large_scale/

# Link individual animals
ln -sf ../Fauna_dataset/large_scale/bear_comb_dinov2_new bear
ln -sf ../Fauna_dataset/large_scale/cow_comb_dinov2_new cow
ln -sf ../Fauna_dataset/large_scale/elephant_comb_dinov2_new elephant
ln -sf ../Fauna_dataset/large_scale/giraffe_comb_dinov2_new giraffe
ln -sf ../Fauna_dataset/large_scale/horse_comb_dinov2_new horse
ln -sf ../Fauna_dataset/large_scale/sheep_comb_dinov2_new sheep
ln -sf ../Fauna_dataset/large_scale/zebra_comb_dinov2_new zebra

# Verify
ls -la
```

#### Option C: Selective Animals

**Example: Mouse + Large Animals Only**

```bash
cd data/fauna/large_scale/

ln -sf ../Fauna_dataset/large_scale/elephant_comb_dinov2_new elephant
ln -sf ../Fauna_dataset/large_scale/horse_comb_dinov2_new horse
ln -sf ../Fauna_dataset/large_scale/giraffe_comb_dinov2_new giraffe
# Keep existing mouse

ls
# Output: mouse elephant horse giraffe
```

---

## 🏃 Running Multi-Animal Training

### Step 1: Verify Setup

```bash
# Check loaded animals
conda run -n 3danimals python run.py --config-name train_mouse_debug 2>&1 | grep "using.*categories"

# Expected output:
# using 8 categories, contains: ['large_scale_bear', 'large_scale_cow', ...]
```

### Step 2: Debug Mode (1-2 hours)

```bash
# Test with all animals (5K iterations)
conda run -n 3danimals python run.py --config-name train_fauna_multi_debug
```

**Expected behavior:**
- Loads all animals
- Batches contain mixed animals
- Loss should stabilize within 5K iterations

### Step 3: Full Training (30-50 hours)

```bash
# Background training recommended
nohup conda run -n 3danimals python run.py \
  --config-name train_fauna_multi \
  > /tmp/fauna_multi_training.log 2>&1 &

echo $!  # Save PID

# Monitor
tail -f /tmp/fauna_multi_training.log
```

---

## ⚙️ Configuration

### Dataset Config Behavior

**Important**: FaunaDataset **automatically loads ALL animals** in `data/fauna/large_scale/`

```yaml
# config/dataset/mouse.yaml (or any fauna dataset config)
train_data_dir: ${oc.env:PWD}/data/fauna

# FaunaDataset will load:
# - data/fauna/large_scale/mouse/
# - data/fauna/large_scale/bear/     (if exists)
# - data/fauna/large_scale/cow/      (if exists)
# ... and so on
```

**To control which animals:**
- Add/remove symlinks in `data/fauna/large_scale/`
- FaunaDataset only loads what exists

### Batch Composition

**Multi-animal batches:**
- FaunaDataset balances categories automatically
- Each batch contains mixed animals
- Helps generalization across species

**Example batch:**
```
Batch 1: [mouse, mouse, horse, bear, cow, elephant]
Batch 2: [giraffe, zebra, sheep, mouse, horse, cow]
```

---

## 📈 Training Recommendations

### Hardware Requirements

| Setup | GPU Memory | Training Time | Recommendation |
|-------|-----------|---------------|----------------|
| Mouse only | 4-6 GB | 2-3 hours | RTX 3060+ |
| 2-3 animals | 6-8 GB | 8-12 hours | RTX 3060 Ti+ |
| All 8 animals | 10-12 GB | 30-50 hours | RTX 3080+ |

### Iteration Guidelines

| Dataset Size | Iterations | Time (RTX 3060) |
|--------------|-----------|-----------------|
| 50 frames | 50K | 2-3 hours |
| 500 frames | 100K | 10-15 hours |
| 5,000 frames | 150K | 25-35 hours |
| 60,000+ frames | 200K | 40-50 hours |

### Learning Rate Adjustment

**For multi-animal training**, consider:

```yaml
# config/train_fauna_multi.yaml
optimizer:
  lr: 0.0001  # Lower LR for stability with diverse data
```

---

## 🔍 Monitoring

### Check Loaded Animals

```bash
# During training, check first few lines:
conda run -n 3danimals python run.py --config-name train_mouse 2>&1 | head -100

# Look for:
# "using 8 categories, contains: ['large_scale_bear', 'large_scale_cow', ...]"
```

### WandB Metrics

**Important metrics for multi-animal:**
- Loss curves per category (if logged)
- Mixed batch performance
- Convergence speed

### Training Logs

```bash
# Check category distribution
grep "using.*categories" /tmp/fauna_multi_training.log

# Check iteration speed
grep "Hz" /tmp/fauna_multi_training.log | tail -20
```

---

## 🎨 Use Cases

### When to Use Mouse Only

✅ **Use single animal when:**
- Quick prototyping
- Testing new features
- Limited compute resources
- Animal-specific fine-tuning
- Debugging

### When to Use Multi-Animal

✅ **Use multi-animal when:**
- Production model development
- Strong generalization needed
- Transfer learning
- Robust to animal variation
- Research on animal motion

---

## 🔄 Switching Between Modes

### Mouse Only → Multi-Animal

```bash
# 1. Setup animals
./scripts/setup_multi_animal_training.sh

# 2. Train
conda run -n 3danimals python run.py --config-name train_fauna_multi_debug
```

### Multi-Animal → Mouse Only

```bash
# 1. Remove other animals
cd data/fauna/large_scale/
rm -rf bear cow elephant giraffe horse sheep zebra

# 2. Keep only mouse
ls
# Output: mouse

# 3. Train
conda run -n 3danimals python run.py --config-name train_mouse
```

### Selective Animals

```bash
# Keep only mouse + horse + elephant
cd data/fauna/large_scale/
rm -rf bear cow giraffe sheep zebra

ls
# Output: mouse horse elephant
```

---

## ⚠️ Important Notes

### Symlink vs Copy

**Symlink (Recommended):**
- ✅ Fast setup (seconds)
- ✅ Saves disk space
- ❌ Requires source availability
- Use for: Development, testing

**Copy:**
- ✅ Standalone (no dependencies)
- ✅ Portable
- ❌ Slow (minutes to hours)
- ❌ Uses 50-100GB disk
- Use for: Production, deployment

### Data Loading

**FaunaDataset behavior:**
```python
# Automatically loads ALL directories in:
data/fauna/large_scale/
  ├── mouse/     → 'large_scale_mouse'
  ├── bear/      → 'large_scale_bear'
  └── horse/     → 'large_scale_horse'
```

**Cannot selectively exclude** via config - control by adding/removing directories.

### Config Reuse

**Same configs work for both modes!**

```bash
# These work with any number of animals:
train_mouse.yaml
train_mouse_debug.yaml
train_fauna_multi.yaml
train_fauna_multi_debug.yaml
```

The only difference: which animals exist in `data/fauna/large_scale/`

---

## 🚨 Troubleshooting

### Issue: "using 1 categories" when expecting more

**Cause**: Animals not properly linked/copied

**Fix:**
```bash
ls data/fauna/large_scale/
# Should show all expected animals

# If missing, re-run setup
./scripts/setup_multi_animal_training.sh
```

### Issue: CUDA Out of Memory with multi-animal

**Fix:**
```yaml
# config/model/mouse.yaml (or fauna.yaml)
cfg_predictor_base:
  cfg_shape:
    grid_res: 32  # Reduce from 64
```

### Issue: Training too slow

**Check:**
```bash
# Iteration speed
grep "Hz" /tmp/training.log | tail -10

# Expected: 2-4 Hz for multi-animal
# If < 1 Hz: Check GPU utilization, reduce workers
```

---

## 📚 Further Reading

- [Fauna Dataset Complete Guide](FAUNA_DATASET_COMPLETE_GUIDE.md)
- [Mouse Training Guide](MOUSE_TRAINING_GUIDE.md)
- [Config Structure Guide](PROJECT_STRUCTURE_GUIDE.md)

---

**Questions?** Check existing configs and scripts for examples!
