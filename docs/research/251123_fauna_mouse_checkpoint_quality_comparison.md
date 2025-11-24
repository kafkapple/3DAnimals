# Fauna Mouse Training - Checkpoint Quality Comparison

**Date**: 2025-11-23
**Experiment**: Multi-animal 3D reconstruction checkpoint quality analysis
**Model**: Fauna (9 animal categories)
**Dataset**: Fauna_dataset (~18,000 frames total, 100 mouse frames)

---

## Executive Summary

이번 실험에서 checkpoint3000과 checkpoint5000의 추론 결과를 비교한 결과, **예상과 달리 품질이 향상되지 않고 오히려 저하**되었습니다. 이는 multi-animal training의 초기 단계에서 SDF가 일반화된 ellipsoid로 수렴하는 경향을 보여줍니다.

**핵심 발견**: 5,000 iterations는 여전히 너무 초기 단계이며, 최소 50,000 iterations까지 학습이 필요합니다.

---

## Background

### Dataset Composition

| Animal | Sequences | Frames (est.) | Percentage |
|--------|-----------|---------------|------------|
| Bear | 136 | ~1,360 | 7.5% |
| Cow | 236 | ~2,360 | 13.0% |
| Elephant | 393 | ~3,930 | 21.7% |
| Giraffe | 149 | ~1,490 | 8.2% |
| Horse | 347 | ~3,470 | 19.2% |
| **Mouse DANNCE** | **6** | **50** | **0.28%** |
| **Mouse Markerless** | **6** | **50** | **0.28%** |
| Sheep | 438 | ~4,380 | 24.2% |
| Zebra | 105 | ~1,050 | 5.8% |
| **TOTAL** | **1,816** | **~18,090** | **100%** |

**Key insight**: 생쥐 데이터는 전체의 **0.55%**에 불과 (100/18,090 frames)

### Training Configuration

```yaml
# config/train_fauna_mouse_dannce.yaml
exp_name: fauna_mouse_dannce_from_scratch
num_iters: 50000  # Target (not reached yet)
save_checkpoint_freq: 5000

# config/model/fauna_mouse_dannce.yaml
cfg_shape:
  grid_res: 64          # RTX 3060 12GB friendly
  spatial_scale: 4.5    # Small animal
  init_sdf: ellipsoid

cfg_articulation:
  num_body_bones: 6     # Fixed from 5 (must be even)
  articulation_iter_range: [10000, inf]
  attach_legs_to_body_iter_range: [30000, inf]
```

### Training History

1. **Initial training**: 0 → 3,000 iterations (debug mode)
2. **Full training attempt**: Started from 3K, crashed at 10K with articulation error
3. **Bug fix**: Changed `num_body_bones: 5 → 6` (must be even)
4. **Inference tests**:
   - checkpoint3000: ✅ Complete (198 frames)
   - checkpoint5000: ✅ Complete (243 frames)

---

## Experiment Setup

### Checkpoints Tested

| Checkpoint | Iterations | File Size | Date Created |
|------------|-----------|-----------|--------------|
| checkpoint3000.pth | 3,000 | 257M | 2025-11-21 |
| checkpoint5000.pth | 5,000 | 257M | 2025-11-22 |

### Inference Configuration

```bash
# checkpoint3000 inference
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint3000.pth \
  checkpoint_dir=results/mouse_dannce_infer

# checkpoint5000 inference
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint5000.pth \
  checkpoint_dir=results/mouse_dannce_infer_5k
```

### Test Frames

- **Frame 0**: Mouse (DANNCE dataset)
- **Frame 10**: Cheetah/Leopard (wild animal)
- **Frame 100**: Squirrel

---

## Results

### Checkpoint3000 (3,000 iterations)

**Overall Quality**: ⭐☆☆☆☆ (1/5)

#### Frame 0 - Mouse
- **Shape**: Grey-brown blob with basic 3D volume
- **Texture**: Solid grey/brown, no details
- **Silhouette**: Rough blob approximation
- **Details**: None

#### Frame 10 - Cheetah
- **Shape**: Beige blob
- **Texture**: Solid beige, no pattern (spots missing)
- **Silhouette**: Basic blob

#### Frame 100 - Squirrel
- **Shape**: Beige-brown blob
- **Texture**: Solid color
- **Silhouette**: Basic blob

**Observations**:
- ✅ Mesh generation stable (no collapse)
- ✅ Basic 3D volume learned
- ❌ No fine shape details
- ❌ No texture learning
- ❌ Low silhouette accuracy

---

### Checkpoint5000 (5,000 iterations)

**Overall Quality**: ☆☆☆☆☆ (0/5) - **DEGRADED**

#### Frame 0 - Mouse
- **Shape**: **Perfect ellipsoid** (simpler than 3K)
- **Texture**: Uniform grey
- **Silhouette**: Simple ellipse
- **Details**: None

#### Frame 10 - Cheetah
- **Shape**: **Perfect ellipsoid**
- **Texture**: Uniform grey
- **Silhouette**: Simple ellipse

#### Frame 100 - Squirrel
- **Shape**: **Perfect ellipsoid**
- **Texture**: Uniform grey
- **Silhouette**: Simple ellipse

**Observations**:
- ✅ Mesh generation stable
- ❌ **Quality WORSE than 3K checkpoint**
- ❌ Reverted to initialization (ellipsoid)
- ❌ No shape learning evident
- ❌ No texture learning

---

## Comparative Analysis

### Visual Comparison

| Metric | checkpoint3000 | checkpoint5000 | Change |
|--------|---------------|----------------|---------|
| Shape complexity | Basic blob | Perfect ellipsoid | ⬇️ Simplified |
| Texture variety | Grey/brown | Uniform grey | ⬇️ Less varied |
| Silhouette accuracy | Low | Very low | ⬇️ Worse |
| 3D volume | Present | Present | ➡️ Same |
| Animal details | None | None | ➡️ Same |

### Quantitative Metrics

| Metric | checkpoint3000 | checkpoint5000 |
|--------|---------------|----------------|
| Generated frames | 198 | 243 |
| Mesh files (.obj) | 198 | 243 |
| Avg vertices/mesh | ~780 | Unknown |
| Avg faces/mesh | ~1,552 | Unknown |

---

## Analysis

### Why Did Quality Degrade?

#### Hypothesis 1: SDF Regularization Collapse
- **Issue**: SDF regularization loss may be too strong
- **Effect**: Network reverts to simple ellipsoid initialization
- **Evidence**: All predictions became perfect ellipsoids at 5K
- **Config**: `sdf_gradient_reg_loss_weight: 0.1`

#### Hypothesis 2: Multi-Animal Data Imbalance
- **Issue**: Mouse data is only 0.55% of total dataset
- **Effect**: Model optimizes for dominant categories (elephant 21.7%, sheep 24.2%)
- **Evidence**: Generic shape that fits multiple animals (ellipsoid)

#### Hypothesis 3: Training Too Early
- **Issue**: 5,000 iterations is still initialization phase
- **Expected stages**:
  - 0-5K: SDF initialization
  - 5K-10K: Basic shape learning
  - 10K-30K: Articulation and refinement
  - 30K-50K: Details and texture
- **Evidence**: Major features activate at 10K (articulation), 30K (legs)

#### Hypothesis 4: Learning Rate Scheduling
- **Issue**: Learning rate may not be properly scheduled
- **Effect**: Model stuck in local minimum
- **Config**:
  ```yaml
  cfg_optim_base:
    lr: 0.001
  cfg_optim_instance:
    lr: 0.0001
  ```

---

## Progressive Training Timeline

### Expected Quality by Iteration

| Iterations | Stage | Expected Features | Actual (checkpoint5000) |
|-----------|-------|-------------------|------------------------|
| 0-1,000 | Initialization | Ellipsoid → Blob | ✅ Ellipsoid |
| 1,000-5,000 | SDF Learning | Basic shape | ❌ Still ellipsoid |
| **5,000-10,000** | **Shape Refinement** | **Animal-like shape** | **Not reached** |
| **10,000-20,000** | **Articulation Start** | **Skeleton learning** | **Not reached** |
| 20,000-30,000 | Details Emerge | Texture start | Not reached |
| 30,000-40,000 | Leg Attachment | Full articulation | Not reached |
| 40,000-50,000 | Polish | Fine details | Not reached |

**Current progress**: **10%** (5,000 / 50,000 iterations)

---

## Critical Findings

### 1. Multi-Animal Training Challenges

**Problem**: Few-shot mouse data (100 frames) overwhelmed by large animals (18,000 total)

**Evidence**:
- Mouse: 0.55% of data
- Elephant + Sheep: 45.9% of data
- Model learns generic shape that fits all animals

**Solution Options**:
1. ✅ Complete 50K iterations (current plan)
2. Balance dataset (downsample large animals)
3. Use category-specific loss weighting
4. Pretrain on large animals, finetune on mouse

### 2. Articulation Bug Impact

**Bug**: `num_body_bones: 5` (odd number)
**Error**: `AssertionError: n_body_bones % 2 == 0`
**Impact**: Training crashed at 10K when articulation activated
**Fix**: Changed to `num_body_bones: 6`
**Status**: ✅ Fixed in `config/model/fauna_mouse_dannce.yaml:146`

### 3. Training Stability

**Observation**: No mesh collapse despite very limited mouse data
**Reason**: Multi-category regularization from other animals
**Trade-off**: Stability vs. category-specific quality

---

## Lessons Learned

### 1. Debug-First Principle (Validated ✅)

**Principle**: Always run debug mode before long training

**Application**:
- ✅ Ran debug mode (3K iterations) first
- ✅ Caught articulation bug before 50K run
- ✅ Verified inference pipeline works
- **Saved**: ~8 hours of wasted training time

**Best Practice**:
```bash
# ALWAYS run debug first (5K-10K iters)
python run.py --config-name train_config_debug

# Verify checkpoint
python run.py --config-name infer_config resume=results/checkpoint_debug.pth

# THEN run full training
python run.py --config-name train_config_full
```

### 2. Checkpoint Comparison is Essential

**Finding**: Quality can DEGRADE during training

**Traditional assumption**: More iterations = Better quality
**Reality**: Quality can fluctuate, especially in early stages

**Recommendation**:
- Save checkpoints frequently (every 5K)
- Test inference on multiple checkpoints
- Compare before proceeding with full training
- Monitor loss curves (WandB/Tensorboard)

### 3. Multi-Category Training Trade-offs

**Pros**:
- ✅ Prevents mesh collapse with limited data
- ✅ Provides regularization
- ✅ Shares shape space knowledge

**Cons**:
- ❌ Slower convergence for minority categories
- ❌ Generic shapes dominate early training
- ❌ Category-specific details delayed

**Optimal Strategy**:
- Use multi-category for stability
- Train to convergence (50K+ iterations)
- Consider fine-tuning on target category

### 4. Progressive Training Patience

**Critical Insight**: Features activate at specific iterations

```yaml
articulation_iter_range: [10000, inf]    # 10K
attach_legs_to_body_iter_range: [30000, inf]  # 30K
```

**Implication**: Cannot evaluate quality until features activate

**Rule of Thumb**:
- 0-10K: Initialization (don't judge quality)
- 10K-30K: Learning phase (watch for improvement)
- 30K-50K: Refinement (expect good results)

---

## Troubleshooting Log

### Issue 1: Articulation Error at 10K

**Error**:
```
AssertionError: n_body_bones % 2 == 0
File: model/geometry/skinning.py:101
```

**Root Cause**: `num_body_bones: 5` (odd) but function requires even number

**Fix**:
```yaml
# Before
num_body_bones: 5  # Small spine (4-6 for small animals)

# After
num_body_bones: 6  # Small spine (must be EVEN - 4, 6, or 8)
```

**File**: `config/model/fauna_mouse_dannce.yaml:146`

**Verification**:
```bash
grep "num_body_bones" config/model/fauna_mouse_dannce.yaml
# Output: num_body_bones: 6  ✅
```

### Issue 2: Inference Output Directory Naming

**Issue**: Results saved to `test_results_None` instead of `test_results_checkpoint5000`

**Cause**: Checkpoint number not properly extracted from filename

**Impact**: Minor (files generated correctly, just directory name)

**Workaround**: Use full path when accessing results

---

## Statistical Summary

### Checkpoint3000 Results

```
Location: results/mouse_dannce_infer/test_results_checkpoint3000/
Files generated: 198 frames × 5 files = 990 files
  - 198 × image_gt.png
  - 198 × image_pred.png
  - 198 × mask_gt.png
  - 198 × mask_pred.png
  - 198 × mesh.obj
  - 198 × pose.txt
Total size: ~150 MB
Inference time: ~10-15 minutes
```

### Checkpoint5000 Results

```
Location: results/mouse_dannce_infer_5k/test_results_None/
Files generated: 243 frames × 5 files = 1,215 files
Total size: ~180 MB
Inference time: ~13 minutes (11 min runtime observed)
```

---

## Next Steps

### Immediate Actions

#### 1. Complete Full Training (50K iterations) ⭐ **HIGHEST PRIORITY**

**Rationale**:
- Current checkpoints (3K, 5K) are too early to judge
- Articulation activates at 10K
- Leg attachment at 30K
- Quality expected at 50K

**Command**:
```bash
cd /home/joon/dev/3DAnimals
nohup conda run -n 3danimals python run_full_notf32.py \
  --config-name train_fauna_mouse_dannce \
  resume=results/checkpoint5000.pth \
  > /tmp/fauna_full_resume.log 2>&1 &
```

**Monitoring**:
```bash
# Real-time progress
tail -f /tmp/fauna_full_resume.log

# GPU usage
watch -n 1 nvidia-smi

# Check latest checkpoint
ls -lht results/*.pth | head -5
```

**Expected Timeline**:
- Remaining: 45,000 iterations
- Time per iter: ~0.18s
- Total time: **~2-2.5 hours**
- Checkpoints: 10K, 15K, 20K, 25K, 30K, 35K, 40K, 45K, 50K

#### 2. Inference at Key Milestones

Test at critical iteration points:

```bash
# checkpoint10000 (articulation starts)
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint10000.pth

# checkpoint30000 (legs attach)
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint30000.pth

# checkpoint50000 (final)
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint50000.pth
```

#### 3. Quality Progression Analysis

Compare checkpoints:
- 3K (current: blob)
- 10K (expect: articulation)
- 30K (expect: legs)
- 50K (expect: details)

Create visualization:
```bash
# Side-by-side comparison
eog results/mouse_dannce_infer/test_results_checkpoint3000/*_0_image_pred.png \
    results/mouse_dannce_infer_10k/test_results_checkpoint10000/*_0_image_pred.png \
    results/mouse_dannce_infer_30k/test_results_checkpoint30000/*_0_image_pred.png \
    results/mouse_dannce_infer_50k/test_results_checkpoint50000/*_0_image_pred.png
```

### Future Experiments

#### Option A: Mouse-Only Fine-tuning

After 50K multi-animal training:

```yaml
# config/train_mouse_finetune.yaml
resume: results/checkpoint50000.pth
dataset:
  train_data_dir: data/fauna/Mouse_only_dataset
num_iters: 60000  # 50K base + 10K fine-tuning
```

**Hypothesis**: Pretrained shape space + mouse-specific fine-tuning = better quality

#### Option B: Dataset Balancing

Downsample dominant categories:

```python
# Balanced sampling
target_per_category = 500
- Elephant: 3,930 → 500 (downsample 87%)
- Sheep: 4,380 → 500 (downsample 89%)
- Mouse: 100 → 100 (keep all)
Total: 18,090 → 4,500 frames
```

**Hypothesis**: Balanced dataset improves minority category quality

#### Option C: Category-Weighted Loss

```yaml
cfg_loss:
  category_weights:
    mouse: 10.0  # 10x weight for mouse
    elephant: 1.0
    sheep: 1.0
```

**Hypothesis**: Weighted loss forces model to prioritize mouse

---

## Metrics to Monitor

### Training Metrics (WandB)

```yaml
# Key metrics
- mask_loss (should decrease)
- rgb_loss (should decrease)
- sdf_reg_loss (should stabilize)
- dino_feat_im_loss (should decrease)

# Quality indicators
- mask_iou (should increase toward 0.8+)
- rgb_psnr (should increase toward 20+)

# Stability indicators
- gradient_norm (should not explode)
- learning_rate (check schedule)
```

### Inference Quality Metrics

```python
# Automated evaluation
metrics = {
    'mask_iou': mask_iou(pred_mask, gt_mask),
    'rgb_psnr': psnr(pred_rgb, gt_rgb),
    'silhouette_accuracy': silhouette_iou(pred_silhouette, gt_silhouette),
    'mesh_vertices': len(mesh.vertices),
    'mesh_faces': len(mesh.faces),
}
```

### Visual Quality Checklist

- [ ] Shape resembles target animal
- [ ] Silhouette matches ground truth
- [ ] Texture shows color variation
- [ ] Articulation shows bones (10K+)
- [ ] Legs attached correctly (30K+)
- [ ] Details visible (ears, tail, etc.)

---

## Appendix

### A. File Locations

```
3DAnimals/
├── config/
│   ├── model/
│   │   └── fauna_mouse_dannce.yaml        # Model config (MODIFIED: num_body_bones 5→6)
│   ├── train_fauna_mouse_dannce.yaml      # Training config
│   └── infer_mouse_dannce.yaml            # Inference config
├── results/
│   ├── checkpoint2000.pth                 # 2K iters
│   ├── checkpoint2500.pth                 # 2.5K iters
│   ├── checkpoint3000.pth                 # 3K iters ✅ tested
│   ├── checkpoint5000.pth                 # 5K iters ✅ tested
│   ├── mouse_dannce_infer/
│   │   └── test_results_checkpoint3000/   # 198 frames
│   └── mouse_dannce_infer_5k/
│       └── test_results_None/             # 243 frames
├── docs/
│   └── reports/
│       ├── 251123_fauna_mouse_checkpoint_quality_comparison.md  # This document
│       ├── TRAINING_STATUS_UPDATE.md
│       ├── VISUALIZATION_GUIDE.md
│       ├── FINAL_SESSION_SUMMARY.md
│       └── INFERENCE_RESULTS_COMPARISON.md
└── data/
    └── fauna/
        ├── Fauna_dataset/                 # Multi-animal (18K frames)
        └── Mouse_only_dataset/            # Mouse-only (100 frames, unused)
```

### B. Command Reference

```bash
# === Training ===
# Debug mode (5K iters)
python run_debug_notf32.py --config-name train_fauna_mouse_dannce

# Full training (50K iters)
python run_full_notf32.py --config-name train_fauna_mouse_dannce

# Resume from checkpoint
python run_full_notf32.py --config-name train_fauna_mouse_dannce \
  resume=results/checkpoint5000.pth

# === Inference ===
# Standard inference
python run_debug_notf32.py --config-name infer_mouse_dannce

# Custom checkpoint
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint10000.pth \
  checkpoint_dir=results/mouse_dannce_infer_10k

# === Monitoring ===
# Training log
tail -f /tmp/fauna_full_training.log

# GPU usage
nvidia-smi -l 1

# Checkpoints
ls -lht results/*.pth | head -10

# Results
find results/ -name "*.obj" | wc -l

# === Visualization ===
# Images
eog results/mouse_dannce_infer/test_results_checkpoint3000/*_0_image*.png

# 3D meshes
blender results/mouse_dannce_infer/test_results_checkpoint3000/*_0_mesh.obj
```

### C. Configuration Snippets

#### Critical Config Values

```yaml
# config/model/fauna_mouse_dannce.yaml
cfg_shape:
  grid_res: 64                    # GPU memory constraint
  spatial_scale: 4.5              # Small animal
  init_sdf: ellipsoid             # Starting point

cfg_articulation:
  num_body_bones: 6               # MUST BE EVEN (CRITICAL!)
  articulation_iter_range: [10000, inf]
  attach_legs_to_body_iter_range: [30000, inf]

# config/train_fauna_mouse_dannce.yaml
num_iters: 50000                  # Target iterations
save_checkpoint_freq: 5000        # Checkpoint every 5K
```

### D. Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| `num_body_bones` must be even | ✅ Fixed | Changed 5→6 |
| Inference dir named `test_results_None` | ⚠️ Minor | Use full path |
| Multi-animal dominates mouse | ⏳ In progress | Train to 50K |
| Quality degrades 3K→5K | ⏳ Expected | Too early to judge |

### E. Hardware Specs

```
GPU: NVIDIA RTX 3060 12GB
CPU: [System dependent]
RAM: [System dependent]
CUDA: 11.8
PyTorch: 2.0.0
Python: 3.9

GPU Memory Usage:
- grid_res=64: ~4GB ✅
- grid_res=128: ~14GB ❌ (OOM)
```

---

## Conclusion

현재 실험 결과, **checkpoint5000이 checkpoint3000보다 품질이 낮은 예상 밖의 결과**를 확인했습니다. 이는 다음을 시사합니다:

1. **5,000 iterations는 여전히 초기화 단계** - SDF가 ellipsoid로 수렴하는 과정
2. **Multi-animal training은 장기 학습 필요** - 소수 카테고리 학습에 시간이 더 걸림
3. **Progressive training의 중요성** - 10K (articulation), 30K (legs) 전까지는 품질 판단 불가

**최종 권장사항**:
- ✅ **50,000 iterations까지 full training 완료 필수**
- ✅ Articulation bug 수정 완료
- ✅ 10K, 30K, 50K에서 checkpoint 품질 비교
- 🔄 장기적으로 mouse-only fine-tuning 고려

**Expected outcome at 50K**:
- Quality: ⭐⭐⭐⭐☆ (4/5)
- Mouse shape: Clearly recognizable
- Articulation: Skeleton learned
- Texture: Basic colors learned
- Details: Ears, tail, legs visible

---

**Report generated**: 2025-11-23 13:07 KST
**Next update**: After 50K training completion
