# Fauna Mouse Inference Execution Guide

**Date**: 2025-11-24
**Purpose**: Execute inference for checkpoints 10K, 30K, 50K to analyze progressive training quality

---

## Overview

After completing full training to 50,000 iterations (2025-11-24 07:00), we need to execute inference for three critical checkpoints to analyze the progressive training quality improvement:

- **checkpoint10000**: Articulation phase activated
- **checkpoint30000**: Leg attachment phase activated
- **checkpoint50000**: Final result

---

## Available Checkpoints

```bash
cd /home/joon/dev/3DAnimals/results

# Generated checkpoints:
checkpoint2000.pth   - 257M (Nov 21)
checkpoint2500.pth   - 257M (Nov 21)
checkpoint3000.pth   - 257M (Nov 21) ✅ Inference completed
checkpoint5000.pth   - 257M (Nov 22) ✅ Inference completed
checkpoint10000.pth  - 257M (Nov 24)
checkpoint15000.pth  - 257M (Nov 24)
checkpoint20000.pth  - 257M (Nov 24)
checkpoint25000.pth  - 257M (Nov 24)
checkpoint30000.pth  - 257M (Nov 24 04:51)
checkpoint35000.pth  - 257M (Nov 24 05:23)
checkpoint40000.pth  - 257M (Nov 24 05:55)
checkpoint45000.pth  - 257M (Nov 24 06:28)
checkpoint50000.pth  - 257M (Nov 24 07:00) ✅ FINAL
```

---

## Sequential Inference Execution

### Prerequisites

```bash
# Navigate to project directory
cd /home/joon/dev/3DAnimals

# Activate conda environment
conda activate 3danimals

# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

### Step 1: Inference for checkpoint10000 (Articulation Phase)

**Progressive Training Context**:
- **articulation_iter_range**: [10000, inf]
- **At 10K iterations**: Articulation (skeletal motion) activated
- **Expected**: Basic shape + articulation starting to work

**Command**:
```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint10000.pth \
  checkpoint_dir=results/infer_10k \
  output_dir=results/infer_10k
```

**Expected Output Location**:
```
results/infer_10k/test_results_*/
├── *_image_gt.png
├── *_image_pred.png
├── *_mask_gt.png
├── *_mask_pred.png
├── *_mesh.obj
└── *_pose.txt
```

**Estimated Time**: ~15-20 minutes (similar to checkpoint3000 inference)

---

### Step 2: Inference for checkpoint30000 (Leg Attachment Phase)

**Progressive Training Context**:
- **legs_iter_range**: [30000, inf]
- **At 30K iterations**: Leg attachment activated
- **Expected**: Shape + articulation + leg bones attached

**Command**:
```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint30000.pth \
  checkpoint_dir=results/infer_30k \
  output_dir=results/infer_30k
```

**Expected Output Location**:
```
results/infer_30k/test_results_*/
├── *_image_gt.png
├── *_image_pred.png
├── *_mask_gt.png
├── *_mask_pred.png
├── *_mesh.obj
└── *_pose.txt
```

**Estimated Time**: ~15-20 minutes

---

### Step 3: Inference for checkpoint50000 (Final Result)

**Progressive Training Context**:
- **Training completed**: 50,000 iterations total
- **All features active**: Shape, articulation, legs, all losses
- **Expected**: Best quality reconstruction

**Command**:
```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint50000.pth \
  checkpoint_dir=results/infer_50k \
  output_dir=results/infer_50k
```

**Expected Output Location**:
```
results/infer_50k/test_results_*/
├── *_image_gt.png
├── *_image_pred.png
├── *_mask_gt.png
├── *_mask_pred.png
├── *_mesh.obj
└── *_pose.txt
```

**Estimated Time**: ~15-20 minutes

---

## Monitoring Inference Progress

### During Execution

```bash
# Watch GPU usage
nvidia-smi -l 1

# Monitor process
ps aux | grep python

# Check output directory
watch -n 5 'ls -lh results/infer_10k/test_results_*/ 2>/dev/null | wc -l'
```

### After Completion

```bash
# Count generated files for each checkpoint
echo "=== Checkpoint 3000 ==="
find results/mouse_dannce_infer/test_results_checkpoint3000 -name "*.obj" | wc -l

echo "=== Checkpoint 5000 ==="
find results/mouse_dannce_infer_5k/test_results_None -name "*.obj" | wc -l

echo "=== Checkpoint 10000 ==="
find results/infer_10k/test_results_* -name "*.obj" | wc -l

echo "=== Checkpoint 30000 ==="
find results/infer_30k/test_results_* -name "*.obj" | wc -l

echo "=== Checkpoint 50000 ==="
find results/infer_50k/test_results_* -name "*.obj" | wc -l
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory
**Solution**: Reduce batch size in config or close other GPU processes

```bash
# Check GPU memory usage
nvidia-smi

# Kill other processes if needed
kill -9 <PID>
```

### Issue 2: Python Not Found

**Symptoms**: `/bin/bash: line 1: python: command not found`
**Solution**: Make sure conda environment is activated

```bash
conda activate 3danimals
which python  # Should show: /home/joon/miniconda3/envs/3danimals/bin/python
```

### Issue 3: Config Not Found

**Symptoms**: `Could not find config file: infer_mouse_dannce.yaml`
**Solution**: Verify working directory

```bash
pwd  # Should be: /home/joon/dev/3DAnimals
ls config/infer_mouse_dannce.yaml  # Should exist
```

---

## Expected Results Summary

| Checkpoint | Iterations | Articulation | Legs | Expected Quality |
|------------|-----------|--------------|------|------------------|
| 3000 | 3,000 | ❌ Not activated | ❌ Not activated | Blob shape |
| 5000 | 5,000 | ❌ Not activated | ❌ Not activated | Ellipsoid (initialization) |
| 10000 | 10,000 | ✅ Just activated | ❌ Not activated | Basic shape + articulation |
| 30000 | 30,000 | ✅ Active | ✅ Just activated | Shape + articulation + legs |
| 50000 | 50,000 | ✅ Fully trained | ✅ Fully trained | Best quality |

---

## Next Steps After Inference

After completing all three inferences:

1. **Quantitative Analysis**:
   - Count frames generated for each checkpoint
   - Compare mesh quality metrics (if available)
   - Analyze mask IoU and RGB PSNR

2. **Qualitative Analysis**:
   - Visual comparison of reconstruction quality
   - Before/after comparison (3K → 5K → 10K → 30K → 50K)
   - Identify visual improvements at each phase

3. **Create Comparison Report**:
   - Document: `251124_checkpoint_progressive_quality_analysis.md`
   - Include side-by-side visualizations
   - Summarize findings and quality progression

---

## Execution Checklist

- [ ] Navigate to `/home/joon/dev/3DAnimals`
- [ ] Activate `3danimals` conda environment
- [ ] Verify CUDA availability
- [ ] Execute checkpoint10000 inference
- [ ] Wait for completion (~15-20 min)
- [ ] Verify output files generated
- [ ] Execute checkpoint30000 inference
- [ ] Wait for completion (~15-20 min)
- [ ] Verify output files generated
- [ ] Execute checkpoint50000 inference
- [ ] Wait for completion (~15-20 min)
- [ ] Verify output files generated
- [ ] Count frames for all checkpoints
- [ ] Proceed to comparison report creation

---

**Estimated Total Time**: ~45-60 minutes for all three inferences

**Note**: Execute these commands **sequentially** (one at a time), not in parallel, to avoid GPU memory conflicts.
