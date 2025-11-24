# CUDA Error Fix Guide

## Error
```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgemm(...)`
```

## System Info
- GPU: RTX 3060 (12GB)
- Driver CUDA: 12.4
- PyTorch CUDA: 11.3
- PyTorch: 1.10.0

## Root Cause
CUBLAS error is typically caused by:
1. PyTorch/CUDA version mismatch
2. Batch size too large for specific operations
3. Mixed precision issues

## Solutions Applied

### Solution 1: Reduce Batch Size ✅

**Changed**: `batch_size: 2` → `batch_size: 1`

**File**: `config/dataset/fauna_mouse_dannce.yaml`

**Rationale**: CUBLAS sometimes fails with larger batches on certain GPU/CUDA combinations

### Solution 2: CUDA Environment Variables

**Option A**: Export before running
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python run.py --config-name train_fauna_mouse_dannce_debug
```

**Option B**: Use the provided script
```bash
bash /tmp/train_mouse_dannce_debug.sh
```

### Solution 3: Update PyTorch (If needed)

**Current**: PyTorch 1.10.0 + CUDA 11.3

**Recommended**: Update to match driver CUDA 12.4

```bash
# Option 1: PyTorch with CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Option 2: PyTorch with CUDA 11.8 (more stable)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

⚠️ **Warning**: Updating PyTorch may require reinstalling other dependencies

## Recommended Action

### Try First (Quickest)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# Set environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run with reduced batch size (already applied)
python run.py --config-name train_fauna_mouse_dannce_debug
```

### If Still Fails

**Update PyTorch to CUDA 11.8**:
```bash
conda activate 3danimals
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Expected: PyTorch: 2.0.0, CUDA: 11.8

# Then retry training
python run.py --config-name train_fauna_mouse_dannce_debug
```

## Testing

### Quick CUDA Test

```bash
conda activate 3danimals
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test tensor operation
x = torch.randn(2, 3, 256, 256).cuda()
y = torch.randn(2, 3, 256, 256).cuda()
z = torch.matmul(x.view(2, -1), y.view(2, -1).T)
print('✅ CUDA operations working!')
"
```

### Test DINO Encoder (Where error occurred)

```bash
conda activate 3danimals
python -c "
import torch
import torch.nn as nn

# Simulate DINO attention operation
B, N, C = 2, 197, 384  # Batch, Num patches, Channels
x = torch.randn(B, N, C).cuda()

# Linear layer (where error occurred)
qkv = nn.Linear(C, C * 3).cuda()
output = qkv(x)
print('✅ DINO-like operation working!')
"
```

## Current Configuration

**Modified Files**:
- `config/dataset/fauna_mouse_dannce.yaml` - batch_size: 1

**Environment Variables** (recommended):
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

## Expected Behavior After Fix

```
Loading training data from /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset
using 141 categories, contains: ['large_scale_mouse', ...]
Archiving code to results/archived_code.zip
Resetting optimizers...
[Training starts without CUDA errors]
T000000: loss=X.XX (training progresses)
```

## Alternative: Use CPU (Not Recommended)

If CUDA continues to fail:
```bash
# Edit config to use CPU (VERY SLOW, not recommended)
# config/train_fauna_mouse_dannce_debug.yaml
device: cpu
```

**Note**: This will be 50-100x slower. Only use as last resort.

---

## Summary

**Immediate Fix Applied**: ✅ Reduced batch_size to 1

**Next Steps**:
1. Try running with environment variables
2. If fails, update PyTorch to 2.0.0 + CUDA 11.8
3. Verify CUDA test passes
4. Retry training

**Command to Run Now**:
```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python run.py --config-name train_fauna_mouse_dannce_debug
```
