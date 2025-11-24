# CUDA Error - Final Fix Guide

## Problem Diagnosis

### Version Mismatch
- **PyTorch**: Built with CUDA 11.3
- **System NVCC**: CUDA 11.8
- **NVIDIA Driver**: CUDA 12.4
- **GPU**: RTX 3060 (sm_86)

### Error Location
```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED
Location: DINO ViT attention layer (qkv linear projection)
```

## Root Cause

**CUBLAS library mismatch** between PyTorch's CUDA 11.3 and system CUDA 11.8/12.4.

## Solutions (Try in Order)

### Solution 1: Reinstall PyTorch with Correct CUDA ✅

**Current environment has Python 3.9**, so we can use PyTorch 1.13:

```bash
conda activate 3danimals

# Uninstall current PyTorch
conda remove pytorch torchvision torchaudio --yes

# Install PyTorch 1.13 with CUDA 11.8 (matches system NVCC)
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected output**:
```
PyTorch: 1.13.1+cu118
CUDA: 11.8
GPU: NVIDIA GeForce RTX 3060
```

**Then retry training**:
```bash
cd /home/joon/dev/3DAnimals
python run_debug_notf32.py
```

---

### Solution 2: Force CPU Mode (Emergency Fallback)

If CUDA continues to fail, use CPU mode (SLOW but works):

**Edit**: `config/train_fauna_mouse_dannce_debug.yaml`
```yaml
# Change:
device: cuda
# To:
device: cpu
```

**Warning**:
- ~50-100x slower than GPU
- Debug mode: ~2-3 hours instead of 15 minutes
- Not recommended for full training

---

### Solution 3: Use LD_LIBRARY_PATH Override

Force PyTorch to use system CUDA libraries:

```bash
# Find PyTorch CUDA libs
PYTORCH_CUDA=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib

# Find system CUDA libs
SYSTEM_CUDA=/usr/local/cuda-11.8/lib64

# Run with override
LD_LIBRARY_PATH=$SYSTEM_CUDA:$PYTORCH_CUDA:$LD_LIBRARY_PATH \
  python run_debug_notf32.py
```

---

### Solution 4: Downgrade CUDA Driver (Not Recommended)

Install CUDA 11.3 toolkit to match PyTorch:

```bash
# Download CUDA 11.3
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run

# Install (DO NOT install driver)
sudo sh cuda_11.3.0_465.19.01_linux.run --toolkit --silent

# Set environment
export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

---

## Recommended Action Plan

### Step 1: Try PyTorch Reinstall (RECOMMENDED)

```bash
conda activate 3danimals

# Backup current environment (optional)
conda list --export > ~/3danimals_backup.txt

# Remove PyTorch
conda remove pytorch torchvision torchaudio --yes

# Install PyTorch 1.13 + CUDA 11.8
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Test CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test CUBLAS operation
x = torch.randn(2, 197, 384).cuda()
linear = torch.nn.Linear(384, 1152).cuda()
y = linear(x)
print('✅ CUBLAS test passed!')
"

# If test passes, run training
cd /home/joon/dev/3DAnimals
python run_debug_notf32.py
```

### Step 2: If Reinstall Fails, Use CPU Mode

```bash
# Edit config
nano config/train_fauna_mouse_dannce_debug.yaml
# Change: device: cuda → device: cpu

# Run (will be slow)
python run_debug_notf32.py
```

### Step 3: Report Results

After trying Solution 1:
- ✅ Success → Continue with training
- ❌ Still fails → Try CPU mode or provide error message

---

## Testing Script

Save as `test_cuda_fix.py`:

```python
#!/usr/bin/env python
"""Test if CUDA fix worked"""
import torch
import torch.nn as nn

print("=" * 80)
print("CUDA FIX TEST")
print("=" * 80)

# Check versions
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Disable TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("TF32 disabled")

# Test 1: Basic tensor
try:
    x = torch.randn(2, 3, 256, 256).cuda()
    print("✅ Test 1: Basic CUDA tensor - PASS")
except Exception as e:
    print(f"❌ Test 1: Basic CUDA tensor - FAIL: {e}")
    exit(1)

# Test 2: Linear layer (where CUBLAS error occurs)
try:
    B, N, C = 2, 197, 384
    x = torch.randn(B, N, C).cuda()
    qkv = nn.Linear(C, C * 3).cuda()
    y = qkv(x)
    print("✅ Test 2: Linear layer (CUBLAS) - PASS")
except Exception as e:
    print(f"❌ Test 2: Linear layer (CUBLAS) - FAIL: {e}")
    exit(1)

# Test 3: Reshape and permute (full DINO operation)
try:
    B, N, C = 2, 197, 384
    num_heads = 6
    x = torch.randn(B, N, C).cuda()
    qkv_layer = nn.Linear(C, C * 3).cuda()
    qkv = qkv_layer(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    print("✅ Test 3: DINO attention operation - PASS")
except Exception as e:
    print(f"❌ Test 3: DINO attention operation - FAIL: {e}")
    exit(1)

print()
print("=" * 80)
print("✅ ALL TESTS PASSED! CUDA is working correctly.")
print("=" * 80)
print()
print("You can now run training:")
print("  cd /home/joon/dev/3DAnimals")
print("  python run_debug_notf32.py")
```

**Run test**:
```bash
conda activate 3danimals
python test_cuda_fix.py
```

---

## Summary

**Immediate Action**: Reinstall PyTorch with CUDA 11.8

```bash
conda activate 3danimals
conda remove pytorch torchvision torchaudio --yes
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

**Fallback**: Use CPU mode (slow but reliable)

**Test before training**: Run `test_cuda_fix.py`

---

**This should fix the CUBLAS error permanently!** 🚀
