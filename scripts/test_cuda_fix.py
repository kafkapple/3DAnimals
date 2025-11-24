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
