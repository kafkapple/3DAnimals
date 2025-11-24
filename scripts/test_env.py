#!/usr/bin/env python
"""Test environment setup"""
import torch
import pytorch3d

print("=" * 80)
print("Environment Verification")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch3D version: {pytorch3d.__version__}")
print("=" * 80)
print("✅ All packages loaded successfully!")
