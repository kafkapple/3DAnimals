#!/bin/bash

# Navigate to project
cd /home/joon/dev/3DAnimals

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3danimals

# Disable TF32 (causes CUBLAS errors on some GPUs)
export NVIDIA_TF32_OVERRIDE=0

# CUDA debugging flags
export CUDA_LAUNCH_BLOCKING=1

# Run training with Python flags to disable TF32
python -c "
import torch
import os

# Disable TF32 for matmul and cudnn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('TF32 disabled')
print(f'MatMul TF32: {torch.backends.cuda.matmul.allow_tf32}')
print(f'cuDNN TF32: {torch.backends.cudnn.allow_tf32}')
print()

# Now run the actual training
import sys
sys.argv = ['run.py', '--config-name', 'train_fauna_mouse_dannce_debug']

exec(open('run.py').read())
"
