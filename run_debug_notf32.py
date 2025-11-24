#!/usr/bin/env python
"""
Run training with TF32 disabled to avoid CUBLAS errors
"""
import torch
import os

# CRITICAL: Disable TF32 before importing anything else
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print("=" * 80)
print("TF32 DISABLED FOR CUBLAS COMPATIBILITY")
print("=" * 80)
print(f"MatMul TF32: {torch.backends.cuda.matmul.allow_tf32}")
print(f"cuDNN TF32: {torch.backends.cudnn.allow_tf32}")
print()

# Set CUDA environment
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Now import and run hydra
import hydra
from omegaconf import DictConfig
from model import Trainer, build_model

@hydra.main(config_path="config", config_name="train_fauna_mouse_dannce_debug", version_base=None)
def main(cfg: DictConfig):
    model = build_model(cfg.model)
    trainer = Trainer(cfg, model)

    if cfg.run_train:
        trainer.train()
    if cfg.run_test:
        trainer.test()

if __name__ == "__main__":
    main()
