# Fauna Mouse Fine-tuning - Quick Start Guide

## Current Status
- ✅ Data conversion completed (50 frames)
- ✅ Config fully debugged and ready
- ⏸️ Training waiting for GPU availability

## GPU Check
```bash
nvidia-smi
# Need: ~5GB free
# Current: 8359MB/12288MB (68% used)
```

## Start Training (When GPU Available)
```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# Start training
nohup python run.py --config-name train_fauna_mouse_finetune > /tmp/fauna_training.log 2>&1 &

# Monitor
tail -f /tmp/fauna_training.log
```

## Training Config
- **Data**: 50 train + 50 val + 50 test frames
- **Iterations**: 50,000 (vs 1M scratch)
- **Batch size**: 4
- **Learning rate**: 0.0001 (10x lower)
- **Spatial scale**: 5 (mice smaller than horses)
- **Articulation start**: 5K iters (vs 20K original)

## Key Files
- Config: `config/train_fauna_mouse_finetune.yaml`
- Data: `data/fauna_mouse/large_scale/mouse_dannce_6view/`
- Pretrained: `results/fauna/pretrained_fauna/pretrained_fauna.pth`
- Log: `/tmp/fauna_training.log`
- Checkpoints: `results/fauna_mouse_finetune/`

## Validation Checkpoints
- **Iter 500**: First validation
- **Iter 5000**: Articulation enabled
- **Iter 25000**: Mid-training checkpoint

## Fixed Issues
1. ✅ `pretrained_sdf` dataclass field added
2. ✅ Absolute paths configured
3. ✅ Empty directories created
4. ✅ Metadata fields completed
5. ✅ `load_dino_feature: false` set

## Next Steps
1. Free GPU memory (kill PID 2499441, 3332901)
2. Start baseline training (ellipsoid init)
3. Prepare MAMMAL prior integration

## Documentation
- Full progress: `~/Documents/Obsidian/40_Areas/2_Research/_Notes/251111_fauna_mouse_integration_progress.md`
- MAMMAL plan: `~/Documents/Obsidian/40_Areas/2_Research/_Notes/251111_mammal_prior_integration_plan.md`
