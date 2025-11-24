# 3DAnimals Project Structure Guide

**Date**: 2025-11-24
**Purpose**: Systematic project organization for data, results, and outputs

---

## Directory Structure Overview

```
3DAnimals/
├── data/                    # Datasets (mixed: files + symlinks)
├── results/                 # Pretrained models (download location)
├── outputs/                 # Training outputs (gitignored)
├── config/                  # Hydra configs
├── model/                   # Model code
├── docs/                    # Documentation
└── scripts/                 # Utility scripts
```

---

## 1. data/ Directory

### Purpose
Dataset storage with selective symlinking for large datasets.

### Structure
```
data/
├── fauna/
│   ├── download_fauna_dataset.sh          # Git tracked
│   ├── Fauna_dataset/                     # Symlink → /media/joon/kafka/...
│   └── Mouse_only_dataset/                # Symlink → /media/joon/kafka/...
├── fauna_mouse/                            # Symlink → /home/joon/dev/project_splatter/...
├── magicpony/
│   ├── download_magicpony_dataset.sh      # Git tracked
│   └── [dataset files]                    # Local copies
├── ponymation/
│   ├── download_ponymation_dataset.sh     # Git tracked
│   └── [dataset files]                    # Local copies
└── tets/
    ├── download_tets.sh                   # Git tracked
    ├── generate_tets.py                   # Git tracked
    └── *.npz                              # Local copies (gitignored)
```

### Git Tracking (.gitignore)
```gitignore
data/*                # Ignore all
!data/**/*.sh         # Track scripts
!data/**/*.py         # Track Python files
!data/**/*.md         # Track docs
*.npz                 # Ignore large mesh files
```

### Rationale
- **Scripts tracked**: Reproducible dataset preparation
- **Large datasets symlinked**: Save disk space, consistent across servers
- **Small datasets copied**: Fast access, no network dependency

---

## 2. results/ Directory

### Purpose
**Pretrained model download location** (NOT training outputs)

### Structure
```
results/
├── fauna/
│   ├── download_pretrained_fauna.sh       # Git tracked
│   └── pretrained_fauna.zip               # Downloaded (gitignored)
├── magicpony/
│   ├── download_pretrained_magicpony.sh   # Git tracked
│   └── [pretrained_*.zip]                 # Downloaded (gitignored)
└── ponymation/
    ├── download_pretrained_ponymation.sh  # Git tracked
    └── [pretrained_*.zip]                 # Downloaded (gitignored)
```

### Git Tracking (.gitignore)
```gitignore
results/*             # Ignore all
!results/**/*.sh      # Track download scripts
```

### Usage
```bash
# Download pretrained model
cd results/fauna
bash download_pretrained_fauna.sh

# Model will be extracted to:
# results/fauna/pretrained_fauna/checkpoint.pth
```

### Why "results/" for Pretrained Models?
- Historical convention from original 3DAnimals repo
- Consistent with inference scripts expecting `results/*/checkpoint.pth`
- Clear separation: `results/` = external models, `outputs/` = our experiments

---

## 3. outputs/ Directory

### Purpose
**All training/inference outputs** from experiments

### Structure
```
outputs/
├── checkpoints/              # Model checkpoints (*.pth)
│   ├── checkpoint10000.pth
│   ├── checkpoint30000.pth
│   └── checkpoint50000.pth
├── training_runs/            # Training experiment results
│   ├── fauna_mouse_debug/
│   ├── fauna_mouse_from_scratch/
│   └── mouse_only_debug/
├── inference_results/        # Inference outputs
│   ├── mouse_dannce_infer/
│   │   ├── test_results_checkpoint3000/
│   │   │   ├── 0000000_image_gt.png
│   │   │   ├── 0000000_image_pred.png
│   │   │   ├── 0000000_mesh.obj
│   │   │   └── 0000000_pose.txt
│   │   └── ...
│   └── mouse_dannce_infer_5k/
├── logs/                     # Training logs
│   ├── tensorboard_logs/
│   ├── metrics.json
│   └── wandb/
├── archives/                 # Old checkpoints, code backups
│   ├── archived_code.zip
│   └── backup_old_checkpoints/
└── YYYY-MM-DD/               # Hydra outputs (daily)
    └── HH-MM-SS/
        ├── .hydra/
        ├── main.log
        └── config.yaml
```

### Git Tracking (.gitignore)
```gitignore
outputs               # Entire directory ignored
```

### Why Separate outputs/?
- **Clean git**: Training artifacts don't clutter git history
- **Flexible cleanup**: Easy to delete old experiments
- **Consistent location**: All outputs in one place
- **Server sync**: Easy to rsync/backup specific subdirectories

---

## 4. Config-Based Output Paths

### Current Default Behavior
```yaml
# config/train_fauna_mouse_dannce.yaml
checkpoint_dir: results        # ❌ Old: saves to results/
output_dir: results/${exp_name}
```

### Recommended Update
```yaml
# config/train_fauna_mouse_dannce.yaml
checkpoint_dir: outputs/checkpoints              # ✅ New
output_dir: outputs/training_runs/${exp_name}
```

### Why Change?
- **Separation of concerns**: `results/` = pretrained, `outputs/` = experiments
- **Gitignore alignment**: `outputs/` already ignored
- **Consistency**: All training outputs in outputs/

---

## 5. Migration Strategy

### For Existing Experiments
Already migrated (2025-11-24):
```bash
results/checkpoint*.pth           → outputs/checkpoints/
results/fauna_mouse_*/            → outputs/training_runs/
results/mouse_dannce_infer*/      → outputs/inference_results/
results/tensorboard_logs/         → outputs/logs/
results/archived_code.zip         → outputs/archives/
```

### For New Experiments
Update configs to use `outputs/` paths:
```yaml
# For training
checkpoint_dir: outputs/checkpoints
save_checkpoint_freq: 5000

# For inference
test_result_dir: outputs/inference_results/${exp_name}_checkpoint${checkpoint_iter}
```

---

## 6. Server Consistency

### For Multi-Server Setup

**data/ symlinking**:
```bash
# On each server, create symlinks to local storage
ln -s /path/to/server/storage/Fauna_dataset data/fauna/Fauna_dataset
ln -s /path/to/server/storage/Mouse_only_dataset data/fauna/Mouse_only_dataset
```

**outputs/ backup**:
```bash
# Periodic backup to shared storage
rsync -avz outputs/ /shared/storage/3DAnimals/outputs/$(hostname)/
```

**results/ pretrained models**:
```bash
# Download once per server
cd results/fauna && bash download_pretrained_fauna.sh
```

---

## 7. Disk Space Management

### Current Usage
```
outputs/:    3.8GB (checkpoints, logs, inference results)
results/:    474MB (pretrained models)
data/:       ~100GB (via symlinks, actual storage elsewhere)
```

### Cleanup Strategy
```bash
# Remove old experiments (keep last 3)
cd outputs/training_runs
ls -t | tail -n +4 | xargs rm -rf

# Remove old checkpoints (keep 10K, 30K, 50K)
cd outputs/checkpoints
rm -f checkpoint[0-9]000.pth  # Keep *0000.pth only

# Compress old logs
cd outputs/logs
tar -czf tensorboard_$(date +%Y%m%d).tar.gz tensorboard_logs/
rm -rf tensorboard_logs/
```

---

## 8. Best Practices

### ✅ DO
- Use `outputs/` for all training/inference outputs
- Keep `results/` clean (only pretrained models)
- Symlink large datasets in `data/`
- Track scripts (`.sh`, `.py`) in git
- Use descriptive experiment names

### ❌ DON'T
- Don't save training checkpoints to `results/`
- Don't commit large files (*.pth, *.npz) to git
- Don't mix pretrained and fine-tuned models
- Don't hardcode absolute paths in configs

---

## 9. Quick Reference

### Download Pretrained Model
```bash
cd results/fauna
bash download_pretrained_fauna.sh
# Output: results/fauna/pretrained_fauna/checkpoint.pth
```

### Train New Model
```bash
python run.py --config-name train_fauna_mouse_dannce
# Outputs:
#   outputs/checkpoints/checkpoint50000.pth
#   outputs/training_runs/fauna_mouse_dannce/
#   outputs/logs/tensorboard_logs/
```

### Run Inference
```bash
python run.py --config-name infer_mouse_dannce \
  resume=outputs/checkpoints/checkpoint50000.pth
# Output: outputs/inference_results/mouse_dannce_infer/
```

### Backup Experiments
```bash
# Backup to external storage
rsync -avz --progress outputs/ /media/backup/3DAnimals/outputs/

# Backup specific experiment
rsync -avz outputs/training_runs/fauna_mouse_from_scratch/ \
  /media/backup/3DAnimals/experiments/
```

---

## 10. Verification

### Check Structure
```bash
# data/ (scripts tracked, datasets symlinked)
ls -la data/fauna/
# Expected:
#   download_fauna_dataset.sh (file)
#   Fauna_dataset (symlink)
#   Mouse_only_dataset (symlink)

# results/ (only pretrained download scripts)
ls -la results/
# Expected:
#   fauna/download_pretrained_fauna.sh
#   magicpony/download_pretrained_magicpony.sh
#   ponymation/download_pretrained_ponymation.sh

# outputs/ (all experiments)
ls -la outputs/
# Expected:
#   checkpoints/
#   training_runs/
#   inference_results/
#   logs/
#   archives/
```

### Git Status
```bash
git status
# Should show:
#   data/**/*.sh (tracked)
#   results/**/*.sh (tracked)
#   outputs/ (ignored)
#   *.npz (ignored)
#   *.pth (ignored if in outputs/)
```

---

**Last Updated**: 2025-11-24
**Maintainer**: Joon
**Status**: Active
