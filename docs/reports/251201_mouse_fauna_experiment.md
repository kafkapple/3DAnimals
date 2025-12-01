# Mouse 3D Reconstruction with Fauna - Experiment Report

**Date**: 2025-12-01
**Project**: 3DAnimals / Pose Splatter Baseline Reproduction

---

## 1. Objective

Pose Splatter 논문의 baseline 실험 재현: 3D Fauna 모델을 사용한 multi-view mouse 3D reconstruction

**Research Question**:
- Multi-view supervision (6 cameras)으로 single-image 3D reconstruction 모델(Fauna)을 학습하면 어떤 결과가 나오는가?
- Pretrained Fauna에서 finetune vs from scratch 성능 차이는?

---

## 2. Data

### Source
- **sam3d_gui session**: `mouse_batch_20251128_163151`
- **원본 구조**: 2 mice × 6 cameras × 6 sequences × 100 frames

### Dataset Configurations

| Config | Images | 용도 |
|--------|--------|------|
| `pose_splatter_debug` | 6 | 코드 검증 |
| `full` | ~7200 | 전체 학습 |

### Preprocessing
- **Crop & Resize**: 원본 1152×1024 → subject 중심 256×256
- **이유**: Subject가 원본의 ~4.7%만 차지 → mesh collapse 방지

---

## 3. Model & Training

### Architecture
- **Base**: 3D Fauna (CVPR 2024)
- **Representation**: DMTet (hybrid SDF-mesh)
- **Features**: DINO ViT-S/8 features

### Training Configurations

| Config | Start | Iterations | Time | 용도 |
|--------|-------|------------|------|------|
| `debug` | Scratch | 5K | ~15min | 검증 |
| `6view_finetune` | Pretrained | 50K | ~5h | 6-view 실험 |
| `large_finetune` | Pretrained | 100K | ~10h | **Main** |
| `large` | Scratch | 200K | ~20h | Ablation |

### Key Hyperparameters
```yaml
grid_res: 64              # SDF resolution
spatial_scale: 5.0        # Mouse size adaptation
sdf_gradient_reg: 0.1     # Regularization
batch_size: 6             # 6 views per batch
```

---

## 4. Expected Results

### From Pose Splatter Paper
> "3D Fauna... fail to maintain shape coherence once the mesh is rotated to a novel viewpoint"

### Evaluation Metrics
- **Input view**: Mask IoU, RGB PSNR
- **Novel views**: Visual quality assessment
- **3D Mesh**: Topology, surface smoothness

---

## 5. Experiment Commands

```bash
# Step 1: Dataset preprocessing
python scripts/setup_multiview_fauna_dataset.py \
    --session_dir /path/to/session \
    --output_dir data/fauna/mouse_large \
    --mode full

# Step 2: Training (recommended)
python run.py --config-name train_fauna_mouse_large_finetune

# Step 3: Visualization
python visualization/visualize_results_fauna.py \
    --config-name test_fauna_mouse_large
```

---

## 6. Output Structure

```
results/fauna_mouse_large_finetune/
├── checkpoints (20K, 40K, 60K, 80K, 100K)
├── training_results/     # Every 10K iter
│   ├── *_mesh.obj        # 3D mesh
│   ├── *_image_pred.png  # Rendered image
│   └── *_mask_pred.png   # Predicted mask
├── test_results/         # Final evaluation
└── wandb/                # Training logs
```

---

## 7. Notes

### Bug Fixed (2025-12-01)
- **Issue**: Symlink을 통해 원본 이미지가 crop된 이미지로 덮어써짐
- **Cause**: `Image.save(symlink_path)` → follows symlink
- **Fix**: Save 전 symlink 삭제 체크 추가

### Key Learnings
1. Subject가 작을 때 반드시 crop preprocessing 필요
2. Fauna는 pretrained에서 finetune이 더 안정적
3. 6-view multi-view supervision이 mesh collapse 방지에 도움

---

## 8. Next Steps

- [ ] Full training 완료 후 결과 분석
- [ ] Novel view synthesis 품질 평가
- [ ] Pose Splatter와 비교 실험
- [ ] Articulation 품질 확인 (20K iter 이후)

---

**Author**: Claude Code Assistant
**Related**: Pose Splatter (arXiv:2024), 3D Fauna (CVPR 2024)
