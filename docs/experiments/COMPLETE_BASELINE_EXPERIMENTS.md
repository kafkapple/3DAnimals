# Pose Splatter 논문 Baseline 실험 완전 가이드

**날짜**: 2025-12-01
**목적**: 3D Fauna 및 MagicPony baseline 실험 재현

---

## 1. 실험 설계 분석

### 1.1 Pose Splatter 논문 원문 해석

논문에서 명시:
> "trained these two models on all six reference views"
> "trained using their prescribed data preprocessing pipelines and hyperparameters"

**해석**:
- "prescribed hyperparameters" = **각 모델의 기본 하이퍼파라미터 사용**
- "trained on all six reference views" = **Mouse 6-view 데이터로 학습**
- **Pretrained → Fine-tuning이 아닌, Mouse 데이터로 직접 학습한 것으로 해석됨**

### 1.2 두 가지 실험 시나리오

| 시나리오 | 설명 | 현실성 |
|----------|------|--------|
| **A: From Scratch** | Mouse 6-view 데이터만으로 처음부터 학습 | 논문 원문과 일치 |
| **B: Pretrained + Fine-tune** | Fauna pretrained → Mouse fine-tune | 더 나은 결과 기대 |

**권장**: 두 시나리오 모두 실험하여 비교

---

## 2. 데이터셋 현황

### 2.1 Pretrained 모델 (이미 보유)

| 모델 | 경로 | 크기 | 상태 |
|------|------|------|------|
| **3D Fauna** | `results/fauna/pretrained_fauna/pretrained_fauna.pth` | 160MB | ✅ 보유 |
| **MagicPony (Horse)** | 다운로드 필요 | - | ❌ 미보유 |
| **MagicPony (기타)** | cow, giraffe, zebra, bird 가능 | - | ❌ 미보유 |

### 2.2 Mouse 데이터셋

| 데이터셋 | 경로 | 이미지 수 | 용도 |
|----------|------|----------|------|
| **6-view Pose Splatter** | `data/fauna/mouse_6view_posesplatter/` | **6장** (train) | 논문 재현 실험 |
| **DANNCE 6-view** | `project_splatter/.../mouse_dannce_6view/` | 50장 | 확장 실험 |
| **원본 비디오** | `markerless_mouse_1_nerf/videos_undist/` | 18,000 frames × 6 views | 추가 데이터 생성 가능 |

### 2.3 6-view Pose Splatter 데이터 상세

```
data/fauna/mouse_6view_posesplatter/large_scale/mouse_6view/
├── train/  # Frame 5000, Camera 0-5 → 6장
├── val/    # Frame 10000, Camera 0-5 → 6장
└── test/   # Frame 10000, Camera 0-5 → 6장
```

| 속성 | 값 |
|------|---|
| 원본 해상도 | 1152 × 1024 |
| Fauna 해상도 | 256 × 256 |
| Train timestep | Frame 5000 |
| Test timestep | Frame 10000 |
| 카메라 수 | 6 |

---

## 3. 실험 A: From Scratch (논문 재현)

### 3.1 3D Fauna - From Scratch

**Config 파일**: `config/train_fauna_mouse_6view.yaml`

**핵심 파라미터** (Fauna 기본값 유지):

| 파라미터 | 값 | 비고 |
|----------|---|------|
| `grid_res` | 64 | GPU 메모리 제한 (기본 256) |
| `grid_res_coarse` | 32 | |
| `spatial_scale` | 5.0 | Mouse 크기 맞춤 (기본 7.0) |
| `batch_size` | 6 | 전체 6 views |
| `learning_rate` | 0.001 (base), 0.0001 (instance) | Fauna 기본값 |
| `num_iters` | 100,000 | |
| `articulation_iter_range` | [20000, inf] | Fauna 기본값 |
| `mask_discriminator` | [80000, 300000] | Fauna 기본값 |

**실행 명령어**:

```bash
# Debug (5K iterations, ~30분)
cd /home/joon/dev/3DAnimals
conda activate 3danimals
nohup python run.py --config-name train_fauna_mouse_6view_debug \
    > /tmp/fauna_6view_scratch_debug.log 2>&1 &

# Full (100K iterations, ~5-6시간)
nohup python run.py --config-name train_fauna_mouse_6view \
    > /tmp/fauna_6view_scratch_full.log 2>&1 &
```

### 3.2 MagicPony - From Scratch

**Config 파일**: `config/train_magicpony_mouse_6view.yaml` (생성 필요)

**핵심 파라미터**:

| 파라미터 | 값 | 비고 |
|----------|---|------|
| `grid_res` | 64 | |
| `spatial_scale` | 5.0 | Mouse 크기 |
| `batch_size` | 6 | |
| `num_iters` | 100,000 | |

**실행 명령어**:

```bash
# Debug
nohup python run.py --config-name train_magicpony_mouse_6view_debug \
    > /tmp/magicpony_6view_scratch_debug.log 2>&1 &

# Full
nohup python run.py --config-name train_magicpony_mouse_6view \
    > /tmp/magicpony_6view_scratch_full.log 2>&1 &
```

---

## 4. 실험 B: Pretrained + Fine-tune

### 4.1 3D Fauna - Fine-tune

**Pretrained 모델**: `results/fauna/pretrained_fauna/pretrained_fauna.pth`

**Config 파일**: `config/train_fauna_mouse_6view_finetune.yaml` (생성 필요)

**핵심 파라미터 변경**:

| 파라미터 | 값 | 비고 |
|----------|---|------|
| `resume` | `results/fauna/pretrained_fauna/pretrained_fauna.pth` | Pretrained 시작 |
| `learning_rate` | 0.0001 (낮춤) | Fine-tuning용 |
| `num_iters` | 50,000 | 더 짧게 |

**실행 명령어**:

```bash
# Fine-tune (50K iterations, ~3시간)
nohup python run.py --config-name train_fauna_mouse_6view_finetune \
    > /tmp/fauna_6view_finetune.log 2>&1 &
```

### 4.2 MagicPony - Fine-tune

**Pretrained 다운로드 필요**:

```bash
cd results/magicpony
bash download_pretrained_magicpony.sh
```

---

## 5. 전체 실험 매트릭스

| 실험 ID | 모델 | 방식 | 데이터 | Iterations | 예상 시간 |
|---------|------|------|--------|------------|----------|
| **A1** | Fauna | From Scratch | 6-view | 100K | 5-6h |
| **A2** | MagicPony | From Scratch | 6-view | 100K | 5-6h |
| **B1** | Fauna | Fine-tune | 6-view | 50K | 3h |
| **B2** | MagicPony | Fine-tune | 6-view | 50K | 3h |

---

## 6. Config 파일 생성 명령어

### 6.1 Fauna Fine-tune Config

```bash
# train_fauna_mouse_6view_finetune.yaml 생성
cat > config/train_fauna_mouse_6view_finetune.yaml << 'EOF'
# Fauna Fine-tuning from Pretrained for Mouse 6-view

hydra:
  run:
    dir: .
  output_subdir: ${checkpoint_dir}

defaults:
  - base_fauna
  - dataset: mouse_6view_posesplatter
  - model: fauna_mouse_6view

dataset:
  in_image_size: 256
  out_image_size: 256
  batch_size: 6
  train_data_dir: data/fauna/mouse_6view_posesplatter
  val_data_dir: data/fauna/mouse_6view_posesplatter
  test_data_dir: data/fauna/mouse_6view_posesplatter
  random_shuffle_samples_train: false
  load_dino_feature: false
  dino_feature_dim: 16
  random_xflip_train: false

run_train: true
run_test: false
seed: 42
gpu: 0
num_iters: 50000  # Shorter for fine-tuning

checkpoint_dir: results/fauna_mouse_6view_finetune
save_checkpoint_freq: 10000
keep_num_checkpoint: 5
resume: results/fauna/pretrained_fauna/pretrained_fauna.pth  # Pretrained!

use_logger: false
log_image_freq: 1000
log_loss_freq: 100
EOF
```

### 6.2 MagicPony 6-view Config

```bash
# train_magicpony_mouse_6view.yaml 생성
cat > config/train_magicpony_mouse_6view.yaml << 'EOF'
# MagicPony for Mouse 6-view (From Scratch)

hydra:
  run:
    dir: .
  output_subdir: ${checkpoint_dir}

defaults:
  - base_magicpony
  - dataset: mouse_6view_posesplatter
  - model: magicpony_mouse

dataset:
  in_image_size: 256
  out_image_size: 256
  batch_size: 6
  train_data_dir: data/fauna/mouse_6view_posesplatter
  val_data_dir: data/fauna/mouse_6view_posesplatter
  test_data_dir: data/fauna/mouse_6view_posesplatter

run_train: true
run_test: false
seed: 42
gpu: 0
num_iters: 100000

checkpoint_dir: results/magicpony_mouse_6view
save_checkpoint_freq: 10000
keep_num_checkpoint: 5
resume: false

use_logger: false
log_image_freq: 1000
log_loss_freq: 100
EOF
```

---

## 7. 실행 순서 권장

### Phase 1: Debug 검증 (각 30분)

```bash
# 1. Fauna From Scratch Debug
nohup python run.py --config-name train_fauna_mouse_6view_debug \
    > /tmp/exp_a1_debug.log 2>&1 &

# 로그 확인
tail -f /tmp/exp_a1_debug.log
```

### Phase 2: Full 실험 (순차 또는 병렬)

```bash
# 2. Fauna From Scratch Full (A1)
nohup python run.py --config-name train_fauna_mouse_6view \
    > /tmp/exp_a1_full.log 2>&1 &

# 3. Fauna Fine-tune (B1) - 다른 서버에서
nohup python run.py --config-name train_fauna_mouse_6view_finetune \
    > /tmp/exp_b1.log 2>&1 &

# 4. MagicPony From Scratch (A2) - 다른 서버에서
nohup python run.py --config-name train_magicpony_mouse_6view \
    > /tmp/exp_a2.log 2>&1 &
```

---

## 8. 예상 결과 비교

| 실험 | Input View | Novel View | Mesh Collapse |
|------|------------|------------|---------------|
| **A1 (Fauna Scratch)** | 중간 | 낮음 | 없음 예상 |
| **A2 (MagicPony Scratch)** | 중간 | 낮음 | 없음 예상 |
| **B1 (Fauna Finetune)** | **높음** | 중간 | 없음 |
| **B2 (MagicPony Finetune)** | 높음 | 중간 | 없음 |

**Pose Splatter 논문 결과**:
> "single-view models accurately reproduce their input view yet fail to maintain shape coherence once the mesh is rotated"

---

## 9. 결과 저장 위치

```
results/
├── fauna_mouse_6view_debug/         # A1 Debug
├── fauna_mouse_6view_posesplatter/  # A1 Full (From Scratch)
├── fauna_mouse_6view_finetune/      # B1 (Fine-tune)
├── magicpony_mouse_6view_debug/     # A2 Debug
├── magicpony_mouse_6view/           # A2 Full (From Scratch)
└── magicpony_mouse_6view_finetune/  # B2 (Fine-tune)
```

---

## 10. 체크리스트

### 데이터 준비
- [x] 6-view mouse 데이터 생성 (`scripts/prepare_6view_fauna_data.py`)
- [x] Fauna 포맷 변환 완료
- [x] 폴더 구조 검증 (large_scale, few_shot_* 폴더)

### Pretrained 모델
- [x] Fauna pretrained 보유 (`pretrained_fauna.pth`)
- [ ] MagicPony pretrained 다운로드 필요

### Config 파일
- [x] `train_fauna_mouse_6view.yaml` (From Scratch)
- [x] `train_fauna_mouse_6view_debug.yaml` (Debug)
- [ ] `train_fauna_mouse_6view_finetune.yaml` (Fine-tune)
- [ ] `train_magicpony_mouse_6view.yaml`
- [ ] `train_magicpony_mouse_6view_debug.yaml`

---

**작성자**: Claude Code
**최종 수정**: 2025-12-01
