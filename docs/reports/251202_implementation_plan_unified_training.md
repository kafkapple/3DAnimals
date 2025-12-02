# 통합 학습 시스템 구현 계획서

**날짜**: 2025-12-02
**목적**: MagicPony, Ponymation, Fauna 세 모델을 일관된 방식으로 학습할 수 있는 통합 시스템 구현

---

## 1. 현재 상태 분석

### 1.1 기존 Fauna 학습 환경 (Working)

```
✓ 설정: config/train_fauna_mouse_large.yaml
✓ 데이터: data/fauna_mouse/large_scale/mouse_dannce_6view/
✓ 스크립트: scripts/train_mouse.sh
✓ 결과: results/fauna_mouse_large/
```

### 1.2 MagicPony/Ponymation 상태

```
✗ 설정: 마우스용 config 없음 (다른 동물만 있음)
✗ 데이터: data/magicpony/, data/ponymation/ 비어있음
✗ 스크립트: 통합 스크립트 없음
```

---

## 2. 구현 목표

### 2.1 최종 결과물

1. **데이터 변환 도구**
   - Fauna → MagicPony 형식 변환
   - Fauna → Ponymation 형식 변환

2. **설정 파일**
   - MagicPony 마우스 학습용
   - Ponymation 마우스 학습용 (Stage 1, 2)

3. **통합 학습 스크립트**
   - 모든 모델 지원
   - 기존 train_mouse.sh 스타일 유지

---

## 3. 상세 구현 계획

### Phase 1: 데이터 변환 스크립트

#### 3.1.1 convert_fauna_to_magicpony.py

```python
"""
입력: data/fauna_mouse/large_scale/mouse_dannce_6view/train/
출력: data/magicpony/mouse/train/

변환 로직:
1. 각 시퀀스 폴더 순회
2. 각 프레임 파일을 개별 폴더로 분리
3. 파일명 변경: {id}_rgb.png → rgb.png
4. box.txt → metadata.json 변환
"""

# 예상 출력 구조
data/magicpony/mouse/
├── train/
│   ├── 000000_0000027/    # {seq}_{frame}
│   │   ├── rgb.png
│   │   ├── mask.png
│   │   └── metadata.json
│   └── ...
└── val/
    └── (5개 이미지)
```

#### 3.1.2 convert_fauna_to_ponymation.py

```python
"""
입력: data/fauna_mouse/large_scale/mouse_dannce_6view/train/
출력: data/ponymation/mouse/train/

변환 로직:
1. 각 시퀀스 폴더 순회
2. 프레임들을 시간순 정렬
3. 10개씩 그룹화하여 시퀀스 생성
4. frame_0/, frame_1/, ... 폴더 구조 생성
"""

# 예상 출력 구조
data/ponymation/mouse/
├── train/
│   ├── seq_000000/
│   │   ├── frame_0/
│   │   │   ├── rgb.png
│   │   │   ├── mask.png
│   │   │   └── metadata.json
│   │   ├── frame_1/
│   │   └── ... (10개)
│   └── ...
└── test/
```

### Phase 2: 설정 파일 생성

#### 3.2.1 MagicPony 마우스 설정

**파일**: `config/model/magicpony_mouse_local.yaml`

```yaml
# Fauna mouse 설정 기반으로 MagicPony에 맞게 조정
name: MagicPony

cfg_predictor_base:
  cfg_shape:
    grid_res: 128           # 마우스용 축소
    grid_res_coarse: 64
    spatial_scale: 5.0      # 마우스 크기에 맞춤

cfg_predictor_instance:
  cfg_articulation:
    num_body_bones: 6       # 마우스용 축소
    num_legs: 4
    num_leg_bones: 3

cfg_loss:
  sdf_gradient_reg_loss_weight: 0.1  # 안정성 강화
```

**파일**: `config/train_magicpony_mouse_local.yaml`

```yaml
defaults:
  - base
  - dataset: image
  - model: magicpony_mouse_local

dataset:
  train_data_dir: data/magicpony/mouse/train
  val_data_dir: data/magicpony/mouse/val
  load_dino_feature: false  # DINO 특징 미생성 시

num_iters: 100000
checkpoint_dir: results/magicpony/mouse_local
```

#### 3.2.2 Ponymation 마우스 설정

**파일**: `config/train_ponymation_mouse_stage1.yaml`

```yaml
defaults:
  - base
  - dataset: sequence
  - model: ponymation_mouse

dataset:
  train_data_dir: data/ponymation/mouse/train
  num_frames: 10
  batch_size: 1

num_iters: 10000
enable_motion_vae: false
checkpoint_path: results/magicpony/mouse_local/checkpoint.pth
checkpoint_dir: results/ponymation/mouse_stage1
```

**파일**: `config/train_ponymation_mouse_stage2.yaml`

```yaml
defaults:
  - base
  - dataset: sequence
  - model: ponymation_mouse

dataset:
  batch_size: 10
  num_frames: 10

num_iters: 500000
enable_motion_vae: true
enable_render: false
checkpoint_path: results/ponymation/mouse_stage1/ckpt-10000.pth
checkpoint_dir: results/ponymation/mouse_stage2
```

### Phase 3: 통합 학습 스크립트

**파일**: `scripts/train_unified.sh`

```bash
#!/bin/bash
# Unified Training Script for MagicPony, Ponymation, Fauna
# Usage: ./train_unified.sh <model> <mode>
#   model: fauna | magicpony | ponymation
#   mode: debug | full | background

set -e

MODEL="${1:-fauna}"
MODE="${2:-debug}"

case "$MODEL" in
    fauna)
        case "$MODE" in
            debug) CONFIG="train_fauna_mouse_6view_debug" ;;
            full)  CONFIG="train_fauna_mouse_large" ;;
        esac
        ;;
    magicpony)
        case "$MODE" in
            debug) CONFIG="train_magicpony_mouse_local_debug" ;;
            full)  CONFIG="train_magicpony_mouse_local" ;;
        esac
        ;;
    ponymation)
        # Ponymation은 2단계 학습
        echo "Ponymation requires 2-stage training"
        echo "Stage 1: ./train_unified.sh ponymation-stage1 $MODE"
        echo "Stage 2: ./train_unified.sh ponymation-stage2 $MODE"
        exit 0
        ;;
    ponymation-stage1)
        CONFIG="train_ponymation_mouse_stage1"
        ;;
    ponymation-stage2)
        CONFIG="train_ponymation_mouse_stage2"
        ;;
esac

# 실행
conda run -n 3danimals python run.py --config-name "$CONFIG"
```

---

## 4. 테스트 계획

### 4.1 데이터 변환 테스트

```bash
# 1. 변환 실행
python scripts/convert_fauna_to_magicpony.py
python scripts/convert_fauna_to_ponymation.py

# 2. 검증
ls data/magicpony/mouse/train/ | wc -l  # 이미지 수 확인
ls data/ponymation/mouse/train/ | head  # 시퀀스 구조 확인
```

### 4.2 Debug 학습 테스트

```bash
# 각 모델 5K iterations 테스트 (15-20분씩)
./scripts/train_unified.sh fauna debug
./scripts/train_unified.sh magicpony debug
./scripts/train_unified.sh ponymation-stage1 debug
```

### 4.3 검증 체크리스트

- [ ] 데이터 로딩 성공
- [ ] Forward pass 성공
- [ ] Backward pass 성공
- [ ] Checkpoint 저장 성공
- [ ] GPU 메모리 12GB 이내

---

## 5. 파일 생성 목록

### 5.1 신규 생성 파일

```
scripts/
├── convert_fauna_to_magicpony.py    # 데이터 변환
├── convert_fauna_to_ponymation.py   # 데이터 변환
└── train_unified.sh                 # 통합 학습

config/
├── model/
│   ├── magicpony_mouse_local.yaml   # MagicPony 모델
│   └── ponymation_mouse.yaml        # Ponymation 모델
├── dataset/
│   └── (기존 image.yaml, sequence.yaml 재사용)
├── train_magicpony_mouse_local.yaml
├── train_magicpony_mouse_local_debug.yaml
├── train_ponymation_mouse_stage1.yaml
├── train_ponymation_mouse_stage1_debug.yaml
├── train_ponymation_mouse_stage2.yaml
└── train_ponymation_mouse_stage2_debug.yaml
```

### 5.2 수정 파일

```
scripts/train_mouse.sh  # train_unified.sh로 통합 안내 추가
```

---

## 6. 의존성 및 제약사항

### 6.1 학습 순서 의존성

```
1. Fauna: 독립적 (바로 학습 가능)
2. MagicPony: 독립적 (바로 학습 가능)
3. Ponymation Stage 1: MagicPony 체크포인트 필요
4. Ponymation Stage 2: Stage 1 체크포인트 필요
```

### 6.2 하드웨어 제약

| 모델 | GPU 메모리 | 예상 시간 (Debug) |
|------|-----------|------------------|
| Fauna | ~4GB | 15-20분 |
| MagicPony | ~6GB | 15-20분 |
| Ponymation Stage1 | ~8GB | 10분 |
| Ponymation Stage2 | ~10GB | 30분 |

---

## 7. 실행 순서

```bash
# Step 1: 데이터 변환 (5분)
cd /home/joon/dev/3DAnimals
python scripts/convert_fauna_to_magicpony.py
python scripts/convert_fauna_to_ponymation.py

# Step 2: 데이터 확인
ls -la data/magicpony/mouse/
ls -la data/ponymation/mouse/

# Step 3: Debug 테스트
./scripts/train_unified.sh fauna debug
./scripts/train_unified.sh magicpony debug
./scripts/train_unified.sh ponymation-stage1 debug

# Step 4: 결과 확인
ls results/fauna_mouse_large/
ls results/magicpony/mouse_local/
ls results/ponymation/mouse_stage1/
```

---

## 8. 예상 산출물

### 8.1 학습 결과

```
results/
├── fauna_mouse_large/
│   ├── checkpoints/
│   └── visualizations/
├── magicpony/mouse_local/
│   ├── checkpoints/
│   └── visualizations/
└── ponymation/
    ├── mouse_stage1/
    └── mouse_stage2/
```

### 8.2 비교 분석

| 메트릭 | Fauna | MagicPony | Ponymation |
|--------|-------|-----------|-----------|
| Mask IoU | TBD | TBD | TBD |
| RGB PSNR | TBD | TBD | TBD |
| 학습 시간/iter | TBD | TBD | TBD |

---

## 9. 다음 단계

1. **즉시 실행**: 데이터 변환 스크립트 작성 및 실행
2. **단기**: 설정 파일 생성 및 Debug 테스트
3. **중기**: Full 학습 실행 및 결과 비교
4. **장기**: 하이퍼파라미터 튜닝 및 최적화
