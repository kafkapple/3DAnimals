# Pose Splatter 논문 Baseline 실험 정확한 재현 가이드

**날짜**: 2025-12-01
**목적**: Pose Splatter 논문의 3D Fauna / MagicPony baseline 실험 정확한 재현

---

## 1. 논문 원문 분석

### 1.1 핵심 인용문

> "We **trained** these two models on all six reference views and evaluated them on a random view from an unseen time-step."

> "Pose Splatter used the same six training images but was tested on all six views of the unseen time step."

> "[MagicPony and 3D Fauna] trained using their **prescribed data preprocessing pipelines and hyperparameters**"

### 1.2 정확한 해석

| 항목 | MagicPony / 3D Fauna | Pose Splatter |
|------|---------------------|---------------|
| **학습 여부** | ✅ **Mouse 데이터로 학습함** | ✅ 학습함 |
| **학습 데이터** | 6개 reference views (1 timestep × 6 cameras = 6장) | 동일 6장 |
| **학습 방식** | 단일 뷰 모델 (한 번에 1장만 입력) | 다중 뷰 모델 (6장 동시 입력) |
| **테스트 입력** | unseen timestep의 **랜덤 1개 뷰** | unseen timestep의 **6개 뷰 전체** |
| **하이퍼파라미터** | 원본 논문 기본값 사용 | - |

### 1.3 핵심 포인트

1. **"trained"** = Mouse 데이터로 **처음부터 학습** (Zero-shot inference가 아님!)
2. **"prescribed hyperparameters"** = 각 모델의 **원본 논문 기본 설정** 사용
3. **Pretrained 모델 fine-tune인지 from scratch인지는 불명확**
   - 그러나 "trained on all six reference views"는 **6장으로 학습**했다는 의미
4. 단일 뷰 모델의 한계: **"never observe the six views simultaneously"**

---

## 2. 실험 설계 (두 가지 해석)

### 해석 A: From Scratch (6장으로 처음부터 학습)

**근거**: "trained on all six reference views" = 6장만으로 학습
**문제점**: 6장으로 from scratch 학습 시 mesh collapse 가능성 높음

### 해석 B: Pretrained + Fine-tune (권장)

**근거**:
- MagicPony/3D Fauna는 원래 대규모 데이터로 학습된 모델
- "prescribed hyperparameters" = 원본 설정 사용 = pretrained 모델 활용 가능성
- 6장만으로 from scratch 시 합리적인 결과 얻기 어려움

**결론**: **Pretrained 모델을 Mouse 6장으로 Fine-tune**하는 것이 논문 의도에 더 부합

---

## 3. 데이터셋 상세

### 3.1 학습 데이터

| 항목 | 값 |
|------|---|
| **위치** | `data/fauna/mouse_6view_posesplatter/large_scale/mouse_6view/train/` |
| **이미지 수** | **6장** (1 timestep × 6 cameras) |
| **Timestep** | Frame 5000 |
| **카메라** | Camera 0, 1, 2, 3, 4, 5 |
| **해상도** | 256 × 256 (Fauna 전처리 후) |
| **원본 해상도** | 1152 × 1024 |

### 3.2 테스트 데이터

| 항목 | 값 |
|------|---|
| **위치** | `data/fauna/mouse_6view_posesplatter/large_scale/mouse_6view/test/` |
| **이미지 수** | **6장** (다른 timestep × 6 cameras) |
| **Timestep** | Frame 10000 (unseen) |

### 3.3 원본 데이터

| 항목 | 값 |
|------|---|
| **위치** | `/home/joon/dev/project_splatter/data/markerless_mouse_1_nerf/` |
| **카메라 수** | 6 |
| **총 프레임** | 18,000 |
| **FPS** | 100 |

---

## 4. 모델 설정

### 4.1 3D Fauna 설정

**Config**: `config/train_fauna_mouse_6view.yaml`

**핵심 파라미터 (Fauna 기본값 = "prescribed hyperparameters")**:

| 파라미터 | 값 | 비고 |
|----------|---|------|
| `grid_res` | 64 | GPU 메모리 제한 (원본 256) |
| `spatial_scale` | 5.0 | Mouse 크기 (원본 7.0) |
| `learning_rate` | 0.001 (base), 0.0001 (instance) | **Fauna 기본값** |
| `batch_size` | 6 | 전체 6장 |
| `articulation_iter_range` | [20000, inf] | **Fauna 기본값** |
| `mask_discriminator` | [80000, 300000] | **Fauna 기본값** |
| `num_body_bones` | 6 | Mouse용 조정 |

### 4.2 MagicPony 설정

**Config**: `config/train_magicpony_mouse_6view.yaml`

**핵심 파라미터 (MagicPony 기본값)**:

| 파라미터 | 값 | 비고 |
|----------|---|------|
| `grid_res` | 64 | GPU 메모리 제한 |
| `spatial_scale` | 5.0 | Mouse 크기 |
| `learning_rate` | 0.0001 | **MagicPony 기본값** |
| `batch_size` | 6 | 전체 6장 |
| `articulation_iter_range` | [10000, inf] | **MagicPony 기본값** |

---

## 5. 실험 시나리오

### 5.1 시나리오 A: From Scratch (논문 문자 그대로)

**설명**: 6장만으로 처음부터 학습
**예상 결과**: Mesh collapse 가능성, 또는 매우 제한적인 품질

```bash
# Fauna From Scratch
nohup python run.py --config-name train_fauna_mouse_6view \
    > /tmp/fauna_6view_scratch.log 2>&1 &

# MagicPony From Scratch
nohup python run.py --config-name train_magicpony_mouse_6view \
    > /tmp/magicpony_6view_scratch.log 2>&1 &
```

### 5.2 시나리오 B: Pretrained + Fine-tune (권장)

**설명**: Pretrained 모델을 6장으로 Fine-tune
**예상 결과**: 논문과 유사한 결과 (입력 뷰 OK, novel view 제한적)

```bash
# Fauna Fine-tune (Pretrained → Mouse)
nohup python run.py --config-name train_fauna_mouse_6view_finetune \
    > /tmp/fauna_6view_finetune.log 2>&1 &
```

### 5.3 Pretrained 모델 현황

| 모델 | 경로 | 상태 |
|------|------|------|
| **3D Fauna** | `results/fauna/pretrained_fauna/pretrained_fauna.pth` | ✅ 보유 (160MB) |
| **MagicPony** | 다운로드 필요 | ❌ |

**MagicPony 다운로드**:
```bash
cd results/magicpony
bash download_pretrained_magicpony.sh
```

---

## 6. 실행 명령어 (전체)

### 6.1 데이터 준비 (이미 완료)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# 6-view 데이터 생성
python scripts/prepare_6view_fauna_data.py \
    --data_dir /home/joon/dev/project_splatter/data/markerless_mouse_1_nerf \
    --output_dir data/fauna/mouse_6view_posesplatter \
    --train_frame 5000 \
    --test_frame 10000
```

### 6.2 Debug 학습 (검증용, 각 ~30분)

```bash
# Fauna Debug (5K iterations)
nohup python run.py --config-name train_fauna_mouse_6view_debug \
    > /tmp/fauna_6view_debug.log 2>&1 &

# MagicPony Debug (5K iterations)
nohup python run.py --config-name train_magicpony_mouse_6view_debug \
    > /tmp/magicpony_6view_debug.log 2>&1 &
```

### 6.3 Full 학습

```bash
# [A1] Fauna From Scratch (100K iterations, ~5-6h)
nohup python run.py --config-name train_fauna_mouse_6view \
    > /tmp/fauna_6view_scratch.log 2>&1 &

# [A2] MagicPony From Scratch (100K iterations, ~5-6h)
nohup python run.py --config-name train_magicpony_mouse_6view \
    > /tmp/magicpony_6view_scratch.log 2>&1 &

# [B1] Fauna Fine-tune (50K iterations, ~3h) - 권장
nohup python run.py --config-name train_fauna_mouse_6view_finetune \
    > /tmp/fauna_6view_finetune.log 2>&1 &
```

### 6.4 로그 확인

```bash
tail -f /tmp/fauna_6view_scratch.log
tail -f /tmp/magicpony_6view_scratch.log
tail -f /tmp/fauna_6view_finetune.log
```

---

## 7. 예상 결과 (논문 기준)

### 7.1 논문 결과 인용

> "the single-view networks accurately reproduce their input view yet **fail to maintain shape coherence once the mesh is rotated**"

> "Pose Splatter preserves plausible anatomy from every angle"

### 7.2 예상 결과 표

| 실험 | Input View | Novel View | Mesh Collapse |
|------|------------|------------|---------------|
| Fauna Scratch | 중간 | 낮음 | 가능 |
| Fauna Fine-tune | **높음** | 중간 | 없음 |
| MagicPony Scratch | 중간 | 낮음 | 가능 |

---

## 8. 결과 저장 위치

```
results/
├── fauna_mouse_6view_debug/          # Debug (5K)
├── fauna_mouse_6view_posesplatter/   # From Scratch (100K)
├── fauna_mouse_6view_finetune/       # Fine-tune (50K)
├── magicpony_mouse_6view_debug/      # Debug (5K)
└── magicpony_mouse_6view/            # From Scratch (100K)
```

---

## 9. Config 파일 목록

| Config | 용도 | Iterations | Pretrained |
|--------|------|------------|------------|
| `train_fauna_mouse_6view.yaml` | Fauna From Scratch | 100K | ❌ |
| `train_fauna_mouse_6view_debug.yaml` | Fauna Debug | 5K | ❌ |
| `train_fauna_mouse_6view_finetune.yaml` | Fauna Fine-tune | 50K | ✅ |
| `train_magicpony_mouse_6view.yaml` | MagicPony From Scratch | 100K | ❌ |
| `train_magicpony_mouse_6view_debug.yaml` | MagicPony Debug | 5K | ❌ |

---

## 10. 권장 실험 순서

1. **Debug 먼저**: `train_fauna_mouse_6view_debug` (30분)
2. **Fine-tune 실험**: `train_fauna_mouse_6view_finetune` (3시간) - **논문 재현에 가장 근접**
3. **From Scratch 비교**: `train_fauna_mouse_6view` (6시간) - 비교용

---

## 11. 핵심 요약

| 질문 | 답변 |
|------|------|
| **논문에서 학습했나?** | ✅ 예, Mouse 데이터 6장으로 학습 |
| **Pretrained 사용?** | 불명확, but Fine-tune이 합리적 해석 |
| **우리 설정으로 재현?** | `train_fauna_mouse_6view_finetune` 권장 |
| **예상 결과?** | 입력 뷰 OK, novel view 제한적 |

---

**작성자**: Claude Code
**최종 수정**: 2025-12-01
