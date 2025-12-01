# Pose Splatter 논문 3D Fauna 실험 재현 가이드

**날짜**: 2025-12-01
**목적**: Pose Splatter 논문에서 수행한 3D Fauna baseline 실험 재현

---

## 1. 실험 배경

### 1.1 논문 원문 (Pose Splatter)

> "We trained these two models [MagicPony, 3D Fauna] on all six reference views and evaluated them on a random view from an unseen time-step."
>
> "trained using their prescribed data preprocessing pipelines and hyperparameters"

### 1.2 핵심 실험 설계

| 항목 | 설정 |
|------|------|
| **학습 데이터** | 1 timestep × 6 camera views = **6장** |
| **테스트 데이터** | 다른 timestep × 6 camera views = **6장** |
| **모델** | 3D Fauna (표준 하이퍼파라미터) |
| **기대 결과** | 입력 뷰 OK, novel view에서 shape coherence 부족 |

### 1.3 실험 의의

- **데이터 부족 시나리오**: 단 6장으로 학습
- **Multi-view supervision**: 같은 시점의 6개 다른 각도
- **Mesh collapse 여부**: 논문에서는 collapse 없이 input view는 성공

---

## 2. 데이터셋

### 2.1 원본 데이터

**경로**: `/home/joon/dev/project_splatter/data/markerless_mouse_1_nerf/`

| 항목 | 값 |
|------|---|
| 카메라 수 | 6 |
| 총 프레임 | 18,000 |
| 해상도 | 1152 × 1024 |
| FPS | 100 |
| 녹화 시간 | 180초 |

### 2.2 Fauna 포맷 데이터

**경로**: `data/fauna/mouse_6view_posesplatter/`

```
data/fauna/mouse_6view_posesplatter/
├── few_shot_animal3d/     # (빈 폴더, FaunaDataset 요구사항)
├── few_shot_web/          # (빈 폴더)
├── few_shot_web_back/     # (빈 폴더)
└── large_scale/
    └── mouse_6view/
        ├── train/         # 6장 (frame 5000, 6 views)
        ├── val/           # 6장 (frame 10000, 6 views)
        └── test/          # 6장 (frame 10000, 6 views)
```

### 2.3 데이터 상세

| Split | Frame Index | 카메라 | 이미지 수 |
|-------|-------------|--------|----------|
| train | 5000 | 0-5 | 6 |
| val | 10000 | 0-5 | 6 |
| test | 10000 | 0-5 | 6 |

**이미지 크기**: 256 × 256 (Fauna 표준)

---

## 3. 모델 설정

### 3.1 핵심 하이퍼파라미터 (Fauna 표준)

| 파라미터 | 값 | 비고 |
|----------|---|------|
| `grid_res` | 64 | GPU 메모리 제한으로 128에서 감소 |
| `spatial_scale` | 5.0 | Mouse 크기에 맞게 7.0에서 감소 |
| `learning_rate` | 0.001 (base), 0.0001 (instance) | Fauna 기본값 |
| `batch_size` | 6 | 전체 6 views를 한 배치로 |
| `num_body_bones` | 6 | Mouse용 (기본 8에서 감소) |
| `articulation_iter_range` | [20000, inf] | Fauna 기본값 |
| `mask_discriminator` | [80000, 300000] | Fauna 기본값 |

### 3.2 Config 파일 위치

```
config/
├── dataset/mouse_6view_posesplatter.yaml
├── model/fauna_mouse_6view.yaml
├── train_fauna_mouse_6view.yaml         # Full training (100K iters)
└── train_fauna_mouse_6view_debug.yaml   # Debug (5K iters)
```

---

## 4. 실행 명령어

### 4.1 데이터 준비 (이미 완료)

```bash
# 6-view 데이터 생성 (train: frame 5000, test: frame 10000)
cd /home/joon/dev/3DAnimals
conda activate 3danimals

python scripts/prepare_6view_fauna_data.py \
    --data_dir /home/joon/dev/project_splatter/data/markerless_mouse_1_nerf \
    --output_dir data/fauna/mouse_6view_posesplatter \
    --train_frame 5000 \
    --test_frame 10000
```

### 4.2 Debug 학습 (5K iterations, ~20-30분)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# Foreground 실행
python run.py --config-name train_fauna_mouse_6view_debug

# Background 실행 (권장)
nohup python run.py --config-name train_fauna_mouse_6view_debug \
    > /tmp/fauna_6view_debug.log 2>&1 &

# 로그 확인
tail -f /tmp/fauna_6view_debug.log
```

### 4.3 Full 학습 (100K iterations, ~5-6시간)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# Background 실행
nohup python run.py --config-name train_fauna_mouse_6view \
    > /tmp/fauna_6view_full.log 2>&1 &

# 로그 확인
tail -f /tmp/fauna_6view_full.log
```

### 4.4 Inference (학습 완료 후)

```bash
python run.py --config-name train_fauna_mouse_6view \
    run_train=false \
    run_test=true \
    checkpoint_name=checkpoint_100000.pth
```

---

## 5. 예상 결과

### 5.1 Pose Splatter 논문 결과

> "single-image pipelines never observe the six views simultaneously, making it difficult to resolve self-occlusions"
> → "fail to maintain shape coherence once the mesh is rotated"

### 5.2 기대 결과

| 평가 항목 | 예상 결과 |
|----------|----------|
| Input view reconstruction | **성공** (reasonable quality) |
| Novel view synthesis | **제한적** (shape coherence 부족) |
| Mesh collapse | **없음** (6-view supervision 효과) |
| Articulation | 20K iterations 이후 활성화 시 가능 |

### 5.3 결과 저장 위치

```
results/fauna_mouse_6view_posesplatter/  # Full training
results/fauna_mouse_6view_debug/          # Debug training
```

---

## 6. 우리 이전 실험과 비교

| 비교 항목 | 이전 실험 (Mouse-only) | 현재 실험 (6-view) |
|----------|----------------------|-------------------|
| 데이터 | 50-100 frames (단일 뷰) | 6 frames (6 동기화 뷰) |
| Multi-view | 없음 | **있음** (같은 시점 6각도) |
| Mesh collapse | **발생** | 없음 예상 |
| 결과 | 실패 | Input view 성공 예상 |

**핵심 차이**: Multi-view supervision이 shape prior 학습에 결정적

---

## 7. 트러블슈팅

### 7.1 CUDA OOM

```yaml
# config/model/fauna_mouse_6view.yaml
grid_res: 64  # 128 → 64로 감소
grid_res_coarse: 32
```

### 7.2 데이터 로딩 에러

FaunaDataset은 특정 폴더 구조를 요구합니다:
- `large_scale/` 필수
- `few_shot_animal3d/`, `few_shot_web/`, `few_shot_web_back/` 빈 폴더 필요

### 7.3 Validation 에러

`val_data_dir` 설정 필수 (train과 동일하게 설정 가능)

---

## 8. 참고 문헌

1. **Pose Splatter**: https://arxiv.org/html/2505.18342v1
2. **3D Fauna**: https://arxiv.org/html/2401.02400v2

---

**작성자**: Claude Code
**최종 수정**: 2025-12-01
