# Fauna Mouse Dataset Setup Guide

## 개요

sam3d_gui 세션 데이터에서 Fauna 학습용 데이터셋을 생성하는 방법.

---

## 데이터 구조

### 원본 데이터 (sam3d_gui session)
```
session_dir/
├── session_metadata.json    # 비디오 메타데이터
├── video_000_0/             # video_idx=0, seq=0
│   ├── frame_0000/
│   │   ├── original.png     # RGB 이미지
│   │   └── mask.png         # Segmentation mask
│   ├── frame_0001/
│   └── ...
├── video_001_12000/         # video_idx=1, seq=12000
└── ...
```

### 메타데이터 구조
- `video_path`: `mouse_{1,2}/Camera{1-6}/{seq}.mp4`
- **mouse_id**: 1 또는 2
- **camera_id**: 1~6 (6개 뷰)
- **sequence**: 0, 3000, 6000, 9000, 12000, 15000

### Fauna 데이터셋 (출력)
```
output_dir/
├── large_scale/
│   └── mouse/
│       ├── train/
│       │   └── seq0/
│       │       ├── 0000001_rgb.png -> (symlink)
│       │       ├── 0000001_mask.png -> (symlink)
│       │       ├── 0000001_box.txt
│       │       └── 0000001_metadata.json
│       ├── val/
│       └── test/
├── few_shot_animal3d/       # placeholder
├── few_shot_web/            # placeholder
├── few_shot_web_back/       # placeholder
└── dataset_info.json
```

---

## 사용 방법

### 1. Pose Splatter 논문 재현 (Debug Mode)

**설정**: 1 timestep × 6 views = 6장 학습

```bash
cd /path/to/3DAnimals

# 데이터셋 생성
python scripts/setup_multiview_fauna_dataset.py \
    --session_dir /path/to/sam3d_gui/outputs/sessions/mouse_batch_YYYYMMDD_HHMMSS \
    --output_dir data/fauna/mouse_6view_posesplatter \
    --mode pose_splatter_debug \
    --mouse_id 1 \
    --train_seq 0 \
    --test_seq 3000 \
    --train_frame 50 \
    --test_frame 50

# 학습 실행
python run.py --config-name train_fauna_mouse_6view_debug
```

### 2. Full 데이터셋 (모든 시퀀스)

```bash
python scripts/setup_multiview_fauna_dataset.py \
    --session_dir /path/to/session \
    --output_dir data/fauna/mouse_full \
    --mode full \
    --train_ratio 0.8

python run.py --config-name train_fauna_mouse_large
```

---

## 스크립트 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--session_dir` | sam3d_gui 세션 경로 | (필수) |
| `--output_dir` | 출력 데이터셋 경로 | (필수) |
| `--mode` | `pose_splatter_debug` 또는 `full` | `pose_splatter_debug` |
| `--mouse_id` | 생쥐 ID (1 또는 2) | 1 |
| `--train_seq` | 학습 시퀀스 | 0 |
| `--test_seq` | 테스트 시퀀스 | 3000 |
| `--train_frame` | 학습 프레임 인덱스 | 0 |
| `--test_frame` | 테스트 프레임 인덱스 | 0 |
| `--copy` | 심볼릭 링크 대신 복사 | false |

---

## 서버별 경로 예시

### 로컬 (joon@local)
```bash
python scripts/setup_multiview_fauna_dataset.py \
    --session_dir /home/joon/dev/sam3d_gui/outputs/sessions/mouse_batch_20251128_163151 \
    --output_dir data/fauna/mouse_6view_posesplatter
```

### GPU 서버 (gpu05)
```bash
python scripts/setup_multiview_fauna_dataset.py \
    --session_dir /home/joon/sam3d_gui/outputs/sessions/mouse_batch_20251128_163151 \
    --output_dir data/fauna/mouse_6view_posesplatter
```

**중요**: `--session_dir`의 경로만 서버에 맞게 변경하면 됨!

---

## 학습 Config 목록

| Config | 용도 | Iterations |
|--------|------|------------|
| `train_fauna_mouse_6view_debug` | 빠른 검증 | 5K |
| `train_fauna_mouse_6view_finetune` | 논문 재현 (Pretrained) | 50K |
| `train_fauna_mouse_6view` | From Scratch | 100K |
| `train_fauna_mouse_large` | Full 데이터셋 | 200K |

---

## 문제 해결

### Mesh Collapse (Empty triangle mesh)
```
AssertionError: Got empty training triangle mesh
```

**원인**: SDF가 붕괴됨
**해결**:
1. `sdf_gradient_reg_loss_weight` 증가 (0.1 → 1.0)
2. `grid_res_coarse_iter_range` 설정으로 점진적 해상도 증가
3. Pretrained 모델로 finetune 시도

### FileNotFoundError: large_scale
```
FileNotFoundError: 'data/fauna/mouse_6view_posesplatter/large_scale'
```

**원인**: 전처리 스크립트 미실행
**해결**: `setup_multiview_fauna_dataset.py` 먼저 실행

---

**작성일**: 2025-12-01
