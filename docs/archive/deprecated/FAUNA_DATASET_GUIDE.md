# Fauna Dataset 구조 및 새로운 동물 추가 가이드

**작성일**: 2025-11-19
**프로젝트**: 3DAnimals (Fauna)
**목적**: Fauna 데이터셋 구조 이해 및 새로운 동물(예: 생쥐) 데이터 준비 가이드

---

## 목차

1. [기존 Fauna 데이터셋 구조](#1-기존-fauna-데이터셋-구조)
2. [필수 데이터 파일 스펙](#2-필수-데이터-파일-스펙)
3. [새로운 동물 추가를 위한 데이터 준비](#3-새로운-동물-추가를-위한-데이터-준비)
4. [Config 파일 작성 가이드](#4-config-파일-작성-가이드)
5. [데이터셋 검증 체크리스트](#5-데이터셋-검증-체크리스트)

---

## 1. 기존 Fauna 데이터셋 구조

### 1.1 전체 디렉토리 구조

```
data/fauna/Fauna_dataset/
├── large_scale/                    # 대규모 비디오 시퀀스 데이터
│   ├── bear_comb_dinov2_new/
│   │   ├── train/
│   │   │   ├── 000002_00002/      # 비디오 시퀀스 ID
│   │   │   │   ├── 0000080_rgb.png
│   │   │   │   ├── 0000080_mask.png
│   │   │   │   ├── 0000080_metadata.json
│   │   │   │   ├── 0000080_box.txt
│   │   │   │   ├── 0000080_feat16.png
│   │   │   │   ├── 0000080_clusters.png
│   │   │   │   └── ...
│   │   │   ├── 000002_00003/
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   ├── cow_comb_dinov2_new/
│   ├── elephant_comb_dinov2_new/
│   ├── giraffe_comb_dinov2_new/
│   ├── horse_comb_dinov2_new/
│   ├── sheep_comb_dinov2_new/
│   └── zebra_comb_dinov2_new/
│
└── few_shot_animal3d/              # Few-shot 학습 데이터 (작은 데이터셋)
    ├── horse/
    │   └── train/
    │       ├── horse_000005_rgb.png
    │       ├── horse_000005_mask.png
    │       ├── horse_000005_metadata.json
    │       ├── horse_000005_box.txt
    │       ├── horse_000005_feat16.png
    │       ├── horse_000005_clusters.png
    │       └── ...
    ├── tiger/
    ├── leopard/
    └── ...
```

### 1.2 데이터셋 타입 구분

| 타입 | 설명 | 디렉토리 | 데이터 구조 |
|------|------|----------|------------|
| **Large Scale** | 대량의 비디오 시퀀스 | `large_scale/{animal}_comb_dinov2_new/` | 시퀀스별 폴더 (`train/`, `val/`, `test/`) |
| **Few Shot** | 적은 수의 이미지 | `few_shot_animal3d/{animal}/` | 단일 `train/` 폴더 |

### 1.3 파일 네이밍 규칙

**Large Scale**:
```
{sequence_id}/{frame_id}_{suffix}.{ext}

예시:
000002_00002/0000080_rgb.png
000002_00002/0000080_mask.png
```

**Few Shot**:
```
{animal}_{frame_id}_{suffix}.{ext}

예시:
horse_000005_rgb.png
horse_000005_mask.png
```

---

## 2. 필수 데이터 파일 스펙

각 프레임마다 다음 파일들이 필요합니다:

### 2.1 필수 파일 (Mandatory)

#### 1) RGB 이미지 (`*_rgb.png`)

**설명**: 원본 RGB 이미지
**형식**: PNG
**크기**: 가변 (일반적으로 256x256 이상)
**채널**: RGB (3 channels)

**예시**: `0000080_rgb.png`

---

#### 2) 마스크 이미지 (`*_mask.png`)

**설명**: 동물 영역 binary mask
**형식**: PNG
**크기**: RGB 이미지와 동일
**값**:
- 255 (흰색): 동물 영역 (foreground)
- 0 (검은색): 배경 (background)

**예시**: `0000080_mask.png`

**생성 방법**:
- 자동: Grounded-SAM, SAM2, Segment Anything 등
- 수동: Photoshop, GIMP, LabelMe 등
- 반자동: CVAT, VGG Image Annotator 등

---

#### 3) 메타데이터 (`*_metadata.json`)

**설명**: 프레임 메타정보 (크롭 박스, 원본 해상도 등)
**형식**: JSON

**필수 필드**:
```json
{
    "video_frame_id": 80,              // 원본 비디오의 프레임 번호
    "crop_box_xyxy": [982, -19, 1802, 801],  // 크롭 영역 [x1, y1, x2, y2]
    "video_frame_width": 1920,         // 원본 비디오 너비
    "video_frame_height": 1080,        // 원본 비디오 높이
    "sharpness": 185.29,               // 선명도 (optional)
    "crop_height": 256,                // 크롭 후 높이
    "crop_width": 256,                 // 크롭 후 너비
    "label": 7                         // 카테고리 라벨 (optional)
}
```

**예시**: `0000080_metadata.json`

**자동 생성 스크립트** (Python):
```python
import json

def create_metadata(frame_id, crop_box, orig_width, orig_height, crop_size=256):
    x1, y1, x2, y2 = crop_box
    metadata = {
        "video_frame_id": frame_id,
        "crop_box_xyxy": [x1, y1, x2, y2],
        "video_frame_width": orig_width,
        "video_frame_height": orig_height,
        "crop_height": crop_size,
        "crop_width": crop_size,
    }
    return metadata

# 예시
metadata = create_metadata(
    frame_id=80,
    crop_box=[982, -19, 1802, 801],
    orig_width=1920,
    orig_height=1080
)

with open("0000080_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
```

---

#### 4) 바운딩 박스 (`*_box.txt`)

**설명**: 단일 라인 텍스트, 크롭 정보
**형식**: Plain text (space-separated)

**구조**:
```
{frame_id} {x} {y} {width} {height} {full_w} {full_h} {sharpness} {label}
```

**예시**:
```
0000080 982.00 -19.00 820.00 820.00 1920.00 1080.00 185.29 7
```

**필드 설명**:
- `frame_id`: 프레임 ID (문자열 또는 숫자)
- `x, y`: 크롭 좌측 상단 좌표
- `width, height`: 크롭 영역 크기
- `full_w, full_h`: 원본 이미지 크기
- `sharpness`: 이미지 선명도 (optional, 0으로 설정 가능)
- `label`: 카테고리 라벨 (optional)

**자동 생성 스크립트**:
```python
def create_box_txt(frame_id, x, y, width, height, full_w, full_h, sharpness=0, label=0):
    line = f"{frame_id} {x:.2f} {y:.2f} {width:.2f} {height:.2f} {full_w:.2f} {full_h:.2f} {sharpness:.2f} {label}\n"
    return line

# 예시
with open("0000080_box.txt", "w") as f:
    f.write(create_box_txt("0000080", 982, -19, 820, 820, 1920, 1080, 185.29, 7))
```

---

### 2.2 선택적 파일 (Optional but Recommended)

#### 5) DINO 특징 맵 (`*_feat16.png`)

**설명**: DINOv2 feature map (16 채널)
**형식**: PNG (3채널로 인코딩된 16차원 feature)
**크기**: RGB 이미지와 동일
**생성**: DINOv2 모델 사용

**생성 방법**:
```python
import torch
from torchvision import transforms
from PIL import Image

# DINOv2 모델 로드
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_model.eval()

# 이미지 로드 및 전처리
img = Image.open("0000080_rgb.png")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)

# Feature 추출
with torch.no_grad():
    features = dinov2_model.forward_features(img_tensor)
    # features를 16채널로 resize/project
    # ... (프로젝트 코드 참고)

# PNG로 저장 (3채널로 인코딩)
# ... (util.py의 encode 함수 참고)
```

**참고**: `model/dataset/util.py`의 `dino_loader()` 및 `read_feat_from_img()` 함수

---

#### 6) 클러스터 맵 (`*_clusters.png`)

**설명**: DINO feature clustering 결과
**형식**: PNG
**용도**: Semantic part segmentation

---

### 2.3 파일 요약표

| 파일 | 필수 여부 | 설명 | 자동 생성 가능 |
|------|-----------|------|----------------|
| `*_rgb.png` | ✅ 필수 | RGB 이미지 | ❌ (원본 필요) |
| `*_mask.png` | ✅ 필수 | Binary mask | ⚠️ (SAM 등 사용) |
| `*_metadata.json` | ✅ 필수 | 메타데이터 | ✅ (스크립트) |
| `*_box.txt` | ✅ 필수 | 바운딩 박스 | ✅ (스크립트) |
| `*_feat16.png` | ⚠️ 권장 | DINO features | ✅ (DINOv2) |
| `*_clusters.png` | ❌ 선택 | Clustering | ✅ (K-means) |

---

## 3. 새로운 동물 추가를 위한 데이터 준비

### 3.1 데이터 수집 요구사항

**최소 요구사항**:
- **Few-shot**: 10-50 이미지 (다양한 자세/각도)
- **Large-scale**: 100+ 이미지 또는 비디오 시퀀스

**권장 사항**:
- 다양한 포즈 (서있기, 앉기, 걷기 등)
- 다양한 시점 (정면, 측면, 후면, 3/4 각도 등)
- 깨끗한 배경 (또는 명확한 전경-배경 구분)
- 고해상도 (512x512 이상)

---

### 3.2 생쥐(Mouse) 데이터 준비 단계별 가이드

#### Step 1: 원본 이미지 수집

**옵션 A: 비디오에서 추출**
```bash
# FFmpeg로 비디오에서 프레임 추출
ffmpeg -i mouse_video.mp4 -vf fps=5 mouse_frames/frame_%06d.png
```

**옵션 B: 공개 데이터셋 활용**
- DANNCE 데이터셋
- OpenMonkeyStudio
- AnimalPose 데이터셋

**옵션 C: 직접 촬영**
- 멀티뷰 카메라 설정 (6+ 각도)
- 고해상도 (1920x1080 이상)
- 다양한 행동 포착

---

#### Step 2: 마스크 생성

**방법 1: Grounded-SAM 자동 생성**
```bash
# Grounded-SAM으로 마스크 자동 생성
python scripts/generate_masks_sam.py \
  --input_dir mouse_frames/ \
  --output_dir mouse_masks/ \
  --prompt "mouse"
```

**방법 2: SAM2 사용**
```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(...)
# ... (SAM2 문서 참고)
```

**방법 3: 수동 annotation**
- CVAT, LabelMe 등 사용
- 정확도가 중요한 경우 권장

---

#### Step 3: 메타데이터 생성

**자동화 스크립트 예시**:
```python
import os
import json
import cv2
from pathlib import Path

def create_dataset_metadata(image_dir, mask_dir, output_dir):
    """
    RGB 이미지와 마스크로부터 메타데이터 자동 생성
    """
    image_files = sorted(Path(image_dir).glob("*.png"))

    for idx, img_path in enumerate(image_files):
        # 이미지 로드
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 마스크 로드
        mask_path = Path(mask_dir) / img_path.name.replace("_rgb", "_mask")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # 바운딩 박스 계산
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, box_w, box_h = cv2.boundingRect(coords)
            x1, y1 = x, y
            x2, y2 = x + box_w, y + box_h

            # 정사각형 크롭 영역 생성 (마진 추가)
            margin = 20
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            size = max(box_w, box_h) + margin * 2

            crop_x1 = max(0, center_x - size // 2)
            crop_y1 = max(0, center_y - size // 2)
            crop_x2 = min(w, center_x + size // 2)
            crop_y2 = min(h, center_y + size // 2)
        else:
            # 마스크가 비어있는 경우 전체 이미지
            crop_x1, crop_y1 = 0, 0
            crop_x2, crop_y2 = w, h

        # Metadata JSON 생성
        frame_id = idx
        metadata = {
            "video_frame_id": frame_id,
            "crop_box_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
            "video_frame_width": w,
            "video_frame_height": h,
            "crop_height": 256,
            "crop_width": 256,
        }

        output_file = Path(output_dir) / f"{frame_id:07d}_metadata.json"
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=4)

        # Box.txt 생성
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        box_line = f"{frame_id:07d} {crop_x1:.2f} {crop_y1:.2f} {crop_w:.2f} {crop_h:.2f} {w:.2f} {h:.2f} 0.00 0\n"

        box_file = Path(output_dir) / f"{frame_id:07d}_box.txt"
        with open(box_file, "w") as f:
            f.write(box_line)

        print(f"Processed: {img_path.name} -> metadata + box")

# 실행
create_dataset_metadata(
    image_dir="mouse_frames/",
    mask_dir="mouse_masks/",
    output_dir="mouse_dataset/train/"
)
```

---

#### Step 4: DINO Feature 추출 (선택적)

```python
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

def extract_dino_features(image_dir, output_dir, feature_dim=16):
    """
    DINOv2로 feature 추출
    """
    # DINOv2 모델 로드
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    model.cuda()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = sorted(Path(image_dir).glob("*_rgb.png"))

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).cuda()

        with torch.no_grad():
            # Feature 추출
            features = model.forward_features(img_tensor)
            # ... (프로젝트별 feature projection 코드)

        # PNG로 저장 (인코딩)
        output_file = Path(output_dir) / img_path.name.replace("_rgb.png", "_feat16.png")
        # ... (저장 코드)

        print(f"Extracted DINO features: {img_path.name}")

# 실행
extract_dino_features("mouse_dataset/train/", "mouse_dataset/train/")
```

---

#### Step 5: 디렉토리 구조 정리

**Few-shot 데이터셋 구조**:
```
data/fauna/Fauna_dataset/few_shot_animal3d/mouse/
└── train/
    ├── mouse_0000001_rgb.png
    ├── mouse_0000001_mask.png
    ├── mouse_0000001_metadata.json
    ├── mouse_0000001_box.txt
    ├── mouse_0000001_feat16.png
    ├── mouse_0000002_rgb.png
    ├── mouse_0000002_mask.png
    └── ...
```

**Large-scale 데이터셋 구조**:
```
data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view/
├── train/
│   ├── 000001_00001/
│   │   ├── 0000001_rgb.png
│   │   ├── 0000001_mask.png
│   │   ├── 0000001_metadata.json
│   │   ├── 0000001_box.txt
│   │   └── 0000001_feat16.png
│   ├── 000001_00002/
│   └── ...
├── val/
└── test/
```

---

### 3.3 데이터 전처리 자동화 스크립트

**통합 전처리 파이프라인**:
```bash
#!/bin/bash
# prepare_mouse_dataset.sh

INPUT_VIDEO="mouse_video.mp4"
OUTPUT_DIR="data/fauna/Fauna_dataset/few_shot_animal3d/mouse/train"

# 1. 프레임 추출
echo "Step 1: Extracting frames..."
ffmpeg -i $INPUT_VIDEO -vf fps=5 temp_frames/frame_%06d.png

# 2. 마스크 생성
echo "Step 2: Generating masks..."
python scripts/generate_masks_sam.py \
  --input_dir temp_frames/ \
  --output_dir temp_masks/ \
  --prompt "mouse"

# 3. 메타데이터 생성
echo "Step 3: Creating metadata..."
python scripts/create_metadata.py \
  --image_dir temp_frames/ \
  --mask_dir temp_masks/ \
  --output_dir $OUTPUT_DIR \
  --prefix "mouse"

# 4. DINO features 추출
echo "Step 4: Extracting DINO features..."
python scripts/extract_dino_features.py \
  --input_dir $OUTPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --feature_dim 16

# 5. 정리
echo "Step 5: Cleaning up..."
rm -rf temp_frames temp_masks

echo "Dataset preparation complete!"
```

---

## 4. Config 파일 작성 가이드

### 4.1 Dataset Config

**위치**: `config/dataset/fauna_mouse.yaml`

```yaml
# fauna_mouse.yaml - Mouse-specific dataset config

# Dataset 기본 설정
data_type: fauna
in_image_size: 256
out_image_size: 256
batch_size: 6  # GPU 메모리에 따라 조정
num_workers: 4

# 데이터 경로 (hydra defaults에서 자동 설정)
train_data_dir: null  # Auto-filled by hydra
val_data_dir: null
test_data_dir: null

# 학습 옵션
random_shuffle_samples_train: false
random_xflip_train: false  # 좌우 반전 augmentation

# 추가 feature 로딩
load_flow: false
background_mode: none  # none (black), white, checkerboard, background, input
load_dino_feature: false  # true로 설정 시 feat16.png 로드
load_dino_cluster: false
dino_feature_dim: 16
```

---

### 4.2 Model Config

**위치**: `config/model/fauna_mouse.yaml`

```yaml
# fauna_mouse.yaml - Mouse-specific model configuration

defaults:
  - fauna  # 기본 fauna config 상속

name: FaunaMouse

# Base predictor - Mouse-specific settings
cfg_predictor_base:
  cfg_shape:
    grid_res: 128  # Mouse는 작은 동물이므로 128로 축소
    grid_res_coarse_iter_range: [0, 300000]
    grid_res_coarse: 64
    spatial_scale: 5.0  # Mouse scale (기본 7.0보다 작음)
    num_layers: 5
    hidden_size: 64  # 메모리 절약
    embedder_freq: 8
    embed_concat_pts: true
    init_sdf: ellipsoid  # 또는 null (pretrained SDF 사용 시)
    pretrained_sdf: null  # 옵션: checkpoints/mouse_sdf_pretrained.pth
    jitter_grid: 0.05
    symmetrize: true

  cfg_dino:
    feature_dim: 16
    num_layers: 5
    hidden_size: 64
    activation: sigmoid
    embedder_freq: 8
    embed_concat_pts: true
    symmetrize: false
    minmax: [0., 1.]

  cfg_bank:
    memory_bank_size: 30  # 작은 데이터셋이므로 30

# Instance predictor
cfg_predictor_instance:
  spatial_scale: 5.0  # Base predictor와 동일

  cfg_texture:
    texture_iter_range: [0, inf]
    cout: 9
    num_layers: 6
    hidden_size: 128
    activation: sigmoid
    kd_minmax: [[0., 1.], [0., 1.], [0., 1.]]
    embed_concat_pts: true
    embedder_freq: 10
    symmetrize: true

  cfg_pose:
    architecture: encoder_dino_patch_key
    cam_pos_z_offset: ${...cfg_render.cam_pos_z_offset}
    fov: ${...cfg_render.fov}
    max_trans_xy_range_ratio: 0.2
    max_trans_z_range_ratio: 0.5
    rot_rep: quadlookat
    rot_temp_scalar: 1
    naive_probs_iter: 2000
    best_pose_start_iter: 6000
    lookat_zeroy: true

  # Deformation
  enable_deform: true
  cfg_deform:
    deform_iter_range: [400000, inf]  # Mouse는 작은 deformation
    num_layers: 4
    hidden_size: 128
    embed_concat_pts: true
    embedder_freq: 10
    symmetrize: true

  # Articulation - Mouse skeleton
  enable_articulation: true
  cfg_articulation:
    articulation_iter_range: [20000, inf]
    architecture: attention
    num_layers: 4
    hidden_size: 128
    embedder_freq: 8
    bone_feature_mode: sample+global
    num_body_bones: 6  # Mouse body bones
    body_bones_mode: z_minmax_y+
    num_legs: 4  # Mouse has 4 legs
    num_leg_bones: 3  # 각 다리당 3개 뼈
    attach_legs_to_body_iter_range: [60000, inf]
    legs_to_body_joint_indices: null  # Auto
    static_root_bones: false
    skinning_temperature: 0.05
    max_arti_angle: 60
    constrain_legs: false
    output_multiplier: 0.1
    enable_refine: false

  # Lighting
  enable_lighting: true
  cfg_light:
    num_layers: 4
    hidden_size: 128
    amb_diff_minmax: [[0.0, 1.0], [0.5, 1.0]]

  cfg_additional:
    iter_leg_rotation_start: 200000

# Rendering
cfg_render:
  spatial_scale: 5.0  # Mouse scale
  background_mode: none
  render_flow: false
  cam_pos_z_offset: 10
  fov: 25
  renderer_spp: 1
```

---

### 4.3 Training Config

**위치**: `config/train_fauna_mouse.yaml`

```yaml
# train_fauna_mouse.yaml - Mouse training configuration

defaults:
  - model: fauna_mouse  # Mouse-specific model config
  - _self_

# 실험 이름
exp_name: fauna_mouse_from_scratch
run_name: ${exp_name}_${now:%Y%m%d_%H%M%S}

# 학습 설정
num_iters: 200000
save_checkpoint_freq: 10000
log_image_freq: 1000
eval_freq: 5000

# 데이터 경로 (실제 경로로 수정)
dataset:
  train_data_dir: data/fauna/Fauna_dataset/few_shot_animal3d/mouse
  val_data_dir: null
  test_data_dir: null

# GPU 설정
device: cuda
gpu_ids: [0]

# Logging
wandb:
  project: fauna_mouse
  entity: your_entity
  mode: online  # online, offline, disabled

# Checkpoint
resume: null  # 또는 checkpoint 경로
```

---

## 5. 데이터셋 검증 체크리스트

### 5.1 파일 구조 검증

```bash
#!/bin/bash
# validate_dataset.sh

DATASET_DIR="data/fauna/Fauna_dataset/few_shot_animal3d/mouse/train"

echo "=== Dataset Validation ==="

# 1. RGB 이미지 개수
rgb_count=$(find $DATASET_DIR -name "*_rgb.png" | wc -l)
echo "RGB images: $rgb_count"

# 2. 마스크 개수
mask_count=$(find $DATASET_DIR -name "*_mask.png" | wc -l)
echo "Mask images: $mask_count"

# 3. Metadata 개수
meta_count=$(find $DATASET_DIR -name "*_metadata.json" | wc -l)
echo "Metadata files: $meta_count"

# 4. Box 개수
box_count=$(find $DATASET_DIR -name "*_box.txt" | wc -l)
echo "Box files: $box_count"

# 5. 일치 여부 확인
if [ $rgb_count -eq $mask_count ] && [ $rgb_count -eq $meta_count ] && [ $rgb_count -eq $box_count ]; then
    echo "✅ All file counts match: $rgb_count files"
else
    echo "❌ File count mismatch!"
    echo "   RGB: $rgb_count, Mask: $mask_count, Meta: $meta_count, Box: $box_count"
fi

# 6. 파일명 일치 검증
echo ""
echo "=== Checking file name consistency ==="
for rgb in $(find $DATASET_DIR -name "*_rgb.png"); do
    base=$(basename $rgb _rgb.png)
    mask="${DATASET_DIR}/${base}_mask.png"
    meta="${DATASET_DIR}/${base}_metadata.json"
    box="${DATASET_DIR}/${base}_box.txt"

    if [ ! -f $mask ]; then
        echo "❌ Missing mask: $mask"
    fi
    if [ ! -f $meta ]; then
        echo "❌ Missing metadata: $meta"
    fi
    if [ ! -f $box ]; then
        echo "❌ Missing box: $box"
    fi
done

echo "✅ Validation complete!"
```

---

### 5.2 데이터 로딩 테스트

```python
import torch
from model.dataset.FaunaDataset import FaunaDataset

# Dataset 로드 테스트
dataset = FaunaDataset(
    root="data/fauna/Fauna_dataset",
    in_image_size=256,
    out_image_size=256,
    shuffle=False,
    load_background=False,
    random_xflip=False,
    load_dino_feature=False,
    load_dino_cluster=False,
    dino_feature_dim=16,
    split='train',
    batch_size=6
)

print(f"Dataset length: {len(dataset)}")
print(f"Categories: {dataset.all_category_names}")

# 첫 번째 샘플 로드
sample = dataset[0]
images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, keypoint, seq_idx, frame_idx = sample

print(f"Image shape: {images.shape}")
print(f"Mask shape: {masks.shape}")
print(f"Bbox: {bboxs}")

print("✅ Dataset loading test passed!")
```

---

### 5.3 학습 시작 전 체크리스트

- [ ] 필수 파일 모두 존재 (`*_rgb.png`, `*_mask.png`, `*_metadata.json`, `*_box.txt`)
- [ ] 파일 개수 일치 (RGB = Mask = Metadata = Box)
- [ ] 파일명 규칙 준수 (`{prefix}_{id}_{suffix}.{ext}`)
- [ ] 마스크 품질 확인 (binary, 255/0 값)
- [ ] 메타데이터 JSON 형식 올바름
- [ ] Dataset config 경로 설정 완료
- [ ] Model config 파라미터 조정 완료
- [ ] Training config 경로 및 설정 확인
- [ ] GPU 메모리 충분 (최소 12GB 권장)
- [ ] Conda 환경 활성화 (`conda activate 3danimals`)
- [ ] CUDA 사용 가능 (`torch.cuda.is_available() == True`)

---

## 부록 A: 자동화 스크립트 모음

### A.1 통합 데이터 준비 스크립트

**위치**: `scripts/prepare_fauna_dataset.py`

```python
#!/usr/bin/env python3
"""
Fauna Dataset Preparation Script

Usage:
    python scripts/prepare_fauna_dataset.py \
        --input_dir raw_images/ \
        --output_dir data/fauna/Fauna_dataset/few_shot_animal3d/mouse/train \
        --animal_name mouse \
        --generate_masks \
        --extract_dino
"""

import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--animal_name", type=str, required=True)
    parser.add_argument("--generate_masks", action="store_true")
    parser.add_argument("--extract_dino", action="store_true")
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()

def create_metadata_and_box(img_path, mask_path, output_dir, frame_id, image_size=256):
    """Create metadata.json and box.txt for a single image"""
    # Load image and mask
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Calculate bounding box from mask
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, box_w, box_h = cv2.boundingRect(coords)

        # Create square crop with margin
        margin = 20
        center_x = x + box_w // 2
        center_y = y + box_h // 2
        size = max(box_w, box_h) + margin * 2

        crop_x1 = max(0, center_x - size // 2)
        crop_y1 = max(0, center_y - size // 2)
        crop_x2 = min(w, center_x + size // 2)
        crop_y2 = min(h, center_y + size // 2)
    else:
        crop_x1, crop_y1 = 0, 0
        crop_x2, crop_y2 = w, h

    # Create metadata
    metadata = {
        "video_frame_id": frame_id,
        "crop_box_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
        "video_frame_width": w,
        "video_frame_height": h,
        "crop_height": image_size,
        "crop_width": image_size,
    }

    meta_file = output_dir / f"{frame_id:07d}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Create box.txt
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    box_line = f"{frame_id:07d} {crop_x1:.2f} {crop_y1:.2f} {crop_w:.2f} {crop_h:.2f} {w:.2f} {h:.2f} 0.00 0\n"

    box_file = output_dir / f"{frame_id:07d}_box.txt"
    with open(box_file, "w") as f:
        f.write(box_line)

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all RGB images
    rgb_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))

    print(f"Found {len(rgb_files)} images")

    for idx, rgb_path in enumerate(tqdm(rgb_files)):
        frame_id = idx + 1
        prefix = f"{args.animal_name}_{frame_id:07d}"

        # Copy RGB image
        rgb_output = output_dir / f"{prefix}_rgb.png"
        img = cv2.imread(str(rgb_path))
        cv2.imwrite(str(rgb_output), img)

        # Generate or copy mask
        if args.generate_masks:
            # TODO: Implement SAM-based mask generation
            print("Mask generation not implemented. Please provide masks manually.")
            continue
        else:
            # Assume masks are provided with same name
            mask_path = rgb_path.parent / rgb_path.name.replace("_rgb", "_mask")
            if not mask_path.exists():
                mask_path = rgb_path.parent / f"{rgb_path.stem}_mask.png"

            if mask_path.exists():
                mask_output = output_dir / f"{prefix}_mask.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(str(mask_output), mask)
            else:
                print(f"Warning: Mask not found for {rgb_path.name}")
                continue

        # Create metadata and box
        create_metadata_and_box(rgb_output, mask_output, output_dir, frame_id, args.image_size)

        # Extract DINO features
        if args.extract_dino:
            # TODO: Implement DINO feature extraction
            pass

    print(f"✅ Dataset preparation complete! Output: {output_dir}")

if __name__ == "__main__":
    main()
```

---

## 부록 B: 자주 묻는 질문 (FAQ)

### Q1: 2D 이미지만 있으면 되나요?

**A**: 예, 2D RGB 이미지와 마스크만 있으면 됩니다. 3D 정보(포즈, depth 등)는 필요하지 않습니다. Fauna는 단안(monocular) 이미지로부터 3D 형상을 학습합니다.

---

### Q2: 비디오 시퀀스가 필수인가요?

**A**: 아니요. Few-shot 데이터셋은 정적 이미지만으로도 가능합니다. 다만 대규모(large-scale) 학습 시 비디오 시퀀스를 사용하면 시간적 일관성(temporal consistency)을 활용할 수 있어 품질이 향상됩니다.

---

### Q3: 최소 몇 장의 이미지가 필요한가요?

**A**:
- **Few-shot**: 최소 10장 (권장 30-50장)
- **Large-scale**: 최소 100장 (권장 1000장 이상)

다양한 자세와 시점이 중요합니다.

---

### Q4: DINO feature가 없어도 학습이 가능한가요?

**A**: 예, 가능합니다. DINO feature는 선택사항입니다. 다만 DINO feature를 사용하면 semantic correspondence 학습이 개선되어 품질이 향상됩니다.

---

### Q5: 마스크 품질이 얼마나 중요한가요?

**A**: 매우 중요합니다. 마스크 품질이 낮으면 형상 학습이 제대로 되지 않습니다. SAM/SAM2로 자동 생성한 후 수동으로 정제하는 것을 권장합니다.

---

### Q6: 다른 종류의 동물도 같은 방식으로 추가할 수 있나요?

**A**: 예. 이 가이드는 모든 4족 보행 동물에 적용 가능합니다. 다만 skeleton 파라미터(`num_body_bones`, `num_legs` 등)는 동물의 해부학적 구조에 맞게 조정해야 합니다.

---

## 요약

1. **필수 데이터**: RGB 이미지 + 마스크 + 메타데이터 + 바운딩 박스
2. **선택 데이터**: DINO features, clusters
3. **자동화**: 메타데이터/박스는 스크립트로 자동 생성 가능
4. **마스크**: SAM/SAM2 활용 또는 수동 annotation
5. **Config**: Dataset + Model + Training config 3개 파일 필요
6. **검증**: 파일 개수 일치 및 로딩 테스트 필수

**다음 단계**: 데이터 준비 완료 후 → Config 작성 → Debug mode 학습 → Full training

---

**문서 버전**: v1.0
**최종 수정**: 2025-11-19
**작성자**: Claude Code Assistant

1. 메인 가이드: docs/FAUNA_DATASET_GUIDE.md

  5개 섹션으로 구성:
  - 기존 Fauna 데이터셋 구조: Large-scale vs Few-shot 구분, 디렉토리 구조, 파일 네이밍 규칙
  - 필수 데이터 파일 스펙: 각 파일 타입별 상세 설명 및 생성 방법
  - 새로운 동물 추가 가이드: 생쥐 예시로 단계별 데이터 준비 과정
  - Config 파일 작성: Dataset, Model, Training config 설명
  - 검증 체크리스트: 자동화 스크립트 및 테스트 방법

  2. 템플릿 Config 파일 3종

  config/dataset/fauna_new_animal_template.yaml

  - 데이터셋 설정 템플릿
  - 배치 크기, 이미지 해상도, augmentation 옵션
  - GPU별 권장 설정

  config/model/fauna_new_animal_template.yaml

  - 모델 아키텍처 템플릿 (500줄 이상 상세 주석)
  - 동물 크기별 파라미터 가이드 (소/중/대형)
  - GPU 메모리 최적화 가이드
  - Skeleton 구조 설정 (body bones, leg bones)

  config/train_fauna_new_animal_template.yaml

  - 학습 설정 템플릿
  - 데이터셋 크기별 학습 전략
  - Debug-first 원칙 강조
  - 시나리오별 설정 예시 (few-shot, large-scale, fine-tuning, debug)

  핵심 발견 사항

  필수 데이터

  2D 이미지만 있으면 됩니다! 3D 정보 불필요:
  1. RGB 이미지 (*_rgb.png)
  2. 마스크 이미지 (*_mask.png) - SAM/SAM2로 자동 생성 가능
  3. 메타데이터 (*_metadata.json) - 스크립트로 자동 생성
  4. 바운딩 박스 (*_box.txt) - 스크립트로 자동 생성

  선택적 데이터

  - DINO features (*_feat16.png) - 품질 향상
  - Cluster maps (*_clusters.png) - Semantic parts

  데이터 준비 자동화

  가이드에 Python 스크립트 포함:
  - 메타데이터 자동 생성
  - 바운딩 박스 계산
  - DINO feature 추출
  - 전체 파이프라인 통합 스크립트

  동물별 파라미터 조정

  | 크기  | 예시     | spatial_scale | grid_res | num_body_bones |
  |-----|--------|---------------|----------|----------------|
  | 소형  | 생쥐, 쥐  | 4-5           | 64-128   | 4-6            |
  | 중형  | 고양이, 개 | 6-7           | 128-256  | 6-8            |
  | 대형  | 말, 코끼리 | 7-10          | 256-512  | 8-12           |

  생쥐 데이터 준비 요약

  1. 이미지 수집: 비디오 → FFmpeg 프레임 추출 또는 DANNCE 데이터셋
  2. 마스크 생성: Grounded-SAM/SAM2 자동 생성
  3. 메타데이터: Python 스크립트로 자동 생성
  4. 디렉토리 정리: data/fauna/Fauna_dataset/few_shot_animal3d/mouse/train/
  5. Config 작성: 템플릿 복사 → 파라미터 조정
  6. 검증: 파일 개수 확인 + 데이터 로딩 테스트
  7. Debug 학습: 5K iters (~15분) 먼저 실행
  8. Full 학습: Debug 성공 후 진행