# Fauna Dataset Complete Guide

**Last Updated**: 2025-11-24
**Purpose**: 완전한 Fauna 데이터셋 준비 가이드 (이론 + 실전 + 자동화)

---

## 목차

### Part 1: Quick Start (바로 시작하기) ⭐
1. [가장 빠른 시작 (3단계)](#1-가장-빠른-시작-3단계)
2. [자주 묻는 질문 (FAQ)](#2-자주-묻는-질문-faq)

### Part 2: Dataset Structure (데이터셋 구조)
3. [Fauna 데이터셋 전체 구조](#3-fauna-데이터셋-전체-구조)
4. [폴더 및 파일 네이밍 규칙](#4-폴더-및-파일-네이밍-규칙)

### Part 3: File Specifications (파일 스펙)
5. [필수 파일 상세 스펙](#5-필수-파일-상세-스펙)
6. [선택적 파일 (DINOv2 features)](#6-선택적-파일-dinov2-features)

### Part 4: Automation Scripts (자동화)
7. [완전 자동화 워크플로우](#7-완전-자동화-워크플로우)
8. [마스크 자동 생성 방법](#8-마스크-자동-생성-방법)
9. [Box.txt & Metadata 자동 생성](#9-boxtxt--metadata-자동-생성)

### Part 5: Animal-Specific Guides (동물별 가이드)
10. [생쥐(Mouse) 데이터 준비](#10-생쥐mouse-데이터-준비)
11. [크기별 파라미터 조정 (소/중/대형)](#11-크기별-파라미터-조정)

### Part 6: Configuration Files (Config 작성)
12. [Dataset Config 작성](#12-dataset-config-작성)
13. [Model Config 작성](#13-model-config-작성)
14. [Training Config 작성](#14-training-config-작성)

### Part 7: Validation & Troubleshooting (검증)
15. [데이터셋 검증 체크리스트](#15-데이터셋-검증-체크리스트)
16. [문제 해결 가이드](#16-문제-해결-가이드)

### Appendix (부록)
17. [전체 Python 스크립트](#17-전체-python-스크립트)
18. [데이터 품질 vs 학습 성공](#18-데이터-품질-vs-학습-성공)

---

# Part 1: Quick Start (바로 시작하기) ⭐

## 1. 가장 빠른 시작 (3단계)

### 단계 1: 이미지만 준비 (10분)

```bash
# 당신이 가진 것:
~/my_animal_images/
├── IMG_001.jpg
├── IMG_002.jpg
├── IMG_003.jpg
└── ... (최소 50-100장 권장)
```

**요구사항**:
- ✅ **어떤 포맷이든 OK**: jpg, png, jpeg, bmp
- ✅ **어떤 크기든 OK**: 자동으로 256×256 리사이징
- ✅ **어떤 이름이든 OK**: 자동 재명명
- ⚠️ **최소 개수**: 50-100장 (품질 확보용)

### 단계 2: 마스크 생성 (10-30분)

**Option A: SAM 사용 (최고 품질, 30분)**
```bash
# 설치 (한 번만)
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 실행
python scripts/generate_masks_sam.py \
  --input_dir ~/my_animal_images \
  --output_dir ~/my_animal_masks
```

**Option B: Background Subtraction (빠름, 10분, 정적 카메라)**
```bash
python scripts/generate_masks_bg_subtraction.py \
  --input_dir ~/my_animal_images \
  --threshold 20
```

### 단계 3: Fauna 포맷 변환 (5분)

```bash
python scripts/convert_to_fauna_format.py \
  --input_dir ~/my_animal_images \
  --output_dir data/fauna/Fauna_dataset/large_scale \
  --animal_name my_animal_custom \
  --split train \
  --auto_generate_metadata \
  --auto_generate_box
```

**완료!** 이제 학습 가능:
```bash
python run.py --config-name train_my_animal
```

---

## 2. 자주 묻는 질문 (FAQ)

### Q1: 이미지 크기가 모두 동일해야 하나요?

**아니요!** 어떤 크기든 상관없습니다.

- ✅ 1920×1080, 640×480, 512×512 등 **모든 크기 가능**
- ✅ 로딩 시 자동으로 256×256으로 리사이징됨
- ✅ 같은 폴더 내 이미지들이 서로 다른 크기여도 OK
- 💡 권장: 512×512 이상 (품질 유지)

### Q2: Mask가 없으면?

**Mask는 필수입니다!** 하지만 자동 생성 가능합니다.

- ❌ Mask 없이는 학습 불가
- ✅ SAM, Grounding DINO 등으로 자동 생성
- ✅ Background subtraction으로 생성
- ✅ CVAT, LabelMe 등으로 수동 생성

**추천 순서** (품질 순):
1. 🥇 **SAM** (Segment Anything Model) - 최고 품질
2. 🥈 **Manual annotation** (CVAT, LabelMe) - 정확
3. 🥉 **GrabCut** - 괜찮음
4. **Background Subtraction** - 정적 카메라만

### Q3: 최소 요구사항은?

**이미지 + 마스크만 있으면 됩니다!**

```
필수 (직접 준비):
✅ {frame_id}_rgb.png    (이미지, 아무 크기)
✅ {frame_id}_mask.png   (마스크, RGB와 같은 크기)

자동 생성 가능:
🤖 {frame_id}_box.txt       (마스크에서 계산)
🤖 {frame_id}_metadata.json (기본값 생성)
```

### Q4: 최소 이미지 개수는?

**동물 크기별 권장 개수**:

| 동물 크기 | 최소 | 권장 | 이상적 |
|----------|------|------|--------|
| **소형** (생쥐, 햄스터) | 50 | 150-200 | 500+ |
| **중형** (고양이, 개) | 100 | 200-300 | 1000+ |
| **대형** (말, 소) | 150 | 300-500 | 2000+ |

**이유**:
- 적을수록: Overfitting 위험, 일반화 부족
- 많을수록: 더 좋은 품질, 새로운 포즈 대응

### Q5: 직접 파일을 넣는다면 어떤 구조?

**가장 간단한 구조**:
```bash
data/fauna/Fauna_dataset/large_scale/my_animal_custom/
└── train/
    └── seq_000/
        ├── 0000000_rgb.png    # 이미지 1
        ├── 0000000_mask.png   # 마스크 1
        ├── 0000001_rgb.png    # 이미지 2
        ├── 0000001_mask.png   # 마스크 2
        ├── 0000002_rgb.png
        ├── 0000002_mask.png
        └── ...
```

**Frame ID 형식**: `0000000`, `0000001`, ... (7자리 zero-padded)

나머지 파일(`box.txt`, `metadata.json`)은 스크립트로 자동 생성!

---

# Part 2: Dataset Structure (데이터셋 구조)

## 3. Fauna 데이터셋 전체 구조

### 3.1 전체 디렉토리 개요

```
data/fauna/Fauna_dataset/
├── large_scale/                    # 대규모 비디오 시퀀스 데이터
│   ├── {animal}_comb_dinov2_new/  # 원본 Fauna 동물들
│   │   ├── train/
│   │   │   ├── {seq_id}/          # 시퀀스별 폴더
│   │   │   │   ├── {frame_id}_rgb.png
│   │   │   │   ├── {frame_id}_mask.png
│   │   │   │   ├── {frame_id}_metadata.json
│   │   │   │   ├── {frame_id}_box.txt
│   │   │   │   ├── {frame_id}_feat16.png      (optional)
│   │   │   │   ├── {frame_id}_clusters.png    (optional)
│   │   │   │   └── ...
│   │   │   ├── {seq_id}/
│   │   │   └── ...
│   │   ├── val/                   # Validation (optional)
│   │   └── test/                  # Test (optional)
│   │
│   └── {your_animal}_custom/      # 사용자 추가 동물
│       └── train/
│           └── seq_000/
│               └── ...
│
└── few_shot_animal3d/              # Few-shot 학습 데이터 (작은 데이터셋)
    ├── {animal}/
    │   └── train/
    │       ├── {animal}_{frame_id}_rgb.png
    │       ├── {animal}_{frame_id}_mask.png
    │       ├── {animal}_{frame_id}_metadata.json
    │       ├── {animal}_{frame_id}_box.txt
    │       └── ...
    └── ...
```

### 3.2 데이터셋 타입 구분

| 타입 | 용도 | 디렉토리 | 데이터 구조 | 권장 개수 |
|------|------|----------|------------|----------|
| **Large Scale** | 대량 학습 데이터 | `large_scale/{animal}_custom/` | 시퀀스별 폴더 (`train/`, `val/`, `test/`) | 100-500+ |
| **Few Shot** | 적은 데이터 학습 | `few_shot_animal3d/{animal}/` | 단일 `train/` 폴더 | 10-50 |

**선택 가이드**:
- ✅ **Large Scale 사용**: 100장 이상, 비디오 시퀀스, 다양한 포즈
- ✅ **Few Shot 사용**: 50장 미만, 정적 이미지, 빠른 테스트

---

## 4. 폴더 및 파일 네이밍 규칙

### 4.1 폴더 네이밍 규칙

**동물별 폴더명 형식**:
```
{animal}_custom          # 사용자 추가 동물
{animal}_comb_dinov2_new # 원본 Fauna 동물 (건드리지 말 것)
```

**예시**:
```
✅ mouse_dannce_6view         (DANNCE 데이터셋에서 온 생쥐)
✅ rabbit_custom              (사용자가 추가한 토끼)
✅ cat_indoor                 (실내 고양이)
✅ horse_jumping              (점프하는 말)

❌ MyRabbit                  (대소문자 섞임)
❌ rabbit-custom             (하이픈 사용)
❌ rabbit custom             (공백 사용)
```

**시퀀스 폴더명 형식** (Large Scale):
```
seq_000, seq_001, seq_002, ...
{video_id}_{seq_id}  (예: 000002_00002)
```

### 4.2 파일 네이밍 규칙

**Large Scale 형식**:
```
{sequence_id}/{frame_id}_{suffix}.{ext}

예시:
seq_000/0000000_rgb.png
seq_000/0000000_mask.png
seq_000/0000001_rgb.png
```

**Few Shot 형식**:
```
{animal}_{frame_id}_{suffix}.{ext}

예시:
rabbit_0000001_rgb.png
rabbit_0000001_mask.png
```

**Frame ID 형식**: `0000000`, `0000001`, `0000002`, ... (7자리 zero-padded)

**Suffix 목록**:
- `_rgb.png`: RGB 이미지 (필수)
- `_mask.png`: Binary mask (필수)
- `_metadata.json`: 메타데이터 (필수, 자동 생성 가능)
- `_box.txt`: Bounding box (필수, 자동 생성 가능)
- `_feat16.png`: DINOv2 features (선택적)
- `_clusters.png`: K-means clusters (선택적)

---

# Part 3: File Specifications (파일 스펙)

## 5. 필수 파일 상세 스펙

### 5.1 RGB 이미지 (`*_rgb.png`)

**설명**: 원본 RGB 이미지

**스펙**:
- **포맷**: PNG, JPG, JPEG, BMP 모두 가능
- **크기**: **아무거나** (1920×1080, 640×480, 256×256, 512×512 등)
- **자동 리사이징**: 로딩 시 256×256으로 자동 변환
- **채널**: RGB (3 channels)
- **권장**: 512×512 이상 (품질 유지)

**예시**: `0000080_rgb.png`, `rabbit_0000001_rgb.png`

---

### 5.2 마스크 이미지 (`*_mask.png`)

**설명**: 동물 영역 binary mask

**스펙**:
- **포맷**: PNG (grayscale)
- **크기**: RGB 이미지와 동일 (다르면 자동 리사이징)
- **값**:
  - `255` (흰색): 동물 영역 (foreground)
  - `0` (검은색): 배경 (background)
- **중간값 금지**: 0 또는 255만 사용

**생성 방법** (품질 순):
1. 🥇 **SAM (Segment Anything Model)** - 최고 품질
2. 🥈 **Manual Annotation (CVAT, LabelMe)** - 정확
3. 🥉 **GrabCut** - 괜찮은 품질
4. **Background Subtraction** - 정적 카메라만

**예시**: `0000080_mask.png`

---

### 5.3 메타데이터 (`*_metadata.json`)

**설명**: 프레임 메타정보 (크롭 박스, 원본 해상도 등)

**스펙**:
- **포맷**: JSON
- **인코딩**: UTF-8

**필수 필드** (4개):
```json
{
    "video_frame_id": 80,                       // 원본 비디오 프레임 번호 (int)
    "crop_box_xyxy": [982, -19, 1802, 801],    // 크롭 영역 [x1, y1, x2, y2] (list)
    "video_frame_width": 1920,                  // 원본 비디오 너비 (int)
    "video_frame_height": 1080                  // 원본 비디오 높이 (int)
}
```

**선택적 필드**:
```json
{
    "sharpness": 185.29,           // 이미지 선명도 (float, optional)
    "crop_height": 256,            // 크롭 후 높이 (int, optional)
    "crop_width": 256,             // 크롭 후 너비 (int, optional)
    "label": 7,                    // 카테고리 라벨 (int, optional)
    "source_image": "IMG_001.jpg", // 원본 파일명 (str, custom)
    "animal_category": "rabbit"    // 동물 종류 (str, custom)
}
```

**자동 생성 코드**:
```python
import json

def create_metadata(frame_id, crop_box, orig_width, orig_height, crop_size=256):
    """
    메타데이터 자동 생성

    Args:
        frame_id: 프레임 번호 (int)
        crop_box: [x1, y1, x2, y2] (list of int)
        orig_width: 원본 이미지 너비 (int)
        orig_height: 원본 이미지 높이 (int)
        crop_size: 크롭 크기 (int, default 256)
    """
    x1, y1, x2, y2 = crop_box
    metadata = {
        "video_frame_id": frame_id,
        "crop_box_xyxy": [x1, y1, x2, y2],
        "video_frame_width": orig_width,
        "video_frame_height": orig_height,
        # Optional fields
        "crop_height": crop_size,
        "crop_width": crop_size,
    }
    return metadata

# 예시 사용
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

### 5.4 바운딩 박스 (`*_box.txt`)

**설명**: 단일 라인 텍스트, 크롭 정보

**스펙**:
- **포맷**: Plain text (공백으로 구분)
- **라인 수**: 1줄
- **값 개수**: 9개

**형식**:
```
{frame_id} {x} {y} {width} {height} {full_w} {full_h} {sharpness} {label}
```

**실제 예시**:
```
0000080 982.00 -19.00 820.00 820.00 1920.00 1080.00 185.29 7
```

**필드 설명**:
- `frame_id`: 프레임 ID (7자리 숫자, 예: 0000080)
- `x, y`: 크롭 좌측 상단 좌표 (float)
- `width, height`: 크롭 영역 크기 (float)
- `full_w, full_h`: 원본 이미지 크기 (float)
- `sharpness`: 이미지 선명도 (float, 0.00으로 설정 가능)
- `label`: 카테고리 라벨 (int, 0으로 설정 가능)

**⚠️ 중요: 올바른 자동 생성 코드**

```python
import numpy as np
import cv2

def create_box_txt(mask_path, frame_id, original_w, original_h, output_path):
    """
    마스크에서 box.txt 자동 생성 (올바른 형식)

    Args:
        mask_path: 마스크 파일 경로 (str)
        frame_id: 프레임 ID (int)
        original_w: 원본 이미지 너비 (int)
        original_h: 원본 이미지 높이 (int)
        output_path: 출력 box.txt 경로 (str)
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Calculate bounding box from mask
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        raise ValueError(f"Empty mask: {mask_path}")

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add 5% padding
    pad_y = int((y_max - y_min) * 0.05)
    pad_x = int((x_max - x_min) * 0.05)
    y_min = max(0, y_min - pad_y)
    y_max = min(mask.shape[0], y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(mask.shape[1], x_max + pad_x)

    # ✅ CORRECT: Calculate width and height (NOT x_max, y_max!)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # ✅ CORRECT: box.txt format is 9 values in one line
    box_line = f"{frame_id:07d} {x_min:.2f} {y_min:.2f} {bbox_w:.2f} {bbox_h:.2f} {original_w:.2f} {original_h:.2f} 0.00 0\n"

    # Write to file
    with open(output_path, "w") as f:
        f.write(box_line)

    return [x_min, y_min, bbox_w, bbox_h]

# 예시 사용
bbox = create_box_txt(
    mask_path="0000080_mask.png",
    frame_id=80,
    original_w=1920,
    original_h=1080,
    output_path="0000080_box.txt"
)
print(f"Generated bbox: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
```

**❌ 잘못된 예시 (사용 금지)**:
```python
# ❌ WRONG: Uses x_max, y_max instead of width, height
bbox = [x_min, y_min, x_max, y_max]  # WRONG!
np.savetxt(box_path, bbox, fmt='%d')  # WRONG! Missing 5 values
```

**✅ 올바른 예시**:
```python
# ✅ CORRECT: Uses width, height
bbox_w = x_max - x_min
bbox_h = y_max - y_min
box_line = f"{frame_id:07d} {x_min:.2f} {y_min:.2f} {bbox_w:.2f} {bbox_h:.2f} {original_w:.2f} {original_h:.2f} 0.00 0\n"
with open(box_path, "w") as f:
    f.write(box_line)
```

---

## 6. 선택적 파일 (DINOv2 Features)

### 6.1 DINOv2 Feature Map (`*_feat16.png`)

**설명**: DINOv2 모델에서 추출한 16채널 feature map

**스펙**:
- **포맷**: PNG (16-channel)
- **크기**: 256×256 (고정)
- **채널**: 16 (semantic features)
- **필수 여부**: ❌ 선택적 (없어도 학습 가능)

**생성 방법**: DINOv2 모델로 추출 (고급 사용자)

---

### 6.2 K-means Clusters (`*_clusters.png`)

**설명**: Feature map을 K-means clustering한 결과

**스펙**:
- **포맷**: PNG (grayscale or RGB)
- **크기**: 256×256
- **필수 여부**: ❌ 선택적

---

# Part 4: Automation Scripts (자동화)

## 7. 완전 자동화 워크플로우

### 7.1 Step 1: 이미지 준비

```bash
# 당신이 가진 것:
~/my_rabbit_images/
├── IMG_001.jpg
├── IMG_002.jpg
└── ...
```

- ✅ 어떤 포맷이든 OK: jpg, png, jpeg, bmp
- ✅ 어떤 크기든 OK
- ✅ 어떤 이름이든 OK

---

### 7.2 Step 2: 마스크 생성 (3가지 방법)

#### Option 1: SAM (Segment Anything Model) - 최고 품질 ⭐

**설치**:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**스크립트**: [섹션 8 참조](#8-마스크-자동-생성-방법)

#### Option 2: Background Subtraction - 빠름 (정적 카메라)

**스크립트**: [섹션 8 참조](#8-마스크-자동-생성-방법)

#### Option 3: Manual Annotation - 최고 정확도

- **CVAT**: https://www.cvat.ai/ (무료, 웹 기반)
- **LabelMe**: https://github.com/wkentaro/labelme (무료, 로컬)

---

### 7.3 Step 3: Fauna 포맷 변환

**스크립트**: [섹션 9 참조](#9-boxtxt--metadata-자동-생성)

```bash
python convert_to_fauna_format.py \
  --input_dir ~/my_rabbit_images \
  --output_dir data/fauna/Fauna_dataset/large_scale \
  --animal_name rabbit_custom \
  --split train
```

**자동 생성되는 것**:
- ✅ `_box.txt`: 마스크에서 계산
- ✅ `_metadata.json`: 기본값 생성
- ✅ Frame ID 재명명: `0000000`, `0000001`, ...

---

### 7.4 Step 4: 검증

```bash
python verify_dataset.py \
  --data_dir data/fauna/Fauna_dataset/large_scale/rabbit_custom/train/seq_000
```

**체크 항목**:
- ✅ 모든 파일 존재 (rgb, mask, box, metadata)
- ✅ 이미지 로딩 가능
- ✅ 마스크 foreground 픽셀 존재
- ✅ Bounding box 유효성
- ✅ JSON 파싱 가능

---

## 8. 마스크 자동 생성 방법

### 8.1 SAM (Segment Anything Model)

**코드**: `scripts/generate_masks_sam.py`

```python
import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

def generate_mask_sam(image_path, output_path):
    """SAM으로 자동 마스크 생성"""
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image_rgb)

    # Get largest mask (assumed to be main object)
    if len(masks) > 0:
        largest_mask = max(masks, key=lambda x: x['area'])
        mask = largest_mask['segmentation'].astype(np.uint8) * 255
    else:
        # Fallback: threshold-based
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Save mask
    cv2.imwrite(str(output_path), mask)
    return mask

# Usage
for img_path in Path("~/my_rabbit_images").glob("*.jpg"):
    mask_path = img_path.with_name(img_path.stem + "_mask.png")
    generate_mask_sam(img_path, mask_path)
```

**장점**:
- 🥇 최고 품질
- ✅ 어떤 배경도 가능
- ✅ 복잡한 형태도 정확

**단점**:
- ⏱️ 느림 (2-5초/이미지)
- 💾 모델 크기 큼 (2.5GB)

---

### 8.2 Background Subtraction

**코드**: `scripts/generate_masks_bg_subtraction.py`

```python
import cv2
from pathlib import Path

def generate_mask_bg_subtract(image_path, output_path, threshold=10):
    """Background subtraction으로 마스크 생성 (정적 카메라 전용)"""
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Simple threshold
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(str(output_path), mask)
    return mask

# Usage
for img_path in Path("~/my_rabbit_images").glob("*.jpg"):
    mask_path = img_path.with_name(img_path.stem + "_mask.png")
    generate_mask_bg_subtract(img_path, mask_path, threshold=20)
```

**장점**:
- ⚡ 매우 빠름 (0.1초/이미지)
- 💡 간단함

**단점**:
- ⚠️ 정적 카메라만 가능
- ⚠️ 단순한 배경 필요

---

### 8.3 GrabCut

**코드**: `scripts/generate_masks_grabcut.py`

```python
import cv2
import numpy as np

def generate_mask_grabcut(image_path, output_path):
    """GrabCut으로 마스크 생성"""
    image = cv2.imread(str(image_path))
    mask = np.zeros(image.shape[:2], np.uint8)

    # Initial rectangle (80% of image)
    h, w = image.shape[:2]
    margin = 0.1
    rect = (int(w*margin), int(h*margin), int(w*(1-2*margin)), int(h*(1-2*margin)))

    # GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask2 = mask2 * 255

    cv2.imwrite(str(output_path), mask2)
    return mask2
```

**장점**:
- 🥉 괜찮은 품질
- ⚡ 빠름 (1초/이미지)

**단점**:
- ⚠️ 중앙에 객체 있어야 함
- ⚠️ 복잡한 배경 어려움

---

## 9. Box.txt & Metadata 자동 생성

### 9.1 완전 자동화 스크립트

**코드**: `scripts/convert_to_fauna_format.py`

```python
import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_to_fauna_format(
    input_dir: str,
    output_dir: str,
    animal_name: str,
    split: str = "train",
    auto_generate_masks: bool = False,
    mask_method: str = "threshold"
):
    """
    이미지(+마스크)를 Fauna 포맷으로 변환

    Args:
        input_dir: 원본 이미지 디렉토리
        output_dir: 출력 디렉토리 (data/fauna/Fauna_dataset/large_scale)
        animal_name: 동물 이름 (예: rabbit_custom)
        split: train/val/test
        auto_generate_masks: True면 마스크 자동 생성
        mask_method: threshold, grabcut, sam 중 선택
    """

    # Create output directory
    seq_dir = Path(output_dir) / animal_name / split / "seq_000"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_exts:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    print(f"Found {len(image_files)} images")

    for idx, img_path in enumerate(tqdm(image_files)):
        frame_id = idx

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load: {img_path}")
            continue

        original_h, original_w = image.shape[:2]

        # 1. Save RGB image
        rgb_path = seq_dir / f"{frame_id:07d}_rgb.png"
        cv2.imwrite(str(rgb_path), image)

        # 2. Load or generate mask
        mask_path_input = img_path.with_name(img_path.stem + "_mask.png")
        if mask_path_input.exists():
            mask = cv2.imread(str(mask_path_input), cv2.IMREAD_GRAYSCALE)
        elif auto_generate_masks:
            if mask_method == "threshold":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            elif mask_method == "grabcut":
                mask = generate_mask_grabcut_inline(image)
            elif mask_method == "sam":
                raise NotImplementedError("SAM requires separate script")
        else:
            raise FileNotFoundError(f"Mask not found: {mask_path_input}")

        # Save mask
        mask_path = seq_dir / f"{frame_id:07d}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        # 3. Generate box.txt (올바른 형식!)
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            print(f"Empty mask: {frame_id}")
            continue

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Add 5% padding
        pad_y = int((y_max - y_min) * 0.05)
        pad_x = int((x_max - x_min) * 0.05)
        y_min = max(0, y_min - pad_y)
        y_max = min(mask.shape[0], y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(mask.shape[1], x_max + pad_x)

        # ✅ CORRECT: Calculate width and height
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # ✅ CORRECT: box.txt format (9 values)
        box_line = f"{frame_id:07d} {x_min:.2f} {y_min:.2f} {bbox_w:.2f} {bbox_h:.2f} {original_w:.2f} {original_h:.2f} 0.00 0\n"

        box_file = seq_dir / f"{frame_id:07d}_box.txt"
        with open(box_file, "w") as f:
            f.write(box_line)

        # 4. Generate metadata.json
        metadata = {
            "video_frame_id": frame_id,
            "crop_box_xyxy": [int(x_min), int(y_min), int(x_min + bbox_w), int(y_min + bbox_h)],
            "video_frame_width": original_w,
            "video_frame_height": original_h,
            "crop_height": 256,
            "crop_width": 256,
            "source_image": img_path.name,
            "animal_category": animal_name
        }

        metadata_file = seq_dir / f"{frame_id:07d}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    print(f"\n✅ Conversion complete!")
    print(f"   Output: {seq_dir}")
    print(f"   Files: {len(image_files) * 4} total")

def generate_mask_grabcut_inline(image):
    """GrabCut inline implementation"""
    mask = np.zeros(image.shape[:2], np.uint8)
    h, w = image.shape[:2]
    margin = 0.1
    rect = (int(w*margin), int(h*margin), int(w*(1-2*margin)), int(h*(1-2*margin)))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2 * 255

# Usage example
if __name__ == "__main__":
    convert_to_fauna_format(
        input_dir="~/my_rabbit_images",
        output_dir="data/fauna/Fauna_dataset/large_scale",
        animal_name="rabbit_custom",
        split="train",
        auto_generate_masks=False  # Masks already exist
    )
```

**실행**:
```bash
python convert_to_fauna_format.py \
  --input_dir ~/my_rabbit_images \
  --output_dir data/fauna/Fauna_dataset/large_scale \
  --animal_name rabbit_custom \
  --split train
```

---

# Part 5: Animal-Specific Guides (동물별 가이드)

## 10. 생쥐(Mouse) 데이터 준비

### 10.1 생쥐 특징

- **크기**: 매우 작음 (5-10cm)
- **권장 데이터 개수**: 최소 50, 권장 150-200
- **카메라**: Multi-view (6-view DANNCE) 또는 단일 뷰
- **배경**: 흰색 또는 단순 배경 권장

### 10.2 DANNCE 데이터셋 활용

**DANNCE 데이터셋 구조**:
```
/media/joon/kafka/data/mouse_dannce_6view/
├── train/
│   ├── 000000_00000/  # Sequence folders
│   │   ├── 0Camera1.jpg
│   │   ├── 1Camera1.jpg
│   │   └── ...
│   ├── 000000_00001/
│   └── ...
└── test/
```

**변환 방법**:
```bash
# 각 뷰를 개별 프레임으로 변환
python convert_dannce_to_fauna.py \
  --dannce_dir /media/joon/kafka/data/mouse_dannce_6view \
  --output_dir data/fauna/Fauna_dataset/large_scale \
  --animal_name mouse_dannce_6view \
  --views 6
```

### 10.3 마스크 생성 (생쥐 특화)

**옵션 1: Background Subtraction (흰색 배경)**
```bash
python generate_masks_bg_subtraction.py \
  --input_dir /media/joon/kafka/data/mouse_dannce_6view/train/000000_00000 \
  --threshold 20
```

**옵션 2: SAM (복잡한 배경)**
```bash
python generate_masks_sam.py \
  --input_dir /media/joon/kafka/data/mouse_dannce_6view/train/000000_00000
```

---

## 11. 크기별 파라미터 조정

### 11.1 동물 크기 분류

| 크기 | 동물 예시 | 특징 |
|------|----------|------|
| **소형** | 생쥐, 햄스터, 새 | 5-15cm, 세밀한 구조 |
| **중형** | 고양이, 토끼, 여우 | 30-60cm, 균형 잡힌 비율 |
| **대형** | 개, 말, 소 | 1-2m, 큰 구조 |

### 11.2 Model Config 파라미터

**소형 동물 (생쥐)**:
```yaml
# config/model/fauna_mouse.yaml
spatial_scale: 4.5      # 작은 스케일
grid_res: 64            # 메모리 친화적
num_body_bones: 5       # 적은 관절
```

**중형 동물 (고양이)**:
```yaml
# config/model/fauna_cat.yaml
spatial_scale: 5.5      # 중간 스케일
grid_res: 128           # 더 높은 해상도
num_body_bones: 6       # 표준 관절
```

**대형 동물 (말)**:
```yaml
# config/model/fauna_horse.yaml
spatial_scale: 7.0      # 큰 스케일
grid_res: 128           # 높은 해상도
num_body_bones: 6       # 표준 관절
```

### 11.3 권장 데이터 개수

| 크기 | 최소 | 권장 | 이상적 |
|------|------|------|--------|
| **소형** | 50 | 150-200 | 500+ |
| **중형** | 100 | 200-300 | 1000+ |
| **대형** | 150 | 300-500 | 2000+ |

---

# Part 6: Configuration Files (Config 작성)

## 12. Dataset Config 작성

**파일**: `config/dataset/{animal_name}.yaml`

**예시**: `config/dataset/rabbit_custom.yaml`

```yaml
_target_: fauna_dataset.FaunaDataset

# Data paths
train_data_dir: data/fauna/Fauna_dataset/large_scale/rabbit_custom
split: train

# Data loading
batch_size: 4
num_workers: 4
shuffle: true

# Data augmentation
use_augmentation: true
rotation_range: 15  # degrees
scale_range: [0.9, 1.1]
flip_horizontal: true

# Image settings
image_size: 256  # Fixed size
normalize: true

# Optional features
use_dinov2_features: false  # Set true if you have feat16.png files
use_clusters: false

# Sequence settings
sequence_length: 10  # Frames per sequence
sample_strategy: uniform  # uniform, random, temporal
```

---

## 13. Model Config 작성

**파일**: `config/model/{animal_name}.yaml`

**예시**: `config/model/rabbit_custom.yaml`

```yaml
_target_: model.FaunaModel

# Animal-specific settings
animal_name: rabbit
animal_category: mammal
size_category: medium  # small, medium, large

# Shape predictor
cfg_predictor_base:
  _target_: model.predictors.BasePredictorBase
  cfg_shape:
    grid_res: 128  # 64 for small animals, 128 for medium/large
    spatial_scale: 5.5  # 4.5 (small), 5.5 (medium), 7.0 (large)
    init_sdf: ellipsoid  # Start from ellipsoid shape

  # Articulation settings
  num_body_bones: 6  # 5 for very small animals, 6 standard
  articulation_iter_range: [20000, inf]  # When to activate articulation

  # Texture settings
  texture_dim: 64
  texture_resolution: 256

# Loss weights
loss_weights:
  mask_loss: 1.0
  rgb_loss: 1.0
  sdf_reg: 0.01
  articulation_reg: 0.001

# Optimizer
learning_rate: 1e-4
weight_decay: 1e-6
```

---

## 14. Training Config 작성

**파일**: `config/train_{animal_name}.yaml`

**예시**: `config/train_rabbit_custom.yaml`

```yaml
defaults:
  - dataset: rabbit_custom
  - model: rabbit_custom

# Experiment settings
exp_name: rabbit_custom_from_scratch
seed: 42

# Training iterations
num_iters: 100000  # 50K for debug, 100-200K for full
save_checkpoint_freq: 5000
log_image_freq: 1000
validate_freq: 5000

# Hardware settings
device: cuda
num_gpus: 1
disable_tf32: true  # Set true for RTX 3060

# Training strategy
strategy: from_scratch  # from_scratch, finetune, resume

# Resume (if strategy == resume or finetune)
resume: null  # Path to checkpoint, e.g., results/checkpoint50000.pth

# Output paths
output_dir: results
checkpoint_dir: results/checkpoints
log_dir: results/logs

# Logging
use_wandb: true
wandb_project: 3DAnimals
wandb_entity: your_username

# Debug mode (for testing)
debug: false
debug_iters: 5000
```

---

# Part 7: Validation & Troubleshooting (검증)

## 15. 데이터셋 검증 체크리스트

### 15.1 자동 검증 스크립트

**코드**: `scripts/verify_dataset.py`

```python
import os
import cv2
import json
from pathlib import Path
from tqdm import tqdm

def verify_dataset(data_dir: str):
    """
    Fauna 데이터셋 검증

    Args:
        data_dir: 데이터 디렉토리 (예: .../train/seq_000)
    """

    data_dir = Path(data_dir)
    errors = []
    warnings = []

    # Find all frames
    rgb_files = sorted(data_dir.glob("*_rgb.png"))
    num_frames = len(rgb_files)

    print(f"Found {num_frames} frames in {data_dir}")

    for rgb_path in tqdm(rgb_files):
        frame_id = rgb_path.stem.replace("_rgb", "")

        # Check required files
        mask_path = data_dir / f"{frame_id}_mask.png"
        box_path = data_dir / f"{frame_id}_box.txt"
        metadata_path = data_dir / f"{frame_id}_metadata.json"

        # 1. Check file existence
        if not mask_path.exists():
            errors.append(f"Missing mask: {frame_id}")
            continue
        if not box_path.exists():
            errors.append(f"Missing box: {frame_id}")
            continue
        if not metadata_path.exists():
            errors.append(f"Missing metadata: {frame_id}")
            continue

        # 2. Check image loading
        rgb = cv2.imread(str(rgb_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if rgb is None:
            errors.append(f"Cannot load RGB: {frame_id}")
            continue
        if mask is None:
            errors.append(f"Cannot load mask: {frame_id}")
            continue

        # 3. Check mask values
        unique_values = set(mask.flatten())
        if not unique_values.issubset({0, 255}):
            warnings.append(f"Mask has non-binary values: {frame_id} (values: {unique_values})")

        # 4. Check mask has foreground
        if mask.max() == 0:
            errors.append(f"Empty mask (all zeros): {frame_id}")
            continue

        # 5. Check box.txt format
        with open(box_path, "r") as f:
            box_line = f.read().strip()

        box_values = box_line.split()
        if len(box_values) != 9:
            errors.append(f"box.txt has {len(box_values)} values (expected 9): {frame_id}")
        else:
            try:
                # Parse values
                fid, x, y, w, h, full_w, full_h, sharp, label = box_values
                x, y, w, h = float(x), float(y), float(w), float(h)

                # Check validity
                if w <= 0 or h <= 0:
                    errors.append(f"Invalid box dimensions (w={w}, h={h}): {frame_id}")
                if x < 0 or y < 0:
                    warnings.append(f"Negative box coordinates (x={x}, y={y}): {frame_id}")
            except ValueError as e:
                errors.append(f"Cannot parse box.txt: {frame_id} ({e})")

        # 6. Check metadata.json
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check required fields
            required_fields = ["video_frame_id", "crop_box_xyxy", "video_frame_width", "video_frame_height"]
            for field in required_fields:
                if field not in metadata:
                    errors.append(f"Missing metadata field '{field}': {frame_id}")

            # Check crop_box_xyxy format
            if "crop_box_xyxy" in metadata:
                crop_box = metadata["crop_box_xyxy"]
                if not isinstance(crop_box, list) or len(crop_box) != 4:
                    errors.append(f"Invalid crop_box_xyxy format: {frame_id}")

        except json.JSONDecodeError as e:
            errors.append(f"Cannot parse metadata.json: {frame_id} ({e})")

    # Print results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"Total frames: {num_frames}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\n🔴 ERRORS:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if warnings:
        print("\n🟡 WARNINGS:")
        for warn in warnings[:10]:  # Show first 10
            print(f"  - {warn}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    if not errors and not warnings:
        print("\n✅ All checks passed!")

    return len(errors) == 0

# Usage
if __name__ == "__main__":
    verify_dataset("data/fauna/Fauna_dataset/large_scale/rabbit_custom/train/seq_000")
```

**실행**:
```bash
python verify_dataset.py --data_dir data/fauna/Fauna_dataset/large_scale/rabbit_custom/train/seq_000
```

---

### 15.2 수동 검증 체크리스트

**디렉토리 구조**:
- [ ] `data/fauna/Fauna_dataset/large_scale/{animal}_custom/` 폴더 존재
- [ ] `train/seq_000/` 하위 폴더 존재
- [ ] 모든 프레임에 대해 4개 파일 존재 (rgb, mask, box, metadata)

**파일 네이밍**:
- [ ] Frame ID가 7자리 zero-padded (예: 0000000, 0000001)
- [ ] 모든 파일이 동일한 frame ID 사용
- [ ] 파일 확장자 정확 (.png, .txt, .json)

**이미지 내용**:
- [ ] RGB 이미지 로딩 가능
- [ ] Mask 이미지 로딩 가능
- [ ] Mask가 binary (0 또는 255)
- [ ] Mask에 foreground 픽셀 존재 (max > 0)

**Box.txt 형식**:
- [ ] 9개 값 존재 (공백으로 구분)
- [ ] Width, height가 양수
- [ ] Frame ID가 7자리 숫자

**Metadata.json 형식**:
- [ ] JSON 파싱 가능
- [ ] 필수 필드 4개 존재 (video_frame_id, crop_box_xyxy, video_frame_width, video_frame_height)
- [ ] crop_box_xyxy가 4개 원소 리스트

---

## 16. 문제 해결 가이드

### 문제 1: "Cannot load image" 에러

**증상**:
```
ValueError: Cannot load image: 0000050_rgb.png
```

**원인**:
- 손상된 이미지 파일
- 지원하지 않는 포맷
- 권한 문제

**해결**:
```bash
# 이미지 다시 확인
file 0000050_rgb.png

# 다시 변환
convert 0000050_rgb.jpg 0000050_rgb.png

# 권한 확인
chmod 644 0000050_rgb.png
```

---

### 문제 2: "Empty mask" 에러

**증상**:
```
ValueError: Empty mask (all zeros): 0000050
```

**원인**:
- 마스크 생성 실패
- Threshold 너무 높음
- 배경과 동물 구분 안 됨

**해결**:
```bash
# Threshold 낮추기
python generate_masks_bg_subtraction.py --threshold 5

# SAM 사용
python generate_masks_sam.py --input_dir ...

# 수동 생성 (CVAT, LabelMe)
```

---

### 문제 3: "Invalid box.txt format" 에러

**증상**:
```
ValueError: box.txt has 4 values (expected 9): 0000050
```

**원인**:
- 잘못된 box.txt 생성 코드 사용
- 수동으로 잘못 작성

**해결**:
```bash
# 올바른 스크립트로 재생성
python convert_to_fauna_format.py \
  --input_dir ... \
  --output_dir ... \
  --animal_name ...
```

**확인**:
```bash
# box.txt 내용 확인
cat 0000050_box.txt
# 출력 예시: 0000050 100.00 50.00 200.00 180.00 640.00 480.00 0.00 0
```

---

### 문제 4: GPU Out of Memory

**증상**:
```
RuntimeError: CUDA out of memory
```

**원인**:
- grid_res 너무 높음 (128)
- batch_size 너무 큼

**해결**:
```yaml
# config/model/{animal}.yaml
cfg_predictor_base:
  cfg_shape:
    grid_res: 64  # 128 → 64로 감소

# config/train_{animal}.yaml
batch_size: 2  # 4 → 2로 감소
```

---

### 문제 5: 학습이 수렴하지 않음

**증상**:
- Loss가 감소하지 않음
- Reconstruction 품질 안 좋음

**원인**:
- 데이터 부족 (50장 미만)
- Learning rate 너무 높거나 낮음
- 부적절한 파라미터 (spatial_scale)

**해결**:
1. **데이터 증가**: 50 → 150-200장
2. **Learning rate 조정**:
   ```yaml
   learning_rate: 5e-5  # 1e-4에서 감소
   ```
3. **Pretrained 모델 사용**:
   ```yaml
   strategy: finetune
   resume: results/fauna/pretrained_fauna/checkpoint.pth
   ```
4. **Longer training**:
   ```yaml
   num_iters: 200000  # 100K → 200K
   ```

---

# Appendix (부록)

## 17. 전체 Python 스크립트

### 17.1 convert_to_fauna_format.py

**위치**: `scripts/convert_to_fauna_format.py`

**전체 코드**: [섹션 9.1 참조](#91-완전-자동화-스크립트)

---

### 17.2 verify_dataset.py

**위치**: `scripts/verify_dataset.py`

**전체 코드**: [섹션 15.1 참조](#151-자동-검증-스크립트)

---

### 17.3 generate_masks_sam.py

**위치**: `scripts/generate_masks_sam.py`

**전체 코드**: [섹션 8.1 참조](#81-sam-segment-anything-model)

---

### 17.4 generate_masks_bg_subtraction.py

**위치**: `scripts/generate_masks_bg_subtraction.py`

**전체 코드**: [섹션 8.2 참조](#82-background-subtraction)

---

## 18. 데이터 품질 vs 학습 성공

### 18.1 데이터 품질 체크리스트

| 항목 | 낮음 (❌) | 중간 (⚠️) | 높음 (✅) |
|------|----------|----------|----------|
| **이미지 개수** | < 50 | 50-100 | 100+ |
| **Mask 품질** | Threshold | GrabCut | SAM/Manual |
| **포즈 다양성** | 단일 포즈 | 2-3 포즈 | 5+ 포즈 |
| **조명 다양성** | 단일 조명 | 2-3 조명 | 다양 |
| **배경 다양성** | 단일 배경 | 2-3 배경 | 다양 |
| **이미지 선명도** | 흐릿 | 보통 | 선명 |

### 18.2 학습 성공 예측

| 데이터 품질 | 예상 Texture 품질 | 예상 Articulation | 예상 Generalization | 학습 시간 |
|------------|------------------|------------------|---------------------|----------|
| **높음** (✅ 전부) | 8-9/10 | 안정적 | 새 포즈 대응 | 10-12h |
| **중간** (⚠️ 일부) | 6-7/10 | 제한적 | 학습 포즈만 | 5-7h |
| **낮음** (❌ 대부분) | 4-5/10 | 불안정 | Overfitting | 2-3h |

### 18.3 개선 우선순위

**우선순위 1: 데이터 개수**
- 50 → 150-200장으로 증가
- 가장 큰 영향

**우선순위 2: Mask 품질**
- Threshold → SAM 전환
- Texture 품질 향상

**우선순위 3: 포즈 다양성**
- 다양한 동작 포함
- Articulation 안정성

**우선순위 4: Pretrained 모델 사용**
- Multi-animal semantic bank
- Few-shot learning 효율

---

## 마무리

이 문서는 Fauna 데이터셋 준비의 모든 것을 담고 있습니다:

1. ⚡ **Quick Start**: 10분 내 시작 가능
2. 📚 **이론**: 전체 구조 이해
3. 🤖 **자동화**: 바로 실행 가능한 스크립트
4. 🐭 **동물별 가이드**: 생쥐, 고양이, 말 등
5. ⚙️ **Config 작성**: Dataset, Model, Training
6. ✅ **검증**: 자동화된 체크
7. 🔧 **문제 해결**: 일반적인 이슈 해결

**다음 단계**:
1. [Quick Start](#1-가장-빠른-시작-3단계)로 바로 시작
2. 문제 발생 시 [문제 해결 가이드](#16-문제-해결-가이드) 참조
3. 고급 사용자: [Config 작성](#part-6-configuration-files-config-작성)으로 커스터마이징

**질문이나 문제가 있다면**:
- GitHub Issues: https://github.com/3DAnimals/Fauna/issues
- 이 문서: `docs/FAUNA_DATASET_COMPLETE_GUIDE.md`

---

**Last Updated**: 2025-11-24
**Version**: 1.0
**Authors**: 3DAnimals Team + Community
