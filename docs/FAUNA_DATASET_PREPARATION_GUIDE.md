# Fauna Dataset Preparation Guide

**Last Updated**: 2025-11-21

---

## Quick Answers

### Q1: 이미지 크기가 모두 동일해야 하나요?

**아니요!** 어떤 크기든 상관없습니다.

- ✅ 1920×1080, 640×480, 512×512 등 **모든 크기 가능**
- ✅ 로딩 시 자동으로 256×256으로 리사이징됨
- ✅ 같은 폴더 내 이미지들이 서로 다른 크기여도 OK

### Q2: Mask가 없으면?

**Mask는 필수입니다!** 하지만 자동 생성 가능합니다.

- ❌ Mask 없이는 학습 불가
- ✅ SAM, Grounding DINO 등으로 자동 생성
- ✅ Background subtraction으로 생성
- ✅ 아래 제공된 스크립트 사용

### Q3: 최소 요구사항은?

**이미지 + 마스크 쌍만 있으면 됩니다!**

```
필수:
- {frame_id}_rgb.png    (이미지, 아무 크기)
- {frame_id}_mask.png   (마스크, RGB와 같은 크기)

자동 생성 가능:
- {frame_id}_box.txt       (마스크에서 계산)
- {frame_id}_metadata.json (기본값 생성)
```

### Q4: 직접 넣는다면 어떤 구조?

**가장 간단한 구조**:

```bash
data/fauna/Fauna_dataset/large_scale/my_animal/
└── train/
    └── seq_000/
        ├── 0000000_rgb.png    # 이미지 1
        ├── 0000000_mask.png   # 마스크 1
        ├── 0000001_rgb.png    # 이미지 2
        ├── 0000001_mask.png   # 마스크 2
        └── ...
```

나머지 파일(`box.txt`, `metadata.json`)은 아래 스크립트로 자동 생성!

---

## 완전 자동화 워크플로우

### Step 1: 이미지만 준비

```bash
# 당신이 가진 것:
~/my_rabbit_images/
├── IMG_001.jpg
├── IMG_002.jpg
├── IMG_003.jpg
└── ...
```

- **어떤 포맷이든 OK**: jpg, png, jpeg, bmp, etc.
- **어떤 크기든 OK**: 자동 리사이징
- **어떤 이름이든 OK**: 자동으로 재명명

### Step 2: 자동 마스크 생성 (선택 1개)

#### Option 1: SAM (Segment Anything Model) - 최고 품질 ⭐

```python
# install_sam.sh
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

```python
# generate_masks_sam.py
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

#### Option 2: Background Subtraction - 간단 (정적 카메라)

```python
# generate_masks_bg_subtraction.py
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

#### Option 3: Manual Annotation - 최고 정확도 (소량 데이터)

- **CVAT**: https://www.cvat.ai/ (무료, 웹 기반)
- **LabelMe**: https://github.com/wkentaro/labelme (무료, 로컬)
- **SimpleClick**: 논문에서 사용한 interactive segmentation tool

### Step 3: Fauna 포맷으로 변환 (완전 자동)

```python
# convert_to_fauna_format.py
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

    print(f"Found {len(image_files)} images in {input_dir}")

    for idx, img_path in enumerate(tqdm(image_files, desc="Converting")):
        frame_id = f"{idx:07d}"

        # ========================================
        # 1. Load and save RGB image
        # ========================================
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to load {img_path}, skipping...")
            continue

        original_h, original_w = img.shape[:2]

        # Save RGB
        rgb_path = seq_dir / f"{frame_id}_rgb.png"
        cv2.imwrite(str(rgb_path), img)

        # ========================================
        # 2. Load or generate mask
        # ========================================
        mask_path_check = img_path.with_name(img_path.stem + "_mask.png")

        if mask_path_check.exists():
            # Use existing mask
            mask = cv2.imread(str(mask_path_check), cv2.IMREAD_GRAYSCALE)
        elif auto_generate_masks:
            # Auto-generate mask
            if mask_method == "threshold":
                mask = auto_mask_threshold(img)
            elif mask_method == "grabcut":
                mask = auto_mask_grabcut(img)
            elif mask_method == "sam":
                mask = auto_mask_sam(img)
            else:
                raise ValueError(f"Unknown mask_method: {mask_method}")
        else:
            raise FileNotFoundError(
                f"Mask not found for {img_path}. "
                f"Either provide {mask_path_check} or set auto_generate_masks=True"
            )

        # Ensure binary mask
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Save mask
        mask_out_path = seq_dir / f"{frame_id}_mask.png"
        cv2.imwrite(str(mask_out_path), mask_binary)

        # ========================================
        # 3. Compute bounding box from mask
        # ========================================
        coords = np.where(mask_binary > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Add small padding (5%)
            pad_y = int((y_max - y_min) * 0.05)
            pad_x = int((x_max - x_min) * 0.05)
            y_min = max(0, y_min - pad_y)
            y_max = min(original_h, y_max + pad_y)
            x_min = max(0, x_min - pad_x)
            x_max = min(original_w, x_max + pad_x)
        else:
            # No foreground pixels, use full image
            x_min, y_min = 0, 0
            x_max, y_max = original_w, original_h

        bbox = [x_min, y_min, x_max, y_max]

        # Save bounding box
        box_path = seq_dir / f"{frame_id}_box.txt"
        np.savetxt(box_path, bbox, fmt='%d')

        # ========================================
        # 4. Create metadata.json
        # ========================================
        metadata = {
            "video_frame_id": idx,
            "video_frame_width": original_w,
            "video_frame_height": original_h,
            "crop_box_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "source_image": str(img_path.name),
            "animal_category": animal_name
        }

        metadata_path = seq_dir / f"{frame_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\n✅ Conversion complete!")
    print(f"Output directory: {seq_dir}")
    print(f"Total frames: {len(image_files)}")
    print(f"\nDataset structure:")
    print(f"  - RGB images: {frame_id}_rgb.png")
    print(f"  - Masks: {frame_id}_mask.png")
    print(f"  - Bounding boxes: {frame_id}_box.txt")
    print(f"  - Metadata: {frame_id}_metadata.json")


def auto_mask_threshold(img, threshold=20):
    """Simple threshold-based mask generation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def auto_mask_grabcut(img):
    """GrabCut-based mask generation"""
    mask = np.zeros(img.shape[:2], np.uint8)

    # Initialize rectangle (assume object in center 80%)
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.1), int(w * 0.1)
    rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Convert to binary mask
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    return mask_binary


def auto_mask_sam(img):
    """SAM-based mask generation (requires SAM installed)"""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "SAM not installed. Install with: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    # Load SAM model (assume checkpoint in current dir)
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate masks
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    # Get largest mask
    if len(masks) > 0:
        largest_mask = max(masks, key=lambda x: x['area'])
        mask = largest_mask['segmentation'].astype(np.uint8) * 255
    else:
        # Fallback to threshold
        mask = auto_mask_threshold(img)

    return mask


# ========================================
# Usage Examples
# ========================================

if __name__ == "__main__":

    # Example 1: You have images + masks already
    # =========================================
    convert_to_fauna_format(
        input_dir="~/my_rabbit_images",  # Contains IMG_001.jpg, IMG_001_mask.png, etc.
        output_dir="data/fauna/Fauna_dataset/large_scale",
        animal_name="rabbit_custom",
        split="train",
        auto_generate_masks=False  # Masks already exist
    )

    # Example 2: Only images, auto-generate masks with threshold
    # ===========================================================
    convert_to_fauna_format(
        input_dir="~/my_cat_images",  # Only IMG_001.jpg, IMG_002.jpg, etc.
        output_dir="data/fauna/Fauna_dataset/large_scale",
        animal_name="cat_custom",
        split="train",
        auto_generate_masks=True,
        mask_method="threshold"  # Simple threshold
    )

    # Example 3: Only images, auto-generate masks with GrabCut
    # =========================================================
    convert_to_fauna_format(
        input_dir="~/my_dog_images",
        output_dir="data/fauna/Fauna_dataset/large_scale",
        animal_name="dog_custom",
        split="train",
        auto_generate_masks=True,
        mask_method="grabcut"  # Better quality
    )

    # Example 4: Only images, auto-generate masks with SAM (best quality)
    # ====================================================================
    convert_to_fauna_format(
        input_dir="~/my_horse_images",
        output_dir="data/fauna/Fauna_dataset/large_scale",
        animal_name="horse_custom",
        split="train",
        auto_generate_masks=True,
        mask_method="sam"  # Requires SAM installed
    )
```

### Step 4: 데이터 검증

```python
# verify_dataset.py
from pathlib import Path
import cv2
import json

def verify_fauna_dataset(data_dir):
    """Fauna 데이터셋 검증"""
    data_path = Path(data_dir)

    # Find all RGB images
    rgb_files = sorted(data_path.glob("*_rgb.png"))

    print(f"Checking {len(rgb_files)} frames in {data_dir}...")

    issues = []
    for rgb_file in rgb_files:
        frame_id = rgb_file.stem.replace("_rgb", "")

        # Check required files
        mask_file = data_path / f"{frame_id}_mask.png"
        box_file = data_path / f"{frame_id}_box.txt"
        meta_file = data_path / f"{frame_id}_metadata.json"

        if not mask_file.exists():
            issues.append(f"Missing mask: {mask_file}")
        if not box_file.exists():
            issues.append(f"Missing box: {box_file}")
        if not meta_file.exists():
            issues.append(f"Missing metadata: {meta_file}")

        # Check image can be loaded
        img = cv2.imread(str(rgb_file))
        if img is None:
            issues.append(f"Cannot load RGB: {rgb_file}")

        # Check mask can be loaded
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues.append(f"Cannot load mask: {mask_file}")
            else:
                # Check mask has foreground pixels
                if mask.max() == 0:
                    issues.append(f"Empty mask: {mask_file}")

    if len(issues) == 0:
        print("✅ All checks passed!")
        print(f"Dataset ready for training with {len(rgb_files)} frames")
    else:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    return len(issues) == 0

# Usage
verify_fauna_dataset("data/fauna/Fauna_dataset/large_scale/rabbit_custom/train/seq_000")
```

---

## 데이터셋 구조 상세

### 최소 구조 (필수만)

```
data/fauna/Fauna_dataset/large_scale/my_animal/
└── train/
    └── seq_000/
        ├── 0000000_rgb.png        # 이미지 (아무 크기)
        ├── 0000000_mask.png       # 마스크 (binary: 0=배경, 255=동물)
        ├── 0000000_box.txt        # 바운딩 박스 [x_min, y_min, x_max, y_max]
        ├── 0000000_metadata.json  # 메타데이터
        ├── 0000001_rgb.png
        ├── 0000001_mask.png
        ├── 0000001_box.txt
        ├── 0000001_metadata.json
        └── ...
```

### 완전 구조 (선택 포함)

```
data/fauna/Fauna_dataset/large_scale/my_animal/
├── train/
│   ├── seq_000/
│   │   ├── 0000000_rgb.png
│   │   ├── 0000000_mask.png
│   │   ├── 0000000_box.txt
│   │   ├── 0000000_metadata.json
│   │   ├── 0000000_keypoint.txt      # 선택: 2D 키포인트
│   │   ├── 0000000_feat16.png        # 선택: DINO features (사전 추출)
│   │   └── ...
│   └── seq_001/  (여러 시퀀스 가능)
├── val/   (선택: 검증 세트)
└── test/  (선택: 테스트 세트)
```

### 파일 포맷 상세

#### 1. `{frame_id}_rgb.png` (필수)

- **포맷**: PNG, JPG, JPEG, BMP 등
- **크기**: 아무거나 (자동 리사이징됨)
- **권장**: 512×512 이상 (품질 위해)
- **예시**: 1920×1080, 640×480, 256×256 모두 OK

#### 2. `{frame_id}_mask.png` (필수)

- **포맷**: PNG (grayscale)
- **값**: 0 (배경) / 255 (동물)
- **크기**: RGB와 동일 (자동으로 맞춰짐)
- **생성 방법**: SAM, GrabCut, Manual annotation

예시:
```python
# Binary mask
mask = np.zeros((height, width), dtype=np.uint8)
mask[animal_region] = 255  # Foreground
cv2.imwrite("0000000_mask.png", mask)
```

#### 3. `{frame_id}_box.txt` (필수, 자동 생성 가능)

- **포맷**: Text file, 4개 숫자
- **내용**: `[x_min, y_min, x_max, y_max]`
- **생성**: Mask에서 자동 계산 가능

예시:
```python
# From mask
coords = np.where(mask > 0)
y_min, y_max = coords[0].min(), coords[0].max()
x_min, x_max = coords[1].min(), coords[1].max()
bbox = [x_min, y_min, x_max, y_max]
np.savetxt("0000000_box.txt", bbox, fmt='%d')
```

#### 4. `{frame_id}_metadata.json` (필수, 자동 생성 가능)

- **포맷**: JSON
- **내용**: Frame ID, 원본 크기, crop box

예시:
```json
{
  "video_frame_id": 0,
  "video_frame_width": 1920,
  "video_frame_height": 1080,
  "crop_box_xyxy": [100, 200, 800, 900],
  "source_image": "IMG_001.jpg",
  "animal_category": "rabbit"
}
```

#### 5. `{frame_id}_keypoint.txt` (선택)

- **포맷**: Text file
- **내용**: [x1, y1, vis1, x2, y2, vis2, ...]
- **용도**: 평가 시 keypoint error 계산

---

## 실전 예제

### 예제 1: 인터넷에서 다운받은 고양이 사진 100장

```bash
# 1. 준비된 것
~/cat_images/
├── cat_001.jpg
├── cat_002.jpg
└── ... (100장)

# 2. SAM으로 마스크 생성
python generate_masks_sam.py --input_dir ~/cat_images

# 3. Fauna 포맷으로 변환
python convert_to_fauna_format.py \
  --input_dir ~/cat_images \
  --output_dir data/fauna/Fauna_dataset/large_scale \
  --animal_name cat_internet \
  --auto_generate_masks False  # 이미 생성됨

# 4. 검증
python verify_dataset.py \
  --data_dir data/fauna/Fauna_dataset/large_scale/cat_internet/train/seq_000

# 5. Config 생성
cp config/dataset/fauna_new_animal_template.yaml config/dataset/fauna_cat_internet.yaml
cp config/model/fauna_new_animal_template.yaml config/model/fauna_cat_internet.yaml
cp config/train_fauna_new_animal_template.yaml config/train_fauna_cat_internet.yaml

# 6. 학습 (Debug 먼저!)
python run.py --config-name train_fauna_cat_internet_debug
```

### 예제 2: 비디오에서 프레임 추출

```python
# extract_frames_from_video.py
import cv2
from pathlib import Path

def extract_frames(video_path, output_dir, fps=2):
    """
    비디오에서 프레임 추출

    Args:
        video_path: 비디오 파일 경로
        output_dir: 출력 디렉토리
        fps: 초당 프레임 수 (2 = 0.5초마다 1프레임)
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_file = output_path / f"frame_{saved_count:07d}.jpg"
            cv2.imwrite(str(output_file), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

# Usage
extract_frames("~/rabbit_video.mp4", "~/rabbit_frames", fps=2)

# Then generate masks and convert
# python generate_masks_sam.py --input_dir ~/rabbit_frames
# python convert_to_fauna_format.py --input_dir ~/rabbit_frames ...
```

### 예제 3: 여러 동물을 하나의 데이터셋으로

```bash
# Fauna는 자동으로 여러 동물을 합쳐서 학습 가능
data/fauna/Fauna_dataset/large_scale/
├── cat_custom/
│   └── train/
│       └── seq_000/
├── dog_custom/
│   └── train/
│       └── seq_000/
└── rabbit_custom/
    └── train/
        └── seq_000/

# Config에서 data_dir만 지정하면 자동으로 모두 로딩됨
# dataset.train_data_dir: data/fauna/Fauna_dataset/large_scale
```

---

## 마스크 품질 체크리스트

좋은 마스크의 조건:

- ✅ **완전한 커버리지**: 동물 전체가 포함됨 (꼬리, 발끝까지)
- ✅ **깨끗한 경계**: 배경과 명확히 구분됨
- ✅ **노이즈 없음**: 작은 점들이나 구멍이 없음
- ✅ **일관성**: 프레임 간 급격한 변화 없음
- ❌ **부분 마스크**: 동물 일부만 포함 (머리만 등)
- ❌ **배경 포함**: 동물 주변 배경이 포함됨
- ❌ **구멍 많음**: 마스크 내부에 구멍이 많음

**품질 향상 팁**:
```python
# Morphological operations으로 마스크 개선
import cv2

# Remove small holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Remove small noise
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

# Smooth edges
mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)
_, mask_clean = cv2.threshold(mask_clean, 127, 255, cv2.THRESH_BINARY)
```

---

## 데이터 품질 vs 학습 성공

| 데이터 품질 | 이미지 수 | 마스크 품질 | 예상 결과 | 학습 시간 |
|-------------|-----------|-------------|-----------|-----------|
| **Excellent** | 200+ | SAM/Manual | High quality | 200K iters (10-12h) |
| **Good** | 100-200 | GrabCut | Good quality | 150K iters (7-8h) |
| **Fair** | 50-100 | Threshold | Acceptable | 100K iters (5-6h) |
| **Poor** | < 50 | Noisy | May fail | N/A |

**권장**:
- 이미지 수 ≥ 100장
- 다양한 포즈 (앉기, 서기, 걷기 등)
- 다양한 뷰포인트 (정면, 측면, 뒤 등)
- 깨끗한 마스크 (SAM 또는 Manual)

---

## FAQ

### Q: RGB 이미지만 있고 마스크가 없는데, 자동으로 생성할 수 있나요?

**A**: 네! 위의 `convert_to_fauna_format.py` 스크립트에서 `auto_generate_masks=True`로 설정하세요. SAM, GrabCut, Threshold 중 선택 가능합니다.

### Q: 마스크 품질이 안 좋으면 어떻게 되나요?

**A**: 학습이 실패하거나 품질이 낮은 3D 모델이 생성됩니다. **마스크가 가장 중요한 supervision signal**이므로, 가능하면 SAM이나 수동 annotation을 권장합니다.

### Q: 이미지가 10장밖에 없는데 학습 가능한가요?

**A**: 가능하지만 품질이 낮을 수 있습니다. Few-shot 설정으로 학습하되, 다양한 viewpoint와 pose를 가진 이미지를 사용하세요. 최소 30-50장을 권장합니다.

### Q: 비디오에서 프레임을 몇 장 추출해야 하나요?

**A**:
- Few-shot: 50-100 프레임 (2-3초 간격)
- Medium: 100-200 프레임 (1-2초 간격)
- Large-scale: 200-500 프레임 (0.5-1초 간격)

중복된 프레임은 피하고, 다양한 동작/각도를 캡처하세요.

### Q: 여러 동물이 한 이미지에 있으면?

**A**: 각 동물을 개별적으로 crop하고 별도의 시퀀스로 저장하세요. Fauna는 한 이미지에 하나의 동물을 가정합니다.

### Q: Keypoint는 꼭 필요한가요?

**A**: 아니요. 선택사항입니다. 평가 시 keypoint error를 측정하려면 필요하지만, 학습 자체에는 필수가 아닙니다.

---

## 요약

**최소 요구사항**:
```
✅ {frame_id}_rgb.png    (이미지, 아무 크기)
✅ {frame_id}_mask.png   (마스크, binary)
```

**나머지는 자동 생성 가능**:
```
🤖 {frame_id}_box.txt       (마스크에서 계산)
🤖 {frame_id}_metadata.json (기본값 생성)
```

**권장 워크플로우**:
1. 이미지 수집 (100+ 장)
2. SAM으로 마스크 생성 (또는 수동 annotation)
3. `convert_to_fauna_format.py` 실행
4. `verify_dataset.py`로 검증
5. Config 파일 생성 (템플릿 복사)
6. Debug 모드 학습 (15-30분)
7. Full 학습 (10-12시간)

**이제 어떤 이미지든 3DAnimals로 학습할 수 있습니다!** 🎉
