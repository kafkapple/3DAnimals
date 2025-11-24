# Mouse Dataset → 3D-Fauna 통합 분석 보고서

**작성일**: 2025-11-10
**목적**: Markerless Mouse 데이터셋을 3D-Fauna 학습에 활용하기 위한 요구사항 및 변환 방법 분석

---

## 1. 현재 train_fauna 데이터셋 스펙 분석

### 1.1 데이터 경로 구조

**Config 설정** (`config/train_fauna.yaml:17-19`):
```yaml
dataset:
  train_data_dir: data/fauna/Fauna_dataset
  val_data_dir: data/fauna/Fauna_dataset
  test_data_dir: data/fauna/Fauna_dataset
```

**실제 디렉토리 구조**:
```
data/fauna/Fauna_dataset/
├── large_scale/              # 대규모 카테고리 데이터
│   ├── bear_comb_dinov2_new/
│   │   ├── train/
│   │   └── test/
│   ├── cow_comb_dinov2_new/
│   ├── elephant_comb_dinov2_new/
│   ├── giraffe_comb_dinov2_new/
│   ├── horse_comb_dinov2_new/
│   ├── sheep_comb_dinov2_new/
│   └── zebra_comb_dinov2_new/
├── few_shot_animal3d/        # 소규모 카테고리 (Animal3D)
│   ├── horse/train/
│   ├── husky_dog/train/
│   ├── ibex/train/
│   ├── impala/train/
│   ├── leopard/train/
│   ├── newfoundland_wolf/train/
│   ├── polar_bear/train/
│   └── red_gazelle/train/
├── few_shot_web/             # 소규모 카테고리 (Web)
└── few_shot_web_back/        # 배경 뷰 (선택적)
```

### 1.2 개별 샘플 파일 구조

**필수 파일들** (예: `horse_000005`):
```
train/
├── horse_000005_rgb.png         # RGB 이미지 (256x256)
├── horse_000005_mask.png        # Binary mask (256x256)
├── horse_000005_feat16.png      # DINO features (16 channels)
├── horse_000005_box.txt         # Bounding box 정보
├── horse_000005_metadata.json   # 메타데이터
├── horse_000005_clusters.png    # DINO clustering (선택적)
└── horse_000005_combine.png     # 시각화용 (선택적)
```

### 1.3 파일 형식 상세

#### **1. RGB 이미지** (`*_rgb.png`)
```
Format: PNG
Size: 256x256 pixels
Channels: 3 (RGB)
Range: [0, 255]
```

#### **2. Mask** (`*_mask.png`)
```
Format: PNG (grayscale)
Size: 256x256 pixels
Values: 0 (배경), 255 (전경)
```

#### **3. DINO Features** (`*_feat16.png`)
```
Format: PNG (16-channel)
Size: 256x256 pixels
Channels: 16 (PCA-reduced DINO features)
Encoding: Float32 → Uint8 (quantized)
```

#### **4. Bounding Box** (`*_box.txt`)
```
Format: Space-separated values
Fields: frame_id x0 y0 width height full_w full_h sharpness
Example: 5 213.00 153.00 214.00 214.00 640.00 480.00 0.00

Explanation:
- frame_id: 원본 비디오 프레임 번호
- x0, y0: Crop 시작 좌표 (원본 해상도 기준)
- width, height: Crop 크기
- full_w, full_h: 원본 프레임 해상도
- sharpness: 프레임 선명도 (0.0 = 미사용)
```

#### **5. Metadata** (`*_metadata.json`)
```json
{
    "video_frame_id": 5,
    "crop_box_xyxy": [213, 153, 427, 367],
    "video_frame_width": 640,
    "video_frame_height": 480,
    "sharpness": 0.0,
    "crop_height": 256,
    "crop_width": 256
}

Explanation:
- video_frame_id: 프레임 ID
- crop_box_xyxy: [x_min, y_min, x_max, y_max] 형식
- video_frame_*: 원본 비디오 해상도
- crop_*: 크롭된 이미지 크기
```

### 1.4 Dataset Loader 동작 방식

**FaunaDataset 클래스** (`model/dataset/FaunaDataset.py:41`):

```python
# 1. 카테고리 스캔
large_scale_paths = {
    'large_scale_{category}': 'path/to/category/train'
}
small_scale_paths = {
    'small_scale_{source}_{category}': 'path/to/category/train'
}

# 2. 이미지 파일 검색 (폴더별)
for folder in category_path:
    files = glob('*_rgb.png')  # 또는 'rgb.*' pattern
    sequences.append(files)

# 3. __getitem__ 시 로드
path = '/path/to/train/horse_000005_{}'
rgb = load(path.format('rgb.png'))
mask = load(path.format('mask.png'))
feat = load(path.format('feat16.png'))
box = load(path.format('box.txt'))
metadata = load(path.format('metadata.json'))
```

**핵심 요구사항**:
- 파일명 패턴: `{prefix}_{id}_{type}.{ext}`
- 모든 파일은 같은 prefix와 ID 공유
- `train/` 또는 `test/` 폴더 내에 위치

---

## 2. Mouse Dataset 구조 분석

### 2.1 현재 데이터 구조

```
/home/joon/dev/data/markerless_mouse/
├── mouse_1/
│   ├── Camera1/
│   │   ├── 0.mp4         (251MB, 1152x1024, 100fps, 3000 frames)
│   │   ├── 3000.mp4      (259MB, 1152x1024, 100fps, 3000 frames)
│   │   ├── 6000.mp4      (280MB)
│   │   ├── 9000.mp4      (280MB)
│   │   ├── 12000.mp4     (280MB)
│   │   └── 15000.mp4     (280MB)
│   ├── Camera2/
│   │   └── [동일 구조]
│   ├── Camera3/
│   ├── Camera4/
│   ├── Camera5/
│   └── Camera6/
└── mouse_2/
    ├── Camera1/
    │   ├── 0.mp4         (278MB, 1152x1024, 100fps, 3000 frames)
    │   ├── 3000.mp4
    │   ├── 6000.mp4
    │   ├── 9000.mp4
    │   ├── 12000.mp4
    │   └── 15000.mp4
    ├── Camera2/
    └── ... (6 cameras)
```

**데이터 통계**:
- **개체 수**: 2마리 (mouse_1, mouse_2)
- **카메라 수**: 6개 (Camera1-6, multi-view setup)
- **비디오 수**: 6개/카메라 (시간대별 분할)
- **총 비디오**: 2 mice × 6 cameras × 6 videos = **72 videos**
- **해상도**: 1152 × 1024 pixels
- **프레임레이트**: 100 FPS
- **프레임 수**: ~3000 frames/video
- **총 프레임 수**: ~216,000 frames

### 2.2 비디오 특성

**해상도**: 1152×1024 (aspect ratio ≈ 1.125:1)
**포맷**: MP4 (H.264)
**내용**: Markerless motion capture (실험용 마우스 행동)
**시점**: Multi-view (6개 고정 카메라)

---

## 3. 데이터셋 호환성 분석

### 3.1 호환 가능 여부

| 항목 | Mouse Dataset | Fauna 요구사항 | 호환성 |
|------|--------------|---------------|--------|
| **동물 종류** | Mouse (설치류) | Quadrupeds | ⚠️ **문제** |
| **해상도** | 1152×1024 | 256×256 (자동 resize) | ✅ 가능 |
| **형식** | MP4 video | PNG images | 🔧 **변환 필요** |
| **Multi-view** | 6 cameras | Single view | ✅ 선택 사용 |
| **프레임 수** | ~216K frames | 수백~수천 | ✅ 충분 |
| **Mask** | 없음 | 필수 | 🔧 **생성 필요** |
| **DINO features** | 없음 | 필수 (training) | 🔧 **추출 필요** |
| **Metadata** | 없음 | 필수 | 🔧 **생성 필요** |

### 3.2 핵심 문제점

#### ❌ **문제 1: 동물 카테고리 불일치**

**3D-Fauna 설계**:
- **Target**: Large quadrupeds (개, 고양이, 말, 사자, 곰 등)
- **특징**: 명확한 4개 다리, 중형~대형 체구
- **Prior shape**: Horse/dog-like articulated skeleton

**Mouse 특성**:
- **Category**: Small rodent (설치류)
- **특징**: 매우 작은 체구, 긴 꼬리, 다른 비율
- **Pose**: Crouched posture, 빠른 움직임

**예상 결과**:
- ⚠️ **Poor reconstruction quality**: Prior shape mismatch
- ⚠️ **Incorrect articulation**: Bone structure 차이
- ⚠️ **Texture artifacts**: 훈련 데이터와 분포 차이

**해결 방안**:
1. **Fine-tuning**: Pretrained Fauna model을 mouse 데이터로 fine-tune
2. **Category-specific training**: Mouse 전용 모델 학습 (많은 데이터 필요)
3. **Alternative models**: MagicPony (category-specific) 사용 고려

---

## 4. 데이터 변환 요구사항

### 4.1 필수 전처리 단계

#### **Step 1: 비디오 → 프레임 추출**

```bash
# 각 비디오에서 균등하게 프레임 샘플링
for video in mouse_1/Camera1/*.mp4; do
    ffmpeg -i $video -vf "select='not(mod(n\,30))'" -vsync vfr \
           output/frames/%05d.png
done

# 설명:
# - 30프레임마다 1개 추출 (100fps → ~3fps)
# - 3000 frames/video → ~100 frames/video
# - 총 72 videos × 100 = ~7200 frames
```

#### **Step 2: 마우스 Segmentation (Mask 생성)**

**Option A - Automatic segmentation**:
```python
# SAM (Segment Anything Model) 사용
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

for frame in frames:
    predictor.set_image(frame)
    # Auto-detect animal (largest connected component)
    mask = detect_largest_object(frame)
    save_mask(mask, f"{frame_id}_mask.png")
```

**Option B - Background subtraction** (if static background):
```python
import cv2

# 배경 모델 학습
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

for frame in frames:
    fg_mask = bg_subtractor.apply(frame)
    # Morphological operations to clean up
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    save_mask(mask, f"{frame_id}_mask.png")
```

**Option C - Manual annotation** (최고 품질):
- CVAT, LabelMe 등 annotation tool 사용
- 시간 소요: ~100-200 frames/hour

#### **Step 3: Bounding Box 계산**

```python
import numpy as np
import cv2

def compute_bbox_from_mask(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Get largest contour (mouse body)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add padding (10%)
    pad = 0.1
    x -= int(w * pad)
    y -= int(h * pad)
    w += int(w * pad * 2)
    h += int(h * pad * 2)

    # Clamp to image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(image.shape[1] - x, w)
    h = min(image.shape[0] - y, h)

    return {
        "x0": x, "y0": y, "width": w, "height": h,
        "full_w": mask.shape[1], "full_h": mask.shape[0]
    }
```

#### **Step 4: Crop & Resize**

```python
def crop_and_resize(image, mask, bbox, target_size=256):
    x0, y0, w, h = bbox["x0"], bbox["y0"], bbox["width"], bbox["height"]

    # Crop
    cropped_image = image[y0:y0+h, x0:x0+w]
    cropped_mask = mask[y0:y0+h, x0:x0+w]

    # Make square (pad shorter dimension)
    if w != h:
        max_dim = max(w, h)
        square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        square_mask = np.zeros((max_dim, max_dim), dtype=np.uint8)

        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2

        square_image[pad_y:pad_y+h, pad_x:pad_x+w] = cropped_image
        square_mask[pad_y:pad_y+h, pad_x:pad_x+w] = cropped_mask

        cropped_image = square_image
        cropped_mask = square_mask

    # Resize to target size
    resized_image = cv2.resize(cropped_image, (target_size, target_size),
                               interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (target_size, target_size),
                              interpolation=cv2.INTER_NEAREST)

    return resized_image, resized_mask
```

#### **Step 5: DINO Feature 추출**

```python
import torch
from torchvision import transforms
from dinov2 import DINOv2

# DINOv2 model 로드
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def extract_dino_features(image, output_dim=16):
    # Forward pass
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0)
        features = dino_model.forward_features(img_tensor)

        # Get dense features (patch tokens)
        patch_features = features['x_norm_patchtokens']  # [1, N_patches, D]

        # Reshape to spatial grid
        h = w = int(np.sqrt(patch_features.shape[1]))
        spatial_features = patch_features.reshape(1, h, w, -1)

        # PCA to reduce to 16 dimensions
        flat_features = spatial_features.reshape(-1, spatial_features.shape[-1])
        pca = PCA(n_components=output_dim)
        reduced_features = pca.fit_transform(flat_features.cpu().numpy())

        # Reshape back to spatial
        reduced_spatial = reduced_features.reshape(1, h, w, output_dim)

        # Upsample to image size
        upsampled = torch.nn.functional.interpolate(
            torch.from_numpy(reduced_spatial).permute(0, 3, 1, 2),
            size=(256, 256), mode='bilinear'
        )

        return upsampled.squeeze(0).permute(1, 2, 0).numpy()  # [256, 256, 16]

def save_dino_features(features, output_path):
    # Normalize to [0, 255]
    features_norm = (features - features.min()) / (features.max() - features.min())
    features_uint8 = (features_norm * 255).astype(np.uint8)

    # Save as 16-channel PNG
    cv2.imwrite(output_path, features_uint8)
```

#### **Step 6: Metadata 생성**

```python
import json

def create_metadata(frame_id, bbox, original_size):
    metadata = {
        "video_frame_id": int(frame_id),
        "crop_box_xyxy": [
            int(bbox["x0"]),
            int(bbox["y0"]),
            int(bbox["x0"] + bbox["width"]),
            int(bbox["y0"] + bbox["height"])
        ],
        "video_frame_width": int(original_size[0]),
        "video_frame_height": int(original_size[1]),
        "sharpness": 0.0,
        "crop_height": 256,
        "crop_width": 256
    }
    return metadata

def save_metadata(metadata, output_path):
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
```

#### **Step 7: Box.txt 생성**

```python
def create_box_txt(frame_id, bbox):
    box_line = f"{frame_id} {bbox['x0']:.2f} {bbox['y0']:.2f} " \
               f"{bbox['width']:.2f} {bbox['height']:.2f} " \
               f"{bbox['full_w']:.2f} {bbox['full_h']:.2f} 0.00"
    return box_line

def save_box_txt(box_line, output_path):
    with open(output_path, 'w') as f:
        f.write(box_line)
```

### 4.2 통합 파이프라인

```python
import os
import cv2
import numpy as np

def process_mouse_video(video_path, output_dir, camera_id, mouse_id,
                        frame_sample_rate=30):
    """
    마우스 비디오를 Fauna 형식으로 변환

    Args:
        video_path: 입력 비디오 경로
        output_dir: 출력 디렉토리
        camera_id: 카메라 ID (1-6)
        mouse_id: 마우스 ID (1-2)
        frame_sample_rate: 프레임 샘플링 간격 (30 = 매 30프레임마다)
    """

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")

    # 프레임 처리
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 샘플링 (매 N 프레임마다)
        if frame_count % frame_sample_rate != 0:
            frame_count += 1
            continue

        # 파일명 생성
        prefix = f"mouse{mouse_id}_cam{camera_id}_{saved_count:05d}"

        # Step 1: Segmentation (예: SAM 또는 background subtraction)
        mask = segment_mouse(frame)  # 구현 필요

        if mask is None or mask.sum() < 100:
            # 마우스 검출 실패
            frame_count += 1
            continue

        # Step 2: Bounding box 계산
        bbox = compute_bbox_from_mask(mask)

        # Step 3: Crop & Resize
        cropped_rgb, cropped_mask = crop_and_resize(frame, mask, bbox)

        # Step 4: DINO features 추출
        dino_features = extract_dino_features(cropped_rgb)

        # Step 5: 파일 저장
        cv2.imwrite(f"{output_dir}/{prefix}_rgb.png", cropped_rgb)
        cv2.imwrite(f"{output_dir}/{prefix}_mask.png", cropped_mask)
        save_dino_features(dino_features, f"{output_dir}/{prefix}_feat16.png")

        # Step 6: Metadata 생성
        metadata = create_metadata(frame_count, bbox, (width, height))
        save_metadata(metadata, f"{output_dir}/{prefix}_metadata.json")

        # Step 7: Box.txt 생성
        box_line = create_box_txt(frame_count, bbox)
        save_box_txt(box_line, f"{output_dir}/{prefix}_box.txt")

        saved_count += 1
        frame_count += 1

        if saved_count % 10 == 0:
            print(f"Processed {saved_count} frames...")

    cap.release()
    print(f"Completed: {saved_count} frames saved to {output_dir}")

    return saved_count

# 실행 예시
if __name__ == "__main__":
    # Mouse 1, Camera 1 처리
    process_mouse_video(
        video_path="/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4",
        output_dir="/home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/few_shot_animal3d/mouse_m1c1/train",
        camera_id=1,
        mouse_id=1,
        frame_sample_rate=30
    )
```

### 4.3 최종 디렉토리 구조

```
data/fauna/Fauna_dataset/few_shot_animal3d/
└── mouse_m1c1/                           # Mouse 1, Camera 1
    └── train/
        ├── mouse1_cam1_00000_rgb.png
        ├── mouse1_cam1_00000_mask.png
        ├── mouse1_cam1_00000_feat16.png
        ├── mouse1_cam1_00000_box.txt
        ├── mouse1_cam1_00000_metadata.json
        ├── mouse1_cam1_00001_rgb.png
        ├── ...
        └── mouse1_cam1_00100_rgb.png      # ~100 frames/video

# 6개 카메라 각각 처리하면:
mouse_m1c1/train/  # Camera 1
mouse_m1c2/train/  # Camera 2
...
mouse_m1c6/train/  # Camera 6
```

---

## 5. 카메라 선택 전략

### 5.1 Single Camera 선택 (권장)

**Option 1 - Best View Selection**:
```python
# 각 카메라별로 샘플 프레임 추출 → 마우스 가시성 평가
for camera in [1, 2, 3, 4, 5, 6]:
    visibility_score = evaluate_visibility(camera)
    # Criteria:
    # - 마우스 전신이 보이는 정도
    # - Occlusion 최소화
    # - 측면 또는 비스듬한 각도 선호

# 최고 점수 카메라 선택
best_camera = select_best_camera()
```

**Option 2 - Lateral View (측면 뷰)**:
- **이유**: 3D-Fauna는 측면 뷰에서 가장 잘 작동
- **선택 기준**: 마우스를 정확히 옆에서 본 카메라

**Option 3 - Data Augmentation**:
- 모든 6개 카메라 사용하여 데이터 증강
- 각 카메라를 별도 샘플로 취급
- 주의: 동일 프레임의 다른 시점이므로 correlation 존재

### 5.2 Multi-Camera 활용 (고급)

**Option A - Separate Categories**:
```yaml
# Config 수정
few_shot_animal3d/
├── mouse_m1c1/train/  # Mouse 1, Camera 1
├── mouse_m1c2/train/  # Mouse 1, Camera 2
...

# 학습 시 각 카메라를 별도 카테고리로 취급
```

**Option B - Multi-view Training** (Ponymation 스타일):
- Sequence dataset으로 변환
- 동일 시간의 다른 시점을 sequence로 구성
- Consistency loss 추가 필요

---

## 6. 추정 작업량 및 리소스

### 6.1 데이터 처리 시간

**프레임 추출**: ~5분/비디오 (CPU)
```
72 videos × 5 min = 360 min = 6시간
```

**Segmentation** (자동):
- SAM: ~1초/프레임 (GPU)
- ~7200 frames × 1초 = 2시간 (A100)

**DINO Feature 추출**: ~0.5초/프레임 (GPU)
```
7200 frames × 0.5초 = 1시간
```

**총 자동 처리 시간**: ~9-10시간 (GPU 1개)

**Manual annotation** (선택):
- ~100-200 frames/hour
- 7200 frames → 36-72시간

### 6.2 저장 공간

**프레임 저장**:
- RGB: 256×256×3 = ~200KB/frame
- Mask: 256×256 = ~20KB/frame
- DINO: 256×256×16 = ~1MB/frame
- **Total**: ~1.2MB/frame

```
7200 frames × 1.2MB = 8.6GB
```

### 6.3 학습 리소스

**Fine-tuning** (Pretrained Fauna → Mouse):
- **GPU**: RTX 3090 (24GB) 이상
- **시간**: ~12-24시간
- **Iterations**: 50K-100K

**From Scratch** (Mouse 전용 모델):
- **GPU**: A100 (40GB) × 4 권장
- **시간**: ~3-7일
- **Iterations**: 500K-1M
- **Data**: 최소 5K-10K frames 권장 (현재 7.2K 충분)

---

## 7. 실현 가능성 평가

### 7.1 기술적 실현 가능성

| 항목 | 난이도 | 소요 시간 | 비고 |
|------|--------|-----------|------|
| 프레임 추출 | ⭐ Easy | 6시간 | 자동화 가능 |
| Segmentation | ⭐⭐ Medium | 2-72시간 | 자동(빠름) vs 수동(정확) |
| DINO 추출 | ⭐ Easy | 1시간 | 자동화 가능 |
| Metadata 생성 | ⭐ Easy | 1시간 | 스크립트 작성 |
| Fine-tuning | ⭐⭐⭐ Hard | 1-2일 | GPU 필요 |
| 품질 검증 | ⭐⭐ Medium | 1일 | 시각적 검사 |

**총 예상 시간**: 3-5일 (자동 segmentation), 10-15일 (수동 annotation)

### 7.2 성능 예측

#### **시나리오 1: Fine-tuning Pretrained Fauna**

**장점**:
- ✅ 빠른 수렴 (12-24시간)
- ✅ 적은 데이터로 가능
- ✅ 안정적인 학습

**단점**:
- ⚠️ Prior shape mismatch (horse/dog → mouse)
- ⚠️ Suboptimal reconstruction quality
- ⚠️ 크기/비율 차이로 인한 artifacts

**예상 품질**: ⭐⭐⭐☆☆ (Fair)

#### **시나리오 2: Train from Scratch (Mouse-specific)**

**장점**:
- ✅ Mouse에 최적화된 prior shape
- ✅ 최고 reconstruction quality
- ✅ Accurate articulation

**단점**:
- ⚠️ 긴 학습 시간 (3-7일)
- ⚠️ 많은 GPU 리소스
- ⚠️ Hyperparameter tuning 필요

**예상 품질**: ⭐⭐⭐⭐⭐ (Excellent)

**데이터 충분성**: ✅ 7.2K frames는 category-specific 모델에 충분

#### **시나리오 3: Alternative - MagicPony**

**고려사항**:
- MagicPony는 category-specific 모델
- Mouse 전용 모델 학습에 더 적합
- Fauna보다 간단한 아키텍처

**장점**:
- ✅ Category-specific design
- ✅ 빠른 수렴
- ✅ 안정적인 학습

**단점**:
- ⚠️ Pan-category features 없음 (Fauna 대비)

---

## 8. 권장 실행 계획

### 8.1 단계별 실행 플랜

#### **Phase 1: 데이터 준비 (1주일)**

**Week 1, Day 1-2: 프레임 추출**
```bash
# 1. 비디오 프레임 추출 스크립트 작성
# 2. Mouse 1, Camera 선택 (Best View)
# 3. 샘플링 (30 frames interval)
# 4. 검증: ~100 frames/video × 6 videos = 600 frames
```

**Week 1, Day 3-4: Segmentation**
```python
# Option A: SAM auto-segmentation (추천)
# - 빠름, 합리적 품질
# - 후처리로 품질 개선

# Option B: Manual annotation (최고 품질)
# - 100-200 frames/hour
# - 팀 작업 가능
```

**Week 1, Day 5: DINO & Metadata**
```python
# 1. DINO features 추출 (1시간)
# 2. Metadata/Box 생성 (1시간)
# 3. 파일 구조 검증
```

**Week 1, Day 6-7: 데이터 검증**
```python
# 1. FaunaDataset으로 로딩 테스트
# 2. 샘플 시각화
# 3. 문제 수정
```

#### **Phase 2: 모델 학습 (1-2주)**

**Option A - Quick Test (Fine-tuning)**:
```bash
# Week 2, Day 1-2: Fine-tuning setup
python run.py --config-name finetune_fauna_mouse

# Config:
dataset:
  train_data_dir: data/fauna/Fauna_dataset
  # Add mouse category

model:
  checkpoint: results/fauna/pretrained_fauna/pretrained_fauna.pth

training:
  num_iters: 50000
  lr: 0.00005  # Lower LR for fine-tuning
```

**Option B - Full Training (Optimal)**:
```bash
# Week 2-3: From scratch
python run.py --config-name train_magicpony_mouse

# Config:
dataset:
  train_data_dir: data/mouse/train

model:
  # MagicPony architecture for category-specific

training:
  num_iters: 200000
```

#### **Phase 3: 평가 및 반복 (3-5일)**

```bash
# Test inference
python run.py --config-name test_mouse

# Evaluate:
# - Reconstruction quality
# - Pose estimation accuracy
# - Articulation plausibility

# Iterate:
# - Adjust hyperparameters
# - Add more data if needed
# - Refine segmentation
```

### 8.2 체크리스트

#### **데이터 준비**
- [ ] 비디오 → 프레임 추출 완료 (~600-1000 frames)
- [ ] 카메라 1개 선택 (Best View 또는 Lateral)
- [ ] Segmentation 완료 (SAM 또는 manual)
- [ ] DINO features 추출 완료
- [ ] Metadata & Box.txt 생성 완료
- [ ] FaunaDataset 로딩 테스트 성공
- [ ] 디렉토리 구조 검증 완료

#### **학습 준비**
- [ ] Config 파일 작성 (train_fauna_mouse.yaml)
- [ ] GPU 리소스 확보 (RTX 3090 이상)
- [ ] Pretrained model 다운로드 (fine-tuning 시)
- [ ] WandB/TensorBoard 설정

#### **학습 실행**
- [ ] 첫 100 iterations 모니터링 (loss 감소 확인)
- [ ] 1K iterations: 시각화 확인
- [ ] 5K iterations: Checkpoint 저장
- [ ] 10K iterations: Intermediate evaluation
- [ ] 50K iterations: Fine-tuning 완료 또는 계속

#### **평가**
- [ ] Test set inference 실행
- [ ] 3D mesh 품질 확인 (Blender/MeshLab)
- [ ] Pose estimation 정확도 평가
- [ ] Articulation 자연스러움 검증
- [ ] Multi-view consistency (여러 카메라 비교)

---

## 9. 리스크 및 대응 방안

### 9.1 데이터 품질 리스크

**Risk 1: Poor Segmentation**
- **Impact**: 학습 품질 저하
- **Mitigation**:
  - SAM + 후처리 (morphological ops)
  - 수동 검증 (샘플링)
  - Active learning (실패 케이스 수동 수정)

**Risk 2: Motion Blur**
- **Impact**: 흐릿한 프레임
- **Mitigation**:
  - Sharpness 계산 후 필터링
  - 높은 샘플링 레이트 → 선명한 프레임 선택

**Risk 3: Occlusion**
- **Impact**: 마우스 일부 가려짐
- **Mitigation**:
  - Visibility 점수 계산
  - 전신이 보이는 프레임만 선택

### 9.2 모델 학습 리스크

**Risk 1: Prior Shape Mismatch (Fine-tuning)**
- **Impact**: Poor reconstruction
- **Mitigation**:
  - MagicPony (category-specific) 사용
  - Prior shape network도 fine-tune

**Risk 2: Overfitting (Small Data)**
- **Impact**: 일반화 실패
- **Mitigation**:
  - Data augmentation (flip, color jitter)
  - Strong regularization
  - Early stopping

**Risk 3: Scale Mismatch**
- **Impact**: Mouse 크기가 학습 데이터와 다름
- **Mitigation**:
  - Normalization by bounding box
  - Multi-scale training

### 9.3 리소스 리스크

**Risk 1: GPU 부족**
- **Impact**: 학습 불가 또는 매우 느림
- **Mitigation**:
  - Cloud GPU 사용 (GCP, AWS)
  - Batch size 줄이기
  - Mixed precision training

**Risk 2: 저장 공간 부족**
- **Impact**: 데이터 생성 실패
- **Mitigation**:
  - 압축 (DINO features)
  - 불필요한 파일 제거 (combine.png 등)

---

## 10. 결론 및 권장사항

### 10.1 실현 가능성 종합 평가

**✅ 기술적 실현 가능**: **가능** (난이도: Medium-High)
- 필요한 모든 도구 및 기술 존재
- 데이터 변환 파이프라인 구축 가능
- 충분한 프레임 수 (7.2K frames)

**⚠️ 품질 예측**: **보통~우수** (접근법에 따라 다름)
- Fine-tuning: 보통 품질 (빠름)
- From Scratch: 우수 품질 (시간 소요)

**💰 비용 예측**:
- 인력: 1명 × 2-3주 (데이터 준비 + 학습)
- GPU: RTX 3090 × 3-7일 또는 A100 × 2-3일
- 저장: ~10GB

### 10.2 최종 권장사항

#### **권장 접근법**: **MagicPony Category-Specific Training**

**이유**:
1. ✅ Mouse는 기존 Fauna 카테고리와 매우 다름 (설치류 vs 대형 quadrupeds)
2. ✅ Category-specific 모델이 더 적합
3. ✅ 충분한 데이터 (7.2K frames)
4. ✅ 합리적인 학습 시간 (3-5일)

#### **구체적 실행 계획**:

**Step 1: Pilot Study (1주)**
```
- Mouse 1, Camera 1 선택
- 1개 비디오만 처리 (~100 frames)
- SAM으로 자동 segmentation
- FaunaDataset 로딩 테스트
- 간단한 fine-tuning 실험 (10K iter)
```

**Step 2: Full Data Preparation (1주)**
```
- 나머지 5개 비디오 처리
- Segmentation 품질 검증 및 수정
- DINO features 추출
- Train/Val split (90% / 10%)
```

**Step 3: Model Training (1-2주)**
```
- MagicPony architecture 사용
- From scratch 학습
- 200K iterations
- Multi-GPU if available
```

**Step 4: Evaluation & Iteration (3-5일)**
```
- Test set 평가
- 품질 분석
- 필요시 hyperparameter tuning
- 최종 모델 선정
```

### 10.3 추가 고려사항

#### **Multi-Camera 활용 (향후)**
- 현재: 1개 카메라로 시작
- 성공 시: 다른 카메라 데이터 추가 (6배 증가)
- Multi-view consistency loss 추가

#### **Mouse 2 데이터**
- Mouse 1로 먼저 실험
- 성공 시 Mouse 2 추가 (2배 증가)
- Total: 2 mice × 6 cameras = 12× data

#### **Temporal Modeling (4D Reconstruction)**
- 현재: Single frame reconstruction
- 향후: Video sequence → 4D (Ponymation 스타일)
- Multi-view + temporal consistency

---

## 11. 필요 피드백 및 결정사항

### 11.1 즉시 결정 필요

1. **접근 방식 선택**:
   - [ ] **Option A**: Fine-tuning Pretrained Fauna (빠름, 낮은 품질)
   - [ ] **Option B**: MagicPony From Scratch (느림, 높은 품질) ← **권장**
   - [ ] **Option C**: Fauna From Scratch (매우 느림, 최고 품질)

2. **카메라 선택**:
   - [ ] 1개 카메라만 (빠른 시작)
   - [ ] 모든 6개 카메라 (더 많은 데이터)

3. **Segmentation 방법**:
   - [ ] SAM 자동 (빠름) ← **권장**
   - [ ] Manual annotation (정확, 느림)
   - [ ] Background subtraction (중간)

4. **데이터 규모**:
   - [ ] Pilot: 1 video (~100 frames) ← **첫 단계 권장**
   - [ ] Small: 6 videos (~600 frames)
   - [ ] Full: All 72 videos (~7200 frames)

### 11.2 리소스 확인 필요

1. **GPU 가용성**:
   - 사용 가능 GPU: _______________
   - 사용 가능 기간: _______________

2. **인력 배정**:
   - 데이터 준비 담당: _______________
   - 학습 모니터링: _______________
   - 평가 및 분석: _______________

3. **일정**:
   - 시작 날짜: _______________
   - 목표 완료일: _______________

### 11.3 기술적 확인 필요

1. **Segmentation Tool**:
   - SAM 모델 사용 가능 여부
   - 대안: U-Net, DeepLabv3 등

2. **DINO Model**:
   - DINOv2 접근 가능 여부
   - 사전 추출 또는 실시간 추출

3. **Training Framework**:
   - 현재 codebase 사용
   - 수정 필요 사항 파악

---

## 부록 A: 전체 변환 스크립트

```python
# convert_mouse_to_fauna.py
import os
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm

class MouseToFaunaConverter:
    def __init__(self,
                 video_dir="/home/joon/dev/data/markerless_mouse",
                 output_dir="/home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/few_shot_animal3d",
                 mouse_id=1,
                 camera_id=1,
                 frame_sample_rate=30,
                 target_size=256):

        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.mouse_id = mouse_id
        self.camera_id = camera_id
        self.frame_sample_rate = frame_sample_rate
        self.target_size = target_size

        # 출력 디렉토리 생성
        self.category_dir = self.output_dir / f"mouse_m{mouse_id}c{camera_id}" / "train"
        self.category_dir.mkdir(parents=True, exist_ok=True)

        # SAM 모델 로드 (segmentation용)
        self.load_sam_model()

        # DINO 모델 로드 (feature 추출용)
        self.load_dino_model()

    def load_sam_model(self):
        # SAM 모델 로드
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
            sam.to("cuda")
            self.sam_predictor = SamPredictor(sam)
            print("✅ SAM model loaded")
        except Exception as e:
            print(f"⚠️ SAM model not available: {e}")
            self.sam_predictor = None

    def load_dino_model(self):
        # DINOv2 모델 로드
        try:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.dino_model.eval()
            self.dino_model.to("cuda")
            print("✅ DINO model loaded")
        except Exception as e:
            print(f"⚠️ DINO model not available: {e}")
            self.dino_model = None

    def process_all_videos(self):
        # 비디오 디렉토리 찾기
        video_path = self.video_dir / f"mouse_{self.mouse_id}" / f"Camera{self.camera_id}"
        video_files = sorted(video_path.glob("*.mp4"))

        print(f"Found {len(video_files)} videos in {video_path}")

        total_frames_saved = 0
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            frames_saved = self.process_video(video_file, total_frames_saved)
            total_frames_saved += frames_saved

        print(f"\n✅ Total frames saved: {total_frames_saved}")
        return total_frames_saved

    def process_video(self, video_path, start_idx=0):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0
        saved_count = start_idx

        pbar = tqdm(total=total_frames // self.frame_sample_rate)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 샘플링
            if frame_count % self.frame_sample_rate != 0:
                frame_count += 1
                continue

            # Segmentation
            mask = self.segment_mouse(frame)

            if mask is None or mask.sum() < 100:
                frame_count += 1
                continue

            # Bounding box
            bbox = self.compute_bbox(mask, width, height)

            if bbox is None:
                frame_count += 1
                continue

            # Crop & Resize
            cropped_rgb, cropped_mask = self.crop_and_resize(frame, mask, bbox)

            # DINO features
            dino_features = self.extract_dino_features(cropped_rgb)

            # 파일명
            prefix = f"mouse{self.mouse_id}_cam{self.camera_id}_{saved_count:05d}"

            # 저장
            self.save_sample(prefix, cropped_rgb, cropped_mask, dino_features,
                           frame_count, bbox, width, height)

            saved_count += 1
            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        return saved_count - start_idx

    def segment_mouse(self, frame):
        if self.sam_predictor is None:
            # Fallback: Simple background subtraction
            return self.simple_segmentation(frame)

        # SAM segmentation
        self.sam_predictor.set_image(frame)

        # Auto-detect largest object
        # (여기서는 간단하게 중앙 point prompt 사용)
        h, w = frame.shape[:2]
        point_coords = np.array([[w//2, h//2]])
        point_labels = np.array([1])

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # 가장 높은 점수의 mask 선택
        best_mask = masks[scores.argmax()]

        return (best_mask * 255).astype(np.uint8)

    def simple_segmentation(self, frame):
        # Simple background subtraction (SAM 없을 때)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def compute_bbox(self, mask, full_w, full_h):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Padding (10%)
        pad = 0.1
        x = max(0, int(x - w * pad))
        y = max(0, int(y - h * pad))
        w = min(full_w - x, int(w * (1 + 2*pad)))
        h = min(full_h - y, int(h * (1 + 2*pad)))

        return {"x0": x, "y0": y, "width": w, "height": h,
                "full_w": full_w, "full_h": full_h}

    def crop_and_resize(self, image, mask, bbox):
        x0, y0, w, h = bbox["x0"], bbox["y0"], bbox["width"], bbox["height"]

        # Crop
        cropped_img = image[y0:y0+h, x0:x0+w]
        cropped_mask = mask[y0:y0+h, x0:x0+w]

        # Make square
        max_dim = max(w, h)
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        square_mask = np.zeros((max_dim, max_dim), dtype=np.uint8)

        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2

        square_img[pad_y:pad_y+h, pad_x:pad_x+w] = cropped_img
        square_mask[pad_y:pad_y+h, pad_x:pad_x+w] = cropped_mask

        # Resize
        resized_img = cv2.resize(square_img, (self.target_size, self.target_size))
        resized_mask = cv2.resize(square_mask, (self.target_size, self.target_size),
                                  interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask

    def extract_dino_features(self, image):
        if self.dino_model is None:
            # Fallback: 빈 features
            return np.zeros((self.target_size, self.target_size, 16), dtype=np.uint8)

        # Preprocessing
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()

        with torch.no_grad():
            features = self.dino_model.forward_features(img_tensor)
            patch_features = features['x_norm_patchtokens'][0]  # [N_patches, D]

            # Reshape to spatial
            h = w = int(np.sqrt(patch_features.shape[0]))
            spatial = patch_features.reshape(h, w, -1)

            # PCA (simplified: just take first 16 dims)
            reduced = spatial[..., :16]

            # Upsample
            upsampled = torch.nn.functional.interpolate(
                reduced.permute(2, 0, 1).unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear'
            )

            # Normalize to [0, 255]
            features_np = upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()
            features_norm = (features_np - features_np.min()) / (features_np.max() - features_np.min())
            features_uint8 = (features_norm * 255).astype(np.uint8)

            return features_uint8

    def save_sample(self, prefix, rgb, mask, dino_features, frame_id, bbox, full_w, full_h):
        # RGB
        cv2.imwrite(str(self.category_dir / f"{prefix}_rgb.png"), rgb)

        # Mask
        cv2.imwrite(str(self.category_dir / f"{prefix}_mask.png"), mask)

        # DINO features (16-channel PNG)
        # Note: OpenCV doesn't support 16-channel, use numpy save or custom encoding
        np.save(str(self.category_dir / f"{prefix}_feat16.npy"), dino_features)

        # Metadata JSON
        metadata = {
            "video_frame_id": int(frame_id),
            "crop_box_xyxy": [
                int(bbox["x0"]),
                int(bbox["y0"]),
                int(bbox["x0"] + bbox["width"]),
                int(bbox["y0"] + bbox["height"])
            ],
            "video_frame_width": int(full_w),
            "video_frame_height": int(full_h),
            "sharpness": 0.0,
            "crop_height": self.target_size,
            "crop_width": self.target_size
        }
        with open(self.category_dir / f"{prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        # Box.txt
        box_line = f"{frame_id} {bbox['x0']:.2f} {bbox['y0']:.2f} " \
                   f"{bbox['width']:.2f} {bbox['height']:.2f} " \
                   f"{full_w:.2f} {full_h:.2f} 0.00"
        with open(self.category_dir / f"{prefix}_box.txt", 'w') as f:
            f.write(box_line)

# 실행
if __name__ == "__main__":
    converter = MouseToFaunaConverter(
        mouse_id=1,
        camera_id=1,
        frame_sample_rate=30  # 매 30프레임마다 (100fps → ~3fps)
    )

    converter.process_all_videos()
```

---

**마지막 업데이트**: 2025-11-10
**작성자**: Claude Code
**문의**: GitHub Issues
