# 3D-Fauna Inference Quick Start Guide

**목적**: 학습된 모델을 사용하여 동물 이미지를 3D로 재구성하는 실전 매뉴얼

---

## 🚀 빠른 시작 (5분 완성)

### Step 1: Pretrained Model 다운로드

```bash
cd results/fauna
sh download_pretrained_fauna.sh
```

### Step 2: 테스트 이미지 준비

**방법 A - 제공된 데이터셋 사용**:
```bash
# 이미 다운로드된 Fauna dataset 사용
config/test_fauna.yaml에서:
test_data_dir: data/fauna/Fauna_dataset/large_scale/bear_comb_dinov2_new/test
```

**방법 B - 자신의 이미지 사용**:
```bash
# 1. 테스트 폴더 생성
mkdir -p data/my_animals

# 2. 이미지 복사 및 이름 변경 (반드시 *_rgb.png 형식)
cp my_dog_photo.jpg data/my_animals/dog001_rgb.png
cp my_cat_photo.jpg data/my_animals/cat001_rgb.png

# 3. config/test_fauna.yaml 수정
dataset:
  test_data_dir: data/my_animals/
```

### Step 3: Inference 실행

```bash
python run.py --config-name test_fauna
```

### Step 4: 결과 확인

```bash
ls results/fauna/pretrained_fauna/visualization/

# 출력 파일:
# - {name}_mesh.obj           # 3D 메쉬 (Blender, MeshLab 등에서 열기)
# - {name}_image_pred.png     # 재구성된 RGB 이미지
# - {name}_mask_pred.png      # 예측된 마스크
# - {name}_pose.txt           # 카메라 포즈 (quaternion + translation)
# - {name}_arti_params.txt    # 관절 파라미터
```

---

## 📋 Input 형식

### 필수 파일
```
test_dir/
└── {name}_rgb.png          # RGB 이미지 (256x256 권장, 자동 resize됨)
```

### 선택적 파일 (품질 향상용)
```
test_dir/
├── {name}_rgb.png
└── {name}_mask.png         # Segmentation mask (texture finetuning 시 필요)
```

### 지원 동물 종류
- ✅ **Quadrupeds (4족 보행 동물)**: 개, 고양이, 말, 사자, 호랑이, 곰, 기린, 얼룩말, 소, 사슴 등
- ❌ **Not supported**: 새, 사람, 물고기, 곤충 등

---

## 📊 Output 형식

### 1. 3D Mesh (`*_mesh.obj`)

**사용법**:
```bash
# Blender에서 열기
blender --python -c "import bpy; bpy.ops.import_scene.obj(filepath='dog001_mesh.obj')"

# MeshLab에서 열기
meshlab dog001_mesh.obj
```

**구조**:
- **Vertices**: 3D 좌표 (canonical space)
- **Faces**: Triangle connectivity
- **Texture**: Albedo (base color)
- **Normals**: Surface normals

### 2. Pose (`*_pose.txt`)

**Format**: `[qw, qx, qy, qz, tx, ty, tz]`
```
0.9659  0.0000  0.2588  0.0000  0.0000  0.0000 -2.5
```

**의미**:
- `[qw, qx, qy, qz]`: Quaternion rotation (camera orientation)
- `[tx, ty, tz]`: Translation (camera position)

**활용**:
```python
import numpy as np
from scipy.spatial.transform import Rotation

pose = np.loadtxt("dog001_pose.txt")
quat = pose[:4]   # [qw, qx, qy, qz]
trans = pose[4:]  # [tx, ty, tz]

# Quaternion → Rotation matrix
rot_matrix = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
```

### 3. Articulation Parameters (`*_arti_params.txt`)

**Format**: `(N_bones, dim)` - 각 관절의 learned representation
```
0.123 -0.456 0.789 ...
0.234 -0.567 0.890 ...
...
```

**의미**:
- 각 행: 하나의 bone/joint에 대한 파라미터
- 물리적 각도가 아닌 **learned latent space**
- Skinning을 통해 mesh에 적용됨

**활용**: Animation 생성 시 보간(interpolation)

---

## 🎨 고급 기능

### Texture Finetuning (품질 향상)

**Config 수정** (`config/test_fauna.yaml`):
```yaml
finetune_texture: true
finetune_iters: 50        # 반복 횟수 (default: 10)
finetune_lr: 0.0005       # Learning rate (default: 0.001)
```

**효과**:
- Input view에서 texture 정밀도 ↑
- 배경 색상이 mesh에 섞이는 현상 방지
- **주의**: Mask 제공 시 더 정확함

### 다양한 시점 렌더링

**Config 수정**:
```yaml
render_modes: [input_view, other_views, rotation]
```

**Modes**:
- `input_view`: 입력 시점에서 렌더링
- `other_views`: 12개 고정 시점 (360도)
- `rotation`: 회전 비디오 생성
- `animation`: 사전 정의된 모션 적용 (보행 등)

**실행**:
```bash
python visualization/visualize_results_fauna.py --config-name test_fauna
```

### Batch Processing (다수 이미지)

**Config 수정**:
```yaml
dataset:
  batch_size: 16          # GPU 메모리에 따라 조절
  test_data_dir: data/animal_collection/
```

**폴더 구조**:
```
data/animal_collection/
├── animal001_rgb.png
├── animal002_rgb.png
├── ...
└── animal100_rgb.png
```

**실행**: 한 번에 모든 이미지 처리됨

---

## 🔧 문제 해결

### Q1: CUDA Out of Memory

**해결책**:
```yaml
dataset:
  batch_size: 1           # Batch size 줄이기
  in_image_size: 128      # 이미지 크기 줄이기
  out_image_size: 128
```

### Q2: 재구성 품질이 나쁨

**원인 1**: 입력 이미지가 흐릿함
- **해결**: 고해상도 이미지 사용, 명확한 동물 형태

**원인 2**: 배경이 복잡함
- **해결**: Segmentation mask 제공 + `finetune_texture: true`

**원인 3**: 지원하지 않는 동물
- **해결**: Quadrupeds만 지원 (4족 보행 동물)

### Q3: Mesh가 이상하게 생김

**원인 1**: 포즈가 극단적 (누워있음, 점프 등)
- **해결**: 서있는 자세의 이미지 사용

**원인 2**: Occlusion (가려짐)
- **해결**: 전신이 보이는 이미지 사용

### Q4: FileNotFoundError: *_dino.npy

**원인**: Config에서 DINO feature 로드 시도
**해결**:
```yaml
dataset:
  load_dino_feature: false   # Test 시 불필요
```

---

## 💡 Best Practices

### 입력 이미지 선택
✅ **Good**:
- 동물 전신이 보임
- 명확한 배경과 대비
- 서있는 자세 (canonical pose에 가까움)
- 고해상도 (512x512 이상)

❌ **Bad**:
- 동물 일부만 보임 (얼굴만, 다리 잘림)
- 복잡한 배경 (나뭇가지, 사람, 다른 동물)
- 극단적 포즈 (누워있음, 공중에 뜀)
- 저해상도, 흐릿함

### 성능 최적화
- **GPU 메모리 부족**: `batch_size=1`, `image_size=128`
- **빠른 처리**: `finetune_texture=false`, `render_modes=[input_view]`
- **고품질 결과**: `finetune_texture=true`, `finetune_iters=50`, mask 제공

---

## 📈 성능 벤치마크

**Hardware**: NVIDIA RTX 3090 (24GB)

| 설정 | Batch Size | 처리 시간/이미지 | 품질 |
|-----|------------|----------------|------|
| Fast (no finetune) | 8 | ~2초 | Good |
| Balanced (finetune 10 iter) | 4 | ~5초 | Better |
| High Quality (finetune 50 iter) | 1 | ~15초 | Best |

---

## 🎯 실전 예제

### 예제 1: 반려견 사진 → 3D 모델

```bash
# 1. 사진 준비
cp my_dog.jpg data/my_test/mydog_rgb.png

# 2. Config 수정
vim config/test_fauna.yaml
# → test_data_dir: data/my_test/

# 3. 실행
python run.py --config-name test_fauna

# 4. 결과 확인
ls results/fauna/pretrained_fauna/visualization/mydog_*

# 5. Blender에서 열기
blender results/fauna/pretrained_fauna/visualization/mydog_mesh.obj
```

### 예제 2: 동물원 사진 100장 배치 처리

```bash
# 1. 이미지 준비 및 이름 변경
for i in {1..100}; do
  cp zoo_photo_$i.jpg data/zoo/animal$(printf "%03d" $i)_rgb.png
done

# 2. Config
dataset:
  batch_size: 16
  test_data_dir: data/zoo/

# 3. 실행
python run.py --config-name test_fauna

# 4. 결과: 100개 3D 모델 생성
ls results/fauna/pretrained_fauna/visualization/*.obj | wc -l
# → 100
```

### 예제 3: 고품질 렌더링 + 애니메이션

```bash
# 1. Config
finetune_texture: true
finetune_iters: 50
render_modes: [input_view, rotation, animation]

# 2. 실행
python visualization/visualize_results_fauna.py --config-name test_fauna

# 3. 결과: 회전 비디오 + 애니메이션 생성
ls results/fauna/pretrained_fauna/visualization/*.mp4
```

---

## 📚 추가 학습 자료

**자세한 가이드**:
- [Training & Inference 완전 가이드](./251110_fauna_training_inference_guide.md)
- [Official README](../../README.md)

**공식 자료**:
- Project Page: https://kyleleey.github.io/3DFauna/
- Paper: https://arxiv.org/abs/2401.02400
- GitHub: https://github.com/3dmagicpony/3DAnimals

**관련 모델**:
- **MagicPony**: Category-specific models (horse, bird, etc.)
- **Ponymation**: Motion generation from video

---

## ✅ 체크리스트

### 시작 전
- [ ] Pretrained model 다운로드 완료
- [ ] Test 이미지 준비 (`*_rgb.png` 형식)
- [ ] CUDA 사용 가능 확인
- [ ] Config 파일 경로 확인

### 실행 후
- [ ] `*_mesh.obj` 파일 생성 확인
- [ ] Blender/MeshLab에서 열어보기
- [ ] Rendered image 품질 확인
- [ ] 필요시 texture finetuning 재실행

---

**마지막 업데이트**: 2025-11-10
**지원**: GitHub Issues - https://github.com/3dmagicpony/3DAnimals/issues
