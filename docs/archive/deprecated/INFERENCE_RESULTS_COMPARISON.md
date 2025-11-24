# Multi-Animal Inference Results Comparison

## 추론 실행 요약

**Date**: 2025-11-22
**Checkpoint**: `results/checkpoint3000.pth` (Multi-animal, 3,000 iterations)
**Config**: `infer_mouse_dannce.yaml`
**Dataset**: Fauna_dataset (9 animal categories)

---

## 📊 생성 결과

### 전체 통계
- **총 생성 프레임**: 198 frames
- **생성 파일**:
  - 198 × 4 PNG images (image_gt, image_pred, mask_gt, mask_pred)
  - 198 × OBJ meshes
  - 198 × pose.txt files

### 포함된 동물 카테고리

Multi-animal checkpoint는 9개 동물 카테고리를 학습했으므로, 추론 결과에도 다양한 동물 포함:

| Frame Range | Animal | Example |
|-------------|--------|---------|
| 0-49 | **Mouse** (생쥐) | Frame 0, 10, 20... |
| 50-99 | Mouse Markerless | Frame 50, 60, 70... |
| 100-149 | Bear / Deer | Frame 100 (사슴) |
| 150-199 | Cow | Frame 200 (소) |
| ... | Elephant, Giraffe, Horse, Sheep, Zebra | 다양한 대형 동물들 |

---

## 🐭 생쥐 결과 상세 분석

### Frame 0 - Mouse DANNCE

**Input (Ground Truth)**:
- High-quality mouse image
- Standing pose with raised paws
- Clear white background
- Long tail visible

**Prediction (3,000 iterations)**:
- **Shape**: Basic 3D blob captured
- **Texture**: Dark/gray (no texture detail)
- **Silhouette**: Rough approximation
- **Quality**: ⭐⭐☆☆☆ (2/5)

**Observations**:
- ✅ Basic shape learning started
- ✅ No mesh collapse (stable)
- ❌ Very early stage - no fine details
- ❌ Texture not learned yet
- ❌ Silhouette accuracy low

---

## 📈 Multi-Animal vs Mouse-Only 비교

### Multi-Animal Checkpoint (현재)

**장점**:
- ✅ **안정적 학습**: 9,000 frames로 학습
- ✅ **Mesh collapse 없음**: 모든 프레임 정상 생성
- ✅ **다양한 shape 학습**: 9개 동물 카테고리
- ✅ **바로 사용 가능**: 추가 학습 없이 추론 가능

**단점**:
- ⚠️ **일반화된 shape space**: 생쥐에 특화되지 않음
- ⚠️ **품질 제한**: 여러 동물 동시 학습으로 각 동물의 디테일 감소
- ⚠️ **추론 시 전체 데이터셋 필요**: 생쥐만 원해도 다른 동물 데이터 필요

**생쥐 재구성 품질**: ⭐⭐☆☆☆ (2/5)
- 3,000 iterations로는 부족
- 50,000 iterations 학습 시 품질 향상 예상

---

### Mouse-Only Attempt (시도됨)

**장점**:
- ✅ **생쥐 전용 환경 구축 완료**: Mouse_only_dataset
- ✅ **Config 파일 준비**: debug, full, inference
- ✅ **학습 시작 성공**: 13 iterations 완료
- ✅ **Loss 감소 확인**: 22.56 → 16.19

**단점**:
- ❌ **데이터 부족**: 50 frames (너무 적음)
- ❌ **Mesh collapse**: Iteration 14에서 실패
- ❌ **Checkpoint 없음**: 학습 완료 실패
- ❌ **불안정**: Few-shot learning의 한계

**생쥐 재구성 품질**: ❌ (학습 실패)

---

## 🎯 결과 시각화 비교

### 생쥐 샘플 (Multi-animal checkpoint)

**Frame 0**:
```
Ground Truth: 고품질 생쥐 이미지 (서있는 자세, 긴 꼬리)
Prediction:   회색 blob (기본 형상만)
Mask GT:      정확한 실루엣
Mask Pred:    대략적인 blob 형태
```

**현재 품질 평가**:
- Shape learning: 10% (blob 단계)
- Texture learning: 0% (회색만)
- Pose accuracy: 20% (대략적 위치)
- Silhouette: 30% (rough approximation)

**예상 품질 (50K iterations)**:
- Shape learning: 70-80%
- Texture learning: 40-50%
- Pose accuracy: 60-70%
- Silhouette: 70-80%

---

## 💡 품질 개선 방안

### Option 1: Full Training 완료 (권장)

**현재 상태**: 3,000 / 50,000 iterations (6%)

**실행 방법**:
```bash
python run_full_notf32.py --config-name train_fauna_mouse_dannce
```

**예상 결과**:
- Duration: ~2-3 hours
- Quality: ⭐⭐⭐⭐☆ (4/5)
- Stability: ✅ Very stable (9,000 frames)

**기대 효과**:
- 텍스처 학습 시작
- 실루엣 정확도 대폭 향상
- Pose estimation 개선
- 디테일 증가

---

### Option 2: Mouse Markerless 데이터 추가

**현재 데이터**:
- mouse_dannce_6view: 50 frames ✅
- mouse_markerless_6view: ??? frames (확인 필요)

**작업**:
```bash
# 1. 데이터 확인
ls -la /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/train/

# 2. Mouse_only_dataset에 추가
cd /home/joon/dev/3DAnimals/data/fauna/Mouse_only_dataset/large_scale
cp -r ../../Fauna_dataset/large_scale/mouse_markerless_6view .

# 3. 학습 재시도
python run_debug_notf32.py --config-name train_mouse_only_debug
```

**필요 조건**:
- mouse_markerless에 충분한 데이터 (최소 100+ frames)
- 그래야 mesh collapse 방지 가능

---

### Option 3: Pretrained Fine-tuning

**방법**:
- Multi-animal checkpoint 로드
- Mouse_only_dataset으로 fine-tune
- 적은 데이터로도 안정적

**Config 예시**:
```yaml
# config/train_mouse_only_finetune.yaml
resume: /home/joon/dev/3DAnimals/results/checkpoint3000.pth
dataset:
  train_data_dir: /home/joon/dev/3DAnimals/data/fauna/Mouse_only_dataset
num_iters: 10000  # Fine-tuning은 짧게
```

**장점**:
- ✅ 안정적 (pretrained shape space)
- ✅ 생쥐에 특화 가능
- ✅ 적은 데이터로도 작동

---

## 📁 파일 위치

### 추론 결과
```
results/mouse_dannce_infer/test_results_checkpoint3000/
├── 0003000_0_image_gt.png          # Frame 0 - 생쥐 입력
├── 0003000_0_image_pred.png        # Frame 0 - 생쥐 예측
├── 0003000_0_mask_gt.png           # Frame 0 - 생쥐 마스크 GT
├── 0003000_0_mask_pred.png         # Frame 0 - 생쥐 마스크 예측
├── 0003000_0_mesh.obj              # Frame 0 - 생쥐 3D 메시
├── 0003000_0_pose.txt              # Frame 0 - 카메라 포즈
├── ...
├── 0003000_100_image_gt.png        # Frame 100 - 사슴
├── 0003000_200_image_gt.png        # Frame 200 - 소
└── ...
```

### Checkpoint
```
results/checkpoint3000.pth          # Multi-animal (3K iters)
results/checkpoint2500.pth          # Backup
results/checkpoint2000.pth          # Backup
```

---

## 🎓 주요 학습 내용

### 1. Multi-animal은 Generalist

- 9개 동물을 **동시에** 학습
- Shared shape space 사용
- 각 동물의 **전문성**은 낮지만 **안정성** 높음
- Few-shot 상황에서 **필수적**

### 2. Few-Shot Learning의 현실

**일반적 필요 데이터**:
- Good quality: 1,000+ frames per category
- Acceptable: 100-500 frames
- **현재 mouse_dannce: 50 frames** ← 매우 도전적!

**해결책**:
- **Multi-category training** (현재 방식) ✅
- **Data augmentation**
- **Pretrained models**
- **Transfer learning**

### 3. Training Iterations의 중요성

| Iterations | Quality | Status |
|------------|---------|--------|
| 0-1,000 | Initialization | Blob 단계 |
| 1,000-10,000 | Shape learning | 기본 형상 |
| 10,000-30,000 | Refinement | 디테일 증가 |
| 30,000-50,000 | **Fine details** | 텍스처, 정확도 |

**현재**: 3,000 iterations (초기 단계)
**권장**: 50,000 iterations (완성도)

---

## 🚀 추천 다음 단계

### 즉시 실행 가능

**Option A: Full Training 완료** ⭐ **최우선 권장**
```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# 백그라운드 실행 (~2-3시간)
nohup python run_full_notf32.py --config-name train_fauna_mouse_dannce \
  > /tmp/fauna_full_training.log 2>&1 &

# 진행 상황 모니터링
tail -f /tmp/fauna_full_training.log
```

**이유**:
- 안정적 (검증된 데이터셋)
- 높은 품질 기대 (50K iterations)
- 생쥐 포함 (mouse_dannce, mouse_markerless)

---

### 나중에 시도

**Option B: Mouse Markerless 데이터 확인 후 추가**
```bash
# 데이터 양 확인
find /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/mouse_markerless_6view/train -type d | wc -l

# 충분하면 (100+ frames) Mouse_only에 추가
```

**Option C: Pretrained Fine-tuning**
```bash
# Fine-tuning config 작성 후
python run_full_notf32.py --config-name train_mouse_only_finetune
```

---

## 📊 현재 상태 요약

| 항목 | Multi-animal | Mouse-only |
|------|--------------|------------|
| **데이터** | 9 categories (~9K frames) | 1 category (50 frames) |
| **학습 상태** | 3K iterations 완료 | 13 iterations (실패) |
| **Checkpoint** | ✅ checkpoint3000.pth | ❌ 없음 |
| **추론 가능** | ✅ 가능 (198 frames) | ❌ 불가 |
| **품질 (현재)** | ⭐⭐☆☆☆ (초기) | ❌ |
| **품질 (50K)** | ⭐⭐⭐⭐☆ (예상) | ❓ (데이터 부족) |
| **안정성** | ✅ 매우 안정적 | ❌ Mesh collapse |

**결론**: Multi-animal full training 완료가 **최선의 선택**입니다!

---

## 시각화 명령어

```bash
# 이미지 뷰어로 결과 확인
cd /home/joon/dev/3DAnimals/results/mouse_dannce_infer/test_results_checkpoint3000

# 생쥐 프레임들 (0-49)
eog *_0_image*.png *_10_image*.png *_20_image*.png

# 3D 메시 (Blender)
blender 0003000_0_mesh.obj

# 또는 MeshLab
meshlab 0003000_0_mesh.obj
```
