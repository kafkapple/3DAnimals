# Mouse-Only Training Setup Guide

## ✅ 완료된 작업

### 1. 생쥐 전용 데이터셋 구조 생성

**위치**: `/home/joon/dev/3DAnimals/data/fauna/Mouse_only_dataset/`

```
Mouse_only_dataset/
├── large_scale/
│   └── mouse_dannce_6view/       # 생쥐 데이터 (50 frames)
│       ├── train/
│       │   ├── 000000_00000/
│       │   ├── 000001_00000/
│       │   ├── 000002_00000/
│       │   ├── 000003_00000/
│       │   └── 000004_00000/
│       ├── val -> train
│       └── test -> train
├── few_shot_animal3d/            # 빈 디렉토리 (required by FaunaDataset)
├── few_shot_web/                 # 빈 디렉토리
└── few_shot_web_back/            # 빈 디렉토리
```

**검증 결과**: ✅ FaunaDataset이 **1개 카테고리만 로딩** 확인됨
```
using 1 categories, contains: ['large_scale_mouse']
```

### 2. Config 파일 생성

#### Debug Training (검증용)
**파일**: `config/train_mouse_only_debug.yaml`
- **num_iters**: 3,000
- **데이터셋**: Mouse_only_dataset (생쥐만)
- **목적**: 빠른 검증 (10-15분)

#### Full Training (본격 학습)
**파일**: `config/train_mouse_only_full.yaml`
- **num_iters**: 50,000
- **데이터셋**: Mouse_only_dataset (생쥐만)
- **목적**: 완전한 학습 (2-3시간)

#### Inference (추론)
**파일**: `config/infer_mouse_only.yaml`
- **목적**: 생쥐 전용 체크포인트로 추론
- **데이터셋**: Mouse_only_dataset

### 3. 학습 시도 결과

**성공 사항**:
- ✅ 생쥐 데이터만 로딩 (9개 동물 → 1개 카테고리)
- ✅ 학습 시작 (13 iterations 완료)
- ✅ Loss 감소 확인 (22.56 → 16.19)

**실패 사항**:
- ❌ Iteration 14에서 mesh 생성 실패
- 에러: `AssertionError: Got empty training triangle mesh`

## ⚠️ 문제 분석: Few-Shot Learning의 한계

### 근본 원인

**데이터 부족**:
- 생쥐 데이터: **50 frames만** (매우 적음)
- 원래 Fauna 학습: 9개 동물 × ~1000 frames = **~9,000 frames**
- 비율: **1/180** (0.56%)

**Mesh Collapse 발생 원인**:
1. 적은 데이터로 SDF 학습 불안정
2. Marching Cubes가 유효한 mesh 추출 실패
3. Early iteration에서 흔한 문제 (특히 few-shot)

### 비교: Multi-animal vs Mouse-only

| 항목 | Multi-animal (이전) | Mouse-only (현재) |
|------|---------------------|------------------|
| **동물 카테고리** | 9개 (bear, cow, elephant, giraffe, horse, 2× mouse, sheep, zebra) | 1개 (mouse only) |
| **총 프레임 수** | ~9,000 frames | 50 frames |
| **학습 안정성** | ✅ 높음 (다양한 데이터) | ❌ 낮음 (적은 데이터) |
| **Mesh collapse** | 거의 없음 | 자주 발생 |
| **결과 품질** | 일반화된 동물 shape | 생쥐 전용 (안정적이면 더 정확) |

## 🎯 해결 방안

### 방안 1: 다른 생쥐 데이터 추가 (권장)

**현재 사용 가능한 데이터**:
- `mouse_dannce_6view`: 50 frames ✅ (현재 사용 중)
- `mouse_markerless_6view`: ??? frames (확인 필요)

**작업**:
```bash
# mouse_markerless도 Mouse_only_dataset에 추가
cd /home/joon/dev/3DAnimals/data/fauna/Mouse_only_dataset/large_scale
cp -r ../../Fauna_dataset/large_scale/mouse_markerless_6view .
```

### 방안 2: Multi-animal 학습 사용 (안정적)

**장점**:
- 이미 학습된 checkpoint3000.pth 활용
- 안정적인 학습 (9,000 frames)
- 생쥐 포함 (mouse_dannce, mouse_markerless)

**단점**:
- 다른 동물들도 함께 학습됨
- Shape space가 넓어 덜 전문화됨

**추론 시 생쥐만 필터링**:
```python
# FaunaDataset에서 test_data_dir로 mouse만 지정 가능
# 또는 추론 후 결과에서 mouse만 선택
```

### 방안 3: Regularization 강화

**config 수정**:
```yaml
# config/model/fauna_mouse_dannce.yaml
sdf_gradient_reg_loss_weight: 0.1 → 1.0  # SDF regularization 10배 증가
grid_res: 64 → 32  # Grid resolution 감소 (더 부드러운 mesh)
```

**목적**: Mesh collapse 방지

### 방안 4: Pretrained Model Fine-tuning

**절차**:
1. Multi-animal checkpoint 로드
2. Mouse_only_dataset으로 fine-tune
3. 더 적은 iteration으로 수렴

**장점**:
- Pretrained shape space 활용
- 적은 데이터로도 안정적
- 생쥐에 특화

## 📊 현재 상태 요약

### 데이터 구조

| 디렉토리 | 동물 카테고리 | 프레임 수 | 용도 |
|----------|--------------|-----------|------|
| `Fauna_dataset` | 9개 (모든 동물) | ~9,000 | 원래 Multi-animal 학습 |
| `Mouse_only_dataset` | 1개 (생쥐만) | 50 | 생쥐 전용 학습 (구축 완료) |

### Checkpoints

| 파일 | 학습 데이터 | Iterations | 상태 |
|------|-------------|-----------|------|
| `results/checkpoint3000.pth` | 9개 동물 혼합 | 3,000 | ✅ 사용 가능 |
| `results/mouse_only_debug/checkpoint*.pth` | 생쥐만 | 실패 | ❌ Mesh collapse |

### Config 파일

| 파일 | 데이터셋 | 상태 |
|------|----------|------|
| `train_fauna_mouse_dannce_debug.yaml` | Multi-animal (9개) | ✅ 작동 확인 |
| `train_mouse_only_debug.yaml` | Mouse-only (1개) | ⚠️ Mesh collapse |
| `infer_mouse_dannce.yaml` | Multi-animal | ✅ 추론 성공 (124 frames) |
| `infer_mouse_only.yaml` | Mouse-only | 미사용 (checkpoint 없음) |

## 🚀 추천 작업 순서

### Option A: Multi-animal 학습 활용 (빠르고 안정적)

```bash
# 1. 이미 있는 checkpoint 사용
# results/checkpoint3000.pth (9개 동물 혼합)

# 2. Full training 실행 (필요시)
python run_full_notf32.py  # 50K iterations, ~2-3 hours

# 3. 추론 (생쥐 포함)
python run_debug_notf32.py --config-name infer_mouse_dannce

# 4. 결과에서 생쥐만 필터링
# mouse_dannce, mouse_markerless 카테고리만 선택
```

### Option B: 생쥐 전용 + 더 많은 데이터

```bash
# 1. mouse_markerless 데이터 추가
cd /home/joon/dev/3DAnimals/data/fauna/Mouse_only_dataset/large_scale
cp -r ../../Fauna_dataset/large_scale/mouse_markerless_6view .

# 2. 데이터 확인
find mouse_markerless_6view/train -type d | wc -l  # 프레임 수 확인

# 3. Debug 학습 재시도
python run_debug_notf32.py --config-name train_mouse_only_debug

# 4. 성공 시 Full training
python run_full_notf32.py --config-name train_mouse_only_full
```

### Option C: Pretrained Fine-tuning

```bash
# 1. Config 수정 (resume checkpoint 추가)
# config/train_mouse_only_finetune.yaml
resume: /home/joon/dev/3DAnimals/results/checkpoint3000.pth
dataset:
  train_data_dir: /home/joon/dev/3DAnimals/data/fauna/Mouse_only_dataset

# 2. Fine-tune 학습
python run_full_notf32.py --config-name train_mouse_only_finetune
```

## 📝 명령어 요약

### 현재 사용 가능한 명령어

```bash
# Multi-animal Full Training
python run_full_notf32.py --config-name train_fauna_mouse_dannce

# Multi-animal Inference (모든 동물 포함 생쥐)
python run_debug_notf32.py --config-name infer_mouse_dannce

# Mouse-only Debug Training (데이터 부족으로 실패 가능)
python run_debug_notf32.py --config-name train_mouse_only_debug

# Mouse-only Full Training (데이터 보강 후)
python run_full_notf32.py --config-name train_mouse_only_full
```

## 🎓 학습 내용

### FaunaDataset 동작 방식

**자동 카테고리 스캔**:
```python
# model/dataset/FaunaDataset.py:52
category_names = sorted(os.listdir(os.path.join(root, 'large_scale')))
```
- `large_scale/` 폴더 내의 **모든 하위 디렉토리**를 카테고리로 인식
- 따라서 `Fauna_dataset/large_scale/`에 9개 동물 → 9개 모두 학습
- `Mouse_only_dataset/large_scale/`에 생쥐만 → 생쥐만 학습

### Few-Shot Learning의 도전

**일반적으로 필요한 데이터**:
- **안정적 학습**: 1,000+ frames per category
- **Few-shot**: 100-500 frames (도전적)
- **현재 생쥐**: 50 frames (매우 도전적)

**해결책**:
1. **Data augmentation**: Flip, rotation, crop
2. **Pretrained model**: Transfer learning
3. **Stronger regularization**: Prevent overfitting
4. **More data**: 다른 생쥐 데이터셋 추가

## 🔍 다음 단계 질문

1. **Multi-animal 학습**을 활용하시겠습니까? (안정적, 빠름)
2. **mouse_markerless 데이터**를 추가하시겠습니까? (생쥐 전용, 더 많은 데이터)
3. **Pretrained fine-tuning**을 시도하시겠습니까? (best of both worlds)

원하시는 방향을 알려주시면 바로 진행하겠습니다!
