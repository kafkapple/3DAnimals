# 3DAnimals Quickstart Manual

**Last Updated**: 2025-11-24
**Purpose**: 데이터 준비부터 학습, 추론까지 전체 워크플로우

---

## 📋 목차

1. [환경 설정](#1-환경-설정)
2. [데이터 준비](#2-데이터-준비)
3. [Debug 모드 실행 (필수)](#3-debug-모드-실행-필수)
4. [Full Training 실행](#4-full-training-실행)
5. [Inference 실행](#5-inference-실행)
6. [결과 시각화](#6-결과-시각화)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. 환경 설정

### 1.1 시스템 요구사항

| 구성 요소 | 요구사항 | 권장사항 |
|-----------|----------|----------|
| **GPU** | CUDA 11.8 지원 | RTX 3060 12GB 이상 |
| **메모리** | 16GB+ | 32GB |
| **디스크** | 50GB+ | 100GB (데이터셋 포함) |
| **OS** | Linux | Ubuntu 20.04+ |

### 1.2 Conda 환경 설정

```bash
# 1. Conda 환경 생성
conda create -n 3danimals python=3.10 -y
conda activate 3danimals

# 2. PyTorch 설치 (CUDA 11.8)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
  pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. PyTorch3D 설치
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# 4. 기타 의존성
pip install hydra-core omegaconf tqdm wandb accelerate einops
```

### 1.3 환경 검증

```bash
cd /home/joon/dev/3DAnimals
python scripts/test_env.py
```

**예상 출력**:
```
================================================================================
Environment Verification
================================================================================
PyTorch version: 2.0.0
CUDA version: 11.8
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
PyTorch3D version: 0.7.4
================================================================================
✅ All packages loaded successfully!
```

### 1.4 RTX 3060 전용: CUDA Fix 검증

```bash
python scripts/test_cuda_fix.py
```

**목적**: TF32 CUBLAS 에러 발생 여부 확인

---

## 2. 데이터 준비

### 2.1 Fauna Mouse DANNCE 데이터셋

#### 데이터셋 구조

```
/home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/
├── mouse_dannce_6view/           # Mouse DANNCE 6-view dataset
│   ├── train/
│   │   ├── 000000_00000/         # Sequence 1
│   │   │   ├── images/
│   │   │   │   ├── view_0.png
│   │   │   │   ├── view_1.png
│   │   │   │   └── ... (6 views)
│   │   │   ├── masks/
│   │   │   ├── cameras.npz       # Camera parameters
│   │   │   ├── dino_features.pth # DINO features (필수!)
│   │   │   └── metadata.json
│   │   ├── 000000_00001/
│   │   └── ...
│   ├── test/                     # Same structure as train
│   └── val/                      # Same structure as train
└── [other animals...]
```

#### 필수 파일 체크리스트

각 시퀀스 폴더마다:
- [ ] `images/` - RGB 이미지 (6 views × 256×256)
- [ ] `masks/` - Segmentation masks
- [ ] `cameras.npz` - Camera intrinsics & extrinsics
- [ ] `dino_features.pth` - **필수!** DINO ViT features
- [ ] `metadata.json` - 메타데이터

### 2.2 데이터셋 검증

```bash
# 데이터셋 경로 확인
ls -lh /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view/train/

# 시퀀스 개수 확인
find /home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view/train/ \
  -name "dino_features.pth" | wc -l
```

**예상 출력**: 50-100개 시퀀스

### 2.3 DINO Features 추출 (필요시)

DINO features가 없는 경우:

```bash
python scripts/extract_dino_features.py \
  --data_dir data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view \
  --split train \
  --gpu 0
```

**소요 시간**: ~1-2분 (100 sequences)

---

## 3. Debug 모드 실행 (필수)

### 3.1 왜 Debug 모드가 필수인가?

**목적**: PoC (Proof of Concept) 검증
- ✅ 데이터 로딩 정상 작동 확인
- ✅ 모델 초기화 성공 확인
- ✅ 학습 루프 안정성 확인
- ✅ GPU 메모리 사용량 확인
- ✅ CUBLAS 에러 여부 확인

**예상 시간**: ~15-20분 (5,000 iterations)

**❌ 주의**: Debug 모드 없이 바로 Full Training 시작하면 11시간 낭비 위험!

### 3.2 Debug Config 확인

**파일**: `config/train_fauna_mouse_dannce_debug.yaml`

```yaml
num_iters: 5000              # Full의 1/10 (5K vs 50K)
save_checkpoint_freq: 1000   # 더 자주 저장
log_image_freq: 100          # 더 자주 로그
```

### 3.3 Debug 실행

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# RTX 3060 (TF32 문제 있음)
python run_debug_notf32.py  # 또는 run.py --config-name ... disable_tf32=true

# 다른 GPU (TF32 정상)
python run.py --config-name train_fauna_mouse_dannce_debug
```

### 3.4 Debug 성공 기준

**로그 확인**:
```bash
tail -f /tmp/fauna_debug.log  # 실시간 모니터링
```

**성공 조건**:
- [ ] TF32 disabled 메시지 (RTX 3060)
- [ ] Data loading 성공 (50 sequences)
- [ ] Model initialization 성공 (grid_res: 64)
- [ ] Training loop 5000 iterations 완료
- [ ] No CUBLAS errors
- [ ] GPU memory < 12GB
- [ ] Checkpoint 저장 성공 (`results/checkpoint5000.pth`)

**실패 시 조치**:
1. 에러 로그 확인
2. [Troubleshooting](#7-troubleshooting) 섹션 참조
3. Debug 모드 재실행

---

## 4. Full Training 실행

### 4.1 Debug 모드 성공 후에만 진행

**확인 사항**:
- [x] Debug 모드 5K iterations 성공
- [x] No CUBLAS errors
- [x] Checkpoint 생성 확인

### 4.2 Full Training Config

**파일**: `config/train_fauna_mouse_dannce.yaml`

```yaml
num_iters: 50000             # 50K iterations (~2-3 hours)
save_checkpoint_freq: 5000   # 5K마다 저장 (10개 checkpoint)
log_image_freq: 500          # 500 iter마다 로그
```

### 4.3 실행

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# 백그라운드 실행 (권장)
nohup python run_full_notf32.py \
  --config-name train_fauna_mouse_dannce \
  > /tmp/fauna_full_training.log 2>&1 &

# PID 확인
echo $!
```

### 4.4 학습 모니터링

#### GPU 사용량 모니터링

```bash
# 실시간 GPU 모니터링
nvidia-smi -l 1

# 프로세스 확인
ps aux | grep python
```

#### 학습 로그 확인

```bash
# 실시간 로그 확인
tail -f /tmp/fauna_full_training.log

# 최근 100줄
tail -100 /tmp/fauna_full_training.log
```

#### WandB 모니터링 (선택)

```bash
# wandb 로그인 (최초 1회)
wandb login

# 브라우저에서 확인
# https://wandb.ai/your-username/fauna_mouse_dannce
```

### 4.5 예상 학습 시간 (RTX 3060)

| Iterations | Time | Checkpoint |
|------------|------|------------|
| 5,000 | ~15min | checkpoint5000.pth |
| 10,000 | ~30min | checkpoint10000.pth (articulation 시작) |
| 20,000 | ~1hr | checkpoint20000.pth |
| 30,000 | ~1.5hr | checkpoint30000.pth (legs attach) |
| 40,000 | ~2hr | checkpoint40000.pth |
| **50,000** | **~2.5-3hr** | **checkpoint50000.pth** (최종) |

### 4.6 Progressive Training Milestones

| Phase | Iterations | 활성화 기능 | 기대 결과 |
|-------|-----------|------------|-----------|
| **Initialization** | 0-5K | SDF, Pose, Texture | Ellipsoid → Mouse 형태 |
| **Shape Refinement** | 5K-10K | Shape, Pose | 디테일 개선 |
| **Articulation** | 10K-30K | **Articulation** | 골격 학습 시작 |
| **Leg Attachment** | 30K-50K | **Legs attach** | 완전한 articulation |

### 4.7 생성되는 Checkpoint

```bash
results/
├── checkpoint5000.pth    (257M)
├── checkpoint10000.pth   (257M) ← Articulation 시작
├── checkpoint15000.pth   (257M)
├── checkpoint20000.pth   (257M)
├── checkpoint25000.pth   (257M)
├── checkpoint30000.pth   (257M) ← Legs attach
├── checkpoint35000.pth   (257M)
├── checkpoint40000.pth   (257M)
├── checkpoint45000.pth   (257M)
└── checkpoint50000.pth   (257M) ← 최종 모델
```

---

## 5. Inference 실행

### 5.1 목적

학습된 모델로 테스트 데이터에 대해 3D 재구성 수행

### 5.2 Inference Config

**파일**: `config/infer_mouse_dannce.yaml`

```yaml
run_train: false
run_test: true
resume: results/checkpoint50000.pth  # 사용할 checkpoint
checkpoint_dir: results/infer_50k    # 출력 디렉토리
```

### 5.3 단일 Checkpoint 추론

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint50000.pth \
  checkpoint_dir=results/infer_50k \
  output_dir=results/infer_50k
```

**예상 시간**: ~15-20분 (100 test sequences)

### 5.4 여러 Checkpoint 비교

Progressive training 품질 비교를 위해 3개 checkpoint 추론:

```bash
# 1. Checkpoint 10K (Articulation 시작)
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint10000.pth \
  checkpoint_dir=results/infer_10k

# 2. Checkpoint 30K (Legs attach)
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint30000.pth \
  checkpoint_dir=results/infer_30k

# 3. Checkpoint 50K (최종)
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/checkpoint50000.pth \
  checkpoint_dir=results/infer_50k
```

### 5.5 결과 파일 구조

```
results/infer_50k/test_results_*/
├── 0000000_00000_image_gt.png      # Ground truth 이미지
├── 0000000_00000_image_pred.png    # 예측 이미지
├── 0000000_00000_mask_gt.png       # Ground truth mask
├── 0000000_00000_mask_pred.png     # 예측 mask
├── 0000000_00000_mesh.obj          # 3D mesh
├── 0000000_00000_pose.txt          # Pose parameters
└── ...
```

### 5.6 결과 파일 개수 확인

```bash
# 생성된 mesh 개수
find results/infer_50k/test_results_* -name "*.obj" | wc -l

# 비교: 모든 checkpoint
echo "=== Checkpoint Results Comparison ==="
echo "10K: $(find results/infer_10k/test_results_* -name "*.obj" 2>/dev/null | wc -l) meshes"
echo "30K: $(find results/infer_30k/test_results_* -name "*.obj" 2>/dev/null | wc -l) meshes"
echo "50K: $(find results/infer_50k/test_results_* -name "*.obj" 2>/dev/null | wc -l) meshes"
```

---

## 6. 결과 시각화

### 6.1 이미지 비교 (GT vs Prediction)

```bash
# 샘플 이미지 확인
eog results/infer_50k/test_results_*/0000000_00000_image_pred.png
```

### 6.2 Mesh 시각화 (MeshLab 사용)

```bash
# MeshLab 설치 (Ubuntu)
sudo apt install meshlab

# Mesh 열기
meshlab results/infer_50k/test_results_*/0000000_00000_mesh.obj
```

### 6.3 정량 평가

**Mask IoU** (Intersection over Union):
```python
# Python으로 계산
import numpy as np
from PIL import Image

def compute_iou(mask_gt_path, mask_pred_path):
    gt = np.array(Image.open(mask_gt_path)) > 0
    pred = np.array(Image.open(mask_pred_path)) > 0
    intersection = (gt & pred).sum()
    union = (gt | pred).sum()
    return intersection / union if union > 0 else 0.0

iou = compute_iou(
    "results/infer_50k/test_results_*/0000000_00000_mask_gt.png",
    "results/infer_50k/test_results_*/0000000_00000_mask_pred.png"
)
print(f"Mask IoU: {iou:.3f}")
```

**기대 성능**:
- Mask IoU > 0.8 (Good)
- RGB PSNR > 20 (Good)

### 6.4 Progressive Quality 비교

checkpoint10K, 30K, 50K 시각적 비교:

```bash
# 동일한 샘플 나란히 비교
eog results/infer_10k/test_results_*/0000000_00000_image_pred.png \
    results/infer_30k/test_results_*/0000000_00000_image_pred.png \
    results/infer_50k/test_results_*/0000000_00000_image_pred.png
```

**기대 개선**:
- 10K: 기본 형태 + 초기 articulation
- 30K: 형태 개선 + legs attached
- 50K: 최고 품질 + 안정적 articulation

---

## 7. Troubleshooting

### 문제 1: CUBLAS Error

**증상**:
```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED
```

**원인**: TF32 활성화 (RTX 3060)

**해결**:
```bash
# run_debug_notf32.py 사용
python run_debug_notf32.py

# 또는 config에 disable_tf32: true 추가
python run.py --config-name train_fauna_mouse_dannce disable_tf32=true
```

**참고**: `docs/guides/RTX_3060_SETUP_GUIDE.md`

---

### 문제 2: Out of Memory (OOM)

**증상**:
```
RuntimeError: CUDA out of memory
```

**원인**: GPU 메모리 부족 (RTX 3060 12GB 초과)

**해결**:
```yaml
# config/model/fauna_mouse_dannce.yaml 수정
grid_res: 64  # 128 → 64로 감소 (메모리 1/8)
```

**Trade-off**: 품질 약간 저하 vs 메모리 절약

---

### 문제 3: Data Loading 실패

**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../dino_features.pth'
```

**원인**: DINO features 누락

**해결**:
```bash
python scripts/extract_dino_features.py \
  --data_dir data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view \
  --split train
```

---

### 문제 4: Checkpoint 로딩 실패

**증상**:
```
ValueError: invalid literal for int() with base 10: ''
```

**원인**: `results/` 디렉토리에 숫자 없는 파일

**해결**:
```bash
# 불필요한 파일 백업
mkdir -p results/backup_old
mv results/mammal_*.pth results/backup_old/
mv results/fauna_mouse_mammal_init.pth results/backup_old/
```

---

### 문제 5: 학습이 매우 느림

**증상**: 1 iteration > 5초

**원인**:
1. DINO features 로딩 느림
2. 데이터셋 경로가 네트워크 드라이브

**해결**:
```bash
# 데이터셋을 로컬 SSD로 복사
cp -r /network/data /local/ssd/data

# Config 수정
dataset:
  train_data_dir: /local/ssd/data/fauna/Fauna_dataset
```

---

### 문제 6: Mesh Collapse (품질 나쁨)

**증상**: 재구성된 mesh가 매우 작거나 collapse됨

**원인**:
1. 학습 iterations 부족 (< 30K)
2. SDF regularization 너무 강함
3. 데이터 부족 (< 50 frames)

**해결**:
```bash
# 1. 더 긴 학습
num_iters: 100000  # 50K → 100K

# 2. Config 조정 (advanced)
# model/fauna_mouse_dannce.yaml
sdf_reg_weight: 0.01  # 0.1 → 0.01로 감소

# 3. 데이터 수집
# 100+ frames 권장
```

---

## 8. 권장 워크플로우 (요약)

### Step-by-Step Checklist

#### Phase 1: 환경 설정 (최초 1회)
- [ ] Conda 환경 생성 및 패키지 설치
- [ ] `python scripts/test_env.py` 성공
- [ ] `python scripts/test_cuda_fix.py` 성공 (RTX 3060)

#### Phase 2: 데이터 준비
- [ ] 데이터셋 다운로드/복사
- [ ] DINO features 추출 확인
- [ ] 시퀀스 개수 확인 (50-100개)

#### Phase 3: Debug 모드 (필수!)
- [ ] `python run_debug_notf32.py` 실행
- [ ] 5K iterations 성공
- [ ] checkpoint5000.pth 생성 확인

#### Phase 4: Full Training
- [ ] `nohup python run_full_notf32.py &` 실행
- [ ] GPU 모니터링 (`nvidia-smi -l 1`)
- [ ] 2-3시간 대기
- [ ] checkpoint50000.pth 생성 확인

#### Phase 5: Inference
- [ ] 3개 checkpoint 추론 (10K, 30K, 50K)
- [ ] 결과 파일 생성 확인
- [ ] Mesh 개수 확인

#### Phase 6: 결과 분석
- [ ] 이미지 시각적 비교
- [ ] Mesh 품질 확인 (MeshLab)
- [ ] Mask IoU 계산 (> 0.8 목표)
- [ ] Progressive 품질 개선 확인

---

## 9. 참고 문서

### 가이드
- **RTX 3060 Setup**: `docs/guides/RTX_3060_SETUP_GUIDE.md`
- **CUDA Fix**: `docs/guides/CUDA_FIX_GUIDE.md`
- **Visualization**: `docs/guides/VISUALIZATION_GUIDE.md`

### 연구 노트
- **Full Training Session**: `docs/research/251123_fauna_mouse_full_training_session.md`
- **Checkpoint Comparison**: `docs/research/251123_fauna_mouse_checkpoint_quality_comparison.md`
- **System Guide**: `docs/research/251121_3danimals_system_comprehensive_guide.md`

### Config 예제
- **Debug**: `config/train_fauna_mouse_dannce_debug.yaml`
- **Full**: `config/train_fauna_mouse_dannce.yaml`
- **Inference**: `config/infer_mouse_dannce.yaml`

---

**Last Updated**: 2025-11-24
**Maintainer**: Joon
**Next Update**: Config 기반 TF32 제어 구현 후
