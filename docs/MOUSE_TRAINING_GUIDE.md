# Mouse Training Guide

**Date**: 2025-11-24
**Dataset**: 50 frames at `data/fauna/Fauna_dataset/large_scale/mouse/train/seq_000/`
**GPU**: RTX 3060 12GB
**Training Strategy**: Few-shot learning from scratch

---

## Quick Start

### 1. Debug Mode (15-20 minutes)

**항상 먼저 디버그 모드로 검증하세요!**

```bash
conda run -n 3danimals python run.py --config-name train_mouse_debug
```

**예상 시간**: 5,000 iterations × 0.18s/iter = ~15-20분
**목적**: Config 오류 검증, 데이터 로딩 확인, GPU 메모리 체크

### 2. Full Training (2-3 hours)

디버그 모드가 성공하면 전체 학습 실행:

```bash
conda run -n 3danimals python run.py --config-name train_mouse
```

**예상 시간**: 50,000 iterations × 0.18s/iter = ~2.5시간
**체크포인트**: 5,000 iteration마다 저장 (총 10개)

### 3. Background Training (장시간 학습)

터미널을 닫아도 학습이 계속되도록:

```bash
nohup conda run -n 3danimals python run.py \
  --config-name train_mouse \
  > /tmp/mouse_training.log 2>&1 &

echo $!  # PID 확인 (프로세스 ID)
```

**모니터링**:
```bash
# 로그 실시간 확인
tail -f /tmp/mouse_training.log

# 진행 상황 확인
grep "Iter:" /tmp/mouse_training.log | tail -20

# GPU 사용량 확인
nvidia-smi
```

**중단**:
```bash
kill <PID>  # PID는 위에서 echo $! 로 확인
```

---

## Configuration Files

### Config 구조 (Hydra 기반)

```
config/
├── train_mouse.yaml          # 메인 설정 (50K iterations)
├── train_mouse_debug.yaml    # 디버그 설정 (5K iterations)
├── dataset/
│   └── mouse.yaml           # 데이터셋 설정
└── model/
    └── mouse.yaml           # 모델 설정
```

### 주요 설정 값

#### train_mouse.yaml (메인)
```yaml
defaults:
  - dataset: mouse
  - model: mouse

exp_name: mouse_50frames
num_iters: 50000              # 50K iterations (2-3 hours)
save_checkpoint_freq: 5000    # 30분마다 저장
log_image_freq: 500           # 15분마다 로그

device: cuda
gpu_ids: [0]
disable_tf32: true            # RTX 3060 필수 설정

wandb:
  project: mouse_fauna
  mode: online
  tags: [mouse, few_shot, 50frames]
```

#### dataset/mouse.yaml
```yaml
data_type: fauna
in_image_size: 256
out_image_size: 256
batch_size: 1                 # RTX 3060: batch 1 필수
num_workers: 2

# 경로 (자동 검색)
train_data_dir: null          # Auto: large_scale/mouse/ 자동 탐색
val_data_dir: null            # Auto: train에서 5장 split
test_data_dir: null           # Few-shot: train과 동일

# Augmentation (few-shot에서 비활성화)
random_shuffle_samples_train: false
random_xflip_train: false
```

#### model/mouse.yaml
```yaml
# 주요 설정
spatial_scale: 4.5            # 작은 동물 (쥐)
grid_res: 64                  # RTX 3060: 128은 OOM!
num_body_bones: 6             # 작은 척추 (짝수: 4, 6, 8)
num_legs: 4                   # 네발 동물
num_leg_bones: 3              # 다리 뼈 개수

# Articulation (관절 학습)
articulation_iter_range: [10000, inf]  # 10K부터 활성화
attach_legs_to_body_iter_range: [30000, inf]  # 30K부터 다리-몸통 결합

# Deformation (비활성화)
enable_deform: false          # Few-shot: 데이터 부족으로 비활성화
```

---

## Data Split 전략

### Few-shot 데이터셋의 Split

**현재 상황**: 50 frames만 존재
**FaunaDataset 동작**: 자동으로 내부 split

```python
# FaunaDataset.py의 동작
train_images: 45장 (0000000-0000044)
val_images: 5장 (0000045-0000049)  # val_num = 5
test_images: 50장 (전체)            # Few-shot: train과 동일
```

### ⚠️ Overfitting 주의

**문제**: Train과 test가 동일한 데이터 → **100% Overfitting 발생**

**해결책**:

1. **더 많은 데이터 수집** (권장)
   - 최소 100+ frames 필요
   - Train 70 / Val 10 / Test 20 비율
   - 다양한 자세와 각도 포함

2. **Few-shot 특성 이해**
   - 50 frames는 "proof of concept" 수준
   - Generalization은 기대하지 말 것
   - 새로운 자세는 재구성 실패할 수 있음

3. **Cross-validation** (차선책)
   - 5-fold cross-validation
   - 10장씩 5개 그룹으로 나누기
   - 각 fold에서 train/val/test 분리

### 데이터 추가 수집 가이드

더 나은 성능을 위해 다음과 같이 데이터 확장:

```bash
# 목표 구조
data/fauna/Fauna_dataset/large_scale/mouse/
├── train/
│   ├── seq_000/  # 기존 50 frames
│   ├── seq_001/  # 새로 추가 50 frames
│   ├── seq_002/  # 새로 추가 50 frames
│   └── seq_003/  # 새로 추가 50 frames
├── val/
│   └── seq_000/  # 새로 추가 20 frames
└── test/
    └── seq_000/  # 새로 추가 20 frames
```

**다양성 확보**:
- 다양한 자세 (서있기, 앉기, 걷기, 뛰기)
- 다양한 각도 (정면, 측면, 후면, 대각선)
- 다양한 조명 환경
- 배경 변화

---

## 학습 모니터링

### WandB 로그

학습 중 실시간 모니터링:

```bash
# 온라인 모드 (train_mouse.yaml 기본값)
wandb login  # 최초 1회만

# 오프라인 모드 (네트워크 없을 때)
# train_mouse.yaml에서 wandb.mode: offline 로 변경
```

**주요 메트릭**:
- `loss/mask_loss`: Mask 재구성 오류 (낮을수록 좋음)
- `loss/rgb_loss`: RGB 재구성 오류 (낮을수록 좋음)
- `metrics/mask_iou`: Mask IoU (높을수록 좋음, 목표: >0.9)
- `metrics/rgb_psnr`: RGB PSNR (높을수록 좋음, 목표: >20dB)

### 로컬 결과 확인

```bash
# 결과 디렉토리
cd results/mouse_50frames/

# 체크포인트 확인
ls -lh checkpoint*.pth

# 로그 이미지 확인 (시각화)
ls -lh log_images/
```

**저장되는 파일**:
- `checkpoint5000.pth`, `checkpoint10000.pth`, ... (모델 가중치)
- `log_images/iter_0500/`: 500 iteration마다 재구성 결과 이미지
- `final_model.pth`: 최종 모델

---

## Troubleshooting

### 1. CUDA Out of Memory (OOM)

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결**:
```yaml
# config/model/mouse.yaml
cfg_predictor_base:
  cfg_shape:
    grid_res: 32  # 64 → 32로 감소
```

**Trade-off**: 메모리 4GB → 1GB, 품질 약간 하락

### 2. 데이터 로딩 실패

**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../mouse/train/seq_000/0000000_rgb.png'
```

**확인**:
```bash
# 데이터 경로 확인
ls data/fauna/Fauna_dataset/large_scale/mouse/train/seq_000/ | head

# 파일 개수 확인
ls data/fauna/Fauna_dataset/large_scale/mouse/train/seq_000/*_rgb.png | wc -l
# 출력: 50 (정상)
```

**수정**: `config/dataset/mouse.yaml`에서 경로 확인

### 3. TF32 오류 (RTX 3060)

**증상**:
```
RuntimeError: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasGemmEx`
```

**해결**: 이미 설정되어 있어야 함
```yaml
# config/train_mouse.yaml
disable_tf32: true  # 반드시 true!
```

### 4. 학습이 너무 느림

**예상 속도**:
- RTX 3060: ~0.18s/iter → 50K iters = ~2.5시간
- RTX 3090: ~0.12s/iter → 50K iters = ~1.7시간

**느린 경우 확인**:
```bash
# GPU 사용률 확인
nvidia-smi

# 다른 프로세스 확인
ps aux | grep python

# CPU 병목 확인
htop
```

### 5. 재구성 품질이 나쁨

**원인**:
1. **데이터 부족** (50 frames는 매우 적음)
2. **자세 다양성 부족** (비슷한 자세만 반복)
3. **학습 부족** (50K iterations는 최소한)

**개선 방법**:
1. 데이터 추가 수집 (100+ frames)
2. 학습 시간 증가 (50K → 100K iterations)
3. 하이퍼파라미터 튜닝

---

## 참고 자료

### 프로젝트 문서
- `docs/FAUNA_DATASET_COMPLETE_GUIDE.md`: 전체 데이터셋 가이드
- `docs/reports/`: 실험 보고서 및 분석

### 관련 Config 파일
- `config/train_fauna_mouse_dannce.yaml`: DANNCE 버전 (참고용)
- `config/archive/deprecated_configs/`: 이전 버전들 (참고용)

### 데이터 생성 스크립트
- `scripts/generate_missing_files.py`: box.txt, metadata.json 생성
- `scripts/prepare_markerless_mouse_dataset.py`: 전체 전처리 (참고용)

---

## FAQ

### Q1: 50 frames로 학습 가능한가?
**A**: 가능하지만 품질 제한적. "Proof of concept" 수준. 실제 사용을 위해서는 100+ frames 권장.

### Q2: 다른 동물에도 사용 가능한가?
**A**: 가능. `config/model/mouse.yaml`를 복사하여 수정:
- `spatial_scale`: 동물 크기에 맞게 조정 (쥐: 4.5, 고양이: 6.0, 말: 10.0)
- `num_body_bones`: 척추 뼈 개수 (작은 동물: 6, 큰 동물: 8)

### Q3: 학습 시간을 단축하려면?
**A**: Debug 모드 사용 (5K iterations, 15분) 또는 `num_iters` 감소. 단, 품질 저하.

### Q4: Checkpoint에서 재시작하려면?
**A**:
```bash
conda run -n 3danimals python run.py \
  --config-name train_mouse \
  resume=results/mouse_50frames/checkpoint10000.pth
```

### Q5: 여러 GPU 사용하려면?
**A**:
```yaml
# config/train_mouse.yaml
gpu_ids: [0, 1]  # GPU 0, 1 사용
```
단, batch_size도 증가해야 효과적 (1 → 2 이상)
