# A6000 원본 논문 설정 학습 가이드

> **목적**: 3D Fauna, MagicPony, Ponymation을 원본 논문의 하이퍼파라미터대로 A6000에서 학습
>
> **작성일**: 2024-12-09
> **대상 GPU**: NVIDIA A6000 (48GB VRAM)

---

## 📋 개요

### 학습 시간 추정

| 모델 | Iterations | 예상 시간 | GPU 메모리 |
|------|-----------|----------|-----------|
| **3D Fauna** | 800K | 5-6일 | ~40GB |
| **MagicPony (Finetune)** | 100K | 1.5-2일 | ~30GB |
| **MagicPony (Scratch)** | 140K | 2-3일 | ~30GB |
| **Ponymation Stage 1** | 280K | 3-4일 | ~35GB |
| **Ponymation Stage 2** | 500K | 4-5일 | ~20GB |

**총 예상 시간**: 약 2주 (순차 실행 시)

### 원본 논문 vs 이전 설정 비교

| 항목 | 원본 논문 | 이전 설정 (RTX 3080Ti) | 차이점 |
|------|----------|----------------------|--------|
| grid_res | 256 | 64 | 4배 감소 |
| Iterations | 800K | 100K | 8배 감소 |
| batch_size | 6-8 | 4-6 | 감소 |
| deform 활성화 | 600-800K | 비활성화 | 기능 제한 |

---

## 🚀 빠른 시작

### 1. 환경 확인

```bash
# GPU 확인
nvidia-smi

# 예상 출력:
# NVIDIA A6000 | 48GB | ...
```

### 2. 데이터 준비

```bash
# Mouse 6-view 데이터 확인
ls data/fauna/mouse_6view_posesplatter/

# MagicPony용 데이터 변환 (필요시)
python scripts/convert_fauna_to_magicpony.py \
    --source data/fauna_mouse/large_scale/mouse_dannce_6view \
    --target data/magicpony/mouse --copy

# Ponymation용 시퀀스 데이터 확인
ls data/ponymation/mouse/train/
```

### 3. Pretrained 모델 다운로드

```bash
# MagicPony pretrained horse
cd results/magicpony
bash download_pretrained_magicpony.sh
cd ../..

# 확인
ls results/magicpony/pretrained_horse/
# → pretrained_horse.pth
```

---

## 📦 설정 파일 목록

### 생성된 A6000 설정 파일

```
config/
├── model/
│   ├── fauna_mouse_a6000.yaml           # Fauna 모델 설정
│   ├── magicpony_mouse_a6000.yaml       # MagicPony 모델 설정
│   └── ponymation_mouse_a6000.yaml      # Ponymation 모델 설정
│
├── train_fauna_mouse_a6000.yaml         # Fauna 학습 (800K)
├── train_magicpony_mouse_a6000.yaml     # MagicPony Finetune (100K)
├── train_magicpony_mouse_scratch_a6000.yaml  # MagicPony Scratch (140K)
├── train_ponymation_mouse_stage1_a6000.yaml  # Ponymation Stage 1 (280K)
└── train_ponymation_mouse_stage2_a6000.yaml  # Ponymation Stage 2 (500K)
```

### 백업된 기존 설정

```
config/backup_20251209/
├── fauna_mouse_6view.yaml
├── finetune_magicpony_mouse.yaml
├── magicpony_mouse_finetune.yaml
└── train_fauna_mouse_6view.yaml
```

---

## 🔧 학습 실행 방법

### Option A: 3D Fauna (800K iterations)

```bash
# 학습 시작 (GPU 1 사용 - A6000 서버 기본값)
python run.py --config-name train_fauna_mouse_a6000

# 백그라운드 실행
nohup python run.py --config-name train_fauna_mouse_a6000 \
    > logs/fauna_a6000.log 2>&1 &

# 로그 모니터링
tail -f logs/fauna_a6000.log
```

**예상 결과 디렉토리**:
```
results/fauna_mouse_a6000/
├── ckpt-*.pth           # 체크포인트 (10K 간격)
├── train_results/       # 학습 중 시각화
└── config.yaml          # 사용된 설정
```

### Option B: MagicPony Finetune (권장)

```bash
# Pretrained horse에서 시작
python run.py --config-name train_magicpony_mouse_a6000

# From scratch 학습 (비교용)
python run.py --config-name train_magicpony_mouse_scratch_a6000
```

### Option C: Ponymation (2-Stage)

```bash
# Stage 1: Articulation 학습 (280K)
python run.py --config-name train_ponymation_mouse_stage1_a6000

# Stage 1 완료 후, Stage 2 설정 수정 필요
# config/train_ponymation_mouse_stage2_a6000.yaml에서:
# checkpoint_path: results/ponymation/mouse_stage1_a6000/ckpt-280000.pth

# Stage 2: Motion VAE 학습 (500K)
python run.py --config-name train_ponymation_mouse_stage2_a6000
```

---

## 📊 학습 모니터링

### WandB 대시보드

```bash
# WandB 로그인 (최초 1회)
wandb login

# 대시보드 URL
# 3D Fauna: https://wandb.ai/{username}/Fauna_mouse_a6000
# MagicPony: https://wandb.ai/{username}/MagicPony_mouse_a6000
# Ponymation: https://wandb.ai/{username}/Ponymation_mouse_a6000
```

### 주요 Loss 지표

| Loss | 정상 수렴 패턴 | 문제 징후 |
|------|---------------|----------|
| `total_loss` | 점진적 감소 | 발산, 진동 |
| `mask_loss` | 초기 높음 → 0.01 이하 | 감소 안함 |
| `rgb_loss` | 0.1 → 0.02~0.05 | 0.1 이상 유지 |
| `dino_feat_im_loss` | 점진적 감소 | 증가 |

### 학습 진행 체크포인트

| Iteration | 예상 상태 |
|-----------|----------|
| 10K | 기본 형태 잡힘 (타원체 → 동물 형상) |
| 50K | Articulation 활성화, 관절 움직임 |
| 100K | 세부 형태 개선 |
| 300K | Texture 품질 향상 |
| 600K+ | Deformation 활성화, 미세 조정 |
| 800K | 최종 수렴 (Fauna 기준) |

---

## 🔍 결과 확인

### 체크포인트에서 Mesh 추출

```bash
# 테스트 실행
python run.py --config-name train_fauna_mouse_a6000 \
    run_train=false run_test=true \
    checkpoint_name=ckpt-800000.pth

# 결과 확인
ls results/fauna_mouse_a6000/test_results/
```

### 시각화

```bash
# 학습 중 저장된 결과
ls results/fauna_mouse_a6000/train_results/

# 구조:
# iter_XXXXXX/
#   ├── mesh.obj        # 3D 메시
#   ├── render.png      # 렌더링 이미지
#   └── ...
```

---

## ⚠️ 트러블슈팅

### CUDA Out of Memory

```bash
# 증상: torch.cuda.OutOfMemoryError

# 해결책 1: batch_size 감소
python run.py --config-name train_fauna_mouse_a6000 \
    dataset.batch_size=4

# 해결책 2: grid_res 감소 (품질 저하)
python run.py --config-name train_fauna_mouse_a6000 \
    model.cfg_predictor_base.cfg_shape.grid_res=128
```

### 학습 중단 후 재개

```bash
# resume=true가 기본 설정이므로 동일 명령어로 재개
python run.py --config-name train_fauna_mouse_a6000 gpu=0

# 특정 체크포인트에서 재개
python run.py --config-name train_fauna_mouse_a6000 \
    checkpoint_name=ckpt-500000.pth
```

### WandB 연결 문제

```bash
# 오프라인 모드로 실행
python run.py --config-name train_fauna_mouse_a6000 \
    use_logger=false

# 또는 텐서보드 사용
python run.py --config-name train_fauna_mouse_a6000 \
    logger_type=tensorboard
```

---

## 📈 정량적 평가 기준

### 수렴 판단 기준

| 메트릭 | 양호 | 보통 | 불량 |
|--------|------|------|------|
| Mask IoU | > 0.85 | 0.7-0.85 | < 0.7 |
| PSNR (dB) | > 25 | 20-25 | < 20 |
| total_loss | < 10 | 10-50 | > 50 |

### From Scratch vs Finetune 결정 기준

**Finetune 선택 조건**:
- Pretrained 모델의 동물과 체형이 유사할 때 (horse → 4족 동물)
- 빠른 실험 반복이 필요할 때
- 데이터셋 크기가 작을 때

**From Scratch 선택 조건**:
- 완전히 새로운 동물 종일 때
- Pretrained 모델과 체형이 크게 다를 때
- 충분한 학습 시간과 데이터가 있을 때

---

## 📚 참고 자료

### 원본 논문

- [3D-Fauna (CVPR 2024)](https://arxiv.org/abs/2401.02400): 800K iterations, A40 GPU
- [MagicPony (CVPR 2023)](https://arxiv.org/abs/2211.12497): 140K iterations
- [Ponymation](https://arxiv.org/abs/2312.13604): Stage 1 (280K) + Stage 2 (500K)

### 관련 문서

- GPU05 학습 가이드: `docs/guides/gpu05_training_guide.md`
- 데이터셋 준비: `docs/guides/dataset_preparation.md`

---

## 📝 변경 이력

| 날짜 | 내용 |
|------|------|
| 2024-12-09 | 초기 문서 작성, A6000 설정 파일 생성 |
