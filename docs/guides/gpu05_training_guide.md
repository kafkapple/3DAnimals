# GPU05 Training Guide

> SSH 원격 서버 (gpu05) 학습 환경 가이드

---

## 환경 정보

| 항목 | 값 |
|------|-----|
| **Server** | gpu05 |
| **GPU** | NVIDIA RTX A6000 x2 (49GB each) |
| **Conda Env** | `3danimals` |
| **Project Path** | `~/3DAnimals` |
| **Local Project** | `/home/joon/dev/3DAnimals` |

---

## 필수 사전 작업

### 1. Git 동기화 (로컬 → gpu05)

**항상 학습 전 동기화 필수!**

```bash
# 로컬에서 커밋 & 푸시
git add -A && git commit -m "your message" && git push origin main

# gpu05에서 풀
ssh gpu05 "cd ~/3DAnimals && git pull origin main"
```

### 2. 데이터 확인

```bash
# 데이터 존재 확인
ssh gpu05 "ls ~/3DAnimals/data/ponymation/mouse/train"
ssh gpu05 "ls ~/3DAnimals/data/magicpony/mouse/train"
```

### 3. Pretrained 모델 확인

```bash
# MagicPony pretrained horse
ssh gpu05 "ls ~/3DAnimals/results/magicpony/pretrained_horse/"

# Ponymation pretrained horse
ssh gpu05 "ls ~/3DAnimals/results/ponymation/pretrained_horse/"
```

---

## 학습 명령어

### MagicPony

```bash
# Finetune (horse → mouse)
ssh gpu05 "./scripts/train_unified.sh magicpony full finetune"

# Debug (빠른 검증)
ssh gpu05 "./scripts/train_unified.sh magicpony debug finetune"
```

### Ponymation

```bash
# Stage 1: Articulation 학습
ssh gpu05 "./scripts/train_unified.sh ponymation-s1 full finetune"

# Stage 2: Motion VAE 학습 (Stage 1 완료 후)
ssh gpu05 "./scripts/train_unified.sh ponymation-s2 full finetune"
```

---

## Resume (중단 후 재시작)

**동일 명령어로 재실행하면 자동으로 이어서 학습**

```bash
# checkpoint_dir에 .pth 파일이 있으면 자동 감지
ssh gpu05 "./scripts/train_unified.sh ponymation-s1 full finetune"
# → [Resume] Found existing checkpoint, continuing training
```

### Resume 동작 방식

| 상황 | 로그 메시지 | 동작 |
|------|------------|------|
| 첫 실행 | `[Transfer] Loading from pretrained` | iter 0부터 시작 |
| 재실행 | `[Resume] Found existing checkpoint` | 저장된 iter부터 이어서 |

---

## 체크포인트 관리

### 저장 위치

```
results/
├── magicpony/
│   ├── mouse_finetune/          # MagicPony finetune 결과
│   │   ├── checkpoint2000.pth
│   │   ├── checkpoint4000.pth
│   │   └── ...
│   └── pretrained_horse/        # 원본 (삭제 금지!)
│
└── ponymation/
    ├── mouse_finetune_stage1/   # Stage 1 결과
    ├── mouse_finetune_stage2/   # Stage 2 결과
    └── pretrained_horse/        # 원본 (삭제 금지!)
```

### 체크포인트 확인

```bash
# Stage 1 체크포인트 확인
ssh gpu05 "ls -lh ~/3DAnimals/results/ponymation/mouse_finetune_stage1/*.pth"

# Stage 2 체크포인트 확인
ssh gpu05 "ls -lh ~/3DAnimals/results/ponymation/mouse_finetune_stage2/*.pth"
```

### 체크포인트 삭제 (재학습 시)

```bash
# 주의: pretrained는 삭제하지 않음!
ssh gpu05 "rm -rf ~/3DAnimals/results/ponymation/mouse_*"
ssh gpu05 "rm -rf ~/3DAnimals/results/magicpony/mouse_*"
```

---

## 모니터링

### tmux 세션 사용 (권장)

```bash
# 새 세션 생성
ssh gpu05 "tmux new -s ponymation"

# 학습 실행 후 detach: Ctrl+B, D

# 세션 재접속
ssh gpu05 "tmux attach -t ponymation"

# 세션 목록
ssh gpu05 "tmux ls"
```

### GPU 사용량 확인

```bash
ssh gpu05 "nvidia-smi"
ssh gpu05 "watch -n 1 nvidia-smi"  # 실시간
```

### 프로세스 확인

```bash
ssh gpu05 "ps aux | grep python | grep -v grep"
```

### Wandb 로그

- Project: `MagicPony_mouse` / `Ponymation_mouse`
- URL: https://wandb.ai/kafkapple-joon-kaist/

---

## 테스트 (추론)

### MagicPony

```bash
ssh gpu05 "cd ~/3DAnimals && conda run -n 3danimals python run.py \
  --config-name test_magicpony_mouse \
  checkpoint_dir=results/magicpony/mouse_finetune \
  checkpoint_name=checkpoint100000.pth"
```

### Ponymation

```bash
# Stage 2 모델로 테스트
ssh gpu05 "cd ~/3DAnimals && conda run -n 3danimals python run.py \
  --config-name test_ponymation_mouse \
  checkpoint_dir=results/ponymation/mouse_finetune_stage2 \
  checkpoint_name=checkpoint200000.pth"
```

---

## 주의사항

### 1. Config 수정 후 반드시 git 동기화

```bash
# 로컬 수정 → 푸시 → gpu05 풀
git push origin main
ssh gpu05 "cd ~/3DAnimals && git pull origin main"
```

### 2. 실행 중인 학습 중단 방법

```bash
# tmux 세션에서 Ctrl+C
# 또는 프로세스 kill
ssh gpu05 "pkill -f 'python run.py'"
```

### 3. Mouse 설정 핵심값

| 파라미터 | Mouse | Horse |
|----------|-------|-------|
| `num_body_bones` | 6 | 8 |
| `legs_to_body_joint_indices` | [1,5,5,1] | [2,7,7,2] |
| `spatial_scale` | 5.0 | 7.0 |
| `extra_constraints` | false | true |

### 4. Stage 간 의존성

```
pretrained_horse.pth
       ↓
[Stage 1] → checkpoint100000.pth (100K iters)
       ↓
[Stage 2] → checkpoint200000.pth (200K iters)
```

- Stage 2는 반드시 Stage 1 완료 후 실행
- `train_unified.sh`가 자동으로 Stage 1 checkpoint 탐지

### 5. 흔한 에러와 해결

| 에러 | 원인 | 해결 |
|------|------|------|
| `size mismatch (20 vs 18)` | Horse config 사용 | mouse config 확인 |
| `does not require grad` | set_train() 누락 | git pull로 최신 코드 반영 |
| `checkpoint not found` | Stage 1 미완료 | Stage 1 먼저 완료 |
| `+checkpoint_path` 에러 | Hydra override | `++` 사용 |

---

## 학습 설정 요약

| 모델 | Config | Iterations | 예상 시간 |
|------|--------|------------|----------|
| MagicPony finetune | `finetune_magicpony_mouse` | 100K | 8-12h |
| Ponymation S1 | `finetune_ponymation_mouse_stage1` | 100K | 10-15h |
| Ponymation S2 | `finetune_ponymation_mouse_stage2` | 200K | 15-24h |

---

## Quick Reference

```bash
# 전체 학습 파이프라인
ssh gpu05 "./scripts/train_unified.sh magicpony full finetune"
ssh gpu05 "./scripts/train_unified.sh ponymation-s1 full finetune"
ssh gpu05 "./scripts/train_unified.sh ponymation-s2 full finetune"

# 상태 확인
ssh gpu05 "nvidia-smi"
ssh gpu05 "ls ~/3DAnimals/results/ponymation/mouse_*/*.pth"

# 코드 동기화
ssh gpu05 "cd ~/3DAnimals && git pull origin main"
```
