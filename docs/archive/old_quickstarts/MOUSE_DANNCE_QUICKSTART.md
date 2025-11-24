# Mouse DANNCE Quick Start

## ✅ 완료된 작업
- Debug 학습 완료 (3,000 iterations)
- 체크포인트 저장: `results/checkpoint3000.pth`
- CUDA 11.8 + PyTorch 2.0.0+cu118 환경 구성

---

## 🎯 추론 실행 (시각화 생성)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# Config 파일이 이제 올바른 위치에 있습니다
python run_debug_notf32.py --config-name infer_mouse_dannce
```

**중요**: `run.py`가 아닌 `run_debug_notf32.py`를 사용해야 TF32가 비활성화되어 CUBLAS 오류를 방지합니다!

---

## 🚀 Full Training 실행

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# 백그라운드 실행 (TF32 disabled)
nohup python run_full_notf32.py > /tmp/mouse_full.log 2>&1 &

# 로그 실시간 확인
tail -f /tmp/mouse_full.log
```

**예상 시간**: ~3시간 (50,000 iterations)

---

## ⚠️ 중요 사항

### TF32 비활성화 필수!

**올바른 명령어:**
```bash
python run_debug_notf32.py --config-name <config_name>
python run_full_notf32.py
```

**잘못된 명령어 (CUBLAS 오류 발생!):**
```bash
python run.py --config-name <config_name>  # ❌ TF32 활성화됨
```

### 왜 TF32를 비활성화해야 하나?
- PyTorch 2.0 + Ampere GPU (RTX 3060)에서 DINO ViT 사용 시
- CUBLAS_STATUS_NOT_SUPPORTED 오류 발생
- TF32 비활성화로 해결됨

---

## 📊 모니터링

### 학습 진행 상황
```bash
# Loss 확인
tail -f /tmp/mouse_full.log | grep "loss:"

# GPU 사용량
watch -n 1 nvidia-smi

# 프로세스 확인
ps aux | grep run_full_notf32
```

### Tensorboard
```bash
tensorboard --logdir results/tensorboard_logs --port 6006
# http://localhost:6006
```

---

## 🔧 문제 해결

### CUBLAS 오류 재발생
```bash
# CUDA 테스트 실행
python test_cuda_fix.py

# TF32 비활성화 확인
python run_debug_notf32.py  # 시작 시 "TF32 DISABLED" 메시지 확인
```

### Config 파일 못 찾음
```bash
# Config 파일 위치 확인
ls config/infer_mouse_dannce.yaml

# 올바른 위치로 이동 (필요시)
mv infer_mouse_dannce.yaml config/
```

---

## 📁 주요 파일 위치

```
/home/joon/dev/3DAnimals/
├── run_debug_notf32.py          # Debug 학습/추론 (TF32 OFF)
├── run_full_notf32.py            # Full 학습 (TF32 OFF)
├── run.py                        # 원본 (TF32 ON) - 사용 금지!
├── test_cuda_fix.py              # CUDA 테스트
├── config/
│   ├── infer_mouse_dannce.yaml       # 추론 config
│   ├── train_fauna_mouse_dannce_debug.yaml
│   └── train_fauna_mouse_dannce.yaml  # Full training config
└── results/
    └── checkpoint3000.pth        # Debug 학습 결과
```

---

## 실행 예시

### 1. 추론 (시각화)
```bash
conda activate 3danimals
cd /home/joon/dev/3DAnimals
python run_debug_notf32.py --config-name infer_mouse_dannce
```

### 2. Full Training
```bash
conda activate 3danimals
cd /home/joon/dev/3DAnimals
nohup python run_full_notf32.py > /tmp/mouse_full.log 2>&1 &
echo "Training started with PID: $!"
```

### 3. 진행 상황 확인
```bash
# 최근 100줄
tail -100 /tmp/mouse_full.log

# Loss만 필터링
grep "loss:" /tmp/mouse_full.log | tail -20
```
