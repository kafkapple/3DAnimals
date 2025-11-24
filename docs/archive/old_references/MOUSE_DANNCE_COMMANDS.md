# Mouse DANNCE Training & Inference Commands

## 현재 상태
- ✅ Debug 학습 완료 (3,000 iterations)
- ✅ 체크포인트 저장: `results/checkpoint3000.pth`
- ✅ CUDA 11.8 환경 구성 완료 (PyTorch 2.0.0+cu118)

---

## 1. 시각화 생성 (추론 실행)

### 방법 1: Config 파일 사용 (권장)

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# 추론 실행 (시각화 생성)
python run_debug_notf32.py --config-name infer_mouse_dannce
```

Config 파일 위치: `config/infer_mouse_dannce.yaml`

### 방법 2: 커맨드라인 오버라이드

```bash
python run_debug_notf32.py \
  --config-name train_fauna_mouse_dannce_debug \
  resume=results/checkpoint3000.pth \
  run_train=false \
  run_test=true
```

### 예상 결과
- 출력 위치: `results/fauna_mouse_dannce_infer/`
- 생성 파일:
  - `*_rgb.png`: 입력 이미지
  - `*_mask.png`: 마스크
  - `*_rendered.png`: 렌더링 결과
  - `*.obj`: 3D 메쉬 (있을 경우)

---

## 2. Full Training 실행 (50K iterations, ~3시간)

### 실행 명령어

```bash
cd /home/joon/dev/3DAnimals
conda activate 3danimals

# 백그라운드 실행
nohup conda run -n 3danimals python run_full_notf32.py \
  > /tmp/mouse_dannce_full.log 2>&1 &

# 프로세스 ID 확인
echo $!
```

### 학습 모니터링

```bash
# 실시간 로그 확인
tail -f /tmp/mouse_dannce_full.log

# Loss 값만 확인
tail -f /tmp/mouse_dannce_full.log | grep "loss:"

# 프로세스 상태 확인
ps aux | grep run_full_notf32

# GPU 사용량 확인
nvidia-smi
```

### 예상 진행 상황
- **총 Iterations**: 50,000
- **예상 시간**: ~2.5-3시간 (5.1 Hz 기준)
- **체크포인트**: 10K, 20K, 30K, 40K, 50K iterations마다 저장
- **메모리 사용량**: ~4GB VRAM (grid_res=64)

---

## 3. 학습 중단 및 재개

### 중단
```bash
# 프로세스 찾기
ps aux | grep run_full_notf32

# 종료 (PID 확인 후)
kill <PID>
```

### 재개
```bash
# 마지막 체크포인트에서 재개
python run_full_notf32.py \
  resume=results/checkpoint40000.pth  # 예시
```

---

## 4. 결과 확인

### 체크포인트 위치
```bash
ls -lh /home/joon/dev/3DAnimals/results/checkpoint*.pth
```

### Tensorboard 실행
```bash
cd /home/joon/dev/3DAnimals
tensorboard --logdir results/tensorboard_logs --port 6006

# 브라우저에서: http://localhost:6006
```

### WandB 동기화 (필요시)
```bash
# Offline 모드였던 로그를 온라인으로 동기화
wandb sync results/tensorboard_logs/
```

---

## 5. Config 파일 위치

```
config/
├── dataset/
│   └── fauna_mouse_dannce.yaml          # 데이터셋 설정
├── model/
│   └── fauna_mouse_dannce.yaml          # 모델 설정
├── train_fauna_mouse_dannce_debug.yaml  # Debug 학습
├── train_fauna_mouse_dannce.yaml        # Full 학습
└── infer_mouse_dannce.yaml              # 추론 설정
```

---

## 6. 환경 정보

```bash
# PyTorch 버전 확인
conda run -n 3danimals python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

# 예상 출력:
# PyTorch: 2.0.0+cu118
# CUDA: 11.8

# CUDA 테스트
conda run -n 3danimals python test_cuda_fix.py
```

---

## 7. 문제 해결

### CUBLAS 오류 재발생 시
```bash
# TF32 비활성화 확인
python run_debug_notf32.py  # TF32 disabled 메시지 확인 필요
```

### OOM (Out of Memory) 오류 시
```bash
# grid_res 감소 (config/model/fauna_mouse_dannce.yaml)
# grid_res: 64 → 32

# 또는 batch_size 감소 (config/dataset/fauna_mouse_dannce.yaml)
# batch_size: 1 (이미 최소값)
```

### 학습 속도가 느릴 경우
```bash
# GPU 사용 확인
nvidia-smi

# 다른 프로세스가 GPU 사용 중인지 확인
# 필요시 다른 프로세스 종료
```

---

## 8. 데이터셋 정보

- **위치**: `/home/joon/dev/3DAnimals/data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view`
- **프레임 수**: 50 frames (5 sequences × 10 frames)
- **해상도**: 256×256
- **카메라 뷰**: 6 views (multi-view)

---

## 참고 자료

- 프로젝트 README: `/home/joon/dev/3DAnimals/README.md`
- Fauna 데이터셋 가이드: `/home/joon/dev/3DAnimals/docs/FAUNA_DATASET_PREPARATION_GUIDE.md`
- 시스템 가이드: `/home/joon/dev/3DAnimals/docs/reports/251121_3danimals_system_comprehensive_guide.md`
