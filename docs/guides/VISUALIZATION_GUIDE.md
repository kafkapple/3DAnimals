# 시각화 결과 확인 가이드

## 🎯 Full Training 시작됨!

**시작 시간**: 2025-11-22 15:16 KST
**예상 완료**: ~2-3시간 후 (17:30-18:30)
**Iterations**: 50,000
**Log 파일**: `/tmp/fauna_full_training.log`

---

## 📊 실시간 모니터링

### 1. 학습 로그 확인

```bash
# 실시간 로그 (전체)
tail -f /tmp/fauna_full_training.log

# Loss만 확인
tail -f /tmp/fauna_full_training.log | grep "loss:"

# 특정 iteration만 확인
tail -f /tmp/fauna_full_training.log | grep "T0"

# 최근 100줄 확인
tail -100 /tmp/fauna_full_training.log
```

### 2. GPU 사용량 모니터링

```bash
# 실시간 GPU 상태
watch -n 1 nvidia-smi

# 또는
nvidia-smi -l 1

# 간단한 확인
nvidia-smi
```

### 3. 프로세스 상태 확인

```bash
# Training 프로세스 확인
ps aux | grep run_full_notf32

# PID 확인
pgrep -af run_full_notf32

# 프로세스 종료 (필요시)
pkill -f run_full_notf32
```

---

## 📁 시각화 결과 위치

### 추론 결과 (Inference)

**위치**: `/home/joon/dev/3DAnimals/results/mouse_dannce_infer/test_results_checkpoint3000/`

**파일 구조**:
```
test_results_checkpoint3000/
├── 0003000_0_image_gt.png          # Ground truth 입력 이미지
├── 0003000_0_image_pred.png        # 모델 예측 렌더링
├── 0003000_0_mask_gt.png           # Ground truth 마스크
├── 0003000_0_mask_pred.png         # 모델 예측 마스크
├── 0003000_0_mesh.obj              # 3D 메시 (Wavefront OBJ)
├── 0003000_0_pose.txt              # 카메라 포즈
└── ... (198 frames total)
```

**확인 방법**:
```bash
cd /home/joon/dev/3DAnimals/results/mouse_dannce_infer/test_results_checkpoint3000

# 이미지 뷰어로 보기
eog *_image_gt.png *_image_pred.png

# 특정 프레임 비교
eog 0003000_0_image_gt.png 0003000_0_image_pred.png

# 마스크 확인
eog *_mask*.png
```

### 3D 메시 확인

**방법 1: Blender** (권장)
```bash
cd /home/joon/dev/3DAnimals/results/mouse_dannce_infer/test_results_checkpoint3000

# Blender로 열기
blender 0003000_0_mesh.obj

# 여러 메시 동시에
blender 0003000_*.obj
```

**방법 2: MeshLab**
```bash
# MeshLab 설치 (없으면)
sudo apt install meshlab

# 메시 확인
meshlab 0003000_0_mesh.obj
```

**방법 3: Python (간단 확인)**
```python
import trimesh

# 메시 로드
mesh = trimesh.load('0003000_0_mesh.obj')

# 정보 출력
print(f"Vertices: {len(mesh.vertices)}")
print(f"Faces: {len(mesh.faces)}")

# 시각화
mesh.show()
```

---

## 📈 WandB (Weights & Biases)

### 현재 상태

**Config 설정**:
```yaml
wandb:
  mode: online  # Online logging enabled
  project: fauna_mouse_dannce
```

### 접근 방법

**1. WandB Dashboard**
```bash
# 브라우저에서
https://wandb.ai/[your-username]/fauna_mouse_dannce
```

**2. 로컬 WandB 확인**
```bash
# WandB sync 디렉토리
ls -la /home/joon/dev/3DAnimals/wandb/

# 최신 run 확인
ls -lat /home/joon/dev/3DAnimals/wandb/ | head -10
```

**3. Offline 모드였던 경우 동기화**
```bash
# Offline 로그를 온라인으로 업로드
cd /home/joon/dev/3DAnimals
wandb sync wandb/latest-run  # latest-run 디렉토리 이름 확인 필요
```

### WandB에서 확인 가능한 것

- **Loss curves**: mask_loss, rgb_loss, sdf_loss 등
- **Images**: 주기적으로 로깅되는 예측 결과
- **Metrics**: 학습 진행 상황
- **System**: GPU 사용률, 메모리 사용량

---

## 📊 Tensorboard (대안)

WandB가 작동하지 않으면 Tensorboard 사용:

### 실행 방법

```bash
cd /home/joon/dev/3DAnimals

# Tensorboard 시작
tensorboard --logdir results/fauna_mouse_dannce_from_scratch/tensorboard_logs --port 6006

# 브라우저에서 접속
http://localhost:6006
```

### 포트 변경 (6006 사용 중일 경우)
```bash
tensorboard --logdir results/fauna_mouse_dannce_from_scratch/tensorboard_logs --port 6007
```

### SSH로 원격 접속 시
```bash
# 로컬 머신에서
ssh -L 6006:localhost:6006 user@remote-server

# 로컬 브라우저에서
http://localhost:6006
```

---

## 🗂️ Checkpoint 위치

**저장 위치**: `/home/joon/dev/3DAnimals/results/fauna_mouse_dannce_from_scratch/`

**파일 구조**:
```
fauna_mouse_dannce_from_scratch/
├── checkpoint5000.pth          # 5K iterations
├── checkpoint10000.pth         # 10K iterations
├── checkpoint15000.pth         # 15K iterations
├── checkpoint20000.pth         # 20K iterations
├── checkpoint25000.pth         # 25K iterations
├── checkpoint30000.pth         # 30K iterations
├── checkpoint35000.pth         # 35K iterations
├── checkpoint40000.pth         # 40K iterations
├── checkpoint45000.pth         # 45K iterations
├── checkpoint50000.pth         # 50K iterations (최종)
├── archived_code.zip           # 코드 백업
├── metrics.json                # 메트릭 기록
└── tensorboard_logs/           # Tensorboard 로그
```

**확인 방법**:
```bash
# Checkpoint 생성 확인
ls -lh /home/joon/dev/3DAnimals/results/fauna_mouse_dannce_from_scratch/*.pth

# 최신 checkpoint 확인
ls -lt /home/joon/dev/3DAnimals/results/fauna_mouse_dannce_from_scratch/*.pth | head -3

# 용량 확인
du -sh /home/joon/dev/3DAnimals/results/fauna_mouse_dannce_from_scratch/
```

---

## 🎬 학습 중 시각화 (Log Images)

### 위치

**Tensorboard/WandB**: 자동 로깅 (매 500 iterations)

**로컬 파일** (있다면):
```
results/fauna_mouse_dannce_from_scratch/training_results/
```

### 확인 시점

- **500 iters**: 첫 시각화
- **1000 iters**: 초기 학습 진행
- **5000 iters**: 첫 checkpoint + 시각화
- **10000 iters**: Articulation 활성화 시점
- **30000 iters**: 다리 부착 시점
- **50000 iters**: 최종 결과

---

## ⏱️ 진행 상황 예상

### Timeline

| Time | Iteration | Milestone |
|------|-----------|-----------|
| 0:00 | 0 | 시작 |
| 0:15 | ~3,000 | 초기 학습 |
| 0:30 | ~6,000 | 첫 checkpoint |
| 1:00 | ~12,000 | Articulation 시작 |
| 1:30 | ~18,000 | Shape 학습 중 |
| 2:00 | ~24,000 | Mid-training |
| 2:30 | ~30,000 | 다리 부착 |
| 3:00 | ~36,000 | 세부 조정 |
| 3:30 | ~42,000 | Fine-tuning |
| 4:00 | ~48,000 | 거의 완료 |
| 4:15 | 50,000 | ✅ 완료! |

**예상 완료 시각**: **~18:30-19:00**

---

## 🔍 품질 확인 체크리스트

### 학습 중 (매 5K checkpoint)

```bash
# 1. Loss 감소 확인
tail -f /tmp/fauna_full_training.log | grep "loss:"

# 2. Checkpoint 생성 확인
ls -lht results/fauna_mouse_dannce_from_scratch/*.pth | head -5

# 3. GPU 사용률 (90-100% 정상)
nvidia-smi

# 4. 프로세스 살아있는지
ps aux | grep run_full_notf32
```

### 학습 완료 후

```bash
# 1. 최종 checkpoint 확인
ls -lh results/fauna_mouse_dannce_from_scratch/checkpoint50000.pth

# 2. 추론 실행
python run_debug_notf32.py --config-name infer_mouse_dannce \
  resume=results/fauna_mouse_dannce_from_scratch/checkpoint50000.pth

# 3. 결과 비교
cd results/mouse_dannce_infer/test_results_checkpoint50000
eog *_image*.png

# 4. 3D 메시 확인
blender *.obj
```

---

## 🚨 문제 해결

### Training이 멈춘 것 같을 때

```bash
# 1. 프로세스 확인
ps aux | grep run_full_notf32

# 2. 로그 마지막 확인
tail -20 /tmp/fauna_full_training.log

# 3. GPU 사용 확인
nvidia-smi

# 4. 에러 확인
grep -i "error\|exception\|fail" /tmp/fauna_full_training.log
```

### OOM (Out of Memory) 발생 시

```bash
# 현재 메모리 사용량
nvidia-smi

# Config에서 grid_res 감소
# config/model/fauna_mouse_dannce.yaml
# grid_res: 64 → 32
```

### CUDA 에러 발생 시

```bash
# TF32 비활성화 확인
grep "TF32" /tmp/fauna_full_training.log

# run_full_notf32.py 사용 확인 (run.py 아님!)
```

---

## 📸 결과 시각화 스크립트

### Python으로 여러 프레임 비교

```python
import matplotlib.pyplot as plt
from PIL import Image
import glob

# 결과 디렉토리
result_dir = "results/mouse_dannce_infer/test_results_checkpoint50000"

# 여러 프레임 로드
frames = [0, 10, 20, 30, 40]

fig, axes = plt.subplots(len(frames), 4, figsize=(16, 4*len(frames)))

for i, frame_id in enumerate(frames):
    # GT image
    img_gt = Image.open(f"{result_dir}/0050000_{frame_id:8d}_image_gt.png")
    axes[i, 0].imshow(img_gt)
    axes[i, 0].set_title(f"Frame {frame_id} - GT")
    axes[i, 0].axis('off')

    # Pred image
    img_pred = Image.open(f"{result_dir}/0050000_{frame_id:8d}_image_pred.png")
    axes[i, 1].imshow(img_pred)
    axes[i, 1].set_title(f"Frame {frame_id} - Pred")
    axes[i, 1].axis('off')

    # Mask GT
    mask_gt = Image.open(f"{result_dir}/0050000_{frame_id:8d}_mask_gt.png")
    axes[i, 2].imshow(mask_gt, cmap='gray')
    axes[i, 2].set_title(f"Frame {frame_id} - Mask GT")
    axes[i, 2].axis('off')

    # Mask Pred
    mask_pred = Image.open(f"{result_dir}/0050000_{frame_id:8d}_mask_pred.png")
    axes[i, 3].imshow(mask_pred, cmap='gray')
    axes[i, 3].set_title(f"Frame {frame_id} - Mask Pred")
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
plt.show()
```

---

## 📋 빠른 명령어 참조

```bash
# === 학습 모니터링 ===
tail -f /tmp/fauna_full_training.log                                    # 실시간 로그
tail -f /tmp/fauna_full_training.log | grep "T0"                        # Iteration만
nvidia-smi                                                               # GPU 상태
ps aux | grep run_full_notf32                                           # 프로세스 확인

# === Checkpoint 확인 ===
ls -lht results/fauna_mouse_dannce_from_scratch/*.pth | head -5        # 최신 5개
watch -n 60 'ls -lh results/fauna_mouse_dannce_from_scratch/*.pth'     # 1분마다 확인

# === 시각화 ===
cd results/mouse_dannce_infer/test_results_checkpoint3000
eog *_0_image*.png                                                      # Frame 0 비교
blender 0003000_0_mesh.obj                                              # 3D 메시

# === Tensorboard ===
tensorboard --logdir results/fauna_mouse_dannce_from_scratch/tensorboard_logs --port 6006

# === 학습 중단 ===
pkill -f run_full_notf32

# === 학습 재개 ===
nohup python run_full_notf32.py --config-name train_fauna_mouse_dannce \
  resume=results/fauna_mouse_dannce_from_scratch/checkpoint25000.pth \
  > /tmp/fauna_resume.log 2>&1 &
```

---

## 🎯 다음 단계 (학습 완료 후)

1. **최종 checkpoint로 추론**:
   ```bash
   python run_debug_notf32.py --config-name infer_mouse_dannce \
     resume=results/fauna_mouse_dannce_from_scratch/checkpoint50000.pth \
     checkpoint_dir=results/mouse_dannce_infer_final
   ```

2. **품질 비교**:
   - checkpoint3000 vs checkpoint50000
   - Loss 감소 확인
   - 시각적 품질 향상 확인

3. **결과 문서화**:
   - Before/After 비교 이미지
   - 메트릭 정리
   - 시각화 비디오 생성

---

**현재 상태**: ✅ Full training 진행 중! (~4시간 소요 예상)
