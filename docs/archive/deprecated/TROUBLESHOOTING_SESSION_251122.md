# Mouse DANNCE Training & Inference - 문제 해결 세션 (2025-11-22)

## 목차
1. [CUDA 환경 문제](#1-cuda-환경-문제)
2. [추론 실행 오류들](#2-추론-실행-오류들)
3. [현재 실행 중인 작업 설명](#3-현재-실행-중인-작업-설명)
4. [모델 구조 및 데이터 흐름](#4-모델-구조-및-데이터-흐름)
5. [결과 확인 방법](#5-결과-확인-방법)

---

## 1. CUDA 환경 문제

### 문제: CUBLAS_STATUS_NOT_SUPPORTED 오류

**증상:**
```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgemm(...)`
Location: DINO ViT attention layer (qkv linear projection)
```

**근본 원인:**
- PyTorch 버전 불일치: 1.10.0 (CUDA 11.3) vs 시스템 CUDA 11.8/12.4
- TF32 (TensorFloat-32) 연산이 Ampere GPU (RTX 3060)에서 CUBLAS 호환성 문제 발생

**해결 과정:**

#### 시도 1: Batch Size 감소 (실패)
```yaml
# config/dataset/fauna_mouse_dannce.yaml
batch_size: 2 → 1
```
❌ 오류 지속

#### 시도 2: TF32 비활성화 (실패)
```python
# run_debug_notf32.py 생성
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```
❌ 오류 지속 (PyTorch 버전 문제가 근본 원인)

#### 시도 3: PyTorch 2.0.0 업그레이드 시도 (의존성 충돌)
```bash
conda install pytorch==2.0.0 pytorch-cuda=11.8
```
❌ Python 3.9와 의존성 충돌 (`sympy`, `antlr-python-runtime`)

#### 최종 해결: PyTorch + PyTorch3D 재설치 ✅

```bash
# 1. 기존 PyTorch 제거
conda remove pytorch torchvision torchaudio pytorch3d --yes

# 2. PyTorch 2.0.0 + CUDA 11.8 설치
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# 3. PyTorch3D 0.7.3 (미리 빌드된 wheel) 설치
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# 4. NumPy 다운그레이드 (호환성)
pip install "numpy<2"
```

**검증:**
```bash
python test_cuda_fix.py
# ✅ Test 1: Basic CUDA tensor - PASS
# ✅ Test 2: Linear layer (CUBLAS) - PASS
# ✅ Test 3: DINO attention operation - PASS
```

**최종 환경:**
- PyTorch: 2.0.0+cu118
- CUDA: 11.8
- PyTorch3D: 0.7.3
- NumPy: 1.26.4
- GPU: RTX 3060 12GB

---

## 2. 추론 실행 오류들

### 오류 2.1: Config 파일 못 찾음

**증상:**
```
Cannot find primary config 'infer_mouse_dannce'
```

**원인:** Config 파일이 `config/` 디렉토리가 아닌 프로젝트 루트에 있음

**해결:**
```bash
mv infer_mouse_dannce.yaml config/
```

---

### 오류 2.2: Test 디렉토리 없음

**증상:**
```
FileNotFoundError: .../mouse_dannce_6view/test
FileNotFoundError: .../mouse_markerless_6view/test
```

**원인:**
- FaunaDataset이 `large_scale/` 아래의 모든 동물 카테고리를 스캔
- `train`, `val`, `test` 폴더 모두 필요하지만, 실제로는 `train`만 존재

**해결:** 심볼릭 링크 생성
```bash
cd data/fauna/Fauna_dataset/large_scale/mouse_dannce_6view
ln -sf train test

cd ../mouse_markerless_6view
ln -sf train test
```

---

### 오류 2.3: 체크포인트 파싱 오류

**증상:**
```
ValueError: invalid literal for int() with base 10: ''
at: int(''.join([c for c in osp.basename(x) if c.isdigit()]))
```

**원인:**
- Trainer가 `checkpoint_dir`에서 모든 `.pth` 파일을 찾음
- `results/` 디렉토리에 숫자가 없는 파일 존재:
  - `fauna_mouse_mammal_init.pth` (❌ 숫자 없음)
  - `mammal_mouse_sdf_mlp.pth` (❌ 숫자 없음)
  - `checkpoint3000.pth` (✅ 숫자 있음)

**해결:** 전용 디렉토리 생성 및 절대 경로 사용
```bash
mkdir -p results/mouse_dannce_infer
cp results/checkpoint3000.pth results/mouse_dannce_infer/
```

```yaml
# config/infer_mouse_dannce.yaml
resume: /home/joon/dev/3DAnimals/results/checkpoint3000.pth
output_dir: /home/joon/dev/3DAnimals/results/mouse_dannce_infer
checkpoint_dir: /home/joon/dev/3DAnimals/results/mouse_dannce_infer  # 명시적 설정!
```

**핵심 교훈:**
- Hydra가 작업 디렉토리를 `outputs/날짜/시간/`으로 변경
- 상대 경로는 작동하지 않음 → **절대 경로 필수**
- `checkpoint_dir`은 기본값 `'results'`를 사용 → **명시적 설정 필요**

---

### 오류 2.4: 함수 import 누락

**증상:**
```
NameError: name 'validate_all_to_device' is not defined
at: model/Trainer.py:144
```

**원인:** Line 144에서 `misc.` 접두사 누락

**해결:**
```python
# model/Trainer.py:144
# Before
batch = validate_all_to_device(batch, device=self.accelerator.device)

# After
batch = misc.validate_all_to_device(batch, device=self.accelerator.device)
```

---

## 3. 현재 실행 중인 작업 설명

### 실행 명령어
```bash
python run_debug_notf32.py --config-name infer_mouse_dannce
```

### 수행 작업: **3D 재구성 추론 (Inference)**

모델이 2D 이미지로부터 3D 형상을 복원하는 과정입니다.

#### 입력 (Input)
- **2D 이미지**: Mouse DANNCE 데이터셋 (50 frames)
  - RGB 이미지: `256×256` 해상도
  - 마스크 이미지: 전경(마우스) vs 배경 분리
  - 6개 카메라 뷰 (multi-view)

#### 출력 (Output) - 각 이미지마다
```
0003000_<frame_id>_image_gt.png      # 입력: Ground Truth 이미지
0003000_<frame_id>_image_pred.png    # 출력: 예측된 렌더링 이미지
0003000_<frame_id>_mask_gt.png       # 입력: Ground Truth 마스크
0003000_<frame_id>_mask_pred.png     # 출력: 예측된 마스크
0003000_<frame_id>_mesh.obj          # 출력: 3D 메쉬 (faces, vertices, normals)
0003000_<frame_id>_pose.txt          # 출력: 카메라 포즈 (6DoF)
```

### 출력 예시 로그 해석

```
writing 778 normals
writing 1552 faces
```

**의미:**
- **778 normals**: 메쉬 표면의 법선 벡터 (표면 방향 정보)
- **1552 faces**: 메쉬를 구성하는 삼각형 면 개수

**왜 매번 다른가?**

✅ **정답: 이미지마다 다른 포즈, 형상**

1. **포즈 차이**:
   - 각 프레임에서 마우스의 자세가 다름 (서있기, 앉기, 걷기 등)
   - 카메라 뷰포인트가 다름 (6개 카메라)

2. **Marching Cubes 알고리즘**:
   - SDF (Signed Distance Function) → 3D 메쉬 변환
   - SDF 값에 따라 메쉬 복잡도가 동적으로 변함
   - 같은 동물이라도 자세에 따라 면 개수가 달라짐

3. **관절 변형 (Articulation)**:
   - 다리, 꼬리 등의 관절 움직임
   - 관절 각도에 따라 로컬 지오메트리 변화

---

## 4. 모델 구조 및 데이터 흐름

### 전체 파이프라인

```
[입력 이미지]
    ↓
[DINO ViT Feature 추출]
    ↓
[Base Predictor: SDF 예측]
    ├─ Shape MLP: 3D 형상 (SDF field)
    ├─ Texture MLP: 색상 정보
    └─ Articulation MLP: 골격 관절 (num_body_bones=5)
    ↓
[Instance Predictor: 포즈 예측]
    ├─ Camera Pose (rotation, translation)
    └─ Bone Transformations
    ↓
[Marching Cubes: SDF → Mesh]
    ↓
[Neural Renderer: Mesh → 2D Image]
    ↓
[출력: 렌더링 이미지 + 3D 메쉬]
```

### 주요 구성 요소

#### 1. DINO ViT (Vision Transformer)
- **역할**: 입력 이미지에서 의미적 특징 추출
- **출력**: Feature map (패치 단위 특징)
- **학습**: Frozen (사전 학습된 가중치 사용)

#### 2. Base Predictor (형상 학습)
```yaml
cfg_shape:
  grid_res: 64              # SDF grid 해상도
  spatial_scale: 4.5        # 작은 동물 (마우스)
  hidden_size: 128          # MLP 크기
  init_sdf: ellipsoid       # 타원체에서 시작
  symmetrize: true          # 좌우 대칭
```

**SDF (Signed Distance Function)**:
- 3D 공간의 각 점에서 표면까지의 거리
- 양수: 외부, 음수: 내부, 0: 표면
- Marching Cubes로 메쉬 추출

#### 3. Articulation (관절 모델링)
```yaml
cfg_articulation:
  num_body_bones: 5         # 척추 관절 개수 (작은 동물)
  num_legs: 4               # 4개 다리
  articulation_iter_range: [10000, inf]  # 10K iteration부터 활성화
```

**골격 구조:**
```
     [머리]
       |
   [척추 5개]
    /  |  \  \
[다리1][다리2][다리3][다리4]
```

#### 4. 렌더링 파이프라인
```
SDF Field → Marching Cubes → Triangle Mesh → Neural Renderer → 2D Image
```

---

## 5. 결과 확인 방법

### 5.1 추론 결과 위치

```bash
# 메인 디렉토리
/home/joon/dev/3DAnimals/results/mouse_dannce_infer/test_results_checkpoint3000/

# 파일 구조
0003000_<frame_id>_image_gt.png      # Ground Truth 이미지
0003000_<frame_id>_image_pred.png    # 모델 예측 렌더링
0003000_<frame_id>_mask_gt.png       # Ground Truth 마스크
0003000_<frame_id>_mask_pred.png     # 모델 예측 마스크
0003000_<frame_id>_mesh.obj          # 3D 메쉬 파일
0003000_<frame_id>_pose.txt          # 카메라 포즈
```

### 5.2 시각화 확인

#### A. 이미지 비교
```bash
# 추론 결과 디렉토리로 이동
cd /home/joon/dev/3DAnimals/results/mouse_dannce_infer/test_results_checkpoint3000/

# 이미지 뷰어로 확인
eog 0003000_0_image_gt.png 0003000_0_image_pred.png    # Ground Truth vs 예측
eog 0003000_0_mask_gt.png 0003000_0_mask_pred.png      # 마스크 비교
```

**확인 사항:**
- 예측 이미지가 GT와 얼마나 유사한가?
- 마스크가 정확하게 동물 형태를 잡아내는가?
- 관절 위치가 자연스러운가?

#### B. 3D 메쉬 확인
```bash
# Blender로 열기 (설치 필요)
blender 0003000_0_mesh.obj

# MeshLab으로 열기 (대안)
meshlab 0003000_0_mesh.obj
```

**확인 사항:**
- 메쉬 토폴로지가 깨끗한가?
- 표면이 부드러운가? (아티팩트 없는가?)
- 마우스 형태가 자연스러운가?

### 5.3 WandB 로깅

**현재 상태:**
```yaml
# config/infer_mouse_dannce.yaml
wandb:
  mode: offline  # 오프라인 모드
```

**온라인 동기화 (선택사항):**
```bash
# WandB 로그 온라인 업로드
wandb login  # 첫 실행 시
wandb sync /home/joon/dev/3DAnimals/results/mouse_dannce_infer/wandb/
```

### 5.4 Tensorboard 로깅

```bash
# Tensorboard 실행
tensorboard --logdir /home/joon/dev/3DAnimals/results/tensorboard_logs --port 6006

# 브라우저에서 확인
# http://localhost:6006
```

**확인 가능한 메트릭:**
- Loss curves (mask_loss, rgb_loss, sdf_reg_loss)
- 이미지 렌더링 (epoch별)
- Mesh 시각화 (epoch별)

### 5.5 정량적 평가 메트릭

추론 완료 후 생성되는 메트릭:

```bash
# 메트릭 파일 (JSON)
cat /home/joon/dev/3DAnimals/results/mouse_dannce_infer/test_metrics.json
```

**예상 메트릭:**
```json
{
  "mask_iou": 0.85,           // 마스크 IoU (높을수록 좋음)
  "rgb_psnr": 22.5,           // RGB PSNR (높을수록 좋음)
  "chamfer_distance": 0.012,  // 3D 형상 오차 (낮을수록 좋음)
  "silhouette_iou": 0.88      // 실루엣 매칭 (높을수록 좋음)
}
```

---

## 6. 학습 vs 추론 비교

### Debug 학습 (완료)
```bash
python run_debug_notf32.py --config-name train_fauna_mouse_dannce_debug
```

- **목적**: Config 검증, 빠른 실험
- **Iterations**: 3,000
- **소요 시간**: ~10-15분
- **출력**: Checkpoints (`checkpoint3000.pth`)
- **run_train**: true, **run_test**: false

### 추론 (현재 실행 중)
```bash
python run_debug_notf32.py --config-name infer_mouse_dannce
```

- **목적**: 학습된 모델로 3D 재구성
- **입력**: Checkpoint + 테스트 이미지
- **출력**: 3D 메쉬, 렌더링 이미지
- **run_train**: false, **run_test**: true

### Full 학습 (다음 단계)
```bash
python run_full_notf32.py
```

- **Iterations**: 50,000
- **소요 시간**: ~2.5-3시간
- **출력**: 고품질 체크포인트

---

## 7. 핵심 교훈

### CUDA 환경
1. **버전 일치 중요**: PyTorch CUDA ↔ System CUDA
2. **TF32 비활성화**: Ampere GPU + DINO ViT 조합
3. **미리 빌드된 Wheel 사용**: PyTorch3D 소스 빌드 회피

### Hydra Config
1. **절대 경로 사용**: Hydra가 작업 디렉토리 변경
2. **명시적 설정**: `checkpoint_dir`, `output_dir` 모두 지정
3. **심볼릭 링크**: 데이터셋 구조 유연하게 대응

### 디버깅
1. **HYDRA_FULL_ERROR=1**: 전체 스택 트레이스 확인
2. **단계별 검증**: CUDA 테스트 → Debug 학습 → 추론
3. **로그 모니터링**: Loss, GPU 메모리, 파일 생성 상태

---

## 부록: 주요 명령어 요약

### 환경 확인
```bash
conda activate 3danimals
python test_cuda_fix.py
nvidia-smi
```

### 학습
```bash
# Debug 모드
python run_debug_notf32.py --config-name train_fauna_mouse_dannce_debug

# Full 학습
nohup python run_full_notf32.py > /tmp/mouse_full.log 2>&1 &
```

### 추론
```bash
python run_debug_notf32.py --config-name infer_mouse_dannce
```

### 결과 확인
```bash
# 파일 리스트
ls -lh results/mouse_dannce_infer/test_results_checkpoint3000/

# 이미지 비교
eog results/mouse_dannce_infer/test_results_checkpoint3000/0003000_0_*.png

# Tensorboard
tensorboard --logdir results/tensorboard_logs --port 6006
```
