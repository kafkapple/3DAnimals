# 3D-Fauna Training & Inference 완전 가이드

**작성일**: 2025-11-10
**모델**: 3D-Fauna (Pan-category Articulated 3D Quadruped Reconstruction)
**목적**: Training 구조 및 Inference 활용 방법 완전 정리

---

## 1. 현재 실행 명령어 분석

```bash
(time python run.py --config-name train_fauna) 2>&1 | tee -a time.log
```

### 실행 흐름

1. **Entry Point**: `run.py:main()` (line 7-19)
2. **Config 로딩**: `config/train_fauna.yaml`
3. **Model 생성**: `build_model(cfg.model)` → `FaunaModel` 인스턴스
4. **Trainer 생성**: `Trainer(cfg, model)`
5. **학습 실행**: `trainer.train()`

---

## 2. Training Input/Output 구조

### 2.1 Input 데이터 구조

**Dataset**: `FaunaDataset` (model/dataset/FaunaDataset.py:41)

```python
# Batch 구조 (forward 함수, AnimalModel.py:362)
input_image     # BxFxCxHxW - RGB 이미지 [0,1] normalized
mask_gt         # BxFxHxW   - Ground truth mask (binary)
mask_dt         # BxFx2xHxW - Distance transform (foreground/background)
mask_valid      # BxFxHxW   - Valid pixel mask
flow_gt         # Bx(F-1)x2xHxW - Optical flow (if video)
bbox            # BxFx8 - [frame_id, x0, y0, w, h, full_w, full_h, sharpness]
bg_image        # BxFxCxHxW - Background image (optional)
dino_feat_im    # BxFxDxHxW - DINO features (D=16 in config)
dino_cluster_im # BxFxKxHxW - DINO clustering (optional)
keypoint        # BxFxKx3   - 2D keypoints (optional)
seq_idx         # BxF       - Sequence index
frame_idx       # BxF       - Frame index within sequence
```

**주요 설정** (config/train_fauna.yaml:13-23):
- `in_image_size`: 256 (입력 이미지 크기)
- `out_image_size`: 256 (출력 렌더링 크기)
- `batch_size`: 6
- `load_dino_feature`: true (DINO feature 사용)
- `dino_feature_dim`: 16
- `random_xflip_train`: true (Data augmentation)

**데이터 경로**:
```yaml
train_data_dir: data/fauna/Fauna_dataset
```

### 2.2 Model Forward Pass

**함수**: `FaunaModel.forward()` → `AnimalModel.forward()` (AnimalModel.py:361)

```python
def forward(batch, epoch, total_iter, is_training=True):
    # 1. Prior Shape 예측 (Category-shared)
    prior_shape, dino_net = netBase(total_iter, is_training)

    # 2. Instance-specific 예측
    shape, pose_raw, pose, mvp, w2c, campos, texture, \
    im_features, deformation, arti_params, light, forward_aux \
        = netInstance(input_image, prior_shape, epoch, total_iter)

    # 3. Rendering
    renders = render(shape, texture, mvp, w2c, campos, ...)
    image_pred, mask_pred, dino_feat_im_pred = renders

    # 4. Loss 계산
    losses = compute_reconstruction_losses(...)
    regularizers = compute_regularizers(...)

    return metrics
```

### 2.3 Output 구조

**학습 중 저장되는 데이터**:

1. **Checkpoint** (`results/fauna/exp/checkpointXXXXX.pth`):
   ```python
   {
       'netBase': ...,         # Prior shape network weights
       'netInstance': ...,     # Instance predictor weights
       'netDisc': ...,         # Discriminator weights (Fauna)
       'optimizerBase': ...,
       'optimizerInstance': ...,
       'optimizerDisc': ...,
       'metrics_trace': ...,   # Training history
       'epoch': int,
       'total_iter': int
   }
   ```

2. **Training Results** (매 `save_train_result_freq` iter):
   - `{iter:07d}_{frame_id:10d}_image_pred.png`
   - `{iter:07d}_{frame_id:10d}_mask_pred.png`
   - `{iter:07d}_{frame_id:10d}_mesh.obj`
   - `{iter:07d}_{frame_id:10d}_pose.txt`
   - `{iter:07d}_{frame_id:10d}_arti_params.txt` (articulation parameters)

3. **Logging** (WandB/TensorBoard):
   - Loss curves
   - Image visualizations
   - Mesh rotation videos
   - Histogram of parameters

---

## 3. Training 과정 의미 해석

### 3.1 Progressive Training

**Grid Resolution Progressive** (AnimalModel.py:381-386):
```python
if in_range(total_iter, cfg.grid_res_coarse_iter_range):
    grid_res = 64  # Coarse stage
else:
    grid_res = 128  # Fine stage
```

**Loss Weight Scheduling**:
- Texture loss는 특정 iteration range에서만 활성화
- Articulation regularization은 점진적 활성화
- Discriminator loss는 특정 iteration 이후 활성화

### 3.2 Two-stage Learning

**Stage 1: Base Predictor (Prior Shape)**
- Category-shared canonical shape 학습
- SDF-based implicit representation (DMTet)
- DINO feature head 학습

**Stage 2: Instance Predictor**
- Instance-specific deformation
- Pose estimation (camera + articulation)
- Texture prediction
- Lighting estimation

### 3.3 Loss Functions

**Reconstruction Losses**:
```python
- mask_loss         # Mask L2 loss
- rgb_loss          # RGB L1 loss (foreground only)
- dino_feat_im_loss # DINO feature loss
- flow_loss         # Optical flow consistency (video)
```

**Regularizers**:
```python
- sdf_bce_reg       # SDF binary cross entropy
- sdf_gradient_reg  # Eikonal loss (|∇SDF| = 1)
- arti_reg_loss     # Articulation parameter regularization
- deform_reg_loss   # Deformation regularization
- laplacian_smooth  # Mesh smoothness
```

**Multi-hypothesis Pose** (QuadLookat):
- 4개의 pose hypothesis 동시 예측
- 각 hypothesis의 reconstruction loss 계산
- Soft selection via probability weighting

---

## 4. Inference 활용 방법

### 4.1 학습된 모델로 Inference 실행

**명령어**:
```bash
python run.py --config-name test_fauna
```

**Config 설정** (config/test_fauna.yaml):
```yaml
run_train: false
run_test: true

checkpoint_dir: results/fauna/pretrained_fauna/
checkpoint_name: pretrained_fauna.pth

dataset:
  test_data_dir: data/fauna/Fauna_dataset/large_scale/bear_comb_dinov2_new/test
  batch_size: 1
  num_frames: 1

output_dir: results/fauna/pretrained_fauna/visualization
render_modes: [input_view, other_views, rotation]
finetune_texture: false
```

### 4.2 Input 이미지 준비

**필수 파일**:
```
test_dir/
├── {name}_rgb.png        # RGB image (필수)
├── {name}_mask.png       # Mask (선택, texture finetuning 시 권장)
└── {name}_box.txt        # Bounding box (자동 생성 가능)
```

**DINO Feature**:
- 테스트 시에는 **DINO feature 불필요**
- 모델이 내부적으로 DINO feature extractor 보유

### 4.3 Inference Forward Pass

**실행 흐름** (Trainer.py:129-146):
```python
def test():
    model.set_eval()
    epoch, total_iter = load_checkpoint()

    for batch in test_loader:
        with torch.no_grad():
            metrics = model.forward(
                batch,
                epoch=epoch,
                total_iter=total_iter,
                save_results=True,
                save_dir=test_result_dir,
                is_training=False
            )
```

### 4.4 Output 결과물

**저장되는 파일들** (AnimalModel.py:643-670):

1. **Rendered Images**:
   ```
   {iter:07d}_{frame_id:10d}_image_pred.png   # Reconstructed RGB
   {iter:07d}_{frame_id:10d}_mask_pred.png    # Predicted mask
   {iter:07d}_{frame_id:10d}_image_gt.png     # Input image
   {iter:07d}_{frame_id:10d}_mask_gt.png      # Input mask (if provided)
   ```

2. **3D Mesh**:
   ```
   {iter:07d}_{frame_id:10d}_mesh.obj         # Textured 3D mesh
   ```

3. **Parameters**:
   ```
   {iter:07d}_{frame_id:10d}_pose.txt         # 7D pose [quat(4), trans(3)]
   {iter:07d}_{frame_id:10d}_arti_params.txt  # Articulation params (N_joints x dim)
   ```

### 4.5 Inference 결과 의미

**3D Mesh (`.obj`)**:
- Vertices: 3D positions of mesh vertices
- Faces: Triangle connectivity
- Texture coordinates: UV mapping
- Material: Albedo + shading

**Pose (`.txt`)**:
```
Format: [qw, qx, qy, qz, tx, ty, tz]
- Quaternion (qw, qx, qy, qz): Camera rotation
- Translation (tx, ty, tz): Camera position
```

**Articulation Parameters (`.txt`)**:
```
Shape: (N_bones, dim)
- Each row: Articulation parameters for one bone
- Learned representation (not physical angles)
- Can be used to animate the mesh via skinning
```

---

## 5. 커스텀 Inference 워크플로우

### 5.1 단일 이미지 입력

```python
# 1. 이미지 준비
input_image = load_image("animal.jpg")  # [H, W, 3]
input_image = resize(input_image, (256, 256))  # Normalize to [0,1]

# 2. Config 설정
cfg = {
    'checkpoint_dir': 'results/fauna/pretrained_fauna/',
    'checkpoint_name': 'pretrained_fauna.pth',
    'test_data_dir': 'path/to/test/images/',
    'output_dir': 'results/my_test/',
}

# 3. 실행
python run.py --config-name test_fauna
```

### 5.2 Batch Inference

**데이터 구조**:
```
test_data_dir/
├── image1_rgb.png
├── image2_rgb.png
├── image3_rgb.png
...
```

**Config 수정**:
```yaml
dataset:
  batch_size: 8  # 원하는 batch size
  test_data_dir: path/to/test/images/
```

### 5.3 Texture Finetuning (Test-time Optimization)

**Config 설정**:
```yaml
finetune_texture: true
finetune_iters: 10     # Finetuning iterations
finetune_lr: 0.001     # Learning rate
```

**효과**:
- Input view에서 texture 정밀도 향상
- Mask가 제공되면 더 정확한 최적화
- 배경 픽셀이 mesh에 투영되는 것 방지

---

## 6. 고급 시각화 (Visualization Script)

### 6.1 Render Modes

**Config 설정** (config/test_fauna.yaml:34):
```yaml
render_modes: [input_view, other_views, rotation]
```

**Available Modes**:
1. **`input_view`**:
   - Input viewpoint에서 렌더링
   - Textured mesh, shading map, gray shape

2. **`other_views`**:
   - 12개 viewpoint에서 렌더링 (360도 회전)
   - Textured mesh, gray shape

3. **`rotation`**:
   - 연속 회전 비디오 생성
   - Textured mesh, gray shape

4. **`animation`** (Quadrupeds only):
   - 사전 정의된 articulation parameters로 애니메이션
   - Side view + rotation view 동시 생성

5. **`canonicalization`**:
   - Input pose → Canonical pose 변형 비디오

### 6.2 Visualization Script 실행

```bash
# Standard visualization
python visualization/visualize_results_fauna.py --config-name test_fauna

# 커스텀 설정
python visualization/visualize_results_fauna.py \
    --config-name test_fauna \
    --render_modes input_view rotation \
    --finetune_texture true
```

---

## 7. 실전 사용 예시

### 예시 1: 새로운 동물 이미지 재구성

```bash
# 1. 테스트 이미지 준비
mkdir -p data/my_test
cp my_dog_photo.jpg data/my_test/dog001_rgb.png

# 2. Config 수정 (test_fauna.yaml)
dataset:
  test_data_dir: data/my_test/

output_dir: results/my_dog_reconstruction/

# 3. 실행
python run.py --config-name test_fauna

# 4. 결과 확인
ls results/my_dog_reconstruction/
# → dog001_mesh.obj
# → dog001_image_pred.png
# → dog001_pose.txt
# → dog001_arti_params.txt
```

### 예시 2: 다수 이미지 배치 처리

```bash
# 1. 테스트 데이터 준비
data/animal_collection/
├── cat001_rgb.png
├── dog001_rgb.png
├── lion001_rgb.png
└── ... (100+ images)

# 2. Config
dataset:
  test_data_dir: data/animal_collection/
  batch_size: 16

# 3. 실행
python run.py --config-name test_fauna

# 4. 모든 결과가 output_dir에 저장됨
```

### 예시 3: 애니메이션 생성

```bash
# 1. Articulation parameters 준비
visualization/animation_params/
├── walk_cycle.txt  # (N_frames, N_bones, dim)
└── run_cycle.txt

# 2. Config
render_modes: [animation]
arti_param_dir: ./visualization/animation_params

# 3. 실행
python visualization/visualize_results_fauna.py --config-name test_fauna

# 4. 결과: 애니메이션 비디오 생성
```

---

## 8. 주요 파라미터 튜닝 가이드

### 8.1 학습 안정성 개선

**문제**: Mesh 붕괴, NaN loss

**해결책** (config/train_fauna.yaml:26-75):
```yaml
training:
  grad_clip_norm: 1.0       # Gradient clipping
  warmup_iters: 1000        # Learning rate warmup

  loss_weights:
    sdf_bce_reg: 2.0        # SDF regularization 강화
    sdf_gradient_reg: 0.3
    laplacian_smooth: 0.01  # Mesh smoothing

optimizer:
  lr: 0.0001                # Learning rate 감소

model:
  dmtet:
    grid_res: 64            # 64 → 128 점진적 증가
```

### 8.2 Inference 품질 향상

**Texture 품질**:
```yaml
finetune_texture: true
finetune_iters: 50        # 기본값 10 → 50
finetune_lr: 0.0005       # 세밀한 조정
```

**Rendering 품질**:
```yaml
model:
  render:
    renderer_spp: 4         # Samples per pixel (default: 1)
    background_mode: white  # 'black', 'white', 'checkerboard'
```

---

## 9. 트러블슈팅

### 9.1 CUDA Out of Memory

**해결책**:
```yaml
dataset:
  batch_size: 2           # Batch size 감소
  in_image_size: 128      # 256 → 128
  out_image_size: 128

model:
  dmtet:
    grid_res: 64          # 128 → 64
```

### 9.2 Poor Reconstruction Quality

**원인 1**: Pose estimation 실패
- **해결**: Multi-hypothesis pose (QuadLookat) 활성화됨 확인

**원인 2**: Texture blurry
- **해결**: Test-time texture finetuning 활성화

**원인 3**: Category mismatch
- **해결**: 3D-Fauna는 **quadrupeds only** (개, 고양이, 말, 사자 등)
  - Birds는 MagicPony bird model 사용

### 9.3 Missing DINO Features

**에러**: `FileNotFoundError: *_dino.npy`

**해결**:
```yaml
dataset:
  load_dino_feature: false  # Training에서만 필요
```

**참고**: Test 시에는 DINO feature 불필요 (모델 내장)

---

## 10. 체크리스트

### Training 시작 전
- [ ] Dataset 다운로드 완료 (`data/fauna/Fauna_dataset/`)
- [ ] Tetrahedral grids 다운로드 (`data/tets/`)
- [ ] CUDA 환경 검증 (`torch.cuda.is_available()`)
- [ ] Config 파일 확인 (`config/train_fauna.yaml`)
- [ ] Checkpoint 저장 경로 설정 (`checkpoint_dir`)
- [ ] WandB/TensorBoard 설정 (`logger_type`)

### Inference 시작 전
- [ ] Pretrained model 다운로드 (`results/fauna/pretrained_fauna/`)
- [ ] Test 이미지 준비 (`*_rgb.png` 형식)
- [ ] Config 파일 수정 (`config/test_fauna.yaml`)
- [ ] Output 경로 설정 (`output_dir`)
- [ ] Render modes 선택 (`render_modes`)

### 결과 검증
- [ ] 3D mesh 파일 생성 확인 (`.obj`)
- [ ] Rendered images 품질 확인
- [ ] Pose parameters 저장 확인 (`.txt`)
- [ ] Articulation parameters 저장 확인 (`.txt`)

---

## 11. 추가 자료

**공식 문서**:
- GitHub: https://github.com/3dmagicpony/3DAnimals
- 3D-Fauna Project Page: https://kyleleey.github.io/3DFauna/
- Paper: https://arxiv.org/abs/2401.02400

**관련 모델**:
- **MagicPony**: Category-specific (horses, birds, etc.)
- **Ponymation**: Motion generation (video-to-4D)

**코드 참조**:
- Training loop: `model/Trainer.py`
- Model forward: `model/models/AnimalModel.py:361`
- Dataset: `model/dataset/FaunaDataset.py`
- Visualization: `visualization/visualize_results_fauna.py`

---

## 요약

**Training**:
```bash
python run.py --config-name train_fauna
```
- Input: RGB images + masks + DINO features
- Output: Checkpoints with learned 3D shape + pose + texture

**Inference**:
```bash
python run.py --config-name test_fauna
```
- Input: Single RGB image (mask optional)
- Output: 3D mesh + pose + articulation params + rendered images

**Key Insight**:
- **Two-stage learning**: Prior shape (category-shared) + Instance deformation
- **Progressive training**: Coarse-to-fine grid resolution
- **Multi-hypothesis pose**: Robust pose estimation
- **Test-time optimization**: Texture finetuning for quality
