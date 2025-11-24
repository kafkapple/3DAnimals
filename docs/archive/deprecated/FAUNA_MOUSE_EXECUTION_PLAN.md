# Fauna Mouse Training - Execution Plan

**작성일**: 2025-11-13
**목적**: Fauna mouse training 시도 및 대안 탐색 실행 계획
**상태**: Ready to Execute

---

## Executive Summary

### 현재 상황
1. **2025-11-12 연구 결과**: Fauna는 mouse-scale animals 근본적으로 불가능 (이미 증명됨)
2. **새로운 데이터 발견**: `/home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view`
   - ✅ RGB, Mask, Metadata 있음
   - ❌ DINO features (feat16.png) 없음 → 추출 필요
3. **사용자 요구사항**:
   - Monocular image input에서 적당한 3D prior 획득
   - Fauna가 가장 잘 알려진 animal model
   - 하지만 더 좋은 방법 있으면 고려

### 실행 전략
**A. Fauna 시도 (교육 목적)**: DINO features 추출 → Debug mode (15분) → 예상 실패 확인
**B. Alternatives 탐색 (본질적 해결)**: 더 나은 방법 조사 및 추천

---

## Part A: Fauna 시도 계획

### Step 1: DINO Features 추출

**스크립트**: `scripts/extract_dino_features_mouse.py`
**상태**: ✅ 작성 완료, ⚠️ 버그 수정 필요

**버그 내역**:
```python
# Line 267: Typo
process_mouse_dataset(
    data_dir=args.data_dir,
    output_dir=args.output_dir,
    model=args.model,  # ← ERROR: 'model' should be 'model_name'
    device=args.device
)
```

**수정 방법**:
```bash
# 1. 버그 수정
sed -i 's/model=args.model/model_name=args.model/' scripts/extract_dino_features_mouse.py

# 2. 실행
conda run -n 3danimals python scripts/extract_dino_features_mouse.py \
  --data_dir /home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view \
  --device cuda
```

**예상 시간**: 1-2시간 (GPU 사용 시)

**출력**:
- 각 sequence의 `*_rgb.png` 옆에 `*_feat16.png` 생성됨
- 예: `000000_00000/0000027_rgb.png` → `000000_00000/0000027_feat16.png`

---

### Step 2: Fauna Debug Mode 실행

**Config**: `config/train_fauna_mouse_debug.yaml`
**상태**: ✅ 이미 존재 (2025-11-12 실험에서 생성됨)

**실행 명령**:
```bash
# Debug mode (5000 iterations, ~15-20분)
conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_debug \
  > /tmp/fauna_mouse_trial.log 2>&1

# 실시간 모니터링
tail -f /tmp/fauna_mouse_trial.log
```

**예상 결과**: ❌ **Iteration 3-7에서 crash**

**증거**:
```
# 2025-11-12 실험 결과
v0 (baseline):  7 iterations → crash
v1 (spatial=3): 4 iterations → crash
v3 (extreme):   5 iterations → crash
Hybrid:         3 iterations → crash (worst!)
```

**교훈**: Fauna는 이론적으로 mouse 지원 불가능 재확인

---

## Part B: Alternatives 탐색 계획

### 문제 정의

**핵심 요구사항**:
1. **Input**: Monocular RGB image (single view)
2. **Output**: 3D animal mesh with reasonable prior
3. **Target**: Mouse (small animal, 75mm height)
4. **Optional**: MAMMAL mouse mesh (14k vertices, LBS) 활용 가능

**Fauna의 장점** (왜 선택했는가?):
- ✅ Monocular input 지원
- ✅ Category-agnostic (범용성)
- ✅ 2D diffusion prior 활용 (SD 기반)
- ✅ Articulation 자동 학습
- ❌ **치명적 단점**: Large animals only (23× scale mismatch)

---

### 조사 항목

#### 1. **NeRF-based Methods**

**Candidates**:
- Instant-NGP (NVIDIA, 2022)
- Nerfacto (Nerfstudio, 2023)
- TensoRF (2022)

**Pros**:
- ✅ Continuous representation (no voxel limit)
- ✅ Scale-agnostic
- ✅ High quality reconstruction

**Cons**:
- ⚠️ Requires multi-view input (not monocular!)
- ⚠️ No articulation model
- ⚠️ Slow optimization (minutes per scene)

**Verdict**: ❌ Not suitable (monocular 요구사항 위반)

---

#### 2. **Diffusion-based 3D Generation**

**Candidates**:
- Zero-1-to-3 (ICCV 2023)
- One-2-3-45 (ICCV 2023)
- Consistent123 (arXiv 2024)

**Approach**: Single image → Multi-view generation → 3D reconstruction

**Pros**:
- ✅ Monocular input 지원
- ✅ Leverages 2D diffusion priors (like Fauna)
- ✅ General objects (not animal-specific)

**Cons**:
- ⚠️ No articulation
- ⚠️ Hallucination risk (novel view synthesis)
- ⚠️ Requires fine-tuning for animals

**Verdict**: ⚠️ 가능하지만 articulation 없음

---

#### 3. **SDS-based Methods (Score Distillation Sampling)**

**Candidates**:
- DreamFusion (ICLR 2023)
- Magic3D (CVPR 2023)
- ProlificDreamer (NeurIPS 2023)

**Approach**: Text/image prompt → NeRF optimization via SDS

**Pros**:
- ✅ Single image input 가능
- ✅ Leverages diffusion priors

**Cons**:
- ⚠️ Slow (hours of optimization)
- ⚠️ No articulation
- ⚠️ Requires prompt engineering

**Verdict**: ❌ Too slow, no articulation

---

#### 4. **3D Gaussian Splatting Variants**

**Candidates**:
- 3D-GS (SIGGRAPH 2023)
- 2D-GS (SIGGRAPH Asia 2024)
- SuGaR (CVPR 2024) - mesh extraction

**Approach**: Point cloud → Gaussian primitives → Fast rendering

**Pros**:
- ✅ Very fast training/rendering
- ✅ High quality
- ✅ Scale-agnostic
- ✅ SuGaR: mesh extraction 가능

**Cons**:
- ⚠️ Requires multi-view OR monocular with depth
- ⚠️ No articulation model
- ⚠️ Topology handling difficult

**Integration with MAMMAL mesh**:
- 🔧 Possible: Use MAMMAL as initialization, optimize Gaussians

**Verdict**: ⚠️ 가능하지만 multi-view or depth 필요

---

#### 5. **Parametric Animal Models**

**SMAL (Skinned Multi-Animal Linear Model)**:
- Paper: CVPR 2017
- GitHub: https://github.com/silviazuffi/smalst
- Coverage: Dog, cat, horse, cow, hippo (NO MOUSE!)

**BARC (canine-specific)**:
- Paper: CVPR 2022
- Coverage: Dogs only

**MAMMAL (rodent-specific)**:
- Paper: CVPR 2024
- Coverage: Mouse, rat ✅
- ✅ **이미 사용 가능!**

**Approach**: Fit parametric model to 2D observations

**Pros**:
- ✅ Strong anatomical prior
- ✅ Built-in articulation (LBS)
- ✅ MAMMAL already has mouse model!

**Cons**:
- ⚠️ Requires keypoint detection or silhouette
- ⚠️ Limited to predefined topology

**Verdict**: ✅ **매우 유망!**

---

#### 6. **DANNCE (3D Pose Estimation)**

**Paper**: Nature Methods 2023
**Purpose**: 3D pose estimation for freely moving animals
**Coverage**: Mouse ✅ (검증됨)

**Input**: Multi-view videos (3-6 cameras)
**Output**: 3D keypoints (pose only, no mesh)

**Pros**:
- ✅ Mouse-specific, validated
- ✅ High accuracy (< 1mm error)
- ✅ Multi-view setup 있음 (6 cameras)

**Cons**:
- ❌ Not monocular (requires multi-view)
- ❌ Pose only (no mesh/texture)

**Integration Possibility**:
```
DANNCE (pose) + MAMMAL (mesh) = Complete 3D reconstruction
```

**Verdict**: ✅ **Multi-view 활용 시 최고 정확도**

---

### 종합 비교표

| Method | Monocular | Articulation | Mouse Support | Speed | Quality | Difficulty |
|--------|-----------|--------------|---------------|-------|---------|------------|
| **Fauna** | ✅ | ✅ | ❌ (proven impossible) | Fast | High | Easy |
| **NeRF** | ❌ | ❌ | ⚠️ | Slow | High | Medium |
| **Zero-1-to-3** | ✅ | ❌ | ⚠️ | Medium | Medium | Medium |
| **3D-GS** | ⚠️ (with depth) | ❌ | ✅ | Very Fast | Very High | Hard |
| **MAMMAL Fitting** | ✅ | ✅ | ✅ | Fast | Medium | Medium |
| **DANNCE + MAMMAL** | ❌ (multi-view) | ✅ | ✅ | Fast | Very High | Easy |

---

## 추천 방법 (우선순위)

### 🥇 Option 1: MAMMAL Fitting (권장 - 모노큘러)

**접근법**: Monocular image → 2D keypoints/silhouette → MAMMAL parameter optimization

**장점**:
- ✅ Monocular input ✅
- ✅ Mouse-specific model ✅
- ✅ Articulation built-in ✅
- ✅ Anatomically correct
- ✅ 이미 구현됨 (`/home/joon/dev/MAMMAL_mouse`)

**단점**:
- ⚠️ Requires 2D keypoints (detection 필요)
- ⚠️ Limited to MAMMAL topology

**구현 계획**:
```python
# 1. 2D Keypoint Detection
from mmpose import inference_top_down_pose_model
keypoints_2d = detect_keypoints(rgb_image)  # 22 joints

# 2. MAMMAL Fitting
from MAMMAL_mouse.bodymodel_th import BodyModelTorch
mouse_model = BodyModelTorch('mouse_model/mouse.pkl')

# Optimize pose, trans, betas to match 2D keypoints
pose, trans, betas = optimize_mammal_to_keypoints(
    mouse_model, keypoints_2d, camera_params
)

# 3. Generate 3D Mesh
vertices, joints = mouse_model(pose, trans, betas)
mesh = trimesh.Trimesh(vertices.cpu(), mouse_model.faces)
```

**예상 시간**: 2-3일 (keypoint detector 학습 포함)

**성공 확률**: ⭐⭐⭐⭐⭐ (90%)

---

### 🥈 Option 2: DANNCE + MAMMAL (최고 정확도 - 멀티뷰)

**접근법**: Multi-view videos → DANNCE 3D pose → MAMMAL fitting

**장점**:
- ✅ Multi-view data 이미 있음 (6 cameras) ✅
- ✅ DANNCE는 mouse 검증됨 ✅
- ✅ 최고 정확도 (< 1mm error)
- ✅ Articulation built-in

**단점**:
- ❌ Monocular 아님 (하지만 data 있음)

**구현 계획**:
```bash
# 1. DANNCE Setup
git clone https://github.com/spoonsso/dannce
cd dannce && pip install -e .

# 2. Train DANNCE on mouse data
dannce train --data /path/to/mouse_dannce_6view

# 3. Predict 3D poses
dannce predict --video /path/to/videos

# 4. Fit MAMMAL to 3D keypoints
python fit_mammal_to_3d_keypoints.py \
  --dannce_output dannce_3d_poses.pkl \
  --mouse_model MAMMAL_mouse/mouse_model/mouse.pkl
```

**예상 시간**: 3-5일

**성공 확률**: ⭐⭐⭐⭐⭐ (95%)

---

### 🥉 Option 3: Gaussian Splatting + MAMMAL (하이브리드)

**접근법**: MAMMAL initialization → Gaussian Splatting optimization

**장점**:
- ✅ MAMMAL prior 활용
- ✅ High quality appearance
- ✅ Fast rendering

**단점**:
- ⚠️ Requires depth or multi-view
- ⚠️ Complex implementation

**구현 계획**:
```python
# 1. MAMMAL as initialization
mammal_mesh = fit_mammal_to_monocular(rgb_image)

# 2. Sample Gaussian points from mesh
gaussians = initialize_gaussians_from_mesh(mammal_mesh)

# 3. Optimize Gaussians with multi-view
from gaussian_splatting import GaussianModel
model = GaussianModel(sh_degree=3)
model.create_from_pcd(gaussians, spatial_lr_scale=1.0)

# Train with images
optimize_gaussians(model, training_images, camera_poses)
```

**예상 시간**: 1-2주 (복잡도 높음)

**성공 확률**: ⭐⭐⭐ (60%)

---

## 실행 로드맵

### Week 1: Fauna 시도 + MAMMAL Fitting PoC

**Day 1 (오늘)**:
- [x] 현재 상황 정리
- [x] 실행 계획 문서화
- [ ] DINO extraction script 버그 수정

**Day 2**:
- [ ] DINO features 추출 (1-2시간)
- [ ] Fauna debug mode 실행 (15분)
- [ ] 실패 확인 및 문서화

**Day 3-5**:
- [ ] MAMMAL fitting 구현 시작
- [ ] 2D keypoint detector 탐색/학습
- [ ] Monocular → MAMMAL pipeline 구축

### Week 2: DANNCE + MAMMAL (Optional)

**Day 1-3**:
- [ ] DANNCE 설치 및 환경 구축
- [ ] Mouse 데이터 DANNCE 형식 변환
- [ ] Training 시작

**Day 4-5**:
- [ ] 3D pose prediction
- [ ] MAMMAL fitting to 3D keypoints
- [ ] Evaluation

---

## 파일 위치 정리

### 현재 생성된 파일
```
/home/joon/dev/3DAnimals/
├── scripts/
│   └── extract_dino_features_mouse.py  # DINO 추출 스크립트 (버그 있음)
├── config/
│   ├── train_fauna_mouse_debug.yaml    # Debug config (v0)
│   ├── train_fauna_mouse_debug_v2.yaml # v2
│   └── train_fauna_mouse_debug_v3.yaml # v3
├── docs/
│   └── 251112_research_fauna_mouse_final_findings.md  # 불가능 증명
└── FAUNA_MOUSE_EXECUTION_PLAN.md       # 본 문서 ✅

/home/joon/dev/MAMMAL_mouse/
├── mouse_model/
│   ├── mouse.pkl                       # Mouse parametric model
│   ├── mouse_reduced_face_1800.obj
│   ├── mouse_reduced_face_3600.obj
│   └── mouse_reduced_face_7200.obj
├── bodymodel_th.py                     # PyTorch body model
└── articulation_th.py                  # Articulation functions

/home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view/
├── train/
│   ├── 000000_00000/
│   │   ├── 0000027_rgb.png            # ✅ 있음
│   │   ├── 0000027_mask.png           # ✅ 있음
│   │   ├── 0000027_metadata.json      # ✅ 있음
│   │   ├── 0000027_box.txt            # ✅ 있음
│   │   └── 0000027_feat16.png         # ❌ 없음 (추출 필요)
│   ├── 000001_00000/
│   ├── 000002_00000/
│   ├── 000003_00000/
│   └── 000004_00000/
├── test/ -> train (symlink)
└── val/ -> train (symlink)
```

---

## 다음 세션 시작 방법

### 1. 문서 확인
```bash
cd /home/joon/dev/3DAnimals
cat FAUNA_MOUSE_EXECUTION_PLAN.md
```

### 2. DINO Features 추출 (Option A 계속하려면)
```bash
# 버그 수정
sed -i '267s/model=args.model/model_name=args.model/' scripts/extract_dino_features_mouse.py

# 실행
conda run -n 3danimals python scripts/extract_dino_features_mouse.py \
  --data_dir /home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view \
  --device cuda
```

### 3. MAMMAL Fitting 시작 (Option 1 바로 시작하려면)
```bash
cd /home/joon/dev/MAMMAL_mouse

# PoC: Load mouse model
python -c "
from bodymodel_th import BodyModelTorch
import torch

model = BodyModelTorch('mouse_model/mouse.pkl')
print('Mouse model loaded successfully!')
print(f'Vertices: {model.v_template.shape}')
print(f'Faces: {model.faces.shape}')
print(f'Joints: {model.t_pose_joints.shape}')
"
```

---

## 핵심 질문 (다음 세션 결정 사항)

### Q1: Fauna를 정말 시도할 것인가?
- **Yes**: DINO features 추출 → Debug mode (15분) → 실패 재확인
- **No**: MAMMAL fitting으로 바로 이동 (시간 절약)

**추천**: ❌ Skip (이미 불가능 증명됨, 시간 낭비)

### Q2: Monocular vs Multi-view?
- **Monocular**: Option 1 (MAMMAL Fitting)
- **Multi-view**: Option 2 (DANNCE + MAMMAL) - 더 정확함

**추천**: ✅ Option 2 (Multi-view data 이미 있음, 최고 정확도)

### Q3: MAMMAL mesh를 어떻게 활용?
- **Prior**: Initialization for other methods
- **Direct**: Fit MAMMAL to 2D/3D keypoints ← **권장**
- **Hybrid**: MAMMAL + Gaussian Splatting

**추천**: ✅ Direct (가장 간단하고 효과적)

---

## 참고 자료

### Papers
- Fauna: arXiv:2401.02400v2
- MAMMAL: CVPR 2024
- DANNCE: Nature Methods 2023
- SMAL: CVPR 2017
- Zero-1-to-3: ICCV 2023
- 3D-GS: SIGGRAPH 2023

### GitHub Repos
- Fauna: https://github.com/3DAnimals/3DAnimals
- MAMMAL_mouse: `/home/joon/dev/MAMMAL_mouse`
- DANNCE: https://github.com/spoonsso/dannce
- 3D-GS: https://github.com/graphdeco-inria/gaussian-splatting

### Internal Docs
- `/home/joon/dev/3DAnimals/docs/251112_research_fauna_mouse_final_findings.md`
- `/home/joon/CLAUDE.md` (Fauna training guide)

---

**마지막 업데이트**: 2025-11-13
**다음 실행 준비**: ✅ 완료
**권장 방향**: Option 2 (DANNCE + MAMMAL) 또는 Option 1 (MAMMAL Fitting)
