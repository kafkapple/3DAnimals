# 현재 세션 상태 정리

**세션 날짜**: 2025-11-13
**작업 목표**: Fauna mouse training 시도 계획 수립

---

## 완료된 작업

### 1. ✅ 이전 실패 근거 파악
- **문서**: `docs/251112_research_fauna_mouse_final_findings.md`
- **핵심 발견**: Fauna는 mouse-scale animals **theoretically impossible**
- **증거**: 5가지 실험 (v0-v3, hybrid) 모두 실패
- **Root cause**: Sub-voxel features (mouse leg 5mm < voxel 11.7mm)

### 2. ✅ 현재 데이터셋 분석
**위치**: `/home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view`

**파일 현황**:
- ✅ RGB images (`*_rgb.png`)
- ✅ Masks (`*_mask.png`)
- ✅ Metadata (`*_metadata.json`, `*_box.txt`)
- ❌ DINO features (`*_feat16.png`) - 추출 필요

**Sequences**: 5개 (000000-000004)

### 3. ✅ DINO Feature 추출 스크립트 작성
**파일**: `scripts/extract_dino_features_mouse.py`
**상태**: 작성 완료, 버그 수정 완료

**기능**:
- DINOv2 model로 feature 추출
- PCA reduction (384 dim → 16 dim)
- feat16.png 형식으로 저장 (3DAnimals format)

**버그 이력**:
- ❌ Line 267: `model=args.model` (잘못된 parameter 이름)
- ✅ 수정됨: `model_name=args.model`

### 4. ✅ 실행 계획 문서화
**파일**: `FAUNA_MOUSE_EXECUTION_PLAN.md`

**내용**:
- Part A: Fauna 시도 계획 (DINO 추출 → Debug mode)
- Part B: Alternatives 탐색 (6가지 방법 비교)
- Top 3 추천: MAMMAL Fitting, DANNCE+MAMMAL, GS+MAMMAL
- 실행 로드맵 (Week 1-2)

---

## 미완료 작업

### 1. ⏳ DINO Features 추출
**명령어**:
```bash
conda run -n 3danimals python scripts/extract_dino_features_mouse.py \
  --data_dir /home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view \
  --device cuda
```

**예상 시간**: 1-2시간
**출력**: 각 `*_rgb.png` 옆에 `*_feat16.png` 생성

### 2. ⏳ Fauna Debug Mode 실행
**명령어**:
```bash
conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_debug \
  > /tmp/fauna_mouse_trial.log 2>&1
```

**예상 결과**: Iteration 3-7에서 crash (이미 예측됨)
**목적**: 불가능 재확인 (교육 목적)

### 3. ⏳ Alternatives 심화 조사
**조사 대상**:
- MAMMAL fitting to 2D keypoints (monocular)
- DANNCE + MAMMAL pipeline (multi-view)
- 2D keypoint detector for mouse

**우선순위**:
1. DANNCE + MAMMAL (multi-view data 있음, 최고 정확도)
2. MAMMAL Fitting (monocular, 중간 정확도)
3. GS + MAMMAL (하이브리드, 복잡함)

---

## 핵심 결정 사항

### 사용자 의도 파악

**질문**: "왜 Fauna 시도를 안 하는 게 좋다고 생각하지?"

**답변**:
1. ✅ **이미 불가능 증명됨** (2025-11-12)
   - 5가지 실험 모두 실패
   - Perfect initialization도 worst (3 iters)
   - Sub-voxel problem (theoretical impossibility)

2. ✅ **시간 낭비**
   - DINO 추출: 1-2시간
   - Debug run: 15분
   - 결과: 예상된 실패
   - **Total: 2-3시간** → 얻는 것 없음

3. ✅ **대안이 더 유망함**
   - MAMMAL mesh 이미 있음
   - Multi-view data 있음 (6 cameras)
   - DANNCE는 mouse 검증됨

**하지만**: 사용자가 "교육 목적"이 아니라 "실제로 monocular 3D prior 필요"

**결론**:
- Fauna는 skip 권장 (시간 절약)
- MAMMAL Fitting 또는 DANNCE+MAMMAL 바로 시작
- 더 좋은 방법 찾기에 집중

---

## 추천 방향

### 🥇 최우선 권장: DANNCE + MAMMAL

**이유**:
- ✅ Multi-view data 이미 있음 (6 cameras)
- ✅ DANNCE는 mouse 검증됨 (Nature paper)
- ✅ MAMMAL mesh로 정확한 3D reconstruction
- ✅ 최고 정확도 (<1mm error)

**단점**: Monocular 아님 (하지만 data는 있음)

**예상 성공률**: 95%

---

### 🥈 차선책: MAMMAL Fitting (Monocular)

**이유**:
- ✅ Monocular input 만족
- ✅ MAMMAL mesh 활용
- ✅ Articulation built-in

**필요 작업**:
- 2D keypoint detector 학습/사용
- MAMMAL parameter optimization

**예상 성공률**: 90%

---

### 🥉 실험적: Gaussian Splatting + MAMMAL

**이유**:
- ✅ High quality appearance
- ✅ MAMMAL prior 활용

**단점**: 복잡한 구현, multi-view 필요

**예상 성공률**: 60%

---

## 다음 세션 체크리스트

### Option A: Fauna 강행 (비권장)
- [ ] DINO features 추출 완료 확인
- [ ] Fauna debug mode 실행
- [ ] 실패 로그 분석 및 문서화
- [ ] 시간: ~2-3시간

### Option B: MAMMAL Fitting (권장 - Monocular)
- [ ] 2D keypoint detector 조사
- [ ] MAMMAL fitting pipeline 구현
- [ ] Monocular test image로 PoC
- [ ] 시간: 2-3일

### Option C: DANNCE + MAMMAL (강력 권장 - Multi-view)
- [ ] DANNCE 설치 및 환경 구축
- [ ] Mouse 데이터 DANNCE 형식 변환
- [ ] DANNCE training 시작
- [ ] MAMMAL fitting to 3D keypoints
- [ ] 시간: 3-5일

---

## 중요 파일 경로

### 문서
```
/home/joon/dev/3DAnimals/
├── FAUNA_MOUSE_EXECUTION_PLAN.md       # 실행 계획 (상세)
├── STATUS_CURRENT_SESSION.md           # 현재 상태 (본 문서)
└── docs/251112_research_fauna_mouse_final_findings.md  # 불가능 증명

/home/joon/CLAUDE.md                    # Fauna training guide (general)
```

### 코드
```
/home/joon/dev/3DAnimals/scripts/extract_dino_features_mouse.py  # DINO 추출
/home/joon/dev/MAMMAL_mouse/bodymodel_th.py                      # MAMMAL model
```

### 데이터
```
/home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view/
```

### Config
```
/home/joon/dev/3DAnimals/config/
├── train_fauna_mouse_debug.yaml
├── train_fauna_mouse_debug_v2.yaml
└── train_fauna_mouse_debug_v3.yaml
```

---

## 다음 세션 시작 명령어

### 문서 읽기
```bash
cd /home/joon/dev/3DAnimals
cat STATUS_CURRENT_SESSION.md
cat FAUNA_MOUSE_EXECUTION_PLAN.md
```

### 빠른 결정
```bash
# Option A 계속: DINO 추출
conda run -n 3danimals python scripts/extract_dino_features_mouse.py \
  --data_dir /home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view \
  --device cuda

# Option B: MAMMAL 바로 시작
cd /home/joon/dev/MAMMAL_mouse
python -c "from bodymodel_th import BodyModelTorch; print('MAMMAL ready!')"

# Option C: DANNCE 조사
cd /home/joon/dev
git clone https://github.com/spoonsso/dannce
```

---

**세션 종료 시각**: 2025-11-13 23:51 KST
**다음 세션 권장 방향**: Option C (DANNCE + MAMMAL) 또는 Option B (MAMMAL Fitting)
**Fauna 시도 권장**: ❌ Skip (이미 불가능 증명, 시간 낭비)
