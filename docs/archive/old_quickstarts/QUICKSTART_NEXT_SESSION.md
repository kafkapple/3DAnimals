# 다음 세션 빠른 시작 가이드

**목적**: 다음 세션에서 5분 안에 상황 파악 및 작업 시작

---

## 📋 현재 상황 (한 줄 요약)

**Fauna mouse training 불가능 확정 → MAMMAL 기반 대안 3가지 제안됨 → 실행 계획 완료**

---

## 🎯 핵심 결정 사항

### Q: Fauna를 시도해야 하나?
**A: ❌ NO** (이미 theoretically impossible 증명됨, 시간 낭비)

### Q: 그럼 어떻게 monocular → 3D mouse를 만드나?
**A: ✅ MAMMAL mesh 활용** (3가지 옵션 제안됨)

---

## 🚀 다음 세션 시작 (3가지 옵션)

### Option 1️⃣: DANNCE + MAMMAL (최고 정확도) ⭐⭐⭐⭐⭐

**장점**: Multi-view data 있음 (6 cameras), 최고 정확도 (<1mm)
**단점**: Monocular 아님 (하지만 data는 있음)

```bash
# 시작 명령어
cd /home/joon/dev
git clone https://github.com/spoonsso/dannce
cd dannce && pip install -e .

# 데이터 확인
ls /home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view/train/
```

**예상 시간**: 3-5일
**성공 확률**: 95%

---

### Option 2️⃣: MAMMAL Fitting (Monocular) ⭐⭐⭐⭐

**장점**: Monocular input 만족, 빠른 PoC
**단점**: 2D keypoint detector 필요

```bash
# 시작 명령어
cd /home/joon/dev/MAMMAL_mouse

# Test MAMMAL model
python -c "
from bodymodel_th import BodyModelTorch
model = BodyModelTorch('mouse_model/mouse.pkl')
print('✅ MAMMAL model ready!')
print(f'Vertices: {model.v_template.shape}')
print(f'Joints: {model.t_pose_joints.shape}')
"
```

**예상 시간**: 2-3일
**성공 확률**: 90%

---

### Option 3️⃣: Fauna 강행 (비권장) ⭐

**왜 비권장?**: 2025-11-12에 이미 불가능 증명됨

```bash
# 그래도 시도하려면...
cd /home/joon/dev/3DAnimals

# 1. DINO features 추출 (1-2시간)
conda run -n 3danimals python scripts/extract_dino_features_mouse.py \
  --data_dir /home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view \
  --device cuda

# 2. Debug mode 실행 (15분)
conda run -n 3danimals python run.py \
  --config-name train_fauna_mouse_debug \
  > /tmp/fauna_mouse_trial.log 2>&1

# 3. 예상 결과: Iteration 3-7에서 crash
```

**예상 시간**: 2-3시간
**예상 결과**: 실패 (이미 알고 있음)

---

## 📚 필독 문서 (우선순위 순)

### 1️⃣ 현재 세션 상태
```bash
cat /home/joon/dev/3DAnimals/STATUS_CURRENT_SESSION.md
```
**내용**: 오늘 한 작업, 미완료 작업, 다음 단계

### 2️⃣ 실행 계획 (상세)
```bash
cat /home/joon/dev/3DAnimals/FAUNA_MOUSE_EXECUTION_PLAN.md
```
**내용**: 6가지 방법 비교, Top 3 추천, 주간 로드맵

### 3️⃣ Fauna 불가능 증명
```bash
cat /home/joon/dev/3DAnimals/docs/251112_research_fauna_mouse_final_findings.md
```
**내용**: 5가지 실험 결과, 왜 불가능한지 상세 분석

---

## 🗂️ 주요 파일 위치

### 문서
- ✅ `QUICKSTART_NEXT_SESSION.md` (본 문서)
- ✅ `STATUS_CURRENT_SESSION.md` (현재 상태)
- ✅ `FAUNA_MOUSE_EXECUTION_PLAN.md` (실행 계획)

### 코드
- ✅ `scripts/extract_dino_features_mouse.py` (버그 수정 완료)
- ✅ `/home/joon/dev/MAMMAL_mouse/bodymodel_th.py`

### 데이터
- ✅ `/home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view/`

---

## 💡 빠른 의사결정 플로우차트

```
START
  │
  ├─ Multi-view 데이터 활용 가능?
  │   YES → Option 1 (DANNCE + MAMMAL) ⭐⭐⭐⭐⭐
  │   NO  → ↓
  │
  ├─ Monocular만 가능?
  │   YES → Option 2 (MAMMAL Fitting) ⭐⭐⭐⭐
  │   NO  → ↓
  │
  └─ Fauna를 꼭 시도해야 하나?
      YES → Option 3 (Fauna 강행) ⭐ [비권장]
      NO  → Option 1 or 2 선택
```

---

## 🎓 왜 Fauna는 안 되는가? (30초 버전)

1. **Scale mismatch**: Mouse (75mm) vs Horse (1750mm) = 23배 차이
2. **Sub-voxel problem**: Mouse leg (5mm) < Fauna voxel (11.7mm)
3. **이미 증명됨**: 5가지 실험 모두 실패 (v0-v3, hybrid)
4. **Perfect initialization도 worst**: 더 정확한 초기화 = 더 빠른 실패
5. **Theoretical impossibility**: Parameter tuning으로 해결 불가능

**결론**: 시간 낭비. 대안 사용.

---

## 🔥 강력 추천 (TL;DR)

### 추천 순서:
1. **DANNCE + MAMMAL** (multi-view, 최고 정확도)
2. **MAMMAL Fitting** (monocular, 중간 정확도)
3. ~~Fauna~~ (skip)

### 시작 명령어:
```bash
# Option 1
cd /home/joon/dev
git clone https://github.com/spoonsso/dannce

# Option 2
cd /home/joon/dev/MAMMAL_mouse
python bodymodel_th.py  # Test
```

---

**작성일**: 2025-11-13
**다음 실행 준비**: ✅ 완료
**예상 소요 시간**: Option 1 (3-5일), Option 2 (2-3일)
