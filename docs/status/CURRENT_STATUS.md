# Training Status Update - 2025-11-22

## ⚠️ 학습 중단 발생

**시간**: 19:30경
**Iteration**: 10,000 / 50,000 (20% 완료)
**원인**: Articulation 활성화 시점에서 에러

### 에러 상세

```
File "model/predictors/InstancePredictorFauna.py", line 84, in get_bones
    bones, self.kinematic_tree, self.bone_aux = estimate_bones(...)
```

**분석**:
- Articulation이 10K iteration에서 활성화됨
- `estimate_bones` 함수에서 에러 발생
- 이전 mouse-only 학습과 유사한 패턴 (arti_params None reference)

---

## ✅ 생성된 Checkpoints

| Checkpoint | Iterations | 상태 | 추론 |
|------------|-----------|------|------|
| checkpoint2000.pth | 2,000 | ✅ 완료 | 미실행 |
| checkpoint2500.pth | 2,500 | ✅ 완료 | 미실행 |
| checkpoint3000.pth | 3,000 | ✅ 완료 | ✅ 완료 (198 frames) |
| **checkpoint5000.pth** | **5,000** | **✅ 완료** | **🔄 진행 중** |

---

## 🎯 현재 진행 중

### Inference (checkpoint5000)

**실행 중**: checkpoint5000.pth로 추론
**위치**: `results/mouse_dannce_infer_5k/`
**목적**: 3,000 vs 5,000 iterations 품질 비교

**예상 결과**:
- 더 나은 shape 학습
- 향상된 silhouette
- 약간의 texture 학습 시작

---

## 🔍 품질 예상 비교

### checkpoint3000 (현재 확인됨)

**품질**: ⭐⭐☆☆☆
- Shape: 기본 blob
- Texture: 회색 단색
- Silhouette: 대략적
- Details: 거의 없음

### checkpoint5000 (예상)

**품질**: ⭐⭐⭐☆☆
- Shape: 더 정확한 형상
- Texture: 약간의 색상 학습
- Silhouette: 향상된 정확도
- Details: 초기 디테일 출현

### checkpoint50000 (목표, 현재 없음)

**품질**: ⭐⭐⭐⭐☆
- Shape: 정확한 동물 형상
- Texture: 세부 텍스처
- Silhouette: 높은 정확도
- Details: 풍부한 디테일

---

## 🛠️ 다음 조치 사항

### 1. 에러 수정 (우선순위)

**문제 파일**: `model/predictors/InstancePredictorFauna.py:84`

**예상 수정**:
```python
# estimate_bones 호출 전 None 체크 추가
if arti_params is not None:
    bones, self.kinematic_tree, self.bone_aux = estimate_bones(...)
else:
    # Handle None case
    bones = None
    ...
```

**참고**:
- 이전 `model/models/AnimalModel.py:549` 수정과 유사
- Articulation 활성화 범위 확인 필요
- Progressive training의 None handling

### 2. 학습 재개

**방법 A: checkpoint5000에서 재개** (권장)
```bash
python run_full_notf32.py --config-name train_fauna_mouse_dannce \
  resume=results/checkpoint5000.pth
```

**방법 B: checkpoint10000이 있다면**
```bash
# 10K까지는 학습 완료되었을 수 있음
ls results/fauna_mouse_dannce_from_scratch/checkpoint10000.pth

# 있다면 그것부터 재개
python run_full_notf32.py --config-name train_fauna_mouse_dannce \
  resume=results/fauna_mouse_dannce_from_scratch/checkpoint10000.pth
```

### 3. checkpoint5000 추론 결과 확인

**완료 후 실행**:
```bash
cd results/mouse_dannce_infer_5k/test_results_checkpoint5000

# 이미지 비교
eog *_0_image*.png

# 3K vs 5K 비교
eog ../mouse_dannce_infer/test_results_checkpoint3000/0003000_0_image*.png \
    mouse_dannce_infer_5k/test_results_checkpoint5000/0005000_0_image*.png
```

---

## 📊 현재 가용 데이터

### Multi-animal Fauna Dataset

**총 데이터**: ~18,000 frames
- Bear: ~1,360 frames
- Cow: ~2,360 frames
- Elephant: ~3,930 frames
- Giraffe: ~1,490 frames
- Horse: ~3,470 frames
- Mouse DANNCE: 50 frames ⭐
- Mouse Markerless: 50 frames ⭐
- Sheep: ~4,380 frames
- Zebra: ~1,050 frames

**학습 상태**:
- ✅ 5,000 iterations 완료
- ❌ 10,000 iterations에서 중단

### Mouse-Only Dataset

**총 데이터**: 100 frames (mouse_dannce + mouse_markerless)
- ❌ 학습 실패 (데이터 부족, mesh collapse)

---

## 🎯 목표 및 계획

### 즉시 목표

1. ✅ **checkpoint5000 추론 완료 대기**
2. **품질 비교** (3K vs 5K)
3. **에러 수정** (articulation None handling)
4. **학습 재개** (5K → 50K)

### 장기 목표

1. **50K iterations 완료**
2. **고품질 multi-animal 3D reconstruction**
3. **생쥐 포함한 9개 동물 모델**

---

## 📂 파일 위치

### Checkpoints
```
results/
├── checkpoint2000.pth          # 2K iters
├── checkpoint2500.pth          # 2.5K iters
├── checkpoint3000.pth          # 3K iters ✅ 추론 완료
├── checkpoint5000.pth          # 5K iters ⭐ 추론 중
└── fauna_mouse_dannce_from_scratch/
    └── checkpoint10000.pth?    # 있을 수도 있음 (확인 필요)
```

### Inference Results
```
results/
├── mouse_dannce_infer/
│   └── test_results_checkpoint3000/    # 3K 추론 결과 (198 frames)
└── mouse_dannce_infer_5k/
    └── test_results_checkpoint5000/    # 5K 추론 결과 (진행 중)
```

### Logs
```
/tmp/fauna_full_training.log            # Full training 로그
```

---

## 🔧 문제 해결

### Articulation 에러

**증상**: iteration 10,000에서 `estimate_bones` 에러

**원인**:
- Articulation이 `arti_reg_loss_iter_range: [30000, 'inf']`로 설정
- 하지만 articulation 자체는 10K부터 활성화
- None handling 누락

**해결 방법**:
1. Config에서 articulation 활성화 시점 확인
2. 관련 None 체크 추가
3. 또는 articulation 비활성화 후 학습

### Config 수정 (임시)

Articulation 문제 우회:
```yaml
# config/model/fauna_mouse_dannce.yaml
# Articulation 활성화 시점 조정
articulation_iter_range: [50000, 'inf']  # 비활성화
```

---

## ⏰ 예상 타임라인

### 현재 진행 상황
- 16:45 - Full training 시작
- 19:30 - 10K iteration에서 중단 (2시간 45분 소요)
- **19:40** - checkpoint5000 추론 시작

### 다음 단계 (에러 수정 후)
- **20:00** - 에러 수정 완료
- **20:10** - 학습 재개 (5K → 50K)
- **23:00-24:00** - 50K iteration 완료 예상 (~3-4시간)

---

## 💡 학습 내용

### Progressive Training의 어려움

1. **Phase 전환 시점**: 각 feature 활성화 iteration
2. **None Handling**: 모든 조건부 변수 체크 필요
3. **Config 설정**: Iteration range 충돌 방지

### Multi-animal Training의 이점

- ✅ 5,000 iterations까지 안정적 학습
- ✅ Mesh collapse 없음
- ✅ 다양한 동물 데이터로 regularization

### Articulation의 복잡성

- 10K iteration부터 활성화
- Skeleton estimation 필요
- Kinematic tree 구성
- None handling 필수

---

**현재 상태**: 🔄 checkpoint5000 추론 진행 중
**다음 단계**: 품질 비교 → 에러 수정 → 학습 재개
