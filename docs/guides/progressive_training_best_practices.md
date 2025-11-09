# Progressive Training Best Practices Guide

**대상:** 3DAnimals 프로젝트 개발자 및 연구자
**목적:** Progressive Training 환경에서 안전하고 견고한 코드 작성
**버전:** 1.0
**최종 수정:** 2025-11-09

---

## 📖 목차

1. [Progressive Training 개요](#1-progressive-training-개요)
2. [조건부 변수 처리 패턴](#2-조건부-변수-처리-패턴)
3. [Config 설정 가이드](#3-config-설정-가이드)
4. [코드 리뷰 체크리스트](#4-코드-리뷰-체크리스트)
5. [디버깅 전략](#5-디버깅-전략)
6. [테스트 시나리오](#6-테스트-시나리오)
7. [자주 하는 실수](#7-자주-하는-실수)

---

## 1. Progressive Training 개요

### 1.1 개념

Progressive Training은 복잡한 특징들을 **단계적으로 활성화**하여 학습 안정성을 높이는 전략입니다.

```
Simple → Complex
├─ Phase 1: Basic features (Shape, Texture, Pose)
├─ Phase 2: + Articulation
├─ Phase 3: + Regularization
└─ Phase 4: + Deformation
```

### 1.2 3DAnimals의 Timeline

#### Fauna 모델 예시

| Iteration | 활성화 기능 | 생성되는 변수 |
|-----------|------------|--------------|
| 0 ~ 20K | Shape, Texture, Pose | `shape`, `texture`, `pose`, `mvp`, `w2c` |
| 20K ~ 60K | + Articulation | + `arti_params` |
| 60K ~ 800K | + Regularization, Attachment | (동일) |
| 800K+ | + Deformation | + `deformation` |

#### 주요 Config 설정

```yaml
# config/model/fauna.yaml
cfg_texture:
  texture_iter_range: [0, inf]        # 항상 활성화

cfg_deform:
  deform_iter_range: [800000, inf]    # 800K부터

cfg_articulation:
  articulation_iter_range: [20000, inf]  # 20K부터
  attach_legs_to_body_iter_range: [60000, inf]  # 60K부터

cfg_loss:
  arti_reg_loss_iter_range: [60000, inf]  # 60K부터
```

### 1.3 장단점

**장점:**
- ✅ 학습 초기 안정성 향상
- ✅ 복잡한 특징 순차적 학습
- ✅ 각 단계별 디버깅 용이

**단점:**
- ⚠️ 각 phase마다 다른 상태 관리 필요
- ⚠️ 조건부 로직 복잡도 증가
- ⚠️ 모든 phase에서 테스트 필요

---

## 2. 조건부 변수 처리 패턴

### 2.1 기본 원칙

> **Rule of Thumb**: 조건부로 생성되는 모든 변수는 **생성과 사용 양쪽에서** None 처리 필요

### 2.2 올바른 패턴

#### Pattern A: 생성 단계

```python
# ✅ Good: 명시적 초기화
def forward(self, ..., total_iter):
    # 1. None으로 초기화
    arti_params = None

    # 2. 조건부 생성
    if self.enable_articulation and in_range(total_iter, self.cfg_articulation.articulation_iter_range):
        shape, arti_params = self.forward_articulation(...)
    else:
        # 3. Dummy operations (DDP 호환성)
        shape.v_pos += sum([p.sum() * 0 for p in self.netArticulation.parameters()])

    # 4. 반환 (None일 수 있음)
    return ..., arti_params
```

#### Pattern B: 사용 단계

```python
# ✅ Good: None 체크 후 사용
def save_results(self, log):
    # 방법 1: if 문
    if log.arti_params is not None:
        save(log.arti_params)

    # 방법 2: 조건부 표현식
    params = log.arti_params if log.arti_params is not None else default_value

    # 방법 3: 조기 반환
    if log.arti_params is None:
        return
    save(log.arti_params)
```

### 2.3 잘못된 패턴

```python
# ❌ Bad: 초기화 없음
def forward(self, ...):
    if condition:
        arti_params = compute()
    # else에서 arti_params가 정의되지 않음!
    return arti_params  # NameError 가능

# ❌ Bad: None 체크 없이 사용
def save_results(self, log):
    save(log.arti_params)  # None이면 에러!

# ❌ Bad: 암묵적 가정
def process(self, log):
    # "arti_params는 항상 존재한다"고 가정
    log.arti_params.cpu()  # AttributeError: 'NoneType' object has no attribute 'cpu'
```

### 2.4 프로젝트 내 실제 예시

#### ✅ 올바른 예시들

```python
# 예시 1: im_features
feat = log.im_features[:b0] if log.im_features is not None else None
misc.save_obj(..., feat=feat, ...)

# 예시 2: flow_pred
if log.flow_pred is not None:
    flow_pred_viz = torch.cat([log.flow_pred, ...], 2) + 0.5
    save_image('flow_pred', flow_pred_viz)

# 예시 3: deformation (사용처 없어서 None 체크 불필요)
deformation = None
if self.enable_deform and in_range(...):
    shape, deformation = self.forward_deformation(...)
return ..., deformation
```

#### ❌ 수정된 예시

```python
# Before (버그):
misc.save_txt(..., rearrange(log.arti_params, ...).cpu().numpy(), ...)

# After (수정):
if log.arti_params is not None:
    misc.save_txt(..., rearrange(log.arti_params, ...).cpu().numpy(), ...)
```

---

## 3. Config 설정 가이드

### 3.1 Iteration Range 설정 원칙

#### 원칙 1: 저장 주기와 feature 활성화 조율

```yaml
# ⚠️ 위험: 충돌 가능
save_train_result_freq: 10000
model:
  cfg_articulation:
    articulation_iter_range: [20000, inf]
# → 10K에서 저장 시도 → arti_params = None → 에러 (코드 수정 전)

# ✅ 안전: 코드에서 None 처리 (현재 상태)
# 어떤 값으로 설정해도 안전

# ✅ 더 안전: 일치시키기
save_train_result_freq: 20000
model:
  cfg_articulation:
    articulation_iter_range: [20000, inf]
```

#### 원칙 2: Feature 간 의존성 고려

```yaml
# ✅ Good: 의존성 순서 지키기
cfg_articulation:
  articulation_iter_range: [20000, inf]          # Articulation 먼저
  attach_legs_to_body_iter_range: [60000, inf]   # 그 다음 attachment

cfg_loss:
  arti_reg_loss_iter_range: [60000, inf]         # Loss는 feature 이후

# ❌ Bad: 순서 뒤바뀜
cfg_loss:
  arti_reg_loss_iter_range: [10000, inf]  # Loss 먼저
cfg_articulation:
  articulation_iter_range: [20000, inf]   # Feature 나중 → arti_params 없는데 loss 계산 시도
```

### 3.2 신규 Feature 추가 시 템플릿

```yaml
# 새로운 feature 추가 시 아래 템플릿 사용
cfg_new_feature:
  # 1. 활성화 여부
  enable_new_feature: true

  # 2. 활성화 시점 (기존 feature들보다 늦게)
  new_feature_iter_range: [100000, inf]

  # 3. Feature 관련 설정
  num_layers: 4
  hidden_size: 256

cfg_loss:
  # 4. Loss 활성화 (feature 이후로 설정)
  new_feature_loss_iter_range: [120000, inf]  # > 100000
  new_feature_loss_weight: 1.0
```

### 3.3 Config Validation (권장)

```python
def validate_config(cfg):
    """Config 설정 간 충돌 검증"""
    warnings = []

    # 1. Save frequency vs feature activation
    if hasattr(cfg.model, 'cfg_articulation'):
        arti_start = cfg.model.cfg_articulation.articulation_iter_range[0]
        save_freq = cfg.get('save_train_result_freq', float('inf'))

        if save_freq < arti_start:
            warnings.append(
                f"⚠️  save_train_result_freq ({save_freq}) < "
                f"articulation_start ({arti_start}). "
                f"arti_params will not be saved in early iterations."
            )

    # 2. Loss activation vs feature activation
    if hasattr(cfg.loss, 'arti_reg_loss_iter_range'):
        loss_start = cfg.loss.arti_reg_loss_iter_range[0]
        if loss_start < arti_start:
            warnings.append(
                f"❌ arti_reg_loss starts at {loss_start} but "
                f"articulation starts at {arti_start}!"
            )

    for w in warnings:
        print(w)

    return len([w for w in warnings if w.startswith('❌')]) == 0
```

---

## 4. 코드 리뷰 체크리스트

### 4.1 새로운 조건부 변수 추가 시

- [ ] **초기화**: `None`으로 명시적 초기화?
- [ ] **생성 로직**: 조건부 생성 로직 명확?
- [ ] **else 처리**: else 브랜치에서 적절한 처리?
- [ ] **사용처 검토**: 모든 사용 지점에서 None 체크?
- [ ] **저장 함수**: `save_results`에서 None 처리?
- [ ] **로그 함수**: `log_visuals`에서 None 처리?
- [ ] **Config 설정**: `iter_range` 설정 합리적?

### 4.2 기존 코드 수정 시

- [ ] **조건 변경**: 기존 조건부 로직 변경 시 영향 범위 확인?
- [ ] **변수 사용처**: 해당 변수를 사용하는 모든 곳 검토?
- [ ] **타입 체크**: 라이브러리 함수의 타입 요구사항 확인?
- [ ] **에러 처리**: 예외 상황 처리?

### 4.3 Config 변경 시

- [ ] **의존성**: 다른 설정과의 의존성 확인?
- [ ] **타임라인**: Feature 활성화 순서 합리적?
- [ ] **저장 주기**: `save_*_freq`와 `iter_range` 충돌 없음?
- [ ] **문서화**: 변경 사항 README 또는 주석에 기록?

---

## 5. 디버깅 전략

### 5.1 Progressive Training 관련 에러 진단

#### Step 1: Iteration 확인

```bash
# 에러 로그에서 iteration 찾기
grep -E "Writing mesh.*[0-9]+" error.log
# 예: Writing mesh: results/fauna/exp/training_results/0010000_3_mesh.obj
#     → Iteration = 10,000
```

#### Step 2: Feature 활성화 범위 확인

```bash
# Config에서 iter_range 찾기
grep -r "iter_range" config/model/fauna.yaml

# 출력 예:
# texture_iter_range: [0, inf]           → 항상 활성화
# deform_iter_range: [800000, inf]       → 800K부터
# articulation_iter_range: [20000, inf]  → 20K부터
```

#### Step 3: 조건 비교

```python
# 에러 발생 iteration과 비교
if iteration < articulation_start:
    # arti_params가 None일 가능성 높음
    print("❌ Articulation not active yet!")
```

#### Step 4: None 체크 확인

```bash
# 변수 사용처 찾기
grep -rn "log.arti_params" model/

# None 체크 있는지 확인
grep -B 2 "log.arti_params" model/ | grep "if.*is not None"
```

### 5.2 일반적인 에러 패턴과 해결

| 에러 메시지 | 원인 | 해결 |
|-----------|------|------|
| `RuntimeError: Tensor type unknown to einops <class 'NoneType'>` | einops 함수에 None 전달 | `if var is not None:` 추가 |
| `AttributeError: 'NoneType' object has no attribute 'cpu'` | None.cpu() 호출 | `if var is not None:` 추가 |
| `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'` | None 연산 시도 | 초기화 또는 기본값 설정 |
| `NameError: name 'variable' is not defined` | 조건부 정의 후 else 없음 | `variable = None` 초기화 |

### 5.3 디버깅 팁

```python
# Tip 1: 변수 상태 로깅
def forward(self, ..., total_iter):
    arti_params = None
    if self.enable_articulation and in_range(total_iter, ...):
        arti_params = self.forward_articulation(...)

    # 디버그 로그 추가
    if total_iter % 1000 == 0:
        print(f"[Iter {total_iter}] arti_params: {type(arti_params)}")

    return ..., arti_params

# Tip 2: Assertion 추가
def save_results(self, log):
    # 개발 중 assertion으로 검증
    assert log.pose is not None, "pose should always exist"

    # Optional 변수는 None 허용
    if log.arti_params is not None:
        save(log.arti_params)
```

---

## 6. 테스트 시나리오

### 6.1 필수 테스트 시나리오

#### 시나리오 1: 초기 학습 (Iteration 0부터)

```bash
# 목적: 모든 phase 경계에서 안정성 확인
python run.py --config-name train_fauna resume=false

# 확인 사항:
# - Iteration 10K: 저장 성공 (arti_params 없어도 OK)
# - Iteration 20K: arti_params 생성 시작
# - Iteration 60K: regularization 활성화
```

#### 시나리오 2: Resume 학습

```bash
# Before articulation
python run.py --config-name train_fauna \
  resume=true checkpoint_path=results/.../checkpoint10000.pth

# After articulation
python run.py --config-name train_fauna \
  resume=true checkpoint_path=results/.../checkpoint50000.pth
```

#### 시나리오 3: Config Override

```bash
# Articulation 조기 활성화
python run.py --config-name train_fauna \
  model.cfg_predictor_instance.cfg_articulation.articulation_iter_range=[5000,inf]

# 저장 주기 변경
python run.py --config-name train_fauna \
  save_train_result_freq=5000
```

### 6.2 Critical Iteration Points

모든 feature 활성화 시점에서 테스트 필요:

```yaml
# Fauna 모델 critical points
- 0      : 학습 시작
- 10,000 : 첫 저장 (기본 설정)
- 20,000 : Articulation 시작 ⚠️
- 60,000 : Regularization 시작 ⚠️
- 80,000 : Discriminator 시작
- 800,000: Deformation 시작 ⚠️
```

### 6.3 자동화 테스트 (권장)

```python
# tests/test_progressive_training.py
import pytest

@pytest.mark.parametrize("iteration", [0, 10000, 20000, 60000])
def test_save_results_at_different_iterations(iteration):
    """모든 critical iteration에서 저장 테스트"""
    model = create_model()
    batch = create_dummy_batch()

    # Forward pass
    with torch.no_grad():
        metrics = model.forward(batch, total_iter=iteration,
                               save_results=True,
                               save_dir="test_output")

    # 에러 없이 완료되어야 함
    assert metrics is not None

    # 저장된 파일 확인
    assert os.path.exists(f"test_output/{iteration:07d}_*_pose.txt")

    # arti_params는 조건부
    if iteration >= 20000:
        assert os.path.exists(f"test_output/{iteration:07d}_*_arti_params.txt")
```

---

## 7. 자주 하는 실수

### 실수 1: "항상 존재한다" 가정

```python
# ❌ Bad
def process(self, log):
    # "arti_params는 항상 있을 거야"
    params = log.arti_params.cpu()

# ✅ Good
def process(self, log):
    # "arti_params는 없을 수도 있어"
    if log.arti_params is not None:
        params = log.arti_params.cpu()
```

### 실수 2: else 브랜치 누락

```python
# ❌ Bad
def forward(self, ..., total_iter):
    if condition:
        arti_params = compute()
    # else?
    return arti_params  # NameError 가능!

# ✅ Good
def forward(self, ..., total_iter):
    arti_params = None  # 초기화
    if condition:
        arti_params = compute()
    return arti_params
```

### 실수 3: Config 의존성 무시

```yaml
# ❌ Bad: Loss가 feature보다 먼저 활성화
cfg_loss:
  arti_reg_loss_iter_range: [10000, inf]
cfg_articulation:
  articulation_iter_range: [20000, inf]  # 10K~20K 구간에서 에러!

# ✅ Good: Loss는 feature 이후
cfg_articulation:
  articulation_iter_range: [20000, inf]
cfg_loss:
  arti_reg_loss_iter_range: [20000, inf]  # 또는 [60000, inf]
```

### 실수 4: 단일 iteration에서만 테스트

```python
# ❌ Bad
# 50,000 iteration에서만 테스트
python run.py ... +trainer.max_iter=50000

# ✅ Good
# 여러 critical point에서 테스트
for iter in [10000, 20000, 60000]:
    python run.py ... +trainer.max_iter=$iter
```

### 실수 5: 라이브러리 동작 오해

```python
# einops는 None을 받지 않음
rearrange(None, "b f n c -> (b f) n c")
# → RuntimeError (타입 체크 실패)

# torch는 더 명확한 에러
None.cpu()
# → AttributeError (메서드 없음)

# 교훈: 라이브러리 호출 전 검증
if tensor is not None:
    rearrange(tensor, ...)
```

---

## 8. 추가 자료

### 8.1 관련 문서

- `docs/reports/20251109_arti_params_none_error.md` - 실제 버그 분석
- `docs/guides/coding_defensive_programming.md` - Defensive programming 가이드

### 8.2 참고 코드

- `model/models/AnimalModel.py:643-669` - save_results 구현
- `model/predictors/InstancePredictorBase.py:680-698` - 조건부 변수 생성
- `config/model/fauna.yaml` - Progressive training 설정

### 8.3 외부 참고

- [Progressive Training in GANs](https://arxiv.org/abs/1710.10196)
- [PyTorch DDP Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-11-09 | 1.0 | 초안 작성 |

---

**이 문서에 대한 질문이나 제안사항이 있으시면 이슈를 등록해주세요.**
