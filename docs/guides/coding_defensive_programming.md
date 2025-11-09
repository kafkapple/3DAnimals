# Defensive Programming Guide for Python

**대상:** Python 개발자, 딥러닝 연구자
**목적:** 견고하고 안전한 코드 작성 방법
**버전:** 1.0
**최종 수정:** 2025-11-09

---

## 📖 목차

1. [Defensive Programming이란?](#1-defensive-programming이란)
2. [핵심 원칙](#2-핵심-원칙)
3. [None 처리 패턴](#3-none-처리-패턴)
4. [타입 안정성](#4-타입-안정성)
5. [에러 처리](#5-에러-처리)
6. [검증과 단언](#6-검증과-단언)
7. [실전 예제](#7-실전-예제)

---

## 1. Defensive Programming이란?

### 1.1 정의

> **Defensive Programming**: 예상치 못한 상황에서도 프로그램이 안전하게 동작하도록 하는 프로그래밍 기법

### 1.2 핵심 철학

```python
# ❌ Optimistic (낙관적)
# "이 변수는 항상 정수일 거야"
result = value * 2

# ✅ Defensive (방어적)
# "이 변수가 정수가 아닐 수도 있어"
if isinstance(value, int):
    result = value * 2
else:
    result = 0  # 또는 에러 처리
```

### 1.3 비용 vs 이득

| 측면 | 비용 | 이득 |
|------|------|------|
| **코드 라인** | +10~20% | - |
| **실행 시간** | +0.01% (거의 없음) | - |
| **버그 발생** | - | -90% |
| **디버깅 시간** | - | -80% |
| **시스템 안정성** | - | +무한대 |

**결론:** 비용은 거의 없고, 이득은 엄청남!

---

## 2. 핵심 원칙

### 원칙 1: 가정하지 말고 검증하라 (Don't Assume, Validate)

```python
# ❌ Bad: 가정
def process(data):
    # data가 리스트일 거라고 가정
    return data[0]

# ✅ Good: 검증
def process(data):
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")
    if len(data) == 0:
        raise ValueError("Empty list")
    return data[0]

# ✅ Better: 기본값 사용
def process(data):
    if not isinstance(data, list) or len(data) == 0:
        return None
    return data[0]
```

### 원칙 2: 조기 반환 (Early Return)

```python
# ❌ Bad: 중첩된 if
def save(data, path):
    if data is not None:
        if path is not None:
            if os.path.exists(os.path.dirname(path)):
                with open(path, 'w') as f:
                    f.write(data)

# ✅ Good: 조기 반환
def save(data, path):
    if data is None:
        return
    if path is None:
        return
    if not os.path.exists(os.path.dirname(path)):
        return

    with open(path, 'w') as f:
        f.write(data)
```

### 원칙 3: Fail Fast (빨리 실패하기)

```python
# ❌ Bad: 늦은 에러
def train_model(config):
    # 10시간 학습 후...
    save_checkpoint(config.checkpoint_path)  # path가 None이면 여기서 에러!

# ✅ Good: 초기 검증
def train_model(config):
    # 시작 전 검증
    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path required")

    # 이제 학습 시작
    for epoch in range(100):
        ...
```

### 원칙 4: 명시적이 암묵적보다 낫다 (Explicit is better than implicit)

```python
# ❌ Bad: 암묵적
def get_value(data):
    return data.get('key')  # None 또는 값

# ✅ Good: 명시적
def get_value(data, default=None):
    """
    Returns value for 'key' or default if not found.

    Args:
        data: Dictionary to search
        default: Value to return if key not found (default: None)

    Returns:
        Value or default
    """
    return data.get('key', default)
```

### 원칙 5: 불변성 선호 (Prefer Immutability)

```python
# ❌ Bad: 가변 기본 인자
def add_item(item, items=[]):
    items.append(item)
    return items

# 문제:
add_item(1)  # [1]
add_item(2)  # [1, 2]  ← 예상: [2]

# ✅ Good: 불변 기본값
def add_item(item, items=None):
    if items is None:
        items = []
    items = items.copy()  # 원본 보존
    items.append(item)
    return items
```

---

## 3. None 처리 패턴

### 3.1 기본 패턴

```python
# Pattern 1: if-else
if variable is not None:
    use(variable)
else:
    handle_none()

# Pattern 2: 조건부 표현식
result = variable if variable is not None else default_value

# Pattern 3: 조기 반환
if variable is None:
    return
use(variable)

# Pattern 4: get with default
value = config.get('key', default_value)
```

### 3.2 PyTorch/NumPy 특화 패턴

```python
# ✅ Tensor 연산 전 None 체크
if tensor is not None:
    tensor = tensor.cpu().detach().numpy()

# ✅ Optional feature 처리
feat = log.im_features[:b0] if log.im_features is not None else None

# ✅ 조건부 저장
if log.arti_params is not None:
    misc.save_txt(..., rearrange(log.arti_params, ...).cpu().numpy(), ...)
```

### 3.3 None 체크 안티패턴

```python
# ❌ Bad: == 사용
if variable == None:  # PEP 8 위반

# ✅ Good: is 사용
if variable is None:

# ❌ Bad: Truthiness에 의존
if variable:  # 0, [], False도 걸러짐!
    use(variable)

# ✅ Good: 명시적 None 체크
if variable is not None:
    use(variable)
```

### 3.4 여러 변수 동시 체크

```python
# Pattern 1: 모두 None이 아닌지
if all(v is not None for v in [var1, var2, var3]):
    use(var1, var2, var3)

# Pattern 2: 하나라도 None인지
if any(v is None for v in [var1, var2, var3]):
    print("At least one is None")

# Pattern 3: 개별 처리
variables = {'var1': var1, 'var2': var2, 'var3': var3}
none_vars = [k for k, v in variables.items() if v is None]
if none_vars:
    raise ValueError(f"These variables are None: {none_vars}")
```

---

## 4. 타입 안정성

### 4.1 타입 힌트 사용

```python
from typing import Optional, List, Dict, Union

# ✅ Good: 타입 힌트로 의도 명확히
def process_data(
    data: List[int],
    config: Optional[Dict] = None,
    verbose: bool = False
) -> Optional[torch.Tensor]:
    """
    Process data with optional config.

    Args:
        data: List of integers
        config: Optional configuration dict
        verbose: Whether to print debug info

    Returns:
        Processed tensor or None if processing failed
    """
    if config is None:
        config = {}

    # ... 처리 ...

    return result
```

### 4.2 isinstance로 타입 검증

```python
# ✅ Good: 타입 검증
def add(a, b):
    if not isinstance(a, (int, float)):
        raise TypeError(f"a must be number, got {type(a)}")
    if not isinstance(b, (int, float)):
        raise TypeError(f"b must be number, got {type(b)}")
    return a + b

# ✅ Better: 타입 힌트 + 런타임 검증
from typing import Union

def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    # IDE가 타입 체크 + 런타임에도 검증
    if not isinstance(a, (int, float)):
        raise TypeError(f"a must be number, got {type(a)}")
    if not isinstance(b, (int, float)):
        raise TypeError(f"b must be number, got {type(b)}")
    return a + b
```

### 4.3 라이브러리 함수 타입 요구사항

```python
# einops: Tensor, ndarray만 허용 (None 불가!)
from einops import rearrange

# ❌ Bad
rearrange(None, "b f n c -> (b f) n c")
# → RuntimeError: Tensor type unknown to einops <class 'NoneType'>

# ✅ Good
if tensor is not None:
    rearrange(tensor, "b f n c -> (b f) n c")

# ✅ Better: 타입 체크
from typing import Union
import torch
import numpy as np

def safe_rearrange(
    tensor: Union[torch.Tensor, np.ndarray, None],
    pattern: str
) -> Union[torch.Tensor, np.ndarray, None]:
    """einops.rearrange with None handling"""
    if tensor is None:
        return None
    return rearrange(tensor, pattern)
```

---

## 5. 에러 처리

### 5.1 Try-Except 패턴

```python
# Pattern 1: 특정 에러만 잡기
try:
    result = risky_operation()
except FileNotFoundError:
    result = None
except PermissionError:
    raise  # 다시 발생시킴

# Pattern 2: 여러 에러 처리
try:
    result = risky_operation()
except (FileNotFoundError, PermissionError) as e:
    print(f"File error: {e}")
    result = None

# Pattern 3: Cleanup (finally)
file = None
try:
    file = open('data.txt')
    process(file)
except IOError as e:
    print(f"Error: {e}")
finally:
    if file is not None:
        file.close()
```

### 5.2 에러 메시지 작성

```python
# ❌ Bad: 불명확한 메시지
raise ValueError("Invalid value")

# ✅ Good: 구체적인 메시지
raise ValueError(
    f"Expected positive integer, got {value}. "
    f"Please check your config at config.yaml:42"
)

# ✅ Best: Context 포함
raise ValueError(
    f"Invalid iteration range: [{start}, {end}]. "
    f"Expected start < end, but got start={start} >= end={end}. "
    f"Config: {config_path}"
)
```

### 5.3 Custom Exception

```python
# 프로젝트 특화 예외
class ConfigurationError(Exception):
    """Configuration validation failed"""
    pass

class IterationRangeError(ConfigurationError):
    """Invalid iteration range in config"""
    pass

# 사용
def validate_config(cfg):
    if cfg.start >= cfg.end:
        raise IterationRangeError(
            f"start ({cfg.start}) must be < end ({cfg.end})"
        )
```

---

## 6. 검증과 단언

### 6.1 입력 검증 (Input Validation)

```python
def train_model(
    data: List,
    epochs: int,
    lr: float,
    checkpoint_dir: Optional[str] = None
):
    """Train model with validation"""

    # 1. 필수 인자 검증
    if not data:
        raise ValueError("data cannot be empty")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if lr <= 0 or lr > 1:
        raise ValueError(f"lr must be in (0, 1], got {lr}")

    # 2. Optional 인자 기본값
    if checkpoint_dir is None:
        checkpoint_dir = "./checkpoints"

    # 3. 환경 검증
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 이제 안전하게 진행
    for epoch in range(epochs):
        ...
```

### 6.2 중간 상태 검증 (Assertion)

```python
# ✅ Good: 개발 중 assertion
def forward(self, x, total_iter):
    # 입력 크기 검증
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
    assert x.shape[1] == 3, f"Expected 3 channels, got {x.shape[1]}"

    # 처리
    features = self.encoder(x)

    # 중간 결과 검증
    assert features is not None, "encoder returned None"
    assert not torch.isnan(features).any(), "NaN in features"

    return features

# 주의: Production에서는 assertion 대신 에러 처리 권장
# (assert는 -O 옵션으로 비활성화 가능)
```

### 6.3 출력 검증 (Output Validation)

```python
def predict(self, x):
    """Predict with output validation"""
    logits = self.model(x)

    # 출력 검증
    if logits is None:
        raise RuntimeError("Model returned None")

    if torch.isnan(logits).any():
        raise RuntimeError("NaN in model output")

    if torch.isinf(logits).any():
        raise RuntimeError("Inf in model output")

    # 범위 검증
    probs = torch.softmax(logits, dim=-1)
    assert (probs >= 0).all() and (probs <= 1).all(), "Invalid probabilities"

    return probs
```

---

## 7. 실전 예제

### 7.1 Before/After: arti_params 버그

#### Before (버그)

```python
def save_results(self, log):
    """Save training results"""
    b0 = log.batch_size * log.num_frames
    fnames = [f'{log.total_iter:07d}_{fid:10d}'
              for fid in collapseBF(log.global_frame_id.int())][:b0]

    # ❌ 문제: arti_params가 None일 수 있음
    misc.save_txt(
        log.save_dir,
        rearrange(log.arti_params, "b f n c -> (b f) n c").cpu().numpy(),
        suffix='arti_params',
        fnames=fnames,
        delim=' '
    )
```

**문제점:**
1. `log.arti_params`가 None일 수 있음 (iteration < 20,000)
2. `rearrange(None, ...)` → RuntimeError
3. None 체크 없음

#### After (수정)

```python
def save_results(self, log):
    """Save training results with defensive checks"""
    b0 = log.batch_size * log.num_frames
    fnames = [f'{log.total_iter:07d}_{fid:10d}'
              for fid in collapseBF(log.global_frame_id.int())][:b0]

    # ✅ 해결: None 체크 추가
    if log.arti_params is not None:
        misc.save_txt(
            log.save_dir,
            rearrange(log.arti_params, "b f n c -> (b f) n c").cpu().numpy(),
            suffix='arti_params',
            fnames=fnames,
            delim=' '
        )
    # else: arti_params 없어도 OK (조기 학습 단계)
```

**개선점:**
1. None 체크로 안전성 확보
2. 다른 optional 변수들과 일관성 유지
3. 모든 iteration에서 안전하게 작동

### 7.2 Before/After: Config 검증

#### Before (검증 없음)

```python
def main():
    cfg = load_config()

    # ❌ Config 검증 없이 바로 사용
    trainer = Trainer(cfg)
    trainer.train()  # 수 시간 후 에러 발생 가능!
```

#### After (검증 추가)

```python
def validate_config(cfg):
    """Validate config before training"""
    errors = []

    # 1. 필수 필드 검증
    required = ['model', 'dataset', 'training']
    for field in required:
        if not hasattr(cfg, field):
            errors.append(f"Missing required field: {field}")

    # 2. 값 범위 검증
    if cfg.training.lr <= 0 or cfg.training.lr > 1:
        errors.append(f"Invalid lr: {cfg.training.lr} (must be in (0, 1])")

    # 3. Iteration range 검증
    if hasattr(cfg.model, 'cfg_articulation'):
        arti_start = cfg.model.cfg_articulation.articulation_iter_range[0]
        if hasattr(cfg.loss, 'arti_reg_loss_iter_range'):
            loss_start = cfg.loss.arti_reg_loss_iter_range[0]
            if loss_start < arti_start:
                errors.append(
                    f"arti_reg_loss starts at {loss_start} "
                    f"but articulation starts at {arti_start}"
                )

    # 4. 에러 처리
    if errors:
        error_msg = "Config validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        raise ConfigurationError(error_msg)

    return True

def main():
    cfg = load_config()

    # ✅ 검증 후 사용
    try:
        validate_config(cfg)
    except ConfigurationError as e:
        print(f"❌ {e}")
        print("Please fix your config and try again.")
        sys.exit(1)

    # 이제 안전하게 진행
    trainer = Trainer(cfg)
    trainer.train()
```

### 7.3 Before/After: 함수 방어 강화

#### Before

```python
def forward_articulation(self, shape, features):
    """Forward pass for articulation"""
    # ❌ 입력 검증 없음
    arti_params = self.netArticulation(features)
    shape = self.apply_articulation(shape, arti_params)
    return shape, arti_params
```

#### After

```python
def forward_articulation(
    self,
    shape: Mesh,
    features: torch.Tensor
) -> Tuple[Mesh, torch.Tensor]:
    """
    Forward pass for articulation with defensive checks.

    Args:
        shape: Input mesh
        features: Encoded features [B, C, H, W]

    Returns:
        shape: Articulated mesh
        arti_params: Articulation parameters [B, F, N, C]

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If articulation fails
    """
    # 1. 입력 검증
    if shape is None:
        raise ValueError("shape cannot be None")
    if features is None:
        raise ValueError("features cannot be None")

    # 2. 타입 검증
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(features)}")

    # 3. 크기 검증
    if features.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {features.ndim}D")

    # 4. 안전하게 처리
    try:
        arti_params = self.netArticulation(features)
    except Exception as e:
        raise RuntimeError(f"Articulation network failed: {e}")

    # 5. 출력 검증
    if arti_params is None:
        raise RuntimeError("Articulation network returned None")

    if torch.isnan(arti_params).any():
        raise RuntimeError("NaN in articulation parameters")

    # 6. 적용
    shape = self.apply_articulation(shape, arti_params)

    return shape, arti_params
```

---

## 8. 체크리스트

### 개발 중

- [ ] 모든 입력에 대해 검증 추가
- [ ] None 가능성 있는 변수는 사용 전 체크
- [ ] 타입 힌트 추가
- [ ] Assertion으로 중간 상태 검증
- [ ] 에러 메시지 명확하게 작성

### 코드 리뷰 시

- [ ] None 체크 누락 없는지
- [ ] 타입 검증 충분한지
- [ ] 에러 처리 적절한지
- [ ] Edge case 고려했는지
- [ ] 입력 검증 있는지

### 테스트 시

- [ ] 정상 입력 테스트
- [ ] None 입력 테스트
- [ ] 잘못된 타입 입력 테스트
- [ ] 경계값 테스트
- [ ] 예외 상황 테스트

---

## 9. 참고 자료

### Python 공식 문서
- [PEP 8 - Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)

### 추천 도서
- "The Pragmatic Programmer" - Andrew Hunt, David Thomas
- "Code Complete" - Steve McConnell
- "Clean Code" - Robert C. Martin

### 프로젝트 내 문서
- `docs/reports/20251109_arti_params_none_error.md`
- `docs/guides/progressive_training_best_practices.md`

---

**방어적 프로그래밍은 습관입니다. 처음에는 번거롭지만, 곧 자연스러워집니다!**
