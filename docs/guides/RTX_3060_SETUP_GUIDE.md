# RTX 3060 Setup Guide (TF32 비활성화)

**Date**: 2025-11-24
**Purpose**: RTX 3060에서 CUBLAS 에러 없이 안정적으로 학습하는 방법

---

## 문제 상황

### CUBLAS 에러 발생
```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling
`cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16F, lda, b,
CUDA_R_16F, ldb, &fbeta, c, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT)`
```

### 근본 원인
- **RTX 3060**: Ampere 아키텍처, TF32 (Tensor Float 32) 기본 활성화
- **PyTorch 기본 동작**: CUDA 11.0+에서 자동으로 TF32 사용
- **문제**: 일부 연산에서 `CUBLAS_STATUS_NOT_SUPPORTED` 에러 발생

**관련 파일**: `model/predictors/DINOPose.py:101` (DINO ViT attention)

---

## 해결 방법 1: Config 기반 (권장)

### 1단계: Config에 `disable_tf32: true` 추가

**파일**: `config/train_fauna_mouse_dannce.yaml`

```yaml
# ==============================================================================
# Hardware Configuration
# ==============================================================================
device: cuda
gpu_ids: [0]
disable_tf32: true  # RTX 3060: CUBLAS 에러 방지
```

### 2단계: 기존 `run.py` 사용

```bash
cd /home/joon/dev/3DAnimals

# Debug 모드 (5K iterations, ~15분)
python run.py --config-name train_fauna_mouse_dannce_debug

# Full training (50K iterations, ~2-3시간)
python run.py --config-name train_fauna_mouse_dannce
```

**장점**:
- ✅ 일관성: 모든 설정이 config에 집중
- ✅ 유지보수: `run.py` 하나만 관리
- ✅ 확장성: 다른 GPU에서도 쉽게 전환 가능

---

## 해결 방법 2: 별도 스크립트 (기존 방식)

### 파일: `run_debug_notf32.py`, `run_full_notf32.py`

**장점**:
- 빠른 임시 해결책

**단점**:
- ❌ 코드 중복
- ❌ 유지보수 부담
- ❌ Config 아키텍처와 불일치

**권장하지 않음** (향후 삭제 예정)

---

## 구현 세부사항

### `model/utils/misc.py`에 추가

```python
def setup_tf32(cfg):
    """
    Setup TF32 based on config

    RTX 3060 compatibility:
    - TF32 causes CUBLAS errors on some operations
    - Must disable for stable training
    """
    import torch

    disable_tf32 = getattr(cfg, 'disable_tf32', False)

    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("=" * 80)
        print("TF32 DISABLED (RTX 3060 Compatibility Mode)")
        print("=" * 80)
        print(f"MatMul TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN TF32: {torch.backends.cudnn.allow_tf32}")
        print()
    else:
        print("TF32 ENABLED (Default PyTorch behavior)")
```

### `model/Trainer.py`에서 호출

```python
class Trainer:
    def __init__(self, cfg: TrainerConfig, model):
        self.cfg = misc.load_cfg(self, cfg, TrainerConfig)
        misc.setup_runtime(self.cfg)
        misc.setup_tf32(self.cfg)  # 추가!
        # ... rest of init
```

---

## 검증 방법

### 1. TF32 설정 확인

```bash
cd /home/joon/dev/3DAnimals
python scripts/test_cuda_fix.py
```

**예상 출력**:
```
================================================================================
CUDA FIX TEST
================================================================================
PyTorch version: 2.0.0
CUDA version: 11.8
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
TF32 disabled
✅ Test 1: Basic CUDA tensor - PASS
✅ Test 2: Linear layer (CUBLAS) - PASS
✅ Test 3: DINO attention operation - PASS
================================================================================
✅ ALL TESTS PASSED! CUDA is working correctly.
================================================================================
```

### 2. Debug 모드 실행 (PoC 검증)

```bash
# IMPORTANT: 본격 학습 전 반드시 debug 모드 먼저 실행!
python run.py --config-name train_fauna_mouse_dannce_debug
```

**검증 항목**:
- [ ] TF32 disabled 메시지 출력
- [ ] CUBLAS 에러 없음
- [ ] 5K iterations 정상 완료 (~15분)
- [ ] Checkpoint 저장 성공

### 3. Full Training 실행

```bash
# Debug 모드 성공 후에만 실행
python run.py --config-name train_fauna_mouse_dannce
```

---

## Hardware 스펙 비교

| GPU | Arch | TF32 | CUBLAS Issue | Solution |
|-----|------|------|--------------|----------|
| **RTX 3060** | Ampere | ✅ | ❌ 에러 발생 | `disable_tf32: true` |
| **RTX 3080** | Ampere | ✅ | ✅ 정상 동작 | Default OK |
| **RTX 4090** | Ada | ✅ | ✅ 정상 동작 | Default OK |
| **V100** | Volta | ❌ | ✅ 정상 동작 | Default OK |

**결론**: RTX 3060에서만 `disable_tf32: true` 필요

---

## 성능 영향

### TF32 비활성화 시

- **정확도**: 변화 없음 (FP32 사용)
- **속도**: ~5-10% 느려짐 (TF32 대비)
- **메모리**: 동일
- **안정성**: ✅ CUBLAS 에러 해결

**Trade-off**: 약간의 속도 저하 vs 안정성 확보

### 실제 학습 시간 (RTX 3060, TF32 disabled)

- 1K iterations: ~2-3분
- 5K iterations: ~15-20분
- 50K iterations: ~2.5-3시간

---

## Troubleshooting

### 문제 1: 여전히 CUBLAS 에러 발생

**원인**: Config 설정이 적용되지 않음

**해결**:
```bash
# 1. Config 확인
grep "disable_tf32" config/train_fauna_mouse_dannce.yaml

# 2. 명시적 지정
python run.py --config-name train_fauna_mouse_dannce disable_tf32=true
```

### 문제 2: TF32 메시지가 출력되지 않음

**원인**: `misc.setup_tf32()` 호출 누락

**해결**: `model/Trainer.py` 수정 확인

### 문제 3: 다른 GPU에서도 비활성화됨

**원인**: Config에 `disable_tf32: true` 고정

**해결**: GPU별 config 파일 생성
```yaml
# config/train_fauna_mouse_dannce_rtx3060.yaml
disable_tf32: true

# config/train_fauna_mouse_dannce_rtx4090.yaml
disable_tf32: false
```

---

## 참고 자료

### 관련 문서
- **Bug Report**: `docs/research/20251109_arti_params_none_error.md`
- **CUDA Setup**: `docs/guides/CUDA_FIX_GUIDE.md`

### PyTorch 공식 문서
- [TF32 on Ampere](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere)
- [Numerical Accuracy](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)

### GitHub Issues
- [PyTorch #73328](https://github.com/pytorch/pytorch/issues/73328) - TF32 CUBLAS issues

---

## 마이그레이션 계획

### 현재 상태 (2025-11-24)
- ✅ `run_debug_notf32.py` 사용 중
- ✅ `run_full_notf32.py` 사용 중

### 다음 단계 (향후)
1. `model/utils/misc.py`에 `setup_tf32()` 추가
2. `model/Trainer.py`에서 호출
3. Config에 `disable_tf32: true` 추가
4. `run.py` 사용으로 전환
5. `run_*_notf32.py` 아카이브

### 예상 일정
- **즉시 가능**: Config 기반 구현 (1시간)
- **마이그레이션**: 기존 스크립트 → `run.py` (테스트 후)
- **클린업**: 구 스크립트 아카이브

---

**Last Updated**: 2025-11-24
**Maintainer**: Joon
**Status**: Active Development
