# 📄 3DAnimals_MagicPony_Ponymation_Fauna_Training_Pipeline

**대화 요약**: MagicPony, Ponymation, Fauna 세 가지 3D 동물 재구성 모델의 학습 파이프라인을 통합 구축하고, mouse 데이터에 대한 finetune/scratch 학습 환경을 완성함

**주요 다룬 주제**:

1. 세 모델(MagicPony, Ponymation, Fauna)의 아키텍처 비교 및 학습 흐름
2. Pretrained 모델 활용 및 Finetune 전략
3. Wandb 로깅 시스템 개선 및 디버깅

---

## 1. 모델 아키텍처 비교

### 1.1 MagicPony

- **핵심개념**: 단일 이미지 → 3D 재구성
- **데이터셋**: `ImageDataset` (개별 이미지)
- **특징**:
  - 1-stage 학습
  - `BasePredictorBase` 사용
  - Horse pretrained 모델 제공

### 1.2 Ponymation

- **핵심개념**: 비디오 기반 4D 모션 재구성
- **데이터셋**: `NFrameSequenceDataset` (시퀀스)
- **2-Stage 학습**:
  - Stage 1: Articulation 학습 (MagicPony checkpoint에서 시작)
  - Stage 2: Motion VAE 학습 (Stage 1 checkpoint에서 시작)
- **데이터 구조**: flat 형식 필요 (`000000rgb.png`, `000000mask.png`)

### 1.3 Fauna

- **핵심개념**: 멀티 동물 지원 범용 3D 재구성
- **데이터셋**: `FaunaDataset`
- **특징**:
  - `BasePredictorBank` (60-shape memory bank)
  - Discriminator 사용
  - Progressive training

## 2. Training Flow 및 Finetune 전략

### 2.1 권장 학습 순서 (Finetune)

```
MagicPony pretrained horse
       ↓
MagicPony mouse finetune (선택적)
       ↓
Ponymation Stage 1 (articulation)
       ↓
Ponymation Stage 2 (Motion VAE)
```

### 2.2 From Scratch vs Finetune

| 방식 | 장점 | 단점 |
|------|------|------|
| From Scratch | 새로운 형상에 최적화 | 200K+ iterations 필요 |
| Finetune | 빠른 수렴 (100K) | Horse prior에 의존 |

### 2.3 Pretrained 모델 현황

- **MagicPony**: `pretrained_horse.pth` (149M) ✓
- **Ponymation**: `pretrained_horse_stage1.pth` (197M), `stage2.pth` (349M) ✓
- **Fauna**: `pretrained_fauna.pth` ✓

## 3. 데이터 변환 및 구조

### 3.1 Ponymation 데이터 구조 문제

- **문제상황**: `frame_0/rgb.png` 구조로 생성했으나 DataLoader가 flat 구조 기대
- **해결방법**: `convert_ponymation_frames_to_flat.py` 스크립트 작성

```python
# Before (잘못된 구조)
sequence/frame_0/rgb.png

# After (올바른 구조)
sequence/000000rgb.png
```

### 3.2 변환 명령어

```bash
python scripts/convert_ponymation_frames_to_flat.py \
    --data-dir data/ponymation/mouse
```

## 4. Wandb 로깅 시스템 개선

### 4.1 문제점

- 모든 모델이 동일한 project명 (`Fauna`, `MagicPony`, `Ponymation`)으로 로깅
- Run 구분 어려움

### 4.2 해결 방법

**Trainer.py 수정**:
```python
# 새로운 config 옵션 추가
wandb_project: str = None  # Override wandb project name
wandb_run_name: str = None  # Override wandb run name
```

**Config 설정 예시**:
```yaml
wandb_project: MagicPony_mouse
wandb_run_name: finetune_${now:%Y%m%d_%H%M%S}
```

### 4.3 프로젝트 구분 체계

| 프로젝트 | Run Name 패턴 |
|---------|--------------|
| `Fauna_mouse` | `large_scratch_...`, `6view_finetune_...` |
| `MagicPony_mouse` | `scratch_...`, `finetune_...` |
| `Ponymation_mouse` | `stage1_finetune_...`, `stage2_finetune_...` |

## 5. 디버깅 및 문제 해결

### 5.1 PDB Breakpoint 문제

- **문제상황**: `skinning.py`에서 `pdb.set_trace()`로 학습 중단
- **원인**: Quadrant에 점이 없는 edge case
- **해결방법**: Fallback 로직으로 대체

```python
# Before
if len(quadrant_points.view(-1)) < 1:
    import pdb; pdb.set_trace()

# After
if len(quadrant_points.view(-1)) < 1:
    print(f"[WARNING] No points in quadrant. Using fallback.")
    quadrant_points = seq_shape[b, f].mean(dim=0, keepdim=True)
```

### 5.2 PDB 기본 명령어

| 명령어 | 설명 |
|--------|------|
| `c` / `continue` | 계속 실행 |
| `q` / `quit` | 프로그램 종료 |
| `n` / `next` | 다음 줄 |
| `p 변수명` | 변수 출력 |

## 6. 통합 학습 스크립트

### 6.1 train_unified.sh 사용법

```bash
# 기본 사용법
./scripts/train_unified.sh <model> <mode> [training_type]

# 예시
./scripts/train_unified.sh magicpony debug finetune
./scripts/train_unified.sh ponymation-s1 full finetune
./scripts/train_unified.sh fauna debug scratch
```

### 6.2 지원 모델 및 모드

- **Models**: `fauna`, `magicpony`, `ponymation-s1`, `ponymation-s2`
- **Modes**: `debug` (빠른 검증), `full` (전체), `background` (백그라운드)
- **Types**: `scratch`, `finetune`

---

## 💡 대화에서 얻은 핵심 인사이트

1. **Finetune 우선 전략**: Horse pretrained 모델이 quadruped 동물에 좋은 prior 제공, mouse 학습 시 finetune이 효율적
2. **데이터 구조의 중요성**: DataLoader가 기대하는 구조와 실제 데이터 구조 불일치 시 `num_samples=0` 에러 발생
3. **디버그 코드 관리**: Production 코드에 `pdb.set_trace()` 남기면 학습 중단 위험, fallback 로직으로 대체 필요

## ❓ 미해결 질문 또는 추가 학습 필요 사항

- Quadrant 비어있는 원인 분석 (mouse 형상 특성? 데이터 품질?)
- Fauna에서 mouse-specific bone structure 최적화 필요 여부
- Stage 2 Motion VAE의 실제 품질 평가 방법

## 🔗 참고 자료 및 키워드

- **키워드**: DMTet, SDF, Progressive Training, Motion VAE, Articulation, Skinning
- **핵심 파일**:
  - `scripts/train_unified.sh` - 통합 학습 스크립트
  - `scripts/convert_ponymation_frames_to_flat.py` - 데이터 변환
  - `model/geometry/skinning.py` - Bone/Joint 계산
  - `model/Trainer.py` - Wandb 설정
- **Config 위치**: `config/finetune_*.yaml`, `config/train_*.yaml`
