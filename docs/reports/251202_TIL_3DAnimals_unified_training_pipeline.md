# 📄 3DAnimals_Unified_Training_Pipeline_Setup

**대화 요약**: 3DAnimals 프로젝트에서 MagicPony, Ponymation, Fauna 세 모델의 통합 학습 파이프라인을 구축하고, mouse 데이터에 대한 finetune/scratch 학습 환경을 완성하며, wandb 로깅 체계와 디버깅 이슈를 해결함

**주요 다룬 주제**:

1. 세 모델(MagicPony, Ponymation, Fauna) 아키텍처 비교 및 학습 의존성
2. Pretrained 모델 활용 전략 및 Finetune vs Scratch 선택
3. 데이터 포맷 변환 및 DataLoader 호환성 문제 해결
4. Wandb 프로젝트/run 구분 체계 구축
5. Production 코드 디버깅 (PDB breakpoint 제거)

---

## 1. 모델 아키텍처 및 학습 흐름

### 1.1 세 모델 핵심 비교

| 모델 | 입력 | Dataset 클래스 | 학습 단계 | 특징 |
|------|------|---------------|----------|------|
| **MagicPony** | 단일 이미지 | `ImageDataset` | 1-stage | Shape/Texture 학습 |
| **Ponymation** | 비디오 시퀀스 | `NFrameSequenceDataset` | 2-stage | Motion VAE |
| **Fauna** | 멀티 동물 이미지 | `FaunaDataset` | 1-stage | Memory Bank (60 shapes) |

### 1.2 Ponymation 2-Stage 학습 흐름

```
MagicPony checkpoint (shape/texture prior)
         ↓
Stage 1: Articulation 학습
- enable_motion_vae: false
- enable_render: true
- artivel_smooth_loss_weight: 1
         ↓
Stage 2: Motion VAE 학습
- enable_motion_vae: true
- enable_render: false (속도 향상)
- arti_recon_loss_weight: 100.0
- kld_loss_weight: 0.001
```

### 1.3 Finetune vs From Scratch

- **Finetune 권장 이유**: Horse pretrained가 quadruped prior 제공, 수렴 빠름 (100K vs 200K)
- **From Scratch 가능**: MagicPony, Fauna 모두 scratch config 존재
- **Ponymation**: 반드시 MagicPony checkpoint 필요 (scratch 불가)

## 2. Pretrained 모델 및 추론

### 2.1 사용 가능한 Pretrained 모델

```
results/
├── magicpony/pretrained_horse/pretrained_horse.pth (149M)
├── ponymation/pretrained_horse/
│   ├── pretrained_horse_stage1.pth (197M)
│   └── pretrained_horse_stage2.pth (349M)
└── fauna/pretrained_fauna/pretrained_fauna.pth
```

### 2.2 추론 실행 방법

```bash
# MagicPony 추론
python run.py --config-name test_magicpony_mouse

# Ponymation 추론 (Stage 2 완료 모델)
python run.py --config-name test_ponymation_mouse
```

**핵심 config 차이**:
```yaml
# Training
run_train: true
run_test: false

# Inference
run_train: false
run_test: true
```

## 3. 데이터 변환 및 호환성

### 3.1 Ponymation 데이터 구조 문제

- **문제상황**: `frame_0/rgb.png` 구조 생성 → DataLoader가 flat 구조 기대 → `num_samples=0` 에러
- **해결방법**: `convert_ponymation_frames_to_flat.py` 스크립트 작성

```python
# Before (잘못된 구조)
sequence/
├── frame_0/
│   ├── rgb.png
│   └── mask.png

# After (올바른 구조)
sequence/
├── 000000rgb.png
├── 000000mask.png
├── 000001rgb.png
└── ...
```

### 3.2 변환 명령어

```bash
# Dry-run으로 확인
python scripts/convert_ponymation_frames_to_flat.py \
    --data-dir data/ponymation/mouse --dry-run

# 실제 변환
python scripts/convert_ponymation_frames_to_flat.py \
    --data-dir data/ponymation/mouse
```

## 4. Wandb 로깅 체계 구축

### 4.1 문제점

- 모든 실험이 `model.name` (Fauna/MagicPony/Ponymation)으로만 구분
- 같은 프로젝트 내 수많은 run이 섞여 구분 어려움

### 4.2 해결: Trainer.py 수정

```python
# TrainerConfig에 새 필드 추가
wandb_project: str = None  # Override wandb project name
wandb_run_name: str = None  # Override wandb run name

# WandbWriter 호출 시 적용
wandb_project = self.wandb_project if self.wandb_project else self.model.name
self.logger = WandbWriter(
    project=wandb_project,
    name=self.wandb_run_name
)
```

### 4.3 Config 설정 패턴

```yaml
# 예: finetune_magicpony_mouse.yaml
use_logger: true
logger_type: wandb
wandb_project: MagicPony_mouse
wandb_run_name: finetune_${now:%Y%m%d_%H%M%S}
```

### 4.4 최종 프로젝트 구조

| Wandb Project | Run Name 패턴 |
|---------------|---------------|
| `Fauna_mouse` | `large_scratch_...`, `6view_finetune_...` |
| `MagicPony_mouse` | `scratch_...`, `finetune_...`, `finetune_debug_...` |
| `MagicPony_mouse_6view` | `scratch_...` |
| `Ponymation_mouse` | `stage1_finetune_...`, `stage2_finetune_debug_...` |

## 5. 통합 학습 스크립트

### 5.1 train_unified.sh 사용법

```bash
./scripts/train_unified.sh <model> <mode> [training_type]

# 예시
./scripts/train_unified.sh magicpony debug finetune
./scripts/train_unified.sh ponymation-s1 full finetune
./scripts/train_unified.sh ponymation-s2 debug finetune
./scripts/train_unified.sh fauna debug scratch
```

### 5.2 지원 옵션

- **Models**: `fauna`, `magicpony`, `ponymation-s1`, `ponymation-s2`
- **Modes**: `debug` (10-20분), `full` (전체), `background`
- **Types**: `scratch` (default for fauna), `finetune` (default for others)

## 6. 디버깅 및 문제 해결

### 6.1 PDB Breakpoint로 학습 중단 문제

- **문제상황**: `skinning.py:184`에서 `pdb.set_trace()`로 학습 멈춤
- **원인**: Quadrant에 점이 없는 edge case (iteration 20,000에서 발생)
- **해결방법**: Fallback 로직으로 대체

```python
# Before (학습 중단)
if len(quadrant_points.view(-1)) < 1:
    import pdb; pdb.set_trace()

# After (계속 진행)
if len(quadrant_points.view(-1)) < 1:
    print(f"[WARNING] No points in quadrant. Using fallback.")
    quadrant_points = seq_shape[b, f].mean(dim=0, keepdim=True)
```

### 6.2 PDB 기본 명령어

| 명령어 | 설명 |
|--------|------|
| `c` / `continue` | 다음 breakpoint까지 실행 |
| `q` / `quit` | 프로그램 종료 |
| `n` / `next` | 다음 줄 실행 |
| `s` / `step` | 함수 안으로 진입 |
| `l` / `list` | 현재 코드 보기 |
| `p 변수명` | 변수 값 출력 |

### 6.3 학습 이어서 하기

```bash
# checkpoint에서 자동 resume (resume: true 설정 시)
python run.py --config-name train_fauna_mouse_large_finetune

# 특정 checkpoint 지정
python run.py --config-name train_fauna_mouse_large_finetune \
    checkpoint_path=results/fauna_mouse_large_finetune/checkpoint19500.pth
```

---

## 💡 대화에서 얻은 핵심 인사이트

1. **Finetune 우선 전략**: Horse pretrained 모델이 quadruped 동물에 강력한 prior 제공, mouse 학습 시 scratch보다 finetune이 효율적 (iteration 절반)

2. **DataLoader 구조 일치 필수**: 모델마다 기대하는 데이터 구조가 다름. `NFrameSequenceDataset`은 flat 구조 (`000000rgb.png`) 필요

3. **Production 코드 디버그 제거**: `pdb.set_trace()`가 남아있으면 학습 중단 위험. Edge case는 fallback 로직으로 처리하고 warning 로그만 남기기

4. **Wandb 체계적 관리**: 프로젝트명에 모델+데이터셋, run name에 학습타입+타임스탬프로 구분하면 실험 추적 용이

## ❓ 미해결 질문 또는 추가 학습 필요 사항

- Quadrant가 비어있는 근본 원인 (mouse 형상 특성? mask 품질?)
- Fauna에서 mouse-specific bone structure 최적화 방법
- Ponymation Stage 2 Motion VAE 품질 정량 평가 방법
- Multi-GPU 분산 학습 설정

## 🔗 참고 자료 및 키워드

**핵심 키워드**:
- DMTet (Deep Marching Tetrahedra)
- SDF (Signed Distance Function)
- Progressive Training
- Motion VAE
- Articulation / Skinning
- Memory Bank

**핵심 파일**:
```
scripts/
├── train_unified.sh                    # 통합 학습 스크립트
├── convert_ponymation_frames_to_flat.py # 데이터 변환

config/
├── finetune_magicpony_mouse*.yaml      # MagicPony finetune
├── finetune_ponymation_mouse_stage*.yaml # Ponymation finetune
├── train_fauna_mouse_large*.yaml       # Fauna 학습

model/
├── Trainer.py                          # wandb_project/run_name 추가
└── geometry/skinning.py                # PDB 제거, fallback 추가
```

**tmux 팁** (bonus):
```bash
# tmux에서 마우스 스크롤 활성화
set -g mouse on
```
