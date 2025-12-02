# MagicPony, Ponymation, Fauna 모델 비교 분석 보고서

**날짜**: 2025-12-02 (업데이트)
**목적**: 세 가지 3D 동물 재구성 모델의 설정, 코드, 데이터셋 요구사항 비교 분석 및 통합 학습/추론 가이드

---

## 1. 모델 개요 비교

| 특성 | MagicPony | Ponymation | Fauna |
|------|-----------|-----------|-------|
| **목적** | 단일 이미지 3D 재구성 | 비디오 기반 4D 동작 생성 | 범용 동물 3D 재구성 |
| **입력** | 단일 이미지 | 비디오 시퀀스 (10 프레임) | 단일 이미지 (다중 뷰 가능) |
| **출력** | 3D 메쉬 + 텍스처 | 3D 메쉬 + 동작 시퀀스 | 3D 메쉬 + 텍스처 |
| **학습 단계** | 1단계 | 2단계 (관절 학습 → VAE) | 1단계 |
| **Base Predictor** | BasePredictorBase | BasePredictorBase | BasePredictorBank (메모리 뱅크) |
| **Instance Predictor** | InstancePredictorBase | InstancePredictorMotionVAE | InstancePredictorFauna |

---

## 2. 사전 학습 모델 다운로드

### 2.1 다운로드 스크립트 위치

```
results/
├── fauna/download_pretrained_fauna.sh      # Fauna 사전학습 모델
├── magicpony/download_pretrained_magicpony.sh  # MagicPony 사전학습 모델
└── ponymation/download_pretrained_ponymation.sh # Ponymation 사전학습 모델
```

### 2.2 다운로드 명령어

```bash
# Fauna 모델 다운로드
cd results/fauna
bash download_pretrained_fauna.sh
# 출력: pretrained_fauna/pretrained_fauna.pth (~160MB)

# MagicPony 모델 다운로드 (동물별)
cd results/magicpony
bash download_pretrained_magicpony.sh
# 출력: pretrained_horse/, pretrained_cow/, pretrained_giraffe/, pretrained_zebra/, pretrained_bird/

# Ponymation 모델 다운로드 (동물별)
cd results/ponymation
bash download_pretrained_ponymation.sh
# 출력: pretrained_horse/, pretrained_cow/, pretrained_giraffe/, pretrained_zebra/
```

### 2.3 다운로드 URL (직접 다운로드 시)

```
# Fauna
https://download.cs.stanford.edu/viscam/3DAnimals/models/fauna/pretrained_fauna.zip

# MagicPony
https://download.cs.stanford.edu/viscam/3DAnimals/models/magicpony/pretrained_{animal}.zip
# animal: horse, cow, giraffe, zebra, bird

# Ponymation
https://download.cs.stanford.edu/viscam/3DAnimals/models/ponymation/pretrained_{animal}.zip
# animal: horse, cow, giraffe, zebra
```

### 2.4 현재 다운로드 상태

| 모델 | 상태 | 위치 |
|------|------|------|
| Fauna | ✅ 다운로드됨 | `results/fauna/pretrained_fauna/pretrained_fauna.pth` |
| MagicPony | ❌ 미다운로드 | `results/magicpony/` (스크립트만 있음) |
| Ponymation | ❌ 미다운로드 | `results/ponymation/` (스크립트만 있음) |

---

## 3. 추론 (Inference) 설정

### 3.1 추론 설정 파일

```
config/
├── test_fauna.yaml              # Fauna 추론
├── test_magicpony_horse.yaml    # MagicPony (말)
├── test_magicpony_cow.yaml      # MagicPony (소)
├── test_magicpony_giraffe.yaml  # MagicPony (기린)
├── test_magicpony_zebra.yaml    # MagicPony (얼룩말)
├── test_magicpony_bird.yaml     # MagicPony (새)
├── test_ponymation_horse.yaml   # Ponymation (말)
├── test_ponymation_cow.yaml     # Ponymation (소)
├── test_ponymation_giraffe.yaml # Ponymation (기린)
└── test_ponymation_zebra.yaml   # Ponymation (얼룩말)
```

### 3.2 추론 핵심 설정

```yaml
# 공통 추론 설정 패턴
run_train: false        # 학습 비활성화
run_test: true          # 추론 활성화
checkpoint_dir: results/{model}/pretrained_{animal}
checkpoint_name: pretrained_{animal}.pth

# 시각화 옵션
output_dir: results/{model}/pretrained_{animal}/visualization
render_modes: [input_view, other_views, rotation]
finetune_texture: false  # 텍스처 미세조정 (선택)
```

### 3.3 추론 실행 명령어

```bash
# Fauna 추론
python run.py --config-name test_fauna

# MagicPony 추론
python run.py --config-name test_magicpony_horse
python run.py --config-name test_magicpony_cow

# Ponymation 추론
python run.py --config-name test_ponymation_horse
python run.py --config-name test_ponymation_cow
```

### 3.4 커스텀 이미지 추론

```yaml
# test_fauna.yaml 수정 예시
dataset:
  test_data_dir: path/to/your/images  # 커스텀 데이터 경로
  # 필요 파일: *_rgb.png, *_mask.png, *_metadata.json
```

---

## 4. 데이터셋 구조 상세

### 4.1 MagicPony 데이터 형식 (ImageDataset)

```
data/magicpony/{animal}/
├── train/
│   ├── {image_id}/              # 이미지별 폴더
│   │   ├── rgb.png              # 256x256 RGB 이미지 (필수)
│   │   ├── mask.png             # 256x256 이진 마스크 (필수)
│   │   ├── metadata.json        # 메타데이터 (필수)
│   │   └── feat16.png           # DINO 특징 (선택, load_dino_feature=true)
│   └── ...
├── val/
│   └── (동일 구조)
└── test/
    └── (동일 구조)
```

**metadata.json 형식**:
```json
{
  "video_frame_id": 0,
  "crop_box_xyxy": [0, 0, 256, 256],
  "video_frame_width": 256,
  "video_frame_height": 256
}
```

### 4.2 Ponymation 데이터 형식 (NFrameSequenceDataset)

```
data/ponymation/{animal}/
├── train/
│   ├── {sequence_id}/           # 시퀀스별 폴더
│   │   ├── frame_0/             # 프레임 0
│   │   │   ├── rgb.png
│   │   │   ├── mask.png
│   │   │   ├── metadata.json
│   │   │   └── feat16.png
│   │   ├── frame_1/             # 프레임 1
│   │   │   └── ...
│   │   └── frame_9/             # 프레임 9 (최소 10 프레임)
│   └── ...
└── test/
    └── (동일 구조)
```

**중요**: Ponymation은 시간적 일관성이 있는 연속 프레임 필요 (최소 10 프레임)

### 4.3 Fauna 데이터 형식 (FaunaDataset)

```
data/fauna/{category}/
├── train/
│   ├── {sequence_id}/           # 시퀀스별 폴더
│   │   ├── {frame_id}_rgb.png   # RGB 이미지
│   │   ├── {frame_id}_mask.png  # 이진 마스크
│   │   ├── {frame_id}_box.txt   # 바운딩 박스 (선택)
│   │   └── {frame_id}_metadata.json  # 메타데이터
│   └── ...
└── val/  # 자동 분할 또는 수동 생성
```

### 4.4 데이터 요구사항 비교표

| 항목 | MagicPony | Ponymation | Fauna |
|------|-----------|-----------|-------|
| **이미지 해상도** | 256x256 권장 | 256x256 권장 | 256x256 권장 |
| **RGB 이미지** | 필수 | 필수 | 필수 |
| **마스크** | 필수 (이진) | 필수 (이진) | 필수 (이진) |
| **메타데이터** | metadata.json | metadata.json | box.txt 또는 metadata.json |
| **DINO 특징** | 선택 | 권장 | 선택 |
| **시퀀스 길이** | 1 (단일 이미지) | 10+ (연속 프레임) | 1+ (유연) |
| **폴더 구조** | 이미지별 폴더 | 시퀀스/프레임 폴더 | 시퀀스별 폴더 |

---

## 5. 현재 데이터셋 위치 및 심볼릭 링크

### 5.1 현재 데이터 구조

```
/home/joon/dev/3DAnimals/data/
├── fauna/                       # Fauna 공식 데이터셋
│   ├── Fauna_dataset -> /media/joon/kafka/data/3DAnimals/fauna/Fauna_dataset (심볼릭 링크)
│   ├── Mouse_only_dataset -> /media/joon/kafka/data/3DAnimals/fauna/Mouse_only_dataset (심볼릭 링크)
│   ├── large_scale/
│   ├── few_shot_animal3d/
│   └── mouse_6view_posesplatter/
├── fauna_mouse -> /home/joon/dev/project_splatter/data/fauna_mouse (심볼릭 링크)
│   └── large_scale/mouse_dannce_6view/train/  # 마우스 DANNCE 데이터
├── magicpony/
│   ├── download_magicpony_dataset.sh
│   └── mouse/                   # ✅ 변환 완료
│       ├── train/ (40개 샘플)
│       └── val/ (10개 샘플)
├── ponymation/
│   ├── download_ponymation_dataset.sh
│   └── mouse/                   # ✅ 변환 완료
│       ├── train/ (4개 시퀀스)
│       └── test/ (1개 시퀀스)
└── tets/
    └── download_tets.sh
```

### 5.2 심볼릭 링크 상태

| 링크 | 대상 | 상태 |
|------|------|------|
| `data/fauna_mouse` | `/home/joon/dev/project_splatter/data/fauna_mouse` | ✅ 정상 |
| `data/fauna/Fauna_dataset` | `/media/joon/kafka/data/3DAnimals/fauna/Fauna_dataset` | ✅ 정상 |
| `data/fauna/Mouse_only_dataset` | `/media/joon/kafka/data/3DAnimals/fauna/Mouse_only_dataset` | ✅ 정상 |

### 5.3 심볼릭 링크 생성 방법 (필요시)

```bash
# 새 데이터셋 심볼릭 링크 생성 예시
cd /home/joon/dev/3DAnimals/data
ln -s /path/to/source/dataset ./target_name

# 예: 외부 마우스 데이터셋 연결
ln -s /media/external/mouse_data ./mouse_external
```

---

## 6. 데이터 변환 워크플로우

### 6.1 변환 스크립트 위치

```
scripts/
├── convert_fauna_to_magicpony.py  # Fauna → MagicPony 변환
└── convert_fauna_to_ponymation.py # Fauna → Ponymation 변환 (신규)
```

### 6.2 Fauna → MagicPony 변환

```bash
cd /home/joon/dev/3DAnimals

# 기본 사용법
python scripts/convert_fauna_to_magicpony.py \
    --source data/fauna_mouse/large_scale/mouse_dannce_6view \
    --target data/magicpony/mouse \
    --copy  # 또는 --symlink (심볼릭 링크)

# 옵션 설명
# --source: Fauna 형식 소스 디렉토리 (train/val/test 하위 폴더 포함)
# --target: MagicPony 형식 출력 디렉토리
# --copy: 파일 복사 (--symlink: 심볼릭 링크)
```

**변환 결과**:
- 입력: `{seq_id}/{frame_id}_rgb.png`, `{frame_id}_mask.png`, `{frame_id}_metadata.json`
- 출력: `{seq_id}_{frame_id}/rgb.png`, `mask.png`, `metadata.json`

### 6.3 Fauna → Ponymation 변환

```bash
cd /home/joon/dev/3DAnimals

# 기본 사용법
python scripts/convert_fauna_to_ponymation.py \
    --source data/fauna_mouse/large_scale/mouse_dannce_6view \
    --target data/ponymation/mouse \
    --num-frames 10 \
    --min-frames 5 \
    --copy

# 옵션 설명
# --source: Fauna 형식 소스 디렉토리
# --target: Ponymation 형식 출력 디렉토리
# --num-frames: 시퀀스당 프레임 수 (기본: 10)
# --min-frames: 최소 프레임 수 (기본: 5)
# --copy: 파일 복사
```

**변환 결과**:
- 입력: `{seq_id}/{frame_id}_rgb.png` (시간순 정렬)
- 출력: `{seq_id}_{chunk}/frame_{n}/rgb.png`, `mask.png`, `metadata.json`

### 6.4 변환 검증

```bash
# MagicPony 변환 확인
ls data/magicpony/mouse/train/ | head -5
ls data/magicpony/mouse/train/$(ls data/magicpony/mouse/train/ | head -1)/

# Ponymation 변환 확인
ls data/ponymation/mouse/train/
ls data/ponymation/mouse/train/$(ls data/ponymation/mouse/train/ | head -1)/
```

---

## 7. 공식 데이터셋 다운로드

### 7.1 데이터셋 다운로드 스크립트

```
data/
├── fauna/download_fauna_dataset.sh       # Fauna 데이터셋 (~수 GB)
├── magicpony/download_magicpony_dataset.sh  # MagicPony 데이터셋
└── ponymation/download_ponymation_dataset.sh # Ponymation 데이터셋
```

### 7.2 다운로드 명령어

```bash
# Fauna 공식 데이터셋
cd data/fauna
bash download_fauna_dataset.sh
# 출력: Fauna_dataset/ (대용량)

# MagicPony 공식 데이터셋
cd data/magicpony
bash download_magicpony_dataset.sh
# 출력: horse_videos_multi/, horse_combined/, cow_coco/, giraffe_coco/, zebra_coco/, bird_videos_bonanza/

# Ponymation 공식 데이터셋
cd data/ponymation
bash download_ponymation_dataset.sh
# 출력: horse/, horse_stage2/, cow/, zebra/, giraffe/
```

---

## 8. 통합 학습 스크립트

### 8.1 사용법

```bash
cd /home/joon/dev/3DAnimals

# 도움말
./scripts/train_unified.sh

# Fauna 학습
./scripts/train_unified.sh fauna debug       # 빠른 테스트 (5K iters)
./scripts/train_unified.sh fauna full        # 전체 학습
./scripts/train_unified.sh fauna background  # 백그라운드 학습

# MagicPony 학습
./scripts/train_unified.sh magicpony debug
./scripts/train_unified.sh magicpony full

# Ponymation 학습 (2단계)
./scripts/train_unified.sh ponymation-s1 debug  # Stage 1
./scripts/train_unified.sh ponymation-s2 full   # Stage 2
```

### 8.2 학습 의존성

```
독립적:
  ├── Fauna       (바로 학습 가능)
  └── MagicPony   (바로 학습 가능)

의존적:
  ├── Ponymation Stage 1 → MagicPony 체크포인트 필요
  └── Ponymation Stage 2 → Stage 1 체크포인트 필요
```

### 8.3 학습 설정 파일

| 모델 | Debug | Full |
|------|-------|------|
| Fauna | `train_fauna_mouse_6view_debug.yaml` | `train_fauna_mouse_large.yaml` |
| MagicPony | `train_magicpony_mouse_debug.yaml` | `train_magicpony_mouse.yaml` |
| Ponymation S1 | `train_ponymation_mouse_stage1_debug.yaml` | `train_ponymation_mouse_stage1.yaml` |
| Ponymation S2 | - | `train_ponymation_mouse_stage2.yaml` |

---

## 9. 아키텍처 비교

### 9.1 모델 클래스 구조

```
MagicPony (AnimalModel)
├── netBase: BasePredictorBase
│   ├── netShape: DMTetGeometry (SDF)
│   └── netDINO: CoordMLP (특징 재구성)
└── netInstance: InstancePredictorBase
    ├── netEncoder: ViTEncoder (DINO ViT-S/8)
    ├── netTexture: CoordMLP
    ├── netPose: Encoder32
    ├── netDeform: CoordMLP
    ├── netArticulation: ArticulationNetwork
    └── netLight: DirectionalLight

Ponymation (AnimalModel)
├── netBase: BasePredictorBase
└── netInstance: InstancePredictorMotionVAE
    ├── (MagicPony 구성요소 모두 포함)
    └── netVAE: ArticulationVAE (Motion VAE)
        ├── Encoder: Transformer 기반
        └── Decoder: Transformer 기반

Fauna (AnimalModel)
├── netBase: BasePredictorBank
│   ├── netShape: DMTetGeometry (SDF)
│   ├── netDINO: CoordMLP
│   └── memory_bank: Shape Bank (60개 형상 저장)
├── netInstance: InstancePredictorFauna
│   ├── (MagicPony 구성요소 모두 포함)
│   └── bank_selection: 메모리 뱅크 선택 로직
└── netDisc: DCDiscriminator (마스크 판별자)
```

### 9.2 핵심 차이점

| 컴포넌트 | MagicPony | Ponymation | Fauna |
|----------|-----------|-----------|-------|
| Shape Prior | 단일 형상 | 단일 형상 | 60개 형상 뱅크 |
| Motion Modeling | 없음 | Motion VAE | 없음 |
| Discriminator | 없음 | 없음 | 마스크 판별자 |
| 시간적 모델링 | 없음 | Transformer 기반 | 없음 |

---

## 10. 학습 매개변수 비교

| 매개변수 | MagicPony | Ponymation Stage1 | Ponymation Stage2 | Fauna |
|---------|-----------|-------------------|-------------------|-------|
| **num_iters** | 100K | 10K | 100K-500K | 200K |
| **batch_size** | 2-8 | 1 | 4-10 | 6-8 |
| **num_frames** | 1 | 10 | 10 | 1 |
| **grid_res** | 64 | 64-128 | 64-128 | 64 |
| **lr_base** | 0.00005 | 0.00005 | 0.00005 | 0.001 |
| **lr_instance** | 0.00005 | 0.00005 | 0.00005 | 0.0001 |

---

## 11. GPU 메모리 요구사항

| 모델 | grid_res | batch_size | 예상 메모리 |
|------|----------|-----------|------------|
| Fauna | 64 | 6 | ~4GB |
| MagicPony | 64 | 2 | ~4GB |
| MagicPony | 128 | 2 | ~8GB |
| Ponymation S1 | 64 | 1 | ~6GB |
| Ponymation S2 | 64 | 4 | ~8GB |

**RTX 3060 (12GB) 권장 설정**:
- grid_res: 64
- batch_size: 2-4

---

## 12. 파일 위치 요약

```
/home/joon/dev/3DAnimals/
├── model/
│   ├── models/
│   │   ├── MagicPony.py
│   │   ├── Ponymation.py
│   │   └── Fauna.py
│   ├── predictors/
│   │   ├── BasePredictorBase.py
│   │   ├── BasePredictorBank.py
│   │   ├── InstancePredictorBase.py
│   │   ├── InstancePredictorMotionVAE.py
│   │   └── InstancePredictorFauna.py
│   ├── networks/
│   │   └── MotionVAE.py
│   └── dataset/
│       ├── ImageDataset.py
│       ├── SequenceDataset.py
│       └── FaunaDataset.py
├── config/
│   ├── model/
│   │   ├── magicpony.yaml
│   │   ├── magicpony_mouse.yaml
│   │   ├── ponymation.yaml
│   │   ├── ponymation_mouse.yaml
│   │   ├── fauna.yaml
│   │   └── fauna_mouse_6view.yaml
│   ├── dataset/
│   │   ├── image.yaml
│   │   ├── sequence.yaml
│   │   └── mouse_large.yaml
│   ├── train_*.yaml
│   └── test_*.yaml
├── data/
│   ├── fauna/
│   ├── fauna_mouse/ (심볼릭 링크)
│   ├── magicpony/mouse/
│   └── ponymation/mouse/
├── results/
│   ├── fauna/pretrained_fauna/
│   ├── magicpony/
│   └── ponymation/
└── scripts/
    ├── train_unified.sh
    ├── convert_fauna_to_magicpony.py
    └── convert_fauna_to_ponymation.py
```

---

## 13. Quick Start 가이드

### 13.1 추론 (사전학습 모델 사용)

```bash
# 1. 사전학습 모델 다운로드
cd results/fauna && bash download_pretrained_fauna.sh
cd results/magicpony && bash download_pretrained_magicpony.sh

# 2. 추론 실행
python run.py --config-name test_fauna
python run.py --config-name test_magicpony_horse
```

### 13.2 커스텀 데이터 학습 (마우스)

```bash
# 1. 데이터 변환
python scripts/convert_fauna_to_magicpony.py \
    --source data/fauna_mouse/large_scale/mouse_dannce_6view \
    --target data/magicpony/mouse --copy

# 2. 학습 실행
./scripts/train_unified.sh magicpony debug  # 테스트
./scripts/train_unified.sh magicpony full   # 전체 학습
```

### 13.3 새 데이터셋 준비

```bash
# 1. 데이터 준비 (Fauna 형식)
mkdir -p data/fauna_custom/train/seq_000
# 파일 배치: {frame_id}_rgb.png, {frame_id}_mask.png, {frame_id}_metadata.json

# 2. MagicPony 형식으로 변환
python scripts/convert_fauna_to_magicpony.py \
    --source data/fauna_custom \
    --target data/magicpony/custom --copy

# 3. 설정 파일 수정 (train_data_dir)
# 4. 학습 실행
```
