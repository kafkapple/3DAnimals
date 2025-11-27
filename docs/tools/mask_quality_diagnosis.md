# Mask Quality Diagnosis Tool

SAM2 video propagation으로 생성된 마스크의 품질을 진단하고 시각화하는 독립 모듈입니다.

## 주요 기능

1. **품질 점수 계산**: 프레임별 마스크 면적 변화를 분석하여 0-100 점수 부여
2. **트렌드 감지**: stable, shrinking, growing, erratic 상태 분류
3. **문제 프레임 식별**: 급격한 변화가 있는 프레임 자동 감지
4. **시각화 생성**:
   - RGB/Mask 비교 그리드 이미지
   - 마스크 면적 변화 그래프

## 설치 요구사항

```bash
pip install numpy pillow matplotlib
```

## 사용법

### 기본 진단
```bash
python scripts/diagnose_mask_quality.py \
    --data_dir path/to/dataset
```

### 시각화 포함
```bash
python scripts/diagnose_mask_quality.py \
    --data_dir path/to/dataset \
    --visualize
```

### 전체 옵션
```bash
python scripts/diagnose_mask_quality.py \
    --data_dir path/to/dataset \
    --output_dir results/my_diagnosis \
    --visualize \
    --threshold 70.0
```

## 지원 데이터 형식

### 1. Fauna 형식
```
dataset/
├── train/
│   └── sequence_name/
│       ├── rgb/
│       │   ├── 000000.png
│       │   └── ...
│       └── mask/
│           ├── 000000.png
│           └── ...
```

### 2. SAM3D Flat 형식
```
dataset/
├── train/
│   └── sequence_name/
│       ├── 0000001_rgb.png
│       ├── 0000001_mask.png
│       ├── 0000002_rgb.png
│       ├── 0000002_mask.png
│       └── ...
```

## 출력 결과

### 콘솔 출력
```
=== Mask Quality Diagnosis ===
Data directory: data/fauna/large_scale/mouse_sam3d
Output directory: results/mask_diagnosis

Analyzing train split...
  sequence_001: Score=85.2, Trend=stable, Frames=100, Problematic=2
  sequence_002: Score=45.3, Trend=erratic, Frames=100, Problematic=8

=== Summary ===
Total sequences: 2
Good quality (score >= 60.0): 1
Problematic (score < 60.0): 1
```

### 생성 파일

```
output_dir/
├── mask_quality_report.json    # JSON 형식 전체 리포트
└── visualizations/
    ├── train_seq001_comparison.png      # RGB/Mask 비교 그리드
    ├── train_seq001_mask_analysis.png   # 면적 변화 그래프
    └── ...
```

## 품질 점수 계산 방식

| 기준 | 감점 |
|------|------|
| 평균 마스크 면적 < 1% | -50점 |
| 평균 마스크 면적 < 5% | -20점 |
| 평균 마스크 면적 > 80% | -30점 |
| 면적 표준편차 | -min(30, std×100) |
| 문제 프레임 수 | -min(20, count×5) |
| erratic 트렌드 | -20점 |
| shrinking/growing 트렌드 | -10점 |

## 트렌드 분류

| 트렌드 | 설명 |
|--------|------|
| **stable** | 면적 변화가 적고 일관됨 |
| **shrinking** | 후반부로 갈수록 면적 감소 (추적 손실) |
| **growing** | 후반부로 갈수록 면적 증가 (배경 포함) |
| **erratic** | 불규칙한 변화 (심각한 추적 오류) |

## Python API 사용

### 전체 데이터셋 분석

```python
from pathlib import Path
from scripts.diagnose_mask_quality import (
    analyze_dataset,
    generate_report,
    generate_visualizations
)

# 데이터셋 분석
data_dir = Path("data/fauna/large_scale/mouse_sam3d")
all_stats = analyze_dataset(data_dir, verbose=True)

# 리포트 생성
report = generate_report(all_stats, data_dir, threshold=60.0)
print(f"Good sequences: {report['good_count']}")
print(f"Bad sequences: {report['bad_count']}")

# 시각화 생성
output_dir = Path("results/visualizations")
generate_visualizations(all_stats, data_dir, output_dir)
```

### 단일 시퀀스 분석

```python
from pathlib import Path
from scripts.diagnose_mask_quality import (
    analyze_sequence,
    visualize_sequence,
    create_comparison_grid,
    SequenceStats
)

# 단일 시퀀스 분석
seq_dir = Path("data/train/sequence_001")
stats = analyze_sequence(seq_dir, split="train")

print(f"Score: {stats.quality_score}")
print(f"Trend: {stats.trend}")
print(f"Problematic frames: {stats.problematic_frames}")

# 시각화 생성
output_dir = Path("results/visualizations")
output_dir.mkdir(exist_ok=True)

visualize_sequence(stats, output_dir)
create_comparison_grid(seq_dir, output_dir / "comparison.png")
```

### SequenceStats 데이터 구조

```python
@dataclass
class SequenceStats:
    name: str              # 시퀀스 이름
    split: str             # train/val/test
    num_frames: int        # 총 프레임 수
    mask_areas: List[float]  # 각 프레임의 마스크 면적 비율 (0~1)
    area_std: float        # 면적 표준편차
    max_change: float      # 최대 프레임간 변화율
    trend: str             # "stable", "shrinking", "growing", "erratic"
    quality_score: float   # 0-100 (높을수록 좋음)
    problematic_frames: List[int]  # 급격한 변화가 있는 프레임 인덱스
```

## 시각화 예시

### RGB/Mask 비교 그리드
8개 프레임을 균등 샘플링하여 RGB 이미지와 마스크를 나란히 표시합니다.

![Comparison Grid](../../results/mask_diagnosis/visualizations/train_cam01_seq_000_comparison.png)

### 마스크 면적 변화 그래프
- 파란선: 프레임별 마스크 면적 비율
- 빨간 점선: 문제 프레임 (급격한 변화 지점)

![Mask Analysis](../../results/mask_diagnosis/visualizations/train_cam01_seq_000_mask_analysis.png)

## SAM2 재주석 권장 사항

품질 점수가 낮은 시퀀스에 대해:

1. **Multi-keyframe annotation**:
   - 첫 프레임 + 문제 프레임(빨간 점)에서 추가 주석
   - 급격한 변화 지점마다 keyframe 추가

2. **Per-frame annotation**:
   - 각 프레임마다 개별 주석 (시간 소모 많음)
   - 가장 정확한 결과

3. **프레임 간격 조정**:
   - 빠른 움직임 구간은 더 촘촘하게 샘플링
   - 정지 구간은 드물게 샘플링

## 관련 스크립트

- `scripts/preprocess_sam3d_dataset.py` - SAM3D 출력을 Fauna 형식으로 변환
- `scripts/run_mouse_pipeline.py` - 전체 학습/추론 파이프라인
