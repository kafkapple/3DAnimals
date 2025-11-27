#!/usr/bin/env python3
"""
Mask Quality Diagnostic Tool
============================

SAM2 video propagation으로 생성된 마스크의 품질을 진단하고 시각화하는 독립 모듈입니다.

Features:
    - 프레임별 마스크 면적 변화 추적
    - 급격한 변화 감지 (SAM2 누적 오류 지표)
    - RGB/Mask 비교 그리드 시각화
    - 면적 변화 그래프 생성
    - JSON 리포트 출력

Supported Data Formats:
    1. Fauna format: sequence_dir/rgb/*.png, sequence_dir/mask/*.png
    2. SAM3D flat format: sequence_dir/*_rgb.png, sequence_dir/*_mask.png

CLI Usage:
    # Basic diagnosis
    python scripts/diagnose_mask_quality.py --data_dir path/to/dataset

    # With visualizations
    python scripts/diagnose_mask_quality.py --data_dir path/to/dataset --visualize

    # Custom threshold
    python scripts/diagnose_mask_quality.py --data_dir path/to/dataset --threshold 70.0

Python API Usage:
    from scripts.diagnose_mask_quality import (
        analyze_sequence,
        visualize_sequence,
        create_comparison_grid,
        SequenceStats
    )

    # Analyze a sequence
    stats = analyze_sequence(Path("data/train/seq_001"), split="train")
    print(f"Quality Score: {stats.quality_score}")
    print(f"Trend: {stats.trend}")

    # Generate visualizations
    visualize_sequence(stats, Path("output/"))
    create_comparison_grid(Path("data/train/seq_001"), Path("output/comparison.png"))

Documentation:
    See docs/tools/mask_quality_diagnosis.md for detailed documentation.

Author: Auto-generated for 3DAnimals project
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from collections import defaultdict

__all__ = [
    'SequenceStats',
    'analyze_mask',
    'analyze_sequence',
    'visualize_sequence',
    'create_comparison_grid',
    'analyze_dataset',
    'generate_report',
    'generate_visualizations',
]


@dataclass
class SequenceStats:
    """시퀀스별 마스크 통계"""
    name: str
    split: str
    num_frames: int
    mask_areas: List[float]  # 각 프레임의 마스크 면적 비율
    area_std: float  # 면적 표준편차
    max_change: float  # 최대 프레임간 변화율
    trend: str  # "stable", "shrinking", "growing", "erratic"
    quality_score: float  # 0-100, 높을수록 좋음
    problematic_frames: List[int]  # 문제 프레임 인덱스


def analyze_mask(mask_path: Path) -> float:
    """마스크 이미지에서 마스크 영역 비율 계산"""
    try:
        img = Image.open(mask_path).convert('L')
        arr = np.array(img)
        # 마스크 영역 비율 (흰색 = 255, 검은색 = 0)
        mask_ratio = np.mean(arr > 127)  # 중간값 기준
        return float(mask_ratio)
    except Exception as e:
        print(f"  Warning: Failed to read {mask_path}: {e}")
        return -1.0


def analyze_sequence(seq_dir: Path, split: str) -> SequenceStats:
    """단일 시퀀스의 마스크 품질 분석

    Supports two data formats:
    1. Fauna format: seq_dir/mask/*.png
    2. SAM3D flat format: seq_dir/*_mask.png
    """
    # Check for Fauna format first
    mask_dir = seq_dir / "mask"
    if mask_dir.exists():
        mask_files = sorted(mask_dir.glob("*.png"))
        if not mask_files:
            mask_files = sorted(mask_dir.glob("*.jpg"))
    else:
        # SAM3D flat format: *_mask.png files in sequence directory
        mask_files = sorted(seq_dir.glob("*_mask.png"))
        if not mask_files:
            mask_files = sorted(seq_dir.glob("*_mask.jpg"))

    if not mask_files:
        return None

    # 각 프레임의 마스크 면적 계산
    mask_areas = []
    for mf in mask_files:
        area = analyze_mask(mf)
        if area >= 0:
            mask_areas.append(area)

    if len(mask_areas) < 2:
        return None

    # 통계 계산
    areas = np.array(mask_areas)
    area_std = float(np.std(areas))

    # 프레임간 변화율 계산
    changes = np.abs(np.diff(areas))
    max_change = float(np.max(changes)) if len(changes) > 0 else 0.0
    avg_change = float(np.mean(changes)) if len(changes) > 0 else 0.0

    # 트렌드 분석
    if len(areas) > 5:
        first_half = np.mean(areas[:len(areas)//2])
        second_half = np.mean(areas[len(areas)//2:])
        trend_diff = second_half - first_half

        if area_std > 0.1 or max_change > 0.2:
            trend = "erratic"
        elif trend_diff < -0.05:
            trend = "shrinking"
        elif trend_diff > 0.05:
            trend = "growing"
        else:
            trend = "stable"
    else:
        trend = "stable" if area_std < 0.05 else "erratic"

    # 문제 프레임 감지 (급격한 변화가 있는 프레임)
    problematic_frames = []
    threshold = max(0.05, avg_change * 3)  # 평균의 3배 이상 변화
    for i, change in enumerate(changes):
        if change > threshold:
            problematic_frames.append(i + 1)  # 변화가 발생한 다음 프레임

    # 품질 점수 계산 (0-100)
    # - 면적이 너무 작거나 크면 감점
    # - 변동성이 크면 감점
    # - 급격한 변화가 많으면 감점
    avg_area = float(np.mean(areas))

    score = 100.0

    # 면적 기반 감점
    if avg_area < 0.01:  # 마스크가 너무 작음
        score -= 50
    elif avg_area < 0.05:
        score -= 20
    elif avg_area > 0.8:  # 마스크가 너무 큼 (배경 포함 가능성)
        score -= 30

    # 변동성 기반 감점
    score -= min(30, area_std * 100)

    # 급격한 변화 기반 감점
    score -= min(20, len(problematic_frames) * 5)

    # 트렌드 기반 감점
    if trend == "erratic":
        score -= 20
    elif trend in ["shrinking", "growing"]:
        score -= 10

    score = max(0, score)

    return SequenceStats(
        name=seq_dir.name,
        split=split,
        num_frames=len(mask_areas),
        mask_areas=mask_areas,
        area_std=area_std,
        max_change=max_change,
        trend=trend,
        quality_score=score,
        problematic_frames=problematic_frames
    )


def visualize_sequence(stats: SequenceStats, output_dir: Path):
    """시퀀스의 마스크 면적 변화 시각화"""
    fig, ax = plt.subplots(figsize=(12, 4))

    frames = range(len(stats.mask_areas))
    ax.plot(frames, stats.mask_areas, 'b-', linewidth=1, label='Mask Area Ratio')
    ax.fill_between(frames, stats.mask_areas, alpha=0.3)

    # 문제 프레임 표시
    for pf in stats.problematic_frames:
        if pf < len(stats.mask_areas):
            ax.axvline(x=pf, color='r', linestyle='--', alpha=0.5)
            ax.scatter([pf], [stats.mask_areas[pf]], color='r', s=50, zorder=5)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Mask Area Ratio')
    ax.set_title(f'{stats.name} ({stats.split}) - Score: {stats.quality_score:.1f}, Trend: {stats.trend}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 저장
    output_path = output_dir / f"{stats.split}_{stats.name}_mask_analysis.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

    return output_path


def create_comparison_grid(seq_dir: Path, output_path: Path, sample_frames: int = 8):
    """RGB와 마스크를 나란히 비교하는 그리드 이미지 생성

    Supports two data formats:
    1. Fauna format: seq_dir/rgb/*.png, seq_dir/mask/*.png
    2. SAM3D flat format: seq_dir/*_rgb.png, seq_dir/*_mask.png
    """
    rgb_dir = seq_dir / "rgb"
    mask_dir = seq_dir / "mask"

    # Check for Fauna format
    if rgb_dir.exists() and mask_dir.exists():
        rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
        mask_files = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
    else:
        # SAM3D flat format
        rgb_files = sorted(seq_dir.glob("*_rgb.png")) + sorted(seq_dir.glob("*_rgb.jpg"))
        mask_files = sorted(seq_dir.glob("*_mask.png")) + sorted(seq_dir.glob("*_mask.jpg"))

    if not rgb_files or not mask_files:
        return None

    # 균등하게 샘플링
    n_files = min(len(rgb_files), len(mask_files))
    indices = np.linspace(0, n_files - 1, sample_frames, dtype=int)

    fig, axes = plt.subplots(2, sample_frames, figsize=(sample_frames * 2, 4))

    for col, idx in enumerate(indices):
        # RGB
        if idx < len(rgb_files):
            rgb_img = Image.open(rgb_files[idx])
            axes[0, col].imshow(rgb_img)
            axes[0, col].set_title(f'Frame {idx}', fontsize=8)
        axes[0, col].axis('off')

        # Mask
        if idx < len(mask_files):
            mask_img = Image.open(mask_files[idx]).convert('L')
            axes[1, col].imshow(mask_img, cmap='gray')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('RGB', fontsize=10)
    axes[1, 0].set_ylabel('Mask', fontsize=10)

    plt.suptitle(seq_dir.name, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

    return output_path


def analyze_dataset(
    data_dir: Path,
    splits: List[str] = ['train', 'val', 'test'],
    verbose: bool = True
) -> Dict[str, List[SequenceStats]]:
    """데이터셋의 모든 시퀀스를 분석합니다.

    Args:
        data_dir: 데이터셋 루트 디렉토리
        splits: 분석할 split 목록
        verbose: 진행 상황 출력 여부

    Returns:
        Dict[split_name, List[SequenceStats]]
    """
    all_stats: Dict[str, List[SequenceStats]] = defaultdict(list)

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        if verbose:
            print(f"Analyzing {split} split...")

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            stats = analyze_sequence(seq_dir, split)
            if stats:
                all_stats[split].append(stats)
                if verbose:
                    print(f"  {seq_dir.name}: Score={stats.quality_score:.1f}, "
                          f"Trend={stats.trend}, Frames={stats.num_frames}, "
                          f"Problematic={len(stats.problematic_frames)}")

    return dict(all_stats)


def generate_report(
    all_stats: Dict[str, List[SequenceStats]],
    data_dir: Path,
    threshold: float = 60.0
) -> dict:
    """분석 결과로부터 리포트를 생성합니다.

    Args:
        all_stats: analyze_dataset()의 반환값
        data_dir: 데이터셋 루트 디렉토리 (리포트에 기록용)
        threshold: 품질 점수 임계값

    Returns:
        리포트 딕셔너리 (JSON 저장 가능)
    """
    good_seqs = []
    bad_seqs = []

    for split, stats_list in all_stats.items():
        for stats in stats_list:
            if stats.quality_score >= threshold:
                good_seqs.append(stats)
            else:
                bad_seqs.append(stats)

    report = {
        "data_dir": str(data_dir),
        "threshold": threshold,
        "total_sequences": len(good_seqs) + len(bad_seqs),
        "good_count": len(good_seqs),
        "bad_count": len(bad_seqs),
        "good_sequences": [
            {"name": s.name, "split": s.split, "score": s.quality_score, "trend": s.trend}
            for s in sorted(good_seqs, key=lambda x: -x.quality_score)
        ],
        "bad_sequences": [
            {"name": s.name, "split": s.split, "score": s.quality_score, "trend": s.trend,
             "problematic_frames": s.problematic_frames}
            for s in sorted(bad_seqs, key=lambda x: x.quality_score)
        ]
    }

    return report


def generate_visualizations(
    all_stats: Dict[str, List[SequenceStats]],
    data_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> List[Path]:
    """모든 시퀀스의 시각화를 생성합니다.

    Args:
        all_stats: analyze_dataset()의 반환값
        data_dir: 데이터셋 루트 디렉토리
        output_dir: 시각화 출력 디렉토리
        verbose: 진행 상황 출력 여부

    Returns:
        생성된 파일 경로 목록
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for split, stats_list in all_stats.items():
        for stats in stats_list:
            # 면적 변화 그래프
            viz_path = visualize_sequence(stats, output_dir)
            if viz_path:
                created_files.append(viz_path)
                if verbose:
                    print(f"  Created: {viz_path}")

            # RGB/Mask 비교 그리드
            seq_dir = data_dir / stats.split / stats.name
            grid_path = output_dir / f"{stats.split}_{stats.name}_comparison.png"
            result = create_comparison_grid(seq_dir, grid_path)
            if result:
                created_files.append(grid_path)
                if verbose:
                    print(f"  Created: {grid_path}")

    return created_files


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose mask quality from SAM2 video propagation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic diagnosis
    python scripts/diagnose_mask_quality.py --data_dir data/fauna/large_scale/mouse

    # With visualizations
    python scripts/diagnose_mask_quality.py --data_dir data/fauna/large_scale/mouse --visualize

    # Custom threshold (flag sequences below 70)
    python scripts/diagnose_mask_quality.py --data_dir data/fauna/large_scale/mouse --threshold 70
        """
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results/mask_diagnosis',
                        help='Output directory for reports (default: results/mask_diagnosis)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--threshold', type=float, default=60.0,
                        help='Quality score threshold (default: 60.0)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Mask Quality Diagnosis ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # 분석 실행
    all_stats = analyze_dataset(data_dir, verbose=True)
    print()

    # 리포트 생성
    report = generate_report(all_stats, data_dir, args.threshold)

    # 결과 출력
    print("=== Summary ===")
    print(f"Total sequences: {report['total_sequences']}")
    print(f"Good quality (score >= {args.threshold}): {report['good_count']}")
    print(f"Problematic (score < {args.threshold}): {report['bad_count']}")
    print()

    if report['bad_sequences']:
        print("=== Problematic Sequences ===")
        for seq in report['bad_sequences']:
            print(f"  [{seq['split']}] {seq['name']}: Score={seq['score']:.1f}, Trend={seq['trend']}")

    if report['good_sequences']:
        print()
        print("=== Good Sequences ===")
        for seq in report['good_sequences']:
            print(f"  [{seq['split']}] {seq['name']}: Score={seq['score']:.1f}, Trend={seq['trend']}")

    # 시각화 생성
    if args.visualize:
        print()
        print("=== Generating Visualizations ===")
        viz_dir = output_dir / "visualizations"
        generate_visualizations(all_stats, data_dir, viz_dir, verbose=True)

    # JSON 리포트 저장
    report_path = output_dir / "mask_quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print()
    print(f"Report saved to: {report_path}")

    # 추천 액션
    print()
    print("=== Recommended Actions ===")
    if report['bad_sequences']:
        print(f"1. Review and re-annotate {report['bad_count']} problematic sequence(s)")
        print("2. Consider using per-frame SAM2 annotation instead of video propagation")
        print("3. Or filter out bad sequences from training data")
        print()
        print("To filter out bad sequences, remove these directories:")
        for seq in report['bad_sequences']:
            seq_path = data_dir / seq['split'] / seq['name']
            print(f"  rm -rf {seq_path}")
    else:
        print("All sequences passed quality check!")


if __name__ == "__main__":
    main()
