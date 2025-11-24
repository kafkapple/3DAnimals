"""
Visualize Fauna training progress and results.
Parse Fauna training log files and create visualization plots.

References:
- /home/joon/dev/pose-splatter/visualize_training.py
- /home/joon/dev/pose-splatter/src/plots.py
"""
__date__ = "2025-11-12"
__author__ = "Claude Code with Joon"

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def parse_fauna_training_log(log_file: str) -> Optional[Dict]:
    """
    Parse Fauna training log file to extract metrics.

    Expected log format:
        T000100: loss: 10.23 | mask_iou: 0.45 | rgb_psnr: 18.34 | ...

    Args:
        log_file: Path to training log file

    Returns:
        Dictionary containing metrics lists or None if file not found
    """
    metrics = {
        'iteration': [],
        'total_loss': [],
        'mask_iou': [],
        'rgb_psnr': [],
        'dino_similarity': [],
        'sdf_bce_reg': [],
        'sdf_gradient_reg': [],
        'arti_reg_loss': [],
    }

    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_file}")
        return None

    print(f"Parsing log file: {log_file}")

    with open(log_path, 'r') as f:
        for line in f:
            # Match iteration line: "T000100: loss: 10.23 | ..."
            iter_match = re.search(r'T(\d+):', line)
            if iter_match:
                iteration = int(iter_match.group(1))
                metrics['iteration'].append(iteration)

                # Extract total loss
                loss_match = re.search(r'loss:\s+([\d.]+)', line)
                if loss_match:
                    metrics['total_loss'].append(float(loss_match.group(1)))

                # Extract mask IoU
                iou_match = re.search(r'mask_iou:\s+([\d.]+)', line)
                if iou_match:
                    metrics['mask_iou'].append(float(iou_match.group(1)))

                # Extract RGB PSNR
                psnr_match = re.search(r'rgb_psnr:\s+([\d.]+)', line)
                if psnr_match:
                    metrics['rgb_psnr'].append(float(psnr_match.group(1)))

                # Extract DINO similarity
                dino_match = re.search(r'dino_similarity:\s+([\d.]+)', line)
                if dino_match:
                    metrics['dino_similarity'].append(float(dino_match.group(1)))

                # Extract SDF BCE regularization
                sdf_bce_match = re.search(r'sdf_bce_reg:\s+([\d.]+)', line)
                if sdf_bce_match:
                    metrics['sdf_bce_reg'].append(float(sdf_bce_match.group(1)))

                # Extract SDF gradient regularization
                sdf_grad_match = re.search(r'sdf_gradient_reg:\s+([\d.]+)', line)
                if sdf_grad_match:
                    metrics['sdf_gradient_reg'].append(float(sdf_grad_match.group(1)))

                # Extract articulation regularization
                arti_match = re.search(r'arti_reg_loss:\s+([\d.]+)', line)
                if arti_match:
                    metrics['arti_reg_loss'].append(float(arti_match.group(1)))

    # Report parsing results
    n_points = len(metrics['iteration'])
    print(f"Parsed {n_points} data points")
    if n_points == 0:
        print("Warning: No metrics found in log file")
        return None

    return metrics


def plot_training_curves(metrics: Dict, output_dir: Path, config_name: str = "fauna"):
    """
    Plot comprehensive training curves for Fauna.

    Creates 4 subplots:
    1. Total Loss
    2. Mask IoU (reconstruction quality)
    3. RGB PSNR (image quality)
    4. Regularization losses (SDF stability)

    Args:
        metrics: Dictionary of metric lists
        output_dir: Output directory for plots
        config_name: Configuration name for title
    """
    if not metrics or not metrics['iteration']:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Fauna Training Progress: {config_name}', fontsize=16, fontweight='bold')

    iterations = np.array(metrics['iteration'])

    # Plot 1: Total Loss
    ax = axes[0, 0]
    if metrics['total_loss']:
        loss = np.array(metrics['total_loss'])
        ax.plot(iterations, loss, 'b-', linewidth=2, label='Total Loss')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Total Training Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add annotations for min loss
        min_idx = np.argmin(loss)
        ax.plot(iterations[min_idx], loss[min_idx], 'r*', markersize=15)
        ax.annotate(f'Min: {loss[min_idx]:.3f}\nIter: {iterations[min_idx]}',
                   xy=(iterations[min_idx], loss[min_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Mask IoU (Reconstruction Quality)
    ax = axes[0, 1]
    if metrics['mask_iou']:
        iou = np.array(metrics['mask_iou'])
        ax.plot(iterations, iou, 'g-', linewidth=2, label='Mask IoU')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('IoU', fontsize=12)
        ax.set_title('Mask IoU (Reconstruction Quality)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target: 0.7')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add annotations for max IoU
        max_idx = np.argmax(iou)
        ax.plot(iterations[max_idx], iou[max_idx], 'r*', markersize=15)
        ax.annotate(f'Max: {iou[max_idx]:.3f}\nIter: {iterations[max_idx]}',
                   xy=(iterations[max_idx], iou[max_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 3: RGB PSNR (Image Quality)
    ax = axes[1, 0]
    if metrics['rgb_psnr']:
        psnr = np.array(metrics['rgb_psnr'])
        ax.plot(iterations, psnr, 'm-', linewidth=2, label='RGB PSNR')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('RGB PSNR (Image Quality)', fontsize=14, fontweight='bold')
        ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Target: 20 dB')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add annotations for max PSNR
        max_idx = np.argmax(psnr)
        ax.plot(iterations[max_idx], psnr[max_idx], 'r*', markersize=15)
        ax.annotate(f'Max: {psnr[max_idx]:.2f} dB\nIter: {iterations[max_idx]}',
                   xy=(iterations[max_idx], psnr[max_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 4: Regularization Losses (SDF Stability)
    ax = axes[1, 1]
    if metrics['sdf_bce_reg']:
        sdf_bce = np.array(metrics['sdf_bce_reg'])
        ax.plot(iterations, sdf_bce, 'r-', linewidth=2, label='SDF BCE Reg', alpha=0.7)
    if metrics['sdf_gradient_reg']:
        sdf_grad = np.array(metrics['sdf_gradient_reg'])
        ax.plot(iterations, sdf_grad, 'orange', linewidth=2, label='SDF Gradient Reg', alpha=0.7)
    if metrics['arti_reg_loss']:
        arti = np.array(metrics['arti_reg_loss'])
        ax.plot(iterations, arti, 'c-', linewidth=2, label='Articulation Reg', alpha=0.7)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Regularization Loss', fontsize=12)
    ax.set_title('Regularization Losses (SDF Stability)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    output_path = output_dir / f'{config_name}_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves: {output_path}")
    plt.close()


def plot_training_summary(metrics: Dict, output_dir: Path, config_name: str = "fauna"):
    """
    Plot single comprehensive summary plot with all key metrics.

    Args:
        metrics: Dictionary of metric lists
        output_dir: Output directory for plots
        config_name: Configuration name for title
    """
    if not metrics or not metrics['iteration']:
        print("No metrics to plot")
        return

    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.suptitle(f'Fauna Training Summary: {config_name}', fontsize=16, fontweight='bold')

    iterations = np.array(metrics['iteration'])

    # Primary axis: Loss
    color = 'tab:blue'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    if metrics['total_loss']:
        loss = np.array(metrics['total_loss'])
        ax1.plot(iterations, loss, color=color, linewidth=2, label='Total Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Mask IoU and RGB PSNR (normalized)
    ax2 = ax1.twinx()

    if metrics['mask_iou']:
        iou = np.array(metrics['mask_iou'])
        ax2.plot(iterations, iou, 'g-', linewidth=2, label='Mask IoU', alpha=0.7)

    if metrics['rgb_psnr']:
        psnr = np.array(metrics['rgb_psnr'])
        # Normalize PSNR to 0-1 range for visualization (assuming PSNR 0-40 dB)
        psnr_norm = np.clip(psnr / 40.0, 0, 1)
        ax2.plot(iterations, psnr_norm, 'm-', linewidth=2, label='RGB PSNR (norm)', alpha=0.7)

    ax2.set_ylabel('Quality Metrics (IoU, PSNR normalized)', fontsize=12)
    ax2.set_ylim([0, 1])

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / f'{config_name}_training_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training summary: {output_path}")
    plt.close()


def plot_phase_analysis(metrics: Dict, output_dir: Path, config_name: str = "fauna"):
    """
    Analyze training phases based on config milestones.

    For from-scratch training:
    - Phase 1 (0-50K): SDF field stabilization
    - Phase 2 (50-100K): Shape learning + articulation
    - Phase 3 (100-150K): Texture & details
    - Phase 4 (150-200K): Refinement

    Args:
        metrics: Dictionary of metric lists
        output_dir: Output directory for plots
        config_name: Configuration name for title
    """
    if not metrics or not metrics['iteration']:
        print("No metrics to plot")
        return

    iterations = np.array(metrics['iteration'])
    max_iter = iterations[-1]

    # Define phases based on iteration count
    if max_iter >= 150000:
        # Full training (200K iters)
        phases = [
            (0, 50000, 'Phase 1: SDF Stabilization', 'lightblue'),
            (50000, 100000, 'Phase 2: Shape + Articulation', 'lightgreen'),
            (100000, 150000, 'Phase 3: Texture & Details', 'lightyellow'),
            (150000, max_iter, 'Phase 4: Refinement', 'lightcoral'),
        ]
    elif max_iter >= 20000:
        # Medium training (50K iters or debug)
        phases = [
            (0, max_iter // 2, 'Phase 1: Initialization', 'lightblue'),
            (max_iter // 2, max_iter, 'Phase 2: Refinement', 'lightgreen'),
        ]
    else:
        # Debug mode (5K iters)
        phases = [
            (0, max_iter, 'Debug: Quick Validation', 'lightgray'),
        ]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(f'Fauna Training Phases: {config_name}', fontsize=16, fontweight='bold')

    # Plot loss with phase backgrounds
    if metrics['total_loss']:
        loss = np.array(metrics['total_loss'])
        ax.plot(iterations, loss, 'b-', linewidth=2, label='Total Loss')

        # Add phase backgrounds
        for start, end, label, color in phases:
            ax.axvspan(start, end, alpha=0.2, color=color, label=label)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss with Phase Annotations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / f'{config_name}_phase_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved phase analysis: {output_path}")
    plt.close()


def print_training_stats(metrics: Dict):
    """Print summary statistics of training."""
    if not metrics or not metrics['iteration']:
        print("No metrics available")
        return

    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)

    iterations = np.array(metrics['iteration'])
    print(f"Total iterations: {len(iterations)}")
    print(f"Iteration range: {iterations[0]} - {iterations[-1]}")

    if metrics['total_loss']:
        loss = np.array(metrics['total_loss'])
        print(f"\nTotal Loss:")
        print(f"  Initial: {loss[0]:.4f}")
        print(f"  Final: {loss[-1]:.4f}")
        print(f"  Min: {np.min(loss):.4f} (iter {iterations[np.argmin(loss)]})")
        print(f"  Improvement: {((loss[0] - loss[-1]) / loss[0] * 100):.2f}%")

    if metrics['mask_iou']:
        iou = np.array(metrics['mask_iou'])
        print(f"\nMask IoU:")
        print(f"  Initial: {iou[0]:.4f}")
        print(f"  Final: {iou[-1]:.4f}")
        print(f"  Max: {np.max(iou):.4f} (iter {iterations[np.argmax(iou)]})")
        print(f"  Target (0.7): {'✓ REACHED' if iou[-1] >= 0.7 else '✗ NOT YET'}")

    if metrics['rgb_psnr']:
        psnr = np.array(metrics['rgb_psnr'])
        print(f"\nRGB PSNR:")
        print(f"  Initial: {psnr[0]:.2f} dB")
        print(f"  Final: {psnr[-1]:.2f} dB")
        print(f"  Max: {np.max(psnr):.2f} dB (iter {iterations[np.argmax(psnr)]})")
        print(f"  Target (20 dB): {'✓ REACHED' if psnr[-1] >= 20 else '✗ NOT YET'}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Fauna training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize debug training
  python scripts/visualize_fauna_training.py \\
    --log_file /tmp/fauna_debug.log \\
    --output_dir results/fauna_mouse_debug/plots \\
    --config_name fauna_mouse_debug

  # Visualize full from-scratch training
  python scripts/visualize_fauna_training.py \\
    --log_file /tmp/fauna_from_scratch.log \\
    --output_dir results/fauna_mouse_from_scratch/plots \\
    --config_name fauna_mouse_from_scratch
        """
    )
    parser.add_argument("--log_file", type=str, required=True,
                       help="Training log file to parse")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for plots")
    parser.add_argument("--config_name", type=str, default="fauna",
                       help="Configuration name for plot titles")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse training log
    print(f"Parsing training log: {args.log_file}")
    metrics = parse_fauna_training_log(args.log_file)

    if not metrics:
        print("Failed to parse metrics. Exiting.")
        return

    # Print statistics
    print_training_stats(metrics)

    # Generate plots
    print("\nGenerating plots...")
    plot_training_curves(metrics, output_dir, args.config_name)
    plot_training_summary(metrics, output_dir, args.config_name)
    plot_phase_analysis(metrics, output_dir, args.config_name)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - {args.config_name}_training_curves.png (4-panel detailed view)")
    print(f"  - {args.config_name}_training_summary.png (single comprehensive view)")
    print(f"  - {args.config_name}_phase_analysis.png (phase-annotated view)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
