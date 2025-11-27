#!/usr/bin/env python3
"""
Training Result Analysis Script
Automatically parse training logs and provide verdict
"""

import re
import sys
from pathlib import Path

def parse_log(log_text):
    """Parse training log and extract key metrics"""

    # Find all iteration lines
    iter_pattern = r'T(\d+)/.*sdf_gradient_reg_loss: ([\d.]+)'
    matches = re.findall(iter_pattern, log_text)

    if not matches:
        return None, None

    # Get last iteration before crash
    last_iter = int(matches[-1][0])
    last_sdf_grad = float(matches[-1][1])

    # Collect trajectory
    trajectory = [(int(m[0]), float(m[1])) for m in matches]

    return last_iter, last_sdf_grad, trajectory

def get_verdict(last_iter, last_sdf_grad, config_name="unknown"):
    """Determine verdict based on results"""

    print("\n" + "="*80)
    print(f"🎯 TRAINING RESULT ANALYSIS: {config_name}")
    print("="*80)

    print(f"\n📊 Final Metrics:")
    print(f"   Last Iteration: T{last_iter:06d}")
    print(f"   Final SDF Gradient Loss: {last_sdf_grad:.2f}")

    # Verdict logic
    if last_iter >= 80:
        verdict = "✅ 대성공"
        emoji = "🎉"
        quality = "Good"
        next_action = "Full training (200 iters) 실행"
    elif last_iter >= 50:
        verdict = "✅ 성공"
        emoji = "👍"
        quality = "Medium"
        next_action = "Stage 1 (reg=0.02) 시도 권장"
    elif last_iter >= 35:
        verdict = "⚠️ 개선"
        emoji = "🤔"
        quality = "Low"
        next_action = "Stage 1 (reg=0.02) 필수"
    elif last_iter >= 25:
        verdict = "❌ 부족"
        emoji = "😟"
        quality = "Poor"
        next_action = "다른 전략 필요 (Pretrained 복원 등)"
    else:
        verdict = "❌ 실패"
        emoji = "😱"
        quality = "Failed"
        next_action = "근본적 재검토 필요"

    print(f"\n{emoji} 판정: {verdict}")
    print(f"   Quality: {quality}")
    print(f"   다음 행동: {next_action}")

    # SDF gradient analysis
    print(f"\n📈 SDF Gradient 분석:")
    if last_sdf_grad > 10.0:
        print(f"   ⚠️ 매우 높음 ({last_sdf_grad:.2f} >> 10.0)")
        print(f"   → 초기 학습 불안정 or Pretrained 부족")
    elif last_sdf_grad > 7.0:
        print(f"   ⚠️ 높음 ({last_sdf_grad:.2f})")
        print(f"   → Regularization 더 완화 필요")
    elif last_sdf_grad > 5.0:
        print(f"   ⚠️ 중간 ({last_sdf_grad:.2f})")
        print(f"   → Acceptable range, 약간 완화 권장")
    else:
        print(f"   ✅ 정상 ({last_sdf_grad:.2f})")
        print(f"   → 안정적인 학습")

    # Comparison with baselines
    print(f"\n📊 Baseline 비교:")
    baselines = {
        "Baseline (32, reg=0.1)": (29, 3.85),
        "Ultimate v2 (64, reg=0.05, pretrained)": (35, 5.26),
        "Stage 0 (64, reg=0.03, from scratch)": (last_iter, last_sdf_grad),
    }

    for name, (iters, grad) in baselines.items():
        if "Stage 0" in name:
            prefix = "→ [현재]"
        else:
            prefix = "   "

        diff_iters = last_iter - iters
        diff_grad = last_sdf_grad - grad

        print(f"{prefix} {name}")
        print(f"      Iters: T{iters:06d} (차이: {diff_iters:+d})")
        print(f"      SDF grad: {grad:.2f} (차이: {diff_grad:+.2f})")

    # Recommendations
    print(f"\n💡 권장 사항:")
    if last_iter < 35:
        print("   1. ⚠️ From scratch 실패 → Pretrained 복원 필수")
        print("   2. Pretrained + reg=0.03 시도")
        print("   3. 또는 Pretrained + reg=0.01 (extreme)")
    elif last_iter < 50:
        print("   1. Stage 1 (reg=0.02) 시도")
        print("   2. Pretrained 유지 권장")
    elif last_iter < 80:
        print("   1. Stage 1 (reg=0.02) 시도")
        print("   2. 성공 시 Full training")
    else:
        print("   1. ✅ 현재 config로 Full training (200 iters)")
        print("   2. Quality 평가 후 inference")

    print("\n" + "="*80 + "\n")

    return verdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_training.py <log_file>")
        print("Or: python analyze_training.py <log_text>")
        sys.exit(1)

    # Read input
    input_path = sys.argv[1]
    if Path(input_path).exists():
        with open(input_path, 'r') as f:
            log_text = f.read()
        config_name = Path(input_path).stem
    else:
        # Assume it's text input
        log_text = input_path
        config_name = "unknown"

    # Parse
    result = parse_log(log_text)
    if result[0] is None:
        print("❌ No training iterations found in log")
        sys.exit(1)

    last_iter, last_sdf_grad, trajectory = result

    # Verdict
    get_verdict(last_iter, last_sdf_grad, config_name)

    # Trajectory plot (optional)
    if len(trajectory) > 5:
        print("📈 SDF Gradient Trajectory:")
        for i, (iter_num, sdf_grad) in enumerate(trajectory[::5]):  # Every 5th
            bar_length = int(sdf_grad * 3)
            bar = "█" * min(bar_length, 50)
            print(f"   T{iter_num:06d}: {sdf_grad:6.2f} {bar}")

if __name__ == "__main__":
    main()
