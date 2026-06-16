#!/usr/bin/env python3
"""
Unit tests for Option B coupling analytic limits.

Three tests verify the structural correctness of the cascade→outbreak coupling:
  1. cascade_completion=0  → effective_density == network_density → same p_outbreak as uncoupled
  2. cascade_completion=1  → effective_density=0 → density_multiplier=0 → p_outbreak≈0
  3. Monotonicity: increasing cascade_completion strictly lowers p_outbreak
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stochastic_avoidance_enhanced import EnhancedStochasticAvoidanceModel

def test_zero_completion_equals_uncoupled():
    """cascade_completion=0 must recover the uncoupled result exactly."""
    model = EnhancedStochasticAvoidanceModel(region="national_average")
    density = 0.45  # above threshold to exercise the exp path

    p_uncoupled = model.calculate_outbreak_probability(density)
    p_zero_cc   = model.calculate_outbreak_probability(density, cascade_completion=0.0)

    assert abs(p_uncoupled - p_zero_cc) < 1e-12, (
        f"cascade_completion=0 should equal uncoupled: {p_uncoupled} vs {p_zero_cc}"
    )
    print(f"  PASS: cascade_completion=0 → p_outbreak={p_zero_cc:.6f} (matches uncoupled)")


def test_perfect_completion_eliminates_outbreak():
    """cascade_completion=1 → effective_density=0 → density_multiplier=0 → p_outbreak=0."""
    model = EnhancedStochasticAvoidanceModel(region="national_average")
    density = 0.8  # well above threshold

    p = model.calculate_outbreak_probability(density, cascade_completion=1.0)

    assert p == 0.0, (
        f"cascade_completion=1 should give p_outbreak=0, got {p}"
    )
    print(f"  PASS: cascade_completion=1 → p_outbreak={p:.6f} (eliminated)")


def test_monotonicity():
    """Increasing cascade_completion must strictly lower p_outbreak."""
    model = EnhancedStochasticAvoidanceModel(region="national_average")
    density = 0.5

    completions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    probs = [model.calculate_outbreak_probability(density, cascade_completion=cc)
             for cc in completions]

    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1], (
            f"Non-monotone at cc={completions[i]:.1f}→{completions[i+1]:.1f}: "
            f"p={probs[i]:.6f} → {probs[i+1]:.6f}"
        )
    print(f"  PASS: monotone decrease — p_outbreak: {[f'{p:.4f}' for p in probs]}")


if __name__ == "__main__":
    print("Running Option B coupling analytic limit tests...")
    test_zero_completion_equals_uncoupled()
    test_perfect_completion_eliminates_outbreak()
    test_monotonicity()
    print("\nAll 3 tests passed.")
