#!/usr/bin/env python3
"""
Cascade Sensitivity Analysis
=============================

Performs probabilistic sensitivity analysis (PSA), barrier removal analysis,
and step importance ranking for the PWID HIV prevention cascade model.

Outputs:
- PSA distributions for cascade completion and P(R₀=0)
- Tornado diagram data
- Barrier removal waterfall analysis
- Step importance ranking

Usage:
    python cascade_sensitivity_analysis.py
    python cascade_sensitivity_analysis.py --output-dir ../data/csv_xlsx --n-samples 1000

Author: AC Demidont, DO / Nyx Dynamics LLC
Date: January 2026
"""

import numpy as np
import random
import json
import csv
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime

# Import from main model
from architectural_barrier_model import (
    ArchitecturalBarrierModel,
    PolicyScenario,
    create_policy_scenarios,
    create_pwid_cascade,
    CascadeStep,
)

# Set seeds
np.random.seed(42)
random.seed(42)


# =============================================================================
# PARAMETER BOUNDS FOR PSA
# =============================================================================

@dataclass
class CascadeParameterBounds:
    """Uncertainty bounds for cascade step parameters"""
    name: str
    base_prob_low: float
    base_prob_high: float
    policy_penalty_low: float = 0.0
    policy_penalty_high: float = 0.0
    stigma_penalty_low: float = 0.0
    stigma_penalty_high: float = 0.0
    infrastructure_penalty_low: float = 0.0
    infrastructure_penalty_high: float = 0.0


# Literature-derived uncertainty bounds
CASCADE_BOUNDS = {
    "awareness": CascadeParameterBounds(
        name="awareness",
        base_prob_low=0.60, base_prob_high=0.80,
        policy_penalty_low=0.15, policy_penalty_high=0.35,
        stigma_penalty_low=0.02, stigma_penalty_high=0.10,
        infrastructure_penalty_low=0.10, infrastructure_penalty_high=0.25,
    ),
    "willingness": CascadeParameterBounds(
        name="willingness",
        base_prob_low=0.70, base_prob_high=0.90,
        policy_penalty_low=0.25, policy_penalty_high=0.45,
        stigma_penalty_low=0.05, stigma_penalty_high=0.15,
    ),
    "healthcare_access": CascadeParameterBounds(
        name="healthcare_access",
        base_prob_low=0.65, base_prob_high=0.85,
        policy_penalty_low=0.05, policy_penalty_high=0.20,
        stigma_penalty_low=0.02, stigma_penalty_high=0.10,
        infrastructure_penalty_low=0.15, infrastructure_penalty_high=0.35,
    ),
    "disclosure": CascadeParameterBounds(
        name="disclosure",
        base_prob_low=0.60, base_prob_high=0.80,
        policy_penalty_low=0.20, policy_penalty_high=0.40,
        stigma_penalty_low=0.10, stigma_penalty_high=0.25,
    ),
    "provider_willing": CascadeParameterBounds(
        name="provider_willing",
        base_prob_low=0.75, base_prob_high=0.90,
        policy_penalty_low=0.02, policy_penalty_high=0.10,
        stigma_penalty_low=0.15, stigma_penalty_high=0.35,
    ),
    "hiv_testing_adequate": CascadeParameterBounds(
        name="hiv_testing_adequate",
        base_prob_low=0.80, base_prob_high=0.95,
        policy_penalty_low=0.02, policy_penalty_high=0.10,
        infrastructure_penalty_low=0.10, infrastructure_penalty_high=0.25,
    ),
    "first_injection": CascadeParameterBounds(
        name="first_injection",
        base_prob_low=0.65, base_prob_high=0.85,
        policy_penalty_low=0.05, policy_penalty_high=0.15,
        stigma_penalty_low=0.02, stigma_penalty_high=0.10,
        infrastructure_penalty_low=0.10, infrastructure_penalty_high=0.25,
    ),
    "sustained_engagement": CascadeParameterBounds(
        name="sustained_engagement",
        base_prob_low=0.60, base_prob_high=0.80,
        policy_penalty_low=0.15, policy_penalty_high=0.30,
        stigma_penalty_low=0.05, stigma_penalty_high=0.15,
        infrastructure_penalty_low=0.05, infrastructure_penalty_high=0.20,
    ),
}


# =============================================================================
# SENSITIVITY ANALYZER
# =============================================================================

class CascadeSensitivityAnalyzer:
    """Performs various sensitivity analyses on the cascade model"""

    def __init__(self):
        self.base_cascade = create_pwid_cascade()
        self.bounds = CASCADE_BOUNDS

    def sample_cascade_parameters(self) -> List[CascadeStep]:
        """Sample cascade parameters from uncertainty distributions"""
        sampled_cascade = []

        for step in self.base_cascade:
            bounds = self.bounds.get(step.name)
            if bounds:
                sampled_step = CascadeStep(
                    name=step.name,
                    description=step.description,
                    base_probability=np.random.uniform(bounds.base_prob_low, bounds.base_prob_high),
                    policy_penalty=np.random.uniform(bounds.policy_penalty_low, bounds.policy_penalty_high),
                    stigma_penalty=np.random.uniform(bounds.stigma_penalty_low, bounds.stigma_penalty_high),
                    infrastructure_penalty=np.random.uniform(bounds.infrastructure_penalty_low, bounds.infrastructure_penalty_high),
                    testing_penalty=step.testing_penalty,
                    research_penalty=step.research_penalty,
                    ml_penalty=step.ml_penalty,
                )
                sampled_cascade.append(sampled_step)
            else:
                sampled_cascade.append(step)

        return sampled_cascade

    def run_probabilistic_sensitivity(
        self,
        n_samples: int = 1000,
        n_individuals: int = 10000
    ) -> Dict:
        """Run probabilistic sensitivity analysis"""
        print(f"Running PSA with {n_samples} samples...")

        cascade_completions = []
        r0_zero_rates = []
        step_prob_samples = {step.name: [] for step in self.base_cascade}

        current_scenario = PolicyScenario("Current Policy")

        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  Sample {i + 1}/{n_samples}")

            # Sample parameters
            sampled_cascade = self.sample_cascade_parameters()
            model = ArchitecturalBarrierModel(cascade=sampled_cascade)

            # Run simulation
            results = model.run_simulation(current_scenario, n_individuals=n_individuals, years=5)

            cascade_completions.append(results["observed_cascade_completion_rate"])
            r0_zero_rates.append(results["observed_r0_zero_rate"])

            for step_name, prob in results["step_probabilities"].items():
                step_prob_samples[step_name].append(prob)

        # Summary statistics
        summary = {
            "cascade_completion": {
                "mean": np.mean(cascade_completions),
                "std": np.std(cascade_completions),
                "p5": np.percentile(cascade_completions, 5),
                "p95": np.percentile(cascade_completions, 95),
            },
            "r0_zero_rate": {
                "mean": np.mean(r0_zero_rates),
                "std": np.std(r0_zero_rates),
                "p5": np.percentile(r0_zero_rates, 5),
                "p95": np.percentile(r0_zero_rates, 95),
            },
            "step_probabilities": {
                name: {
                    "mean": np.mean(probs),
                    "std": np.std(probs),
                    "p5": np.percentile(probs, 5),
                    "p95": np.percentile(probs, 95),
                }
                for name, probs in step_prob_samples.items()
            }
        }

        return {"summary": summary, "n_samples": n_samples}

    def barrier_removal_analysis(self, n_individuals: int = 100000) -> Dict:
        """Analyze effect of removing individual barrier types"""
        print("Running barrier removal analysis...")

        model = ArchitecturalBarrierModel()
        results = {}

        # Baseline (current policy)
        baseline = model.run_simulation(PolicyScenario("Current Policy"), n_individuals=n_individuals)
        results["baseline"] = {
            "r0_zero_rate": baseline["observed_r0_zero_rate"],
            "cascade_completion": baseline["observed_cascade_completion_rate"],
            "ci_95": list(baseline["r0_zero_95ci"]),
        }

        # Scenario configurations for barrier removal
        barrier_scenarios = [
            ("no_criminalization", PolicyScenario("No Criminalization", decriminalization=True, incarceration_modifier=0.3)),
            ("no_stigma", PolicyScenario("No Stigma", stigma_reduction=1.0, bias_training=True)),
            ("low_barrier_access", PolicyScenario("Low Barrier Access", ssp_integrated=True, peer_navigation=True, low_barrier=True)),
            ("full_inclusion", PolicyScenario("Full Research Inclusion", trial_inclusion=True, ml_debiasing=True)),
            ("all_removed", PolicyScenario("All Barriers Removed", decriminalization=True, incarceration_modifier=0.0,
                                           stigma_reduction=1.0, bias_training=True, ssp_integrated=True,
                                           peer_navigation=True, low_barrier=True, trial_inclusion=True, ml_debiasing=True)),
        ]

        for name, scenario in barrier_scenarios:
            print(f"  Testing: {name}")
            result = model.run_simulation(scenario, n_individuals=n_individuals)
            results[name] = {
                "r0_zero_rate": result["observed_r0_zero_rate"],
                "cascade_completion": result["observed_cascade_completion_rate"],
                "ci_95": list(result["r0_zero_95ci"]),
            }

        return results

    def step_importance_analysis(self, n_individuals: int = 100000) -> Dict:
        """Analyze importance of each cascade step"""
        print("Running step importance analysis...")

        model = ArchitecturalBarrierModel()
        current_scenario = PolicyScenario("Current Policy")

        # Baseline
        baseline = model.run_simulation(current_scenario, n_individuals=n_individuals)
        baseline_r0_zero = baseline["observed_r0_zero_rate"]

        effects = {}
        for i, step in enumerate(model.cascade):
            # Create modified cascade with step fixed at 100%
            modified_cascade = []
            for j, s in enumerate(model.cascade):
                if i == j:
                    # Fix this step at near-perfect probability
                    modified_step = CascadeStep(
                        name=s.name,
                        description=s.description,
                        base_probability=0.99,
                        policy_penalty=0.0,
                        stigma_penalty=0.0,
                        infrastructure_penalty=0.0,
                        testing_penalty=0.0,
                        research_penalty=0.0,
                        ml_penalty=0.0,
                    )
                    modified_cascade.append(modified_step)
                else:
                    modified_cascade.append(s)

            modified_model = ArchitecturalBarrierModel(cascade=modified_cascade)
            result = modified_model.run_simulation(current_scenario, n_individuals=n_individuals)

            effects[step.name] = {
                "original_probability": baseline["step_probabilities"][step.name],
                "cascade_if_fixed": result["observed_cascade_completion_rate"],
                "r0_zero_if_fixed": result["observed_r0_zero_rate"],
                "improvement": result["observed_r0_zero_rate"] - baseline_r0_zero,
            }

        # Rank by improvement
        ranked = sorted(effects.keys(), key=lambda x: effects[x]["improvement"], reverse=True)

        return {"ranked": ranked, "effects": effects}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cascade Sensitivity Analysis")
    parser.add_argument("--output-dir", type=str, default="../data/csv_xlsx",
                       help="Output directory")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="PSA samples (default: 1000)")
    parser.add_argument("--n-individuals", type=int, default=100000,
                       help="Individuals per simulation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("CASCADE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()

    analyzer = CascadeSensitivityAnalyzer()

    # Run analyses
    psa_results = analyzer.run_probabilistic_sensitivity(
        n_samples=args.n_samples, n_individuals=10000)
    barrier_results = analyzer.barrier_removal_analysis(n_individuals=args.n_individuals)
    step_results = analyzer.step_importance_analysis(n_individuals=args.n_individuals)

    # Compile output
    output = {
        "timestamp": datetime.now().isoformat(),
        "probabilistic_sensitivity": psa_results,
        "barrier_removal": barrier_results,
        "step_importance": step_results,
    }

    # Save
    json_path = os.path.join(args.output_dir, "cascade_sensitivity_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nPSA Results (Current Policy):")
    print(f"  P(R₀=0) mean: {psa_results['summary']['r0_zero_rate']['mean']:.6f}")
    print(f"  P(R₀=0) 95% CI: ({psa_results['summary']['r0_zero_rate']['p5']:.6f}, "
          f"{psa_results['summary']['r0_zero_rate']['p95']:.6f})")

    print("\nBarrier Removal Effects:")
    for name, result in barrier_results.items():
        print(f"  {name}: P(R₀=0) = {result['r0_zero_rate']*100:.2f}%")

    print("\nStep Importance Ranking:")
    for i, step_name in enumerate(step_results["ranked"][:5], 1):
        effect = step_results["effects"][step_name]
        print(f"  {i}. {step_name}: +{effect['improvement']*100:.4f}% if fixed")

    print("\nDone!")


if __name__ == "__main__":
    main()
