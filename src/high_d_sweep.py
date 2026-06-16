#!/usr/bin/env python3
"""
High-dimensional sensitivity sweep — Option B coupled model.

Sweeps over 9 dimensions with defensible bounds using Latin Hypercube Sampling (LHS).
Degenerate-bound parameters (density weights, ssp/oat/prevalence_normalization) are
excluded until AC sets real ranges.

Swept dimensions:
  1. outbreak_escalation_rate    [1.0, 5.0]  — AC-approved
  2. critical_network_threshold  [0.25, 0.45] — AC-approved
  3. baseline_outbreak_prob      [0.01, 0.08] — sourced (Des Jarlais 2022)
  4. baseline_network_density    [0.08, 0.25] — sourced (Des Jarlais modeling)
  5. meth_annual_growth_rate     [0.01, 0.05] — sourced (NHBS extrapolation)
  6. housing_instability_rate    [0.55, 0.80] — sourced (NHBS 23-city)
  7. incarceration_annual_rate   [0.20, 0.45] — sourced (Stone 2018)
  8. ssp_coverage                [0.15, 0.30] — sourced (Van Handel 2016)
  9. oat_coverage                [0.04, 0.15] — sourced (UNAIDS/WHO)

Cascade completion scenarios tested at each LHS point:
  - Current policy PWID (<1%)  — 0.00005
  - Decriminalization           — 0.001
  - Full harm reduction         — 0.05
  - Theoretical maximum         — 0.50

Outputs (to sweep/):
  - sweep_results.csv
  - Fig_sweep_escalation_robustness.png  — headline make-or-break stress test
  - Fig_sweep_sobol_indices.png          — first-order + total Sobol indices
  - Fig_sweep_coupled_causal.png         — coupled p_outbreak vs cascade completion
"""

import sys
import os
import numpy as np
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_avoidance_enhanced import (
    EnhancedStochasticAvoidanceModel, KEY_PARAMETERS, REGIONAL_PROFILES
)

# BMC figure settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP DIMENSIONS — only parameters with defensible bounds
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_DIMS = [
    ("outbreak_escalation_rate",   1.0,  5.0),
    ("critical_network_threshold", 0.25, 0.45),
    ("baseline_outbreak_prob",     0.01, 0.08),
    ("baseline_network_density",   0.08, 0.25),
    ("meth_annual_growth_rate",    0.01, 0.05),
    ("housing_instability_rate",   0.55, 0.80),
    ("incarceration_annual_rate",  0.20, 0.45),
    ("ssp_coverage",               0.15, 0.30),
    ("oat_coverage",               0.04, 0.15),
]

DIM_NAMES = [d[0] for d in SWEEP_DIMS]
N_DIMS = len(SWEEP_DIMS)

CASCADE_SCENARIOS = {
    "current_policy":     0.000050,   # <1% PWID under current policy
    "decriminalization":  0.001000,
    "full_harm_reduction": 0.050000,
    "theoretical_max":    0.500000,
}


def latin_hypercube(n_samples: int, n_dims: int) -> np.ndarray:
    """Generate LHS design matrix in [0,1]^n_dims."""
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = RNG.permutation(n_samples)
        result[:, d] = (perm + RNG.uniform(size=n_samples)) / n_samples
    return result


def scale_lhs(lhs: np.ndarray) -> np.ndarray:
    """Scale [0,1] LHS to parameter bounds."""
    scaled = np.zeros_like(lhs)
    for i, (name, lo, hi) in enumerate(SWEEP_DIMS):
        scaled[:, i] = lo + lhs[:, i] * (hi - lo)
    return scaled


def make_params_from_row(row: np.ndarray) -> Dict:
    """Build a modified KEY_PARAMETERS dict from a sweep row."""
    import copy
    params = copy.deepcopy(KEY_PARAMETERS)
    for i, name in enumerate(DIM_NAMES):
        val = float(row[i])
        params[name].point_estimate = val
        params[name].lower_bound = val
        params[name].upper_bound = val
    return params


def run_single_point(row: np.ndarray, cascade_completion: float, year: int = 2030) -> float:
    """Evaluate p_outbreak at a single LHS point for a given cascade completion."""
    params = make_params_from_row(row)
    model = EnhancedStochasticAvoidanceModel(region="national_average", params=params)
    density = model.calculate_network_density(year=year)
    return model.calculate_outbreak_probability(
        density,
        cascade_completion=cascade_completion
    )


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run LHS sweep. Returns (scaled_samples, results_matrix, scenario_names)."""
    print(f"  LHS sweep: {n_samples} samples × {N_DIMS} dims × {len(CASCADE_SCENARIOS)} scenarios")
    lhs = latin_hypercube(n_samples, N_DIMS)
    scaled = scale_lhs(lhs)

    scenario_names = list(CASCADE_SCENARIOS.keys())
    results = np.zeros((n_samples, len(scenario_names)))

    for j, (scen_name, cc) in enumerate(CASCADE_SCENARIOS.items()):
        for i in range(n_samples):
            results[i, j] = run_single_point(scaled[i], cascade_completion=cc)
        print(f"    {scen_name}: mean p_outbreak = {results[:, j].mean():.4f}, "
              f"P(>50%) = {(results[:, j] > 0.5).mean():.3f}")

    return scaled, results, scenario_names


# ─────────────────────────────────────────────────────────────────────────────
# SOBOL INDICES (Saltelli estimator, first-order + total)
# ─────────────────────────────────────────────────────────────────────────────

def sobol_indices(n_sobol: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate first-order (S1) and total-effect (ST) Sobol indices via Saltelli method.
    Uses current_policy cascade completion for the coupled model.
    """
    cc = CASCADE_SCENARIOS["current_policy"]

    # Build A and B matrices
    lhs_a = latin_hypercube(n_sobol, N_DIMS)
    lhs_b = latin_hypercube(n_sobol, N_DIMS)
    A = scale_lhs(lhs_a)
    B = scale_lhs(lhs_b)

    # Evaluate A and B
    f_A = np.array([run_single_point(A[i], cc) for i in range(n_sobol)])
    f_B = np.array([run_single_point(B[i], cc) for i in range(n_sobol)])

    # Total variance
    f_all = np.concatenate([f_A, f_B])
    var_total = np.var(f_all)

    if var_total < 1e-14:
        return np.zeros(N_DIMS), np.zeros(N_DIMS)

    S1 = np.zeros(N_DIMS)
    ST = np.zeros(N_DIMS)

    for k in range(N_DIMS):
        # A_B: A with column k replaced by B
        A_Bk = A.copy()
        A_Bk[:, k] = B[:, k]
        f_ABk = np.array([run_single_point(A_Bk[i], cc) for i in range(n_sobol)])

        # First-order: S1_k = V[E[Y|X_k]] / V[Y]
        S1[k] = np.mean(f_B * (f_ABk - f_A)) / var_total

        # Total-effect: ST_k = E[V[Y|X_~k]] / V[Y]
        ST[k] = 0.5 * np.mean((f_A - f_ABk) ** 2) / var_total

    # Clip to [0,1] to handle Monte Carlo noise
    S1 = np.clip(S1, 0, 1)
    ST = np.clip(ST, 0, 1)

    return S1, ST


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_escalation_robustness(output_dir: str):
    """
    Fig 1 (headline): p_outbreak vs escalation_rate across cascade completion scenarios.
    Tests whether the outbreak conclusion is robust across [1,5].
    """
    escalation_vals = np.linspace(1.0, 5.0, 50)
    fig, ax = plt.subplots(figsize=(5.5, 4))

    colors = ["#D62728", "#FF7F0E", "#2CA02C", "#1F77B4"]
    for (scen_name, cc), color in zip(CASCADE_SCENARIOS.items(), colors):
        p_vals = []
        for esc in escalation_vals:
            import copy
            params = copy.deepcopy(KEY_PARAMETERS)
            params["outbreak_escalation_rate"].point_estimate = esc
            model = EnhancedStochasticAvoidanceModel(region="national_average", params=params)
            density = model.calculate_network_density(year=2030)
            p = model.calculate_outbreak_probability(density, cascade_completion=cc)
            p_vals.append(p)
        ax.plot(escalation_vals, p_vals, color=color,
                label=f"{scen_name.replace('_', ' ')} (cc={cc:.4f})", linewidth=1.5)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Outbreak escalation rate (above κ)")
    ax.set_ylabel("Annual outbreak probability (2030)")
    ax.set_title("Escalation-rate robustness: does conclusion hold across [1,5]?")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "Fig_sweep_escalation_robustness.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: Fig_sweep_escalation_robustness.png")


def plot_sobol_indices(S1: np.ndarray, ST: np.ndarray, output_dir: str):
    """Fig 2: First-order and total-effect Sobol indices."""
    labels = [d.replace('_', '\n') for d in DIM_NAMES]
    x = np.arange(N_DIMS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width/2, S1, width, label='First-order S1', color='steelblue', alpha=0.8)
    barsT = ax.bar(x + width/2, ST, width, label='Total-effect ST', color='tomato', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Sobol index")
    ax.set_title("Sobol sensitivity indices — coupled model (current policy scenario)")
    ax.legend()
    ax.set_ylim(0, max(ST.max(), S1.max()) * 1.2 + 0.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "Fig_sweep_sobol_indices.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: Fig_sweep_sobol_indices.png")


def plot_coupled_causal(output_dir: str):
    """Fig 3: p_outbreak vs cascade completion — the new causal link visualized."""
    cc_vals = np.linspace(0, 1, 200)

    model_baseline = EnhancedStochasticAvoidanceModel(region="national_average")
    density_2030 = model_baseline.calculate_network_density(year=2030)

    # Point estimate
    p_vals_pt = [model_baseline.calculate_outbreak_probability(density_2030, cascade_completion=cc)
                 for cc in cc_vals]

    # Uncertainty band: vary escalation_rate and kappa across AC-approved ranges
    n_band = 200
    esc_samples  = RNG.uniform(1.0, 5.0, n_band)
    kap_samples  = RNG.uniform(0.25, 0.45, n_band)

    band_matrix = np.zeros((n_band, len(cc_vals)))
    import copy
    for s in range(n_band):
        params_s = copy.deepcopy(KEY_PARAMETERS)
        params_s["outbreak_escalation_rate"].point_estimate = esc_samples[s]
        params_s["critical_network_threshold"].point_estimate = kap_samples[s]
        m = EnhancedStochasticAvoidanceModel(region="national_average", params=params_s)
        d = m.calculate_network_density(year=2030)
        for ci, cc in enumerate(cc_vals):
            band_matrix[s, ci] = m.calculate_outbreak_probability(d, cascade_completion=cc)

    p5  = np.percentile(band_matrix, 5, axis=0)
    p95 = np.percentile(band_matrix, 95, axis=0)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.fill_between(cc_vals, p5, p95, alpha=0.25, color='steelblue',
                    label='90% UI (escalation [1,5], κ [0.25,0.45])')
    ax.plot(cc_vals, p_vals_pt, color='steelblue', linewidth=2, label='Point estimate')

    # Mark current-policy PWID cascade completion
    ax.axvline(CASCADE_SCENARIOS["current_policy"], color='red', linestyle='--',
               linewidth=1.0, label=f"PWID current policy\n(cc={CASCADE_SCENARIOS['current_policy']:.5f})")

    ax.set_xlabel("Cascade completion rate")
    ax.set_ylabel("Annual outbreak probability (2030, national avg)")
    ax.set_title("Option B coupling: cascade failure → outbreak risk")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "Fig_sweep_coupled_causal.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: Fig_sweep_coupled_causal.png")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(scaled: np.ndarray, results: np.ndarray, scenario_names: List[str],
                 S1: np.ndarray, ST: np.ndarray, output_dir: str):
    """Save sweep results to CSV and a summary JSON."""
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    header = DIM_NAMES + [f"p_outbreak_{s}" for s in scenario_names]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(scaled)):
            writer.writerow(list(scaled[i]) + list(results[i]))
    print(f"  Saved: sweep_results.csv ({len(scaled)} rows)")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(scaled)),
        "n_dims": N_DIMS,
        "sweep_dimensions": {
            name: {"lower": float(lo), "upper": float(hi)}
            for name, lo, hi in SWEEP_DIMS
        },
        "cascade_scenarios": CASCADE_SCENARIOS,
        "scenario_summaries": {
            name: {
                "mean": float(results[:, j].mean()),
                "std": float(results[:, j].std()),
                "p5": float(np.percentile(results[:, j], 5)),
                "median": float(np.median(results[:, j])),
                "p95": float(np.percentile(results[:, j], 95)),
                "frac_above_50pct": float((results[:, j] > 0.5).mean()),
            }
            for j, name in enumerate(scenario_names)
        },
        "sobol": {
            "first_order": {DIM_NAMES[k]: float(S1[k]) for k in range(N_DIMS)},
            "total_effect": {DIM_NAMES[k]: float(ST[k]) for k in range(N_DIMS)},
        },
        "escalation_robustness": _escalation_robustness_summary(),
    }

    with open(os.path.join(output_dir, "sweep_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: sweep_summary.json")
    return summary


def _escalation_robustness_summary() -> Dict:
    """Check whether qualitative outbreak conclusion holds across escalation [1,5]."""
    escalation_vals = np.linspace(1.0, 5.0, 20)
    cc_pwid = CASCADE_SCENARIOS["current_policy"]
    import copy

    p_at_rate = {}
    for esc in escalation_vals:
        params = copy.deepcopy(KEY_PARAMETERS)
        params["outbreak_escalation_rate"].point_estimate = esc
        model = EnhancedStochasticAvoidanceModel(region="national_average", params=params)
        density = model.calculate_network_density(year=2030)
        p = model.calculate_outbreak_probability(density, cascade_completion=cc_pwid)
        p_at_rate[round(float(esc), 2)] = round(float(p), 6)

    above_50_rates = [r for r, p in p_at_rate.items() if p > 0.5]
    return {
        "p_outbreak_by_escalation_rate": p_at_rate,
        "n_rates_above_50pct": len(above_50_rates),
        "min_rate_exceeding_50pct": min(above_50_rates) if above_50_rates else None,
        "conclusion_robust_across_full_range": len(above_50_rates) == 20,
        "conclusion_robust_at_lower_half": any(r <= 3.0 and p > 0.5
                                               for r, p in p_at_rate.items()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(n_samples: int = 2000, n_sobol: int = 1000, output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sweep")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  HIGH-DIMENSIONAL SWEEP — Option B coupled model")
    print("=" * 70)

    print("\n1. LHS sweep...")
    scaled, results, scenario_names = run_sweep(n_samples)

    print("\n2. Sobol indices...")
    S1, ST = sobol_indices(n_sobol)
    for k in range(N_DIMS):
        print(f"    {DIM_NAMES[k]:<35}: S1={S1[k]:.3f}  ST={ST[k]:.3f}")

    print("\n3. Figures...")
    plot_escalation_robustness(output_dir)
    plot_sobol_indices(S1, ST, output_dir)
    plot_coupled_causal(output_dir)

    print("\n4. Saving results...")
    summary = save_results(scaled, results, scenario_names, S1, ST, output_dir)

    # Print headline finding
    rob = summary["escalation_robustness"]
    print("\n" + "=" * 70)
    print("  ESCALATION-RATE ROBUSTNESS (headline finding)")
    print("=" * 70)
    print(f"  Fraction of escalation range [1,5] where p_outbreak>50%: "
          f"{rob['n_rates_above_50pct']}/20")
    if rob["conclusion_robust_across_full_range"]:
        print("  ROBUST: outbreak conclusion holds across the full approved range.")
    elif rob["min_rate_exceeding_50pct"] is not None:
        print(f"  PARTIAL: conclusion holds for escalation_rate ≥ {rob['min_rate_exceeding_50pct']}")
    else:
        print("  SENSITIVE: outbreak conclusion does NOT hold across [1,5] — review required.")

    return summary


if __name__ == "__main__":
    main()
