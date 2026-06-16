"""
Stage 3: Mechanistic epidemic threshold from the generated degree distribution.

Computes τ_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩) (Newman 2002) and R₀ on the configuration
network, then sweeps the explorer fraction and r_g tail to bound τ_c.

GUARDRAIL: This module does NOT modify critical_network_threshold = 0.35 or
any part of the outbreak path. It only computes the mechanistic alternative and
compares it. Replacing the phenomenological threshold is a separate human-
reviewed step — see docs/SOURCING_mobility_network.md §6.

Reference:
  Newman MEJ. Spread of epidemic disease on networks.
  Phys Rev E 66:016128 (2002). DOI 10.1103/PhysRevE.66.016128
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mobility.network_generator import run_generator, NetworkStats

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

# Current hardcoded value in stochastic_avoidance_enhanced.py KEY_PARAMETERS
CURRENT_PHENOMENOLOGICAL_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def network_threshold(degree_seq: List[int]) -> float:
    """
    Compute the network epidemic threshold τ_c from a degree sequence.

    τ_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩)

    This is the per-partnership transmissibility at which the epidemic
    transitions from subcritical to supercritical (R₀ = 1).

    Reference: Newman MEJ. Phys Rev E 66:016128 (2002). DOI 10.1103/PhysRevE.66.016128

    NOTE: τ_c → ∞ as ⟨k²⟩ → ⟨k⟩ (star graphs / no heterogeneity).
    In scale-free networks (⟨k²⟩ → ∞), τ_c → 0 — epidemics spread at
    any transmissibility. The high-degree tail is therefore the dominant
    uncertainty source.

    Returns float or np.inf if ⟨k²⟩ ≤ ⟨k⟩ (degenerate case).
    """
    k = np.array(degree_seq, dtype=float)
    mean_k = np.mean(k)
    mean_k2 = np.mean(k ** 2)
    denom = mean_k2 - mean_k
    if denom <= 0:
        return float('inf')
    return float(mean_k / denom)


def r0_on_network(T: float, degree_seq: List[int]) -> float:
    """
    Basic reproduction number on the configuration network.

    R₀ = T · (⟨k²⟩ − ⟨k⟩) / ⟨k⟩

    where T is per-partnership transmissibility.

    Reference: Newman MEJ. Phys Rev E 66:016128 (2002).
    """
    k = np.array(degree_seq, dtype=float)
    mean_k = np.mean(k)
    mean_k2 = np.mean(k ** 2)
    if mean_k == 0:
        return 0.0
    return float(T * (mean_k2 - mean_k) / mean_k)


# ─────────────────────────────────────────────────────────────────────────────
# EXPLORER FRACTION / r_g TAIL SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def sweep_tau_c(
    n_agents: int = 300,
    n_steps: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep EPR exploration parameter ρ (which controls the explorer fraction
    and hence ⟨k²⟩ / tail) and r_g scale, computing τ_c at each point.

    Returns (rho_vals, rg_vals, tau_c_matrix) where tau_c_matrix[i,j] is
    the threshold at rho_vals[i], rg_vals[j].
    """
    rho_vals = np.linspace(0.30, 0.90, 8)
    rg_vals = np.array([1.0, 2.4, 5.0])  # km — low / point estimate / high

    tau_c_matrix = np.zeros((len(rho_vals), len(rg_vals)))

    for i, rho in enumerate(rho_vals):
        for j, rg in enumerate(rg_vals):
            _, stats, _ = run_generator(
                n_agents=n_agents,
                n_steps=n_steps,
                rg_scale_km=rg,
                epr_rho=rho,
                seed=seed,
            )
            tau_c_matrix[i, j] = network_threshold(stats.degree_sequence)

    return rho_vals, rg_vals, tau_c_matrix


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_tau_c_sensitivity(
    rho_vals: np.ndarray,
    rg_vals: np.ndarray,
    tau_c_matrix: np.ndarray,
    tau_c_point: float,
    output_dir: str,
):
    """τ_c vs explorer fraction (ρ) and r_g — with current 0.35 reference line."""
    fig, ax = plt.subplots(figsize=(6, 4))

    rg_labels = [f"r_g={rg:.1f} km" for rg in rg_vals]
    colors = ['steelblue', 'tomato', 'seagreen']

    for j, (rg_label, color) in enumerate(zip(rg_labels, colors)):
        tc_col = tau_c_matrix[:, j]
        finite_mask = np.isfinite(tc_col)
        ax.plot(rho_vals[finite_mask], tc_col[finite_mask],
                marker='o', markersize=4, color=color, label=rg_label, linewidth=1.5)

    ax.axhline(CURRENT_PHENOMENOLOGICAL_THRESHOLD, color='black', linestyle='--',
               linewidth=1.5, label=f'Current κ = {CURRENT_PHENOMENOLOGICAL_THRESHOLD} '
                                     f'(phenomenological)')

    if np.isfinite(tau_c_point):
        ax.axhline(tau_c_point, color='purple', linestyle=':',
                   linewidth=1.5, label=f'Mechanistic τ_c = {tau_c_point:.4f} '
                                         f'(point est.)')

    ax.set_xlabel("EPR exploration scaling ρ (controls explorer fraction → ⟨k²⟩ → τ_c)")
    ax.set_ylabel("Network epidemic threshold τ_c")
    ax.set_title("Mechanistic τ_c vs explorer fraction and activity radius\n"
                 "(τ_c is where R₀=1; below τ_c = epidemic)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "Fig_tau_c_sensitivity.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REPORT
# ─────────────────────────────────────────────────────────────────────────────

def run_threshold_analysis(
    n_agents: int = 300,
    n_steps: int = 50,
    seed: int = 42,
    output_dir: str = None,
) -> Dict:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

    print("\n" + "=" * 72)
    print("  MECHANISTIC EPIDEMIC THRESHOLD — Stage 3")
    print("=" * 72)

    # Point estimate network
    print("\n1. Generating point-estimate network...")
    _, stats, _ = run_generator(n_agents=n_agents, n_steps=n_steps, seed=seed)
    tau_c_point = network_threshold(stats.degree_sequence)

    # Per-partnership transmissibility for HIV (from params.py)
    T_hiv = 0.008  # per-shared-injection; Baggaley meta-analysis
    r0_point = r0_on_network(T_hiv, stats.degree_sequence)

    print(f"\n  Network statistics (point estimate):")
    print(f"    ⟨k⟩   = {stats.mean_degree:.3f}")
    print(f"    ⟨k²⟩  = {stats.k2_moment:.3f}  ← SENSITIVITY RANGE, not a point")
    print(f"    τ_c   = {tau_c_point:.4f}  (epidemic threshold; T=τ_c → R₀=1)")
    print(f"    R₀    = {r0_point:.4f}  (at T={T_hiv}, per-shared-injection HIV β)")

    # Comparison paragraph
    print("\n" + "-" * 72)
    print("  COMPARISON: mechanistic τ_c vs current κ = 0.35")
    print("-" * 72)

    if np.isfinite(tau_c_point):
        direction = "LOWER" if tau_c_point < CURRENT_PHENOMENOLOGICAL_THRESHOLD else "HIGHER"
        ratio = tau_c_point / CURRENT_PHENOMENOLOGICAL_THRESHOLD
        print(f"\n  Mechanistic τ_c = {tau_c_point:.4f}")
        print(f"  Current κ        = {CURRENT_PHENOMENOLOGICAL_THRESHOLD:.4f}")
        print(f"  Direction: mechanistic threshold is {direction} than κ (ratio = {ratio:.2f})")
        if tau_c_point < CURRENT_PHENOMENOLOGICAL_THRESHOLD:
            print(
                f"\n  IMPLICATION: the mechanistic threshold is more permissive —\n"
                f"  at the generated network density, R₀ > 1 (supercritical) at a\n"
                f"  lower transmissibility than the current κ implies. The current\n"
                f"  κ=0.35 effectively underestimates outbreak risk at this network\n"
                f"  structure. HOWEVER: τ_c is dominated by ⟨k²⟩ which is the\n"
                f"  most under-captured quantity — see sweep below."
            )
        else:
            print(
                f"\n  IMPLICATION: the mechanistic threshold is more conservative —\n"
                f"  the current κ=0.35 implies greater outbreak risk than this\n"
                f"  network structure would predict. Note: ⟨k²⟩ uncertainty dominates."
            )
    else:
        print(f"\n  τ_c → ∞ (degenerate degree distribution; ⟨k²⟩ ≈ ⟨k⟩).")
        print(f"  This means no hub structure — epidemics require high T to spread.")

    # Sweep
    print("\n2. Sweeping explorer fraction (ρ) and r_g tail...")
    rho_vals, rg_vals, tau_c_matrix = sweep_tau_c(n_agents=n_agents, n_steps=n_steps, seed=seed)

    # Summary statistics across the sweep
    finite_tcs = tau_c_matrix[np.isfinite(tau_c_matrix)]
    if len(finite_tcs) > 0:
        tc_min, tc_max, tc_med = finite_tcs.min(), finite_tcs.max(), np.median(finite_tcs)
        print(f"\n  τ_c across sweep:")
        print(f"    min  = {tc_min:.4f}")
        print(f"    median = {tc_med:.4f}")
        print(f"    max  = {tc_max:.4f}")
        print(f"    Current κ = {CURRENT_PHENOMENOLOGICAL_THRESHOLD:.4f}")
        frac_below = (finite_tcs < CURRENT_PHENOMENOLOGICAL_THRESHOLD).mean()
        print(f"    Fraction of sweep where τ_c < 0.35: {frac_below:.2f}")
        print(f"    (i.e., fraction where mechanistic threshold is more permissive than κ=0.35)")
    else:
        tc_min = tc_max = tc_med = float('nan')

    plot_tau_c_sensitivity(rho_vals, rg_vals, tau_c_matrix, tau_c_point, output_dir)

    # Save JSON report
    report = {
        "point_estimate": {
            "mean_degree": stats.mean_degree,
            "k2_moment": stats.k2_moment,
            "tau_c": tau_c_point if np.isfinite(tau_c_point) else None,
            "r0_at_T_hiv": r0_point,
            "T_hiv": T_hiv,
        },
        "comparison": {
            "tau_c_mechanistic": tau_c_point if np.isfinite(tau_c_point) else None,
            "kappa_phenomenological": CURRENT_PHENOMENOLOGICAL_THRESHOLD,
            "direction": direction if np.isfinite(tau_c_point) else "degenerate",
            "ratio": ratio if np.isfinite(tau_c_point) else None,
        },
        "sweep": {
            "rho_vals": rho_vals.tolist(),
            "rg_vals_km": rg_vals.tolist(),
            "tau_c_matrix": [[v if np.isfinite(v) else None for v in row]
                              for row in tau_c_matrix.tolist()],
            "tau_c_min": float(tc_min) if not np.isnan(tc_min) else None,
            "tau_c_median": float(tc_med) if not np.isnan(tc_med) else None,
            "tau_c_max": float(tc_max) if not np.isnan(tc_max) else None,
            "frac_below_kappa": float(frac_below) if len(finite_tcs) > 0 else None,
        },
        "guardrail": (
            "This report does NOT modify critical_network_threshold=0.35 or any "
            "outbreak-path code. Replacing the phenomenological threshold is a "
            "separate human-reviewed step."
        ),
    }

    report_path = os.path.join(output_dir, "threshold_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {report_path}")

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Mechanistic τ_c (point) = {tau_c_point:.4f}")
    print(f"  τ_c sweep range         = [{tc_min:.4f}, {tc_max:.4f}]")
    print(f"  Current κ               = {CURRENT_PHENOMENOLOGICAL_THRESHOLD}")
    print(f"  Note: ⟨k²⟩ uncertainty dominates τ_c. Do not report a single τ_c")
    print(f"  number without the sweep range.")

    return report


if __name__ == "__main__":
    run_threshold_analysis()
