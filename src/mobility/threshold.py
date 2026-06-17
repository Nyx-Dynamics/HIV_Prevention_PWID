"""
Mechanistic epidemic threshold from the generated degree distribution.

Computes τ_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩) (Newman 2002) and R₀ = T / τ_c, where T
is the DURATION-INTEGRATED, ACUTE-WEIGHTED per-edge transmissibility (not the
per-act β). Sweeps explorer fraction and r_g tail to bound τ_c; propagates
full parameter uncertainty into an R₀ credible interval.

GUARDRAIL: This module does NOT modify critical_network_threshold = 0.35 or
any part of the outbreak path. It only computes the mechanistic quantities.
κ = 0.35 is a density threshold (different dimension from τ_c, a
transmissibility threshold); no numerical comparison between them is made.
Replacing the phenomenological cliff is a separate human-reviewed step.

Reference:
  Newman MEJ. Spread of epidemic disease on networks.
  Phys Rev E 66:016128 (2002). DOI 10.1103/PhysRevE.66.016128
"""

from __future__ import annotations

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mobility.network_generator import run_generator, NetworkStats
from mobility.params import MOBILITY_PARAMS, BETA_CAP
import copy as _copy

RNG = np.random.default_rng(42)

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


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def network_threshold(degree_seq: List[int]) -> float:
    """
    Compute network epidemic threshold τ_c from a degree sequence.

    τ_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩)

    τ_c is the critical per-EDGE transmissibility: when T > τ_c, R₀ > 1.
    It is NOT a density threshold and cannot be compared to κ=0.35 (a density
    variable). The only valid comparison is T vs τ_c (same units).

    Returns float('inf') when ⟨k²⟩ ≤ ⟨k⟩ (degenerate / star-free graph).
    """
    k = np.array(degree_seq, dtype=float)
    mean_k = np.mean(k)
    mean_k2 = np.mean(k ** 2)
    denom = mean_k2 - mean_k
    if denom <= 0:
        return float('inf')
    return float(mean_k / denom)


# ─────────────────────────────────────────────────────────────────────────────
# PER-EDGE TRANSMISSIBILITY T (duration-integrated, acute-weighted)
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_edge_T(
    params: dict = None,
    n_samples: int = 5000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Sample the per-EDGE transmissibility T from parameter distributions.

    T integrates β over the full partnership duration, weighted by acute-phase
    elevation:

        β_acute   = min(β_chronic × acute_multiplier, BETA_CAP)
        m_acute   = injection_freq × shared_fraction × acute_duration_days
        m_chronic = injection_freq × shared_fraction × chronic_window_days
        T = 1 − (1−β_acute)^m_acute × (1−β_chronic)^m_chronic

    Parameters
    ----------
    params : MOBILITY_PARAMS dict (or None → use defaults)
    n_samples : number of Monte Carlo draws from parameter distributions
    rng : seeded Generator

    Returns
    -------
    T_samples : (n_samples,) array of per-edge transmissibility values

    SANITY GUARD: if all T_samples < 0.05, a T value from per-act β was likely
    used instead of per-edge T — stop and recheck the inputs.
    """
    if params is None:
        params = MOBILITY_PARAMS
    if rng is None:
        rng = np.random.default_rng(42)

    beta_c = params["beta_chronic_per_shared_injection"].sample(n_samples)
    acute_mult = params["acute_multiplier"].sample(n_samples)
    acute_dur = params["acute_duration_days"].sample(n_samples)
    inj_freq = params["injection_freq_per_day"].sample(n_samples)
    shared_frac = params["shared_fraction_per_partner"].sample(n_samples)
    chronic_win = params["chronic_window_days"].sample(n_samples)

    beta_acute = np.minimum(beta_c * acute_mult, BETA_CAP)

    m_acute = inj_freq * shared_frac * acute_dur
    m_chronic = inj_freq * shared_frac * chronic_win

    T = 1.0 - (1.0 - beta_acute) ** m_acute * (1.0 - beta_c) ** m_chronic
    T = np.clip(T, 0.0, 1.0)

    # Sanity guard: per-act β produces T ≈ 0.02; per-edge T should be ≥ 0.05
    if np.all(T < 0.05):
        raise ValueError(
            "SANITY GUARD: all T < 0.05. Per-act β was likely passed as per-edge T. "
            "R₀ should straddle 1 — if R₀ ≈ 0.02, T is still per-act. Recheck inputs."
        )

    return T


def compute_per_edge_T_from_dyad_intensities(
    edge_intensity_dict: Dict,
    params: dict = None,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 3: Compute per-edge T from dyad intensities.

    For each edge (a, b) with dyad_intensity intensity_ij, T_ij integrates
    β over shared injections:
        m_acute_ij = injection_freq × intensity_ij × acute_duration_days
        T_ij = 1 − (1−β_acute)^m_acute_ij × (1−β_chronic)^m_chronic_ij

    Returns (T_array, intensities_array) in the same order as edge_intensity_dict.

    T is now heterogeneous across edges and positively correlated with node
    propensity (high-propensity hubs have high-T edges → underestimates of
    R₀ from mean T assumption; the actual R₀ is higher due to k-T correlation).
    """
    if params is None:
        params = MOBILITY_PARAMS
    if rng is None:
        rng = np.random.default_rng(42)
    if not edge_intensity_dict:
        return np.array([]), np.array([])

    beta_c = params["beta_chronic_per_shared_injection"].point_estimate
    mult = params["acute_multiplier"].point_estimate
    acute_dur = params["acute_duration_days"].point_estimate
    inj_freq = params["injection_freq_per_day"].point_estimate
    chronic_win = params["chronic_window_days"].point_estimate

    beta_acute = min(beta_c * mult, BETA_CAP)

    intensities = np.array(list(edge_intensity_dict.values()))
    m_acute = inj_freq * intensities * acute_dur
    m_chronic = inj_freq * intensities * chronic_win

    T = 1.0 - (1.0 - beta_acute) ** m_acute * (1.0 - beta_c) ** m_chronic
    T = np.clip(T, 0.0, 1.0)
    return T, intensities


def saturation_diagnostic(params: dict = None) -> Dict:
    """
    Task 1: Report the shared_fraction at which T crosses 0.95 (saturation).

    At point estimates for β_chronic, acute_multiplier, acute_duration,
    injection_freq, and chronic_window, computes the crossover fraction where
    T exceeds 0.95 — i.e., the partnership is essentially certain to transmit.
    Reports whether the current shared_fraction is past that point.

    Returns a dict with crossover_shared_fraction, current_shared_fraction,
    current_T_point, and is_saturated flag.
    """
    if params is None:
        params = MOBILITY_PARAMS

    beta_c = params["beta_chronic_per_shared_injection"].point_estimate
    mult = params["acute_multiplier"].point_estimate
    acute_dur = params["acute_duration_days"].point_estimate
    inj_freq = params["injection_freq_per_day"].point_estimate
    chronic_win = params["chronic_window_days"].point_estimate
    current_sf = params["shared_fraction_per_partner"].point_estimate

    beta_acute = min(beta_c * mult, BETA_CAP)

    # Find crossover via binary search: T(sf) = 0.95
    def T_at_sf(sf):
        m_a = inj_freq * sf * acute_dur
        m_c = inj_freq * sf * chronic_win
        return 1.0 - (1 - beta_acute) ** m_a * (1 - beta_c) ** m_c

    # Binary search in [0, 1]
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if T_at_sf(mid) < 0.95:
            lo = mid
        else:
            hi = mid
    crossover = (lo + hi) / 2

    current_T = T_at_sf(current_sf)

    return {
        "crossover_shared_fraction": float(crossover),
        "current_shared_fraction": float(current_sf),
        "current_T_point_estimate": float(current_T),
        "is_saturated": bool(current_sf >= crossover),
        "saturation_note": (
            f"T saturates (>0.95) at shared_fraction ≥ {crossover:.4f}. "
            f"Current value {current_sf:.4f} is {'PAST' if current_sf >= crossover else 'BELOW'} "
            f"the saturation point (T={current_T:.4f}). "
            "Default placeholder 0.15 was past saturation; calibrated 0.001 is well below it."
        ),
    }


def compute_T_edge_distribution(
    params: dict = None,
    n_samples: int = 5000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Task 1: Sample T_edge from the collapsed, directly-bounded parameter.

    Returns T_samples from MOBILITY_PARAMS["T_edge"] distribution.
    This is the identifiable alternative to the 5-parameter decomposed path.
    """
    if params is None:
        params = MOBILITY_PARAMS
    if rng is None:
        rng = np.random.default_rng(42)
    return params["T_edge"].sample(n_samples)


def r0_on_network(T: float, degree_seq: List[int]) -> float:
    """
    Basic reproduction number on the configuration network.

    R₀ = T · (⟨k²⟩ − ⟨k⟩) / ⟨k⟩  =  T / τ_c

    T MUST be the per-EDGE transmissibility (duration-integrated), not per-act β.
    """
    k = np.array(degree_seq, dtype=float)
    mean_k = np.mean(k)
    mean_k2 = np.mean(k ** 2)
    if mean_k == 0:
        return 0.0
    return float(T * (mean_k2 - mean_k) / mean_k)


def compute_r0_distribution(
    T_samples: np.ndarray,
    tau_c_samples: np.ndarray,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Compute R₀ = T / τ_c sample-by-sample and summarize.

    Uses broadcast: pairs each T draw with the corresponding τ_c draw
    (or a single τ_c value if the degree distribution is fixed).

    Returns
    -------
    r0_samples : (n,) array
    p_r0_gt_1 : P(R₀ > 1)
    summary : dict with credible interval, median, P(R₀>1)
    """
    finite_mask = np.isfinite(tau_c_samples)
    if not np.any(finite_mask):
        return np.zeros_like(T_samples), 0.0, {"error": "all tau_c infinite"}

    T_fin = T_samples[finite_mask]
    tc_fin = tau_c_samples[finite_mask]

    r0 = T_fin / tc_fin
    p_gt1 = float(np.mean(r0 > 1.0))

    summary = {
        "mean": float(np.mean(r0)),
        "median": float(np.median(r0)),
        "p5": float(np.percentile(r0, 5)),
        "p25": float(np.percentile(r0, 25)),
        "p75": float(np.percentile(r0, 75)),
        "p95": float(np.percentile(r0, 95)),
        "p_r0_gt_1": p_gt1,
        "straddles_1": bool(np.percentile(r0, 5) < 1.0 < np.percentile(r0, 95)),
    }

    return r0, p_gt1, summary


# ─────────────────────────────────────────────────────────────────────────────
# EXPLORER FRACTION / r_g TAIL SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def sweep_tau_c(
    n_agents: int = 300,
    n_steps: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep EPR ρ and r_g scale, computing τ_c at each point.

    Returns (rho_vals, rg_vals, tau_c_matrix) — τ_c is a transmissibility,
    not a density. No comparison to κ=0.35 (different dimension).
    """
    rho_vals = np.linspace(0.30, 0.90, 8)
    rg_vals = np.array([1.0, 2.4, 5.0])

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
# FIGURE — τ_c with R₀=1 locus (T=τ_c line)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tau_c_with_r0_locus(
    rho_vals: np.ndarray,
    rg_vals: np.ndarray,
    tau_c_matrix: np.ndarray,
    T_summary: Dict,
    output_dir: str,
):
    """
    τ_c vs explorer fraction (ρ) and r_g.

    Reference line = T_median (per-edge transmissibility point estimate).
    Where τ_c < T_median, R₀ > 1 (epidemic above this line).
    κ=0.35 is NOT plotted — it is a density variable (different dimension).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    rg_labels = [f"r_g={rg:.1f} km" for rg in rg_vals]
    colors = ['steelblue', 'tomato', 'seagreen']

    for j, (rg_label, color) in enumerate(zip(rg_labels, colors)):
        tc_col = tau_c_matrix[:, j]
        finite_mask = np.isfinite(tc_col)
        ax.plot(rho_vals[finite_mask], tc_col[finite_mask],
                marker='o', markersize=4, color=color, label=rg_label, linewidth=1.5)

    T_med = T_summary.get("T_median", None)
    T_p5 = T_summary.get("T_p5", None)
    T_p95 = T_summary.get("T_p95", None)

    if T_med is not None:
        ax.axhline(T_med, color='black', linestyle='--', linewidth=1.5,
                   label=f'T median = {T_med:.3f} (R0=1 locus)')
    if T_p5 is not None and T_p95 is not None:
        ax.axhspan(T_p5, T_p95, alpha=0.10, color='black',
                   label=f'T [5th–95th pct] = [{T_p5:.3f}, {T_p95:.3f}]')

    ax.set_xlabel("EPR exploration scaling ρ (controls explorer fraction → ⟨k²⟩ → τ_c)")
    ax.set_ylabel("τ_c (critical per-edge transmissibility)")
    ax.set_title(
        "Mechanistic τ_c vs explorer fraction / activity radius\n"
        "Reference: T (per-edge, acute-weighted) — where τ_c < T, R₀ > 1\n"
        "Note: κ=0.35 is a DENSITY threshold (different dimension) — not shown"
    )
    ax.legend(fontsize=7.5)
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
    n_T_samples: int = 5000,
    output_dir: str = None,
) -> Dict:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

    rng = np.random.default_rng(seed)

    print("\n" + "=" * 72)
    print("  MECHANISTIC EPIDEMIC THRESHOLD — Stage 3 (corrected)")
    print("=" * 72)

    # ── 1. Generate point-estimate network ──────────────────────────────────
    print("\n1. Generating point-estimate network...")
    _, stats, _ = run_generator(n_agents=n_agents, n_steps=n_steps, seed=seed)
    tau_c_point = network_threshold(stats.degree_sequence)

    print(f"\n  Network statistics:")
    print(f"    ⟨k⟩   = {stats.mean_degree:.3f}")
    print(f"    ⟨k²⟩  = {stats.k2_moment:.3f}  [SENSITIVITY RANGE — dominant τ_c input]")
    if np.isfinite(tau_c_point):
        print(f"    τ_c   = {tau_c_point:.4f}  (critical per-edge T; R₀=1 when T=τ_c)")
    else:
        print(f"    τ_c   = ∞  (degenerate degree sequence; ⟨k²⟩ ≈ ⟨k⟩)")

    # ── 2a. Saturation diagnostic ────────────────────────────────────────────
    print("\n2a. Saturation diagnostic (Task 1)...")
    sat = saturation_diagnostic(MOBILITY_PARAMS)
    print(f"    {sat['saturation_note']}")

    # ── 2b. Decomposed T distribution ───────────────────────────────────────
    print("\n2b. Decomposed T (duration-integrated, acute-weighted)...")
    T_samples = compute_per_edge_T(params=MOBILITY_PARAMS, n_samples=n_T_samples, rng=rng)

    T_summary = {
        "T_mean": float(np.mean(T_samples)),
        "T_median": float(np.median(T_samples)),
        "T_p5": float(np.percentile(T_samples, 5)),
        "T_p25": float(np.percentile(T_samples, 25)),
        "T_p75": float(np.percentile(T_samples, 75)),
        "T_p95": float(np.percentile(T_samples, 95)),
    }

    print(f"    T_decomposed median = {T_summary['T_median']:.4f}")
    print(f"    T_decomposed [5th, 95th] = [{T_summary['T_p5']:.4f}, {T_summary['T_p95']:.4f}]")

    if T_summary['T_p95'] < 0.05:
        print("  ⚠ SANITY GUARD: T_p95 < 0.05 — possible per-act β error. Stop and recheck.")

    # ── 2c. Collapsed T_edge ─────────────────────────────────────────────────
    print("\n2c. Collapsed T_edge (directly bounded, identifiable)...")
    T_edge_samples = compute_T_edge_distribution(params=MOBILITY_PARAMS, n_samples=n_T_samples, rng=rng)
    T_edge_summary = {
        "T_edge_median": float(np.median(T_edge_samples)),
        "T_edge_p5": float(np.percentile(T_edge_samples, 5)),
        "T_edge_p95": float(np.percentile(T_edge_samples, 95)),
    }
    print(f"    T_edge median = {T_edge_summary['T_edge_median']:.4f}")
    print(f"    T_edge [5th, 95th] = [{T_edge_summary['T_edge_p5']:.4f}, {T_edge_summary['T_edge_p95']:.4f}]")

    # ── 3. R₀ distribution (both paths) ─────────────────────────────────────
    print("\n3. Computing R₀ = T / τ_c distribution (both paths)...")

    # Propagate τ_c uncertainty via the sweep (ρ, r_g)
    # For R₀ credible interval, sample τ_c from the sweep range as a uniform proxy
    rho_vals, rg_vals, tau_c_matrix = sweep_tau_c(n_agents=n_agents, n_steps=n_steps, seed=seed)
    finite_tcs = tau_c_matrix[np.isfinite(tau_c_matrix)].flatten()

    if len(finite_tcs) > 0:
        tc_min, tc_max = float(finite_tcs.min()), float(finite_tcs.max())
        # Sample τ_c uniformly from sweep range as proxy for uncertainty
        tau_c_samples = rng.uniform(tc_min, tc_max, n_T_samples)
    else:
        tau_c_samples = np.full(n_T_samples, tau_c_point)

    # Decomposed path
    r0_samples, p_r0_gt1, r0_summary = compute_r0_distribution(T_samples, tau_c_samples)

    print(f"    [decomposed] R₀ median = {r0_summary['median']:.3f}  "
          f"[5–95th: {r0_summary['p5']:.3f}–{r0_summary['p95']:.3f}]  "
          f"P(R₀>1)={p_r0_gt1:.3f}  straddles={r0_summary['straddles_1']}")

    # Collapsed T_edge path
    r0_edge_samples, p_r0_edge_gt1, r0_edge_summary = compute_r0_distribution(
        T_edge_samples, tau_c_samples
    )
    print(f"    [T_edge]     R₀ median = {r0_edge_summary['median']:.3f}  "
          f"[5–95th: {r0_edge_summary['p5']:.3f}–{r0_edge_summary['p95']:.3f}]  "
          f"P(R₀>1)={p_r0_edge_gt1:.3f}  straddles={r0_edge_summary['straddles_1']}")

    # ── 4. Dimension note (replaces the retired κ comparison) ───────────────
    print("\n" + "-" * 72)
    print("  NOTE: κ = 0.35 is a DENSITY threshold; τ_c is a TRANSMISSIBILITY")
    print("  threshold. They live on orthogonal axes and cannot be compared.")
    print("  κ = 0.35 will be RETIRED (not matched) in the eventual swap.")
    print("  The valid comparison is T vs τ_c (same units): R₀ = T / τ_c.")
    print("-" * 72)

    # ── 5. τ_c sweep summary ─────────────────────────────────────────────────
    print(f"\n  τ_c sweep range [{tc_min:.4f}, {tc_max:.4f}]")
    print(f"  τ_c point estimate: {tau_c_point:.4f}")

    plot_tau_c_with_r0_locus(rho_vals, rg_vals, tau_c_matrix, T_summary, output_dir)

    # ── 6. Save JSON ─────────────────────────────────────────────────────────
    report = {
        "seed": seed,
        "network": {
            "mean_degree": stats.mean_degree,
            "k2_moment": stats.k2_moment,
            "k2_is_sensitivity_range": True,
        },
        "saturation_diagnostic": sat,
        "tau_c": {
            "point_estimate": tau_c_point if np.isfinite(tau_c_point) else None,
            "sweep_min": tc_min,
            "sweep_median": float(np.median(finite_tcs)) if len(finite_tcs) else None,
            "sweep_max": tc_max,
            "note": "tau_c is a per-edge transmissibility threshold (T units). "
                    "NOT comparable to kappa=0.35 (a density variable).",
        },
        "T_decomposed": T_summary,
        "T_edge": T_edge_summary,
        "r0_decomposed": {
            **r0_summary,
            "formula": "R0 = T_decomposed / tau_c",
            "T_units": "per-edge (duration-integrated, acute-weighted)",
        },
        "r0_T_edge": {
            **r0_edge_summary,
            "formula": "R0 = T_edge / tau_c",
            "T_units": "per-edge (collapsed, directly bounded)",
        },
        "dimension_note": (
            "kappa=0.35 (density threshold) and tau_c (transmissibility threshold) "
            "are on orthogonal axes. No numerical comparison is made. "
            "kappa=0.35 will be RETIRED, not matched, in the eventual swap."
        ),
        "guardrail": (
            "critical_network_threshold=0.35 and all outbreak-path code are untouched. "
            "Replacing the phenomenological threshold is a separate human-reviewed step."
        ),
    }

    report_path = os.path.join(output_dir, "threshold_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {report_path}")

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  τ_c (point)                  = {tau_c_point:.4f}  [sweep: {tc_min:.4f}–{tc_max:.4f}]")
    print(f"  Saturation crossover         = sf ≥ {sat['crossover_shared_fraction']:.4f}  "
          f"(current sf={sat['current_shared_fraction']:.5f}, {'SATURATED' if sat['is_saturated'] else 'below'})")
    print(f"  T_decomposed median          = {T_summary['T_median']:.4f}  "
          f"[5–95th: {T_summary['T_p5']:.4f}–{T_summary['T_p95']:.4f}]")
    print(f"  T_edge median (collapsed)    = {T_edge_summary['T_edge_median']:.4f}  "
          f"[5–95th: {T_edge_summary['T_edge_p5']:.4f}–{T_edge_summary['T_edge_p95']:.4f}]")
    print(f"  R₀ (decomposed T)  median    = {r0_summary['median']:.3f}  "
          f"[{r0_summary['p5']:.3f}–{r0_summary['p95']:.3f}]  P(>1)={p_r0_gt1:.3f}  "
          f"straddles={r0_summary['straddles_1']}")
    print(f"  R₀ (T_edge collapsed) median = {r0_edge_summary['median']:.3f}  "
          f"[{r0_edge_summary['p5']:.3f}–{r0_edge_summary['p95']:.3f}]  P(>1)={p_r0_edge_gt1:.3f}  "
          f"straddles={r0_edge_summary['straddles_1']}")

    return report


if __name__ == "__main__":
    run_threshold_analysis()
