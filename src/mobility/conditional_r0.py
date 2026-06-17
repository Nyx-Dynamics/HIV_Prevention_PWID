"""
Task 5: Conditional R₀ headline — network under each candidate proxy.

Runs the network generator under each of the 3 attractor-layer candidates
(SSP-only, SSP+diffuse markets, SSP+concentrated hotspots), validates the
network structure, computes R₀ = T / τ_c for both the decomposed and T_edge
paths, and reports P(R₀>1) and the straddle-1 verdict per candidate.

Also produces a parameter-sensitivity ranking via one-at-a-time (OAT) sweep
over the key parameters driving R₀.

ALL results are conditional on the candidate proxy and flagged pending AC review.
No proxy is enshrined as final. The drug-market proxy choice is AC's judgment call.
"""

from __future__ import annotations

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mobility.network_generator import run_generator
from mobility.threshold import (
    network_threshold, compute_per_edge_T, compute_T_edge_distribution,
    compute_r0_distribution, saturation_diagnostic
)
from mobility.validate_network import validate, print_validation_report
from mobility.attractor_proxies import all_proxies, PROXY_DESCRIPTIONS
from mobility.params import MOBILITY_PARAMS

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

RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK UNDER EACH PROXY
# ─────────────────────────────────────────────────────────────────────────────

def run_proxy_network(
    proxy_name: str,
    venues: list,
    n_agents: int = 300,
    n_steps: int = 80,
    venue_return_boost: float = 5.0,
    seed: int = 42,
) -> Dict:
    """
    Run network generator under a given proxy venue layer.
    venue_return_boost=5.0 ensures venues create meaningful clustering
    (agents strongly prefer returning to venue locations).
    """
    _, stats, _ = run_generator(
        n_agents=n_agents,
        n_steps=n_steps,
        venues=venues,
        venue_return_boost=venue_return_boost,
        seed=seed,
    )

    tau_c = network_threshold(stats.degree_sequence)
    return {"proxy": proxy_name, "stats": stats, "tau_c": tau_c}


# ─────────────────────────────────────────────────────────────────────────────
# R₀ UNDER EACH PROXY
# ─────────────────────────────────────────────────────────────────────────────

def compute_conditional_r0(
    tau_c: float,
    n_samples: int = 3000,
) -> Dict:
    """
    Compute R₀ = T / τ_c for both T paths at a given τ_c value.

    Returns dict with results for decomposed and T_edge paths.
    """
    rng = np.random.default_rng(42)

    if not np.isfinite(tau_c) or tau_c <= 0:
        return {
            "tau_c": tau_c,
            "note": "tau_c degenerate (infinite or zero); R₀ not computable.",
        }

    T_decomp = compute_per_edge_T(params=MOBILITY_PARAMS, n_samples=n_samples, rng=rng)
    T_edge = compute_T_edge_distribution(params=MOBILITY_PARAMS, n_samples=n_samples,
                                         rng=np.random.default_rng(43))

    tau_c_arr = np.full(n_samples, tau_c)

    _, p_decomp, s_decomp = compute_r0_distribution(T_decomp, tau_c_arr)
    _, p_edge, s_edge = compute_r0_distribution(T_edge, tau_c_arr)

    return {
        "tau_c": float(tau_c),
        "decomposed": {**s_decomp, "p_r0_gt1": p_decomp},
        "T_edge": {**s_edge, "p_r0_gt1": p_edge},
    }


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY RANKING (one-at-a-time)
# ─────────────────────────────────────────────────────────────────────────────

def oat_sensitivity_ranking(
    tau_c_reference: float = 0.33,
    n_samples: int = 500,
) -> List[Dict]:
    """
    One-at-a-time sensitivity: vary each parameter ±1 std from its distribution
    and measure the change in R₀ median (using T_edge for clarity).

    Returns list sorted by |ΔR₀| descending.
    """
    import copy

    if not np.isfinite(tau_c_reference) or tau_c_reference <= 0:
        return []

    tau_c_arr = np.full(n_samples, tau_c_reference)

    def r0_median(params):
        rng = np.random.default_rng(42)
        T = compute_T_edge_distribution(params=params, n_samples=n_samples, rng=rng)
        tau_samp = np.full(n_samples, tau_c_reference)
        r0, _, _ = compute_r0_distribution(T, tau_samp)
        return float(np.median(r0[np.isfinite(r0)]))

    base_r0 = r0_median(MOBILITY_PARAMS)

    results = []
    for name, param in MOBILITY_PARAMS.items():
        if param.lower_bound >= param.upper_bound:
            continue  # degenerate/placeholder — skip
        try:
            low_params = copy.deepcopy(MOBILITY_PARAMS)
            low_params[name].point_estimate = param.lower_bound
            low_params[name].lower_bound = param.lower_bound
            low_params[name].upper_bound = param.lower_bound

            high_params = copy.deepcopy(MOBILITY_PARAMS)
            high_params[name].point_estimate = param.upper_bound
            high_params[name].lower_bound = param.upper_bound
            high_params[name].upper_bound = param.upper_bound

            r0_low = r0_median(low_params)
            r0_high = r0_median(high_params)
            delta = abs(r0_high - r0_low)
            results.append({
                "parameter": name,
                "r0_at_low": round(r0_low, 4),
                "r0_at_high": round(r0_high, 4),
                "delta_r0": round(delta, 4),
                "source_note": param.source[:60],
            })
        except Exception:
            pass

    return sorted(results, key=lambda x: x["delta_r0"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_conditional_r0(proxy_results: List[Dict], output_dir: str):
    """
    Plot R₀ distributions per proxy (T_edge path — the identifiable estimate).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for ax, result in zip(axes, proxy_results):
        proxy = result["proxy"]
        r0_data = result.get("r0_results", {})
        tau_c = result.get("tau_c", None)

        if not r0_data or not np.isfinite(tau_c or float('nan')):
            ax.text(0.5, 0.5, "N/A\n(degenerate τ_c)", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(proxy.replace('_', '\n'), fontsize=9)
            continue

        edge_r = r0_data.get("T_edge", {})
        decomp_r = r0_data.get("decomposed", {})

        x = np.linspace(0, 4, 200)
        ax.set_title(proxy.replace('_', ' '), fontsize=8, wrap=True)

        # Draw boxplot-style bars
        for label, r, color in [
            ("T_edge", edge_r, "steelblue"),
            ("decomposed", decomp_r, "tomato"),
        ]:
            if r:
                ax.barh([label], [r.get("p95", 0) - r.get("p5", 0)],
                        left=r.get("p5", 0), height=0.35,
                        color=color, alpha=0.6, label=f"{label}")
                ax.scatter([r.get("median", 0)], [label], color=color, zorder=5)

        ax.axvline(1.0, color='black', linestyle='--', linewidth=1.0, label='R₀=1')
        ax.set_xlabel("R₀ = T / τ_c")
        ax.set_xlim(0, max(4, (edge_r.get("p95", 3) * 1.1) if edge_r else 4))

        p_gt1 = edge_r.get("p_r0_gt1", None)
        straddles = edge_r.get("straddles_1", None)
        if p_gt1 is not None:
            ax.text(0.02, 0.02, f"P(R₀>1)={p_gt1:.2f}\nstraddles={straddles}",
                    transform=ax.transAxes, fontsize=7.5,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.suptitle(
        "Conditional R₀ by attractor proxy — 5th–95th pct bars, median dot\n"
        "ALL RESULTS PENDING AC REVIEW — drug-market proxy is AC's judgment call",
        fontsize=9
    )
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Fig_conditional_r0.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sensitivity_ranking(ranking: List[Dict], output_dir: str):
    """Bar chart of OAT sensitivity by parameter."""
    if not ranking:
        return

    params = [r["parameter"][:25] for r in ranking[:10]]
    deltas = [r["delta_r0"] for r in ranking[:10]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(params[::-1], deltas[::-1], color='steelblue', alpha=0.75)
    ax.set_xlabel("|ΔR₀| (high−low bound, T_edge path)")
    ax.set_title("One-at-a-time sensitivity ranking (top 10)\nLarger bar = parameter drives R₀ more")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "Fig_sensitivity_ranking.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_conditional_r0(
    n_agents: int = 300,
    n_steps: int = 80,
    venue_return_boost: float = 5.0,
    seed: int = 42,
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  CONDITIONAL R₀ — network under each attractor proxy")
    print("  ALL RESULTS CONDITIONAL ON PROXY; PENDING AC REVIEW")
    print("=" * 72)

    proxies = all_proxies()
    proxy_results = []

    for proxy_name, venues in proxies.items():
        print(f"\n{'─'*60}")
        print(f"  PROXY: {proxy_name}")
        print(f"  {PROXY_DESCRIPTIONS[proxy_name][:120]}")
        print(f"{'─'*60}")

        result = run_proxy_network(proxy_name, venues, n_agents, n_steps,
                                   venue_return_boost, seed)
        stats = result["stats"]
        tau_c = result["tau_c"]

        # Validate network
        checks = validate(stats)
        print_validation_report(stats, checks)

        # R₀ at this τ_c
        r0_results = compute_conditional_r0(tau_c, n_samples=1000)
        result["r0_results"] = r0_results

        d_r0 = r0_results.get("decomposed", {})
        e_r0 = r0_results.get("T_edge", {})

        print(f"\n  τ_c = {tau_c:.4f}")
        if d_r0:
            print(f"  R₀ (decomposed T): median={d_r0.get('median',0):.3f}  "
                  f"[{d_r0.get('p5',0):.3f}–{d_r0.get('p95',0):.3f}]  "
                  f"P(R₀>1)={d_r0.get('p_r0_gt1',0):.3f}  "
                  f"straddles={d_r0.get('straddles_1')}")
        if e_r0:
            print(f"  R₀ (T_edge):       median={e_r0.get('median',0):.3f}  "
                  f"[{e_r0.get('p5',0):.3f}–{e_r0.get('p95',0):.3f}]  "
                  f"P(R₀>1)={e_r0.get('p_r0_gt1',0):.3f}  "
                  f"straddles={e_r0.get('straddles_1')}")

        proxy_results.append(result)

    # Sensitivity ranking (using reference τ_c from SSP-only)
    print("\n" + "=" * 72)
    print("  SENSITIVITY RANKING (OAT, T_edge path)")
    print("=" * 72)
    ref_tau_c = proxy_results[0]["tau_c"] if proxy_results else 0.33
    ranking = oat_sensitivity_ranking(tau_c_reference=ref_tau_c, n_samples=300)
    for r in ranking[:10]:
        print(f"  {r['parameter']:<35}: |ΔR₀|={r['delta_r0']:.4f}  "
              f"[{r['r0_at_low']:.3f}→{r['r0_at_high']:.3f}]")

    # Figures
    plot_conditional_r0(proxy_results, output_dir)
    plot_sensitivity_ranking(ranking, output_dir)

    # Save JSON
    report = {
        "seed": seed,
        "venue_return_boost": venue_return_boost,
        "saturation_diagnostic": saturation_diagnostic(MOBILITY_PARAMS),
        "proxies": [
            {
                "proxy": r["proxy"],
                "description": PROXY_DESCRIPTIONS[r["proxy"]],
                "network_stats": {
                    "mean_degree": r["stats"].mean_degree,
                    "k2_moment": r["stats"].k2_moment,
                    "dispersion": (r["stats"].k2_moment / r["stats"].mean_degree**2
                                   if r["stats"].mean_degree > 0 else None),
                    "clustering": r["stats"].clustering_coefficient,
                    "giant_component_fraction": r["stats"].giant_component_fraction,
                    "mean_path_length": r["stats"].mean_path_length,
                    "n_edges": r["stats"].n_edges,
                },
                "tau_c": r["tau_c"] if np.isfinite(r.get("tau_c", float('nan'))) else None,
                "r0_conditional": r.get("r0_results", {}),
            }
            for r in proxy_results
        ],
        "sensitivity_ranking_top10": ranking[:10],
        "disclaimer": (
            "ALL RESULTS ARE CONDITIONAL ON THE ATTRACTOR PROXY. "
            "No proxy is enshrined as final. Drug-market proxy = AC's judgment call. "
            "kappa_share and shared_fraction_per_partner PENDING AC SIGN-OFF. "
            "Outbreak path and critical_network_threshold=0.35 are UNTOUCHED."
        ),
    }

    report_path = os.path.join(output_dir, "conditional_r0_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=lambda x: None if not isinstance(x, (int, float, str, bool, list, dict)) else x)
    print(f"\n  Saved: {report_path}")

    # Print headline summary
    print("\n" + "=" * 72)
    print("  HEADLINE SUMMARY — conditional R₀")
    print("=" * 72)
    for r in proxy_results:
        e = r.get("r0_results", {}).get("T_edge", {})
        d = r.get("r0_results", {}).get("decomposed", {})
        tc = r.get("tau_c", float('nan'))
        print(f"\n  {r['proxy']}  (τ_c={tc:.4f})")
        if e:
            print(f"    T_edge:     R₀={e.get('median',0):.3f} [{e.get('p5',0):.2f}–{e.get('p95',0):.2f}]  "
                  f"P(>1)={e.get('p_r0_gt1',0):.3f}  straddles={e.get('straddles_1')}")
        if d:
            print(f"    decomposed: R₀={d.get('median',0):.3f} [{d.get('p5',0):.2f}–{d.get('p95',0):.2f}]  "
                  f"P(>1)={d.get('p_r0_gt1',0):.3f}  straddles={d.get('straddles_1')}")

    return report


if __name__ == "__main__":
    run_conditional_r0()
