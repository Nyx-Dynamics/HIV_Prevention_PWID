"""
Handoff 6: Dyad-based R₀ analysis.

Runs the dyad pipeline, validates the network, and reports R₀ using the
heterogeneous per-edge T distribution. The key question: does node-level-
correlated, heavy-tailed T lift R₀ off the subcritical floor from Handoff 5?

ALL RESULTS CONDITIONAL AND PENDING AC REVIEW.
"""

from __future__ import annotations

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mobility.network_generator import run_generator_dyads
from mobility.threshold import (
    network_threshold,
    compute_per_edge_T_from_dyad_intensities,
    compute_r0_distribution,
    saturation_diagnostic,
    sweep_tau_c,
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


def run_dyad_analysis(
    n_agents: int = 300,
    n_steps: int = 80,
    venue_return_boost: float = 5.0,
    kappa_dyad: float = 2.0,
    seed: int = 42,
    output_dir: str = None,
) -> Dict:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    print("\n" + "=" * 72)
    print("  DYAD-BASED R₀ ANALYSIS (Handoff 6)")
    print("  node-level intensity + persistent dyads")
    print("=" * 72)

    # ── Saturation check ──────────────────────────────────────────────────────
    sat = saturation_diagnostic(MOBILITY_PARAMS)
    print(f"\n  Saturation: {sat['saturation_note']}")

    # ── Run proxies ───────────────────────────────────────────────────────────
    proxies = all_proxies()
    proxy_results = []

    for proxy_name, venues in proxies.items():
        print(f"\n{'─'*60}")
        print(f"  PROXY: {proxy_name}")
        print(f"{'─'*60}")

        G, stats, agents, node_intensities, edge_intensities = run_generator_dyads(
            n_agents=n_agents,
            n_steps=n_steps,
            venues=venues,
            venue_return_boost=venue_return_boost,
            kappa_dyad=kappa_dyad,
            seed=seed,
        )

        n_sharers = int(np.sum(node_intensities > 0))
        print(f"  Nodes: {stats.n_nodes}  Edges: {stats.n_edges}  "
              f"Sharers: {n_sharers}/{n_agents} ({n_sharers/n_agents*100:.0f}%)  "
              f"Edge-intensity pairs: {len(edge_intensities)}")

        # Validate
        checks = validate(stats)
        print_validation_report(stats, checks)

        # τ_c
        tau_c = network_threshold(stats.degree_sequence)
        print(f"\n  τ_c = {tau_c:.4f}")

        # Per-edge T from dyad intensities
        T_dyad, intensities = compute_per_edge_T_from_dyad_intensities(
            edge_intensities, params=MOBILITY_PARAMS, rng=rng
        )

        if len(T_dyad) == 0:
            print("  No edges with intensities — skipping R₀.")
            proxy_results.append({
                "proxy": proxy_name,
                "stats": stats,
                "tau_c": tau_c,
                "T_dyad_summary": None,
                "r0_summary": None,
            })
            continue

        T_summary = {
            "n_edges": len(T_dyad),
            "T_mean": float(np.mean(T_dyad)),
            "T_median": float(np.median(T_dyad)),
            "T_p5": float(np.percentile(T_dyad, 5)),
            "T_p95": float(np.percentile(T_dyad, 95)),
            "intensity_mean": float(np.mean(intensities)),
            "intensity_p95": float(np.percentile(intensities, 95)),
        }
        print(f"  T_dyad median={T_summary['T_median']:.4f}  "
              f"[{T_summary['T_p5']:.4f}–{T_summary['T_p95']:.4f}]")

        # R₀ = T / τ_c (use edge T distribution directly)
        if np.isfinite(tau_c) and tau_c > 0:
            tau_c_arr = np.full(len(T_dyad), tau_c)
            r0_samples, p_gt1, r0_sum = compute_r0_distribution(T_dyad, tau_c_arr)
            print(f"  R₀ median={r0_sum['median']:.3f}  "
                  f"[{r0_sum['p5']:.3f}–{r0_sum['p95']:.3f}]  "
                  f"P(R₀>1)={p_gt1:.3f}  straddles={r0_sum['straddles_1']}")
        else:
            r0_sum = {"error": "degenerate tau_c"}
            p_gt1 = None

        proxy_results.append({
            "proxy": proxy_name,
            "stats": stats,
            "tau_c": tau_c,
            "T_dyad_summary": T_summary,
            "r0_summary": r0_sum,
            "p_r0_gt1": p_gt1,
        })

    # ── Figure ────────────────────────────────────────────────────────────────
    _plot_dyad_summary(proxy_results, output_dir)

    # ── Frequency-data gap note ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FREQUENCY-DATA GAP STATUS")
    print("=" * 72)
    print(
        "  Mixing weight (27% any-sharing prevalence): sourced from NHBS 2018.\n"
        "  Within-sharer intensity: NHBS reports frequency categories\n"
        "  (every time / >half / <half) but exact % breakdown is not pre-loaded.\n"
        "  Current intensity: Gamma(shape=0.5, scale=0.05) — PLACEHOLDER.\n"
        "  Mean within-sharer intensity = 0.025 injections shared per partner.\n"
        "  PENDING AC SIGN-OFF on intensity scale. Note: 'every time' fraction\n"
        "  should set the upper tail; the NHBS tabulation (MMWR 67(1)) likely\n"
        "  has this breakdown in supplementary tables."
    )

    # ── Headline summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  HEADLINE SUMMARY — dyad R₀")
    print("=" * 72)
    for r in proxy_results:
        stats = r["stats"]
        er_floor = stats.mean_degree / stats.n_nodes if stats.n_nodes > 0 else 0
        rs = r.get("r0_summary") or {}
        print(f"\n  {r['proxy']}:")
        print(f"    ⟨k⟩={stats.mean_degree:.2f}  cc={stats.clustering_coefficient:.4f}  "
              f"ER_floor={er_floor:.5f}  "
              f"cc/ER={stats.clustering_coefficient/er_floor:.1f}×  "
              f"dispersion={stats.k2_moment/stats.mean_degree**2 if stats.mean_degree>0 else 0:.2f}")
        if rs and "median" in rs:
            print(f"    τ_c={r['tau_c']:.4f}  R₀={rs.get('median',0):.3f}  "
                  f"[{rs.get('p5',0):.2f}–{rs.get('p95',0):.2f}]  "
                  f"P(R₀>1)={r.get('p_r0_gt1',0):.3f}  straddles={rs.get('straddles_1')}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def _safe(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return x

    report = {
        "seed": seed,
        "kappa_dyad": kappa_dyad,
        "venue_return_boost": venue_return_boost,
        "saturation_diagnostic": sat,
        "proxies": [
            {
                "proxy": r["proxy"],
                "description": PROXY_DESCRIPTIONS.get(r["proxy"], ""),
                "network": {
                    "n_edges": r["stats"].n_edges,
                    "mean_degree": float(r["stats"].mean_degree),
                    "k2": float(r["stats"].k2_moment),
                    "dispersion": float(r["stats"].k2_moment / r["stats"].mean_degree**2)
                        if r["stats"].mean_degree > 0 else None,
                    "clustering": float(r["stats"].clustering_coefficient),
                    "er_floor": float(r["stats"].mean_degree / r["stats"].n_nodes),
                    "clustering_vs_er": float(r["stats"].clustering_coefficient /
                        (r["stats"].mean_degree / r["stats"].n_nodes))
                        if r["stats"].mean_degree > 0 else None,
                    "giant_component": float(r["stats"].giant_component_fraction),
                    "mean_path_length": float(r["stats"].mean_path_length)
                        if r["stats"].mean_path_length else None,
                },
                "tau_c": float(r["tau_c"]) if np.isfinite(r.get("tau_c", float("nan"))) else None,
                "T_dyad": r.get("T_dyad_summary"),
                "r0": r.get("r0_summary"),
                "p_r0_gt1": r.get("p_r0_gt1"),
            }
            for r in proxy_results
        ],
        "frequency_gap": (
            "Within-sharer intensity uses Gamma(0.5, 0.05) PLACEHOLDER. "
            "NHBS frequency breakdown (every time / >half / <half) not pre-loaded. "
            "PENDING AC SIGN-OFF on intensity scale."
        ),
        "guardrail": (
            "Outbreak path and critical_network_threshold=0.35 are untouched. "
            "No fabrication. All results conditional and pending AC review."
        ),
    }

    out_path = os.path.join(output_dir, "dyad_r0_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_safe)
    print(f"\n  Saved: {out_path}")

    return report


def _plot_dyad_summary(proxy_results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: clustering vs ER floor per proxy
    proxies = [r["proxy"] for r in proxy_results]
    ccs = [r["stats"].clustering_coefficient for r in proxy_results]
    er_floors = [r["stats"].mean_degree / r["stats"].n_nodes for r in proxy_results]
    x = range(len(proxies))

    axes[0].bar(x, ccs, color="steelblue", alpha=0.75, label="Clustering coefficient")
    axes[0].bar(x, er_floors, color="red", alpha=0.4, label="ER floor (⟨k⟩/n)")
    axes[0].axhline(0.1, color="orange", linestyle="--", linewidth=1.2,
                    label="Target lower bound (0.1)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([p.replace("_", "\n") for p in proxies], fontsize=8)
    axes[0].set_ylabel("Clustering")
    axes[0].set_title("Clustering vs ER floor per proxy\n(orange = empirical target floor)")
    axes[0].legend(fontsize=8)

    # Right: R₀ distribution per proxy
    colors = ["steelblue", "tomato", "seagreen"]
    for i, (r, color) in enumerate(zip(proxy_results, colors)):
        rs = r.get("r0_summary") or {}
        if "median" not in rs:
            continue
        axes[1].barh([i], [rs.get("p95", 0) - rs.get("p5", 0)],
                     left=rs.get("p5", 0), height=0.4,
                     color=color, alpha=0.65)
        axes[1].scatter([rs["median"]], [i], color=color, zorder=5)
        axes[1].text(rs.get("p95", 0) + 0.05, i,
                     f"P(>1)={r.get('p_r0_gt1',0):.2f}", fontsize=7.5, va="center")

    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(range(len(proxies)))
    axes[1].set_yticklabels([p.replace("_", "\n") for p in proxies], fontsize=8)
    axes[1].set_xlabel("R₀ = T_dyad / τ_c")
    axes[1].set_title("Conditional R₀ (dyad T) per proxy\n5th–95th pct bar, median dot")

    fig.suptitle(
        "Dyad-based network: clustering and R₀\nALL RESULTS PENDING AC REVIEW",
        fontsize=9
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "Fig_dyad_r0.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    run_dyad_analysis()
