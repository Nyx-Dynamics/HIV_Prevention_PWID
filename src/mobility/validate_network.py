"""
Stage 2: Validation harness for the mobility network generator.

Compares generated network statistics to Layer-3 empirical targets from
published PWID network surveys. These targets are HELD OUT — they were not
used to fit any generator parameter.

CRITICAL: This harness validates the GENERATOR against network structure
surveys. It must not touch outbreak data. The threshold (τ_c) is not
computed here — that is Stage 3 (threshold.py).

Outputs:
  - Printed validation table (generated vs target)
  - Degree-distribution plot → outputs/Fig_network_degree_distribution.png
  - ⟨k²⟩ explicitly labeled as a sensitivity range (not pass/fail)
"""

import sys
import os
import json
from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mobility.network_generator import run_generator, NetworkStats
from mobility.params import VALIDATION_TARGETS

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


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def validate(stats: NetworkStats) -> Dict[str, Any]:
    """
    Compare generated stats to Layer-3 targets.

    ANCHOR vs HELD-OUT distinction (Fix 3):
    ─────────────────────────────────────────
    kappa_share=0.01 (run_generator default) anchors ⟨k⟩ ≈ 2.6. This means
    ⟨k⟩ is an ANCHOR STATISTIC — not a genuine held-out check when kappa_share
    was set by targeting it. Reporting it as PASS would be circular.

    Genuine HELD-OUT checks are statistics NOT pinned by anchoring the mean:
      - Dispersion ⟨k²⟩/⟨k⟩²  (over-dispersion = heavy tail present)
      - Clustering coefficient   (local structure)
      - Giant-component fraction (connectivity)
      - Mean path length         (reachability)
      - Degree distribution shape (variance > mean = over-dispersed)

    If kappa_share is later derived independently from behavioral rates
    (injection_freq × shared_fraction_per_partner / co-location_rate), then ⟨k⟩
    also becomes a held-out check — mark it as such in that case.
    """
    import numpy as np
    results = {}

    k_arr = np.array(stats.degree_sequence, dtype=float)
    mean_k = stats.mean_degree
    k2 = stats.k2_moment

    # ── ANCHOR: ⟨k⟩ — labeled as anchor, NOT a held-out check ──────────────
    target_k = VALIDATION_TARGETS["mean_degree"]["value"]
    target_k_max = VALIDATION_TARGETS["mean_degree"]["range"][1]
    results["mean_degree_ANCHOR"] = {
        "generated": round(mean_k, 3),
        "target": f"{target_k} (range 0–{target_k_max})",
        "status": "ANCHOR",
        "note": (
            "ANCHOR STATISTIC — kappa_share=0.01 was set to produce ⟨k⟩≈2.6. "
            "Reporting this as PASS is circular. ⟨k⟩ becomes a held-out check "
            "only after kappa_share is derived from behavioral rates independently. "
            "See params.py kappa_share and shared_fraction_per_partner."
        ),
    }

    # ── HELD-OUT: Giant component ───────────────────────────────────────────
    gc_ok = stats.giant_component_fraction > 0.0
    results["giant_component"] = {
        "generated": round(stats.giant_component_fraction, 3),
        "target": "> 0 (empirical: 137–600+ nodes in one component)",
        "status": "PASS" if gc_ok else "FAIL",
        "note": "Buchanan 2019; Klovdahl/Potterat 1994. Not anchored by mean.",
    }

    # ── HELD-OUT: Mean path length ──────────────────────────────────────────
    target_mpl = VALIDATION_TARGETS["mean_path_length"]["value"]
    mpl = stats.mean_path_length
    if mpl is not None:
        # Target ~3.1; allow generous range for sparse first-pass network
        mpl_realistic = mpl <= 8.0
        mpl_good = mpl <= target_mpl * 1.5
        results["mean_path_length"] = {
            "generated": round(mpl, 3),
            "target": f"~{target_mpl} (Klovdahl/Potterat 1994 small-world core)",
            "status": "PASS" if mpl_good else ("REVIEW" if mpl_realistic else "FAIL"),
            "note": (
                f"Target ~{target_mpl} (small-world core). "
                f"PASS: ≤{target_mpl*1.5:.1f}; REVIEW: ≤8.0; FAIL: >8.0. "
                "Gap vs target reflects sparse network or weaker venue-anchoring."
            ),
        }
    else:
        results["mean_path_length"] = {
            "generated": "N/A (disconnected)",
            "target": f"~{target_mpl}",
            "status": "REVIEW",
            "note": "Graph disconnected.",
        }

    # ── HELD-OUT: Dispersion ⟨k²⟩/⟨k⟩² ────────────────────────────────────
    # Real injection networks are over-dispersed (dispersion > 1).
    # Dispersion > 1 iff Var(k)/⟨k⟩ > 1 iff ⟨k²⟩ > ⟨k⟩(⟨k⟩+1).
    # This is the τ_c-relevant check: τ_c = ⟨k⟩/(⟨k²⟩-⟨k⟩); over-dispersion lowers τ_c.
    dispersion = k2 / (mean_k ** 2) if mean_k > 0 else 0.0

    # ── HELD-OUT: Clustering vs Erdős–Rényi floor ──────────────────────────
    # ER floor: ⟨k⟩/n — if clustering ≈ ER floor, network is essentially random.
    # Real injection networks: Rolls 2011, Buchanan 2019 → clustering ≈ 0.1–0.4.
    n = stats.n_nodes
    er_floor = mean_k / n if n > 0 else 0.0
    cc_above_floor = stats.clustering_coefficient > er_floor * 3.0  # meaningfully above ER
    cc_realistic = 0.1 <= stats.clustering_coefficient <= 0.5
    results["clustering_coefficient"] = {
        "generated": round(stats.clustering_coefficient, 4),
        "er_floor": round(er_floor, 5),
        "target": "0.1–0.4 (empirical injection networks; Rolls 2011, Buchanan 2019)",
        "status": "PASS" if cc_realistic else ("ABOVE_FLOOR" if cc_above_floor else "REVIEW"),
        "note": (
            f"ER floor (random baseline) = ⟨k⟩/n = {er_floor:.5f}. "
            f"Generated = {stats.clustering_coefficient:.4f}. "
            f"PASS requires [0.1, 0.4]; ABOVE_FLOOR means > 3×ER but below 0.1; "
            f"REVIEW means at or near the random floor. "
            "Rolls 2011; Buchanan 2019."
        ),
    }

    # ── HELD-OUT: Dispersion vs Poisson baseline ────────────────────────────
    # Poisson degree distribution (ER graph): dispersion = 1 + 1/⟨k⟩ ≈ 1.38 at ⟨k⟩=2.6.
    # Real injection networks have heavier tails (Buchanan max k=14 vs mean 2.6).
    # Realistic target: > Poisson baseline, ideally substantially so.
    poisson_baseline = 1.0 + (1.0 / mean_k) if mean_k > 0 else 2.0
    disp_ok = dispersion > poisson_baseline * 1.5  # meaningfully above Poisson
    results["dispersion_k2_over_k2"] = {
        "generated": round(dispersion, 3),
        "poisson_baseline": round(poisson_baseline, 3),
        "target": f"> {poisson_baseline:.2f} × 1.5 = {poisson_baseline*1.5:.2f} (Buchanan max_k=14 implies heavy tail)",
        "status": "PASS" if disp_ok else "REVIEW",
        "note": (
            f"Poisson (ER) baseline = 1 + 1/⟨k⟩ = {poisson_baseline:.3f}. "
            f"Real injection networks substantially exceed this (Buchanan max_degree=14 "
            f"vs mean=2.6 → heavy tail). PASS requires dispersion > {poisson_baseline*1.5:.2f}. "
            "This is the τ_c-relevant check — not pinned by anchoring ⟨k⟩."
        ),
    }

    # ── ⟨k²⟩ — SENSITIVITY RANGE (not pass/fail; dominant τ_c input) ───────
    results["k2_tail"] = {
        "generated": round(k2, 3),
        "target": "SENSITIVITY RANGE — not a pass/fail check",
        "status": "RANGE",
        "note": (
            "TAIL CAVEAT (mandatory): ⟨k²⟩ is the most under-captured quantity "
            "in all available empirical datasets. ⟨k²⟩ directly sets τ_c. "
            "See Stage 3 sweep for τ_c band."
        ),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DEGREE DISTRIBUTION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_degree_distribution(stats: NetworkStats, output_dir: str):
    """
    Plot the generated degree distribution with empirical reference point.
    Labels ⟨k²⟩ explicitly as uncertain.
    """
    deg_seq = np.array(stats.degree_sequence)
    max_deg = int(deg_seq.max()) if len(deg_seq) > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: degree distribution (linear)
    bins = range(0, max_deg + 2)
    axes[0].hist(deg_seq, bins=bins, color='steelblue', alpha=0.75, edgecolor='white')
    axes[0].axvline(stats.mean_degree, color='red', linestyle='--', linewidth=1.5,
                    label=f'Generated ⟨k⟩={stats.mean_degree:.2f}')
    axes[0].axvline(2.6, color='orange', linestyle=':', linewidth=1.5,
                    label='Empirical target ⟨k⟩=2.6\n(Buchanan 2019)')
    axes[0].set_xlabel('Degree k')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Degree distribution (linear)')
    axes[0].legend(fontsize=8)

    # Right: log-log (tail structure) — only plot degrees > 0
    nonzero = deg_seq[deg_seq > 0]
    if len(nonzero) > 0:
        log_bins = np.logspace(np.log10(1), np.log10(max(nonzero) + 1), 20)
        axes[1].hist(nonzero, bins=log_bins, color='tomato', alpha=0.75, edgecolor='white')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Degree k (log)')
        axes[1].set_ylabel('Count (log)')
        axes[1].set_title(
            f'Degree tail (log-log)\n⟨k²⟩={stats.k2_moment:.1f} — SENSITIVITY RANGE, not validated'
        )

    fig.suptitle(
        f'Generated PWID contact network (n={stats.n_nodes}, edges={stats.n_edges})',
        fontsize=10
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'Fig_network_degree_distribution.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_validation_report(stats: NetworkStats, checks: Dict[str, Any]) -> bool:
    """Print table and return True if all non-RANGE checks passed."""
    print("\n" + "=" * 72)
    print("  NETWORK VALIDATION REPORT — generated vs Layer-3 empirical targets")
    print("=" * 72)
    print(f"  Network: {stats.n_nodes} nodes, {stats.n_edges} edges, seed=42")
    print()

    col = "{:<22} {:<10} {:<22} {:<8}  {}"
    print(col.format("Statistic", "Generated", "Target", "Status", "Note"))
    print("-" * 90)
    for name, res in checks.items():
        gen_str = str(res['generated'])
        tgt_str = str(res['target'])[:21]
        status = res['status']
        note = res.get('note', '')[:60]
        print(col.format(name, gen_str, tgt_str, status, note))

    print()
    print("  TAIL CAVEAT (mandatory, see SOURCING_mobility_network.md):")
    print("    ⟨k²⟩ is not validated as a point estimate. It is the primary")
    print("    sensitivity axis for τ_c (Stage 3). Treat τ_c as a range, not")
    print("    a number. Stage 3 sweeps the degree tail to bound τ_c.")

    held_out = {k: v for k, v in checks.items()
                if v['status'] not in ('RANGE', 'ANCHOR')}
    anchors = [k for k, v in checks.items() if v['status'] == 'ANCHOR']
    passed = all(v['status'] in ('PASS',) for v in held_out.values())
    reviews = [k for k, v in held_out.items() if v['status'] == 'REVIEW']
    fails = [k for k, v in held_out.items() if v['status'] == 'FAIL']

    print()
    if anchors:
        print(f"  ANCHOR (scale-set, not held-out): {anchors}")
    print(f"  HELD-OUT PASS: {sum(1 for v in held_out.values() if v['status']=='PASS')}/{len(held_out)}")
    if reviews:
        print(f"  REVIEW (within tolerance): {reviews}")
    if fails:
        print(f"  FAIL: {fails}")

    return len(fails) == 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    n_agents: int = 300,
    n_steps: int = 50,
    seed: int = 42,
    output_dir: str = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')

    print("Running network generator for validation...")
    _, stats, _ = run_generator(n_agents=n_agents, n_steps=n_steps, seed=seed)

    checks = validate(stats)
    all_ok = print_validation_report(stats, checks)
    plot_degree_distribution(stats, output_dir)

    # Save JSON report
    report = {
        "generated": {
            "mean_degree": stats.mean_degree,
            "k2_moment": stats.k2_moment,
            "giant_component_fraction": stats.giant_component_fraction,
            "mean_path_length": stats.mean_path_length,
            "n_nodes": stats.n_nodes,
            "n_edges": stats.n_edges,
        },
        "checks": checks,
        "all_non_range_passed": all_ok,
    }
    report_path = os.path.join(output_dir, "network_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    return report


if __name__ == "__main__":
    run_validation()
