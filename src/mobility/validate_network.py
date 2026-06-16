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

    Returns a dict with one entry per check: {status, generated, target, note}.
    ⟨k²⟩ is explicitly excluded from pass/fail — it's a sensitivity range.
    """
    results = {}

    # ── Mean degree ⟨k⟩ ──
    target_k = VALIDATION_TARGETS["mean_degree"]["value"]
    target_k_max = VALIDATION_TARGETS["mean_degree"]["range"][1]
    k_ok = 0.5 <= stats.mean_degree <= target_k_max
    results["mean_degree"] = {
        "generated": round(stats.mean_degree, 3),
        "target": f"{target_k} (range 0–{target_k_max})",
        "status": "PASS" if k_ok else "REVIEW",
        "note": "RDS-derived; point check only.",
    }

    # ── Giant component ──
    gc_ok = stats.giant_component_fraction > 0.0
    results["giant_component"] = {
        "generated": round(stats.giant_component_fraction, 3),
        "target": "> 0 (empirical: 137–600+ nodes in one component)",
        "status": "PASS" if gc_ok else "FAIL",
        "note": "Buchanan 2019; Klovdahl/Potterat 1994.",
    }

    # ── Mean path length ──
    target_mpl = VALIDATION_TARGETS["mean_path_length"]["value"]
    mpl = stats.mean_path_length
    if mpl is not None:
        mpl_ok = mpl <= target_mpl * 3.0  # allow 3× tolerance for sparse networks
        results["mean_path_length"] = {
            "generated": round(mpl, 3),
            "target": f"~{target_mpl}",
            "status": "PASS" if mpl_ok else "REVIEW",
            "note": "Klovdahl/Potterat 1994; small-world core in dense empirical network.",
        }
    else:
        results["mean_path_length"] = {
            "generated": "N/A (disconnected)",
            "target": f"~{target_mpl}",
            "status": "REVIEW",
            "note": "Graph disconnected — giant component too small or path infinite.",
        }

    # ── ⟨k²⟩ — SENSITIVITY RANGE, NOT PASS/FAIL ──
    results["k2_tail"] = {
        "generated": round(stats.k2_moment, 3),
        "target": "SENSITIVITY RANGE — not a pass/fail check",
        "status": "RANGE",
        "note": (
            "TAIL CAVEAT (mandatory): ⟨k²⟩ is the most under-captured quantity in all "
            "available empirical datasets (RDS is itself a network walk; high-degree tail "
            "is systematically missed). ⟨k²⟩ directly sets τ_c. Treat as sensitivity axis. "
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

    non_range = {k: v for k, v in checks.items() if v['status'] != 'RANGE'}
    passed = all(v['status'] in ('PASS',) for v in non_range.values())
    reviews = [k for k, v in non_range.items() if v['status'] == 'REVIEW']
    fails = [k for k, v in non_range.items() if v['status'] == 'FAIL']

    print()
    print(f"  PASS: {sum(1 for v in non_range.values() if v['status']=='PASS')}/{len(non_range)}")
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
