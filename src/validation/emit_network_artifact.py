"""
Emit a fresh network_validation_report.json using the dyad generator
or (default) the new core-periphery generator (Handoff 8).

This script lives in src/validation/ and does NOT modify validate_network.py
or any other existing module.

Usage:
    python src/validation/emit_network_artifact.py --output outputs/
    python src/validation/emit_network_artifact.py --output outputs/ --no-core-periphery
    python src/validation/emit_network_artifact.py --output outputs/ --proxy ssp_concentrated_hotspot
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from mobility.network_generator import (
    run_generator_dyads,
    run_generator_core_periphery,
    NetworkStats,
)
from mobility.attractor_proxies import all_proxies, PROXY_DESCRIPTIONS


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def _stats_to_checks(stats: NetworkStats) -> dict:
    """Build checks dict in the network_validation_report.json schema."""
    mean_k = stats.mean_degree
    n = stats.n_nodes or 1
    er_floor = mean_k / n
    poisson_baseline = 1 + 1 / mean_k if mean_k > 0 else 2.0

    def _disp_status(disp, pb):
        if disp > pb * 1.5:
            return "PASS"
        elif disp > pb:
            return "REVIEW"
        return "FAIL"

    def _cc_status(cc, er):
        if 0.1 <= cc <= 0.4:
            return "PASS"
        elif cc > er * 3:
            return "ABOVE_FLOOR"
        return "REVIEW"

    def _mpl_status(mpl):
        if mpl is None:
            return "REVIEW"
        if 2.5 <= mpl <= 4.5:
            return "PASS"
        return "FAIL"

    def _gc_status(gc):
        return "PASS" if gc >= 0.45 else "FAIL"

    dispersion = stats.k2_moment / (mean_k ** 2) if mean_k > 0 else 0

    return {
        "mean_degree_ANCHOR": {
            "generated": round(mean_k, 4),
            "status": "ANCHOR",
            "note": "ANCHOR — kappa_dyad-set; not a held-out check.",
        },
        "giant_component": {
            "generated": round(stats.giant_component_fraction, 4),
            "target": "45-100%",
            "status": _gc_status(stats.giant_component_fraction),
        },
        "mean_path_length": {
            "generated": round(stats.mean_path_length, 4) if stats.mean_path_length else None,
            "target": "~3.1 [2.5, 4.5]",
            "status": _mpl_status(stats.mean_path_length),
        },
        "clustering_coefficient": {
            "generated": round(stats.clustering_coefficient, 5),
            "er_floor": round(er_floor, 5),
            "target": "0.1-0.4",
            "status": _cc_status(stats.clustering_coefficient, er_floor),
        },
        "dispersion_k2_over_k2": {
            "generated": round(dispersion, 4),
            "poisson_baseline": round(poisson_baseline, 4),
            "target": f">{poisson_baseline:.3f} x 1.5 = {poisson_baseline * 1.5:.3f}",
            "status": _disp_status(dispersion, poisson_baseline),
        },
    }


def emit_artifact(
    proxy_name: str = "ssp_concentrated_hotspot",
    output_dir: str = "outputs",
    n_agents: int = 300,
    n_steps: int = 80,
    venue_return_boost: float = 5.0,
    kappa_dyad: float = 2.0,
    seed: int = 42,
    use_core_periphery: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    sha = _git_sha()

    proxies = all_proxies()
    venues = proxies.get(proxy_name)
    if venues is None:
        raise ValueError(f"Unknown proxy: {proxy_name}. Choose from: {list(proxies)}")

    if use_core_periphery:
        print(f"Generating core-periphery network (proxy={proxy_name}, seed={seed})...")
        G, stats, _, _, _, cp_stats = run_generator_core_periphery(
            n_agents=n_agents,
            n_steps=n_steps,
            venues=venues,
            venue_return_boost=venue_return_boost,
            kappa_dyad=kappa_dyad,
            seed=seed,
        )
        generator_name = "run_generator_core_periphery"
    else:
        print(f"Generating dyad network (proxy={proxy_name}, seed={seed})...")
        _, stats, _, _, _ = run_generator_dyads(
            n_agents=n_agents,
            n_steps=n_steps,
            venues=venues,
            venue_return_boost=venue_return_boost,
            kappa_dyad=kappa_dyad,
            seed=seed,
        )
        cp_stats = None
        generator_name = "run_generator_dyads"

    mean_k = stats.mean_degree
    disp = stats.k2_moment / (mean_k ** 2) if mean_k > 0 else 0
    checks = _stats_to_checks(stats)
    all_held_out_pass = all(
        v.get("status") in ("PASS", "ANCHOR", "ABOVE_FLOOR")
        for v in checks.values()
    )

    artifact = {
        "git_sha": sha,
        "timestamp": datetime.now().isoformat(),
        "proxy": proxy_name,
        "proxy_description": PROXY_DESCRIPTIONS.get(proxy_name, ""),
        "generator": generator_name,
        "params": {
            "n_agents": n_agents,
            "n_steps": n_steps,
            "venue_return_boost": venue_return_boost,
            "kappa_dyad": kappa_dyad,
            "seed": seed,
            "use_core_periphery": use_core_periphery,
        },
        "generated": {
            "mean_degree": stats.mean_degree,
            "k2_moment": stats.k2_moment,
            "giant_component_fraction": stats.giant_component_fraction,
            "mean_path_length": stats.mean_path_length,
            "n_nodes": stats.n_nodes,
            "n_edges": stats.n_edges,
        },
        "checks": checks,
        "all_non_range_passed": all_held_out_pass,
    }

    # Embed core-periphery fields (Handoff 8) — only when using new pipeline
    if cp_stats is not None:
        artifact["core_periphery"] = {
            "two_core_size": cp_stats["two_core_size"],
            "two_core_fraction": cp_stats["two_core_fraction"],
            "two_core_density": cp_stats["two_core_density"],
            "component_sizes": cp_stats["component_sizes"][:10],
            "n_components": cp_stats["n_components"],
            "largest_component_fraction": cp_stats["largest_component_fraction"],
            "isolate_fraction": cp_stats["isolate_fraction"],
            "core_dispersion": cp_stats["core_dispersion"],
            "core_clustering": cp_stats["core_clustering"],
        }

    out_path = os.path.join(output_dir, "network_validation_report.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"  Wrote: {out_path}")
    print(f"  git_sha: {sha[:16]}")
    print(f"  <k>={stats.mean_degree:.3f}  gc={stats.giant_component_fraction:.3f}  "
          f"cc={stats.clustering_coefficient:.4f}  "
          f"disp={disp:.3f}  "
          f"mpl={stats.mean_path_length:.3f}" if stats.mean_path_length else "  mpl=None")
    if cp_stats is not None:
        print(f"  isolate_frac={cp_stats['isolate_fraction']:.4f}  "
              f"largest_comp_frac={cp_stats['largest_component_fraction']:.4f}  "
              f"two_core_frac={cp_stats['two_core_fraction']:.4f}  "
              f"core_disp={cp_stats['core_dispersion']:.4f}  "
              f"core_cc={cp_stats['core_clustering']:.4f}")

    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emit fresh network artifact for validation harness.")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--proxy", default="ssp_concentrated_hotspot",
                        choices=["ssp_only", "ssp_diffuse_market", "ssp_concentrated_hotspot"],
                        help="Attractor proxy to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-core-periphery", action="store_true",
                        help="Use legacy run_generator_dyads instead of core-periphery pipeline")
    args = parser.parse_args()
    emit_artifact(
        proxy_name=args.proxy,
        output_dir=args.output,
        seed=args.seed,
        use_core_periphery=not args.no_core_periphery,
    )
