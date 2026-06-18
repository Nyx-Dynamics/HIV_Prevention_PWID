"""
Geometry sweep: single-hotspot vs multi-venue proxy, 20 seeds each.

Frozen knobs (H8-fix calibration): contact_cap=2, min_colocs=4.
Reports mean ± 95% CI for 7 metrics per proxy.

Pre-registered verdicts (decide before reading numbers):
  isolate_fraction ↓ toward <0.10 AND <k>/dispersion hold
      → geometry artifact: single hotspot stranded low-r_g agents
  isolate_fraction stays ~0.20
      → min_colocs=4 is the isolate source (not geometry)
  <k> or dispersion move substantially under multi-venue
      → structure is geometry-sensitive; report as finding, do not re-tune

Usage:
    python src/validation/sweep_geometry.py [--output outputs/]
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from mobility.network_generator import run_generator_core_periphery
from mobility.attractor_proxies import all_proxies, PROXY_DESCRIPTIONS


PROXY_NAMES = ["ssp_concentrated_hotspot", "ssp_diffuse_market"]
SEEDS = list(range(42, 62))  # 20 seeds: 42-61
METRICS = [
    "isolate_fraction",
    "mean_degree",
    "dispersion",
    "two_core_fraction",
    "largest_component_fraction",
    "clustering",
    "mean_path_length",
]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_one(proxy_name: str, seed: int, proxies: dict) -> dict:
    venues = proxies[proxy_name]
    G, stats, _, _, _, cp = run_generator_core_periphery(
        venues=venues,
        seed=seed,
    )
    mean_k = stats.mean_degree
    disp = stats.k2_moment / (mean_k ** 2) if mean_k > 0 else 0.0
    return {
        "isolate_fraction": cp["isolate_fraction"],
        "mean_degree": mean_k,
        "dispersion": disp,
        "two_core_fraction": cp["two_core_fraction"],
        "largest_component_fraction": cp["largest_component_fraction"],
        "clustering": stats.clustering_coefficient,
        "mean_path_length": stats.mean_path_length if stats.mean_path_length else float("nan"),
    }


def ci95(values: list) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)])
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    t = scipy_stats.t.ppf(0.975, df=n - 1)
    return {
        "mean": round(mean, 5),
        "ci95_lo": round(mean - t * se, 5),
        "ci95_hi": round(mean + t * se, 5),
        "std": round(float(np.std(arr, ddof=1)), 5),
        "n": n,
    }


def run_sweep(output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    sha = _git_sha()
    proxies = all_proxies()

    raw: dict = {p: {m: [] for m in METRICS} for p in PROXY_NAMES}

    for proxy_name in PROXY_NAMES:
        print(f"\nProxy: {proxy_name}  ({len(SEEDS)} seeds)")
        for seed in SEEDS:
            row = run_one(proxy_name, seed, proxies)
            for m in METRICS:
                raw[proxy_name][m].append(row[m])
            print(f"  seed={seed}  <k>={row['mean_degree']:.2f}  "
                  f"iso={row['isolate_fraction']:.3f}  "
                  f"gc={row['largest_component_fraction']:.3f}  "
                  f"cc={row['clustering']:.3f}  "
                  f"mpl={row['mean_path_length']:.2f}")

    summary: dict = {}
    for proxy_name in PROXY_NAMES:
        summary[proxy_name] = {
            m: ci95(raw[proxy_name][m]) for m in METRICS
        }

    # Pre-registered verdict
    iso_hotspot = summary["ssp_concentrated_hotspot"]["isolate_fraction"]["mean"]
    iso_multi = summary["ssp_diffuse_market"]["isolate_fraction"]["mean"]
    k_hotspot = summary["ssp_concentrated_hotspot"]["mean_degree"]["mean"]
    k_multi = summary["ssp_diffuse_market"]["mean_degree"]["mean"]
    disp_hotspot = summary["ssp_concentrated_hotspot"]["dispersion"]["mean"]
    disp_multi = summary["ssp_diffuse_market"]["dispersion"]["mean"]

    iso_drop = iso_hotspot - iso_multi
    k_shift_frac = abs(k_multi - k_hotspot) / k_hotspot if k_hotspot > 0 else 0
    disp_shift_frac = abs(disp_multi - disp_hotspot) / disp_hotspot if disp_hotspot > 0 else 0

    if iso_drop > 0.08 and k_shift_frac < 0.25 and disp_shift_frac < 0.25:
        verdict = "GEOMETRY_ARTIFACT"
        verdict_note = (
            "isolate_fraction fell >0.08 under multi-venue AND <k>/dispersion held "
            "(<25% relative shift). Single hotspot stranded low-r_g agents; "
            "multiple venues give them co-location opportunities."
        )
    elif abs(iso_drop) < 0.04:
        verdict = "MIN_COLOCS_ISOLATE_SOURCE"
        verdict_note = (
            f"isolate_fraction stayed ~{iso_hotspot:.3f} vs {iso_multi:.3f} "
            "(|delta|<0.04). Geometry is not the source. "
            "min_colocs=4 is the isolate driver — follow-up: test min_colocs=3 "
            "with compensating contact_cap."
        )
    else:
        verdict = "GEOMETRY_SENSITIVE"
        verdict_note = (
            f"<k> shifted {k_shift_frac*100:.1f}% or dispersion shifted "
            f"{disp_shift_frac*100:.1f}% across proxies. "
            "Structure is geometry-sensitive; no single proxy is sufficient. "
            "Sensitivity analysis across proxies is itself a finding."
        )

    artifact = {
        "git_sha": sha,
        "timestamp": datetime.now().isoformat(),
        "frozen_knobs": {"contact_cap": 2, "min_colocs": 4},
        "proxies": {
            p: {
                "description": PROXY_DESCRIPTIONS[p],
                "n_seeds": len(SEEDS),
                "seeds": SEEDS,
            }
            for p in PROXY_NAMES
        },
        "summary": summary,
        "raw": {p: {m: [round(v, 6) for v in raw[p][m]] for m in METRICS} for p in PROXY_NAMES},
        "verdict": verdict,
        "verdict_note": verdict_note,
        "verdict_inputs": {
            "iso_hotspot_mean": round(iso_hotspot, 4),
            "iso_multi_mean": round(iso_multi, 4),
            "iso_drop": round(iso_drop, 4),
            "k_hotspot_mean": round(k_hotspot, 4),
            "k_multi_mean": round(k_multi, 4),
            "k_shift_frac": round(k_shift_frac, 4),
            "disp_hotspot_mean": round(disp_hotspot, 4),
            "disp_multi_mean": round(disp_multi, 4),
            "disp_shift_frac": round(disp_shift_frac, 4),
        },
    }

    out_path = os.path.join(output_dir, "geometry_sweep.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote: {out_path}")
    print(f"Verdict: {verdict}")
    print(f"  {verdict_note}")

    # Print comparison table
    print("\n--- COMPARISON TABLE (mean [95% CI]) ---")
    print(f"{'Metric':<30}  {'ssp_concentrated_hotspot':>32}  {'ssp_diffuse_market':>32}")
    for m in METRICS:
        h = summary["ssp_concentrated_hotspot"][m]
        d = summary["ssp_diffuse_market"][m]
        hstr = f"{h['mean']:.4f} [{h['ci95_lo']:.4f}, {h['ci95_hi']:.4f}]"
        dstr = f"{d['mean']:.4f} [{d['ci95_lo']:.4f}, {d['ci95_hi']:.4f}]"
        print(f"  {m:<28}  {hstr:>32}  {dstr:>32}")

    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    run_sweep(args.output)
