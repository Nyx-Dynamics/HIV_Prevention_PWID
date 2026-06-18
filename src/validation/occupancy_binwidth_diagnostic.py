"""
Occupancy / bin-width diagnostic (Handoff 12).

Question: is max venue-bin occupancy 145/300 (48%) a physical co-presence
density or a time-aggregation artifact of the current 5-day bin width?

Method: re-bin the same trajectory data at five bin widths
{1, 2, 5, 10, 20} steps/bin and compute max venue-bin occupancy.
Two sub-questions:
  (A) Does max occupancy scale linearly with bin width (artifact)?
      → fit log(max_occ) vs log(bin_width) via scipy.stats.linregress.
      slope ≈ 1 → linear/artifact  |  slope ≈ 0 → plateau/genuine
  (B) Is the hotspot/diffuse occupancy ratio bin-width-invariant (verdict)?
      → flat = Outcome 2 (static-blind) stands; shrinks = verdict wobbles.

ABSOLUTE GUARDRAIL: NO dynamics. NO transmission. NO β. NO γ.
Trajectories are RE-DERIVED (EPR walk only, identical seeds ≡ stored data).
No invocation of run_temporal_with_gamma or run_static.

Pre-registered decision rule (§4, Handoff 12 — committed before data):
  (A) linear, (B) shrinks → ARTIFACT, VERDICT WOBBLES
  (A) linear, (B) flat   → ARTIFACT, VERDICT ROBUST
  (A) plateau, (B) flat  → GENUINE DENSITY, VERDICT ROBUST
  (A) plateau, (B) shrinks → CONTRADICTORY — debug
  (A) intermediate, any  → PARTIAL ARTIFACT — report exponent; verdict per (B)

House-style constraints (§5, Handoff 11):
  No sklearn/statsmodels/lifelines/scipy.integrate.
  Binning: numpy.  Scaling fit: scipy.stats.linregress (log-log).
  CIs: numpy bootstrap (seed=42, 1000 iterations).
  Sign convention for ratios: hotspot / diffuse.

Acceptance gates:
  1. Toy case validated at ≥2 bin widths (units/counts correct).
  2. No run_temporal_with_gamma or run_static invocations (grep check).
  3. No prohibited imports.
  4. Re-binning floor stated; resolution limit reported if applicable.
  5. Decision rule applied from pre-registered table, verbatim.

Usage:
    python src/validation/occupancy_binwidth_diagnostic.py [--output outputs/]
"""

from __future__ import annotations

import argparse
import json
import os
import re as _re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.stats import linregress

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from mobility.network_generator import sample_traversement_potential, generate_walks
from mobility.attractor_proxies import all_proxies

# ── Constants ─────────────────────────────────────────────────────────────────

NATIVE_STEP_DAYS = 1         # 1 EPR step = 1 day (Trajectory.visits stores step index)
N_STEPS          = 80        # n_steps for trajectory (same as H9-H11)
SPACE_BIN        = 0.5       # spatial bin width (km) — fixed; not varied here
W0               = 5         # current baseline time_bin (w₀)
BIN_WIDTHS       = [1, 2, 5, 10, 20]   # integer time_bin values; covers {0.2×,0.4×,1×,2×,4×} w₀
PROXIES          = ["ssp_concentrated_hotspot", "ssp_diffuse_market"]
BOOTSTRAP_SEED   = 42
BOOTSTRAP_N      = 1000

# Slope thresholds for (A) classification
SLOPE_LINEAR_THRESHOLD = 0.75    # slope ≥ this → linear (artifact)
SLOPE_PLATEAU_THRESHOLD = 0.25   # slope ≤ this → plateau (genuine)
# Ratio-invariance threshold for (B) — DIRECTIONAL
# SHRINKING: ratio at finest bin < ratio at w₀ by >threshold → geometry weakens
# GROWING:   ratio at finest bin > ratio at w₀ by >threshold → geometry strengthens
# FLAT:      within ±threshold
RATIO_DRIFT_THRESHOLD = 0.10


# ── Toy validation (T1) ───────────────────────────────────────────────────────

def _toy_validation() -> dict:
    """
    Minimal hand-checkable case: 4 agents at (0,0), each visiting at a
    different step within a 4-step trajectory.

    time_bin=1:  each bin has 1 agent → max_occ=1 (no events)
    time_bin=2:  bin 0 (steps 0-1) has agents {0,1} → max_occ=2
                 bin 1 (steps 2-3) has agents {2,3} → max_occ=2
    time_bin=4:  bin 0 (steps 0-3) has agents {0,1,2,3} → max_occ=4

    Shows occupancy scales linearly with bin width in this pure-spread case.
    """
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class _Traj:
        agent_id: int
        visits: list

    trajs = [_Traj(agent_id=i, visits=[(0.0, 0.0, i)]) for i in range(4)]

    results = {}
    for tb in [1, 2, 4]:
        bin_map: dict = {}
        for traj in trajs:
            for x, y, step in traj.visits:
                sx = int(np.floor(x / SPACE_BIN))
                sy = int(np.floor(y / SPACE_BIN))
                key = (step // tb, sx, sy)
                bin_map.setdefault(key, []).append(traj.agent_id)
        occupancies = [len(set(v)) for v in bin_map.values() if len(set(v)) >= 2]
        results[tb] = {"max_occ": max(occupancies) if occupancies else 0}

    # Validate
    assert results[1]["max_occ"] == 0, f"time_bin=1 toy: expected 0 events, got {results[1]}"
    assert results[2]["max_occ"] == 2, f"time_bin=2 toy: expected max_occ=2, got {results[2]}"
    assert results[4]["max_occ"] == 4, f"time_bin=4 toy: expected max_occ=4, got {results[4]}"

    # Second toy: all 4 agents at (0,0) at step 0 (physical co-presence)
    trajs2 = [_Traj(agent_id=i, visits=[(0.0, 0.0, 0)]) for i in range(4)]
    for tb in [1, 2, 4]:
        bin_map2: dict = {}
        for traj in trajs2:
            for x, y, step in traj.visits:
                sx = int(np.floor(x / SPACE_BIN))
                sy = int(np.floor(y / SPACE_BIN))
                key = (step // tb, sx, sy)
                bin_map2.setdefault(key, []).append(traj.agent_id)
        occ2 = [len(set(v)) for v in bin_map2.values() if len(set(v)) >= 2]
        max_occ2 = max(occ2) if occ2 else 0
        # All 4 in same step → same bin regardless of bin_width → max_occ=4 at all widths
        assert max_occ2 == 4, f"time_bin={tb} plateau toy: expected max_occ=4, got {max_occ2}"

    return {
        "gate_passed": True,
        "spread_case": results,
        "plateau_case": {"max_occ": 4, "all_bin_widths": True},
        "note": (
            "Spread toy: 4 agents at distinct steps → max_occ scales linearly with bin_width. "
            "Plateau toy: 4 agents at same step → max_occ=4 at all bin_widths. ✓"
        ),
    }


# ── Derive occupancy from trajectories ───────────────────────────────────────

def _derive_occupancy(trajectories, time_bin: int, space_bin: float = SPACE_BIN) -> dict:
    """
    Compute per-bin occupancy from raw trajectories at a given time_bin.
    Returns: {max_occ, mean_occ, p90_occ, n_events, histogram}
    """
    bin_map: dict = {}
    for traj in trajectories:
        for x, y, step in traj.visits:
            sx = int(np.floor(x / space_bin))
            sy = int(np.floor(y / space_bin))
            tb = step // time_bin
            key = (tb, sx, sy)
            bin_map.setdefault(key, []).append(traj.agent_id)

    occupancies = [len(set(v)) for v in bin_map.values() if len(set(v)) >= 2]
    if not occupancies:
        return {"max_occ": 0, "mean_occ": 0.0, "p90_occ": 0.0,
                "n_events": 0, "all_occupancies": []}

    arr = np.array(occupancies)
    return {
        "max_occ":         int(np.max(arr)),
        "mean_occ":        float(np.mean(arr)),
        "p90_occ":         float(np.percentile(arr, 90)),
        "n_events":        len(arr),
        "all_occupancies": arr.tolist(),
    }


def one_seed_sweep(proxy_name: str, seed: int, proxies_dict: dict) -> dict:
    """
    Re-derive trajectories (EPR walk, no transmission) and compute occupancy
    at all 5 bin widths.
    """
    rng = np.random.default_rng(seed)
    venues = proxies_dict[proxy_name]
    agents = sample_traversement_potential(
        n=300, rg_scale_km=2.4, rg_growth_exponent=1.65,
        seed_hiv_prevalence=0.07, rng=rng,
    )
    trajectories = generate_walks(
        agents=agents, venues=venues, n_steps=N_STEPS,
        epr_rho=0.60, epr_gamma=0.21, jump_length_exponent=0.60,
        rng=rng, venue_return_boost=5.0,
    )
    return {
        tb: _derive_occupancy(trajectories, time_bin=tb)
        for tb in BIN_WIDTHS
    }


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def _ci95(values: list) -> dict:
    arr = np.array([v for v in values if v is not None])
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci95_lo": None, "ci95_hi": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = rng.choice(arr, size=(BOOTSTRAP_N, n), replace=True).mean(axis=1)
    return {
        "mean":    round(float(np.mean(arr)), 2),
        "ci95_lo": round(float(np.percentile(boots, 2.5)), 2),
        "ci95_hi": round(float(np.percentile(boots, 97.5)), 2),
        "std":     round(float(np.std(arr, ddof=1)), 2) if n > 1 else 0.0,
        "n":       n,
    }


# ── Main sweep + analysis ─────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_diagnostic(output_dir: str = "outputs", n_seeds: int = 20, seeds: list = None):
    if seeds is None:
        seeds = list(range(42, 42 + n_seeds))

    os.makedirs(output_dir, exist_ok=True)
    sha = _git_sha()
    proxies_dict = all_proxies()

    print("=" * 70)
    print("HANDOFF 12 — OCCUPANCY / BIN-WIDTH DIAGNOSTIC")
    print(f"  git_sha: {sha[:16]}  n_seeds={n_seeds}  seeds {seeds[0]}–{seeds[-1]}")
    print(f"  bin_widths={BIN_WIDTHS} (steps)  w₀={W0}  native_step={NATIVE_STEP_DAYS}d")
    print(f"  space_bin={SPACE_BIN}km  n_steps={N_STEPS}")
    print("=" * 70)

    # ── T0: Resolution statement ───────────────────────────────────────────
    print(f"\n[T0] Resolution statement")
    print(f"  Stored artifacts: summary JSONs only (outputs/*.json). No raw trajectories on disk.")
    print(f"  Native resolution: 1 step = {NATIVE_STEP_DAYS} day (Trajectory.visits stores step index).")
    print(f"  Re-binning floor: time_bin = 1 (finest integer bin width).")
    print(f"  Re-derivation: trajectories regenerated from EPR walk, same seeds (deterministic = stored).")
    print(f"  No transmission invoked. Coverage: full grid {{1,2,5,10,20}} — no resolution limit.")

    # ── T1: Toy validation ─────────────────────────────────────────────────
    print(f"\n[T1] Toy validation...")
    toy_result = _toy_validation()
    print(f"  {toy_result['note']}")

    # ── Acceptance gate: no prohibited imports or dynamics invocations ────────
    this_file = os.path.join(_ROOT, "src/validation/occupancy_binwidth_diagnostic.py")
    prohibited_patterns = [
        r"^\s*(import|from)\s+sklearn",
        r"^\s*(import|from)\s+statsmodels",
        r"^\s*(import|from)\s+lifelines",
        r"^\s*(import|from)\s+scipy\.integrate",
    ]
    # Transmission/dynamics calls that must not appear (as actual code, not strings)
    prohibited_call_patterns = [
        r"\brun_temporal_with_gamma\s*\(",
        r"\brun_static\s*\(",
    ]
    for line in open(this_file):
        for pat in prohibited_patterns:
            if _re.match(pat, line):
                raise AssertionError(f"Prohibited import: {line.strip()}")
        # Only flag actual calls (word + open-paren), not string literals
        stripped = line.split("#")[0]  # remove comments
        stripped = _re.sub(r'"[^"]*"', '', stripped)  # remove string literals
        stripped = _re.sub(r"'[^']*'", '', stripped)
        for pat in prohibited_call_patterns:
            if _re.search(pat, stripped):
                raise AssertionError(f"Prohibited call in: {line.strip()}")
    print("[Acceptance gates] No prohibited imports or dynamics invocations. ✓")

    # ── T2: Sweep ──────────────────────────────────────────────────────────
    print(f"\n[T2] Bin-width sweep ({n_seeds} seeds × 2 geometries × {len(BIN_WIDTHS)} bin widths)...")
    all_data: dict = {proxy: {tb: [] for tb in BIN_WIDTHS} for proxy in PROXIES}

    for proxy in PROXIES:
        print(f"  {proxy}")
        for seed in seeds:
            per_tb = one_seed_sweep(proxy, seed, proxies_dict)
            for tb in BIN_WIDTHS:
                all_data[proxy][tb].append(per_tb[tb]["max_occ"])
            print(f"    seed={seed}  " + "  ".join(
                f"tb={tb}:max={per_tb[tb]['max_occ']}" for tb in BIN_WIDTHS))

    # Aggregate
    agg = {
        proxy: {
            tb: _ci95(all_data[proxy][tb])
            for tb in BIN_WIDTHS
        }
        for proxy in PROXIES
    }

    # ── T3: Hotspot/diffuse ratio per bin width ────────────────────────────
    ratios = {}
    for tb in BIN_WIDTHS:
        h_vals = np.array(all_data["ssp_concentrated_hotspot"][tb], dtype=float)
        d_vals = np.array(all_data["ssp_diffuse_market"][tb], dtype=float)
        valid = (d_vals > 0)
        if valid.sum() > 0:
            ratio_per_seed = h_vals[valid] / d_vals[valid]
            ratios[tb] = _ci95(ratio_per_seed.tolist())
        else:
            ratios[tb] = {"mean": None}

    # ── T4: Log-log linregress (A test) ────────────────────────────────────
    h_max_means = np.array([agg["ssp_concentrated_hotspot"][tb]["mean"] for tb in BIN_WIDTHS], dtype=float)
    d_max_means = np.array([agg["ssp_diffuse_market"][tb]["mean"] for tb in BIN_WIDTHS], dtype=float)
    log_bw  = np.log(BIN_WIDTHS)

    loglog_h = linregress(log_bw, np.log(np.maximum(h_max_means, 1e-9)))
    loglog_d = linregress(log_bw, np.log(np.maximum(d_max_means, 1e-9)))

    slope_h = round(loglog_h.slope, 4)
    slope_d = round(loglog_d.slope, 4)
    slope_mean = round((slope_h + slope_d) / 2, 4)

    # (B) ratio drift test — DIRECTIONAL
    ratio_w0  = ratios[W0]["mean"] or 1.0
    ratio_w1  = ratios[BIN_WIDTHS[0]]["mean"] or 1.0   # finest bin (tb=1)
    signed_drift = (ratio_w1 - ratio_w0) / ratio_w0 if ratio_w0 > 0 else 0.0
    ratio_drift = abs(signed_drift)

    # ── T5: Pre-registered decision rule ──────────────────────────────────
    if slope_mean >= SLOPE_LINEAR_THRESHOLD:
        axis_a = "LINEAR_ARTIFACT"
    elif slope_mean <= SLOPE_PLATEAU_THRESHOLD:
        axis_a = "PLATEAU_GENUINE"
    else:
        axis_a = f"INTERMEDIATE_slope={slope_mean:.2f}"

    if ratio_drift <= RATIO_DRIFT_THRESHOLD:
        axis_b = "FLAT"
    elif signed_drift < 0:
        axis_b = "SHRINKING"   # ratio decreases at finer bins → geometry weakens
    else:
        axis_b = "GROWING"     # ratio increases at finer bins → geometry strengthens

    decision = _decision_rule(axis_a, axis_b, slope_mean)

    # ── Print tables ───────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  MAX OCCUPANCY vs BIN WIDTH (mean [95% CI], n=300 agents)")
    print(f"{'═'*70}")
    print(f"  {'bin_width':>10}  {'hotspot max_occ':>24}  {'diffuse max_occ':>24}  {'ratio':>10}")
    for tb in BIN_WIDTHS:
        h = agg["ssp_concentrated_hotspot"][tb]
        d = agg["ssp_diffuse_market"][tb]
        r = ratios[tb]
        hstr = f"{h['mean']:.0f}[{h['ci95_lo']:.0f},{h['ci95_hi']:.0f}]"
        dstr = f"{d['mean']:.0f}[{d['ci95_lo']:.0f},{d['ci95_hi']:.0f}]"
        rstr = f"{r['mean']:.3f}" if r["mean"] else "—"
        flag = "← w₀" if tb == W0 else ""
        print(f"  {tb:>10}d  {hstr:>24}  {dstr:>24}  {rstr:>10}  {flag}")

    print(f"\n  Log-log slope (max_occ vs bin_width):")
    print(f"    hotspot: slope={slope_h:.4f}  diffuse: slope={slope_d:.4f}  mean={slope_mean:.4f}")
    print(f"    Interpretation threshold: ≥{SLOPE_LINEAR_THRESHOLD} = linear; ≤{SLOPE_PLATEAU_THRESHOLD} = plateau")
    print(f"\n  Hotspot/diffuse ratio at finest bin (tb=1):  {ratio_w1:.3f}")
    print(f"  Hotspot/diffuse ratio at baseline (tb=5):    {ratio_w0:.3f}")
    print(f"  Signed drift: (ratio_w1-ratio_w0)/ratio_w0 = {signed_drift:+.3f}  "
          f"(threshold ±{RATIO_DRIFT_THRESHOLD}: {axis_b})")

    print(f"\n{'═'*70}")
    print("  PRE-REGISTERED DECISION RULE (§4, Handoff 12)")
    print(f"{'═'*70}")
    print(f"  (A) Max-occ vs bin-width: {axis_a}")
    print(f"  (B) Geometry ratio vs bin-width: {axis_b}")
    print(f"  CELL FIRED: {decision['cell']}")
    print(f"  VERDICT:    {decision['verdict']}")
    print(f"  CONSEQUENCE: {decision['consequence']}")
    print(f"  H10 status:  {decision['h10_status']}")
    print(f"  H11 status:  {decision['h11_status']}")

    # ── Occupancy distribution tail ─────────────────────────────────────────
    print(f"\n  Finest-bin occupancy distribution (tb=1, hotspot, seed=42):")
    rng42 = np.random.default_rng(42)
    agents42 = sample_traversement_potential(n=300, rg_scale_km=2.4, rg_growth_exponent=1.65,
        seed_hiv_prevalence=0.07, rng=rng42)
    traj42 = generate_walks(agents=agents42, venues=proxies_dict["ssp_concentrated_hotspot"],
        n_steps=N_STEPS, epr_rho=0.60, epr_gamma=0.21, jump_length_exponent=0.60,
        rng=rng42, venue_return_boost=5.0)
    occ_tb1 = _derive_occupancy(traj42, time_bin=1)
    arr1 = np.array(occ_tb1["all_occupancies"])
    print(f"    n_events={occ_tb1['n_events']}  mean={occ_tb1['mean_occ']:.1f}  "
          f"p90={occ_tb1['p90_occ']:.0f}  max={occ_tb1['max_occ']}")
    for thresh in [5, 10, 20, 30, 50]:
        pct = (arr1 > thresh).sum() / len(arr1) * 100 if len(arr1) > 0 else 0
        print(f"    fraction bins > {thresh:3d} occupants: {pct:.1f}%")

    # ── Artifact-vs-physical per-step rate ─────────────────────────────────
    hotspot_tb1_mean = agg["ssp_concentrated_hotspot"][1]["mean"] or 0
    diffuse_tb1_mean = agg["ssp_diffuse_market"][1]["mean"] or 0
    print(f"\n  Physical single-day max co-presence (tb=1):")
    print(f"    hotspot:  {hotspot_tb1_mean:.0f}/300 = {hotspot_tb1_mean/300*100:.1f}%")
    print(f"    diffuse:  {diffuse_tb1_mean:.0f}/300 = {diffuse_tb1_mean/300*100:.1f}%")
    print(f"    Field (Friedman 1997 shooting gallery): ~5–20 concurrent users")
    print(f"    Density overshoot at tb=1: hotspot {hotspot_tb1_mean:.0f} vs ~12 = "
          f"×{hotspot_tb1_mean/12:.1f}")
    print(f"    Layer-2 FOI excess at tb=1: ~{(hotspot_tb1_mean/12)**2:.0f}× (quadratic)")

    # ── Artifact share ─────────────────────────────────────────────────────
    h_tb5 = agg["ssp_concentrated_hotspot"][W0]["mean"] or 1
    artifact_ratio = h_tb5 / hotspot_tb1_mean if hotspot_tb1_mean > 0 else None
    print(f"\n  Bin-width artifact factor: max_occ(tb=5)/max_occ(tb=1) = "
          f"{h_tb5:.0f}/{hotspot_tb1_mean:.0f} = {artifact_ratio:.2f}× at hotspot")
    print(f"  (This is how much the 5-day bin inflates occupancy vs single-day)")

    # ── Revised density estimate ────────────────────────────────────────────
    revised_density_ratio = hotspot_tb1_mean / 12  # vs field ~12 (mid Friedman)
    revised_h11_gap = revised_density_ratio ** 2   # Layer-2 quadratic
    print(f"\n  Revised Layer-2 density excess at physical bin (tb=1):")
    print(f"    {hotspot_tb1_mean:.0f}/12 = ×{revised_density_ratio:.1f} occupancy")
    print(f"    Layer-2 FOI: ×{revised_h11_gap:.0f}  (vs H11 claim of ×{(145/12)**2:.0f} at tb=5)")

    # Build artifact
    artifact = {
        "git_sha": sha,
        "timestamp": datetime.now().isoformat(),
        "resolution": {
            "native_step_days": NATIVE_STEP_DAYS,
            "stored_artifacts": "summary JSONs only; no raw trajectories on disk",
            "re_binning_floor": f"time_bin=1 (1 day); full grid {BIN_WIDTHS} available",
            "re_derivation_note": "Trajectories re-derived from EPR walk, same seeds; no transmission.",
        },
        "params": {
            "n_steps": N_STEPS, "space_bin": SPACE_BIN,
            "bin_widths": BIN_WIDTHS, "w0": W0,
        },
        "seeds": seeds,
        "toy_validation": toy_result,
        "max_occupancy_table": {
            proxy: {str(tb): agg[proxy][tb] for tb in BIN_WIDTHS}
            for proxy in PROXIES
        },
        "hotspot_diffuse_ratio_per_binwidth": {
            str(tb): ratios[tb] for tb in BIN_WIDTHS
        },
        "loglog_regression": {
            "hotspot": {"slope": slope_h, "r2": round(loglog_h.rvalue**2, 4)},
            "diffuse": {"slope": slope_d, "r2": round(loglog_d.rvalue**2, 4)},
            "mean_slope": slope_mean,
        },
        "decision": {
            "axis_a": axis_a,
            "axis_b": axis_b,
            "ratio_at_w0": round(ratio_w0, 4),
            "ratio_at_w1_finest": round(ratio_w1, 4),
            "signed_drift": round(signed_drift, 4),
            "ratio_drift_abs": round(ratio_drift, 4),
            **decision,
        },
        "physical_bin_stats": {
            "hotspot_tb1_mean": round(hotspot_tb1_mean, 1),
            "diffuse_tb1_mean": round(diffuse_tb1_mean, 1),
            "artifact_factor_tb5_vs_tb1": round(artifact_ratio, 2) if artifact_ratio else None,
            "revised_density_excess_vs_field": round(revised_density_ratio, 1),
            "revised_layer2_foi_excess": round(revised_h11_gap, 0),
        },
    }

    out_path = os.path.join(output_dir, "occupancy_binwidth_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote: {out_path}")
    return artifact


def _decision_rule(axis_a: str, axis_b: str, slope: float) -> dict:
    """
    Apply the pre-registered decision rule (§4, Handoff 12) mechanically.
    Returns verdict dict.
    """
    if axis_a.startswith("LINEAR") and axis_b == "SHRINKING":
        return {
            "cell": "(A)=LINEAR, (B)=SHRINKING",
            "verdict": "ARTIFACT, VERDICT WOBBLES",
            "consequence": "Occupancy-realism handoff REQUIRED. H10 geometry verdict AND "
                           "H11 density both provisional pending re-run at physical bin width.",
            "h10_status": "PROVISIONAL — geometry verdict contaminated by binning artifact",
            "h11_status": "PROVISIONAL — density magnitude inflated by binning",
        }
    elif axis_a.startswith("LINEAR") and axis_b == "FLAT":
        return {
            "cell": "(A)=LINEAR, (B)=FLAT",
            "verdict": "ARTIFACT, VERDICT ROBUST",
            "consequence": "Occupancy-realism handoff required for H11 density number. "
                           "H10 geometry verdict stands (ratio bin-invariant).",
            "h10_status": "CONFIRMED — geometry ratio bin-width-invariant",
            "h11_status": "PROVISIONAL — density magnitude (×222) inflated by binning artifact; "
                          "revised estimate uses tb=1 occupancy",
        }
    elif axis_a.startswith("PLATEAU") and axis_b == "FLAT":
        return {
            "cell": "(A)=PLATEAU, (B)=FLAT",
            "verdict": "GENUINE DENSITY, VERDICT ROBUST",
            "consequence": "H11 density story stands as-is. Proceed to HCV sentinel.",
            "h10_status": "CONFIRMED",
            "h11_status": "CONFIRMED",
        }
    elif axis_a.startswith("PLATEAU") and axis_b == "SHRINKING":
        return {
            "cell": "(A)=PLATEAU, (B)=SHRINKING",
            "verdict": "CONTRADICTORY — INVESTIGATE",
            "consequence": "Should not co-occur cleanly. Debug occupancy metric or re-binning "
                           "before drawing any conclusion.",
            "h10_status": "INDETERMINATE",
            "h11_status": "INDETERMINATE",
        }
    else:  # INTERMEDIATE
        if axis_b == "GROWING":
            # Ratio INCREASES at finer bins: geometry separation STRONGER at physical scale
            # Not in original pre-registered table — report as NOT PRE-REGISTERED finding
            return {
                "cell": f"(A)=INTERMEDIATE (slope={slope:.2f}), (B)=GROWING (NOT PRE-REGISTERED)",
                "verdict": "PARTIAL ARTIFACT — VERDICT POTENTIALLY STRENGTHENED",
                "consequence": (
                    f"Scaling exponent = {slope:.2f}. H11 density magnitude overstated; "
                    f"revised estimate at tb=1. "
                    f"IMPORTANT: geometry ratio INCREASES at finer bins (hotspot more differentiated "
                    f"from diffuse at 1-day vs 5-day scale). The 5-day binning UNDERESTIMATED the "
                    f"geometry separation. Outcome 2 ('static-blind') rests on a ratio measured at "
                    f"a bin width that understated the effect. Re-running dynamics at tb=1 could "
                    f"push the hotspot/diffuse peak-incidence ratio above the 1.25× threshold, "
                    f"reversing Outcome 2 → Outcome 1. This is the most important finding of H12."
                ),
                "h10_status": (
                    "PROVISIONAL — but in the STRENGTHENING direction: geometry separation "
                    "at physical scale (tb=1) is larger than what H10/H11 measured at tb=5. "
                    "Re-confirmation at physical bin width may STRENGTHEN (not weaken) the verdict."
                ),
                "h11_status": (
                    f"PARTIAL ARTIFACT — ×222 gap should be revised downward using tb=1 occupancy. "
                    f"But geometry effect is larger at physical scale."
                ),
            }
        else:
            b_verdict = "ROBUST" if axis_b == "FLAT" else "WOBBLES"
            return {
                "cell": f"(A)=INTERMEDIATE (slope={slope:.2f}), (B)={axis_b}",
                "verdict": f"PARTIAL ARTIFACT — VERDICT {b_verdict}",
                "consequence": (
                    f"Scaling exponent = {slope:.2f} (between 0 and 1). "
                    f"H11 density magnitude overstated by bin-width artifact; "
                    f"revised figure uses tb=1 occupancy. "
                    + ("H10 geometry verdict stands (ratio bin-width-invariant)."
                       if axis_b == "FLAT"
                       else "H10 geometry verdict needs re-confirmation at physical bin width.")
                ),
                "h10_status": ("CONFIRMED — geometry ratio flat across bin widths"
                               if axis_b == "FLAT"
                               else "PROVISIONAL — ratio drifts with bin width"),
                "h11_status": (
                    f"PARTIAL ARTIFACT — density magnitude inflated by binning "
                    f"(artifact factor = max_occ(w₀)/max_occ(w=1)); "
                    f"×222 should be replaced by revised estimate at tb=1."
                ),
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    run_diagnostic(output_dir=args.output, n_seeds=args.seeds)
