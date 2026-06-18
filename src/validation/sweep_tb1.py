"""
Physical-scale verdict re-run at time_bin=1 day (Handoff 13).

H12 showed the geometry verdict (Outcome 2, static-blind) was measured at
tb=5, a bin width where the hotspot/diffuse occupancy ratio (1.66×) is LOWER
than at physical tb=1 (1.84×).  This re-run checks whether the peak-incidence
geometry ratio crosses the 1.25× pre-registered threshold at tb=1.

ONLY PARAMETER CHANGED: time_bin = 1 day (was 5).
All else identical to H10 desaturation sweep:
  - SeedSequence root and spawn structure (seeds 42–61)
  - Contact cap = 2, min_colocs = 4
  - γ anchors (GAMMA_LOW, GAMMA_HIGH)
  - β values (sourced, never tuned)
  - Arm structure (0/1/2)
  - No re-tuning of ⟨k⟩, β, or any dynamics parameter

Time-bin-1 overrides (derived, not fit):
  TIME_BIN_DAYS = 1
  N_BINS_PER_TILE = N_STEPS_STRUCTURE // 1 = 80
  ACUTE_DURATION_BINS = 90  (90 days / 1 day/bin = 90 bins)
  LATE_START_BINS = 2920  (8 years / 1 day/bin; unreachable in 800-day sim)
  BETA_STATIC_PER_BIN = β_syringe × inj_freq × 1d × shared_frac

Sanity gate (§2): report ⟨k⟩ at tb=1 vs tb=5; flag drift; do NOT re-tune.

Pre-registered decision rule (§4):
  CI upper < 1.25× → Outcome 2 CONFIRMED at physical scale
  CI lower > 1.25× → Outcome 1 PROVISIONAL (arms H14 convergence check)
  CI straddles 1.25× → INDETERMINATE (arms H14, widened)

House-style constraints:
  No sklearn/statsmodels/lifelines/scipy.integrate.
  Bootstrap: numpy, 1000×, paired resample, seed=42.

Usage:
    python src/validation/sweep_tb1.py [--output outputs/] [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import networkx as nx
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from mobility.network_generator import (
    sample_traversement_potential, generate_walks,
    colocation_counter_capped, build_contact_edges, build_contact_graph,
)
from mobility.attractor_proxies import all_proxies
from outbreak.event_stream import build_event_stream, tile_event_stream
from outbreak.transmission import run_temporal_with_gamma
from outbreak import params as P
from outbreak.desaturation import (
    assign_gamma_zero, assign_gamma_uniform, assign_gamma_heterogeneous,
    verify_mean_match, GAMMA_LOW, GAMMA_HIGH, TAU_DAYS,
)
from outbreak.observation_layer import compute_lambda_obs, check_envelope
from validation.commensurability_audit import period_incidence_per_100py

# ── tb=1 parameter overrides ──────────────────────────────────────────────────
TB1_TIME_BIN_DAYS      = 1
TB1_N_BINS_PER_TILE    = P.N_STEPS_STRUCTURE // TB1_TIME_BIN_DAYS   # 80 bins/tile
TB1_ACUTE_DURATION_BINS = P.ACUTE_DURATION_DAYS // TB1_TIME_BIN_DAYS # 90 bins
TB1_LATE_START_BINS    = (8 * 365) // TB1_TIME_BIN_DAYS              # 2920 (unreachable)
TB1_BETA_STATIC_PER_BIN = (
    P.BETA_SYRINGE_CHRONIC * P.INJECTION_FREQ_PER_DAY
    * TB1_TIME_BIN_DAYS * P.SHARED_FRACTION_PER_PARTNER
)  # 0.000024

TB1_EPIDEMIC_DAYS = TB1_N_BINS_PER_TILE * P.N_TILES_OUTBREAK * TB1_TIME_BIN_DAYS  # 800 d
W1_DAYS           = 365   # same 1-year comparison window

PROXY_KNOBS = {
    "ssp_concentrated_hotspot": {"contact_cap": 2, "min_colocs": 4},
    "ssp_diffuse_market":       {"contact_cap": 2, "min_colocs": 4},
}
PROXIES  = ["ssp_concentrated_hotspot", "ssp_diffuse_market"]
ANCHORS  = {"gamma_low": GAMMA_LOW, "gamma_high": GAMMA_HIGH}
BOOTSTRAP_SEED = 42
BOOTSTRAP_N    = 1000

# Pre-registered threshold (§4)
VERDICT_THRESHOLD = 1.25


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def _ci95_bootstrap(values: list) -> dict:
    arr = np.array([v for v in values if v is not None and not np.isnan(float(v))])
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci95_lo": None, "ci95_hi": None, "n": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = rng.choice(arr, size=(BOOTSTRAP_N, n), replace=True).mean(axis=1)
    return {
        "mean":    round(float(np.mean(arr)), 4),
        "ci95_lo": round(float(np.percentile(boots, 2.5)), 4),
        "ci95_hi": round(float(np.percentile(boots, 97.5)), 4),
        "std":     round(float(np.std(arr, ddof=1)), 4) if n > 1 else 0.0,
        "n": n,
    }


def _ratio_bootstrap_ci(hotspot_vals: list, diffuse_vals: list) -> dict:
    """
    Paired bootstrap CI on the geometry ratio = mean(hotspot) / mean(diffuse).
    Resamples (hotspot, diffuse) pairs with replacement.
    """
    h = np.array(hotspot_vals, dtype=float)
    d = np.array(diffuse_vals, dtype=float)
    assert len(h) == len(d), "Paired bootstrap requires equal-length arrays."
    n = len(h)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ratios = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        h_b = h[idx].mean()
        d_b = d[idx].mean()
        ratios.append(h_b / d_b if d_b > 0 else np.nan)
    arr = np.array([r for r in ratios if not np.isnan(r)])
    observed = h.mean() / d.mean() if d.mean() > 0 else np.nan
    return {
        "observed_ratio":  round(float(observed), 4),
        "ci95_lo": round(float(np.percentile(arr, 2.5)), 4),
        "ci95_hi": round(float(np.percentile(arr, 97.5)), 4),
        "std":     round(float(np.std(arr)), 4),
        "n_seeds": n,
    }


# ── Single run at tb=1 ────────────────────────────────────────────────────────

def one_run_tb1(
    proxy_name: str,
    seed: int,
    proxies_dict: dict,
    arm: int,
    gamma_anchor: float,
    n_agents: int = 300,
) -> dict:
    """
    Re-run transmission dynamics at time_bin=1.
    Same SeedSequence root as H10; only time_bin changes.
    """
    ss = np.random.SeedSequence(seed)
    rng_struct, rng_tx, rng_rm = [np.random.default_rng(c) for c in ss.spawn(3)]

    knobs  = PROXY_KNOBS[proxy_name]
    venues = proxies_dict[proxy_name]

    agents = sample_traversement_potential(
        n=n_agents, rg_scale_km=2.4, rg_growth_exponent=1.65,
        seed_hiv_prevalence=0.07, rng=rng_struct,
    )
    trajectories = generate_walks(
        agents=agents, venues=venues, n_steps=P.N_STEPS_STRUCTURE,
        epr_rho=0.60, epr_gamma=0.21, jump_length_exponent=0.60,
        rng=rng_struct, venue_return_boost=5.0,
    )

    # Static graph at tb=1 (for ⟨k⟩ sanity gate and seed_agent selection)
    coloc = colocation_counter_capped(
        trajectories,
        contact_cap=knobs["contact_cap"],
        rng=rng_struct,
        time_bin=TB1_TIME_BIN_DAYS,          # ← time_bin=1
    )
    edges  = build_contact_edges(coloc, min_colocs=knobs["min_colocs"])
    G, stats = build_contact_graph(edges, n_agents=n_agents)

    comps  = sorted(nx.connected_components(G), key=len, reverse=True)
    gc     = sorted(comps[0]) if comps else list(range(n_agents))
    seed_agent = gc[seed % len(gc)]

    agent_rgs = np.array([a.rg for a in agents])
    if arm == 0:
        gamma_arr = assign_gamma_zero(n_agents)
    elif arm == 1:
        gamma_arr = assign_gamma_uniform(n_agents, gamma_anchor)
    else:
        gamma_arr = assign_gamma_heterogeneous(agent_rgs, gamma_anchor)
        assert verify_mean_match(gamma_arr, gamma_anchor)["pass"]

    # Event stream at tb=1
    base_stream = build_event_stream(
        trajectories, space_bin=0.5, time_bin=TB1_TIME_BIN_DAYS
    )
    stream = tile_event_stream(
        base_stream, P.N_TILES_OUTBREAK, TB1_N_BINS_PER_TILE
    )

    raw = run_temporal_with_gamma(
        event_stream=stream,
        n_agents=n_agents,
        seed_agent=seed_agent,
        contact_cap=knobs["contact_cap"],
        beta_syringe=P.BETA_SYRINGE_CHRONIC,
        beta_env=P.BETA_ENV_CHRONIC,
        gamma_array=gamma_arr,
        relocation_fraction=0.0,
        acute_dur_bins=TB1_ACUTE_DURATION_BINS,     # ← 90 bins at tb=1
        late_start_bins=TB1_LATE_START_BINS,
        acute_mult=P.ACUTE_MULTIPLIER,
        late_mult=P.LATE_MULTIPLIER,
        rng_transmission=rng_tx,
        rng_removal=rng_rm,
        time_bin_days=TB1_TIME_BIN_DAYS,            # ← 1 day/bin
        outbreak_threshold=P.OUTBREAK_THRESHOLD,
    )

    # W1 period incidence at tb=1
    w1 = period_incidence_per_100py(
        raw["infection_bin_raw"], raw["removed_bin_by_agent"],
        n_agents, seed_agent, W1_DAYS, TB1_TIME_BIN_DAYS,
    )

    raw["mean_degree"]      = stats.mean_degree
    raw["gc_fraction"]      = stats.giant_component_fraction
    raw["w1_rate_per_100py"]= w1["rate_per_100py"]
    raw["w1_susceptible_py"]= w1["susceptible_py"]
    raw["gamma_anchor"]     = gamma_anchor
    raw["arm"]              = arm
    return raw


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(output_dir: str = "outputs", n_seeds: int = 20, seeds: list = None):
    if seeds is None:
        seeds = list(range(42, 42 + n_seeds))

    os.makedirs(output_dir, exist_ok=True)
    sha = _git_sha()
    proxies_dict = all_proxies()

    print("=" * 70)
    print("HANDOFF 13 — tb=1 PHYSICAL-SCALE VERDICT RE-RUN")
    print(f"  git_sha: {sha[:16]}  n_seeds={n_seeds}  seeds {seeds[0]}–{seeds[-1]}")
    print(f"  time_bin=1d  N_BINS_PER_TILE={TB1_N_BINS_PER_TILE}  "
          f"epidemic_days={TB1_EPIDEMIC_DAYS}")
    print(f"  acute_dur_bins={TB1_ACUTE_DURATION_BINS}  "
          f"β_syringe={P.BETA_SYRINGE_CHRONIC}  β_env={P.BETA_ENV_CHRONIC:.5f}")
    print(f"  β_static/bin={TB1_BETA_STATIC_PER_BIN:.7f}")
    print(f"  verdict threshold: 1.25×  bootstrap_seed={BOOTSTRAP_SEED}  n={BOOTSTRAP_N}")
    print("=" * 70)

    # ── T2 sanity: ⟨k⟩ at tb=1 (Arm0 cells, no removal noise) ────────────────
    print("\n[T2 sanity] ⟨k⟩ at tb=1 vs tb=5 (5 seeds)...")
    k_tb1 = {proxy: [] for proxy in PROXIES}
    for proxy in PROXIES:
        for seed in seeds[:5]:
            r = one_run_tb1(proxy, seed, proxies_dict, arm=0, gamma_anchor=0.0)
            k_tb1[proxy].append(r["mean_degree"])
    k_tb5 = {"ssp_concentrated_hotspot": 3.37, "ssp_diffuse_market": 3.80}
    print(f"  {'proxy':<32} ⟨k⟩_tb1         ⟨k⟩_tb5  drift")
    for proxy in PROXIES:
        mean_tb1 = float(np.mean(k_tb1[proxy]))
        drift = (mean_tb1 - k_tb5[proxy]) / k_tb5[proxy] * 100
        print(f"  {proxy:<32} {mean_tb1:.2f}  {k_tb5[proxy]:.2f}  {drift:+.1f}%")
        if abs(drift) > 20:
            print(f"    *** ⟨k⟩ DRIFT > 20% — tb=1 incidence comparison is partly a ⟨k⟩ comparison.")
            print(f"    *** DO NOT re-tune. Caveat on verdict interpretation.")

    # ── T1 full sweep ─────────────────────────────────────────────────────────
    all_rows: dict = {}
    for anchor_name, gamma_anchor in ANCHORS.items():
        for proxy in PROXIES:
            for arm in [0, 1, 2]:
                key = (proxy, arm, anchor_name)
                rows = []
                print(f"\n[{proxy}  arm={arm}  {anchor_name}]  tb=1")
                for seed in seeds:
                    r = one_run_tb1(proxy, seed, proxies_dict, arm, gamma_anchor)
                    rows.append(r)
                    print(f"  s={seed}  fs={r['final_size']:3d}  "
                          f"peak={r['peak_incidence_per_100py']:6.0f}  "
                          f"W1={r['w1_rate_per_100py']:6.0f}  "
                          f"k={r['mean_degree']:.2f}")
                all_rows[key] = rows

    # ── T3 geometry ratio + bootstrap CI ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GEOMETRY RATIO AT tb=1  (pre-registered decision target)")
    print("=" * 70)

    ratio_results = {}
    for anchor_name in ANCHORS:
        for metric_key, metric_label in [
            ("peak_incidence_per_100py", "peak_inc/100py"),
            ("w1_rate_per_100py", "W1 rate/100py"),
        ]:
            h_rows = all_rows.get(("ssp_concentrated_hotspot", 1, anchor_name), [])
            d_rows = all_rows.get(("ssp_diffuse_market",       1, anchor_name), [])
            if not h_rows or not d_rows:
                continue
            h_vals = [r[metric_key] for r in h_rows]
            d_vals = [r[metric_key] for r in d_rows]
            ratio_ci = _ratio_bootstrap_ci(h_vals, d_vals)
            env_status = check_envelope(ANCHORS[anchor_name])["status"]
            ratio_results[(anchor_name, metric_key)] = ratio_ci

            print(f"\n  {anchor_name}  {env_status}  metric={metric_label}")
            print(f"    hotspot mean: {np.mean(h_vals):.0f}/100py  "
                  f"diffuse mean: {np.mean(d_vals):.0f}/100py")
            print(f"    ratio = {ratio_ci['observed_ratio']:.4f}  "
                  f"95% CI [{ratio_ci['ci95_lo']:.4f}, {ratio_ci['ci95_hi']:.4f}]")

    # Also compare tb=1 vs tb=5 for context
    print(f"\n  tb=5 (H10/H11 reference, peak_inc geometry ratio): 1.24×")
    print(f"  tb=1 (this run):  see table above")

    # ── T4 verdict ─────────────────────────────────────────────────────────────
    key_ratio = ratio_results.get(("gamma_low", "peak_incidence_per_100py"), {})
    r_obs  = key_ratio.get("observed_ratio") or 0.0
    r_lo   = key_ratio.get("ci95_lo") or 0.0
    r_hi   = key_ratio.get("ci95_hi") or 0.0

    if r_hi < VERDICT_THRESHOLD:
        verdict = "OUTCOME_2_CONFIRMED"
        verdict_note = (
            f"CI upper bound {r_hi:.4f} < 1.25×. "
            "Outcome 2 (static-blind) CONFIRMED at physical scale (tb=1). "
            "The finer scale where geometry separation is largest still did not cross. "
            "H10 verdict upgraded from provisional to CONFIRMED. H14 NOT armed."
        )
        h14_armed = False
    elif r_lo > VERDICT_THRESHOLD:
        verdict = "OUTCOME_1_PROVISIONAL"
        verdict_note = (
            f"CI lower bound {r_lo:.4f} > 1.25×. "
            "Outcome 1 (geometry-legible) — PROVISIONAL. "
            "H14 convergence check ARMED (cannot claim Outcome 1 until tb=1 is confirmed "
            "as a plateau, not a continuing upward trend)."
        )
        h14_armed = True
    else:
        verdict = "INDETERMINATE"
        verdict_note = (
            f"CI straddles 1.25× [{r_lo:.4f}, {r_hi:.4f}]. "
            "Ratio is threshold-adjacent and resolution-sensitive. "
            "H14 convergence check ARMED (widened)."
        )
        h14_armed = True

    print(f"\n{'═'*70}")
    print("  PRE-REGISTERED DECISION RULE (§4, Handoff 13)")
    print(f"{'═'*70}")
    print(f"  Peak-inc geometry ratio (Arm1, γ_low, tb=1): {r_obs:.4f} [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"  Threshold: {VERDICT_THRESHOLD}×")
    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_note}")
    print(f"  H14 armed: {h14_armed}")

    # ── Comparison table tb=1 vs tb=5 ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  tb=1 vs tb=5 comparison (Arm1, γ_low)")
    print(f"{'─'*70}")
    print(f"  {'Metric':<30}  {'hotspot tb=5':>14}  {'hotspot tb=1':>14}  "
          f"{'diffuse tb=5':>14}  {'diffuse tb=1':>14}")
    tb5_ref = {
        "ssp_concentrated_hotspot": {"peak": 3690, "w1": 4124, "k": 3.37},
        "ssp_diffuse_market":       {"peak": 3317, "w1": 3185, "k": 3.80},
    }
    for metric, label in [("peak", "peak_inc/100py"), ("w1", "W1 rate/100py"), ("k", "mean_degree")]:
        hkey = "peak_incidence_per_100py" if metric == "peak" else ("w1_rate_per_100py" if metric == "w1" else "mean_degree")
        h1_rows = all_rows.get(("ssp_concentrated_hotspot", 1, "gamma_low"), [])
        d1_rows = all_rows.get(("ssp_diffuse_market",       1, "gamma_low"), [])
        h1_mean = float(np.mean([r[hkey] for r in h1_rows])) if h1_rows else 0
        d1_mean = float(np.mean([r[hkey] for r in d1_rows])) if d1_rows else 0
        h5 = tb5_ref["ssp_concentrated_hotspot"][metric]
        d5 = tb5_ref["ssp_diffuse_market"][metric]
        print(f"  {label:<30}  {h5:>14.0f}  {h1_mean:>14.0f}  {d5:>14.0f}  {d1_mean:>14.0f}")

    # ── Build artifact ─────────────────────────────────────────────────────────
    def _agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return {"mean": None}
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            "n":    len(vals),
        }

    artifact = {
        "git_sha": sha,
        "timestamp": datetime.now().isoformat(),
        "time_bin": TB1_TIME_BIN_DAYS,
        "params": {
            "time_bin_days": TB1_TIME_BIN_DAYS,
            "n_bins_per_tile": TB1_N_BINS_PER_TILE,
            "acute_duration_bins": TB1_ACUTE_DURATION_BINS,
            "epidemic_days": TB1_EPIDEMIC_DAYS,
            "beta_syringe": P.BETA_SYRINGE_CHRONIC,
            "beta_env": P.BETA_ENV_CHRONIC,
            "gamma_low": GAMMA_LOW,
            "gamma_high": GAMMA_HIGH,
        },
        "seeds": seeds,
        "k_drift_sanity": {
            proxy: {
                "mean_k_tb1": round(float(np.mean(k_tb1[proxy])), 3),
                "ref_k_tb5":  k_tb5[proxy],
                "drift_pct":  round((float(np.mean(k_tb1[proxy])) - k_tb5[proxy]) / k_tb5[proxy] * 100, 1),
            }
            for proxy in PROXIES
        },
        "cell_summaries": {
            f"{proxy}__arm{arm}__{anc}": {
                m: _agg(all_rows[(proxy, arm, anc)], m)
                for m in ["final_size", "peak_incidence_per_100py",
                           "w1_rate_per_100py", "mean_degree",
                           "removed_while_S", "removed_while_I"]
            }
            for proxy, arm, anc in all_rows
        },
        "geometry_ratios": {
            f"{anc}__{mk}": ratio_results[(anc, mk)]
            for anc, mk in ratio_results
        },
        "verdict": {
            "code": verdict,
            "note": verdict_note,
            "ratio_observed": round(r_obs, 4),
            "ci95_lo": round(r_lo, 4),
            "ci95_hi": round(r_hi, 4),
            "threshold": VERDICT_THRESHOLD,
            "h14_armed": h14_armed,
            "h10_status": (
                "CONFIRMED (strongest static-blind result)"
                if verdict == "OUTCOME_2_CONFIRMED" else
                "FLIPPED — Outcome 1 provisional (arms H14)"
                if verdict == "OUTCOME_1_PROVISIONAL" else
                "INDETERMINATE — arms H14 (widened)"
            ),
        },
    }

    out_path = os.path.join(output_dir, "tb1_verdict.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nWrote: {out_path}")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    run_sweep(output_dir=args.output, n_seeds=args.seeds)
