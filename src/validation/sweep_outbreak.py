"""
Stage A/B/C outbreak sweep (Handoff 9).

Architecture:
  Trajectories: n_steps=80 (preserves calibrated ⟨k⟩≈3; same as 8-fix/8-geometry).
  Event stream: tiled × N_TILES_OUTBREAK = 10 tiles × 16 bins × 5 d = 800 days.
  Static baseline: contact graph from n_steps=80 trajectories (same knobs).

Stage A — temporal vs static, Layer 1 only, concentrated hotspot.
           Tests: timing/burstiness alone vs uniform-rate static expectation.
Stage B — temporal, add Layer 2 (environmental pulse), concentrated hotspot.
           Tests: does uncapped communal exposure amplify over Stage A?
Stage C — full model (L1+L2), both geometries, ⟨k⟩-matched.
           Tests: does static core-periphery tradeoff predict dynamic explosiveness?

Pre-registered outcomes (decided before reading numbers):
  1. Pulsing amplifies AND concentrated >> diffuse → static tradeoff predicts dynamics.
  2. Pulsing amplifies BUT concentrated ≈ diffuse → static DOES NOT predict outbreak risk.
  3. Pulsing does NOT amplify vs collapsed-graph → static adequate; null, report.

Usage:
    python src/validation/sweep_outbreak.py [--output outputs/] [--seeds 20]
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
from scipy import stats as scipy_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from mobility.network_generator import (
    sample_traversement_potential,
    generate_walks,
    colocation_counter_capped,
    build_contact_edges,
    build_contact_graph,
)
from mobility.attractor_proxies import all_proxies, PROXY_DESCRIPTIONS
from outbreak.event_stream import build_event_stream, tile_event_stream
from outbreak.transmission import run_temporal, run_static
from outbreak import params as P


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def ci95(values: list) -> dict:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci95_lo": None, "ci95_hi": None, "n": 0}
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t = scipy_stats.t.ppf(0.975, df=max(n - 1, 1))
    return {
        "mean": round(mean, 4),
        "ci95_lo": round(mean - t * se, 4),
        "ci95_hi": round(mean + t * se, 4),
        "std": round(float(np.std(arr, ddof=1)), 4) if n > 1 else 0.0,
        "n": n,
    }


def p_outbreak(bools: list) -> float:
    return round(sum(bools) / len(bools), 4) if bools else 0.0


# ── Single run ────────────────────────────────────────────────────────────────

def one_run(
    proxy_name: str,
    seed: int,
    proxies: dict,
    contact_cap: int,
    min_colocs: int,
    beta_env: float,
    mode: str,           # "temporal" | "static"
    n_agents: int = 300,
) -> dict:
    """
    Generate trajectories (n_steps=80, calibrated), build event stream and
    static graph, then run the SI outbreak model.

    Seed agent: random node from the giant component (varies per seed).
    Event stream is tiled × N_TILES_OUTBREAK to extend epidemic horizon.
    """
    rng_gen = np.random.default_rng(seed)
    venues = proxies[proxy_name]

    agents = sample_traversement_potential(
        n=n_agents, rg_scale_km=2.4, rg_growth_exponent=1.65,
        seed_hiv_prevalence=0.07, rng=rng_gen,
    )
    trajectories = generate_walks(
        agents=agents, venues=venues, n_steps=P.N_STEPS_STRUCTURE,
        epr_rho=0.60, epr_gamma=0.21, jump_length_exponent=0.60,
        rng=rng_gen, venue_return_boost=5.0,
    )

    # Build static graph (calibrated knobs)
    coloc = colocation_counter_capped(
        trajectories, contact_cap=contact_cap, rng=rng_gen,
    )
    contact_edges = build_contact_edges(coloc, min_colocs=min_colocs)
    G, stats = build_contact_graph(contact_edges, n_agents=n_agents)

    # Seed agent: random node from giant component (or any node if GC empty)
    if G.number_of_nodes() > 0:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        gc_nodes = sorted(comps[0])
        seed_agent = gc_nodes[seed % len(gc_nodes)]
    else:
        seed_agent = seed % n_agents

    rng_tx = np.random.default_rng(seed + 99999)
    n_bins_total = P.N_BINS_PER_TILE * P.N_TILES_OUTBREAK

    if mode == "temporal":
        base_stream = build_event_stream(trajectories, space_bin=0.5, time_bin=5)
        stream = tile_event_stream(base_stream, P.N_TILES_OUTBREAK, P.N_BINS_PER_TILE)
        result = run_temporal(
            event_stream=stream,
            n_agents=n_agents,
            seed_agent=seed_agent,
            contact_cap=contact_cap,
            beta_syringe=P.BETA_SYRINGE_CHRONIC,
            beta_env=beta_env,
            acute_dur_bins=P.ACUTE_DURATION_BINS,
            late_start_bins=P.LATE_START_BINS,
            acute_mult=P.ACUTE_MULTIPLIER,
            late_mult=P.LATE_MULTIPLIER,
            rng=rng_tx,
            time_bin_days=P.TIME_BIN_DAYS,
            outbreak_threshold=P.OUTBREAK_THRESHOLD,
        )
    else:  # static
        result = run_static(
            contact_edges=contact_edges,
            n_agents=n_agents,
            seed_agent=seed_agent,
            n_bins=n_bins_total,
            beta_static_per_bin=P.BETA_STATIC_PER_BIN_CHRONIC,
            acute_dur_bins=P.ACUTE_DURATION_BINS,
            late_start_bins=P.LATE_START_BINS,
            acute_mult=P.ACUTE_MULTIPLIER,
            late_mult=P.LATE_MULTIPLIER,
            rng=rng_tx,
            time_bin_days=P.TIME_BIN_DAYS,
            outbreak_threshold=P.OUTBREAK_THRESHOLD,
        )

    result["mean_degree"] = stats.mean_degree
    result["giant_component_fraction"] = stats.giant_component_fraction
    return result


# ── ⟨k⟩-matching (5-seed estimate, n_steps=80) ───────────────────────────────

def _mean_k(proxy_name, contact_cap, min_colocs, proxies,
            seeds=(42, 43, 44, 45, 46)):
    ks = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        agents = sample_traversement_potential(
            n=300, rg_scale_km=2.4, rg_growth_exponent=1.65,
            seed_hiv_prevalence=0.07, rng=rng,
        )
        traj = generate_walks(agents=agents, venues=proxies[proxy_name],
                               n_steps=P.N_STEPS_STRUCTURE,
                               epr_rho=0.60, epr_gamma=0.21, jump_length_exponent=0.60,
                               rng=rng, venue_return_boost=5.0)
        coloc = colocation_counter_capped(traj, contact_cap=contact_cap, rng=rng)
        edges = build_contact_edges(coloc, min_colocs=min_colocs)
        _, stats = build_contact_graph(edges, n_agents=300)
        ks.append(stats.mean_degree)
    return float(np.mean(ks))


def find_matched_knobs(proxy_name, target_k, proxies):
    """Grid-search (cap, min_c) → ⟨k⟩ closest to target_k."""
    best = (2, 4, _mean_k(proxy_name, 2, 4, proxies))
    for cap in [1, 2, 3]:
        for mc in [2, 3, 4, 5, 6, 7, 8, 10]:
            k = _mean_k(proxy_name, cap, mc, proxies)
            if abs(k - target_k) < abs(best[2] - target_k):
                best = (cap, mc, k)
    return best


# ── Sweep ─────────────────────────────────────────────────────────────────────

METRICS = ["final_size", "single_cluster_fraction", "doubling_time_days",
           "peak_incidence_per_100py", "mean_degree", "giant_component_fraction"]


def aggregate(rows: list) -> dict:
    out = {}
    for m in METRICS:
        out[m] = ci95([r[m] for r in rows])
    out["P_outbreak"] = p_outbreak([r["is_outbreak"] for r in rows])
    return out


def run_sweep(output_dir: str = "outputs", n_seeds: int = 20, seeds: list = None):
    if seeds is None:
        seeds = list(range(42, 42 + n_seeds))
    os.makedirs(output_dir, exist_ok=True)
    sha = _git_sha()
    proxies = all_proxies()

    epidemic_days = P.N_BINS_PER_TILE * P.N_TILES_OUTBREAK * P.TIME_BIN_DAYS

    print("=" * 70)
    print("HANDOFF 9 OUTBREAK SWEEP")
    print(f"  git_sha: {sha[:16]}  n_seeds={n_seeds}  seeds {seeds[0]}–{seeds[-1]}")
    print(f"  n_steps_structure={P.N_STEPS_STRUCTURE}  "
          f"n_tiles={P.N_TILES_OUTBREAK} × {P.N_BINS_PER_TILE} bins × "
          f"{P.TIME_BIN_DAYS} d = {epidemic_days} epidemic days")
    print(f"  β_syringe={P.BETA_SYRINGE_CHRONIC}  β_env={P.BETA_ENV_CHRONIC:.5f}  "
          f"acute_mult={P.ACUTE_MULTIPLIER}  β_static/bin={P.BETA_STATIC_PER_BIN_CHRONIC:.6f}")
    print(f"  acute_dur_bins={P.ACUTE_DURATION_BINS}  outbreak_threshold={P.OUTBREAK_THRESHOLD}")
    print("=" * 70)

    # ── Stage C calibration ───────────────────────────────────────────────────
    print("\n[Stage C calibration] 5-seed ⟨k⟩ estimates (n_steps=80)...")
    hotspot_k = _mean_k("ssp_concentrated_hotspot", 2, 4, proxies)
    diffuse_k  = _mean_k("ssp_diffuse_market",       2, 4, proxies)
    target_k   = hotspot_k
    print(f"  hotspot (cap=2,mc=4): ⟨k⟩={hotspot_k:.2f}")
    print(f"  diffuse (cap=2,mc=4): ⟨k⟩={diffuse_k:.2f}  target={target_k:.2f}")
    d_cap, d_mc, d_k = find_matched_knobs("ssp_diffuse_market", target_k, proxies)
    print(f"  diffuse matched: cap={d_cap}, mc={d_mc} → ⟨k⟩={d_k:.2f}")

    # ── Stage A ───────────────────────────────────────────────────────────────
    print(f"\n[Stage A] Temporal vs static, hotspot, L1 only ({n_seeds} seeds)")
    rows_At, rows_As = [], []
    for seed in seeds:
        rt = one_run("ssp_concentrated_hotspot", seed, proxies, 2, 4, 0.0, "temporal")
        rs = one_run("ssp_concentrated_hotspot", seed, proxies, 2, 4, 0.0, "static")
        rows_At.append(rt); rows_As.append(rs)
        print(f"  s={seed}  temporal fs={rt['final_size']:3d} out={rt['is_outbreak']}  "
              f"dt={rt['doubling_time_days']}d  "
              f"static fs={rs['final_size']:3d} out={rs['is_outbreak']}  "
              f"dt={rs['doubling_time_days']}d")
    agg_At = aggregate(rows_At)
    agg_As = aggregate(rows_As)

    # ── Stage B ───────────────────────────────────────────────────────────────
    print(f"\n[Stage B] L1+L2 temporal, hotspot ({n_seeds} seeds)")
    rows_B = []
    for seed in seeds:
        r = one_run("ssp_concentrated_hotspot", seed, proxies, 2, 4,
                    P.BETA_ENV_CHRONIC, "temporal")
        rows_B.append(r)
        print(f"  s={seed}  fs={r['final_size']:3d} out={r['is_outbreak']}  "
              f"dt={r['doubling_time_days']}d  "
              f"peak_inc={r['peak_incidence_per_100py']:.0f}/100py")
    agg_B = aggregate(rows_B)

    # ── Stage C ───────────────────────────────────────────────────────────────
    print(f"\n[Stage C] Both geometries, ⟨k⟩-matched, L1+L2 ({n_seeds} seeds)")
    rows_Ch, rows_Cdm, rows_Cdn = [], [], []
    for seed in seeds:
        rh  = one_run("ssp_concentrated_hotspot", seed, proxies, 2, 4,
                      P.BETA_ENV_CHRONIC, "temporal")
        rdm = one_run("ssp_diffuse_market", seed, proxies, d_cap, d_mc,
                      P.BETA_ENV_CHRONIC, "temporal")
        rdn = one_run("ssp_diffuse_market", seed, proxies, 2, 4,
                      P.BETA_ENV_CHRONIC, "temporal")
        rows_Ch.append(rh); rows_Cdm.append(rdm); rows_Cdn.append(rdn)
        print(f"  s={seed}  hotspot fs={rh['final_size']:3d} dt={rh['doubling_time_days']}d  "
              f"diffuse(matched) fs={rdm['final_size']:3d} dt={rdm['doubling_time_days']}d")
    agg_Ch  = aggregate(rows_Ch)
    agg_Cdm = aggregate(rows_Cdm)
    agg_Cdn = aggregate(rows_Cdn)

    # ── Pre-registered verdict ────────────────────────────────────────────────
    B_fs  = agg_B["final_size"]["mean"] or 0
    At_fs = agg_At["final_size"]["mean"] or 0
    B_pi  = agg_B["peak_incidence_per_100py"]["mean"] or 0
    At_pi = agg_At["peak_incidence_per_100py"]["mean"] or 0
    Ch_pi = agg_Ch["peak_incidence_per_100py"]["mean"] or 0
    Cdm_pi= agg_Cdm["peak_incidence_per_100py"]["mean"] or 0
    Ch_P  = agg_Ch["P_outbreak"]
    Cdm_P = agg_Cdm["P_outbreak"]

    # "Amplifies" if Stage B peak_inc or P_outbreak substantially exceeds Stage A
    pulse_amplifies = (B_pi > At_pi * 1.25) or (agg_B["P_outbreak"] > agg_At["P_outbreak"] + 0.10)
    # "concentrated >> diffuse" on peak dynamics
    hotspot_dominates = (Ch_pi > Cdm_pi * 1.25) or (Ch_P > Cdm_P + 0.15)

    if pulse_amplifies and hotspot_dominates:
        verdict = "OUTCOME_1_STATIC_PREDICTS"
        verdict_note = (
            "Pulsing amplifies (Stage B > Stage A ×1.25 or P_outbreak shift >0.10) "
            "AND concentrated >> diffuse (peak_inc or P_outbreak). "
            "Static core-periphery tradeoff predicts dynamic explosiveness. "
            "Gallery mechanism confirmed; concentrated structure is more dangerous."
        )
    elif pulse_amplifies:
        verdict = "OUTCOME_2_STATIC_BLIND"
        verdict_note = (
            "Pulsing amplifies BUT concentrated ≈ diffuse in dynamics. "
            "HEADLINE: static structure does NOT predict outbreak risk. "
            "Diffuse network pulses as hard as concentrated despite appearing "
            "homogeneous on the collapsed graph. Cannot read outbreak risk off "
            "static structure — the whole case for the temporal model."
        )
    else:
        verdict = "OUTCOME_3_NULL_STATIC_ADEQUATE"
        verdict_note = (
            "Pulsing does NOT amplify vs static-graph expectation (Stage B ≈ Stage A). "
            "NULL RESULT: static graph was adequate; single-hit/SI deceleration "
            "dominates (Unicomb 2021). Report as null; do not bury."
        )

    # ── Validation targets ────────────────────────────────────────────────────
    from mobility.params import OUTBREAK_VALIDATION_TARGETS as OVT
    tgt_scott   = OVT["final_size"]["scott_county_in"]
    tgt_cabell  = OVT["final_size"]["cabell_county_wv"]
    tgt_inc     = OVT["explosive_hiv_incidence_per_100py"]["value"]
    tgt_cluster = OVT["cluster_dominance"]["scott_county_pct"] / 100.0

    B_fs_mean  = agg_B["final_size"]["mean"] or 0
    B_pi_mean  = agg_B["peak_incidence_per_100py"]["mean"] or 0
    B_cl_mean  = agg_B["single_cluster_fraction"]["mean"] or 0

    validation = {
        "final_size_generated":        round(B_fs_mean, 1),
        "final_size_empirical_range":  f"{tgt_cabell}–{tgt_scott}",
        "final_size_plausible":        tgt_cabell <= B_fs_mean <= tgt_scott * 3,
        "peak_incidence_generated":    round(B_pi_mean, 2),
        "peak_incidence_target_18_6":  tgt_inc,
        "peak_incidence_pass":         B_pi_mean >= tgt_inc * 0.5,
        "single_cluster_generated":    round(B_cl_mean, 3),
        "single_cluster_target":       f"≥{tgt_cluster:.2f} (Scott County)",
        "single_cluster_pass":         B_cl_mean >= tgt_cluster,
    }

    # ── Print tables ──────────────────────────────────────────────────────────
    _print_table("Stage A — temporal vs static (hotspot, L1 only)",
                 {"temporal L1": agg_At, "static baseline": agg_As})
    _print_table("Stage B — L1+L2 vs L1 only (hotspot, temporal)",
                 {"L1+L2 temporal": agg_B, "L1 only temporal": agg_At})
    _print_table("Stage C — ⟨k⟩-matched geometries, full model (L1+L2)",
                 {f"hotspot (cap=2,mc=4,k≈{hotspot_k:.1f})": agg_Ch,
                  f"diffuse matched (cap={d_cap},mc={d_mc},k≈{d_k:.1f})": agg_Cdm,
                  f"diffuse native (cap=2,mc=4,k≈{diffuse_k:.1f})": agg_Cdn})

    print(f"\nPRE-REGISTERED VERDICT: {verdict}")
    print(f"  {verdict_note}")
    print(f"\nVALIDATION TARGETS:")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    # ── Artifact ──────────────────────────────────────────────────────────────
    artifact = {
        "git_sha": sha,
        "timestamp": datetime.now().isoformat(),
        "architecture": {
            "n_steps_structure": P.N_STEPS_STRUCTURE,
            "n_tiles_outbreak": P.N_TILES_OUTBREAK,
            "n_bins_per_tile": P.N_BINS_PER_TILE,
            "epidemic_days": epidemic_days,
            "note": "Trajectories generated at n_steps=80 (calibrated ⟨k⟩); "
                    "event stream tiled × 10 for 800-day epidemic horizon.",
        },
        "params": {
            "beta_syringe_chronic": P.BETA_SYRINGE_CHRONIC,
            "beta_env_chronic": round(P.BETA_ENV_CHRONIC, 6),
            "env_syringe_ratio": P.ENV_SYRINGE_RATIO,
            "acute_multiplier": P.ACUTE_MULTIPLIER,
            "acute_duration_days": P.ACUTE_DURATION_DAYS,
            "late_multiplier": P.LATE_MULTIPLIER,
            "beta_static_per_bin": round(P.BETA_STATIC_PER_BIN_CHRONIC, 7),
            "outbreak_threshold": P.OUTBREAK_THRESHOLD,
            "sources": P.SOURCES,
        },
        "seeds": seeds,
        "stage_c_calibration": {
            "hotspot_native_k": round(hotspot_k, 3),
            "diffuse_native_k": round(diffuse_k, 3),
            "target_k": round(target_k, 3),
            "diffuse_matched_cap": d_cap,
            "diffuse_matched_min_colocs": d_mc,
            "diffuse_matched_k": round(d_k, 3),
        },
        "stage_A": {"temporal_L1": agg_At, "static_baseline": agg_As},
        "stage_B": {"temporal_L1_L2": agg_B},
        "stage_C": {
            "hotspot_native": agg_Ch,
            "diffuse_matched": agg_Cdm,
            "diffuse_native": agg_Cdn,
        },
        "verdict": verdict,
        "verdict_note": verdict_note,
        "verdict_inputs": {
            "B_peak_inc": round(B_pi, 2),
            "At_peak_inc": round(At_pi, 2),
            "pulse_amplifies": pulse_amplifies,
            "Ch_peak_inc": round(Ch_pi, 2),
            "Cdm_peak_inc": round(Cdm_pi, 2),
            "Ch_P_outbreak": Ch_P,
            "Cdm_P_outbreak": Cdm_P,
            "hotspot_dominates": hotspot_dominates,
        },
        "validation_targets": validation,
    }

    out_path = os.path.join(output_dir, "outbreak_sweep.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote: {out_path}")
    return artifact


def _print_table(title, conditions):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    cols = list(conditions.keys())
    hdr = f"  {'Metric':<30}" + "".join(f"  {c[:24]:>24}" for c in cols)
    print(hdr)
    spec = {
        "final_size": "final_size",
        "P_outbreak": "P(outbreak)",
        "single_cluster_fraction": "single_cluster_frac",
        "doubling_time_days": "doubling_time_days",
        "peak_incidence_per_100py": "peak_inc/100py",
        "mean_degree": "mean_degree",
        "giant_component_fraction": "gc_fraction",
    }
    for key, label in spec.items():
        row = f"  {label:<30}"
        for cond in cols:
            v = conditions[cond]
            if key == "P_outbreak":
                s = f"{v['P_outbreak']:.3f}"
            else:
                vv = v.get(key, {})
                if isinstance(vv, dict) and vv.get("mean") is not None:
                    s = f"{vv['mean']:.2f}[{vv['ci95_lo']:.2f},{vv['ci95_hi']:.2f}]"
                else:
                    s = str(vv)
            row += f"  {s:>24}"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    run_sweep(output_dir=args.output, n_seeds=args.seeds)
