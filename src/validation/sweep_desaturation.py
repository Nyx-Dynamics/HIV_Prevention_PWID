"""
Desaturation ladder driver (Handoff 10).

Controlled 3-arm experiment × 2 geometries × 2 γ anchors × 20 seeds.

Arms:
  Arm 0 — γ=0          Falsification anchor; must reproduce 299/300 saturation.
  Arm 1 — uniform-γ    Desaturation per se; isolates ceiling-lifting effect.
  Arm 2 — γ(U) hetero  Agent-level differential loading, mean-matched to γ_anchor.

RNG: numpy.random.SeedSequence(seed).spawn(3) → rng_structure, rng_transmission,
     rng_removal.  Arms 0/1/2 share rng_structure and rng_transmission byte-for-byte.
     Only rng_removal differs; Arm 0 never draws from it.

Attribution:
  Arm0 → Arm1 = desaturation effect (ceiling lifting, uniform removal).
  Arm1 → Arm2 = concentration effect (differential loading; hotspot self-limitation).

γ parameterization: JAIDS ms "Calibration-to-Deployment Mismatch" (Demidont),
  Eq. 2-3, Eq. 10, Table S4 PWID rows.
Canonical numerics: Nyx-Dynamics/nyx-kassanjee-letter.

Usage:
    python src/validation/sweep_desaturation.py [--output outputs/] [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

import networkx as nx
import numpy as np
from scipy import stats as scipy_stats

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
    verify_mean_match, GAMMA_LOW, GAMMA_HIGH, TAU_DAYS, ANCHORS,
)
from outbreak.observation import compute_lambda_obs, check_envelope


PROXIES = ["ssp_concentrated_hotspot", "ssp_diffuse_market"]
PROXY_KNOBS = {
    "ssp_concentrated_hotspot": {"contact_cap": 2, "min_colocs": 4},
    "ssp_diffuse_market":       {"contact_cap": 2, "min_colocs": 4},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def ci95(values: list) -> dict:
    arr = np.array([v for v in values if v is not None and not np.isnan(float(v))])
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci95_lo": None, "ci95_hi": None, "std": None, "n": 0}
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t = scipy_stats.t.ppf(0.975, df=max(n - 1, 1))
    return {
        "mean": round(mean, 4),
        "ci95_lo": round(mean - t * se, 4),
        "ci95_hi": round(mean + t * se, 4),
        "std": round(float(np.std(arr, ddof=1)), 4) if n > 1 else 0.0,
        "var": round(float(np.var(arr, ddof=1)), 4) if n > 1 else 0.0,
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "n": n,
    }


def p_outbreak(bools: list) -> float:
    return round(sum(bools) / len(bools), 4) if bools else 0.0


# ── One run ───────────────────────────────────────────────────────────────────

def one_run(
    proxy_name: str,
    seed: int,
    proxies_dict: dict,
    arm: int,               # 0, 1, or 2
    gamma_anchor: float,
    n_agents: int = 300,
) -> dict:
    """
    One generator + transmission run.

    3-stream RNG: SeedSequence(seed).spawn(3) → rng_struct, rng_tx, rng_rm.
    Arms 0/1/2 share rng_struct and rng_tx identically.
    Only rng_rm differs; Arm 0 never draws from it.
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(3)
    rng_struct = np.random.default_rng(child_seeds[0])
    rng_tx     = np.random.default_rng(child_seeds[1])
    rng_rm     = np.random.default_rng(child_seeds[2])

    knobs = PROXY_KNOBS[proxy_name]
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
    coloc = colocation_counter_capped(
        trajectories, contact_cap=knobs["contact_cap"], rng=rng_struct,
    )
    contact_edges = build_contact_edges(coloc, min_colocs=knobs["min_colocs"])
    G, stats = build_contact_graph(contact_edges, n_agents=n_agents)

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    gc_nodes = sorted(comps[0]) if comps else list(range(n_agents))
    seed_agent = gc_nodes[seed % len(gc_nodes)]

    # γ assignment (arm-dependent)
    agent_rgs = np.array([a.rg for a in agents])
    if arm == 0:
        gamma_arr = assign_gamma_zero(n_agents)
    elif arm == 1:
        gamma_arr = assign_gamma_uniform(n_agents, gamma_anchor)
    else:  # arm == 2
        gamma_arr = assign_gamma_heterogeneous(agent_rgs, gamma_anchor)
        mmatch = verify_mean_match(gamma_arr, gamma_anchor)
        assert mmatch["pass"], f"Mean-match failed: {mmatch}"

    # Event stream (tiled)
    base_stream = build_event_stream(trajectories, space_bin=0.5, time_bin=5)
    stream = tile_event_stream(base_stream, P.N_TILES_OUTBREAK, P.N_BINS_PER_TILE)

    result = run_temporal_with_gamma(
        event_stream=stream,
        n_agents=n_agents,
        seed_agent=seed_agent,
        contact_cap=knobs["contact_cap"],
        beta_syringe=P.BETA_SYRINGE_CHRONIC,
        beta_env=P.BETA_ENV_CHRONIC,
        gamma_array=gamma_arr,
        relocation_fraction=0.0,
        acute_dur_bins=P.ACUTE_DURATION_BINS,
        late_start_bins=P.LATE_START_BINS,
        acute_mult=P.ACUTE_MULTIPLIER,
        late_mult=P.LATE_MULTIPLIER,
        rng_transmission=rng_tx,
        rng_removal=rng_rm,
        time_bin_days=P.TIME_BIN_DAYS,
        outbreak_threshold=P.OUTBREAK_THRESHOLD,
    )

    result["mean_degree"] = stats.mean_degree
    result["gc_fraction"] = stats.giant_component_fraction
    result["gamma_anchor"] = gamma_anchor
    result["arm"] = arm
    result["mean_gamma_i"] = float(gamma_arr.mean())

    # Observation layer (post-hoc; firewall: never inside transmission loop)
    lambda_obs_info = compute_lambda_obs(
        result["peak_incidence_per_100py"], gamma_anchor,
    )
    result["observation"] = lambda_obs_info

    return result


# ── Arm-0 falsification gate ──────────────────────────────────────────────────

def arm0_gate(proxies_dict: dict, seeds: list, n_agents: int = 300) -> dict:
    """Verify Arm 0 reproduces saturation on all seeds."""
    print("\n[Arm-0 gate] Verifying saturation (γ=0) on all seeds...")
    failures = []
    for proxy in ["ssp_concentrated_hotspot"]:
        for seed in seeds:
            r = one_run(proxy, seed, proxies_dict, arm=0, gamma_anchor=0.0, n_agents=n_agents)
            if r["final_size"] < 295:  # allow tiny stochastic variation near ceiling
                failures.append((proxy, seed, r["final_size"]))
                print(f"  FAIL: proxy={proxy} seed={seed} final_size={r['final_size']}")
            else:
                print(f"  pass: seed={seed} fs={r['final_size']} removed={r['total_removed']}")
    if failures:
        raise RuntimeError(
            f"Arm-0 gate FAILED on {len(failures)} runs — transmission streams may be crossed.\n"
            f"Failures: {failures}"
        )
    print("  Arm-0 gate PASSED: saturation confirmed.")
    return {"gate_passed": True, "failures": failures}


# ── Main sweep ────────────────────────────────────────────────────────────────

METRICS = [
    "final_size", "removed_while_S", "removed_while_I", "total_removed",
    "single_cluster_fraction", "doubling_time_days",
    "peak_incidence_per_100py", "mean_degree",
]


def aggregate(rows: list, gamma_anchor: float) -> dict:
    out = {m: ci95([r[m] for r in rows]) for m in METRICS}
    out["P_outbreak"] = p_outbreak([r["is_outbreak"] for r in rows])
    # Observation layer: mean λ_true and λ_obs across seeds
    lambda_trues = [r["peak_incidence_per_100py"] for r in rows]
    out["lambda_obs"] = compute_lambda_obs(
        float(np.mean([v for v in lambda_trues if v is not None])), gamma_anchor,
    )
    return out


def run_sweep(output_dir: str = "outputs", n_seeds: int = 20, seeds: list = None):
    if seeds is None:
        seeds = list(range(42, 42 + n_seeds))

    os.makedirs(output_dir, exist_ok=True)
    sha = _git_sha()
    proxies_dict = all_proxies()
    epidemic_days = P.N_BINS_PER_TILE * P.N_TILES_OUTBREAK * P.TIME_BIN_DAYS

    print("=" * 70)
    print("HANDOFF 10 — DESATURATION LADDER")
    print(f"  git_sha: {sha[:16]}  n_seeds={n_seeds}  seeds {seeds[0]}–{seeds[-1]}")
    print(f"  γ_low={GAMMA_LOW}/d (γτ={GAMMA_LOW*TAU_DAYS:.3f})  "
          f"γ_high={GAMMA_HIGH}/d (γτ={GAMMA_HIGH*TAU_DAYS:.3f})")
    print(f"  epidemic_days={epidemic_days}  n_steps_structure={P.N_STEPS_STRUCTURE}")
    for a, info in {"low": GAMMA_LOW * TAU_DAYS, "high": GAMMA_HIGH * TAU_DAYS}.items():
        env = check_envelope({"low": GAMMA_LOW, "high": GAMMA_HIGH}[a])
        print(f"  anchor_{a}: γτ={info:.4f}  envelope={env['status']}")
    print("=" * 70)

    # ── Arm-0 gate ────────────────────────────────────────────────────────────
    gate = arm0_gate(proxies_dict, seeds[:10])   # 10 seeds sufficient for gate

    # ── Full sweep ────────────────────────────────────────────────────────────
    results = {}  # {(proxy, arm, anchor_name): [rows]}

    for anchor_name, gamma_anchor in [("gamma_low", GAMMA_LOW), ("gamma_high", GAMMA_HIGH)]:
        for proxy in PROXIES:
            for arm in [0, 1, 2]:
                key = (proxy, arm, anchor_name)
                rows = []
                print(f"\n[{proxy}  arm={arm}  {anchor_name}={gamma_anchor}/d]  {n_seeds} seeds")
                for seed in seeds:
                    r = one_run(proxy, seed, proxies_dict, arm, gamma_anchor)
                    rows.append(r)
                    print(f"  s={seed}  fs={r['final_size']:3d}  "
                          f"rmS={r['removed_while_S']:3d}  rmI={r['removed_while_I']:3d}  "
                          f"out={r['is_outbreak']}  dt={r['doubling_time_days']}d")
                results[key] = rows

    # ── Aggregate and print tables ────────────────────────────────────────────
    agg = {}
    for (proxy, arm, anc), rows in results.items():
        ga = {"gamma_low": GAMMA_LOW, "gamma_high": GAMMA_HIGH}[anc]
        agg[(proxy, arm, anc)] = aggregate(rows, ga)

    for anchor_name in ["gamma_low", "gamma_high"]:
        print(f"\n{'═'*70}")
        print(f"  {anchor_name} ({ANCHORS[anchor_name]/1e-4:.0f}×10⁻⁴/d)  "
              f"γτ={ANCHORS[anchor_name]*TAU_DAYS:.4f}")
        print(f"{'═'*70}")
        for proxy in PROXIES:
            _print_arm_table(proxy, anchor_name, agg)

    # ── Rung-attributed verdict ───────────────────────────────────────────────
    verdicts = {}
    for anchor_name in ["gamma_low", "gamma_high"]:
        verdict = _compute_verdict(agg, anchor_name)
        verdicts[anchor_name] = verdict
        print(f"\n[Verdict  {anchor_name}]")
        print(f"  {verdict['verdict']}")
        print(f"  {verdict['note']}")

    # ── Validation ────────────────────────────────────────────────────────────
    from mobility.params import OUTBREAK_VALIDATION_TARGETS as OVT
    tgt_inc = OVT["explosive_hiv_incidence_per_100py"]["value"]
    tgt_cl  = OVT["cluster_dominance"]["scott_county_pct"] / 100.0
    # Use gamma_low Arm1 hotspot as the primary validation case
    val_rows = results[("ssp_concentrated_hotspot", 1, "gamma_low")]
    val_agg  = agg[("ssp_concentrated_hotspot", 1, "gamma_low")]
    validation = {
        "arm": 1,
        "anchor": "gamma_low",
        "proxy": "ssp_concentrated_hotspot",
        "final_size_mean": val_agg["final_size"]["mean"],
        "final_size_empirical": f"{OVT['final_size']['cabell_county_wv']}–"
                                f"{OVT['final_size']['scott_county_in']}",
        "peak_inc_true": val_agg["peak_incidence_per_100py"]["mean"],
        "peak_inc_obs":  val_agg["lambda_obs"].get("lambda_obs_per_100py"),
        "peak_inc_target_strathdee": tgt_inc,
        "peak_inc_pass": (val_agg["lambda_obs"].get("lambda_obs_per_100py") or 0) >= tgt_inc * 0.5,
        "single_cluster_mean": val_agg["single_cluster_fraction"]["mean"],
        "single_cluster_pass": (val_agg["single_cluster_fraction"]["mean"] or 0) >= tgt_cl,
    }
    print(f"\n[Validation — Arm1, γ_low, hotspot]")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    # ── Build artifact ────────────────────────────────────────────────────────
    def _agg_serializable(a):
        # convert defaultdict / np types
        return json.loads(json.dumps(a, default=lambda x: float(x) if hasattr(x, '__float__') else str(x)))

    artifact = {
        "git_sha": sha,
        "timestamp": datetime.now().isoformat(),
        "params": {
            "beta_syringe": P.BETA_SYRINGE_CHRONIC,
            "beta_env": P.BETA_ENV_CHRONIC,
            "acute_multiplier": P.ACUTE_MULTIPLIER,
            "n_steps_structure": P.N_STEPS_STRUCTURE,
            "n_tiles": P.N_TILES_OUTBREAK,
            "epidemic_days": epidemic_days,
            "gamma_low":  GAMMA_LOW,
            "gamma_high": GAMMA_HIGH,
            "tau_days":   TAU_DAYS,
        },
        "seeds": seeds,
        "arm0_gate": gate,
        "results": {
            f"{proxy}__arm{arm}__{anc}": _agg_serializable(agg[(proxy, arm, anc)])
            for proxy, arm, anc in agg
        },
        "verdicts": {k: v for k, v in verdicts.items()},
        "validation": validation,
    }

    out_path = os.path.join(output_dir, "desaturation_ladder.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote: {out_path}")
    return artifact


def _print_arm_table(proxy: str, anchor_name: str, agg: dict):
    print(f"\n  {proxy}  ({anchor_name})")
    print(f"  {'Metric':<30}  {'Arm0(γ=0)':>22}  {'Arm1(uniform-γ)':>22}  {'Arm2(γ(U))':>22}")
    metrics_labels = {
        "final_size": "final_size",
        "P_outbreak": "P(outbreak)",
        "removed_while_S": "removed_while_S",
        "removed_while_I": "removed_while_I",
        "single_cluster_fraction": "single_cluster_frac",
        "doubling_time_days": "doubling_time_days",
        "peak_incidence_per_100py": "peak_inc/100py",
        "mean_degree": "mean_degree",
    }
    for key, label in metrics_labels.items():
        row = f"  {label:<30}"
        for arm in [0, 1, 2]:
            k = (proxy, arm, anchor_name)
            if k not in agg:
                row += f"  {'—':>22}"
                continue
            v = agg[k]
            if key == "P_outbreak":
                s = f"{v['P_outbreak']:.3f}"
            else:
                vv = v.get(key, {})
                if isinstance(vv, dict) and vv.get("mean") is not None:
                    s = f"{vv['mean']:.1f}[{vv['ci95_lo']:.1f},{vv['ci95_hi']:.1f}]"
                else:
                    s = str(vv)
            row += f"  {s:>22}"
        print(row)

    # Observation layer for Arms 1 and 2
    for arm in [1, 2]:
        k = (proxy, arm, anchor_name)
        if k in agg:
            lobs = agg[k].get("lambda_obs", {})
            lt   = lobs.get("lambda_true_per_100py", "—")
            lo   = lobs.get("lambda_obs_per_100py", "—")
            env  = lobs.get("envelope_status", "—")
            print(f"  {'λ_true/λ_obs (Arm'+str(arm)+')':<30}  "
                  f"{'—':>22}  "
                  f"{str(lt)+'/'+str(lo)+' '+env:>22}" if arm == 1
                  else f"  {'λ_true/λ_obs (Arm'+str(arm)+')':<30}  "
                       f"{'—':>22}  {'—':>22}  "
                       f"{str(lt)+'/'+str(lo)+' '+env:>22}")


def _compute_verdict(agg: dict, anchor_name: str) -> dict:
    """Rung-attributed verdict for one anchor."""
    def _fs(proxy, arm):
        k = (proxy, arm, anchor_name)
        return (agg[k]["final_size"]["mean"] or 0) if k in agg else 0
    def _pi(proxy, arm):
        k = (proxy, arm, anchor_name)
        return (agg[k]["peak_incidence_per_100py"]["mean"] or 0) if k in agg else 0
    def _var(proxy, arm):
        k = (proxy, arm, anchor_name)
        return (agg[k]["final_size"]["var"] or 0) if k in agg else 0

    # Acceptance gate: variance returned in Arms 1 & 2
    var_returned = (
        _var("ssp_concentrated_hotspot", 1) > 0
        or _var("ssp_concentrated_hotspot", 2) > 0
    )

    # Arm0→Arm1: desaturation effect
    desat_hotspot = _fs("ssp_concentrated_hotspot", 0) - _fs("ssp_concentrated_hotspot", 1)
    desat_diffuse = _fs("ssp_diffuse_market", 0) - _fs("ssp_diffuse_market", 1)

    # Arm1→Arm2: concentration effect
    conc_hotspot = _fs("ssp_concentrated_hotspot", 1) - _fs("ssp_concentrated_hotspot", 2)
    conc_diffuse = _fs("ssp_diffuse_market", 1) - _fs("ssp_diffuse_market", 2)

    # Stage-C style comparison at Arm 1 (uniform removal, clean desaturation)
    fs_h1 = _fs("ssp_concentrated_hotspot", 1)
    fs_d1 = _fs("ssp_diffuse_market", 1)
    pi_h1 = _pi("ssp_concentrated_hotspot", 1)
    pi_d1 = _pi("ssp_diffuse_market", 1)

    conc_still_dominates = (
        pi_h1 > pi_d1 * 1.25 or
        _fs("ssp_concentrated_hotspot", 1) < _fs("ssp_diffuse_market", 1) - 20
    )

    if not var_returned:
        verdict = "STILL_SATURATED — variance did not return; γ too low to desaturate"
        note = (
            f"final_size variance ≈ 0 in Arms 1 & 2. γ={ANCHORS[anchor_name]}/d "
            f"is insufficient to desaturate the concentrated epidemic. "
            "Acceptance gate FAILED."
        )
    elif conc_still_dominates:
        verdict = "OUTCOME_1_SURVIVES — concentrated hotspot more explosive after desaturation"
        note = (
            f"Arm1 hotspot peak_inc={pi_h1:.0f} > diffuse×1.25={pi_d1*1.25:.0f}. "
            f"Desaturation (Arm0→Arm1: Δfs_hotspot={desat_hotspot:.1f}) uncovers "
            "geometry dominance. Outcome 2 (static-blind) does NOT survive full model."
        )
    else:
        verdict = "OUTCOME_2_SURVIVES — static-blind verdict survives desaturation"
        note = (
            f"After desaturation: hotspot peak_inc={pi_h1:.0f} ≈ diffuse={pi_d1:.0f} "
            f"(ratio={pi_h1/pi_d1:.3f} < 1.25). "
            f"Arm0→Arm1 Δfs: hotspot={desat_hotspot:.1f}, diffuse={desat_diffuse:.1f}. "
            f"Arm1→Arm2 Δfs: hotspot={conc_hotspot:.1f}, diffuse={conc_diffuse:.1f}. "
            "Outcome 2 (static structure does not predict outbreak risk) survives the ladder."
        )

    return {
        "verdict": verdict,
        "note": note,
        "var_returned": var_returned,
        "arm0_arm1_desat_hotspot": round(desat_hotspot, 2),
        "arm0_arm1_desat_diffuse": round(desat_diffuse, 2),
        "arm1_arm2_conc_hotspot": round(conc_hotspot, 2),
        "arm1_arm2_conc_diffuse": round(conc_diffuse, 2),
        "pi_hotspot_arm1": round(pi_h1, 2),
        "pi_diffuse_arm1": round(pi_d1, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    run_sweep(output_dir=args.output, n_seeds=args.seeds)
