"""
Temporal SI outbreak engine (Handoff 9).

Two-layer per-event transmission:
  Layer 1 — direct syringe-sharing: capped (≤ contact_cap partners), high β.
             Each cap-contact represents one sharing act at the venue.
             β = BETA_SYRINGE_CHRONIC, stage-adjusted.
  Layer 2 — environmental/paraphernalia: uncapped, low β (Corson ratio 1/8).
             Applied to every viremic co-present agent.
             β = BETA_ENV_CHRONIC, stage-adjusted.

Set beta_env=0 for Stage A (Layer 1 only; tests timing/burstiness alone).

Per-event acquisition (dose-dependent, saturating — Takaguchi 2013; Unicomb 2021):
  p_acq = 1 − ∏(1−β_s,i) × ∏(1−β_e,j)
  where i ∈ viremic direct partners, j ∈ all viremic co-present.

Acute coupling (Eaton-Hallett-Garnett 2010): each source's β is multiplied
by ACUTE_MULTIPLIER for t_since_infection < ACUTE_DURATION_BINS.  The pulse
is dangerous only when it coincides with a source's acute window.

GUARDRAIL: β values are sourced and fixed.  Do not modify to fit outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Stage-adjusted β helper ───────────────────────────────────────────────────

def _stage_beta(
    beta_base: float,
    t_since_bins: int,
    acute_dur_bins: int,
    late_start_bins: int,
    acute_mult: float,
    late_mult: float,
) -> float:
    """Return stage-adjusted per-act β for a source at t_since_bins post-infection."""
    if t_since_bins < 0:
        return 0.0
    if t_since_bins < acute_dur_bins:
        return min(beta_base * acute_mult, 1.0)
    if t_since_bins >= late_start_bins:
        return min(beta_base * late_mult, 1.0)
    return beta_base


# ── Metrics helper ────────────────────────────────────────────────────────────

def _metrics(
    infection_bin: Dict[int, int],
    new_by_bin: Dict[int, int],
    n_agents: int,
    time_bin_days: int,
    outbreak_threshold: int,
) -> Dict:
    final_size = len(infection_bin) - 1   # secondary only (exclude seed)
    is_outbreak = final_size >= outbreak_threshold

    times = sorted(new_by_bin.keys())

    # Doubling time: bins from first secondary until cumulative ≥ 2
    doubling_time_days: Optional[float] = None
    if times:
        cumulative = 0
        t_first = times[0]
        for t in times:
            cumulative += new_by_bin[t]
            if cumulative >= 2:
                doubling_time_days = float((t - t_first) * time_bin_days)
                break

    # Peak incidence per 100 person-years (cases/bin / n_agents × 365/bin_days × 100)
    peak_incidence_per_100py = 0.0
    if new_by_bin:
        peak_new = max(new_by_bin.values())
        peak_incidence_per_100py = (peak_new / n_agents) * (365.0 / time_bin_days) * 100.0

    # Single-cluster fraction: under SI with one seed all cases form one chain;
    # report 1.0 when there are cases, 0.0 otherwise (matches empirical ≥93%).
    single_cluster_fraction = 1.0 if final_size > 0 else 0.0

    return {
        "final_size": final_size,
        "is_outbreak": is_outbreak,
        "single_cluster_fraction": single_cluster_fraction,
        "doubling_time_days": doubling_time_days,
        "peak_incidence_per_100py": peak_incidence_per_100py,
    }


# ── Temporal model ────────────────────────────────────────────────────────────

def run_temporal(
    event_stream: List[Tuple[int, Tuple, List[int]]],
    n_agents: int,
    seed_agent: int,
    contact_cap: int,
    beta_syringe: float,
    beta_env: float,
    acute_dur_bins: int,
    late_start_bins: int,
    acute_mult: float,
    late_mult: float,
    rng: np.random.Generator,
    time_bin_days: int = 5,
    outbreak_threshold: int = 10,
) -> Dict:
    """
    Run SI outbreak on the time-ordered event stream.

    Parameters
    ----------
    event_stream    : from build_event_stream()
    seed_agent      : agent_id of initial infection (t=0)
    beta_syringe    : per-cap-contact β at event (Layer 1; Baggaley chronic base)
    beta_env        : per-co-present-viremic β (Layer 2; 0 → Stage A)
    """
    infection_bin: Dict[int, int] = {seed_agent: 0}
    new_by_bin: Dict[int, int] = defaultdict(int)

    for t_bin, _cell, agents in event_stream:
        m = len(agents)
        if m < 2:
            continue

        infected_here = [(a, infection_bin[a]) for a in agents if a in infection_bin]
        if not infected_here:
            continue

        susceptibles = [a for a in agents if a not in infection_bin]
        if not susceptibles:
            continue

        # Stage-adjusted β per source
        src_b1 = []  # (src, beta_syringe_adjusted)
        src_b2 = []  # (src, beta_env_adjusted)
        for src, t_inf in infected_here:
            t_since = t_bin - t_inf
            src_b1.append(
                (src, _stage_beta(beta_syringe, t_since, acute_dur_bins,
                                  late_start_bins, acute_mult, late_mult))
            )
            if beta_env > 0:
                src_b2.append(
                    (src, _stage_beta(beta_env, t_since, acute_dur_bins,
                                      late_start_bins, acute_mult, late_mult))
                )

        new_this_bin: List[int] = []
        for susc in susceptibles:
            others = [a for a in agents if a != susc]
            n_sample = min(contact_cap, len(others))
            sampled_set = set(int(x) for x in
                              rng.choice(others, size=n_sample, replace=False))

            # Log-survival (numerically stable via log1p)
            log_surv = 0.0
            for src, b in src_b1:
                if src in sampled_set and b > 0.0:
                    log_surv += np.log1p(-b)
            for src, b in src_b2:
                if b > 0.0:
                    log_surv += np.log1p(-b)

            if log_surv < 0.0:
                p_acq = -np.expm1(log_surv)   # = 1 - exp(log_surv), stable
                if rng.random() < p_acq:
                    new_this_bin.append(susc)

        for a in new_this_bin:
            if a not in infection_bin:
                infection_bin[a] = t_bin
                new_by_bin[t_bin] += 1

    return _metrics(infection_bin, new_by_bin, n_agents, time_bin_days, outbreak_threshold)


# ── Static baseline (Stage A comparison) ─────────────────────────────────────

def run_static(
    contact_edges: List[Tuple[int, int]],
    n_agents: int,
    seed_agent: int,
    n_bins: int,
    beta_static_per_bin: float,
    acute_dur_bins: int,
    late_start_bins: int,
    acute_mult: float,
    late_mult: float,
    rng: np.random.Generator,
    time_bin_days: int = 5,
    outbreak_threshold: int = 10,
) -> Dict:
    """
    Static-graph SI baseline for Stage A comparison.

    At each time-bin every susceptible acquires from ALL infected neighbours
    with compound dose-dependent probability.  β_static_per_bin is the
    per-bin population-average sharing rate (Baggaley × inj_freq × bin_days
    × shared_fraction).  No concurrency — pairwise sequential.

    The contrast with run_temporal() isolates the pure timing/burstiness
    effect: same expected β per dyad, but temporal contacts are bursty
    (event-driven co-presence) while static contacts are uniform per-bin.
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in contact_edges:
        adj[a].append(b)
        adj[b].append(a)

    infection_bin: Dict[int, int] = {seed_agent: 0}
    new_by_bin: Dict[int, int] = defaultdict(int)

    for t_bin in range(n_bins):
        new_this_bin: List[int] = []
        for agent in range(n_agents):
            if agent in infection_bin:
                continue
            log_surv = 0.0
            for nbr in adj[agent]:
                if nbr not in infection_bin:
                    continue
                t_since = t_bin - infection_bin[nbr]
                b = _stage_beta(beta_static_per_bin, t_since, acute_dur_bins,
                                late_start_bins, acute_mult, late_mult)
                if b > 0.0:
                    log_surv += np.log1p(-b)
            if log_surv < 0.0:
                p_acq = -np.expm1(log_surv)
                if rng.random() < p_acq:
                    new_this_bin.append(agent)

        for a in new_this_bin:
            if a not in infection_bin:
                infection_bin[a] = t_bin
                new_by_bin[t_bin] += 1

    return _metrics(infection_bin, new_by_bin, n_agents, time_bin_days, outbreak_threshold)


# ── Temporal model with γ removal (Handoff 10 — desaturation ladder) ─────────

def run_temporal_with_gamma(
    event_stream,
    n_agents: int,
    seed_agent: int,
    contact_cap: int,
    beta_syringe: float,
    beta_env: float,
    gamma_array,                      # np.ndarray, per-agent γ/day; zeros → Arm 0
    relocation_fraction: float,       # = 0 for all H10 runs (switch for future)
    acute_dur_bins: int,
    late_start_bins: int,
    acute_mult: float,
    late_mult: float,
    rng_transmission,                 # acquisition draws ONLY
    rng_removal,                      # removal draws ONLY — never crossed
    time_bin_days: int = 5,
    outbreak_threshold: int = 10,
) -> dict:
    """
    SI outbreak with terminal agent removal (γ structural-censoring).

    RNG ARCHITECTURE (§3, Handoff 10):
      rng_transmission — all acquisition/transmission draws.
      rng_removal      — removal draws ONLY; never touched by rng_transmission.
      Arm 0 (γ=0): rng_removal is NEVER drawn from → transmission draws are
      byte-for-byte identical to Arms 1/2 on the same SeedSequence root.
      Verify: Arm 0 final_size == event-stream baseline (299/300) on each seed.

    Removal mechanics:
      At the start of each time-bin, each non-removed agent is tested for
      removal: p = 1 − exp(−γ_i × time_bin_days), drawn from rng_removal.
      Removal is terminal and serostatus-blind (affects S and I equally).
      Removed agents are excluded from all subsequent events.

    relocation_fraction: 0 for all H10 runs.  Switch exposes the parameter
    for future displacement/bridge modelling without re-architecting.

    Additional returned metrics:
      removed_while_S  — susceptibles removed before infection (pool dilution)
      removed_while_I  — infected removed (source extinction)
      total_removed    — total removals
    """
    import numpy as _np
    from collections import defaultdict as _dd

    if relocation_fraction != 0.0:
        raise NotImplementedError("relocation_fraction > 0 is reserved for future builds.")

    gamma_arr = _np.asarray(gamma_array, dtype=float)
    any_removal = bool(_np.any(gamma_arr > 0))
    p_rm = _np.array([
        float(1.0 - _np.exp(-g * time_bin_days)) if g > 0 else 0.0
        for g in gamma_arr
    ])

    # Build time-bin indexed event lookup
    events_by_t: dict = _dd(list)
    max_t_bin = 0
    for t_bin, cell, agents in event_stream:
        events_by_t[t_bin].append((cell, agents))
        if t_bin > max_t_bin:
            max_t_bin = t_bin

    infection_bin: dict = {seed_agent: 0}
    new_by_bin: dict = _dd(int)
    removed_set: set = set()
    removed_while_S: set = set()
    removed_while_I: set = set()

    for t_bin in range(max_t_bin + 1):

        # ── Removal phase (rng_removal only; Arm 0 skips entirely) ────────────
        if any_removal:
            draws = rng_removal.random(n_agents)
            for agent in range(n_agents):
                if agent in removed_set:
                    continue
                if p_rm[agent] > 0.0 and draws[agent] < p_rm[agent]:
                    removed_set.add(agent)
                    if agent in infection_bin:
                        removed_while_I.add(agent)
                    else:
                        removed_while_S.add(agent)

        # ── Transmission phase (rng_transmission only) ────────────────────────
        for _cell, agents in events_by_t[t_bin]:
            active = [a for a in agents if a not in removed_set]
            m = len(active)
            if m < 2:
                continue

            infected_here = [(a, infection_bin[a]) for a in active if a in infection_bin]
            if not infected_here:
                continue
            susceptibles = [a for a in active if a not in infection_bin]
            if not susceptibles:
                continue

            src_b1 = []
            src_b2 = []
            for src, t_inf in infected_here:
                t_since = t_bin - t_inf
                src_b1.append(
                    (src, _stage_beta(beta_syringe, t_since, acute_dur_bins,
                                      late_start_bins, acute_mult, late_mult))
                )
                if beta_env > 0:
                    src_b2.append(
                        (src, _stage_beta(beta_env, t_since, acute_dur_bins,
                                          late_start_bins, acute_mult, late_mult))
                    )

            new_this_bin = []
            for susc in susceptibles:
                others = [a for a in active if a != susc]
                n_sample = min(contact_cap, len(others))
                sampled_set = set(int(x) for x in
                                  rng_transmission.choice(others, size=n_sample, replace=False))

                log_surv = 0.0
                for src, b in src_b1:
                    if src in sampled_set and b > 0.0:
                        log_surv += _np.log1p(-b)
                for src, b in src_b2:
                    if b > 0.0:
                        log_surv += _np.log1p(-b)

                if log_surv < 0.0:
                    p_acq = -_np.expm1(log_surv)
                    if rng_transmission.random() < p_acq:
                        new_this_bin.append(susc)

            for a in new_this_bin:
                if a not in infection_bin and a not in removed_set:
                    infection_bin[a] = t_bin
                    new_by_bin[t_bin] += 1

    base = _metrics(infection_bin, new_by_bin, n_agents, time_bin_days, outbreak_threshold)
    base["removed_while_S"] = len(removed_while_S)
    base["removed_while_I"] = len(removed_while_I)
    base["total_removed"]   = len(removed_set)
    return base
