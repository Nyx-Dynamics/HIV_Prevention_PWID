"""
Agent-level removal (γ) mechanics for the desaturation ladder (Handoff 10).

γ_i = per-day hazard of structural censoring: overdose, incarceration,
displacement.  Sourced from JAIDS manuscript "Calibration-to-Deployment
Mismatch" (Demidont), Eq. 2–3 (Ω*(γ)), Eq. 10 (γ_city), Table S4 PWID rows.

Arm assignments:
  Arm 0  — γ=0.     Falsification anchor (must reproduce H9 saturation).
  Arm 1  — uniform  γ_anchor for all agents.
  Arm 2  — agent-level γ(U_i), mean-matched to γ_anchor.

Mean-matching is mandatory: population-mean γ_i = γ_anchor exactly (to
numerical tolerance) so that Arm 1 → Arm 2 isolates only the *variance /
concentration* of γ, never a shift in total expected removal.

Canonical numerics (spec-of-record):
  Nyx-Dynamics/nyx-kassanjee-letter:
    compute_covid_deficit.py, kassanjee_invariance_test.py
  Future reconciliation should verify γ/Ω* numerics against those files
  line-for-line rather than re-deriving.

GUARDRAIL: do NOT touch the network generator or κ/critical_network_threshold.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ── Published anchors (Table S4, Demidont JAIDS) ──────────────────────────────
GAMMA_LOW  = 12e-4   # /day  "PWID US (PURPOSE 4 projected)", r=0.75
GAMMA_HIGH = 20e-4   # /day  "PWID severe structural (Hartford-like)", r=0.65
TAU_DAYS   = 173     # days  Sedia LAg-EIA exponential constant (Supplement §S1.1)

# γτ products used in envelope checks (§5, main letter)
GAMMA_TAU_LOW  = round(GAMMA_LOW  * TAU_DAYS, 6)   # ≈ 0.2076  — inside envelope
GAMMA_TAU_HIGH = round(GAMMA_HIGH * TAU_DAYS, 6)   # ≈ 0.3460  — outside envelope

ANCHORS = {
    "gamma_low":  GAMMA_LOW,
    "gamma_high": GAMMA_HIGH,
}


# ── Arm assignment functions ───────────────────────────────────────────────────

def assign_gamma_zero(n_agents: int) -> np.ndarray:
    """Arm 0: γ=0 for all agents (falsification anchor)."""
    return np.zeros(n_agents)


def assign_gamma_uniform(n_agents: int, gamma_anchor: float) -> np.ndarray:
    """Arm 1: every agent carries identical γ_anchor per day."""
    return np.full(n_agents, float(gamma_anchor))


def assign_gamma_heterogeneous(
    agent_rgs: np.ndarray,
    gamma_anchor: float,
) -> np.ndarray:
    """
    Arm 2: agent-level γ(U_i) ∝ 1/rg_i, mean-matched to γ_anchor.

    U proxy: 1/rg_i.  Low-mobility agents (small rg) have more concentrated
    activity at fixed venues, higher structural exposure to incarceration/
    overdose — therefore higher γ.  The high-γ concentration at the hotspot
    *emerges* from where these agents co-locate; it is not imposed on the venue.

    γ is a property of the person's structural position, NOT the venue (JAIDS
    ms Eq. 10 framing).  Rescaling ensures mean(γ_i) = γ_anchor exactly so
    that Arm 1 → Arm 2 measures only differential loading, not more removal.

    Sources: JAIDS ms Eq. 10 (γ_city); structural-functions framing (ms §2).
    Canonical: Nyx-Dynamics/nyx-kassanjee-letter.
    """
    rgs = np.asarray(agent_rgs, dtype=float)
    if np.any(rgs <= 0):
        raise ValueError("All agent rg values must be > 0 for Arm-2 γ assignment.")
    raw = 1.0 / rgs
    normed = raw / raw.mean()          # E[normed] = 1.0 exactly
    return gamma_anchor * normed       # E[γ_i] = γ_anchor exactly


def verify_mean_match(
    gamma_i: np.ndarray,
    gamma_anchor: float,
    tol: float = 1e-10,
) -> dict:
    """
    Acceptance gate for Arm 2: confirm mean(γ_i) == γ_anchor to tolerance.
    Returns a dict with pass/fail and the measured error.
    """
    diff = float(abs(gamma_i.mean() - gamma_anchor))
    return {
        "pass": diff < tol,
        "mean_gamma_i": float(gamma_i.mean()),
        "gamma_anchor": float(gamma_anchor),
        "abs_diff": diff,
        "tol": tol,
    }


def removal_p_per_bin(gamma_per_day: float, time_bin_days: int) -> float:
    """P(removed in one time-bin) = 1 − exp(−γ × bin_days)."""
    return float(1.0 - np.exp(-gamma_per_day * time_bin_days))
