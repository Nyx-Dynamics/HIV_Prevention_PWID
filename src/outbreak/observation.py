"""
Observation layer: Ω*(γ) incidence deflation (Handoff 10).

══════════════════════════════════════════════════════════════════════
ARCHITECTURAL FIREWALL
  This module must ONLY be called AFTER the dynamics are complete.
  It must NEVER be imported or called inside the transmission loop.
  Applying Ω*(γ) inside the dynamics bakes measurement bias into the
  mechanism — the exact circularity this program is built to avoid.
  The dynamics produce TRUE incidence (already reduced by removal).
  The observation layer produces the *estimated* incidence a RITA-based
  estimator would report, for estimator-to-estimator field comparison.
══════════════════════════════════════════════════════════════════════

Deflation formula — exponential basis (Eq. 3 / Eq. 5, Demidont JAIDS):
  λ_obs = λ_true × (Ω*/Ω) = λ_true / (1 + γτ)

  τ = 173 days (Sedia LAg-EIA constant, Supplement §S1.1)
  γ = per-day structural-censoring hazard

  BASIS DISCIPLINE: 1/(1+γτ) is the incidence/b_HIV deflation (exponential
  basis, Eq. 3).  It is NOT exp(−γτ/2), which is ρ_screen for the IRR
  machinery (uniform basis, Eq. 7).  These are different.  Do not mix.

Envelope gate (§4.5, main letter):
  Apply ONLY when γτ ∈ [0.087, 0.273].
  γ_low  anchor: γτ = 0.208 → IN  envelope; apply deflation.
  γ_high anchor: γτ = 0.346 → OUT of envelope; emit OUT_OF_ENVELOPE flag.

Canonical numerics (spec-of-record):
  Nyx-Dynamics/nyx-kassanjee-letter (compute_covid_deficit.py,
  kassanjee_invariance_test.py).  Future reconciliation should verify
  against those files line-for-line.
"""

from __future__ import annotations

TAU_DAYS    = 173.0     # Sedia LAg-EIA exponential constant (Supplement §S1.1)
ENVELOPE_LO = 0.087     # lower bound of well-characterized deflation envelope
ENVELOPE_HI = 0.273     # upper bound


def check_envelope(gamma_per_day: float, tau: float = TAU_DAYS) -> dict:
    """
    Return envelope status for a given γ.

    Returns
    -------
    dict with keys: gamma_tau, in_envelope, status ("IN_ENVELOPE" |
    "OUT_OF_ENVELOPE"), envelope_range.
    """
    gt = gamma_per_day * tau
    in_env = ENVELOPE_LO <= gt <= ENVELOPE_HI
    return {
        "gamma_tau": round(gt, 6),
        "in_envelope": in_env,
        "status": "IN_ENVELOPE" if in_env else "OUT_OF_ENVELOPE",
        "envelope_range": f"[{ENVELOPE_LO}, {ENVELOPE_HI}]",
    }


def compute_lambda_obs(
    lambda_true_per_100py: float,
    gamma_per_day: float,
    tau: float = TAU_DAYS,
) -> dict:
    """
    Return λ_true and, where γτ is in-envelope, λ_obs = λ_true / (1 + γτ).

    FIREWALL: call only after dynamics are complete, never inside the loop.

    Parameters
    ----------
    lambda_true_per_100py : true model incidence (per 100 person-years)
    gamma_per_day         : effective enrolled hazard for this run/arm

    Returns
    -------
    dict with lambda_true, lambda_obs (None if out-of-envelope), envelope
    status, deflation factor, and an audit note citing Eq. 3.
    """
    env = check_envelope(gamma_per_day, tau)
    gt  = env["gamma_tau"]

    result = {
        "lambda_true_per_100py": round(float(lambda_true_per_100py), 4),
        "gamma_per_day":         float(gamma_per_day),
        "gamma_tau":             gt,
        "envelope_status":       env["status"],
        "tau_days":              tau,
    }

    if env["in_envelope"]:
        deflation = 1.0 / (1.0 + gt)
        result["lambda_obs_per_100py"]  = round(float(lambda_true_per_100py) * deflation, 4)
        result["deflation_factor"]       = round(deflation, 6)
        result["note"] = (
            f"λ_obs = λ_true / (1 + γτ) = λ_true / {1.0 + gt:.4f} "
            f"≈ λ_true × {deflation:.4f}. "
            "Exponential basis, Eq. 3 / Eq. 5 (Demidont JAIDS). "
            "Canonical: Nyx-Dynamics/nyx-kassanjee-letter."
        )
    else:
        result["lambda_obs_per_100py"] = None
        result["deflation_factor"]      = None
        result["note"] = (
            f"OUT_OF_ENVELOPE: γτ = {gt:.4f} outside [{ENVELOPE_LO}, {ENVELOPE_HI}]. "
            "Estimator-bias sign is not guaranteed in this regime. "
            "Clean deflation is not well-posed. Report λ_true only. "
            "Canonical: Nyx-Dynamics/nyx-kassanjee-letter."
        )

    return result
