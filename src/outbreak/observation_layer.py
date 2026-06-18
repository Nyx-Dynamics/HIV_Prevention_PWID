"""
Canonical observation layer for incidence-estimator bias correction (Handoff 11).

══════════════════════════════════════════════════════════════════════
CANONICAL STATUS
  This module is the first executable implementation of Eq. 3 / Eq. 5
  (Ω*/Ω = 1/(1+γτ)) in the Nyx codebase.
  Future revisions of Nyx-Dynamics/nyx-kassanjee-letter should
  RECONCILE AGAINST THIS FILE, not the reverse.
  Cross-repo reconciliation target:
    compute_covid_deficit.py, kassanjee_invariance_test.py
    (Nyx-Dynamics/nyx-kassanjee-letter)
══════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════
ARCHITECTURAL FIREWALL
  Ω*(γ) deflation is applied ONLY post-hoc to emitted incidence rates.
  It must NEVER be called inside the transmission loop.
  Dynamics produce TRUE incidence (already reduced by γ removal).
  This module produces the *estimated* rate a RITA-based estimator
  would report — for estimator-to-estimator comparison with field data.
══════════════════════════════════════════════════════════════════════

Deflation formula — exponential basis (Eq. 3 / Eq. 5, Demidont JAIDS):
  λ_obs = λ_true × (Ω*/Ω) = λ_true / (1 + γτ)

  BASIS DISCIPLINE: 1/(1+γτ) is the incidence/b_HIV deflation.
  It is NOT exp(−γτ/2) (ρ_screen for IRR machinery, uniform basis, Eq. 7).
  Apply to PERIOD incidence (W1, Strathdee-commensurable); NOT to peak rate.

Envelope gate (§4.5, main letter):
  Apply ONLY when γτ ∈ [0.087, 0.273].
  γ_low  (12×10⁻⁴/d): γτ = 0.208 → IN_ENVELOPE  → apply deflation.
  γ_high (20×10⁻⁴/d): γτ = 0.346 → OUT_OF_ENVELOPE → flag, no deflation.
  γ_high is deliberately supra-empirical (above any real MSA in Eq. 10's
  range). The OUT_OF_ENVELOPE flag firing is the design confirming itself.

kassanjee_factor anchor:
  γ_low  ≈ 12×10⁻⁴/d corresponds to late-dx ≈ 30% city (near top of
  Eq.-10-reachable MSA range).  γ_high ≈ 20×10⁻⁴/d ≫ Hartford 37.2%
  late-dx → 15.4×10⁻⁴/d.

STALE-CLONE CAVEAT (decision-log record):
  kassanjee_invariance_test.py imports run_policy_iteration from
  ~/gitrepo/HIV_Prevention_PWID (months-stale main-only clone from the
  git-recovery episode).  Invariance numbers (ρ=0.9979, 34/34) were
  generated against that stale mdp_engine — fine for the letter (MDP ≠
  outbreak path), but any regeneration of the invariance test MUST
  repoint COMPANION_REPO to the air/ clone or the pushed remote, or the
  two papers could silently diverge on the MDP version.  Flag this in the
  Corner-4 decision log before next regeneration.
"""

from __future__ import annotations

TAU_DAYS    = 173.0
ENVELOPE_LO = 0.087
ENVELOPE_HI = 0.273

# ── kassanjee_factor constants — pinned to kassanjee_invariance_test.py ───────
# γ_city = SELECTION_AMP × (late_dx / LATE_DX_NATIONAL)^ALPHA × GAMMA_BASE
# DO NOT re-derive α or the amplification factor.
GAMMA_BASE          = 5e-4
ALPHA               = 1.2
SELECTION_AMP       = 1.5
LATE_DX_NATIONAL    = 20.0   # national late-dx % reference (Demidont Eq. 10)


def kassanjee_factor(late_dx_pct: float) -> float:
    """
    γ_city per day from city-level late-dx prevalence (Eq. 10, Demidont JAIDS).

    Pinned to kassanjee_invariance_test.py (Nyx-Dynamics/nyx-kassanjee-letter).
    Constants: GAMMA_BASE=5e-4, ALPHA=1.2, SELECTION_AMP=1.5,
               LATE_DX_NATIONAL=20.0.  Do not re-derive.

    Parameters
    ----------
    late_dx_pct : city late-diagnosis percentage (e.g., 30.0 for 30%)

    Returns
    -------
    γ_city per day
    """
    return SELECTION_AMP * (late_dx_pct / LATE_DX_NATIONAL) ** ALPHA * GAMMA_BASE


def check_envelope(gamma_per_day: float, tau: float = TAU_DAYS) -> dict:
    """Envelope status for γ (whether Eq. 3 deflation is well-posed)."""
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
    note: str = "",
) -> dict:
    """
    Compute λ_obs = λ_true / (1 + γτ) where γτ is in-envelope.

    Apply ONLY to W1 period incidence (Strathdee-commensurable).
    Never to peak instantaneous rate.

    FIREWALL: call only after dynamics are complete.
    """
    env = check_envelope(gamma_per_day, tau)
    gt  = env["gamma_tau"]

    result = {
        "lambda_true_per_100py":  round(float(lambda_true_per_100py), 4),
        "gamma_per_day":          float(gamma_per_day),
        "gamma_tau":              gt,
        "envelope_status":        env["status"],
        "tau_days":               tau,
    }

    if env["in_envelope"]:
        deflation = 1.0 / (1.0 + gt)
        result["lambda_obs_per_100py"] = round(float(lambda_true_per_100py) * deflation, 4)
        result["deflation_factor"]      = round(deflation, 6)
        result["note"] = (
            f"λ_obs = λ_true / (1+γτ) = λ_true / {1.0+gt:.4f} ≈ λ_true × {deflation:.4f}. "
            "Eq. 3 / Eq. 5 (Demidont JAIDS), exponential basis. "
            "Canonical: Nyx-Dynamics/nyx-kassanjee-letter. " + note
        )
    else:
        result["lambda_obs_per_100py"] = None
        result["deflation_factor"]      = None
        result["note"] = (
            f"OUT_OF_ENVELOPE: γτ = {gt:.4f} outside [{ENVELOPE_LO}, {ENVELOPE_HI}]. "
            "γ_high is deliberately supra-empirical — above what Eq. 10 maps for any "
            "real MSA. This flag firing confirms the design, not a defect. "
            "Report λ_true only; no deflation applied. " + note
        )

    return result
