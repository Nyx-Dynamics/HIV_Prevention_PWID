"""
Sourced transmission parameters for the temporal outbreak engine (Handoff 9).

All β values from published sources.  Never tuned to outbreak outcomes.
Time unit: steps = days.  time_bin=5 → 5-day venue-event window.

GUARDRAIL: do not modify β values to improve agreement with outbreak
validation targets.  The outbreak shape is the emergent, held-out output.
"""

# ── Layer 1 — direct syringe-sharing (per cap-contact at venue) ──────────────
# Each cap-contact at an event represents one sharing act (needle/syringe).
# Baggaley RF et al. Int J Epidemiol 35(5):1329 (2006). DOI 10.1093/ije/dyl176
# Per-act HIV transmission probability, injection route, chronic-phase source.
# NOTE: Rolls et al. 2011 1–3% is HCV — do NOT use for HIV.
BETA_SYRINGE_CHRONIC = 0.008   # HIV per sharing act, chronic source (Baggaley)

# ── Layer 2 — environmental / paraphernalia (per co-present viremic) ─────────
# Uncapped; applied to every viremic agent present at the event (not gated by c).
# Corson S et al. Drug Alcohol Depend 132(1-2):130 (2013).
# DOI 10.1016/j.drugalcdep.2013.05.014
# HCV paraphernalia β / needle β ≈ 0.19–0.30% / 2.5% ≈ 1/8.
# Ratio transfers to HIV environmental exposure (mechanism, not absolute HCV β).
ENV_SYRINGE_RATIO    = 1.0 / 8.0
BETA_ENV_CHRONIC     = BETA_SYRINGE_CHRONIC * ENV_SYRINGE_RATIO   # ≈ 0.001

# ── Stage-dependent transmissibility multipliers ──────────────────────────────
# Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501
# Acute-phase ×26 for ~90 days; late-stage ×7 (concentrated ~270 d pre-death).
ACUTE_MULTIPLIER       = 26.0
ACUTE_DURATION_DAYS    = 90
LATE_MULTIPLIER        = 7.0
LATE_START_DAYS        = 8 * 365   # ≈ 8 years; not reached in 1-year simulation

# ── Derived time-bin constants (time_bin = 5 days per venue event) ────────────
TIME_BIN_DAYS       = 5
N_STEPS_STRUCTURE   = 80               # same n_steps used for network calibration
N_BINS_PER_TILE     = N_STEPS_STRUCTURE // TIME_BIN_DAYS     # 16 bins per tile
N_TILES_OUTBREAK    = 10               # 10 × 16 bins × 5 days = 800 days (~26 mo)
ACUTE_DURATION_BINS = ACUTE_DURATION_DAYS // TIME_BIN_DAYS   # 18 time-bins
LATE_START_BINS     = LATE_START_DAYS  // TIME_BIN_DAYS      # 584  (unreachable)

# ── Static baseline (Stage A comparison) ─────────────────────────────────────
# Per-bin β for the collapsed static-graph SI model.
# Represents: β_chronic × injection_freq/day × days/bin × sharing_fraction
# Sources: Baggaley (β); Burnett 2018/Strathdee 1997 (inj_freq);
#          params.py derivation (shared_fraction ≈ 0.001 from NHBS forward).
INJECTION_FREQ_PER_DAY       = 3.0
SHARED_FRACTION_PER_PARTNER  = 0.001
BETA_STATIC_PER_BIN_CHRONIC  = (
    BETA_SYRINGE_CHRONIC
    * INJECTION_FREQ_PER_DAY
    * TIME_BIN_DAYS
    * SHARED_FRACTION_PER_PARTNER
)   # ≈ 0.000120

# ── Outbreak threshold ────────────────────────────────────────────────────────
# Min secondary cases to classify a run as an "outbreak" (vs. fade-out).
OUTBREAK_THRESHOLD = 10

# ── Source audit ──────────────────────────────────────────────────────────────
SOURCES = {
    "beta_syringe": (
        "Baggaley RF et al. Int J Epidemiol 35(5):1329 (2006). "
        "DOI 10.1093/ije/dyl176 — per-act HIV β, injection route."
    ),
    "env_ratio": (
        "Corson S et al. Drug Alcohol Depend 132(1-2):130 (2013). "
        "DOI 10.1016/j.drugalcdep.2013.05.014 — "
        "paraphernalia/needle HCV β ratio; mechanism transfers to HIV env layer."
    ),
    "stage_multipliers": (
        "Hollingsworth TQ et al. Nat Med 14:1096 (2008). "
        "DOI 10.1086/590501 — acute ×26, late ×7 vs chronic."
    ),
    "injection_freq": (
        "Burnett JC et al. MMWR 67(1) 2018. DOI 10.15585/mmwr.mm6701a5; "
        "Strathdee SA et al. AIDS 11(8) 1997. DOI 10.1097/00002030-199708000-00001."
    ),
    "shared_fraction": (
        "Calibrated forward from NHBS p_any=0.27 (Burnett 2018) ÷ "
        "(injection_freq × mean_degree). See src/mobility/params.py derivation."
    ),
    "acute_coupling": (
        "Eaton JW et al. AIDS 24(Suppl 1):S29 (2010). "
        "DOI 10.1007/s10461-010-9787-8 — concurrency amplifies when "
        "infectiousness is transiently elevated (acute-window coincidence)."
    ),
    "dose_dependence": (
        "Takaguchi Y et al. PLoS ONE 8(7):e68629 (2013). "
        "DOI 10.1371/journal.pone.0068629; "
        "Unicomb S et al. Nat Commun 12:554 (2021). "
        "DOI 10.1038/s41467-020-20398-4 — "
        "dose-dependent acquisition at bursty events."
    ),
}
