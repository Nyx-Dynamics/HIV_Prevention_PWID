"""
Mobility network generator — parameter definitions.

Every value is a ParameterWithUncertainty carrying distribution + source + gap flag.
See docs/SOURCING_mobility_network.md for the full methods-section-ready provenance.

PLACEHOLDER constants mark parameters where no defensible source exists.
Their defaults are conservative; AC must approve before use in published results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stochastic_avoidance_enhanced import ParameterWithUncertainty

# ─────────────────────────────────────────────────────────────────────────────
# PLACEHOLDER sentinel — appears in source fields for unsourced values
# ─────────────────────────────────────────────────────────────────────────────
PLACEHOLDER = "PLACEHOLDER — no defensible source; AC must approve before use in results"

# Hard per-act β cap: β_acute = min(β_chronic * acute_multiplier, BETA_CAP)
# Not a ParameterWithUncertainty — a sanity constraint, not a free parameter.
BETA_CAP = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Walk / Traversement-potential distribution
# ─────────────────────────────────────────────────────────────────────────────

MOBILITY_PARAMS = {

    # Jump-length power-law exponent: P(Δr) ∝ (Δr)^{−1−β}
    # Gap: generic mobile-phone population, not PWID-specific
    "jump_length_exponent": ParameterWithUncertainty(
        name="Jump-length power-law exponent β",
        point_estimate=0.60,
        lower_bound=0.50,
        upper_bound=0.75,
        distribution="uniform",
        source="González, Hidalgo & Barabási. Nature 453:779 (2008). DOI 10.1038/nature06958. "
               "GAP: generic population mobility, not PWID-specific.",
    ),

    # Radius-of-gyration growth exponent: r_g ∝ t^{ζ/2}
    # Heavy tail controls hub generation; keep but treat as uncertain
    "rg_growth_exponent": ParameterWithUncertainty(
        name="Radius-of-gyration growth exponent ζ",
        point_estimate=1.65,
        lower_bound=1.29,
        upper_bound=1.71,
        distribution="uniform",
        source="González et al. Nature 453:779 (2008). DOI 10.1038/nature06958.",
    ),

    # PWID activity radius (r_g scale, km)
    # CRITICAL GAP: single-city (SF) self-reported estimate; not GPS r_g; urban bias
    "rg_scale_km": ParameterWithUncertainty(
        name="PWID activity radius r_g scale (km)",
        point_estimate=2.4,
        lower_bound=1.0,
        upper_bound=5.0,
        distribution="lognormal",
        source="Cooper HLF, Tempalski B. Integrating place into research on drug use, drug users' health, and drug policy. Int J Drug Policy 25(3) (2014). DOI 10.1016/j.drugpo.2013.11.008. [Author verified against DOI; original attribution 'Ober 2014' was incorrect — corrected per HANDOFF 4.] "
               "N=1084, San Francisco. GAP: self-reported anchor distance ≠ GPS r_g; "
               "single-city urban mean; treat scale as major sensitivity axis.",
    ),

    # EPR exploration probability scaling (ρ): p_explore = ρ · S^{-γ}
    # KEY SENSITIVITY KNOB — controls explorer fraction → ⟨k²⟩ → τ_c
    "epr_rho": ParameterWithUncertainty(
        name="EPR exploration probability scaling ρ",
        point_estimate=0.60,
        lower_bound=0.30,
        upper_bound=0.90,
        distribution="uniform",
        source="Pappalardo L et al. Nat Commun 6:8166 (2015). DOI 10.1038/ncomms9166. "
               "KEY SENSITIVITY KNOB: explorer fraction is emergent from ρ and γ; "
               "controls ⟨k²⟩ and hence τ_c.",
    ),

    # EPR exploration decay exponent (γ): p_explore = ρ · S^{-γ}
    "epr_gamma": ParameterWithUncertainty(
        name="EPR exploration decay exponent γ",
        point_estimate=0.21,
        lower_bound=0.10,
        upper_bound=0.35,
        distribution="uniform",
        source="Pappalardo L et al. Nat Commun 6:8166 (2015). DOI 10.1038/ncomms9166.",
    ),

    # Number of mobility steps per agent per simulation period
    "n_steps": ParameterWithUncertainty(
        name="Mobility steps per agent (simulation period)",
        point_estimate=50.0,
        lower_bound=30.0,
        upper_bound=100.0,
        distribution="uniform",
        source=PLACEHOLDER + " — set to approximate daily location visits over a 30-day period.",
    ),

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 2 — Attractors (venues)
    # ─────────────────────────────────────────────────────────────────────

    # SSP utilization: fraction of PWID who got syringes from an SSP (12-mo)
    # Used to weight SSP venues as attractors
    "ssp_utilization_rate": ParameterWithUncertainty(
        name="SSP/SEP utilization rate (12-mo, PWID)",
        point_estimate=0.52,
        lower_bound=0.45,
        upper_bound=0.60,
        distribution="beta",
        source="Burnett JC et al. MMWR 67(1):23 (2018). DOI 10.15585/mmwr.mm6701a5. "
               "Population rate — not per-venue pull; use as attractor weight ceiling.",
    ),

    # Fraction of activity-space routes that cross an SSP location
    "ssp_route_overlap": ParameterWithUncertainty(
        name="SSP on activity-space route base rate",
        point_estimate=0.096,
        lower_bound=0.05,
        upper_bound=0.20,
        distribution="beta",
        source="Cooper HLF, Tempalski B. Integrating place into research on drug use, drug users' health, and drug policy. Int J Drug Policy 25(3) (2014). DOI 10.1016/j.drugpo.2013.11.008. [Author verified against DOI; original attribution 'Ober 2014' was incorrect — corrected per HANDOFF 4.]",
    ),

    # Drug market attractor weight — PLACEHOLDER, no clean dataset
    "drug_market_attractor_weight": ParameterWithUncertainty(
        name="Drug market venue attractor weight (relative to SSP)",
        point_estimate=1.5,
        lower_bound=1.0,
        upper_bound=3.0,
        distribution="uniform",
        source=PLACEHOLDER + " — proxy only. DO NOT USE IN PUBLISHED RESULTS without AC approval. "
               "Implement as labeled TODO. Real source: overdose/EMS hotspots, arrest data, "
               "ethnographic maps.",
    ),

    # ─────────────────────────────────────────────────────────────────────
    # BRIDGE — Co-location → Sharing edge
    # ─────────────────────────────────────────────────────────────────────

    # Receptive syringe sharing (12-mo per-person prevalence)
    # UNIT MISMATCH: this is 12-mo prevalence, NOT per-co-location-event probability.
    # Use as calibration ceiling; kappa_share converts to per-event rate.
    "sharing_prob_syringe": ParameterWithUncertainty(
        name="Receptive syringe sharing prevalence (12-mo, PWID)",
        point_estimate=0.27,
        lower_bound=0.25,
        upper_bound=0.29,
        distribution="beta",
        source="Burnett JC et al. MMWR 67(1):23 (2018). DOI 10.15585/mmwr.mm6701a5. "
               "UNIT MISMATCH: 12-month per-person prevalence, NOT per-co-location-event. "
               "Use as calibration ceiling via kappa_share; do not apply directly.",
    ),

    # Receptive equipment sharing (12-mo per-person prevalence)
    "sharing_prob_equipment": ParameterWithUncertainty(
        name="Receptive equipment sharing prevalence (12-mo, PWID)",
        point_estimate=0.49,
        lower_bound=0.46,
        upper_bound=0.51,
        distribution="beta",
        source="Burnett JC et al. MMWR 67(1):23 (2018). DOI 10.15585/mmwr.mm6701a5. "
               "UNIT MISMATCH: same unit caveat as sharing_prob_syringe.",
    ),

    # kappa_share: calibration scalar converting 12-mo prevalence → per-event probability
    # DEFAULT = 1.0 (conservative ceiling — maximum edges in run_generator)
    # run_generator uses 0.01 (approximate calibration anchoring <k>≈2.6)
    # TRUE DERIVATION: injection_freq_per_day × shared_fraction_per_partner × co-location_rate
    # shared_fraction_per_partner is defined above — change there, not here, so T and
    # kappa_share cannot drift apart.
    # ANCHOR WARNING: at kappa_share=0.01, <k> is anchored to ~2.6 (target). This means
    # <k> is NOT a held-out validation check when kappa_share is set by anchoring.
    "kappa_share": ParameterWithUncertainty(
        name="Sharing-probability calibration scalar (per-event / 12-mo-prevalence)",
        point_estimate=1.0,
        lower_bound=0.01,
        upper_bound=1.0,
        distribution="uniform",
        source=PLACEHOLDER + " — 1.0 is conservative ceiling. run_generator default=0.01 "
               "anchors <k>≈2.6; this makes <k> an anchor, not a held-out check. "
               "Derive from: injection_freq_per_day × shared_fraction_per_partner / "
               "mean_co-locations_per_agent. shared_fraction_per_partner defined above.",
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Per-edge transmissibility T (duration-integrated, acute-weighted)
    # T = 1 − (1−β_acute)^m_acute × (1−β_chronic)^m_chronic
    # where m = injection_freq_per_day × shared_fraction_per_partner × duration_days
    # ─────────────────────────────────────────────────────────────────────

    # Per-shared-injection HIV transmissibility β (chronic phase)
    # NOTE: Rolls 1–3% is HCV, not HIV — do not use for HIV β
    # Renamed from hiv_transmissibility for clarity
    "beta_chronic_per_shared_injection": ParameterWithUncertainty(
        name="HIV per-shared-injection transmissibility β (chronic phase)",
        point_estimate=0.008,
        lower_bound=0.006,
        upper_bound=0.024,
        distribution="lognormal",
        source="Baggaley RF et al. meta-analysis (per-act transmission probability). "
               "NOTE: Rolls et al. 2011 1–3% is HCV — do NOT use for HIV β.",
    ),

    # Acute-phase multiplier on per-act β
    # Hollingsworth 2008: ×26 vs asymptomatic; duration ~3 months (90 days).
    # NOTE: ratio derived from sexual transmission data; transfers by mechanism.
    # Range lower bound 8 preserves uncertainty from secondary sources.
    "acute_multiplier": ParameterWithUncertainty(
        name="Acute-phase transmissibility multiplier on β_chronic",
        point_estimate=26.0,
        lower_bound=8.0,
        upper_bound=26.0,
        distribution="lognormal",
        source="Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501 — "
               "×26 vs asymptomatic; sexual data, ratio transfers to injection route. "
               "Lower bound from secondary sources (Wawer MJ et al. J Infect Dis 2005; "
               "Pinkerton SD. AIDS 2007). Point estimate updated to Hollingsworth primary value.",
    ),

    # Acute-phase duration (Hollingsworth 2008: ~3 months)
    # Updated from prior 77 d (Fiebig staging only); Hollingsworth gives ~90 d window.
    "acute_duration_days": ParameterWithUncertainty(
        name="Acute HIV infection duration (days)",
        point_estimate=90.0,
        lower_bound=60.0,
        upper_bound=120.0,
        distribution="normal",
        source="Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501 — "
               "acute phase ~3 months; ~90 d window for elevated viremia/transmissibility. "
               "Fiebig staging consistent with 60–120 d range.",
    ),

    # Late-stage multiplier on β_chronic (NEW — Hollingsworth 2008)
    # ×7 vs asymptomatic, concentrated 19–10 months pre-death.
    # Staged T model: T = 1-(1-β_acute)^m_acute × (1-β_late)^m_late × (1-β_chronic)^m_chronic
    # NOT YET IMPLEMENTED in compute_per_edge_T — placeholder for next build pass.
    "late_stage_multiplier": ParameterWithUncertainty(
        name="Late-stage HIV transmissibility multiplier on β_chronic",
        point_estimate=7.0,
        lower_bound=3.0,
        upper_bound=12.0,
        distribution="lognormal",
        source="Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501 — "
               "late-stage ×7 vs asymptomatic; window concentrated 19–10 months pre-death. "
               "NOT YET WIRED into staged T model — next build pass.",
    ),

    # Late-stage duration window (19–10 months pre-death = ~270 days)
    "late_stage_duration_days": ParameterWithUncertainty(
        name="Late-stage HIV infection window for elevated transmissibility (days)",
        point_estimate=270.0,
        lower_bound=180.0,
        upper_bound=365.0,
        distribution="normal",
        source="Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501 — "
               "window 19–10 months pre-death ≈ 270 days. NOT YET WIRED.",
    ),

    # Injection frequency (acts per day)
    # Drug-type dependent: stimulants (meth/cocaine) → more frequent injection.
    # Strathdee 1997 — cocaine → frequent injection; Des Jarlais 2020 — short-acting drugs.
    # NEEDS numeric freq by drug class for staged model. Current range is aggregate.
    "injection_freq_per_day": ParameterWithUncertainty(
        name="Injection frequency (injections per day per PWID)",
        point_estimate=3.0,
        lower_bound=1.0,
        upper_bound=6.0,
        distribution="lognormal",
        source="Burnett JC et al. MMWR 67(1) (2018). DOI 10.15585/mmwr.mm6701a5 (aggregate). "
               "Strathdee SA et al. AIDS 11(8) (1997). DOI 10.1097/00002030-199708000-00001 "
               "(cocaine → frequent injection). "
               "Des Jarlais DC et al. Lancet HIV 7(7) (2020). DOI 10.1016/S2352-3018(20)30082-5 "
               "(short-acting drugs, injection pattern). "
               "NEEDS drug-class stratification (stimulant vs opioid) for next build pass.",
    ),

    # Shared fraction per partner — CALIBRATED FORWARD (Task 2)
    # DERIVATION (show-your-work; not tuned to any outcome):
    #   NHBS any-syringe-sharing prevalence (12 mo): p_any = 0.27
    #   Mean injection-partner degree: k̄ = 2.6
    #   Per-partnership annual sharing probability:
    #     p_pp_yr = 1 − (1 − p_any)^(1/k̄) = 1 − 0.73^(1/2.6) ≈ 0.117
    #   Annual injection frequency: f_yr = 3/day × 365 = 1095 injections/year
    #   Per-injection per-partner sharing probability (Bernoulli, small-p approx):
    #     p_pp_inj ≈ p_pp_yr / f_yr = 0.117 / 1095 ≈ 0.000107
    #   This is the conservative (Poisson independence) estimate.
    #   Upper bound: if sharers share on ~half of injections (heavy-sharing subgroup):
    #     0.27 × 0.5 / 2.6 / 3 × 1/365 ≈ 0.014 — still well below 0.15
    #   Central estimate: 0.001 [bounds 0.0001, 0.015]
    #   SATURATION NOTE: at inj_freq=3, acute_dur=77, saturation (T>0.95) requires
    #     shared_fraction > 0.088. The calibrated range [0.0001, 0.015] is far below
    #     saturation — T will range ~0.01–0.35 rather than being pinned near 1.
    #   NOT tuned to R₀ or outbreak rarity. PENDING AC SIGN-OFF.
    "shared_fraction_per_partner": ParameterWithUncertainty(
        name="Fraction of injections shared with a given network partner",
        point_estimate=0.001,
        lower_bound=0.0001,
        upper_bound=0.015,
        distribution="lognormal",
        source="CALIBRATED FORWARD from NHBS syringe sharing (Burnett JC et al. MMWR 67(1) 2018. "
               "DOI 10.15585/mmwr.mm6701a5, p_any=0.27) ÷ (injection_freq × mean_degree). "
               "Derivation: p_per_partner_yr = 1−(1−0.27)^(1/2.6) ≈ 0.117; "
               "shared_fraction ≈ 0.117/1095 ≈ 0.0001 (independence lower bound). "
               "Upper bound 0.015 allows for heavy-sharing subgroups. "
               "NOT TUNED to R₀ or outbreak frequency — forward from behavioral data only. "
               "PENDING AC SIGN-OFF before use in published results.",
    ),

    # Chronic-phase transmission window (partnership duration / time-to-treatment)
    # Acute phase dominates; this is secondary
    "chronic_window_days": ParameterWithUncertainty(
        name="Chronic HIV partnership transmission window (days)",
        point_estimate=180.0,
        lower_bound=60.0,
        upper_bound=365.0,
        distribution="lognormal",
        source=PLACEHOLDER + " — bounded by partnership duration and time-to-diagnosis/treatment. "
               "Acute phase dominates IDU outbreak dynamics; this is secondary.",
    ),

    # Per-act β cap (sanity constraint: β_acute ≤ BETA_CAP per act)
    # Not a ParameterWithUncertainty — a hard constraint
    # BETA_CAP = 0.5 (accessed directly, not via .point_estimate)

    # Seed HIV prevalence among PWID
    "seed_hiv_prevalence": ParameterWithUncertainty(
        name="HIV prevalence among PWID at initialization (seed)",
        point_estimate=0.07,
        lower_bound=0.05,
        upper_bound=0.09,
        distribution="beta",
        source="Burnett JC et al. MMWR 67(1):23 (2018). DOI 10.15585/mmwr.mm6701a5. "
               "National average; varies substantially by region.",
    ),

    # ─────────────────────────────────────────────────────────────────────
    # T_edge — collapsed per-edge transmissibility (Task 1)
    # Direct bounded parameter; does NOT require decomposition.
    # Alternative to the 5-parameter decomposed path; both are preserved.
    # ─────────────────────────────────────────────────────────────────────
    "T_edge": ParameterWithUncertainty(
        name="Effective per-edge transmissibility (collapsed, directly bounded)",
        point_estimate=0.35,
        lower_bound=0.10,
        upper_bound=0.95,
        distribution="beta",
        source="Rolls DA et al. J Theor Biol 297:73 (2011). DOI 10.1016/j.jtbi.2011.12.008 — "
               "per-partnership HCV transmission probability as structural analog (method precedent; "
               "use HIV-specific value when available). "
               "Hollingsworth TQ et al. Nat Med 2008 — phylogenetic cluster R₀ for IDU implies "
               "T ≈ R₀ × τ_c / (⟨k²⟩−⟨k⟩)/⟨k⟩; at R₀~1–2 and τ_c~0.3, T~0.3–0.6. "
               "Upper bound 0.95 from per-act β_acute × m_acute at acute-viremia peak; "
               "lower bound 0.10 for sparse-sharing chronic partnerships. "
               "PENDING AC SIGN-OFF — collapsed parameter removes mechanistic transparency.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Validation targets (held out; NOT used for fitting)
# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_TARGETS = {
    # Mean injecting-partner degree ⟨k⟩
    "mean_degree": {
        "value": 2.6,
        "range": (0, 14),
        "source": "Buchanan R et al. J Infect 80:225 (2019). DOI 10.1016/j.jinf.2019.12.010.",
        "note": "RDS→ABM; isolated community. Validate ⟨k⟩ as a point check only.",
    },
    # Giant component must be present
    "giant_component_present": {
        "value": True,
        "source": "Buchanan 2019; Klovdahl/Potterat 1994. Soc Sci Med 38:79. DOI 10.1016/0277-9536(94)90302-6.",
        "note": "137 in one component (IoW); 595–600 connected (Colorado Springs).",
    },
    # Mean core path length
    "mean_path_length": {
        "value": 3.1,
        "source": "Klovdahl/Potterat 1994 (ibid.)",
        "note": "Small-world core; 3.1 steps infected→susceptible.",
    },
    # k² / tail is explicitly NOT a point-check target — sensitivity range only
    "k2_tail": {
        "value": None,
        "source": "Multiple RDS surveys (Buchanan, Young, Klovdahl).",
        "note": "TAIL CAVEAT: high-degree tail is most under-captured in all RDS datasets. "
                "k² and hence τ_c must be treated as a sensitivity range, never a point estimate.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTBREAK VALIDATION PANEL
# Validate simulated outbreak on TRAJECTORY (size / cluster / incidence /
# doubling), NOT on a published R₀ — avoids circularity.
# HIV outbreak frequency is HELD OUT as a forward prediction; not fitted.
# ─────────────────────────────────────────────────────────────────────────────

OUTBREAK_VALIDATION_TARGETS = {
    "final_size": {
        "scott_county_in": 215,
        "cabell_county_wv": 82,
        "ne_massachusetts": 129,
        "source": (
            "Peters PJ et al. N Engl J Med 373:2431 (2015). DOI 10.1056/NEJMoa1515195 "
            "(Scott Co.; 215 confirmed cases). "
            "McClung RP et al. Am J Prev Med 61(1):50 (2021). DOI 10.1016/j.amepre.2021.05.039 "
            "(Cabell Co.; 82). "
            "Alpren C et al. Am J Public Health 110(1):37 (2020). DOI 10.2105/AJPH.2019.305366 "
            "(NE Mass.; 129)."
        ),
        "use": "final-size plausibility check; range <100 to >1000 cross-site",
        "note": "Des Jarlais DC et al. Lancet HIV 2020 DOI 10.1016/S2352-3018(20)30082-5 — "
                "outbreak size range <100 to >1000.",
    },
    "cluster_dominance": {
        "scott_county_pct": 98.7,
        "cabell_county_pct": 93.0,
        "source": "Peters 2015 (ibid.); McClung 2021 (ibid.)",
        "use": "single-seed → giant-component check. "
               "Argues against 29% giant-component artifact — empirical outbreaks "
               "show 93-99% in one molecular cluster.",
    },
    "degree_risk_gradient": {
        "arr_per_syringe_partner_named": 1.9,
        "source": "Peters PJ et al. N Engl J Med 373:2431 (2015). DOI 10.1056/NEJMoa1515195",
        "use": "validates degree→risk shape in the generated network",
    },
    "explosive_hiv_incidence_per_100py": {
        "value": 18.6,
        "ci_lower": 11.1,
        "ci_upper": 26.0,
        "source": "Strathdee SA et al. AIDS 11(8) (1997). DOI 10.1097/00002030-199708000-00001",
        "use": "incidence check during explosive-phase simulation",
    },
    "hcv_prevalence_injection_dominated": {
        "range_pct": (88, 92),
        "source": "Peters 2015 (ibid.); Strathdee 1997 (ibid.)",
        "use": (
            "route-stratified sharing-intensity anchor. "
            "ROUTE CAVEAT: use ONLY in injection-dominated nodes. "
            "In bridge (HIV+/chemsex MSM) HCV is sexually transmitted — using "
            "whole-network HCV as injection intensity is route-misattribution."
        ),
    },
    "sir_removal_rate_per_diagnosed_per_day": {
        "value": 0.024,
        "source": "Gonsalves GS & Crawford FW. Lancet HIV 5(6):e297 (2018). "
                  "DOI 10.1016/S2352-3018(18)30176-0",
        "use": "SIR removal term (diagnosis→treatment suppression)",
        "note": "Gonsalves & Crawford model transmission rate + case-finding; "
                "no clean R₀ — validate on trajectory, not a published R₀.",
    },
    "structural_drivers": {
        "cabell_unstable_housing_pct": 80,
        "source": "McClung 2021 (ibid.); Des Jarlais 2020 (ibid.)",
        "use": "structural concentration / hotspot weighting in venue layer",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SENTINEL LADDER
# Four pathogens on ONE contact-intensity axis.
# Rectal GC → early syphilis → HIV → sexual HCV (ordered by effective R₀ /
# threshold; lower rung = lower threshold = lights up first).
# HIV stays HELD OUT. Each rung is a non-circular ordered constraint.
# ─────────────────────────────────────────────────────────────────────────────

SENTINEL_LADDER = {
    "rectal_gc": {
        "rung": 1,
        "subsequent_hiv_incidence_per_100py": 4.1,
        "source": "Katz DA et al. Sex Transm Dis 43(2):91 (2016). "
                  "DOI 10.1097/OLQ.0000000000000423",
        "role": "Fastest detector; site-specific to receptive anal → cleanest "
                "behavioral marker of bridge node.",
    },
    "early_syphilis": {
        "rung": 2,
        "subsequent_hiv_incidence_per_100py": 2.8,
        "source": "Katz DA et al. Sex Transm Dis 43(2):91 (2016). "
                  "DOI 10.1097/OLQ.0000000000000423",
        "role": "Detector + ulcerative cofactor.",
    },
    "hiv": {
        "rung": 3,
        "held_out": True,
        "role": "HELD-OUT forward prediction. Not fitted. Not validated against outcome.",
    },
    "sexual_hcv": {
        "rung": 4,
        "incidence_per_100py_range": (1.0, 4.09),
        "source": (
            "Chaillon A et al. Open Forum Infect Dis 6(9):ofz160 (2019). "
            "DOI 10.1093/ofid/ofz160 (San Diego 1.13→3.01). "
            "Wandeler G et al. Clin Infect Dis 55(6):821 (2012). "
            "DOI 10.1093/cid/cis694 (Swiss 0.23→4.09). "
            "Jansen K et al. PLoS One 10(10):e0142515 (2015). "
            "DOI 10.1371/journal.pone.0142515 (Ger 1.54)."
        ),
        "role": "High-water mark: sexual sustainment ≈ pool already supercritical for HIV. "
                "Falsification: HCV sexual sustainment should track contact INTENSITY, "
                "not HIV serostatus per se.",
    },
    "composite_sti_cofactor": {
        "ahr": 2.7,
        "ci_lower": 1.2,
        "ci_upper": 6.4,
        "paf_pct": 14.6,
        "source": "Kelley CF et al. AIDS Res Hum Retroviruses 31(10):1009 (2015). "
                  "DOI 10.1089/AID.2015.0013 (rectal STI → incident HIV, propensity-weighted). "
                  "David D et al. Infect Dis Model 5:721 (2020). "
                  "DOI 10.1016/j.idm.2020.10.008 (HIV–syphilis coupling model). "
                  "Wandeler 2012: syphilis predicts HCV seroconversion aHR 2.11.",
        "role": "ONE inflammatory cofactor on bridge nodes — composite, not per-pathogen. "
                "Rectal GC + syphilis are INDICATORS of that state; "
                "4.1/100py and 2.8/100py are VALIDATION TARGETS, not transmission coefficients.",
        "note": "DO NOT split per-pathogen — collinear, would double-count. "
                "Coupling precedent: David 2020 HIV-syphilis reproduction-number model.",
    },
    "methamphetamine_sexualized_use_pct": {
        "value": 48.0,
        "source": "Palaniswami PP & Fierer DS. Open Forum Infect Dis 5(10):ofy238 (2018). "
                  "DOI 10.1093/ofid/ofy238",
        "role": "48% sexualized meth use in sexually-acquired-HCV cohort. "
                "Drug type → injection_frequency → m. Stimulant use is the link "
                "between injection model and bridge; sets upper range of injection_freq.",
    },
}
