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
        source="Ober AJ et al. Int J Drug Policy 25(3):308 (2014). DOI 10.1016/j.drugpo.2013.11.008. "
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
        source="Ober AJ et al. Int J Drug Policy 25(3):308 (2014). DOI 10.1016/j.drugpo.2013.11.008.",
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
    # DEFAULT = 1.0 (conservative ceiling — maximum edges)
    # TODO: calibrate from daily injection frequency × sharing fraction × co-location rate
    "kappa_share": ParameterWithUncertainty(
        name="Sharing-probability calibration scalar (per-event / 12-mo-prevalence)",
        point_estimate=1.0,
        lower_bound=0.01,
        upper_bound=1.0,
        distribution="uniform",
        source=PLACEHOLDER + " — default=1.0 is conservative ceiling (maximum edges). "
               "TODO: calibrate from daily injection frequency × fraction shared × "
               "mean network co-location rate before use in threshold or R₀ calculations.",
    ),

    # HIV per-shared-injection transmissibility β
    # NOTE: Rolls 1–3% is HCV, not HIV — do not use for HIV β
    "hiv_transmissibility": ParameterWithUncertainty(
        name="HIV per-shared-injection transmissibility β",
        point_estimate=0.008,
        lower_bound=0.006,
        upper_bound=0.024,
        distribution="lognormal",
        source="Baggaley RF et al. meta-analysis (per-act transmission probability). "
               "NOTE: Rolls et al. 2011 1–3% is HCV — do NOT use for HIV β.",
    ),

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
