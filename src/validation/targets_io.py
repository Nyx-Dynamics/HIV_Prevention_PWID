"""
Target definitions and classification for the validation harness.

Every target has exactly one class (one line to flip):
  SCORED    — model output vs sourced range; compute overlap pass/fail
  HELD_OUT  — HIV outcome we predict, never tune to; report-only by construction
  PROVENANCE — sourced input parameter; record DOI + value, no pass/fail
  STUBBED   — needs a layer that doesn't exist yet; return not_computable_yet

HELD_OUT is a frozenset checked by the harness before any scoring.
Nothing in HELD_OUT ever contributes to pass rate or fail rate.
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELD_OUT — hard-coded; checked first; never scored under any circumstances
# ─────────────────────────────────────────────────────────────────────────────

HELD_OUT = frozenset({
    "hiv_r0",
    "hiv_r0_decomposed",
    "hiv_r0_T_edge",
    "hiv_incidence",
    "hiv_outbreak_frequency",
    "hiv_p_outbreak",
    "hiv_p_r0_gt1",
    "hiv_forward_prediction",
})

# ─────────────────────────────────────────────────────────────────────────────
# TARGET_CLASSES — one line per target; flip class to change harness behavior
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CLASSES = {
    # ── Network benchmarks (SCORED against sourced ranges) ────────────────
    # 7B AMENDMENT: giant_component_fraction 45-100% RETIRED.
    # Friedman 1997 shows largest component ~30% of 767 — not 45-100%.
    # The correct target is largest_component_fraction (core-periphery below).
    "giant_component_fraction":   "RETIRED",  # → replaced by core-periphery targets
    "clustering_coefficient":     "SCORED",   # target 0.1–0.4 (empirical)
    "dispersion_k2_over_k2":      "SCORED",   # target > Poisson_baseline × 1.5
    "mean_path_length":           "SCORED",   # target ~3.1

    # ── Core-periphery targets (SCORED; 7B amendment; consume fields emitted by 8) ──
    "largest_component_fraction": "SCORED",   # target 0.20–0.55 (Friedman ~30%)
    "isolate_fraction":           "SCORED",   # target < 0.10 (< 10% degree-0)
    "core_dispersion":            "SCORED",   # over-Poisson on 2-core
    "core_clustering":            "SCORED",   # target 0.30–0.65 (Kwan 2019)

    # ── Outbreak trajectory (SCORED once outbreak_sim.json is emitted) ───
    "final_size":                 "SCORED",   # target: 82–215 cross-site range
    "single_cluster_fraction":    "SCORED",   # target 0.93–0.99
    "degree_risk_gradient":       "SCORED",   # target aRR ~1.9
    "explosive_incidence":        "SCORED",   # target 11.1–26.0 /100 py

    # ── HIV — HELD OUT (forward predictions, never scored) ───────────────
    "hiv_r0":                     "HELD_OUT",
    "hiv_r0_decomposed":          "HELD_OUT",
    "hiv_r0_T_edge":              "HELD_OUT",
    "hiv_incidence":              "HELD_OUT",
    "hiv_outbreak_frequency":     "HELD_OUT",
    "hiv_p_outbreak":             "HELD_OUT",
    "hiv_p_r0_gt1":               "HELD_OUT",

    # ── Provenance (input parameters; record + DOI, no pass/fail) ────────
    "sir_removal_rate":           "PROVENANCE",
    "composite_sti_cofactor_ahr": "PROVENANCE",
    "beta_chronic":               "PROVENANCE",
    "acute_multiplier":           "PROVENANCE",
    "late_stage_multiplier":      "PROVENANCE",
    "sharing_prevalence":         "PROVENANCE",

    # ── Stubbed (need bridge / multi-pathogen layer; not_computable_yet) ─
    "rectal_gc_hiv_incidence":    "STUBBED",
    "syphilis_hiv_incidence":     "STUBBED",
    "sexual_hcv_incidence":       "STUBBED",
    "hcv_consistency_check":      "STUBBED",
    "bridge_subpopulation":       "STUBBED",
}


def classify(target_key: str) -> str:
    """Return the class of a target. HELD_OUT takes precedence over dict."""
    if target_key in HELD_OUT:
        return "HELD_OUT"
    return TARGET_CLASSES.get(target_key, "UNKNOWN")


def is_retired(target_key: str) -> bool:
    """True if the target has been retired (replaced by a better target)."""
    return TARGET_CLASSES.get(target_key) == "RETIRED"


# ─────────────────────────────────────────────────────────────────────────────
# SCORED TARGET DEFINITIONS
# Sourced ranges + DOIs for all SCORED targets.
# Keys match TARGET_CLASSES above; each entry has value_range, source, type.
# ─────────────────────────────────────────────────────────────────────────────

SCORED_TARGETS = {
    # ── Network benchmarks ────────────────────────────────────────────────
    "giant_component_fraction": {
        "type": "fraction",
        "range": (0.45, 1.00),
        "source": "Buchanan R et al. J Infect 80:225 (2019). DOI 10.1016/j.jinf.2019.12.010 "
                  "(137/595 in giant, ~23–100%); Klovdahl/Potterat 1994. Minimum ~45% as "
                  "operationally relevant connected component.",
        "note": "Current dyad model: ~29% — fails this check; "
                "Handoff 7A connectivity fix expected to raise this.",
    },
    "clustering_coefficient": {
        "type": "fraction",
        "range": (0.10, 0.40),
        "source": "Rolls DA et al. J Theor Biol 297:73 (2011). DOI 10.1016/j.jtbi.2011.12.008; "
                  "Buchanan 2019.",
        "note": "ABOVE_FLOOR (~0.08) currently; Proxy C first to reach PASS.",
    },
    "dispersion_k2_over_k2": {
        "type": "lower_bound_factor",
        "factor": 1.5,       # must exceed Poisson baseline × 1.5
        "source": "Buchanan 2019 — max degree 14 vs mean 2.6 implies heavy tail "
                  "substantially above Poisson (1 + 1/⟨k⟩).",
        "note": (
            "Poisson baseline = 1 + 1/⟨k⟩ computed from the SAME RUN's ⟨k⟩ "
            "(not hardcoded) so the target moves correctly when 7A reconnects isolates. "
            "Target = baseline × 1.5."
        ),
    },
    "mean_path_length": {
        "type": "range",
        # Two-sided range: stale baseline 5.18 and dyad net 2.0–2.3 both FAIL;
        # healthy injection networks (Klovdahl ~3.1) PASS.
        "range": (2.5, 4.5),
        "ideal": 3.1,
        "source": "Klovdahl AS, Potterat JJ et al. Soc Sci Med 38:79 (1994). "
                  "DOI 10.1016/0277-9536(94)90302-6",
        "note": (
            "Target ~3.1 (small-world injection-network core). "
            "Range [2.5, 4.5] is two-sided so both stale-long (≥5) and "
            "dyad-short (≤2.3) register as FAIL."
        ),
    },

    # ── Core-periphery targets (7B amendment — consume fields emitted by 8) ──
    "largest_component_fraction": {
        "type": "range",
        "range": (0.20, 0.55),
        "source": "Friedman SR et al. Am J Public Health 87(8):1289 (1997). "
                  "DOI 10.2105/ajph.87.8.1289 — largest component ~30% of 767 nodes. "
                  "Lin/Boodram 2023 DOI 10.1016/j.drugpo.2023.104217 — concentrated West Side core.",
        "note": (
            "RETIRED the 45-100% giant_component_fraction target (was wrong). "
            "Friedman empirical: one large ~30% + tail of smaller; this replaces the old check."
        ),
    },
    "isolate_fraction": {
        "type": "upper_bound",
        "range": (0.0, 0.10),   # target < 10% isolated
        "source": "Friedman 1997 (ibid.); Lin/Boodram 2023 — contact layer should "
                  "connect most agents who move through venue geography.",
        "note": "Target < 10% degree-0 nodes after contact-layer decoupling (Handoff 8).",
    },
    "core_dispersion": {
        "type": "lower_bound_factor",
        "factor": 1.5,   # must exceed Poisson baseline × 1.5 on 2-core nodes
        "source": "Buchanan 2019 DOI 10.1016/j.jinf.2019.12.010; Friedman 1997 — "
                  "2-core concentrates high-degree nodes; Poisson baseline = 1 + 1/⟨k⟩ on core.",
        "note": "Over-Poisson dispersion on 2-core validates that hub structure is present.",
    },
    "core_clustering": {
        "type": "range",
        "range": (0.30, 0.70),   # Kwan 2019 injection-network specific
        "source": "Kwan TH et al. PLoS One 14(5):e0216727 (2019). "
                  "DOI 10.1371/journal.pone.0216727 — clustering 0.53–0.65 on injection networks.",
        "note": (
            "Measured on 2-core (core-restricted). Range [0.30, 0.70] allows for network "
            "size/density variation; Kwan target is 0.53-0.65 in dense networks."
        ),
    },

    # ── Outbreak trajectory ───────────────────────────────────────────────
    "final_size": {
        "type": "range",
        "range": (50, 250),   # 82 Cabell → 215 Scott County
        "source": "Peters PJ et al. N Engl J Med 373:2431 (2015). DOI 10.1056/NEJMoa1515195 "
                  "(215); McClung RP et al. Am J Prev Med 61:50 (2021). DOI 10.1016/j.amepre.2021.05.039 "
                  "(82); Alpren C et al. Am J Public Health 110:37 (2020). DOI 10.2105/AJPH.2019.305366 "
                  "(129).",
        "note": "range <100 to >1000 cross-site (Des Jarlais 2020); 50–250 = named-outbreak range.",
    },
    "single_cluster_fraction": {
        "type": "fraction",
        "range": (0.93, 1.00),
        "source": "Peters 2015 (98.7%); McClung 2021 (93%).",
        "note": "Single dominant molecular cluster. Argues against 29% giant-component artifact.",
    },
    "degree_risk_gradient": {
        "type": "range",
        "range": (1.5, 3.0),   # aRR ~1.9 per partner named
        "source": "Peters PJ et al. N Engl J Med 373:2431 (2015). DOI 10.1056/NEJMoa1515195",
        "note": "aRR 1.9 per syringe-sharing partner named; validates degree→risk shape.",
    },
    "explosive_incidence": {
        "type": "range",
        "range": (11.1, 26.0),  # 95% CI from Strathdee 1997
        "units": "per 100 person-years",
        "source": "Strathdee SA et al. AIDS 11(8) (1997). DOI 10.1097/00002030-199708000-00001",
        "note": "Trajectory check during explosive phase — NOT used to calibrate model.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PROVENANCE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

PROVENANCE_REGISTRY = {
    "sir_removal_rate": {
        "value": 0.024,
        "units": "per diagnosed per day",
        "source": "Gonsalves GS & Crawford FW. Lancet HIV 5(6):e297 (2018). "
                  "DOI 10.1016/S2352-3018(18)30176-0",
    },
    "composite_sti_cofactor_ahr": {
        "value": 2.7,
        "ci": (1.2, 6.4),
        "source": "Kelley CF et al. AIDS Res Hum Retroviruses 31(10):1009 (2015). "
                  "DOI 10.1089/AID.2015.0013",
    },
    "beta_chronic": {
        "value": 0.008,
        "range": (0.006, 0.024),
        "source": "Baggaley RF et al. AIDS (2006). DOI 10.1097/01.aids.0000218543.46963.6d",
    },
    "acute_multiplier": {
        "value": 26.0,
        "source": "Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501",
    },
    "late_stage_multiplier": {
        "value": 7.0,
        "source": "Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501",
    },
    "sharing_prevalence": {
        "value": 0.27,
        "source": "Burnett JC et al. MMWR 67(1) (2018). DOI 10.15585/mmwr.mm6701a5",
    },
}
