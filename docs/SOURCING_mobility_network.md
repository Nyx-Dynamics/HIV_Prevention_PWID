# Sourcing Memo: Mobility-Driven Network Generator
### HIV_Prevention_PWID — `src/mobility/`

*Methods-section-ready parameter provenance for the network generator module.*
*Generated: 2026-06-16. Branch: `mobility/network-generator`.*

---

## Purpose

This memo records every parameter value, its source, its uncertainty range, and any
gap flags for the mobility → contact-network → epidemic-threshold pipeline. It is
intended to be cited directly in the methods section and to make all assumptions
auditable by reviewers who lack access to the code.

**Ethical invariant (reproduced from module docstring):**
This module generates an *ensemble sampled from population distributions*. It must
never track, reconstruct, or require real individuals' trajectories. A digital twin
of PWID would be surveillance of a criminalized population and is explicitly out of
scope. Any extension toward individual-level tracking is prohibited.

---

## Layer 1 — Walk / Traversement-Potential Distribution

### Walk engine

The module uses the **Exploration and Preferential Return (EPR)** model (Song et al.
2010). At each step an agent either *explores* (jumps to a new location with probability
`ρ·S^−γ`, where S is the number of distinct locations already visited) or returns
*preferentially* to a previously visited location with probability proportional to
visit frequency.

**Reference:** Song C, Koren T, Wang P, Barabási AL. Modelling the scaling properties of
human mobility. *Nature Physics* 6:818–823 (2010). DOI 10.1038/nphys1760.

Attractor-biased (density-EPR) extension:
**Reference:** Pappalardo L et al. Returners and explorers dichotomy in human mobility.
*Nature Communications* 6:8166 (2015). DOI 10.1038/ncomms9166.

*Note:* scikit-mobility (`skmob`) ships a ready implementation of DensityEPR. It was
not used here due to GDAL/geopandas build failures on the current environment; the
module implements a pure-NumPy EPR fallback that is functionally equivalent for
first-pass network generation (venue gravity reduces to a weighting column on the
venue array; no CRS-aware geo-computation is performed).

### Jump-length distribution

| Symbol | Meaning | Point estimate | Uncertainty | Distribution | Source |
|---|---|---|---|---|---|
| β | Jump-length power-law exponent (`P(Δr) ∝ (Δr)^{−1−β}`) | 0.60 | [0.50, 0.75] | uniform | González, Hidalgo & Barabási 2008 |
| ζ | Radius-of-gyration growth exponent (`r_g ∝ t^ζ/2`) | 1.65 | [1.29, 1.71] | uniform | González et al. 2008 |

**Gap flag:** Both exponents are from general mobile-phone population data. PWID-specific
mobility data do not exist at scale. Treat as population priors; the activity-radius
scale below adjusts to the PWID context.

**Reference:** González MC, Hidalgo CA, Barabási AL. Understanding individual human
mobility patterns. *Nature* 453:779–782 (2008). DOI 10.1038/nature06958.

### PWID activity radius (r_g scale)

| Symbol | Meaning | Point estimate | Uncertainty | Distribution | Source |
|---|---|---|---|---|---|
| `r_g_scale_km` | Mean radius-of-gyration for PWID activity space | 2.4 km | [1.0, 5.0] km | lognormal | SF activity-space study (N=1 084), *Int J Drug Policy* 2014 |

**Gap flag (critical):** The 2.4 km value is the mean self-reported anchor-point distance
from one city (San Francisco, high-density urban). It is not a GPS-derived radius of
gyration. It substantially underestimates the activity space of rural PWID, who are
the highest-risk group. Treat the scale as a major sensitivity axis. All results should
be re-run at r_g_scale = 1.0, 2.4, 5.0 km before any quantitative claims.

**Reference:** Ober AJ et al. Activity spaces among injection drug users in San Francisco.
*International Journal of Drug Policy* 25(3) (2014). DOI 10.1016/j.drugpo.2013.11.008.

### EPR model parameters

| Parameter | Symbol | Value | Dist | Source |
|---|---|---|---|---|
| Exploration probability scaling | ρ | 0.6 | uniform [0.3, 0.9] | Pappalardo et al. 2015 |
| Exploration decay exponent | γ | 0.21 | uniform [0.10, 0.35] | Pappalardo et al. 2015 |

**Gap flag (KEY SENSITIVITY KNOB):** The explorer fraction is emergent from ρ and γ —
it is not a fixed parameter. It directly controls ⟨k²⟩ and hence τ_c. The sweep in
Stage 3 varies ρ and γ to bound τ_c; this must be reported as the primary uncertainty
range for the threshold result.

---

## Layer 2 — Attractors (Venues)

| Venue type | Weighting | Source | Gap flag |
|---|---|---|---|
| SSP/SEP locations | High (real attractors) | NASEN directory (nasen.org); AmFAR Opioid & Health Indicators | Real and geocodable. Module uses synthetic SSP layer in absence of locally loaded data. |
| SSP utilization (anchor strength) | 52% of PWID got syringes from an SSP in the past 12 months | CDC NHBS / MMWR 67(1) 2018. DOI 10.15585/mmwr.mm6701a5 | Population rate, not per-venue pull — calibration adjustment needed |
| SSP-on-activity-space-route base rate | 9.6% of activity-space locations overlap with SSP route | SF activity-space study 2014 (ibid.) | Template for "venue intersects route" |
| Drug market geography | **PLACEHOLDER — no clean dataset** | Proxy: overdose/EMS hotspots, arrest data, ethnographic maps | **DO NOT INVENT A DATASET. Implemented as clearly labeled placeholder with TODO. AC must provide or approve proxy before this attractor type is used in any published result.** |

---

## Bridge — Co-location → Sharing Edge

| Symbol | Meaning | Point estimate | Uncertainty | Distribution | Source | Unit caveat |
|---|---|---|---|---|---|---|
| `sharing_prob_syringe` | Receptive syringe sharing | 0.27 | [0.25, 0.29] | beta | NHBS 20 cities 2015. DOI 10.15585/mmwr.mm6701a5 | **12-month per-person prevalence, NOT per-co-location-event.** Used as calibration ceiling; `kappa_share` converts to per-event rate. |
| `sharing_prob_equipment` | Receptive equipment sharing | 0.49 | [0.46, 0.51] | beta | NHBS 2015 (ibid.) | Same unit mismatch — see above |
| `kappa_share` | Calibration scalar (per-event rate / 12-mo prevalence) | 1.0 | [0.01, 1.0] | uniform | **PLACEHOLDER** — must be calibrated | **Default = 1.0 (conservative ceiling). TODO: calibrate from daily injection frequency × sharing fraction data.** |
| `hiv_transmissibility` | Per-shared-injection HIV transmissibility β | 0.008 | [0.006, 0.024] | lognormal | Baggaley et al. meta-analysis | HIV value. Do NOT use Rolls 1–3% (that is HCV). |
| `seed_hiv_prevalence` | HIV prevalence among PWID at initialization | 0.07 | [0.05, 0.09] | beta | NHBS 2015/2018. DOI 10.15585/mmwr.mm6701a5 | National average; varies substantially by region |

---

## Layer 3 — Validation Targets (Held Out)

These statistics are from independent network surveys of PWID. They are **not used to fit
any parameter** — they are held-out checks on whether the generated network has plausible
structure. They must not be used to tune the generator or to validate outbreak conclusions.

| Statistic | Target value | Source |
|---|---|---|
| Mean injecting-partner degree ⟨k⟩ | 2.6 (range 0–14) | Buchanan et al. 2019, *J Infect* 80:225. DOI 10.1016/j.jinf.2019.12.010 |
| Giant-component existence | Yes (137–600+ in empirical studies) | Buchanan 2019; Klovdahl/Potterat 1994 |
| Mean core path length | ~3.1 steps | Klovdahl/Potterat 1994, *Soc Sci Med* 38:79. DOI 10.1016/0277-9536(94)90302-6 |
| Degree heterogeneity | Present (qualitative) | Rolls et al. 2011, *J Theor Biol* 297:73. DOI 10.1016/j.jtbi.2011.12.008 |
| Rural network: 433 users, 488 risk ties | ~1.1 ties/person | Young et al. 2014, *PLoS One* 9:e101047. DOI 10.1371/journal.pone.0101047 |
| Near-miss: 1 HIV transmission in 3 yr among 595 | Qualitative structural evidence | Potterat, Rothenberg & Muth 1999, *Int J STD AIDS* 10:182. DOI 10.1258/0956462991913853 |

**Tail caveat (mandatory):** Every degree dataset is RDS/convenience, small and local.
RDS is itself a network walk, so the **high-degree tail (and hence ⟨k²⟩) is the most
under-captured quantity** — and it is exactly what τ_c is most sensitive to. ⟨k⟩ and
component structure are point checks; ⟨k²⟩ and the tail must be treated as a
sensitivity range, never a point estimate.

---

## Threshold Math

The **network epidemic threshold** is derived from the generated degree distribution via
the configuration-model formula (Newman 2002):

```
τ_c = ⟨k⟩ / (⟨k²⟩ − ⟨k⟩)
```

The basic reproduction number on the configuration network:

```
R₀ = T · (⟨k²⟩ − ⟨k⟩) / ⟨k⟩
```

where T is per-partnership transmissibility (compounded from per-event β and sharing-event
count). Epidemic threshold is where R₀ = 1, i.e. T = τ_c. κ (mechanistic) := the network
density/connectivity at which R₀ = 1.

**Current hardcoded value:** `critical_network_threshold = 0.35` (sourced as
"Theoretical/modeling"). The generator computes the mechanistic alternative. They are not
yet wired together — that is a separate human-reviewed step.

**Reference:** Newman MEJ. Spread of epidemic disease on networks. *Physical Review E*
66:016128 (2002). DOI 10.1103/PhysRevE.66.016128.

---

## Full Reference List

1. González MC, Hidalgo CA, Barabási AL. Understanding individual human mobility patterns.
   *Nature* 453:779–782 (2008). DOI 10.1038/nature06958

2. Song C, Koren T, Wang P, Barabási AL. Modelling the scaling properties of human mobility.
   *Nature Physics* 6:818–823 (2010). DOI 10.1038/nphys1760

3. Pappalardo L et al. Returners and explorers dichotomy in human mobility.
   *Nature Communications* 6:8166 (2015). DOI 10.1038/ncomms9166

4. Ober AJ et al. Activity spaces among injection drug users in San Francisco.
   *International Journal of Drug Policy* 25(3):308–318 (2014).
   DOI 10.1016/j.drugpo.2013.11.008

5. Burnett JC et al. HIV infection and HIV-associated behaviors among persons who inject
   drugs — 20 cities, United States, 2015. MMWR 67(1):23–28 (2018).
   DOI 10.15585/mmwr.mm6701a5

6. Klovdahl AS, Potterat JJ et al. Social networks and infectious disease: the Colorado
   Springs Study. *Social Science & Medicine* 38:79–88 (1994).
   DOI 10.1016/0277-9536(94)90302-6

7. Potterat JJ, Rothenberg RB, Muth SQ. Network structural dynamics and infectious disease
   propagation. *International Journal of STD & AIDS* 10:182–185 (1999).
   DOI 10.1258/0956462991913853

8. Rolls DA et al. Modelling a disease-relevant contact network of people who inject drugs.
   *Journal of Theoretical Biology* 297:73–87 (2011). DOI 10.1016/j.jtbi.2011.12.008

9. Buchanan R et al. HIV transmission network characteristics of people who inject drugs.
   *Journal of Infection* 80:225–231 (2019). DOI 10.1016/j.jinf.2019.12.010

10. Young AM et al. Social-spatial network characteristics among rural young adults who use
    drugs. *PLoS One* 9:e101047 (2014). DOI 10.1371/journal.pone.0101047

11. Newman MEJ. Spread of epidemic disease on networks. *Physical Review E* 66:016128 (2002).
    DOI 10.1103/PhysRevE.66.016128

---

## Gaps Requiring AC Decision

1. **Drug-market geography:** No clean dataset. Current implementation uses a clearly
   labeled `PLACEHOLDER` attractor layer. AC must provide data source or approve a specific
   proxy (overdose/EMS hotspots, arrest data, ethnographic map) before this attractor
   type appears in any published result.

2. **`kappa_share` calibration:** The NHBS sharing prevalence is a 12-month per-person
   rate, not a per-co-location-event probability. The module uses `kappa_share = 1.0`
   (conservative ceiling) by default. Calibration requires: daily injection frequency ×
   fraction of injections shared × mean network co-location rate. AC should provide or
   approve this calibration step.

3. **r_g scale beyond San Francisco:** The 2.4 km activity-radius estimate is from a
   single high-density urban study. Rural PWID (the highest-risk group) likely have
   substantially larger activity spaces. Results should be presented across r_g = 1.0,
   2.4, 5.0 km, with rural scenario as a primary sensitivity axis.

4. **⟨k²⟩ / tail bound:** The high-degree tail is the most under-captured quantity in
   all available empirical datasets. Any published τ_c value must be presented as a range
   driven by the tail uncertainty, not a point estimate. AC should review the sensitivity
   band produced by Stage 3 and set conservative / liberal bounds for the manuscript.

5. **NHBS receptive-sharing frequency breakdown:** The module has the any-sharing
   prevalence (27%) but needs the frequency distribution (every time / >half / <half)
   to set within-sharer intensity in the dyad model. This is likely in the NHBS MMWR
   67(1) supplementary tables. **Highest priority for next sourcing pass.**

6. **injection_frequency by drug class:** Stimulant (meth/cocaine) injection frequency
   differs from opioid. Strathdee 1997 and Des Jarlais 2020 document the direction;
   specific per-day counts by drug class are needed for the staged T model.

7. **`intensity_scale` in `assign_node_sharing_intensity()`:** Currently
   `Gamma(shape=0.5, scale=0.05)` — PLACEHOLDER. Will be retired once the staged-β
   model (Hollingsworth 2008 acute ×26, late ×7) is wired into `compute_per_edge_T`.
   See `params.py` entries `acute_multiplier`, `late_stage_multiplier`.

8. **Giant component gap:** Current dyad model produces only 29% giant-component
   fraction (vs 90%+ empirical). Cause: 71% of nodes have zero sharing intensity.
   Two options for Handoff 7: (a) add rare-sharing tier for non-zero-intensity nodes
   at low probability, (b) make kappa_dyad produce edges even for low-intensity pairs.

---

## T_dyad Staging — Sourcing (feeds Handoff 7 implementation)

Staged per-injection β: **T = 1 − (1−β_stage)^m** per stage, where
**m = injection_freq × sharing_fraction × window_days**.

| Stage | Multiplier vs β_chronic | Window | Source |
|---|---|---|---|
| Acute | ×26 | ~90 days | Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/590501 |
| Chronic/asymptomatic | ×1 (baseline) | partnership duration − acute − late | (baseline) |
| Late-stage | ×7 | ~270 days (19–10 mo pre-death) | Hollingsworth 2008 (ibid.) |

The acute ×26 window (~3 months) is the non-circular takeoff form: explosiveness
emerges from viremia staging rather than from a phenomenological cliff. This is what
validates against Scott County trajectory (18.6/100 py explosive incidence) forward,
not backward.

## Sentinel Ladder — Sourcing

Four pathogens ordered by effective R₀ threshold on one contact-intensity axis.
HIV is held out. Each rung provides an independent ordered constraint.

| Rung | Pathogen | Validation target | Source |
|---|---|---|---|
| 1 | Rectal GC | Subsequent HIV 4.1/100 py | Katz DA et al. Sex Transm Dis 43(2):91 (2016). DOI 10.1097/OLQ.0000000000000423 |
| 2 | Early syphilis | Subsequent HIV 2.8/100 py | Katz 2016 (ibid.) |
| 3 | **HIV** | **HELD OUT — forward prediction** | — |
| 4 | Sexual HCV | Incidence 1–4/100 py (HIV+ MSM) | Chaillon 2019 DOI 10.1093/ofid/ofz160; Wandeler 2012 DOI 10.1093/cid/cis694; Jansen 2015 DOI 10.1371/journal.pone.0142515 |

Composite STI cofactor: rectal STI → incident HIV aHR 2.7 [1.2–6.4], PAF 14.6%
(Kelley 2015 DOI 10.1089/AID.2015.0013). ONE cofactor on bridge nodes; GC + syphilis
are indicators, not independent transmission coefficients.

**Do NOT let the 4.1/100 py incidence figures enter as transmission coefficients.**
They are validation targets for the model's output, not inputs.

## Outbreak Validation Panel — Sourcing

Validate on trajectory (size / cluster / incidence / doubling), not a published R₀.

| Target | Value | Source |
|---|---|---|
| Final size (Scott Co.) | 215 | Peters PJ et al. N Engl J Med 373:2431 (2015). DOI 10.1056/NEJMoa1515195 |
| Final size (Cabell Co.) | 82 | McClung RP et al. Am J Prev Med 61(1):50 (2021). DOI 10.1016/j.amepre.2021.05.039 |
| Final size (NE Mass.) | 129 | Alpren C et al. Am J Public Health 110(1):37 (2020). DOI 10.2105/AJPH.2019.305366 |
| Cluster dominance | 93–99% in one cluster | Peters 2015; McClung 2021 |
| Degree→risk gradient | aRR 1.9 per syringe-sharing partner named | Peters 2015 |
| Explosive incidence | 18.6/100 py [11.1–26.0] | Strathdee SA et al. AIDS 11(8) (1997). DOI 10.1097/00002030-199708000-00001 |
| SIR removal rate | 0.024 /diagnosed/day | Gonsalves GS & Crawford FW. Lancet HIV 5(6):e297 (2018). DOI 10.1016/S2352-3018(18)30176-0 |
