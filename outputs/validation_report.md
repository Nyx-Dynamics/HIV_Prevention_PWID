# Validation Report
Run dir: `/Users/acdstudpro/air/HIV_Prevention_PWID/outputs`
Timestamp: 2026-06-17T12:56:30.908386

## Schema Discovery
| Artifact | Source file | Status |
|---|---|---|
| network_stats | network_validation_report.json | realized name |
| outbreak_sim | — | NOT EMITTED |

## SCORED Targets
| Target | Model value | Target range | Status | CI overlaps | Median in range |
|---|---|---|---|---|---|
| giant_component_fraction | 0.8866666666666667 | (0.45, 1.0) | **PASS** | None | True |
| clustering_coefficient | 0.0091 | (0.1, 0.4) | **FAIL** | None | False |
| mean_path_length | 5.180451127819549 | (2.0, 6.0) | **PASS** | None | True |
| dispersion_k2_over_k2 | 1.501 | (2.07, '∞') | **FAIL** | None | False |
| final_size | not_emitted | — | **NOT_EMITTED** | None | None |
| single_cluster_fraction | not_emitted | — | **NOT_EMITTED** | None | None |
| degree_risk_gradient | not_emitted | — | **NOT_EMITTED** | None | None |
| explosive_incidence | not_emitted | — | **NOT_EMITTED** | None | None |

**SCORED pass rate: 2/4 (0.5)** (4 not emitted yet)

## HELD_OUT — Forward Predictions (never scored)
| Key | Model value | Note |
|---|---|---|
| hiv_r0 | not_emitted | HIV forward prediction — reported here, NEVER scored. |
| hiv_p_r0_gt1 | not_emitted | HIV forward prediction — reported here, NEVER scored. |

## PROVENANCE
| Parameter | Value | Source |
|---|---|---|
| sir_removal_rate | 0.024 | Gonsalves GS & Crawford FW. Lancet HIV 5(6):e297 (2018). DOI |
| composite_sti_cofactor_ahr | 2.7 | Kelley CF et al. AIDS Res Hum Retroviruses 31(10):1009 (2015 |
| beta_chronic | 0.008 | Baggaley RF et al. AIDS (2006). DOI 10.1097/01.aids.00002185 |
| acute_multiplier | 26.0 | Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/ |
| late_stage_multiplier | 7.0 | Hollingsworth TQ et al. Nat Med 14:1096 (2008). DOI 10.1086/ |
| sharing_prevalence | 0.27 | Burnett JC et al. MMWR 67(1) (2018). DOI 10.15585/mmwr.mm670 |

## STUBBED
| Target | Status | Note |
|---|---|---|
| rectal_gc_hiv_incidence | not_computable_yet | Requires bridge / multi-pathogen layer (Handoff 7+). |
| syphilis_hiv_incidence | not_computable_yet | Requires bridge / multi-pathogen layer (Handoff 7+). |
| sexual_hcv_incidence | not_computable_yet | Requires bridge / multi-pathogen layer (Handoff 7+). |
| hcv_consistency_check | not_computable_yet | Requires bridge / multi-pathogen layer (Handoff 7+). |

> HELD_OUT items (HIV R₀, HIV incidence) appear in the 'held_out' section and are NOT counted in scored_pass_rate. HIV outbreak frequency remains the held-out forward prediction.