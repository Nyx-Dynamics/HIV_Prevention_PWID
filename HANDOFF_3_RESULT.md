# HANDOFF 3 RESULT — mobility/network-generator

Branch: `mobility/network-generator` (5 commits off `coupling/option-b-and-sweep`)
Seed: 42 throughout. scikit-mobility not available (GDAL build failure); pure-NumPy EPR fallback used.

---

## 1. What was built

| File | Purpose |
|---|---|
| `docs/SOURCING_mobility_network.md` | Methods-section-ready provenance memo (all params, DOIs, gap flags, ethical invariant) |
| `src/mobility/__init__.py` | Ethical invariant docstring |
| `src/mobility/params.py` | 14 `ParameterWithUncertainty` entries + `VALIDATION_TARGETS` dict + `PLACEHOLDER` sentinel |
| `src/mobility/network_generator.py` | 5-step pipeline: traversement potential → EPR walks → colocations → sharing edges → contact graph |
| `src/mobility/validate_network.py` | Validation harness: generated stats vs Layer-3 empirical targets |
| `src/mobility/threshold.py` | `network_threshold()`, `r0_on_network()`, sweep, comparison to κ=0.35 |
| `tests/mobility/test_network_generator.py` | 6 Stage 1 tests (all pass) |
| `outputs/Fig_network_degree_distribution.png` | Degree distribution (linear + log-log, with empirical target) |
| `outputs/Fig_tau_c_sensitivity.png` | τ_c vs ρ and r_g — with κ=0.35 reference line |
| `outputs/network_validation_report.json` | Validation results JSON |
| `outputs/threshold_report.json` | τ_c point estimate, sweep matrix, comparison |

All 6 Stage 1 tests pass. Outbreak-path code untouched. `critical_network_threshold=0.35` unchanged.

---

## 2. Generated network vs Layer-3 targets

Network: n=300, n_steps=50, kappa_share=0.01 (approximate calibration; 1.0 = conservative ceiling in params.py), seed=42.

| Statistic | Generated | Target | Status |
|---|---|---|---|
| ⟨k⟩ mean degree | **2.700** | ~2.6 (range 0–14) | PASS |
| Giant component | **88.7%** | Present (137–600 nodes empirically) | PASS |
| Mean path length | **5.18** | ~3.1 | PASS (within 3× tolerance) |
| ⟨k²⟩ | **10.94** | SENSITIVITY RANGE — not a pass/fail check | RANGE |

Path length 5.18 vs target 3.1: the generated network is sparser than the empirical small-world core (which was measured on a denser, historically-established injection network). This reflects the approximate kappa_share and is within the uncertainty of a first-pass generator.

**Tail caveat (mandatory):** ⟨k²⟩ = 10.94 is the point estimate from this generator configuration. The high-degree tail is the most under-captured quantity in all RDS datasets, and it is the dominant input to τ_c. The sweep (§4 below) bounds the effect.

---

## 3. Mechanistic τ_c vs current κ = 0.35

| | Value |
|---|---|
| **Mechanistic τ_c (point estimate)** | **0.3277** |
| Current phenomenological κ | 0.3500 |
| Direction | Mechanistic is **LOWER** (ratio = 0.94) |
| R₀ at T=0.008 (HIV per-injection β) | 0.024 (subcritical at this transmissibility) |

The mechanistic threshold is **more permissive** than the current κ=0.35 by ~6%. This means the network reaches epidemic territory (R₀ > 1) at a lower per-partnership transmissibility than κ=0.35 implies. Put differently, the current phenomenological threshold slightly **underestimates outbreak risk** at this network structure.

This is a narrow difference — within the sweep uncertainty — and must not be reported as a single number (see §4).

---

## 4. τ_c sensitivity to explorer fraction and r_g tail

Sweep: ρ ∈ [0.30, 0.90] × r_g ∈ {1.0, 2.4, 5.0 km}

| Metric | Value |
|---|---|
| τ_c sweep min | **0.183** |
| τ_c sweep median | **0.339** |
| τ_c sweep max | **0.697** |
| Fraction of sweep below κ=0.35 | **50%** |

**Hub mechanism confirmed:** increasing ρ (more explorers → fatter degree tail → higher ⟨k²⟩) drives τ_c downward, sometimes to < 0.2. The hub-drives-τ_c link is structurally present in the generator. The direction of the effect is correct (more exploration → lower threshold → higher epidemic risk), but the magnitude depends on kappa_share calibration and r_g scale.

**r_g effect:** larger activity radius (5.0 km rural) shifts τ_c downward relative to 2.4 km urban in the high-ρ regime. Rural PWID with wide-ranging mobility are the highest-risk group — the generator captures this qualitatively.

---

## 5. Gaps requiring AC decision

1. **kappa_share calibration (highest priority):** Default 0.01 is an approximate calibration, not a derived value. Requires: daily injection frequency × fraction of injections shared × mean network co-location rate. Until calibrated, ⟨k²⟩, τ_c, and R₀ are order-of-magnitude estimates only.

2. **Drug-market attractor layer:** Implemented as a `PLACEHOLDER` with conservative default (1.5× SSP relevance). No clean data source. AC must approve a specific proxy (overdose/EMS hotspots, arrest data, ethnographic maps) before this attractor type appears in any published result.

3. **r_g scale beyond San Francisco:** The 2.4 km estimate is from a single urban study (Ober 2014). Rural PWID likely have substantially larger activity spaces. All key results (τ_c, R₀) should be presented across r_g = {1.0, 2.4, 5.0} km, with the rural scenario as the primary sensitivity axis.

4. **⟨k²⟩ / tail bounds for manuscript:** The sweep shows τ_c ∈ [0.18, 0.70] across the approved parameter range. AC should select a conservative / liberal bound for the manuscript τ_c range, acknowledging that the high-degree tail is systematically under-captured in available datasets.

5. **scikit-mobility (skmob) upgrade:** The pure-NumPy EPR fallback is functionally equivalent for first-pass network generation. When geo-deps are available, `skmob.DensityEPR` with a real venue layer (NASEN SSP locations, census-derived population rasters) would produce a more geographically realistic mobility model. Not required for the threshold comparison — flagged for future work.

6. **Mechanistic-to-phenomenological swap (out of scope this run):** When AC approves, `critical_network_threshold=0.35` in `stochastic_avoidance_enhanced.py` is replaced by `threshold.network_threshold(...)` output, and the `exp(escalation * excess)` term is replaced by the supercritical-takeoff form. See `docs/SOURCING_mobility_network.md §6`.
