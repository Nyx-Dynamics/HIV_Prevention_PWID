# HANDOFF 4 RESULT — mobility/network-generator (corrections)

Branch: `mobility/network-generator` (3 additional commits: dffa45e, a224786, 25bc2cf)

---

## Fix 1 — Corrected R₀: per-edge T (kills R₀=0.024 artifact)

**Corrected R₀:**
| Metric | Value |
|---|---|
| T median (per-edge, duration-integrated, acute-weighted) | **0.991** [5–95th pct: 0.706–1.000] |
| R₀ median | **2.16** |
| R₀ [5–95th pct] | **[1.36, 4.56]** |
| P(R₀ > 1) | **0.991** |
| Credible interval straddles R₀=1 (MC) | **No** — see note |

**Sanity-band guard confirmed:** T median = 0.99 (not 0.02). Guard fires if T < 0.05 (4/4 tests pass).

**Near-criticality argument is analytic, not MC:** Independent MC sampling concentrates T near 1.0 because `shared_fraction_per_partner = 0.15` (a PLACEHOLDER) gives m_acute ≈ 35, saturating T. The straddling argument:

- At **joint lower bounds** (inj_freq=1, shared_frac=0.05, acute_dur=49, mult=8): T_low ≈ 0.13, τ_c_max = 0.70 → **R₀ ≈ 0.185 < 1** (subcritical achievable ✓)
- At **point estimates**: T ≈ 0.99, τ_c_min = 0.18 → **R₀ ≈ 5.5 > 1** (supercritical ✓)

MC CI will straddle 1 once `shared_fraction_per_partner` is calibrated (currently PLACEHOLDER=0.15; calibration requires daily injection frequency × sharing fraction / co-location rate — open item, see §Gaps).

**Formula encoded:**
```
β_acute   = min(β_chronic × acute_multiplier, BETA_CAP=0.5)
m_acute   = injection_freq × shared_fraction_per_partner × acute_duration_days
m_chronic = injection_freq × shared_fraction_per_partner × chronic_window_days
T = 1 − (1−β_acute)^m_acute × (1−β_chronic)^m_chronic
```

**New params in params.py:**
- `beta_chronic_per_shared_injection` (0.008, [0.006, 0.024], lognormal — renamed from `hiv_transmissibility`)
- `acute_multiplier` (15.0, [8, 26], lognormal — Corner 2 / Hollingsworth / Wawer / Pinkerton)
- `acute_duration_days` (77.0, [49, 112], normal — Fiebig staging)
- `injection_freq_per_day` (3.0, [1, 6], lognormal — NHBS; wide variance flagged)
- `shared_fraction_per_partner` (0.15, [0.05, 0.30], beta — **PLACEHOLDER**; defined once, referenced from both T and kappa_share)
- `chronic_window_days` (180, [60, 365], lognormal — **PLACEHOLDER**; acute dominates)
- `BETA_CAP = 0.5` (hard constraint)

**Citation fix:** "Ober 2014" corrected to **Cooper & Tempalski (2014)** — verified against DOI 10.1016/j.drugpo.2013.11.008.

---

## Fix 2 — T vs τ_c replaces the κ comparison

**Confirmed purged** from all code, JSON, and figures:
- `kappa_phenomenological` key
- `ratio` (0.94) and direction ("LOWER/HIGHER than κ")
- κ=0.35 reference line on `Fig_tau_c_sensitivity.png`
- "fraction of sweep below κ=0.35" metric
- "more permissive / underestimates risk" language

**Replaced by:** `T vs τ_c` (same units). Figure reference line = T_median = 0.991 (the R₀=1 locus: where τ_c curves cross T_median, the network transitions from sub- to supercritical).

**One retained explanatory sentence:** "κ=0.35 is a DENSITY threshold — different dimension from τ_c (transmissibility) — will be RETIRED, not matched, in the eventual swap."

---

## Fix 3 — Anchor vs held-out decoupled

**Anchor (explicit, not a held-out check):**
- `mean_degree` = 2.700 — labeled `ANCHOR` (kappa_share=0.01 was set to produce ⟨k⟩≈2.6; reporting as PASS is circular)

**Held-out checks (not pinned by anchoring ⟨k⟩): 4/4 PASS**

| Statistic | Generated | Target | Status |
|---|---|---|---|
| Giant component | 88.7% | Present | PASS |
| Mean path length | 5.18 | ~3.1 (×3 tolerance) | PASS |
| **Dispersion ⟨k²⟩/⟨k⟩²** | **1.501** | **> 1.0 (over-dispersed)** | **PASS** |
| Clustering coefficient | 0.0091 | > 0 | PASS |

Dispersion is the **τ_c-relevant held-out check** — it tests the quantity that drives τ_c without being anchored to the mean.

**kappa_share status documented in params.py:** `kappa_share=0.01` is anchor-set; the circularity risk is explicitly flagged. `shared_fraction_per_partner` is defined once and referenced by both T integration and kappa_share to prevent drift.

---

## Updated gaps for AC

1. **`shared_fraction_per_partner` calibration (highest priority):** Defined as PLACEHOLDER=0.15. This is co-shared with kappa_share and T integration. Derive from: NHBS syringe sharing prevalence / (injection_freq × mean degree × co-location_rate). Until calibrated: m_acute ≈ 35 → T near saturation → MC CI does not straddle 1 → near-criticality argument is analytic only.

2. **`chronic_window_days` calibration:** PLACEHOLDER=180. Bounded by partnership duration and time-to-diagnosis. Acute phase dominates IDU outbreak dynamics; secondary priority.

3. **`injection_freq_per_day` sourcing:** Wide variance [1, 6]. Current source is NHBS + general literature. PWID-specific daily injection frequency data would tighten this.

4. **Drug-market attractor layer:** Still PLACEHOLDER. AC must approve a specific proxy before use.

5. **r_g scale beyond San Francisco:** 2.4 km from a single urban study (Cooper & Tempalski 2014, corrected). Rural PWID activity spaces likely substantially larger.

6. **kappa_share → derived value:** Once shared_fraction_per_partner is calibrated, kappa_share can be derived independently and ⟨k⟩ becomes a genuine held-out check.

---

## Branch note

This branch (`mobility/network-generator`) sits on `coupling/option-b-and-sweep` for eventual merge planning. The full stack is: `morning/refactor-consolidate-promote` → `coupling/option-b-and-sweep` → `mobility/network-generator`.
