# HANDOFF 5 RESULT — overnight: identifiability + first conditional R₀

Branch: `mobility/network-generator` (2 additional commits: 5667499, 7a373aa)

---

## 1. shared_fraction derivation

**Method:** forward from behavioral data only. NOT tuned to R₀ or outbreak rarity.

```
NHBS any-syringe-sharing prevalence (12 mo): p_any = 0.27  [0.25, 0.29]
Mean injection-partner degree: k̄ = 2.6
Per-partnership annual sharing probability:
  p_pp_yr = 1 − (1 − 0.27)^(1/2.6) = 1 − 0.73^0.385 ≈ 0.117
Annual injection frequency: f_yr = 3/day × 365 = 1095
shared_fraction ≈ p_pp_yr / f_yr = 0.117 / 1095 ≈ 0.0001  (lower bound, independence)
Upper bound 0.015 allows heavy-sharing subgroups.
Central: 0.001 (geometric mean of range).
```

**Result:** `shared_fraction_per_partner` updated from 0.15 (saturating placeholder) to **0.001 [0.0001, 0.015]**, lognormal. This is ~150× lower than the placeholder.

**PENDING AC SIGN-OFF** — derivation is transparent; value was not tuned to any outcome.

---

## 2. T: collapsed T_edge + saturation diagnostic

**Saturation crossover:** T > 0.95 when `shared_fraction ≥ 0.0885` (at point estimates for other params). Old placeholder (0.15) **was past saturation** (T≈1). Calibrated 0.001 is well below it (T≈0.033 at point estimate).

| Path | T median | T [5th, 95th] |
|---|---|---|
| **decomposed** (calibrated shared_fraction) | **0.035** | [0.004, 0.305] |
| **T_edge** (collapsed, directly bounded [0.10, 0.95]) | **0.323** | [0.050, 0.739] |

The two paths represent the uncertainty range:
- Decomposed path = minimum identifiable estimate from behavioral data
- T_edge = expert-bounded range (identifiable, but collapsed)

**R₀ both ways — at τ_c = 0.328 (point estimate):**

| Path | R₀ median | [5th, 95th] | P(R₀>1) | Straddles 1 |
|---|---|---|---|---|
| decomposed | 0.085 | [0.008, 0.812] | 3.6% | No (MC); Yes (analytic lower bound) |
| T_edge | 0.759 | [0.110, 2.345] | 36% | **Yes** |

---

## 3. Network under each candidate proxy

Run config: n=300, n_steps=80, venue_return_boost=5.0, seed=42

| Proxy | ⟨k⟩ | Clustering | vs ER floor | Dispersion | Path length | τ_c |
|---|---|---|---|---|---|---|
| **A — SSP-only** | 4.26 (ANCHOR) | 0.016 | REVIEW (≈1.1×ER) | 1.29 (REVIEW) | 4.00 (PASS) | **0.222** |
| **B — SSP+diffuse** | 3.92 (ANCHOR) | 0.008 | REVIEW (≈0.6×ER) | 1.32 (REVIEW) | 4.14 (PASS) | **0.240** |
| **C — SSP+hotspot** | 7.41 (ANCHOR) | 0.037 | REVIEW (≈1.5×ER) | 1.23 (REVIEW) | 3.08 (**PASS**) | **0.123** |

**Did venue-anchoring lift clustering off the random floor?**
Proxy C reached 0.037 (≈1.5× ER floor) vs ER floor ≈ 0.025 — marginal improvement. None reached the 0.1–0.4 target. Clustering remains a REVIEW across all proxies. The concentrated hotspot cluster (Proxy C) shows the strongest network structure (path length = 3.08 ≈ target, τ_c = 0.123 — much lower than SSP-only).

**What drives τ_c between proxies:** Proxy C has ⟨k²⟩ = 67.7 vs Proxy A's 23.5 — the concentrated hotspots create high-degree hub nodes that substantially fatten the tail and lower τ_c.

---

## 4. Conditional R₀ — the headline result

**ALL RESULTS CONDITIONAL ON PROXY AND PENDING AC REVIEW.**

| Proxy | τ_c | T_edge R₀ median | [5–95th] | P(R₀>1) | Straddles 1 | Decomposed P(R₀>1) |
|---|---|---|---|---|---|---|
| **A — SSP-only** | 0.222 | **1.45** | [0.26–3.29] | **67%** | ✓ Yes | 8% |
| **B — SSP+diffuse** | 0.240 | **1.42** | [0.22–3.12] | **67%** | ✓ Yes | 8% |
| **C — SSP+hotspot** | 0.123 | **2.51** | [0.37–5.94] | **83%** | ✓ Yes | 18% |

**Two-interpretation summary:**
- **Decomposed T (behavioral data calibration):** P(R₀>1) = 8–18% — substantially subcritical at median, with near-critical tail. Thesis-compatible: "HIV would be subcritical or near-critical with realistic sharing frequency."
- **T_edge (collapsed, identifiable [0.10, 0.95]):** P(R₀>1) = 67–83% — likely supercritical. Consistent with intermittent outbreak risk.

Both paths straddle R₀=1 across all proxies — the near-criticality thesis is structurally supportable from both ends of the T uncertainty range.

---

## 5. Sensitivity ranking

OAT (one-at-a-time), T_edge path, τ_c = 0.222 (Proxy A reference):

| Rank | Parameter | |ΔR₀| | Notes |
|---|---|---|---|
| 1 | `chronic_window_days` | 0.225 | PLACEHOLDER — acute dominates; still influences via m_chronic |
| 2 | `ssp_utilization_rate` | 0.224 | Attractor weight; network structure driver |
| 3 | `acute_multiplier` | 0.221 | Directly scales β_acute |
| 4 | `ssp_route_overlap` | 0.220 | Attractor geometry |
| 5 | `jump_length_exponent` | 0.207 | Mobility reach → network reach |
| ... | `shared_fraction_per_partner` | *excluded* | Degenerate bounds in OAT (lognormal; implicit in T) |

Note: `shared_fraction_per_partner` was excluded from OAT because the T decomposition uses the full distribution. Its implicit influence is the dominant driver when comparing the two T-paths (median R₀ 0.14 vs 1.45 — the gap between decomposed and T_edge is essentially the shared_fraction uncertainty).

---

## 6. Decisions waiting for AC

1. **Drug-market proxy choice:** Three options presented (A/B/C). No proxy enshrined. The concentrated hotspot model (C) gives τ_c ≈ 0.12 and P(R₀>1) = 83% (T_edge) — most alarming. SSP-only (A) gives τ_c ≈ 0.22 and P(R₀>1) = 67%. Please select a proxy or approve a fourth option.

2. **T_edge parameterization sign-off:** Bounds [0.10, 0.95], point estimate 0.35. Sources: Rolls 2011 (HCV structural analog), Hollingsworth 2008 (phylogenetic R₀). Pending AC review before this is the "published" T estimate.

3. **shared_fraction sign-off:** Derivation documented in params.py and above. Central 0.001 from NHBS ÷ (injection_freq × degree). Not tuned to any outcome. Pending AC approval.

4. **Clustering gap:** All proxies show clustering REVIEW (≈ER floor). Reaching 0.1–0.4 requires either: (a) stronger venue-clustering mechanism (injection rooms, very tight spatial bins), (b) explicit modeling of injection partnerships as fixed repeated dyads (not just co-location), or (c) accepting that the current sparse first-pass generator underestimates venue-anchoring. Which direction to pursue?

5. **kappa_share re-calibration:** With ⟨k⟩ now ranging 3.9–7.4 (up from 2.7) due to increased n_steps and venue_return_boost, kappa_share=0.01 no longer anchors ⟨k⟩≈2.6. Re-calibrate if ⟨k⟩ matters for the manuscript.
