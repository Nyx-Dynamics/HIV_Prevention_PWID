# HANDOFF 6 RESULT — persistent dyads + node-level sharing intensity

Branch: `mobility/network-generator` (1 commit: 2fb2188)

---

## 1. Did dyads lift clustering off the ER floor?

**Yes — dramatically.**

| Proxy | cc (dyads) | cc (before) | ER floor | cc/ER | Status |
|---|---|---|---|---|---|
| A — SSP-only | **0.0825** | 0.024 | 0.00942 | 8.8× | ABOVE_FLOOR |
| B — SSP+diffuse | **0.0791** | 0.008 | 0.00971 | 8.1× | ABOVE_FLOOR |
| **C — SSP+hotspot** | **0.1163** | 0.037 | 0.01622 | **7.2×** | **PASS** ✓ |

**Proxy C (concentrated hotspot) is the first to reach the 0.1–0.4 target.**

Did venue-anchoring fat the degree tail? Yes:

| Proxy | k² / ⟨k⟩² (dispersion) | Poisson baseline | Status |
|---|---|---|---|
| A | **5.14** | 1.35 | **PASS** |
| B | **5.33** | 1.34 | **PASS** |
| C | **4.85** | 1.21 | **PASS** |

All three proxies pass the dispersion check for the first time (previously 1.3 REVIEW).

**What created the improvement:** node-level intensity correlates edges across a node's neighborhood. High-intensity nodes cluster at venues → they form edges with multiple other high-intensity venue-visitors → triangles. This is the mechanism stated in the handoff. It works.

**Giant component gap (new issue):** All proxies have only 29% in the giant component. The cause: 73% of nodes have zero sharing intensity (non-sharers by the mixture model) and form no dyad edges. This differs from empirical networks where 45–100% of injectors are in the largest component (Buchanan, Klovdahl). **AC decision needed** — see §3.

---

## 2. R₀ off the subcritical floor?

**Yes — decidedly off, possibly too far. Interpret with the intensity-scale caveat.**

| Proxy | τ_c | T_dyad median | [5–95th pct] | R₀ median | [5–95th] | P(R₀>1) | Straddles 1 |
|---|---|---|---|---|---|---|---|
| A — SSP-only | 0.0740 | 0.568 | [0.13, 0.94] | **7.7** | [1.7, 12.7] | 98% | No |
| B — SSP+diffuse | 0.0689 | 0.573 | [0.13, 0.94] | **8.3** | [1.9, 13.6] | 99% | No |
| C — SSP+hotspot | 0.0443 | 0.524 | [0.11, 0.92] | **11.8** | [2.6, 20.7] | 99% | No |

**Off the subcritical floor:** Confirmed. Handoff 5 had P(R₀>1)=8% (decomposed) to 67% (T_edge). Dyad T now gives P(R₀>1)=98–99%.

**Caution — intensity scale is PLACEHOLDER:** R₀ = T_dyad / τ_c. T_dyad median ≈ 0.55 is driven by `intensity_scale=0.05` (mean within-sharer intensity = 0.025 per injection). If the true intensity is 5× lower (scale=0.01), T_dyad falls to ≈0.15, and R₀ falls to ≈2–3 (still supercritical but more plausible). If 10× lower (scale=0.005), T_dyad ≈ 0.06 and R₀ ≈ 0.8–1.5 — near-critical. **The intensity scale is the dial that controls this.**

τ_c is also low (0.04–0.07) because ⟨k²⟩ is large (41–115 — hub formation from dyad mechanism). This is structurally correct — heterogeneous networks have low τ_c — but the exact value depends on kappa_dyad calibration.

---

## 3. Frequency-data gap and placeholders

**What is sourced:**
- Mixing weight (27% any-sharing prevalence): NHBS 2018, Burnett JC et al. MMWR 67(1). ✓

**What is PLACEHOLDER (flagged, PENDING AC SIGN-OFF):**
- **Within-sharer intensity scale:** `Gamma(shape=0.5, scale=0.05)` — mean = 0.025. NHBS reports frequency categories (every time / >half / <half) but the exact % breakdown was not pre-loaded. The NHBS MMWR 67(1) tabulation likely has this in supplementary data. **This is the highest-priority number to retrieve.** It directly sets T_dyad and hence R₀.
- **kappa_dyad = 2.0:** Calibration scalar for dyad formation rate. Set to produce ⟨k⟩ ≈ 2–3 in typical conditions. **ANCHOR WARNING** — ⟨k⟩ is anchored to this, not a held-out check.

---

## 4. Decisions waiting for AC

1. **Intensity scale (highest priority):** Retrieve the NHBS frequency breakdown for receptive syringe sharing (every time / >half / <half as % of 12-mo sharers). Current scale=0.05 gives T_median=0.55 and R₀=8–12. At scale=0.01, R₀≈2–3. At scale=0.005, R₀ near-critical. This single number determines whether the thesis is "near-critical" or "comfortably supercritical."

2. **Giant component gap:** 29% vs 90%+ empirical. The fix options:
   - Increase kappa_dyad to allow lower-intensity pairs to form edges
   - Add a second tier: rare sharers (intensity ≈ 0) can still form edges with low probability (e.g., one-time sharing event)
   - The current mechanism requires BOTH nodes to have non-zero intensity — too strict

3. **Path length too short (2.0–2.3 vs target 3.1):** Dense dyad formation at venues creates short paths. Consistent with a venue-dominated network but not the small-world injection-network structure. Fixing the giant-component issue (which segments the network) would lengthen paths.

4. **Bridge subpopulation (AC design input):** The handoff notes this comes next. The dyad infrastructure is ready; the design choice is yours.
