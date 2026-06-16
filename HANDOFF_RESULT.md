# HANDOFF RESULT — morning/refactor-consolidate-promote

Branch: `morning/refactor-consolidate-promote` (3 commits, not merged to main)

---

## Pre-existing issue (not introduced by this branch)

`reproduce_all.py` Step 3 (Sensitivity Analysis) was already failing before any change:

```
TypeError: CascadeStep.__init__() missing 1 required positional argument: 'barrier_layer'
```

Location: `src/cascade_sensitivity_analysis.py:140` — `sample_cascade_parameters()` constructs a `CascadeStep` without the `barrier_layer` argument that `CascadeStep.__init__` now requires. This was present on `main` before the branch was created. All baseline comparisons were made against the 5/6-step run (Step 3 excluded).

---

## Task 1 — README Quickstart clone URL

**Done.** Changed `Prevention-Theorem` → `HIV_Prevention_PWID` in the `git clone` URL and directory name. Commit: `1172689`.

---

## Task 2 — Promote 5 inline coefficients to named ParameterWithUncertainty

**Done. reproduce_all outputs identical to baseline (timestamp field only differs).**

Five coefficients promoted in `src/stochastic_avoidance_enhanced.py`, added to `KEY_PARAMETERS` with degenerate placeholder bounds (`lower_bound = upper_bound = point_estimate`) so behavior is numerically unchanged. Commit: `2ecb08e`.

| New param name | point_estimate | Location promoted from |
|---|---|---|
| `outbreak_escalation_rate` | 3.0 | `calculate_outbreak_probability` — `np.exp(3 * excess)` |
| `meth_density_weight` | 0.5 | `calculate_network_density` — `meth_prevalence * meth_multiplier * 0.5` |
| `housing_density_weight` | 0.3 | `calculate_network_density` — `housing_instability * 0.3` |
| `incarceration_density_weight` | 0.2 | `calculate_network_density` — `incarceration_rate * 0.2` |
| `sexwork_bridge_weight` | 0.15 | `calculate_network_density` — `meth_prevalence * 0.15` |

All five have `source="UNSOURCED — flagged for AC"` and are **not enrolled in any sensitivity sweep** (degenerate bounds make sweep runs meaningless until AC sets real ranges).

**Additional inline magic numbers found (not promoted, listed for AC):**

In `calculate_outbreak_probability`:
- `0.4` — SSP coverage effectiveness (`ssp_protection = 1 - (ssp_coverage * 0.4)`)
- `0.3` — OAT coverage effectiveness (`oat_protection = 1 - (oat_coverage * 0.3)`)
- `0.10` — prevalence normalization baseline (`prevalence_multiplier = 1 + (hiv_prevalence / 0.10)`)

These three were left in place pending AC decision on sourcing/promotion.

---

## Task 3 — Consolidate duplicate outbreak logic

**Clean removal — outputs identical to baseline.**

**Canonical module:** `src/stochastic_avoidance_enhanced.py` (`EnhancedStochasticAvoidanceModel`)

**Stale copy location:** `src/structural_barrier_model.py` — `StochasticAvoidanceModel` class (~L735) with `calculate_network_density` and `calculate_annual_outbreak_probability` methods.

**Caller inventory (grep across reproduce_all.py, scripts/, src/*):**

| Caller | Class used | Module |
|---|---|---|
| `reproduce_all.py` | `EnhancedStochasticAvoidanceModel` | `stochastic_avoidance_enhanced` |
| `scripts/reproduce_supplementary_results.py` | `EnhancedStochasticAvoidanceModel` | `stochastic_avoidance_enhanced` |
| `src/stochastic_avoidance_v2.py` | `EnhancedStochasticAvoidanceModel` (subclass) | `stochastic_avoidance_enhanced` |
| `src/hood_parameter_comparison.py` | `EnhancedStochasticAvoidanceModel` + V2 | `stochastic_avoidance_enhanced` |

**Zero reproduction-path callers of the stale `StochasticAvoidanceModel`.** The class was only used internally within `structural_barrier_model.py` (the `main()` script entrypoint, not the reproduction pipeline).

**Divergence found:** The stale copy lacked the `prevalence_multiplier` term present in the canonical version. No published result traced to the impoverished copy — `reproduce_all.py` never called it.

**Action taken:** Removed `StochasticAvoidanceModel` class and cleaned up `main()` / `__main__` references. A deprecation notice is left as a section header comment for blame/history. Reproduction outputs confirmed numerically identical to baseline. Commit: `67b1dbe`.

---

## Open items awaiting AC

1. **Set real bounds and source for the 5 promoted coefficients** (especially `outbreak_escalation_rate = 3.0`) before enrolling in sensitivity sweeps. Current bounds are degenerate placeholders.
2. **Source or justify `κ` (`critical_network_threshold = 0.35`)** — currently `source="Theoretical/modeling"`. Already in `KEY_PARAMETERS`; bounds need justification.
3. **Source or justify the `3` escalation rate** (now named `outbreak_escalation_rate`) — single-handedly drives superlinear blow-up above threshold. Highest-priority item per GAP_REPORT.
4. **Decide coupling form (Option B):** route `(1 − cascade_completion_rate)` into `calculate_network_density` as effective susceptible fraction so the paper's structural-barriers → cascade failure → network inflation → outbreak chain is the model's actual structure.
5. **Address the 3 additional inline coefficients** in `calculate_outbreak_probability` (SSP 0.4, OAT 0.3, prevalence normalization 0.10) — promote and source, or document rationale.
6. **Fix pre-existing Step 3 failure** (`CascadeStep.__init__()` missing `barrier_layer` argument in `src/cascade_sensitivity_analysis.py:140`). Not introduced by this branch; flagging for AC awareness.
