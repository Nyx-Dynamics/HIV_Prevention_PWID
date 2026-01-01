# HIV Prevention Barrier Modeling Project Guide
## Claude Instructions and Feature File (CIFF)

### Project Overview

This project develops computational models demonstrating that structural barriers prevent effective HIV prevention for people who inject drugs (PWID) despite highly efficacious pharmacological interventions. The work supports a manuscript submitted to **The Lancet HIV**.

**Manuscript ID**: thelancethiv-D-25-00576  
**Submitted**: December 29, 2025

---

## Core Research Thesis

**Mathematical Proof**: Under current U.S. policy conditions, achieving R₀ = 0 (elimination of HIV acquisition risk) for PWID is mathematically impossible due to barrier multiplication across the prevention cascade.

**Key Insight**: The novel contribution isn't identifying barriers (documented by experts for decades) but **quantifying the timeline before stochastic avoidance fails** (~63% probability of major outbreak within 5 years).

---

## Key Files Reference

### Code
| File | Purpose |
|------|---------|
| `architectural_barrier_model.py` | Main Monte Carlo simulation (refactored from `manufactured_death_model.py`) |
| `visualize_md_results.py` | Publication figure generation |
| `PEP__mucosal.py` | Supporting PEP pharmacokinetic modeling |

### Outputs
| File | Contents |
|------|----------|
| `architectural_barrier_results.json` | Full Monte Carlo outputs (100K iterations) |
| `pwid_simulation_results.json` | PWID-specific scenario analyses |
| `*_results.csv` | Summary tables for manuscript |

### Manuscript
| File | Status |
|------|--------|
| `lancet_hiv_manuscript_acd_2025.tex` | Main manuscript LaTeX |
| `manuscript_lancet_hiv_2025_acd.pdf` | Compiled PDF (thelancethiv-D-25-00576) |
| `supplement_S1_lancet_hiv_2025_acd.pdf` | Mathematical foundations supplement |

### Figures (Publication-Ready)
- `Fig1_CascadeComparison.png` - PWID vs MSM cascade comparison
- `Fig3_PolicyScenarios.png` - Policy intervention analysis
- `Fig5_SNR_LOOCV.png` - Signal-to-noise validation

---

## Three-Layer Barrier Framework

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: PATHOGEN BIOLOGY                                   │
│ • HIV establishes R₀ > 0 within hours (irreversible)        │
│ • No intervention restores R₀ = 0 post-exposure             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: HIV TESTING FAILURES                               │
│ • Acute infection detection gaps                            │
│ • RNA testing not standard of care                          │
│ • 4th gen Ag/Ab misses first 2-4 weeks                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: ARCHITECTURAL BARRIERS                             │
│ ├── Policy: Criminalization, incarceration                  │
│ ├── Stigma: Healthcare discrimination, disclosure barriers  │
│ ├── Infrastructure: MSM-centric cascade design              │
│ ├── Research: LOOCV framework (systematic exclusion)        │
│ └── ML/AI: Algorithmic deprioritization (WMD framework)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Eight-Step Prevention Cascade

Each step has barrier probability that multiplies:

```
P(sustained protection) = Π₁⁸ P(pass step i)

Step 1: Awareness of LAI-PrEP availability
Step 2: Willingness to seek care (despite system visibility)
Step 3: Healthcare access
Step 4: Disclosure of injection drug use
Step 5: Provider willingness to prescribe
Step 6: Adequate HIV testing (exclude acute infection)
Step 7: First injection received
Step 8: Sustained engagement
```

**Result**: Even with 90% success at each step: 0.90⁸ = 43%
With PWID-specific barriers (70-95% per step): ~0.003%

---

## Policy Scenarios (Key Results)

| Scenario | P(R₀=0) | Notes |
|----------|---------|-------|
| Current Policy | 0.003% | Baseline |
| Decriminalization Only | 0.20% | Single intervention |
| Decrim + Stigma Reduction | 0.45% | Combined |
| SSP-Integrated Delivery | 5.0% | Infrastructure change |
| Full Harm Reduction | 9.5% | Policy + Stigma + Infrastructure |
| Full HR + PURPOSE-4 Data | 11.9% | Add missing efficacy data |
| Full HR + Algorithmic Debiasing | 18.6% | Remove ML bias |
| Theoretical Maximum | 19.7% | All barriers removed |
| **MSM (Comparison)** | **16.3%** | Same drug, different population |

---

## Terminology Decisions

**Use "Architectural Barriers"** — NOT "Manufactured Death"

A colleague warned that "manufactured death" framing would trigger immediate rejection as advocacy. The mathematics are identical; the framing is strategic.

| Avoid | Use Instead |
|-------|-------------|
| Manufactured death | Architectural barriers |
| Systematic murder | Structural failures |
| Policy violence | Policy-determined outcomes |

The evidence speaks for itself without inflammatory language.

---

## Key Literature References

| Topic | Primary Source |
|-------|----------------|
| Global PWID epidemiology | Degenhardt et al. (2017), *Lancet* |
| Criminalization effects | DeBeck et al. (2017), *Lancet HIV* |
| PrEP cascade failures | Mistler et al. (2021), *AIDS Behav* |
| Incarceration HIV risk | Altice et al. (2016), *Lancet* |
| LAI-PrEP efficacy | PURPOSE-1/2, Bekker et al. (2024), *NEJM* |
| Healthcare stigma | Biancarelli et al. (2019), *Soc Sci Med* |
| Algorithmic bias | O'Neil (2016), *Weapons of Math Destruction* |

---

## Claude Guidance for This Project

### When Helping with Code
- The model uses **NumPy random seed 42** for reproducibility
- Monte Carlo runs 100,000 iterations with 10,000 bootstrap replicates
- Output JSON files must match the schema in existing results files
- Figures follow **Lancet style**: 300 DPI, Arial font, specific dimensions

### When Helping with Writing
- Use **Lancet HIV** formatting guidelines
- Cite all claims with literature-derived parameters
- Maintain mathematical rigor while accessible prose
- Frame as **validation of expert consensus** + **novel timeline quantification**

### When Discussing Results
- Core finding: **0.003% vs 16.3%** (PWID vs MSM, same drug)
- The 5,000x disparity is structural, not individual
- Emphasize **policy-determined** outcomes, not choice
- Timeline to outbreak failure (~5 years) is the key novel contribution

### Sensitive Topics
- This research documents **preventable deaths** in marginalized populations
- Maintain scientific objectivity while acknowledging human impact
- The math proves what advocates have said for decades
- Frame as "evidence for policy change" not "accusation"

---

## GitHub Repository

**URL**: https://github.com/Nyx-Dynamics/hiv-prevention-master

### Repository Structure (Target)
```
hiv-prevention-master/
├── README.md
├── CITATION.cff
├── LICENSE (MIT)
├── RELEASE_INSTRUCTIONS.md
├── requirements.txt
├── architectural_barrier_model.py
├── visualize_md_results.py
├── data/
│   ├── architectural_barrier_results.json
│   └── pwid_simulation_results.json
├── figures/
│   ├── Fig1_CascadeComparison.png
│   ├── Fig3_PolicyScenarios.png
│   └── ...
└── manuscript/
    ├── lancet_hiv_manuscript_acd_2025.tex
    └── unified_bibliography.bib
```

---

## Quick Commands

### Run simulation
```bash
python architectural_barrier_model.py
```

### Generate figures
```bash
python visualize_md_results.py
```

### Git workflow
```bash
git add .
git commit -m "Description of changes"
git push origin main
```

### Create release
```bash
git tag -a v1.0.0 -m "Lancet HIV submission"
git push origin v1.0.0
```

---

## Contact

**AC Demidont, DO**  
CEO, Nyx Dynamics LLC  
Email: acdemidont@nyxdynamics.org  
ORCID: 0000-0002-9216-8569
