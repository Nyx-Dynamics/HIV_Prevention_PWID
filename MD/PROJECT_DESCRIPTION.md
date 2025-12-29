# HIV Prevention Master Theorem Project

## Formal Project Description

**Principal Investigator:** A.C. Demidont, DO  
**Institution:** Nyx Dynamics LLC  
**Date:** December 24, 2025

---

## 1. PROJECT NAME

**Manufactured Death: Computational Modeling of Nested Policy and Algorithmic Barriers to HIV Prevention for People Who Inject Drugs**

*Core thesis:* HIV prevention research has inadvertently created a leave-one-out cross-validation framework at population scale. PWID are the systematically held-out test population; ML systems trained on this exclusionary literature cannot generalize to a population absent from their training data. The 8,833-fold disparity between PWID and MSM—using identical drug efficacy—is the mathematically inevitable consequence of this structure.

*Working titles for component manuscripts:*

- **Canonical manuscript:** "Manufactured Death: Computational Modeling of Nested Policy and Algorithmic Barriers to HIV Prevention for People Who Inject Drugs"

- **Viruses submissions:** "Computational Validation of a Clinical Decision Support Algorithm for LAI-PrEP Bridge Period Navigation at UNAIDS Scale" and "Bridging the Gap: The PrEP Cascade Paradigm Shift for Long-Acting Injectable HIV Prevention"

- **Framework terminology:** "Manufactured Death"—nested barriers (policy + implementation + algorithmic) that foreclose all biomedical pathways to R(0)=0 regardless of drug efficacy

---

## 2. PROJECT GOALS

### 2.1 Primary Theoretical Goal

**Prove mathematically that R(0)=0 is the unique closed-form solution to HIV prevention.**

This theorem establishes that prevention—not treatment, not functional cure, not decades of antiretroviral management—is the only mathematical path to epidemic termination. The proof is trivial but profound: if the initial reservoir R(0)=0, then R(t)=0 for all t. No other closed-form solution exists because the HIV reservoir, once established, is permanent and self-renewing via T-cell homeostatic proliferation.

### 2.2 Secondary Theoretical Goal

**Formalize the Nested Barrier Framework demonstrating that policy and algorithmic barriers create cascade attrition that drug efficacy cannot overcome.**

Three nested barrier layers operate multiplicatively:

- **Layer 1: Policy barriers** (52.5% of total attrition) — Criminalization creating fear of disclosure, defunded harm reduction infrastructure, and healthcare stigma

- **Layer 2: Implementation barriers** (25.4% of total attrition) — Cascade attrition from awareness through sustained engagement, reflecting the absence of PWID-specific implementation pathways

- **Layer 3: Algorithmic barriers** (22.1% of total attrition) — ML-mediated resource allocation that systematically deprioritizes PWID based on patterns learned from 44 years of exclusionary training data

**The complete probability model:**
```
P(protection) = ε × Π(cascade steps) × P(no incarceration) × P(algorithmic access)
```

Where P(algorithmic access) = 0.15 for PWID vs. 0.92 for MSM, reflecting the 120-fold signal-to-noise ratio disparity in training data.

These barriers are **multiplicative, not additive**. Each layer decays probability independently such that moderate barriers (30–55% at each step) compound to impossibility (0.006% completion with algorithmic mediation). The framework proves that even 99% effective drugs cannot achieve R(0)=0 when nested barriers decay cascade completion toward zero.

### 2.3 Applied Goal

**Validate a 21-intervention clinical decision support algorithm at UNAIDS global scale (21.2 million people) demonstrating mathematically necessary improvements for achieving R(0)=0 in high-burden populations.**

Key outcomes:
- PWID: +265% relative improvement (10.36% → 37.8%)
- Adolescents: +147% relative improvement
- Sub-Saharan Africa: +91.4% improvement (21.69% → 41.5%)
- Global average: +81.6% improvement (23.96% → 43.50%)

### 2.4 Policy Goal

**Expose the structural contradiction between CDC "same-day initiation" marketing claims and CDC testing guidelines that mathematically prevent same-day initiation.**

Demonstrate that this contradiction:
1. Creates conditions for capsid inhibitor monotherapy in undetected acute infections
2. Guarantees 100% resistance in breakthrough cases (PURPOSE-2 precedent)
3. Systematically excludes populations with highest HIV burden (PWID: 10.36% baseline success)
4. Maintains R(0) > 0 through guideline architecture rather than pathogen biology

### 2.5 Evidentiary Goal

**Document real-time evidence of policy-manufactured prevention failure using the Boston PWID HIV Cluster (2018–2024).**

The Boston outbreak (205+ cases over 6 years, 20 miles from Harvard) demonstrates:
- Standard individual-level interventions fail at outbreak scale
- Even diagnosed individuals transmit (33% detectable viral loads)
- System explicitly "cannot accomplish optimal engagement"
- Outbreak persists due to absence of outbreak-level infrastructure, not absence of resources

---

## 3. CURRENT STATE OF THE PROJECT

### 3.1 Completed Work

#### Theoretical Framework
- ✅ Prevention Theorem formalized: R(0)=0 ⟹ R(t)=0 ∀t (unique closed-form solution)
- ✅ **Inadvertent Leave-One-Out Cross-Validation (LOOCV) framework:** PWID identified as systematically held-out test population in HIV prevention research (2/11 trials, 0/9 Best Practices, 0 LAI-PrEP participants)
- ✅ **3-layer nested barrier model:** Policy (52.5%) + Implementation (25.4%) + Algorithmic (22.1%)
- ✅ **Signal-to-noise ratio analysis:** 120-fold disparity (MSM SNR=9,180 vs PWID SNR=76.4); undefined (∞) for LAI-PrEP
- ✅ **Algorithmic access probability:** P(alg|PWID)=0.15 vs P(alg|MSM)=0.92
- ✅ Sensitivity analysis: 75-90% ML-attributable attrition across plausible parameter range (0.10-0.25)
- ✅ "Manufactured Death" formally defined: nested barriers foreclosing all biomedical pathways to R(0)=0
- ✅ Human-AI collaboration framework explicitly disclosed per ICMJE 2024

#### Computational Validation
- ✅ 21-intervention algorithm developed with mechanism diversity scoring
- ✅ Progressive validation at four scales: 1K, 1M, 10M, 21.2M
- ✅ 100% unit test pass rate (18/18 edge cases)
- ✅ Convergence demonstrated: 1K (±2.6pp) → 21.2M (±0.018pp)
- ✅ Population-specific, setting-specific, and regional analyses complete
- ✅ Dose-response relationship for barriers quantified (R²=0.998)

#### Manuscript Preparation

**Viruses Journal (Under Review):**
- ✅ **viruses-4063895:** "Computational Validation of a Clinical Decision Support Algorithm for LAI-PrEP Bridge Period Navigation at UNAIDS Global Target Scale" — Under Review (submitted December 9, 2025) — SI: Long-Acting Antiretrovirals
- ✅ **viruses-4064402:** "Bridging the Gap: The PrEP Cascade Paradigm Shift for Long-Acting Injectable HIV Prevention" — Under Review (submitted December 10, 2025) — SI: Long-Acting Antiretrovirals

**Preprints (Pending Check):**
- ✅ **Preprints ID: 191398:** Computational Validation manuscript — Pending Check (December 24, 2025)
- ✅ **Preprints ID: 191275:** Bridging the Gap manuscript — Pending Check (December 24, 2025)

**Other Materials:**
- ✅ Supplementary materials S1–S5 complete (December 18, 2025 versions)
- ✅ Lancet correspondence draft complete (3,847 words)
- ✅ Boston outbreak case study documented with MDPH evidence

#### Supporting Documentation
- ✅ Signal-to-noise analysis (120:1 MSM:PWID disparity)
- ✅ Literature novelty analysis with citation strategy
- ✅ Figure set: cascades, workflow, policy scenarios, regional analysis

### 3.2 In Progress

#### Immediate Priorities
- ✅ Both Viruses manuscripts under review (viruses-4063895, viruses-4064402)
- ✅ Both preprints submitted, pending check (191398, 191275)
- 🔄 Lancet submission preparation—final formatting and reference verification

#### Short-Term (January 2026)
- ⏳ Lancet submission (Manufactured Death canonical manuscript)
- ⏳ Response to Viruses reviewer comments
- ⏳ PURPOSE-4 data monitoring (12+ months past endpoint, still unpublished)

### 3.3 Pending External Events

#### PURPOSE-4 Trial (NCT06101342)
- **Status:** Active, Not Recruiting
- **Expected:** Week 52 data (endpoint reached ~December 2024)
- **Prediction:** 1–3 seroconversions with 100% capsid resistance in 181 PWID
- **Significance:** If results match prediction, validates entire theoretical framework

#### Boston PWID Outbreak
- **Status:** Ongoing transmission (December 2024)
- **Monitoring:** MDPH alerts, case counts, viral suppression rates
- **Significance:** Real-time evidence of manufactured prevention failure

### 3.4 Project Outputs Summary

| Output | Status | ID | Target |
|--------|--------|-----|--------|
| Computational Validation | Under Review | viruses-4063895 | Viruses SI |
| Bridging the Gap | Under Review | viruses-4064402 | Viruses SI |
| Computational Validation Preprint | Pending Check | 191398 | Preprints |
| Bridging the Gap Preprint | Pending Check | 191275 | Preprints |
| Manufactured Death (canonical) | Ready | — | Lancet |
| Boston Case Study | Complete | — | Supporting |
| Mathematical Supplement | Complete | — | O'Neill framework |

### 3.5 Key Metrics

**From Canonical Manuscript (Manufactured Death COMPLETE_FINAL):**
- **Monte Carlo simulation:** n = 100,000 per scenario, 1,000 bootstrap iterations
- **Cascade steps modeled:** 8 sequential stages
- **PWID cascade completion (without ML):** 0.04%
- **PWID cascade completion (with ML):** 0.006%
- **MSM cascade completion:** 53%
- **Total disparity factor:** 8,833-fold (1,325× cascade × 6.7× algorithmic)
- **Signal-to-noise ratio:** MSM=9,180, PWID=76.4 (120-fold disparity)
- **P(algorithmic access):** PWID=0.15, MSM=0.92
- **Drug efficacy assumed:** 99%
- **ML-attributable attrition:** 75-90% (sensitivity range)
- **Barrier decomposition:** Policy 52.5%, Implementation 25.4%, Algorithmic 22.1%
- **Policy scenarios modeled:** 8 (current → full harm reduction + debiasing)
- **Best outcome (full HR + debiasing):** 31.2% cascade completion

**From Viruses Computational Validation:**
- **Scale validated:** 21.2 million people (UNAIDS global target)
- **Populations modeled:** 7 groups
- **Healthcare settings:** 8 types
- **Global regions:** 5
- **Interventions optimized:** 21
- **Structural barriers quantified:** 13
- **Improvement demonstrated:** +81.6% global average, +265% PWID

---

## 4. SIGNIFICANCE

This project establishes that HIV prevention failure for PWID reflects an **inadvertent leave-one-out cross-validation framework operating at population scale**:

1. **PWID are the held-out test population:** 2/11 trials (18%), 0/9 Best Practices studies (0%), 0 LAI-PrEP participants. This is not underrepresentation—it is systematic exclusion.

2. **ML systems cannot generalize to excluded populations:** The 120-fold SNR disparity means algorithms receive high-precision MSM evidence and low-precision (or zero) PWID evidence. Poor PWID outcomes are the mathematically inevitable consequence of training set exclusion, not evidence of a "hard to reach" population.

3. **The algorithmic barrier is invisible but substantial:** Traditional cascade analyses miss the 22.1% attrition attributable to ML-mediated deprioritization. Sensitivity analysis confirms 75-90% ML-attributable reduction across plausible parameters.

4. **Nested barriers compound to impossibility:** Policy (52.5%) × Implementation (25.4%) × Algorithmic (22.1%) = 0.006% cascade completion. No individual barrier exceeds 55%, yet the product approaches zero.

5. **The terminology "Manufactured Death" is precise:** Deaths are manufactured not by algorithmic bias but by experimental design—PWID were excluded from training data that defines "evidence-based" prevention. The algorithm performs correctly; it simply cannot generalize to populations it never encountered.

**The core reframing:** The "hard to reach" narrative misattributes a training set exclusion problem to a population characteristic. The solution is not debiasing algorithms trained on exclusionary data—it is including PWID in training data.

**Policy can change. Training data can change. The mathematics cannot.**

---

*Document version: 1.0*  
*Last updated: December 24, 2025*
