# HIV Prevention Master Algorithm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18116991.svg)](https://doi.org/10.5281/zenodo.18116991)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Computational models for HIV prevention barrier analysis and policy simulation among people who inject drugs (PWID).

## Overview

This repository contains Monte Carlo simulation models demonstrating how policy, stigma, and infrastructure barriers compound across an 8-step prevention cascade to produce near-zero population-level effectiveness despite highly efficacious pharmacological interventions. The model quantifies the "manufactured death" phenomenon where structural barriers, not biology or individual choice, determine prevention outcomes.

### Key Findings

| Metric | Value |
|--------|-------|
| Current PWID prevention rate | ~0.003% |
| MSM comparison (identical intervention) | 16.3% |
| Disparity factor | 5,434-fold |
| 5-year outbreak probability | 63% |

## Theoretical Background

This structural barrier analysis builds upon the **Prevention Theorem** framework developed in earlier work (Demidont, 2025). The Prevention Theorem establishes the biological constraint that HIV prevention (achieving R₀ = 0) is only possible before proviral integration into host DNA.

### Core Principle

Prevention requires:
- **R₀(e,t) = 0** — complete prevention of infection establishment
- This is achievable only while **P_int(t) < 1** (before irreversible integration)

Once HIV integrates into host chromosomes (typically within 24-72 hours depending on exposure route), the individual transitions to an **irreducible state** where R₀ > 0 regardless of subsequent intervention.

### Relationship to Structural Barriers

The current analysis extends this framework by asking: *Given that prevention is biologically time-limited, what structural factors prevent timely intervention?*

| Framework | Question Addressed |
|-----------|-------------------|
| **Prevention Theorem** | *When* must prevention occur? (biological constraints) |
| **Structural Barrier Model** | *Why* doesn't prevention occur in time? (policy constraints) |

### Files Derived from Prevention Theorem Work

The `AIDS and Behavior/` directory contains figures and code from the Prevention Theorem analysis:

- `prevention_theorem_figures.py` — Generates theoretical framework visualizations (Figure 1: establishment dynamics, Figure 2: window compression)
- `pep_mucosal.py` — PEP timing analysis extending the theorem to exposure-route-specific windows

These materials provide context for understanding why the narrow biological window makes structural barriers so consequential.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Nyx-Dynamics/HIV-Prevention_Master-Algorithm.git
cd HIV-Prevention_Master-Algorithm

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import numpy, scipy, matplotlib, pandas; print('All dependencies installed successfully')"
```

## Reproducing Results

### Step 1: Run Main Simulation

Generate the core Monte Carlo results for all policy scenarios:

```bash
python architectural_barrier_model.py --output-dir ./results --n-individuals 100000
```

**Arguments:**
- `--output-dir`: Directory for output files (default: current directory)
- `--n-individuals`: Simulated individuals per scenario (default: 100,000)
- `--n-sa-sims`: Stochastic avoidance simulations (default: 10,000)

**Outputs:**
- `architectural_barrier_results.json` - Full simulation data
- `architectural_barrier_results.csv` - Tabular summary

### Step 2: Run Sensitivity Analysis

Perform probabilistic sensitivity analysis and barrier removal analysis:

```bash
python cascade_sensitivity_analysis.py --output-dir ./outputs --n-samples 1000
```

**Arguments:**
- `--output-dir`: Output directory (default: `outputs`)
- `--n-samples`: PSA samples (default: 1,000)

**Outputs:**
- `cascade_sensitivity_results.json/csv` - PSA results
- Figures: `FigS4_CascadeUncertainty.png`, `FigS5_BarrierWaterfall.png`, `FigS6_StepImportance.png`

### Step 3: Run Stochastic Avoidance Model

Forecast outbreak probabilities and methamphetamine prevalence trajectories:

```bash
python stochastic_avoidance_enhanced.py --output-dir ./outputs --n-sims 2000 --n-psa 500
```

**Arguments:**
- `--output-dir`: Output directory (default: `outputs`)
- `--n-sims`: National forecast simulations (default: 2,000)
- `--n-psa`: PSA samples (default: 500)

**Outputs:**
- `stochastic_avoidance_sensitivity_results.json/csv/xlsx`
- Figures: `FigS1_MethTrajectories.png`, `FigS2_OutbreakForecast.png`, etc.

### Step 4: Generate Publication Figures

Create all manuscript figures from simulation results:

```bash
python visualize_md_results.py --input architectural_barrier_results.json --output-dir ./figures
```

**Outputs (in `figures/`):**
- `Fig1_CascadeComparison.png` - MSM vs PWID cascade
- `Fig2_BarrierDecomposition.png` - Three-layer barrier breakdown
- `Fig3_PolicyScenarios.png` - Policy scenario comparison
- `Fig4_StochasticAvoidance.png` - Outbreak prediction
- `Fig5_SNR_LOOCV.png` - Signal-to-noise ratio analysis

### Quick Visualization

Generate a simple scenario comparison chart:

```bash
python visualize_results.py --input architectural_barrier_results.json --output ./figures/scenario_comparison.png
```

## Repository Structure

```
HIV-Prevention_Master-Algorithm/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── architectural_barrier_model.py      # Core Monte Carlo simulation
├── cascade_sensitivity_analysis.py     # PSA and sensitivity analyses
├── stochastic_avoidance_enhanced.py    # Outbreak prediction model
├── visualize_md_results.py             # Publication figure generator (md = manuscript data)
├── visualize_results.py                # Quick visualization script
├── pep_mucosal.py                      # PEP timing analysis
├── pwid_cascade_v1.py                  # Annotated cascade model (educational)
│
├── MD/SRC/                             # Modular source code
│   ├── structural_barrier_model.py     # Core model classes
│   ├── cascade_sensitivity_analysis.py
│   ├── stochastic_avoidance_enhanced.py
│   └── ...
│
├── AIDS and Behavior/                  # AIDS & Behavior manuscript materials
│   ├── SRC/                            # Source code for that analysis
│   └── data/                           # Outputs and figures
│
├── legacy/                             # Archived development versions
└── figures/                            # Generated figures
```

## Model Architecture

### Three-Layer Barrier Framework

| Layer | Description | Contribution |
|-------|-------------|--------------|
| **Pathogen Biology** | HIV establishes irreversible R(0) > 0 within hours | 0% |
| **Testing Barriers** | Acute infection detection gaps | 6.9% |
| **Architectural Barriers** | Policy, stigma, infrastructure | 93.1% |

### Eight-Step Prevention Cascade

1. Awareness of LAI-PrEP availability
2. Willingness to seek care despite system visibility
3. Healthcare access
4. Disclosure of injection drug use
5. Provider willingness to prescribe
6. Adequate HIV testing
7. First injection received
8. Sustained engagement

### Policy Scenarios

| Scenario | P(R(0)=0) | Description |
|----------|-----------|-------------|
| Current Policy | 0.003% | All barriers active |
| Decriminalization Only | 0.20% | Remove criminalization penalties |
| Decrim + Stigma Reduction | 0.45% | Add stigma reduction |
| SSP-Integrated Delivery | 5.0% | Syringe service integration |
| Full Harm Reduction | 9.5% | Comprehensive harm reduction |
| Full HR + PURPOSE-4 Data | 11.9% | Add PWID trial data |
| Full HR + ML Debiasing | 18.6% | Add algorithmic fairness |
| Theoretical Maximum | 19.7% | All barriers removed |
| **MSM (Comparison)** | **16.3%** | Same intervention, different population |

## Reproducibility

All analyses are designed for full reproducibility:

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Monte Carlo iterations | 100,000 per scenario |
| Bootstrap replicates | 10,000 |
| PSA samples | 1,000 (default) |

To reproduce exact results:
```bash
# Run complete analysis pipeline
python architectural_barrier_model.py --output-dir ./results
python cascade_sensitivity_analysis.py --output-dir ./outputs
python stochastic_avoidance_enhanced.py --output-dir ./outputs
python visualize_md_results.py --input ./results/architectural_barrier_results.json --output-dir ./figures
```

## Dependencies

### Required
- **numpy** >= 1.21.0 - Numerical computing
- **scipy** >= 1.7.0 - Statistical functions
- **matplotlib** >= 3.5.0 - Visualization
- **pandas** >= 1.3.0 - Data manipulation

### Optional
- **openpyxl** >= 3.0.0 - Excel export support

### Development
- **mypy** >= 1.0.0 - Static type checking

## Data Sources

All model parameters are derived from peer-reviewed literature:

| Parameter | Source |
|-----------|--------|
| PWID epidemiology | Degenhardt et al. (2017), *Lancet* |
| Criminalization effects | DeBeck et al. (2017), *Lancet HIV* |
| PrEP cascade | Mistler et al. (2021), *AIDS Behav* |
| Incarceration impact | Altice et al. (2016), *Lancet* |
| LAI-PrEP efficacy | PURPOSE-1/2 trials, Bekker et al. (2024), *NEJM* |
| Healthcare stigma | Biancarelli et al. (2019), *Soc Sci Med* |
| Methamphetamine trends | NHBS 2012-2018 |
| Network modeling | Des Jarlais et al. (2022) |

## Citation

If you use this code or data in your research, please cite:

```bibtex
@software{demidont2025hivprevention,
  author={Demidont, AC},
  title={HIV Prevention Master Algorithm: Structural Barrier Modeling for PWID},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.18116991},
  url={https://github.com/Nyx-Dynamics/HIV_Prevention_PWID}
}
```

## License

This project is licensed under the **MIT License**, which permits reuse, modification, and distribution with attribution. See the [LICENSE](LICENSE) file for full details.

**Summary of permitted uses:**
- Academic research and publication
- Commercial applications
- Modification and derivative works
- Redistribution

**Required:** Include copyright notice and license text in copies or substantial portions.

## Contact

**AC Demidont, DO**
Nyx Dynamics LLC
Email: acdemidont@nyxdynamics.org
ORCID: [0000-0002-9216-8569](https://orcid.org/0000-0002-9216-8569)

## Acknowledgments

- HIV prevention research community for published parameters
- PWID community advocates for barrier characterization
- AI tools (Anthropic Claude) used as assistive technology for code development. Author maintains full responsibility for all aspects of project.
