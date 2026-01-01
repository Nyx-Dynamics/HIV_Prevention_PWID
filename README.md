# HIV Prevention Barrier Modeling for PWID

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Code repository for:** Demidont AC. Structural barriers drive near-zero population-level effectiveness of Long Acting Injectable HIV prevention (LAI-PrEP) among people who inject drugs: A Computational Modeling Study. *The Lancet HIV* (2025). [Manuscript ID: thelancethiv-D-25-00576, Submitted Dec 29, 2025]

## Overview

This repository contains the computational models, simulation code, and visualization scripts supporting the analysis of structural barriers to HIV prevention among people who inject drugs (PWID). The model demonstrates how policy, stigma, and infrastructure barriers compound across an 8-step prevention cascade to produce near-zero population-level effectiveness despite highly efficacious pharmacological interventions.

## Key Findings

- **Current policy achieves ~0.003% sustained HIV prevention** among PWID vs. 16.3% for MSM receiving identical interventions
- **Barrier multiplication effect**: Even moderate barriers at each cascade step compound to near-total prevention failure
- **Policy-determined outcomes**: Results are structural, not individual choice-based
- **63% outbreak probability within 5 years** under current conditions

## Repository Structure

```
├── architectural_barrier_model.py    # Main simulation model
├── visualize_md_results.py           # Publication figure generation
├── architectural_barrier_results.json # Monte Carlo simulation outputs
├── pwid_simulation_results.json       # PWID-specific analyses
├── manuscript/                        # LaTeX manuscript files
│   └── lancet_hiv_manuscript_acd_2025.tex
└── figures/                           # Publication-ready figures
    ├── Fig1_CascadeComparison.png
    ├── Fig3_PolicyScenarios.png
    └── Fig5_SNR_LOOCV.png
```

## Model Architecture

### Three-Layer Barrier Framework

**Layer 1: Pathogen Biology**
- HIV establishes irreversible R₀ > 0 within hours of infection
- No intervention can restore R₀ = 0 post-exposure

**Layer 2: Testing Barriers**
- Acute infection detection gaps (RNA testing not standard of care)
- 4th gen Ag/Ab tests miss first 2-4 weeks

**Layer 3: Architectural Barriers**
- Policy (criminalization, incarceration)
- Stigma (healthcare discrimination, disclosure barriers)
- Infrastructure (MSM-centric cascade design)
- Research exclusion (LOOCV framework)
- Machine learning (algorithmic deprioritization)

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

| Scenario | P(R₀=0) |
|----------|---------|
| Current Policy | 0.003% |
| Decriminalization Only | 0.20% |
| Decrim + Stigma Reduction | 0.45% |
| SSP-Integrated Delivery | 5.0% |
| Full Harm Reduction | 9.5% |
| Full HR + PURPOSE-4 Data | 11.9% |
| Full HR + Algorithmic Debiasing | 18.6% |
| Theoretical Maximum | 19.7% |
| MSM (Comparison) | 16.3% |

## Installation

### Requirements
- Python 3.9+
- NumPy
- Matplotlib
- SciPy (optional, for advanced statistics)

### Setup
```bash
git clone https://github.com/Nyx-Dynamics/hiv-prevention-master.git
cd hiv-prevention-master
pip install -r requirements.txt
```

## Usage

### Run Main Simulation
```bash
python architectural_barrier_model.py
```

### Generate Figures
```bash
python visualize_md_results.py
```

### Outputs
- `architectural_barrier_results.json` - Full Monte Carlo results
- `pwid_simulation_results.csv` - Summary statistics
- `figures/` - Publication-ready PNG files

## Data Sources

All model parameters are derived from peer-reviewed literature:

| Parameter | Source |
|-----------|--------|
| PWID epidemiology | Degenhardt et al. (2017), *Lancet* |
| Criminalization effects | DeBeck et al. (2017), *Lancet HIV* |
| PrEP cascade | Mistler et al. (2021), *AIDS Behav* |
| Incarceration impact | Altice et al. (2016), *Lancet* |
| LAI-PrEP efficacy | PURPOSE-1/2 trials, Bekker et al. (2024), *NEJM* |
| Healthcare stigma | Biancarelli et al. (2019), *Social Science & Medicine* |

## Reproducibility

All analyses are fully reproducible:
- Random seed: 42
- Monte Carlo iterations: 100,000
- Bootstrap replicates: 10,000

## Citation

If you use this code, please cite:

```bibtex
@article{demidont2025structural,
  title={Structural barriers drive near-zero population-level effectiveness 
         of Long Acting Injectable HIV prevention (LAI-PrEP) among people 
         who inject drugs: A Computational Modeling Study},
  author={Demidont, AC},
  journal={The Lancet HIV},
  year={2025},
  note={Manuscript ID: thelancethiv-D-25-00576, Submitted}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**AC Demidont, DO**  
Nyx Dynamics LLC  
Email: acdemidont@nyxdynamics.org  
ORCID: [0000-0002-9216-8569](https://orcid.org/0000-0002-9216-8569)

## Acknowledgments

- HIV prevention research community for published parameters
- PWID community advocates for barrier characterization
- AI tools (Anthropic Claude 4.5), Jetbrains Junie, Zotero AI, Numpy, Matplotlib, Scipy, OpenAI GPT 5.2 used as assistive technology for literature synthesis, reference management and code refinement.  Author maintainces responsibility for all aspects of project and responsbility for use and software use.
