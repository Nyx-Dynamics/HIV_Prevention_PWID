
## Repository Overview
This repository contains the code, simulation outputs, and supplementary analyses supporting the manuscript *Structural Violation of HIV Prevention Timing Constraints Among People Who Inject Drugs*. The study examines why people who inject drugs (PWID) continue to experience high HIV incidence despite the availability of highly efficacious biomedical prevention tools. Building on a feasibility-based definition of prevention, the analyses demonstrate how strict biological timing constraints on post-exposure prophylaxis interact with structural barriers to care, network dynamics, and incarceration to place reactive prevention strategies outside the feasible biological domain. The materials provided here support the computational modeling, sensitivity analyses, and scenario evaluations described in the manuscript and are intended to enable transparency and reproducibility of the reported findings.

## Author
**AC Demidont, DO**
Independent Researcher
Nyx Dynamics LLC
GitHub: [Nyx-Dynamics](https://github.com/Nyx-Dynamics)

## Repository Contents

```
├── SRC/                                    # Source code
│   ├── architectural_barrier_model.py      # Cascade simulation framework (Monte Carlo)
│   ├── cascade_sensitivity_analysis.py     # Probabilistic sensitivity analysis
│   ├── stochastic_avoidance_enhanced.py    # Models outbreak probability under stochastic avoidance over 5-10 year horizons 
│   └── generate_figures.py                 # Publication figure generator
├── data/
│   ├── csv_xlsx/                           # Model outputs (JSON, CSV, XLSX)
│   └── figures/                            # Generated figures (TIFF, PNG, EPS)
├── config/
│   └── parameters.json                     # Model configuration and literature values
├── LICENSE                                 # MIT License
└── requirements.txt                        # Python dependencies
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `architectural_barrier_model.py` | Simulates 100,000 individuals through 8-step prevention cascade |
| `cascade_sensitivity_analysis.py` | Runs 1,000-sample probabilistic sensitivity analysis |
| `stochastic_avoidance_enhanced.py` | Models outbreak probability over 5-10 year horizons |
| `generate_figures.py` | Generates all manuscript figures (Figs 1-5) |

## Reproducibility:
The commands below reproduce the analyses reported in the manuscript and Supplementary Materials.

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Step-by-Step Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Nyx-Dynamics/HIV_Prevention_PWID.git
cd HIV_Prevention_PWID

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run main simulation (generates cascade results)
cd SRC
python architectural_barrier_model.py --output-dir ../data/csv_xlsx --n-individuals 100000

# 5. Run sensitivity analysis
python cascade_sensitivity_analysis.py --output-dir ../data/csv_xlsx --n-samples 1000

# 6. Run stochastic outbreak model
python stochastic_avoidance_enhanced.py --output-dir ../data/csv_xlsx

# 7. Generate publication figures
python generate_figures.py --input-dir ../data/csv_xlsx --output-dir ../data/figures
```

### Expected Runtime

- Main simulation: ~2-5 minutes (100,000 individuals)
- Sensitivity analysis: ~5-10 minutes (1,000 samples)
- Figure generation: ~1 minute

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20.0 | Numerical computation |
| scipy | >=1.7.0 | Statistical functions |
| matplotlib | >=3.5.0 | Figure generation |
| pandas | >=1.3.0 | Data manipulation |
| openpyxl | >=3.0.0 | Excel export |

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Data Provenance
No individual-level or identifiable data are used in this repository.

### Model Parameters

All epidemiological parameters in `config/parameters.json` are derived from peer-reviewed literature. Key sources include:

- **PWID population estimates**: UNODC World Drug Report 2023
- **HIV prevalence**: Degenhardt et al. (2017), Lancet HIV
- **Cascade probabilities**: Derived from empirical studies cited in manuscript
- **Outbreak parameters**: Based on Scott County (2015), Massachusetts (2018-2019), and Kanawha County (2019) outbreak data

### Output Files

| File | Description |
|------|-------------|
| `architectural_barrier_results.*` | Cascade completion rates by scenario |
| `cascade_sensitivity_results.*` | Parameter sensitivity rankings |
| `stochastic_avoidance_sensitivity_results.*` | Outbreak probability distributions |
| `structural_barrier_results.*` | Three-layer barrier decomposition |

### Key Results
Summary statistics below reflect simulated outcomes under specified structural scenarios and are provided for transparency; they are not intended as population forecasts.
| Metric | Value | 95% CI |
|--------|-------|--------|
| PWID cascade completion (current policy) | 0.003% | (0.000%, 0.006%) |
| MSM cascade completion (comparison) | 16.30% | — |
| Disparity ratio | 5,434-fold | — |
| P(outbreak within 5 years) | 63.3% | (60.1%, 66.5%) |

## Manuscript

This code accompanies the manuscript:

> Demidont AC. *Structural Violation of HIV Prevention Timing Constraints Among People Who Inject Drugs*. AIDS and Behavior (submitted 1/2026).

### Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `Fig1_CascadeComparison.*` | LAI-PrEP cascade comparison (MSM vs PWID) |
| Fig 2 | `Fig2_BarrierDecomposition.*` | Three-layer barrier decomposition |
| Fig 3 | `Fig3_PolicyScenarios.*` | Policy scenario analysis |
| Fig 4 | `Fig4_StochasticAvoidance.*` | Stochastic avoidance failure prediction |
| Fig 5 | `Fig5_SNR_LOOCV.*` | Signal-to-noise ratio / LOOCV framework |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

You are free to:
- Use, copy, modify, and distribute this code
- This project is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the terms of the license.
- Include in derivative works

With the requirement to include the original copyright notice.

## Citation

If you use this code or data, please cite:

```bibtex
@article{demidont2026pwid,
  author = {Demidont, AC},
  title = {Structural Violation of HIV Prevention Timing Constraints Among People Who Inject Drugs},
  journal = {AIDS and Behavior},
  year = {2026},
  publisher = {Springer}
}
```

For the code repository:

```bibtex
@software{demidont2026pwid_code,
  author = {Demidont, AC},
  title = {Structural Violation of HIV Prevention Timing Constraints Among People Who Inject Drugs Source Code},
  year = {2026},
  url = {https://github.com/Nyx-Dynamics/HIV_Prevention_PWID}
}
```
