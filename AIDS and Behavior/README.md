
## Repository Overview
This repository contains the code, simulation outputs, and supplementary analyses supporting the manuscript *Structural Violation of HIV Prevention Timing Constraints Among People Who Inject Drugs*. The study examines why people who inject drugs (PWID) continue to experience high HIV incidence despite the availability of highly efficacious biomedical prevention tools. Building on a feasibility-based definition of prevention, the analyses demonstrate how strict biological timing constraints on post-exposure prophylaxis interact with structural barriers to care, network dynamics, and incarceration to place reactive prevention strategies outside the feasible biological domain. The materials provided here support the computational modeling, sensitivity analyses, and scenario evaluations described in the manuscript and are intended to enable transparency and reproducibility of the reported findings.

## Author
**AC Demidont, DO**
Independent Researcher
Nyx Dynamics LLC
GitHub: [Nyx-Dynamics](https://github.com/Nyx-Dynamics)

## Repository Contents

```
├── SRC/                                           # Source code
│   ├── architectural_barrier_model.py             # Cascade simulation framework (Monte Carlo)
│   ├── cascade_sensitivity_analysis.py            # Probabilistic sensitivity analysis
│   ├── stochastic_avoidance_enhanced.py           # Outbreak probability modeling (5-10 year horizons)
│   ├── structural_barrier_model.py                # Three-layer barrier framework simulation
│   ├── figures_AIDS_Behavior.py                   # Graphical abstract generator
│   ├── reproduce_supplementary_results.py         # Reproduces supplementary figures S1-S5 and data
│   ├── generate_outputs.py                        # Output generation utilities
│   ├── prevention_theorem_figures.py              # Prevention theorem visualization
│   ├── pwid_cascade_v1.py                         # PWID cascade model (v1)
│   └── PEP_mucosal.py                             # PEP mucosal exposure modeling
├── data/
│   ├── csv_xlsx/                                  # Model outputs (JSON, CSV, XLSX)
│   ├── figures/                                   # Main manuscript figures (TIFF, PNG, EPS)
│   └── aids_behavior_figures_canonical/           # Canonical figure versions
├── config/
│   └── parameters.json                            # Model configuration and literature values
├── AIDS_Behvior_submission/                       # Journal submission files
│   ├── AIDS_Behavior_Manuscript_BLINDED.docx/pdf  # Blinded manuscript
│   ├── AIDS_Behavior_OnlineResource_S1.docx       # Online Resource S1
│   ├── AIDS_Behavior_OnlineResource_S2.docx       # Online Resource S2
│   ├── Fig1-5.tif                                 # Main figures (TIFF format)
│   └── FigS1-S4.tif                               # Supplementary figures
├── Structural_Barriers_PWIDS_preprints_submission/ # Preprint submission files
├── Supplementary Figures/                         # High-resolution supplementary figures (EPS, TIFF)
├── LICENSE                                        # MIT License
└── requirements.txt                               # Python dependencies
```

## Extended Stochastic Avoidance and Outbreak Modeling

This repository includes additional data and figures related to stochastic avoidance failure and outbreak probability that are not shown in the main manuscript. These materials provide extended context on network-driven instability, regional heterogeneity, and sensitivity of outbreak risk to structural parameters. They are included to support transparency and robustness assessment rather than to expand the scope of the main analysis.

### Key Scripts

| Script | Description |
|--------|-------------|
| `architectural_barrier_model.py` | Monte Carlo cascade simulation (100,000 individuals) |
| `structural_barrier_model.py` | Three-layer barrier framework with policy scenarios |
| `cascade_sensitivity_analysis.py` | 1,000-sample probabilistic sensitivity analysis |
| `stochastic_avoidance_enhanced.py` | Outbreak probability modeling (5-10 year horizons) |
| `reproduce_supplementary_results.py` | Reproduces all supplementary figures and data tables |
| `figures_AIDS_Behavior.py` | Generates graphical abstract with corrected values |

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

# 7. Generate supplementary results and figures
python reproduce_supplementary_results.py

# 8. Generate graphical abstract
python figures_AIDS_Behavior.py
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
| `national_forecast_summary.csv` | National outbreak forecast summary |
| `regional_comparison.csv` | Regional heterogeneity analysis |
| `scenario_comparison.csv` | Policy scenario comparisons |
| `tornado_analysis.csv` | Tornado diagram sensitivity data |
| `supplementary_data_master.xlsx` | Combined supplementary data workbook |

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

### Main Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `Fig1_CascadeComparison.*` | LAI-PrEP cascade comparison (MSM vs PWID) |
| Fig 2 | `Fig2_BarrierDecomposition.*` | Three-layer barrier decomposition |
| Fig 3 | `Fig3_PolicyScenarios.*` | Policy scenario analysis |
| Fig 4 | `Fig4_StochasticAvoidance.*` | Stochastic avoidance failure prediction |
| Fig 5 | `Fig5_SNR_LOOCV.*` | Signal-to-noise ratio / LOOCV framework |

### Supplementary Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig S1 | `FigS1_ContextualStochasticFailureDriver.*` | Contextual stochastic failure trajectories |
| Fig S2 | `FigS2_OutbreakForecast.*` | Outbreak probability forecast |
| Fig S3 | `FigS3_TornadoDiagram.*` | Sensitivity tornado diagram |
| Fig S4 | `FigS4_ScenarioComparison.*` | Policy scenario comparison |
| Fig S5 | `FigS5_ScenarioComparisonDetail.png` | Detailed scenario comparison |

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
