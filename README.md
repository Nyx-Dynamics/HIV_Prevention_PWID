# AIDS and Behavior - PWID HIV Prevention Cascade Analysis

## Repository Structure

```
AIDS and Behavior/
├── SRC/
│   ├── architectural_barrier_model.py    # Main cascade simulation model
│   ├── cascade_sensitivity_analysis.py   # PSA and sensitivity analyses
│   ├── stochastic_avoidance_enhanced.py  # Outbreak prediction model
│   └── generate_figures.py               # Publication figure generator
├── data/
│   ├── csv_xlsx/                         # Model outputs (JSON, CSV, XLSX)
│   └── figures/                          # Generated figures (TIFF, PNG, EPS)
└── config/
    └── parameters.json                   # Model configuration
```

## Usage

### 1. Run Main Simulation

```bash
cd SRC
python architectural_barrier_model.py --output-dir ../data/csv_xlsx --n-individuals 100000
```

### 2. Run Sensitivity Analysis

```bash
python cascade_sensitivity_analysis.py --output-dir ../data/csv_xlsx --n-samples 1000
```

### 3. Generate Figures

```bash
python generate_figures.py --input-dir ../data/csv_xlsx --output-dir ../data/figures
```

## Key Findings

### Cascade Completion by Policy Scenario

| Scenario | P(R₀=0) | 95% CI |
|----------|---------|--------|
| Current Policy | 0.003% | (0.000%, 0.006%) |
| Decriminalization Only | 0.20% | (0.17%, 0.23%) |
| Decrim + Stigma Reduction | 0.45% | (0.41%, 0.50%) |
| SSP-Integrated Delivery | 5.00% | (4.87%, 5.14%) |
| Full Harm Reduction | 9.55% | (9.37%, 9.73%) |
| Full HR + PURPOSE-4 Data | 11.87% | (11.67%, 12.07%) |
| Full HR + ML Debiasing | 18.57% | (18.33%, 18.81%) |
| Theoretical Maximum | 19.74% | (19.49%, 19.98%) |
| **MSM (Comparison)** | **16.30%** | — |

**Disparity: 5,434-fold** (MSM vs PWID under identical pharmacology)

### Three-Layer Barrier Decomposition

| Layer | Contribution |
|-------|-------------|
| Pathogen Biology | 0.0% |
| HIV Testing | 6.9% |
| **Architectural** | **93.1%** |

### Architectural Barrier Subtypes

| Subtype | Contribution |
|---------|-------------|
| Policy (Criminalization) | 38.4% |
| Infrastructure (MSM-centric) | 21.9% |
| Stigma (Healthcare) | 20.6% |
| Machine Learning (Algorithmic) | 8.2% |
| Research Exclusion | 4.1% |

### Stochastic Avoidance Failure Prediction

| Metric | Value |
|--------|-------|
| P(outbreak within 5 years) | 63.3% |
| P(outbreak within 10 years) | 87.5% |
| Median years to outbreak | 4.0 |

## Figures

### Main Manuscript (5 figures)

1. **Fig 1**: LAI-PrEP Cascade Comparison (MSM vs PWID)
2. **Fig 2**: Three-Layer Barrier Decomposition
3. **Fig 3**: Policy Scenario Analysis
4. **Fig 4**: Stochastic Avoidance Failure Prediction
5. **Fig 5**: Signal-to-Noise Ratio / LOOCV Framework

### Online Resource S2 (Supplementary Figures)

- **Fig S1**: Methamphetamine Prevalence Trajectories
- **Fig S2**: Tornado Diagram (Parameter Sensitivity)
- **Fig S3**: Policy Scenario Extended Comparison
- **Fig S4**: Cascade Uncertainty (PSA Distribution)
- **Fig S5**: Barrier Removal Waterfall
- **Fig S6**: Step Importance Analysis

## Requirements

```
numpy>=1.20.0
matplotlib>=3.5.0
openpyxl>=3.0.0  # Optional, for Excel export
```

## Citation

Demidont AC. Structural Barriers, Stochastic Avoidance Failure, and Outbreak Risk 
in HIV Prevention for People Who Inject Drugs. AIDS and Behavior. 2026.

## Author

AC Demidont, DO  
Nyx Dynamics LLC  
https://github.com/Nyx-Dynamics/HIV_Prevention_PWID

## License

MIT License
