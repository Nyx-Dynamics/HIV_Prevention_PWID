# PROJECT GUIDE: AIDS and Behavior Publication Pipeline

This guide provides instructions for executing the computational pipeline and reproducing the figures and data tables for the *AIDS and Behavior* submission.

## 1. Environment Setup
The scripts require Python 3.9+ and the following dependencies:
```bash
pip install numpy matplotlib scipy pandas openpyxl
```

## 2. Execution Sequence
For full reproducibility, execute the scripts in the following order from within the `AIDS and Behavior/SRC/` directory.

### Step 1: Core Simulation
Run the `architectural_barrier_model.py` to generate the primary dataset.
```bash
python architectural_barrier_model.py --n-individuals 100000
```
- **Outputs**: `architectural_barrier_results.json`, `.csv`, and `.xlsx` in `../data/csv_xlsx/`.

### Step 2: Sensitivity Analysis
Run the `cascade_sensitivity_analysis.py` to evaluate parameter uncertainty.
```bash
python cascade_sensitivity_analysis.py --n-samples 1000
```
- **Outputs**: `cascade_sensitivity_results.json`, `.csv`, and `.xlsx` in `../data/csv_xlsx/`.
- **Figures**: Supplemental Figures (S5-S8) in `../data/figures/`.

### Step 3: Outbreak Forecasting
Run `stochastic_avoidance_enhanced.py` for regional and temporal projections.
```bash
python stochastic_avoidance_enhanced.py --n-sims 2000 --n-psa 500
```
- **Outputs**: `stochastic_avoidance_sensitivity_results.json`, `.csv`, and `.xlsx` in `../data/csv_xlsx/`.
- **Figures**: Supplemental Figures (S1-S4) in `../data/figures/`.

### Step 4: Supplementary Results and Figures
Generate all supplementary figures (S1-S5) and export data tables.
```bash
python reproduce_supplementary_results.py
```
- **Outputs**: Supplementary Figures (S1-S5) in `../Supplementary Figures/` and data in `../data/csv_xlsx/`.

### Step 5: Graphical Abstract
Generate the graphical abstract with corrected values.
```bash
python figures_AIDS_Behavior.py
```
- **Outputs**: Graphical abstract in publication-ready format.

## 3. Data Interpretation
- **`../data/csv_xlsx/`**: Contains the raw numerical data. The Excel workbooks are organized with multiple sheets (e.g., "Scenario Results", "Barrier Decomposition") for direct use in manuscript table preparation.
- **`../data/figures/`**: Contains main manuscript figures (Figs 1-5). EPS files are preferred for vector graphics (Line Art), while TIFF files are provided for high-resolution halftones/combination art.
- **`../data/aids_behavior_figures_canonical/`**: Contains canonical versions of main figures in EPS, PNG, and TIFF formats.
- **`../Supplementary Figures/`**: Contains high-resolution supplementary figures (S1-S5) in EPS and TIFF formats.

## 4. Key Output Files

| File | Description |
|------|-------------|
| `architectural_barrier_results.*` | Cascade completion rates by policy scenario |
| `structural_barrier_results.*` | Three-layer barrier decomposition analysis |
| `stochastic_avoidance_sensitivity_results.*` | Outbreak probability distributions |
| `national_forecast_summary.csv` | National-level outbreak forecasts |
| `regional_comparison.csv` | Regional heterogeneity analysis |
| `tornado_analysis.csv` | Parameter sensitivity rankings |
| `supplementary_data_master.xlsx` | Combined workbook for all supplementary data |

## 5. Submission Directories

### AIDS_Behvior_submission/
Contains all files formatted for AIDS and Behavior journal submission:
- `AIDS_Behavior_Manuscript_BLINDED.docx/pdf` - Blinded manuscript
- `AIDS_Behavior_OnlineResource_S1.docx` - Mathematical foundations supplement
- `AIDS_Behavior_OnlineResource_S2.docx` - Supplementary figures
- `Fig1.tif` through `Fig5.tif` - Main figures
- `FigS1.tif` through `FigS4.tif` - Supplementary figures

### Structural_Barriers_PWIDS_preprints_submission/
Contains files formatted for preprint submission.

## 6. Troubleshooting
- **Missing Data**: Ensure all scripts are run from the `SRC/` directory so that relative paths (`../data/...`) resolve correctly.
- **Font Warnings**: If Arial is not found, the scripts will fallback to Helvetica or a generic sans-serif. For final production, ensure Arial is installed on the system.
- **Import Errors**: Ensure `stochastic_avoidance_enhanced.py` is in the same directory when running `reproduce_supplementary_results.py`.
