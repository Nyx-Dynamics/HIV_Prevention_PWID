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

### Step 4: Figure Generation
Generate the primary manuscript figures using the data from Step 1.
```bash
python visualize_md_results.py
python aids_behavior_figs.py
```
- **Outputs**: Main Figures (1-5) in `../data/figures/` (EPS and TIFF formats).

## 3. Data Interpretation
- **`../data/csv_xlsx/`**: Contains the raw numerical data. The Excel workbooks are organized with multiple sheets (e.g., "Scenario Results", "Barrier Decomposition") for direct use in manuscript table preparation.
- **`../data/figures/`**: Contains final artwork. EPS files are preferred for vector graphics (Line Art), while TIFF files are provided for high-resolution halftones/combination art.

## 4. Troubleshooting
- **Missing Data**: Ensure all scripts are run from the `SRC/` directory so that relative paths (`../data/...`) resolve correctly.
- **Font Warnings**: If Arial is not found, the scripts will fallback to Helvetica or a generic sans-serif. For final production, ensure Arial is installed on the system.
