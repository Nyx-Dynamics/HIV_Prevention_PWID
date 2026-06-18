# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monte Carlo simulation study modeling structural barriers to HIV prevention for people who inject drugs (PWID). The model implements a three-layer barrier framework across an 8-step prevention cascade, comparing PWID vs MSM populations. Under peer review at BMC Public Health; preprint DOI: 10.20944/preprints202601.0948.v1.

## Commands

### Setup
```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, pandas, openpyxl
```

### Run everything
```bash
python reproduce_all.py           # Full pipeline: simulation → sensitivity → figures → validation
```

### Run individual modules
```bash
# All run from src/ unless noted
python src/structural_barrier_model.py
python src/stochastic_avoidance_enhanced.py --output-dir data/figures --data-dir data/csv_xlsx
python src/cascade_sensitivity_analysis.py --output-dir data/csv_xlsx
python src/generate_outputs.py
python src/generate_cascade_schematic.py
python scripts/reproduce_supplementary_results.py
```

### Manuscript prep
```bash
python manuscript_prep/build_additional_files.py    # Generates BMC Word .docx with embedded TIFF figures
python manuscript_prep/revise_manuscript_v2.py      # Enhanced revision workflow
```

## Architecture

### Core Data Flow
```
config/parameters.json
    → structural_barrier_model.py    (core simulation engine)
    → stochastic_avoidance_enhanced.py / _v2.py  (outbreak risk models)
    → cascade_sensitivity_analysis.py  (PSA, barrier removal scenarios)
    → generate_outputs.py + generate_cascade_schematic.py  (publication figures)
    → data/csv_xlsx/  (JSON/CSV/XLSX results)
    → data/figures/   (PNG/EPS/TIFF, BMC-compliant)
```

`reproduce_all.py` orchestrates all steps via the `ReproducibilityRunner` class and runs built-in validation checks (PWID cascade <1%, MSM ~21%, disparity >20-fold, R₀=0 feasibility).

### Three-Layer Barrier Model
- **Layer 1 — Pathogen Biology:** HIV integration timeline (120 hrs), transmission probabilities (0.63% injection, 0.8% sexual)
- **Layer 2 — Testing:** Acute infection window periods, detection delays
- **Layer 3 — Architectural:** Policy/criminalization (52.5% cascade attrition), stigma/implementation (25.4%), infrastructure gaps (15%), research exclusion (10%), ML/algorithmic bias (22.1%)

Each cascade step in `config/parameters.json` specifies a `base_probability` plus penalty values for each architectural barrier category.

### Key Modules
- **`src/structural_barrier_model.py`** — Canonical core. Contains `StructuralBarrierModel`, `PolicyScenario`, `StochasticAvoidanceParams`, `CascadeStep`. Monte Carlo: 100,000 individuals × 5 years, fixed seed 42.
- **`src/architectural_barrier_model.py`** — Backward-compatibility shim; re-exports from `structural_barrier_model`.
- **`src/stochastic_avoidance_enhanced.py`** — Outbreak probability forecasting (`EnhancedStochasticAvoidanceModel`, `SensitivityAnalyzer`); models regional variation (Appalachia, Pacific Northwest, Northeast urban).
- **`src/stochastic_avoidance_v2.py`** — Extends v1 with multiplicative methamphetamine × housing interaction.
- **`src/cascade_sensitivity_analysis.py`** — Probabilistic sensitivity analysis (PSA), barrier removal scenarios, step importance ranking.
- **`src/hood_parameter_comparison.py`** — Adjusts parameters against Hood et al. (2018) published data.
- **`manuscript_prep/`** — Word document generation for BMC submission; not part of the simulation pipeline.

### Reproducibility
- Random seed is fixed at **42** across all modules.
- All outputs (figures, JSON, CSV) are deterministic given the same `config/parameters.json`.
- `data/reproducibility_log.txt` is written by `reproduce_all.py` and logs validation results.

### Output Formats
Figures are generated in PNG + EPS + TIFF (300 DPI) to meet BMC Public Health submission requirements (sans-serif fonts, specific size constraints). Figure generation code in `generate_outputs.py` and `generate_cascade_schematic.py` should preserve these settings when modified.
