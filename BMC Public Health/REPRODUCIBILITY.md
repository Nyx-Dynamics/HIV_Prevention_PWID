# Reproducibility Suite for "BMC Public Health"

This document describes how to reproduce the results and figures presented in the manuscript submitted to *BMC Public Health*.

## Prerequisites

- Python 3.8 or higher
- `make` utility

## Installation

1. Clone the repository and navigate to the `BMC Public Health` directory:
   ```bash
   cd "BMC Public Health"
   ```

2. Install the required Python dependencies:
   ```bash
   make install-deps
   ```

## Reproducing Results

The reproducibility suite is managed via a `Makefile`. You can run individual components or the entire suite.

### Full Reproduction
To run all simulations and generate all figures (main and supplementary):
```bash
make reproduce
```

### Main Figures Only
To generate only the main publication figures (Figures 1, 2, and 3):
```bash
make figures
```

### Supplementary Results Only
To generate supplementary figures (S1–S4) and the associated data files:
```bash
make supplementary
```

## Output Locations

- **Main Figures:** `data/figures/` (available in EPS and TIFF formats)
- **Supplementary Figures:** `Supplementary Figures/`
- **Supplemental Data:** `Supplemental Data/data/` (CSV and XLSX formats)

## Cleanup

To remove generated outputs and start from a clean state:
```bash
make clean
```
