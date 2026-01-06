# AIDS and Behavior: Publication Materials

This directory contains the computational models, simulation scripts, and publication-quality artwork for the manuscript submitted to *AIDS and Behavior*.

## Overview
All materials in this directory have been specifically formatted to comply with the **AIDS and Behavior Artwork and Illustrations Guidelines**. This includes standardized typography (Arial/Helvetica), high-resolution outputs (EPS and TIFF), and accessibility features (hatching patterns for color-blind friendliness).

## Directory Structure
- **`SRC/`**: Primary Python scripts for simulation and visualization.
  - `architectural_barrier_model.py`: Core Monte Carlo simulation engine.
  - `stochastic_avoidance_enhanced.py`: Outbreak forecasting and regional trajectory modeling.
  - `cascade_sensitivity_analysis.py`: Probabilistic sensitivity analysis and barrier importance ranking.
  - `aids_behavior_figs.py`: Dedicated figure generator for journal-compliant artwork.
- **`data/figures/`**: Publication-ready illustrations.
  - Formats: EPS (vector), TIFF (600 DPI, LZW compressed), PDF, and PNG.
  - Named following the `Fig[N]` convention (e.g., `Fig1_CascadeComparison.eps`).
- **`data/csv_xlsx/`**: Raw and processed simulation data.
  - Formats: CSV, JSON, and multi-sheet Excel (.xlsx) workbooks for easy review.

## Guideline Compliance
- **Typography**: Arial/Helvetica sans-serif fonts (8-12pt).
- **Resolution**: 600 DPI for combination art, 300 DPI minimum for halftones.
- **Accessibility**: High-contrast palettes and distinct hatching patterns for black-and-white legibility.
- **Technical**: Removed all transparency (`alpha` settings) for full PostScript compatibility.

## Usage
Refer to the `PROJECT_GUIDE.md` within this directory for detailed instructions on running the scripts and reproducing the results.

## Contact
**AC Demidont, DO**
Nyx Dynamics LLC
Email: acdemidont@nyxdynamics.org
