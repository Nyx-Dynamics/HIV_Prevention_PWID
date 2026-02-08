# CHANGELOG: BMC Public Health Publication Materials
## Version 1.0.0 (January 6, 2026)
### Initial Release for BMC Public Health
- **Consolidated Publication Materials:** Created a dedicated `BMC Public Health` directory to house all scripts, data, and figures specifically formatted for the journal submission.
- **Guideline Compliance:**
  - Standardized all figure generation to use **Arial/Helvetica** sans-serif fonts.
  - Set font sizes consistently between 8pt and 12pt (e.g., 10pt for labels, 9pt for ticks).
  - Switched output formats to **EPS** (vector) and **TIFF** (300/600 DPI with LZW compression).
  - Removed internal titles and captions from figures as per journal requirements.
  - Implemented **hatching patterns** for accessibility and black-and-white clarity.
- **Script Enhancements:**
  - Migrated primary modeling scripts: `architectural_barrier_model.py`, `stochastic_avoidance_enhanced.py`, and `cascade_sensitivity_analysis.py`.
  - Added full **Excel (.xlsx)** export support to all modeling scripts using `pandas` and `openpyxl`.
  - Removed all transparency (`alpha` settings) from plot elements to ensure PostScript (EPS) backend compatibility.
- **Directory Structure:**
  - `SRC/`: Updated source scripts and figure generators.
  - `data/figures/`: Publication-quality artwork in EPS, TIFF, PDF, and PNG formats.
  - `data/csv_xlsx/`: Raw simulation results in CSV, JSON, and multi-sheet Excel workbooks.
