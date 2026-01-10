# Prevention Theorem: Time-Dependent Constraints on Post-Exposure Prophylaxis for HIV

## Overview
This directory contains the code, data, and submission materials for the Prevention Theorem manuscript, which establishes the mathematical and biological constraints on HIV post-exposure prophylaxis (PEP) efficacy.

## Author
**AC Demidont, DO**
Independent Researcher
Nyx Dynamics LLC
GitHub: [Nyx-Dynamics](https://github.com/Nyx-Dynamics)

## Publication Status
- **Preprints.org**: Accepted (ID: 193808) - January 10, 2026
- **Epidemics Journal**: Under Review (EPIDEMICS-S-26-00002)
- **medRxiv**: Published (MEDRXIV-2026-343483v1)

## Repository Structure

```
Prevention_Theorem/
├── SRC/                                    # Source code
│   ├── PEP_mucosal.py                      # Mucosal exposure and PEP timing model
│   ├── prevention_theorem_figures.py       # Figure generation (v1)
│   ├── prevention_theorem_figures_v2.py    # Figure generation (v2)
│   └── graphical_abstracts.py              # Graphical abstract generator
├── data/
│   └── figures/                            # Generated figures
│       ├── Figure_1_Prevention_Theorem_Dynamics.*
│       ├── Figure_2_Window_Compression.*
│       └── fig1_mucosal_timeline.pdf
├── submissions/
│   ├── Epidemics/                          # Epidemics journal submission
│   │   ├── epidemics_manuscript.pdf
│   │   ├── epidemics_manuscript_LaTeX_source_file.tex
│   │   ├── EPIDEMICS-S-26-00002.pdf        # Submission confirmation
│   │   ├── figures/
│   │   ├── graphical_abstract.*
│   │   └── Epidemics_cover_letter.docx
│   └── Preprints/                          # Preprints.org submission
│       ├── preprints_prevention_theorem.docx
│       ├── preprints_prevention_theorem_figures.zip
│       └── MEDRXIV-2026-343483v1-Demidont-3.pdf
├── references/                             # Bibliography files
│   ├── prevention_theorem_clean.bib
│   ├── prevention_theorem_clean-2.bib
│   └── Prevention_theorem_references.bib
└── README.md
```

## Key Concepts

### The Prevention Theorem
The Prevention Theorem establishes that HIV prevention efficacy is fundamentally constrained by:

1. **Biological Timing**: HIV establishes irreversible infection within 24-72 hours of mucosal exposure
2. **PEP Window**: Post-exposure prophylaxis must be initiated within 72 hours for efficacy
3. **Access Delays**: Structural barriers create systematic delays that exceed biological windows

### Main Findings
- PEP efficacy declines exponentially after exposure
- Structural barriers (stigma, criminalization, healthcare access) create delays incompatible with biological requirements
- The "prevention window" is systematically violated for marginalized populations

## Figures

| Figure | Description |
|--------|-------------|
| Figure 1 | Prevention Theorem Dynamics - Time-dependent efficacy curves |
| Figure 2 | Window Compression - Structural barrier impact on prevention windows |

## Running the Code

```bash
# Generate figures
cd SRC
python prevention_theorem_figures.py

# Generate graphical abstract
python graphical_abstracts.py
```

## Dependencies
- Python 3.8+
- numpy
- matplotlib
- scipy

## Citation

```bibtex
@article{demidont2026prevention,
  author = {Demidont, AC},
  title = {The Prevention Theorem: Time-Dependent Constraints on Post-Exposure Prophylaxis for HIV},
  journal = {Preprints.org},
  year = {2026},
  note = {Preprints ID: 193808}
}
```

## License
MIT License - See LICENSE file in parent directory.
