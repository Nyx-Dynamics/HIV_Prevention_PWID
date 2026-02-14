# Structural Barriers, Stochastic Avoidance, and Outbreak Risk in HIV Prevention for People Who Inject Drugs

**A Monte Carlo Simulation Study**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Abstract

People who inject drugs (PWID) face nested structural barriers to HIV prevention that are qualitatively different from those encountered by other key populations. We developed a Monte Carlo simulation modeling three barrier layers — pathogen biology, HIV testing gaps, and architectural failures (policy, stigma, infrastructure, research exclusion, algorithmic bias) — to test the hypothesis that these barriers render the basic reproduction number R(0) = 0 mathematically unachievable under current policy conditions.

Using an 8-step prevention cascade parameterized from published literature, we simulated outcomes across policy scenarios for PWID versus MSM reference populations. Cascade completion under current policy was <1% for PWID versus ~21% for MSM, a disparity driven primarily by criminalization, stigma, and infrastructure barriers. Stochastic avoidance modeling revealed that the absence of catastrophic outbreaks among PWID reflects probabilistic luck rather than policy success, with outbreak probability increasing under methamphetamine-driven network expansion.

**Keywords:** HIV prevention, people who inject drugs, structural barriers, Monte Carlo simulation, PrEP cascade, harm reduction, stochastic avoidance

## Quickstart\
\
````bash\
git clone https://github.com/Nyx-Dynamics/Prevention-Theorem.git\
cd Prevention-Theorem\
python -m venv .venv \&\& source .venv/bin/activate\
pip install -r requirements.txt\
python reproduce_all.py\
```\

## Repository Structure

```
HIV_Prevention_PWID/
├── src/
│   ├── structural_barrier_model.py       # Core 3-layer barrier model (canonical)
│   ├── architectural_barrier_model.py    # Backward-compatibility import shim
│   ├── stochastic_avoidance_enhanced.py  # Outbreak probability & sensitivity analysis
│   ├── stochastic_avoidance_v2.py        # V2: multiplicative meth × housing interaction
│   ├── hood_parameter_comparison.py      # Hood et al. (2018) parameter adjustment
│   ├── cascade_sensitivity_analysis.py   # PSA, barrier removal, step importance
│   ├── generate_outputs.py              # Publication-quality figures (BMC style)
│   └── generate_cascade_schematic.py    # Figure 1: cascade with barrier layers
├── config/
│   └── parameters.json                  # All cascade parameters (PWID + MSM)
├── data/
│   ├── figures/                         # Generated figures (PNG, EPS, TIFF)
│   └── csv_xlsx/                        # Simulation output data
├── scripts/
│   └── reproduce_supplementary_results.py  # Reproduce all supplementary figures/data
├── manuscript_prep/                     # Document preparation scripts (not analysis)
│   ├── build_additional_files.py
│   ├── revise_manuscript.py
│   └── revise_manuscript_v2.py
├── requirements.txt
└── README.md
```

## Key Findings

1. **Cascade completion:** <1% for PWID vs ~21% for MSM under current policy — a >20-fold disparity
2. **R(0) = 0 unachievable:** Current policy conditions make elimination mathematically impossible for PWID
3. **Stochastic avoidance:** Absence of outbreaks reflects probabilistic luck, not policy effectiveness
4. **Methamphetamine risk:** Network bridging through stimulant use increases outbreak probability over time
5. **Policy levers:** Decriminalization + harm reduction integration required for meaningful cascade improvement

## Reproducibility

```bash
# Clone and install
git clone https://github.com/Nyx-Dynamics/HIV_Prevention_PWID.git
cd HIV_Prevention_PWID
pip install -r requirements.txt

# Run core simulation
cd src
python structural_barrier_model.py

# Run stochastic avoidance analysis
python stochastic_avoidance_enhanced.py --output-dir ../data/figures --data-dir ../data/csv_xlsx

# Run sensitivity analysis
python cascade_sensitivity_analysis.py --output-dir ../data/csv_xlsx

# Generate publication figures
python generate_outputs.py
python generate_cascade_schematic.py

# Reproduce all supplementary results
cd ../scripts
python reproduce_supplementary_results.py
```

**Environment:** Python 3.10+, NumPy, SciPy, Matplotlib, Pandas, openpyxl
**Random seed:** 42 (fixed for reproducibility)

## Three-Layer Barrier Framework

| Layer | Barriers | Mechanism |
|-------|----------|-----------|
| Pathogen Biology | Irreversible R(0)>0 within hours of exposure | Biological constraint — not modifiable by policy |
| HIV Testing | Acute infection detection gaps, window periods | Limits early intervention regardless of access |
| Architectural | Policy (criminalization), Stigma, Infrastructure, Research Exclusion, ML/Algorithmic Bias | Modifiable structural barriers — targets for intervention |

## Data Sources

Cascade parameters derived from peer-reviewed literature including:
- CDC HIV Surveillance Reports
- Hood et al. (2018) — King County MSM-PWID methamphetamine data
- Legal Services Corporation — structural access barriers
- Published PrEP cascade studies for PWID populations

Full parameter derivation logic: `config/parameters.json`

## Preprint & Submission

- **Preprints.org:** DOI: [10.20944/preprints202601.0948.v1](https://doi.org/10.20944/preprints202601.0948.v1)
- **Under review:** BMC Public Health

## Citation

```bibtex
@article{demidont2026structural,
  title={Structural Barriers, Stochastic Avoidance, and Outbreak Risk in
         HIV Prevention for People Who Inject Drugs:
         A Monte Carlo Simulation Study},
  author={Demidont, AC},
  journal={BMC Public Health},
  year={2026},
  note={Under review}
}
```

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Author

**AC Demidont, DO**
Nyx Dynamics LLC
acdemidont@nyxdynamics.org

---

*Updated: February 2026*
