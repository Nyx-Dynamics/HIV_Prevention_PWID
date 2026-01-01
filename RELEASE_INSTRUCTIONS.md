# Release and Push Instructions

## Pre-Release Checklist

- [x] All code tested and runs without errors
- [x] Results files regenerated with final code
- [x] All scripts updated with configurable CLI arguments (`argparse`)
- [x] Visualization scripts updated to handle new and legacy JSON formats
- [x] No "manufactured death" terminology in any production files
- [x] README.md and CHANGELOG.md complete (Version 1.2.0)
- [x] CITATION.cff complete
- [x] LICENSE file present

---

## Step 1: Prepare Local Repository

```bash
# Navigate to your repository
cd /path/to/hiv-prevention-master

# Ensure you're on main branch
git checkout main

# Pull any remote changes
git pull origin main
```

## Step 2: Clean Up Duplicates and Old Versions

If you have legacy files or redundant copies, move them to the `archieve/` directory or delete them:

```bash
# Remove redundant downloads (if present)
rm *_dl.py

# Ensure legacy 'manufactured_death' scripts are archived
mv manufactured_death_model.py archieve/ 2>/dev/null
mv manufactured_death_results.json archieve/ 2>/dev/null
```

## Step 3: Stage and Commit Changes

```bash
# Stage all changes
git add .

# Review what will be committed
git status

# Commit with descriptive message
git commit -m "Version 1.2.0: Terminology shift and CLI enhancement

- Completed transition to 'Architectural Barrier Model'
- Added argparse to all scripts for configurable input/output paths
- Optimized requirements.txt and streamlined dependencies
- Improved visualization script robustness for JSON formats"
```

## Step 4: Push to GitHub

```bash
git push origin main
```

---

## Step 5: Create GitHub Release

### Via GitHub Web Interface:

1. Go to https://github.com/Nyx-Dynamics/hiv-prevention-master
2. Click **Releases** → **Draft a new release**
3. Click **Choose a tag** → type `v1.2.0` → **Create new tag**
4. **Release title:** `v1.2.0 - Terminology Refactor and CLI Update`
5. **Description:**

```markdown
## Architectural Barrier HIV Prevention Modeling

Major update implementing the final terminology shift and enhancing user control.

### Key Features (v1.2.0)
- **Unified Terminology:** Shift from "Manufactured Death" to "Architectural Barrier Model".
- **CLI Integration:** All primary scripts now support `--output-dir` and other configurable parameters.
- **Improved Privacy:** Ability to redirect sensitive simulation results outside the project folder.
- **Robust Visualizations:** Figures FigS1-FigS8 and Fig1-Fig5 fully supported with dual PNG/PDF export.

### Primary Scripts
- `architectural_barrier_model.py`: Main Monte Carlo simulation.
- `cascade_sensitivity_analysis.py`: PSA and step importance analysis.
- `stochastic_avoidance_enhanced.py`: Regional outbreak forecasting.
- `visualize_md_results.py`: Manuscript-ready figure generation.

### Citation
Demidont AC. Structural barriers drive near-zero population-level effectiveness of Long Acting Injectable HIV prevention (LAI-PrEP) among people who inject drugs: A Computational Modeling Study. The Lancet HIV (2025). [Submitted]
```

6. Click **Publish release**

---

## Step 6: Verify Deployment

```bash
# Clone fresh copy to verify
cd /tmp
git clone https://github.com/Nyx-Dynamics/hiv-prevention-master.git
cd hiv-prevention-master

# Test main simulation with CLI flag
python architectural_barrier_model.py --output-dir ./test_run --n-individuals 1000

# Verify figures can be generated
python visualize_md_results.py --input ./test_run/architectural_barrier_results.json --output-dir ./test_figures
```

---

## Quick Reference - Single Command Block

```bash
# All-in-one (run from repository root)
git add .
git commit -m "Release v1.2.0: Terminology refactor and CLI update"
git push origin main
git tag -a v1.2.0 -m "Version 1.2.0 release"
git push origin v1.2.0
```
