# Release and Push Instructions

## Pre-Release Checklist

- [ ] All code tested and runs without errors
- [ ] Results files regenerated with final code
- [ ] Visualization script updated to use `architectural_barrier_results.json`
- [ ] No "manufactured death" terminology in any files
- [ ] README.md complete
- [ ] CITATION.cff complete
- [ ] LICENSE file present

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

## Step 2: Add/Update Files

Copy these files to your repository:

```bash
# Core model (renamed)
cp architectural_barrier_model.py /path/to/hiv-prevention-master/

# Update visualization script filename reference
# In visualize_results.py, change:
#   manufactured_death_results.json → architectural_barrier_results.json

# Results files
cp architectural_barrier_results.json /path/to/hiv-prevention-master/
cp architectural_barrier_results.csv /path/to/hiv-prevention-master/

# Documentation
cp README.md /path/to/hiv-prevention-master/
cp CITATION.cff /path/to/hiv-prevention-master/
cp LICENSE /path/to/hiv-prevention-master/
```

## Step 3: Remove Old Files (if present)

```bash
git rm manufactured_death_model.py
git rm manufactured_death_results.json
git rm manufactured_death_results.csv
```

## Step 4: Stage and Commit

```bash
# Stage all changes
git add .

# Review what will be committed
git status

# Commit with descriptive message
git commit -m "Refactor: rename to architectural barrier model for Lancet HIV submission

- Rename ManufacturedDeathModel → ArchitecturalBarrierModel
- Update all output filenames and terminology
- Add README, CITATION.cff, LICENSE
- Preserve all mathematical logic and results"
```

## Step 5: Push to GitHub

```bash
git push origin main
```

---

## Step 6: Create GitHub Release

### Via GitHub Web Interface:

1. Go to https://github.com/Nyx-Dynamics/hiv-prevention-master
2. Click **Releases** → **Draft a new release**
3. Click **Choose a tag** → type `v1.0.0` → **Create new tag**
4. **Release title:** `v1.0.0 - Lancet HIV Submission`
5. **Description:**

```markdown
## HIV Prevention Barrier Modeling for PWID

Initial release accompanying submission to The Lancet HIV.

### Key Features
- Monte Carlo simulation of 8-step HIV prevention cascade
- Three-layer barrier framework (biology, testing, architectural)
- Policy scenario analysis across 8 intervention levels
- Stochastic avoidance failure modeling
- MSM vs PWID disparity quantification

### Files
- `architectural_barrier_model.py` - Main simulation model
- `visualize_results.py` - Figure generation
- `architectural_barrier_results.json` - Full simulation outputs

### Citation
Demidont AC. Structural barriers drive near-zero population-level effectiveness of HIV prevention among people who inject drugs: A computational modelling study. The Lancet HIV (2025). [Submitted]
```

6. Click **Publish release**

---

## Step 7: Get Zenodo DOI (Optional but Recommended)

1. Go to https://zenodo.org
2. Log in with GitHub
3. Enable the repository for Zenodo archiving
4. The GitHub release will automatically create a Zenodo DOI
5. Update README.md badge with actual DOI

---

## Step 8: Verify

```bash
# Clone fresh copy to verify
cd /tmp
git clone https://github.com/Nyx-Dynamics/hiv-prevention-master.git
cd hiv-prevention-master

# Test that model runs
python architectural_barrier_model.py

# Verify outputs match expected values
```

---

## Quick Reference - Single Command Block

```bash
# All-in-one (run from repository root)
git checkout main
git pull origin main
git add .
git status
git commit -m "Refactor: rename to architectural barrier model for Lancet HIV submission"
git push origin main
git tag -a v1.0.0 -m "Lancet HIV submission release"
git push origin v1.0.0
```

---

## Troubleshooting

**Problem:** Push rejected  
**Solution:** `git pull --rebase origin main` then push again

**Problem:** Large file error  
**Solution:** Check for PDFs or data files >100MB, add to .gitignore

**Problem:** Authentication failed  
**Solution:** Use GitHub CLI (`gh auth login`) or personal access token
