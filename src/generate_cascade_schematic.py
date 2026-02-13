#!/usr/bin/env python3
"""
Generate Fig 1: Prevention cascade with structural barrier layers.

Eight-step cascade showing base probability, barrier penalties, and effective
probability for PWID compared to MSM. Barrier types are color-coded.

Output:
  data/figures/Fig1_CascadeSchematic.tif
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "parameters.json"
OUT_DIR = ROOT / "data" / "figures"

# ── Load data ────────────────────────────────────────────────────────────────
with open(CONFIG) as f:
    params = json.load(f)

pwid_steps = params['cascade_steps']['pwid']
msm_steps = params['cascade_steps']['msm']

step_names = ['awareness', 'willingness', 'healthcare_access', 'disclosure',
              'provider_willing', 'hiv_testing_adequate', 'first_injection',
              'sustained_engagement']
step_labels = [
    '1. Awareness\n   of PrEP',
    '2. Willingness\n   to use PrEP',
    '3. Healthcare\n   access',
    '4. Risk\n   disclosure',
    '5. Provider willing\n   to prescribe',
    '6. Adequate\n   HIV testing',
    '7. First\n   injection/dose',
    '8. Sustained\n   engagement',
]

# Barrier categories and their colors
BARRIER_COLORS = {
    'policy_penalty': '#C0392B',       # Red — Policy/Criminalization
    'stigma_penalty': '#E67E22',       # Orange — Stigma
    'infrastructure_penalty': '#2980B9', # Blue — Infrastructure
    'research_penalty': '#8E44AD',     # Purple — Research exclusion
    'testing_penalty': '#16A085',      # Teal — Testing
    'ml_penalty': '#7F8C8D',           # Gray — ML bias
}

BARRIER_LABELS = {
    'policy_penalty': 'Policy / Criminalization',
    'stigma_penalty': 'Stigma',
    'infrastructure_penalty': 'Infrastructure',
    'research_penalty': 'Research Exclusion',
    'testing_penalty': 'HIV Testing',
    'ml_penalty': 'ML / Algorithmic Bias',
}

# ── Compute data ─────────────────────────────────────────────────────────────
n_steps = len(step_names)

pwid_base = []
pwid_barriers = []  # list of dicts {barrier_key: value}
pwid_effective = []
msm_vals = []

for step in step_names:
    step_data = pwid_steps[step]
    base = step_data['base_probability']
    barriers = {k: v for k, v in step_data.items() if k != 'base_probability'}
    effective = base - sum(barriers.values())
    pwid_base.append(base)
    pwid_barriers.append(barriers)
    pwid_effective.append(effective)
    msm_vals.append(msm_steps[step])

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

y_positions = np.arange(n_steps)[::-1]  # top-to-bottom
bar_height = 0.35

# Draw PWID bars (top of each pair)
for i, step in enumerate(step_names):
    y = y_positions[i] + bar_height / 2 + 0.02
    base = pwid_base[i]
    barriers = pwid_barriers[i]
    effective = pwid_effective[i]

    # Draw effective probability (remaining bar)
    ax.barh(y, effective, height=bar_height, color='#27AE60', alpha=0.85,
            edgecolor='white', linewidth=0.5, zorder=3)

    # Stack barrier penalties on top of effective
    left = effective
    # Sort barriers by their position in BARRIER_COLORS for consistent stacking
    barrier_order = ['policy_penalty', 'stigma_penalty', 'infrastructure_penalty',
                     'research_penalty', 'testing_penalty', 'ml_penalty']
    for bk in barrier_order:
        bv = barriers.get(bk, 0)
        if bv > 0:
            ax.barh(y, bv, height=bar_height, left=left,
                    color=BARRIER_COLORS[bk], alpha=0.85,
                    edgecolor='white', linewidth=0.5, zorder=3)
            left += bv

    # Label effective probability
    ax.text(effective - 0.01, y, f'{effective:.2f}', ha='right', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)

# Draw MSM bars (bottom of each pair)
for i in range(n_steps):
    y = y_positions[i] - bar_height / 2 - 0.02
    val = msm_vals[i]
    ax.barh(y, val, height=bar_height, color='#3498DB', alpha=0.5,
            edgecolor='white', linewidth=0.5, zorder=2)
    ax.text(val - 0.01, y, f'{val:.2f}', ha='right', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)

# Y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels(step_labels, fontsize=8, fontfamily='Arial')

# X-axis
ax.set_xlim(0, 1.0)
ax.set_xlabel('Probability', fontsize=10, fontfamily='Arial')
ax.set_title('Prevention Cascade with Structural Barrier Layers\nPWID (colored) vs MSM (blue reference)',
             fontsize=12, fontweight='bold', fontfamily='Arial', pad=12)

# Grid
ax.xaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color='#27AE60', alpha=0.85, label='PWID Effective Probability'),
]
for bk in ['policy_penalty', 'stigma_penalty', 'infrastructure_penalty',
           'research_penalty', 'testing_penalty', 'ml_penalty']:
    legend_patches.append(
        mpatches.Patch(color=BARRIER_COLORS[bk], alpha=0.85,
                       label=f'Barrier: {BARRIER_LABELS[bk]}')
    )
legend_patches.append(
    mpatches.Patch(color='#3498DB', alpha=0.5, label='MSM Reference')
)

ax.legend(handles=legend_patches, loc='lower right', fontsize=7,
          framealpha=0.9, edgecolor='gray')

# ── Cumulative annotation ────────────────────────────────────────────────────
pwid_cum = 1.0
msm_cum = 1.0
for i in range(n_steps):
    pwid_cum *= pwid_effective[i]
    msm_cum *= msm_vals[i]

ax.text(0.98, 0.98,
        f'Cumulative: PWID = {pwid_cum:.2e}  |  MSM = {msm_cum:.4f}\n'
        f'Disparity: {msm_cum / pwid_cum:.0f}-fold',
        transform=ax.transAxes, fontsize=8, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.9))

plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "Fig1_CascadeSchematic.tif"
fig.savefig(str(out_path), dpi=300, format='tiff',
            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
print(f"Saved: {out_path}")

# Also save PNG for quick preview
png_path = OUT_DIR / "Fig1_CascadeSchematic.png"
fig.savefig(str(png_path), dpi=150, format='png', bbox_inches='tight')
print(f"Preview: {png_path}")

plt.close()
