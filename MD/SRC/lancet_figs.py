#!/usr/bin/env python3
"""
Publication-Quality Figures for The Lancet
===========================================

Generates Lancet-standard figures from existing simulation results.

Author: AC Demidont, MD / Nyx Dynamics LLC
Date: December 2024
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Lancet-compliant styling
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# Color-blind friendly palette
COLORS = {
    'blue': '#648FFF',
    'purple': '#785EF0',
    'magenta': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
    'green': '#00A878',
    'red': '#C1272D',
    'gray': '#767676',
}

# Create output directory
OUTPUT_DIR = Path("archieve/lancet_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load results
print("Loading results...")
with open('pwid_simulation_results.json', 'r') as f:
    results = json.load(f)
print(f"✓ Loaded {len(results)} scenarios\n")

# Extract data
scenarios = [r['scenario'] for r in results]
p_values = np.array([r['observed_r0_zero_rate'] * 100 for r in results])
ci_lower = np.array([r['r0_zero_95ci'][0] * 100 for r in results])
ci_upper = np.array([r['r0_zero_95ci'][1] * 100 for r in results])
cascade_completion = np.array([r['observed_cascade_completion_rate'] * 100 for r in results])

# ============================================================================
# FIGURE 1: Main Scenario Comparison
# ============================================================================
print("Generating Figure 1: Scenario Comparison...")

fig, ax = plt.subplots(figsize=(7.5, 4.5))

x_pos = np.arange(len(scenarios))
colors_gradient = plt.colormaps.get_cmap('RdYlGn')(np.linspace(0.2, 0.9, len(scenarios)))

# Error bars
yerr_lower = p_values - ci_lower
yerr_upper = ci_upper - p_values

# Bars
bars = ax.bar(x_pos, p_values, color=colors_gradient, alpha=0.8,
              edgecolor='black', linewidth=0.5)

# Error bars
ax.errorbar(x_pos, p_values, yerr=[yerr_lower, yerr_upper],
            fmt='none', ecolor='black', capsize=3, capthick=0.5, linewidth=0.5)

# Value labels
for bar, val in zip(bars, p_values):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

# Styling
ax.set_xlabel('Policy Scenario', fontsize=8, fontweight='bold')
ax.set_ylabel('Probability of Achieving R(0) = 0 (%)', fontsize=8, fontweight='bold')
ax.set_title('PWID LAI-PrEP Success Rate by Policy Scenario\n(with 95% Confidence Intervals)',
             fontsize=9, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=7)
ax.set_ylim(0, max(p_values) * 1.25)
ax.grid(axis='y', alpha=0.2, linewidth=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Reference line
current_val = p_values[0]
ax.axhline(y=current_val, color=COLORS['red'], linestyle='--',
           linewidth=1, label=f'Current Policy: {current_val:.2f}%', alpha=0.7)
ax.legend(loc='upper left', fontsize=7, frameon=False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure1_ScenarioComparison.png', dpi=600, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/Figure1_ScenarioComparison.png")
plt.close()

# ============================================================================
# FIGURE 2: Population Impact Dashboard
# ============================================================================
print("Generating Figure 2: Population Impact...")

fig = plt.figure(figsize=(7.5, 6))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30)

# Extract impact data
n_protected = np.array([r['impact']['n_protected'] / 1_000_000 for r in results])
annual_prevented = np.array([r['impact']['annual_infections_prevented'] / 1000 for r in results])
five_year_prevented = np.array([r['impact']['five_year_infections_prevented'] / 1000 for r in results])
cost_savings = np.array([r['impact']['five_year_cost_averted_billions'] for r in results])

# Panel A: People Protected
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(x_pos, n_protected, color=COLORS['blue'], alpha=0.75, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars1, n_protected):
    if val > 0.1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{val:.2f}M', ha='center', va='bottom', fontsize=6)
ax1.set_ylabel('Millions of PWID', fontsize=7)
ax1.set_title('A. People Achieving Sustained Protection', fontsize=8, fontweight='bold', loc='left')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=6)
ax1.grid(axis='y', alpha=0.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Annual Infections Prevented
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(x_pos, annual_prevented, color=COLORS['green'], alpha=0.75, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars2, annual_prevented):
    if val > 0.5:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{val:.1f}K', ha='center', va='bottom', fontsize=6)
ax2.set_ylabel('Thousands of Infections', fontsize=7)
ax2.set_title('B. Annual HIV Infections Prevented', fontsize=8, fontweight='bold', loc='left')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=6)
ax2.grid(axis='y', alpha=0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel C: 5-Year Cumulative
ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(x_pos, five_year_prevented, color=COLORS['orange'], alpha=0.75, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars3, five_year_prevented):
    if val > 2:
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{val:.1f}K', ha='center', va='bottom', fontsize=6)
ax3.set_ylabel('Thousands of Infections', fontsize=7)
ax3.set_title('C. 5-Year Cumulative Infections Prevented', fontsize=8, fontweight='bold', loc='left')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=6)
ax3.grid(axis='y', alpha=0.2)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Panel D: Cost Savings
ax4 = fig.add_subplot(gs[1, 1])
bars4 = ax4.bar(x_pos, cost_savings, color=COLORS['purple'], alpha=0.75, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars4, cost_savings):
    if val > 0.5:
        ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'${val:.2f}B', ha='center', va='bottom', fontsize=6)
ax4.set_ylabel('Billions of Dollars ($)', fontsize=7)
ax4.set_title('D. 5-Year Healthcare Cost Savings', fontsize=8, fontweight='bold', loc='left')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=6)
ax4.grid(axis='y', alpha=0.2)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

fig.suptitle('Population-Level Impact Across Policy Scenarios',
             fontsize=9, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / 'Figure2_PopulationImpact.png', dpi=600, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/Figure2_PopulationImpact.png")
plt.close()

# ============================================================================
# FIGURE 3: Cascade Waterfall (Multi-panel)
# ============================================================================
print("Generating Figure 3: Cascade Waterfall...")

fig = plt.figure(figsize=(7.5, 9))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.30)

panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

for idx, r in enumerate(results):
    row = idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])

    # Get step probabilities and calculate cumulative
    step_probs = r['step_probabilities']
    steps = list(step_probs.keys())
    cumulative = []
    current = 1.0
    for step in steps:
        current *= step_probs[step]
        cumulative.append(current * 100)

    # Color based on survival rate
    colors = [COLORS['red'] if c < 10 else COLORS['orange'] if c < 30 else
    COLORS['yellow'] if c < 50 else COLORS['green'] for c in cumulative]

    bars = ax.bar(range(len(steps)), cumulative, color=colors, alpha=0.75,
                  edgecolor='black', linewidth=0.5)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, cumulative)):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width() / 2., val - 3,
                    f'{val:.1f}', ha='center', va='top', fontsize=6, color='white', fontweight='bold')

    ax.set_ylabel('% Remaining', fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[:4] for s in steps], rotation=45, ha='right', fontsize=6)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel label and title
    ax.text(-0.15, 1.05, panel_labels[idx], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.set_title(f'{r["scenario"]}\nFinal: {cumulative[-1]:.2f}%',
                 fontsize=7, fontweight='bold', pad=8)

fig.suptitle('Cascade Attrition Across Policy Scenarios',
             fontsize=9, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / 'Figure3_CascadeWaterfall.png', dpi=600, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/Figure3_CascadeWaterfall.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("ALL LANCET FIGURES GENERATED")
print("=" * 70)
print(f"\nLocation: {OUTPUT_DIR.absolute()}/")
print("\nGenerated files:")
print("  - Figure1_ScenarioComparison.png (600 DPI)")
print("  - Figure2_PopulationImpact.png (600 DPI)")
print("  - Figure3_CascadeWaterfall.png (600 DPI)")
print("\n✓ All figures ready for publication!")