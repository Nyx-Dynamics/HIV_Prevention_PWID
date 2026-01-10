#!/usr/bin/env python3
"""
Publication-Quality Figures for AIDS and Behavior
=================================================

Generates figures from existing simulation results following 'AIDS and Behavior' 
Artwork and Illustrations Guidelines.

Author: AC Demidont, MD / Nyx Dynamics LLC
Date: January 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# --- AIDS and Behavior STYLING ---
# Guidelines:
# - Helvetica or Arial (sans serif fonts).
# - Font size: 8-12 pt (consistently sized).
# - No titles or captions within illustrations.
# - Lines at least 0.1 mm (0.3 pt) wide.
# - Resolution: 600 dpi (combination art), 300 dpi (halftones).
# - Format: EPS (preferred for vector) or TIFF (halftone).

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 10, # Used for panel labels A, B, etc.
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
})

# High-contrast, color-blind friendly palette with patterns for accessibility
COLORS = [
    '#000000', # Black
    '#E69F00', # Orange
    '#56B4E9', # Sky Blue
    '#009E73', # Bluish Green
    '#F0E442', # Yellow
    '#0072B2', # Blue
    '#D55E00', # Vermillion
    '#CC79A7'  # Reddish Purple
]

PATTERNS = ['', '/', '\\', 'x', '.', '*', 'o', 'O', '-']

# Figure dimensions (AIDS and Behavior)
# 84 mm for double-column, 174 mm for single-column
MM_TO_INCH = 1 / 25.4
WIDTH_SINGLE = 174 * MM_TO_INCH
WIDTH_DOUBLE = 84 * MM_TO_INCH

FIG_DIR = Path("../data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("../data/csv_xlsx")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_publication_fig(fig, fig_name):
    """Saves figure in EPS and TIFF formats as per guidelines."""
    # EPS for vector
    fig.savefig(FIG_DIR / f"{fig_name}.eps", format='eps', dpi=600)
    # TIFF for high-quality raster
    fig.savefig(FIG_DIR / f"{fig_name}.tif", format='tif', dpi=600, 
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"✓ Saved: {fig_name}.eps and .tif")

# Load results
print("Loading results...")
try:
    # Try local data dir first, then root
    if (DATA_DIR / 'pwid_simulation_results.json').exists():
        results_path = DATA_DIR / 'pwid_simulation_results.json'
    else:
        results_path = 'pwid_simulation_results.json'
        
    with open(results_path, 'r') as f:
        results = json.load(f)
    print(f"✓ Loaded {len(results)} scenarios (from {results_path})\n")
except FileNotFoundError:
    print("Error: pwid_simulation_results.json not found.")
    results = []

if results:
    scenarios = [r['scenario'] for r in results]
    p_values = np.array([r['observed_r0_zero_rate'] * 100 for r in results])
    ci_lower = np.array([r['r0_zero_95ci'][0] * 100 for r in results])
    ci_upper = np.array([r['r0_zero_95ci'][1] * 100 for r in results])

    # ============================================================================
    # FIGURE 1: Main Scenario Comparison (Single Column Width)
    # ============================================================================
    print("Generating Fig1: Scenario Comparison...")
    
    # Guidelines: No titles within illustration. Arabic numerals for numbering.
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 5))
    
    x_pos = np.arange(len(scenarios))
    yerr_lower = p_values - ci_lower
    yerr_upper = ci_upper - p_values
    
    # Use patterns and colors for accessibility
    for i in range(len(scenarios)):
        ax.bar(x_pos[i], p_values[i], color=COLORS[i % len(COLORS)], 
               alpha=1.0, edgecolor='black', linewidth=0.8,
               hatch=PATTERNS[i % len(PATTERNS)], label=scenarios[i])

    ax.errorbar(x_pos, p_values, yerr=[yerr_lower, yerr_upper],
                fmt='none', ecolor='black', capsize=3, capthick=0.8, linewidth=0.8)

    # Value labels - kept minimal
    for i, val in enumerate(p_values):
        ax.text(i, val + (max(p_values)*0.05), f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Policy Scenario')
    ax.set_ylabel('Probability of Achieving R(0) = 0 (%)')
    # ax.set_title(...) -> Removed as per guidelines (no titles within illustrations)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylim(0, 110) # Enough space for labels
    ax.grid(axis='y', color='lightgray', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reference line for current policy
    current_val = p_values[0]
    ax.axhline(y=current_val, color='black', linestyle=':', 
               linewidth=1)
    
    plt.tight_layout()
    save_publication_fig(fig, 'Fig1')
    plt.close()

    # ============================================================================
    # FIGURE 2: Population Impact (Double Column Width)
    # ============================================================================
    print("Generating Fig2: Population Impact...")
    
    fig = plt.figure(figsize=(WIDTH_SINGLE, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    n_protected = np.array([r['impact']['n_protected'] / 1_000_000 for r in results])
    annual_prevented = np.array([r['impact']['annual_infections_prevented'] / 1000 for r in results])
    five_year_prevented = np.array([r['impact']['five_year_infections_prevented'] / 1000 for r in results])
    cost_savings = np.array([r['impact']['five_year_cost_averted_billions'] for r in results])

    metrics = [
        (n_protected, 'Millions of PWID', 'a'),
        (annual_prevented, 'Thousands of Infections', 'b'),
        (five_year_prevented, 'Thousands of Infections', 'c'),
        (cost_savings, 'Billions of Dollars ($)', 'd')
    ]

    for i, (data, ylabel, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        for j in range(len(scenarios)):
            ax.bar(x_pos[j], data[j], color=COLORS[j % len(COLORS)], 
                   alpha=1.0, edgecolor='black', linewidth=0.5,
                   hatch=PATTERNS[j % len(PATTERNS)])
        
        ax.set_ylabel(ylabel)
        # Panel labels (a, b, c, d)
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', color='lightgray', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # fig.tight_layout() - potentially problematic with multi-panel
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95, bottom=0.15, left=0.1, right=0.95)
    save_publication_fig(fig, 'Fig2')
    plt.close()

    # ============================================================================
    # FIGURE 3: Cascade Waterfall (Single Column Width)
    # ============================================================================
    print("Generating Fig3: Cascade Waterfall...")
    
    # We'll plot a selection or a combined view to fit better
    fig = plt.figure(figsize=(WIDTH_SINGLE, 10))
    # guidelines: Figure parts denoted by lowercase letters
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.6, wspace=0.3)
    
    sub_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    for idx, r in enumerate(results[:7]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        step_probs = r['step_probabilities']
        steps = list(step_probs.keys())
        cumulative = []
        current = 1.0
        for step in steps:
            current *= step_probs[step]
            cumulative.append(current * 100)

        # High contrast colors for cascade
        bars = ax.bar(range(len(steps)), cumulative, color=COLORS[1], 
                      alpha=1.0, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('% Remaining')
        ax.set_ylim(0, 105)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([s[:4] for s in steps], rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', color='lightgray', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.text(-0.2, 1.15, f"{sub_labels[idx]}) {r['scenario']}", 
                transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    # fig.tight_layout()
    fig.subplots_adjust(hspace=0.7, wspace=0.3, top=0.95, bottom=0.1, left=0.1, right=0.95)
    save_publication_fig(fig, 'Fig3')
    plt.close()

print("\n" + "=" * 70)
print("AIDS AND BEHAVIOR FIGURES GENERATED")
print("=" * 70)
print(f"Location: {FIG_DIR.absolute()}/")
print("Files: Fig1.eps/tif, Fig2.eps/tif, Fig3.eps/tif")
