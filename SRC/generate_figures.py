#!/usr/bin/env python3
"""
AIDS and Behavior Figure Generator
===================================

Generates publication-quality figures for AIDS & Behavior submission.
Reads data from ../data/csv_xlsx/ and outputs to ../data/figures/

AIDS & Behavior Requirements:
- 300 DPI minimum
- TIFF preferred format  
- Single column: 84mm (3.31")
- Double column: 174mm (6.85")
- Font: 8-12pt sans-serif (Arial)

Main Figures:
    Fig 1: LAI-PrEP Cascade Comparison (MSM vs PWID)
    Fig 2: Three-Layer Barrier Decomposition
    Fig 3: Policy Scenario Analysis
    Fig 4: Stochastic Avoidance Failure Prediction
    Fig 5: Signal-to-Noise Ratio / LOOCV Framework

Usage:
    python generate_figures.py
    python generate_figures.py --input-dir ../data/csv_xlsx --output-dir ../data/figures

Author: AC Demidont, DO / Nyx Dynamics LLC
Date: January 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
from pathlib import Path

# =============================================================================
# AIDS & BEHAVIOR SPECIFICATIONS
# =============================================================================

SINGLE_COL_MM = 84
DOUBLE_COL_MM = 174
MM_TO_INCH = 1 / 25.4

SINGLE_COL = SINGLE_COL_MM * MM_TO_INCH
DOUBLE_COL = DOUBLE_COL_MM * MM_TO_INCH

# Publication settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
})

# Colorblind-friendly palette
COLORS = {
    'pwid': '#D55E00',
    'msm': '#0072B2',
    'policy': '#CC79A7',
    'stigma': '#F0E442',
    'infrastructure': '#009E73',
    'research': '#56B4E9',
    'ml': '#E69F00',
    'testing': '#999999',
    'pathogen': '#000000',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(input_dir: str) -> dict:
    """Load results from JSON file"""
    json_path = os.path.join(input_dir, "architectural_barrier_results.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Results file not found: {json_path}")


# =============================================================================
# FIGURE FUNCTIONS
# =============================================================================

def save_figure(fig, name: str, output_dir: str):
    """Save figure in TIFF, PNG, and EPS formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # TIFF (preferred)
    fig.savefig(os.path.join(output_dir, f"{name}.tif"), format='tiff', 
                dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    
    # PNG (preview)
    fig.savefig(os.path.join(output_dir, f"{name}.png"), format='png',
                dpi=300, bbox_inches='tight')
    
    # EPS (vector)
    fig.savefig(os.path.join(output_dir, f"{name}.eps"), format='eps',
                bbox_inches='tight')
    
    print(f"Saved: {name} (.tif, .png, .eps)")


def fig1_cascade_comparison(data: dict, output_dir: str):
    """Figure 1: LAI-PrEP Cascade Comparison (MSM vs PWID)"""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    
    pwid_data = data["cascade_results"][0]
    msm_data = data["msm_comparison"]
    
    steps = list(pwid_data["step_probabilities"].keys())
    step_labels = [s.replace('_', '\n').title() for s in steps]
    
    pwid_probs = [pwid_data["step_probabilities"][s] * 100 for s in steps]
    msm_probs = [msm_data["cascade_steps"][s] * 100 for s in steps]
    
    x = np.arange(len(steps))
    width = 0.35
    
    bars_msm = ax.bar(x - width/2, msm_probs, width,
                      label=f'MSM (P(R₀=0)={msm_data["p_r0_zero"]*100:.1f}%)',
                      color=COLORS['msm'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_pwid = ax.bar(x + width/2, pwid_probs, width,
                       label=f'PWID (P(R₀=0)={pwid_data["observed_r0_zero_rate"]*100:.3f}%)',
                       color=COLORS['pwid'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Step Completion Probability (%)')
    ax.set_xlabel('Prevention Cascade Step')
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=7)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    
    # Disparity annotation
    disparity = data.get("disparity_fold", "N/A")
    ax.annotate(f'Disparity: {disparity:,.0f}-fold',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    save_figure(fig, "Fig1_CascadeComparison", output_dir)
    plt.close(fig)


def fig2_barrier_decomposition(data: dict, output_dir: str):
    """Figure 2: Three-Layer Barrier Decomposition"""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    
    current = data["cascade_results"][0]
    three_layer = current["three_layer_decomposition"]
    barrier_pct = current["barrier_decomposition_pct"]
    
    # Panel A: Three-layer pie
    ax = axes[0]
    labels = ['HIV Testing\n(6.9%)', 'Architectural\n(93.1%)']
    sizes = [three_layer["hiv_testing"], three_layer["architectural"]]
    colors_pie = [COLORS['testing'], COLORS['infrastructure']]
    
    wedges, _ = ax.pie(sizes, explode=(0, 0.05), colors=colors_pie,
                       startangle=90, wedgeprops=dict(edgecolor='black', linewidth=0.5))
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(-0.1, 0.5), fontsize=8)
    ax.set_title('A. Three-Layer Decomposition', fontweight='bold', fontsize=10)
    ax.text(0.5, -0.15, 'Pathogen Biology: 0.0%', transform=ax.transAxes, 
            ha='center', fontsize=7, style='italic')
    
    # Panel B: Architectural subtypes
    ax = axes[1]
    arch_data = {
        'Policy': barrier_pct["policy"],
        'Infrastructure': barrier_pct["infrastructure"],
        'Stigma': barrier_pct["stigma"],
        'Machine Learning': barrier_pct["machine_learning"],
        'Research Exclusion': barrier_pct["research_exclusion"],
    }
    
    labels_arch = list(arch_data.keys())
    values = list(arch_data.values())
    colors_arch = [COLORS['policy'], COLORS['infrastructure'], COLORS['stigma'],
                   COLORS['ml'], COLORS['research']]
    
    bars = ax.barh(labels_arch, values, color=colors_arch, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Contribution to Prevention Failure (%)')
    ax.set_title('B. Architectural Barrier Subtypes', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 45)
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, "Fig2_BarrierDecomposition", output_dir)
    plt.close(fig)


def fig3_policy_scenarios(data: dict, output_dir: str):
    """Figure 3: Policy Scenario Analysis"""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))
    
    scenarios = [r["scenario"] for r in data["cascade_results"]]
    p_r0_zero = [r["observed_r0_zero_rate"] * 100 for r in data["cascade_results"]]
    ci_lower = [(r["observed_r0_zero_rate"] - r["r0_zero_95ci"][0]) * 100 
                for r in data["cascade_results"]]
    ci_upper = [(r["r0_zero_95ci"][1] - r["observed_r0_zero_rate"]) * 100 
                for r in data["cascade_results"]]
    
    short_labels = ['Current\nPolicy', 'Decrim\nOnly', 'Decrim+\nStigma',
                    'SSP-\nIntegrated', 'Full HR', 'HR+\nPURPOSE-4',
                    'HR+ML\nDebias', 'Theoretical\nMax']
    
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(scenarios)))
    
    bars = ax.bar(short_labels, p_r0_zero, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5,
                  yerr=[ci_lower, ci_upper], capsize=3, error_kw={'linewidth': 0.8})
    
    # MSM reference
    msm_p = data["msm_comparison"]["p_r0_zero"] * 100
    ax.axhline(y=msm_p, color=COLORS['msm'], linestyle='--', linewidth=2,
               label=f'MSM Reference ({msm_p:.1f}%)')
    
    ax.set_ylabel('P(R₀=0) %')
    ax.set_xlabel('Policy Scenario')
    ax.legend(loc='upper left', fontsize=8)
    
    for bar, val in zip(bars, p_r0_zero):
        label = f'{val:.2f}%' if val < 1 else f'{val:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha='center', va='bottom', fontsize=7)
    
    ax.set_ylim(0, max(p_r0_zero) * 1.2)
    plt.tight_layout()
    save_figure(fig, "Fig3_PolicyScenarios", output_dir)
    plt.close(fig)


def fig4_stochastic_avoidance(data: dict, output_dir: str):
    """Figure 4: Stochastic Avoidance Failure Prediction"""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    
    sa = data["stochastic_avoidance"]
    
    # Panel A: Cumulative probability
    ax = axes[0]
    years = np.arange(0, 16)
    p_annual = 0.18
    p_cumulative = 1 - (1 - p_annual) ** years
    p_cumulative = p_cumulative * (sa["probability_outbreak_10_years"] / p_cumulative[10])
    p_cumulative = np.clip(p_cumulative, 0, 1)
    
    ax.plot(years + 2024, p_cumulative * 100, color=COLORS['pwid'], linewidth=2)
    ax.fill_between(years + 2024, 0, p_cumulative * 100, color=COLORS['pwid'], alpha=0.2)
    
    ax.axhline(y=sa["probability_outbreak_5_years"] * 100, color='red', linestyle=':',
               label=f'5-year: {sa["probability_outbreak_5_years"]*100:.1f}%')
    ax.axhline(y=sa["probability_outbreak_10_years"] * 100, color='orange', linestyle=':',
               label=f'10-year: {sa["probability_outbreak_10_years"]*100:.1f}%')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Outbreak Probability (%)')
    ax.set_title('A. Outbreak Probability Over Time', fontweight='bold', fontsize=10)
    ax.set_xlim(2024, 2039)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=7)
    
    # Panel B: Distribution
    ax = axes[1]
    np.random.seed(42)
    outbreak_times = np.random.exponential(scale=sa["mean_years_to_outbreak"], size=5000)
    outbreak_times = outbreak_times[outbreak_times <= 15]
    
    ax.hist(outbreak_times, bins=15, color=COLORS['pwid'], alpha=0.7,
            edgecolor='black', linewidth=0.5, density=True)
    ax.axvline(x=sa["median_years_to_outbreak"], color='black', linewidth=2,
               label=f'Median: {sa["median_years_to_outbreak"]:.0f} years')
    ax.axvline(x=5, color='red', linestyle='--', label='5-year horizon')
    ax.axvline(x=10, color='orange', linestyle='--', label='10-year horizon')
    
    ax.set_xlabel('Years to Outbreak')
    ax.set_ylabel('Density')
    ax.set_title('B. Time-to-Outbreak Distribution', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 15)
    ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    save_figure(fig, "Fig4_StochasticAvoidance", output_dir)
    plt.close(fig)


def fig5_snr_loocv(data: dict, output_dir: str):
    """Figure 5: Signal-to-Noise Ratio / LOOCV Framework"""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    
    # Panel A: SNR comparison
    ax = axes[0]
    populations = ['MSM', 'PWID']
    snr_values = [9180, 76.4]
    colors_snr = [COLORS['msm'], COLORS['pwid']]
    
    bars = ax.bar(populations, snr_values, color=colors_snr, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Signal-to-Noise Ratio (log scale)')
    ax.set_title('A. Literature Evidence Base', fontweight='bold', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(10, 20000)
    
    ax.annotate('120-fold\ndisparity', xy=(0.5, 400), fontsize=10, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    for bar, val in zip(bars, snr_values):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
    
    # Panel B: LOOCV framework
    ax = axes[1]
    y_pos = [0, 1]
    widths = [19800, 2413]
    labels = ['Training Set\n(9+ trials)', 'Held-Out Set\n(Bangkok only)']
    colors_loocv = [COLORS['msm'], COLORS['pwid']]
    
    bars = ax.barh(y_pos, widths, color=colors_loocv, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Trial Participants')
    ax.set_title('B. LOOCV Framework', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 25000)
    
    ax.text(19800 + 500, 0, '19,800', va='center', fontsize=8)
    ax.text(2413 + 500, 1, '2,413', va='center', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, "Fig5_SNR_LOOCV", output_dir)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate AIDS & Behavior figures")
    parser.add_argument("--input-dir", type=str, default="../data/csv_xlsx",
                       help="Directory with JSON results")
    parser.add_argument("--output-dir", type=str, default="../data/figures",
                       help="Directory for figure output")
    args = parser.parse_args()
    
    print("=" * 70)
    print("AIDS AND BEHAVIOR FIGURE GENERATOR")
    print("=" * 70)
    print()
    
    # Load data
    print(f"Loading data from: {args.input_dir}")
    data = load_results(args.input_dir)
    
    # Generate figures
    print(f"\nGenerating figures to: {args.output_dir}")
    print()
    
    fig1_cascade_comparison(data, args.output_dir)
    fig2_barrier_decomposition(data, args.output_dir)
    fig3_policy_scenarios(data, args.output_dir)
    fig4_stochastic_avoidance(data, args.output_dir)
    fig5_snr_loocv(data, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
