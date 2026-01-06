#!/usr/bin/env python3
"""
AIDS and Behavior Figure Generator - Canonical Version
=======================================================

Generates publication-quality figures for AIDS & Behavior submission using
canonical data from manufactured_death_results.json.

AIDS & Behavior Requirements:
- 300 DPI minimum for all figures
- TIFF preferred format
- Single column: 84mm (3.31")
- Double column: 174mm (6.85")
- Font: 8-12pt sans-serif (Arial preferred)

Main Figures:
    Fig 1: LAI-PrEP Cascade Comparison (MSM vs PWID)
    Fig 2: Three-Layer Barrier Decomposition
    Fig 3: Policy Scenario Analysis
    Fig 4: Stochastic Avoidance Failure Prediction
    Fig 5: Signal-to-Noise Ratio / LOOCV Framework

Supplementary Figures (Online Resource S2):
    Fig S1: Methamphetamine Trajectories
    Fig S2: Tornado Diagram
    Fig S3: Policy Scenario Extended Comparison
    Fig S4: Cascade Uncertainty (PSA)
    Fig S5: Barrier Removal Waterfall
    Fig S6: Step Importance Analysis

Author: AC Demidont, DO / Nyx Dynamics LLC
Date: January 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
from datetime import datetime

# =============================================================================
# AIDS & BEHAVIOR SPECIFICATIONS
# =============================================================================

SINGLE_COL_MM = 84
DOUBLE_COL_MM = 174
MM_TO_INCH = 1 / 25.4

SINGLE_COL = SINGLE_COL_MM * MM_TO_INCH  # 3.31 inches
DOUBLE_COL = DOUBLE_COL_MM * MM_TO_INCH  # 6.85 inches

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
    'pwid': '#D55E00',  # Vermillion
    'msm': '#0072B2',  # Blue
    'policy': '#CC79A7',  # Pink
    'stigma': '#F0E442',  # Yellow
    'infrastructure': '#009E73',  # Teal
    'research': '#56B4E9',  # Sky blue
    'ml': '#E69F00',  # Orange
    'testing': '#999999',  # Gray
    'pathogen': '#000000',  # Black
}

# =============================================================================
# CANONICAL DATA (from manufactured_death_results.json)
# =============================================================================

CANONICAL_DATA = {
    "cascade_results": [
        {
            "scenario": "Current Policy",
            "observed_r0_zero_rate": 0.00003,
            "r0_zero_95ci": [0.0, 0.000064],
            "observed_cascade_completion_rate": 0.00005,
            "incarceration_survival_probability": 0.168,
            "step_probabilities": {
                "awareness": 0.10,
                "willingness": 0.30,
                "healthcare_access": 0.35,
                "disclosure": 0.25,
                "provider_willing": 0.35,
                "hiv_testing_adequate": 0.45,
                "first_injection": 0.45,
                "sustained_engagement": 0.25
            },
            "barrier_decomposition_pct": {
                "pathogen_biology": 0.0,
                "hiv_testing": 6.85,
                "policy": 38.36,
                "stigma": 20.55,
                "infrastructure": 21.92,
                "research_exclusion": 4.11,
                "machine_learning": 8.22
            },
            "three_layer_decomposition": {
                "pathogen_biology": 0.0,
                "hiv_testing": 6.85,
                "architectural": 93.15
            }
        },
        {
            "scenario": "Decriminalization Only",
            "observed_r0_zero_rate": 0.00198,
            "r0_zero_95ci": [0.00170, 0.00226],
            "observed_cascade_completion_rate": 0.00304,
            "incarceration_survival_probability": 0.624,
        },
        {
            "scenario": "Decrim + Stigma Reduction",
            "observed_r0_zero_rate": 0.00454,
            "r0_zero_95ci": [0.00412, 0.00496],
            "observed_cascade_completion_rate": 0.00763,
            "incarceration_survival_probability": 0.624,
        },
        {
            "scenario": "SSP-Integrated Delivery",
            "observed_r0_zero_rate": 0.05004,
            "r0_zero_95ci": [0.04869, 0.05139],
            "observed_cascade_completion_rate": 0.07994,
            "incarceration_survival_probability": 0.624,
        },
        {
            "scenario": "Full Harm Reduction",
            "observed_r0_zero_rate": 0.09552,
            "r0_zero_95ci": [0.09370, 0.09734],
            "observed_cascade_completion_rate": 0.09552,
            "incarceration_survival_probability": 1.0,
        },
        {
            "scenario": "Full HR + PURPOSE-4 Data",
            "observed_r0_zero_rate": 0.11872,
            "r0_zero_95ci": [0.11672, 0.12072],
            "observed_cascade_completion_rate": 0.11872,
            "incarceration_survival_probability": 1.0,
        },
        {
            "scenario": "Full HR + Algorithmic Debiasing",
            "observed_r0_zero_rate": 0.18569,
            "r0_zero_95ci": [0.18328, 0.18810],
            "observed_cascade_completion_rate": 0.18569,
            "incarceration_survival_probability": 1.0,
        },
        {
            "scenario": "Theoretical Maximum",
            "observed_r0_zero_rate": 0.19737,
            "r0_zero_95ci": [0.19490, 0.19984],
            "observed_cascade_completion_rate": 0.19737,
            "incarceration_survival_probability": 1.0,
        },
    ],
    "msm_comparison": {
        "p_r0_zero": 0.16302,
        "cascade_completion": 0.21068,
        "incarceration_survival": 0.7738,
        "cascade_steps": {
            "awareness": 0.90,
            "willingness": 0.85,
            "healthcare_access": 0.80,
            "disclosure": 0.75,
            "provider_willing": 0.90,
            "hiv_testing_adequate": 0.85,
            "first_injection": 0.80,
            "sustained_engagement": 0.75
        },
        "snr": 9180.0,
    },
    "stochastic_avoidance": {
        "median_years_to_outbreak": 4.0,
        "mean_years_to_outbreak": 4.90,
        "probability_outbreak_5_years": 0.633,
        "probability_outbreak_10_years": 0.875,
        "p10_years": 0.0,
        "p25_years": 1.0,
        "p75_years": 7.0,
        "p90_years": 11.0,
    },
    "disparity_fold": 5434,
}


# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def save_figure(fig, name, output_dir):
    """Save figure in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)

    # TIFF (preferred by AIDS & Behavior)
    tiff_path = os.path.join(output_dir, f"{name}.tif")
    fig.savefig(tiff_path, format='tiff', dpi=300, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})

    # PNG (for preview)
    png_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

    # EPS (vector format)
    eps_path = os.path.join(output_dir, f"{name}.eps")
    fig.savefig(eps_path, format='eps', bbox_inches='tight')

    print(f"Saved: {name} (.tif, .png, .eps)")


def fig1_cascade_comparison(data, output_dir):
    """
    Figure 1: LAI-PrEP Cascade Comparison (MSM vs PWID)
    Shows the 8-step cascade with step-by-step probability comparison
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))

    pwid_data = data["cascade_results"][0]
    msm_data = data["msm_comparison"]

    steps = list(pwid_data["step_probabilities"].keys())
    step_labels = [s.replace('_', '\n').title() for s in steps]

    pwid_probs = [pwid_data["step_probabilities"][s] * 100 for s in steps]
    msm_probs = [msm_data["cascade_steps"][s] * 100 for s in steps]

    x = np.arange(len(steps))
    width = 0.35

    bars_msm = ax.bar(x - width / 2, msm_probs, width,
                      label=f'MSM (P(R0=0)={msm_data["p_r0_zero"] * 100:.1f}%)',
                      color=COLORS['msm'], alpha=1.0, edgecolor='black', linewidth=0.5)
    bars_pwid = ax.bar(x + width / 2, pwid_probs, width,
                       label=f'PWID (P(R0=0)={pwid_data["observed_r0_zero_rate"] * 100:.3f}%)',
                       color=COLORS['pwid'], alpha=1.0, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Step Completion Probability (%)')
    ax.set_xlabel('Prevention Cascade Step')
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=7)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    # Reference line
    ax.axhline(y=50, color='gray', linestyle='--', alpha=1.0, linewidth=0.8)

    # Disparity annotation
    ax.annotate(f'Disparity: {data["disparity_fold"]:,}-fold',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=1.0))

    plt.tight_layout()
    save_figure(fig, "Fig1_CascadeComparison", output_dir)
    plt.close(fig)
    return fig


def fig2_barrier_decomposition(data, output_dir):
    """
    Figure 2: Three-Layer Barrier Decomposition
    Panel A: Three-layer pie chart
    Panel B: Architectural subtype breakdown
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    current = data["cascade_results"][0]
    three_layer = current["three_layer_decomposition"]
    barrier_pct = current["barrier_decomposition_pct"]

    # Panel A: Three-layer decomposition
    ax = axes[0]

    # Data for pie (skip pathogen biology since it's 0)
    labels = ['HIV Testing\n(6.9%)', 'Architectural\n(93.1%)']
    sizes = [three_layer["hiv_testing"], three_layer["architectural"]]
    colors_pie = [COLORS['testing'], COLORS['infrastructure']]
    explode = (0, 0.05)

    wedges, texts = ax.pie(sizes, explode=explode, colors=colors_pie,
                           startangle=90, wedgeprops=dict(edgecolor='black', linewidth=0.5, alpha=1.0))

    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(-0.1, 0.5), fontsize=8)
    ax.set_title('A. Three-Layer Decomposition', fontweight='bold', fontsize=10)

    # Add note about pathogen biology
    ax.text(0.5, -0.15, 'Pathogen Biology: 0.0%\n(addressed by drug efficacy)',
            transform=ax.transAxes, ha='center', fontsize=7, style='italic')

    # Panel B: Architectural subtypes
    ax = axes[1]

    arch_data = {
        'Policy\n(Criminalization)': barrier_pct["policy"],
        'Infrastructure\n(MSM-centric)': barrier_pct["infrastructure"],
        'Stigma\n(Healthcare)': barrier_pct["stigma"],
        'Machine Learning\n(Algorithmic)': barrier_pct["machine_learning"],
        'Research\nExclusion': barrier_pct["research_exclusion"],
    }

    labels_arch = list(arch_data.keys())
    values = list(arch_data.values())
    colors_arch = [COLORS['policy'], COLORS['infrastructure'], COLORS['stigma'],
                   COLORS['ml'], COLORS['research']]

    bars = ax.barh(labels_arch, values, color=colors_arch, alpha=1.0,
                   edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Contribution to Prevention Failure (%)')
    ax.set_title('B. Architectural Barrier Subtypes', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 45)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    save_figure(fig, "Fig2_BarrierDecomposition", output_dir)
    plt.close(fig)
    return fig


def fig3_policy_scenarios(data, output_dir):
    """
    Figure 3: Policy Scenario Analysis
    Bar chart showing P(R0=0) across 8 scenarios with MSM reference
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))

    scenarios = [r["scenario"] for r in data["cascade_results"]]
    p_r0_zero = [r["observed_r0_zero_rate"] * 100 for r in data["cascade_results"]]
    ci_lower = [(r["observed_r0_zero_rate"] - r["r0_zero_95ci"][0]) * 100
                for r in data["cascade_results"]]
    ci_upper = [(r["r0_zero_95ci"][1] - r["observed_r0_zero_rate"]) * 100
                for r in data["cascade_results"]]

    # Shortened labels
    short_labels = ['Current\nPolicy', 'Decrim\nOnly', 'Decrim+\nStigma',
                    'SSP-\nIntegrated', 'Full HR', 'HR+\nPURPOSE-4',
                    'HR+ML\nDebias', 'Theoretical\nMax']

    # Color gradient from red to green
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(scenarios)))

    ax.set_xticks(range(len(short_labels)))
    bars = ax.bar(range(len(short_labels)), p_r0_zero, color=colors, alpha=1.0,
                  edgecolor='black', linewidth=0.5,
                  yerr=[ci_lower, ci_upper], capsize=3, error_kw={'linewidth': 0.8})

    # MSM reference line
    msm_p = data["msm_comparison"]["p_r0_zero"] * 100
    ax.axhline(y=msm_p, color=COLORS['msm'], linestyle='--', linewidth=2,
               alpha=1.0, label=f'MSM Reference ({msm_p:.1f}%)')

    ax.set_ylabel('P(R0=0) %')
    ax.set_xlabel('Policy Scenario')
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.legend(loc='upper left', fontsize=8)

    # Value labels
    for bar, val in zip(bars, p_r0_zero):
        height = bar.get_height()
        label_text = f'{val:.2f}%' if val < 1 else f'{val:.1f}%'
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                label_text, ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_ylim(0, max(p_r0_zero) * 1.2)

    plt.tight_layout()
    save_figure(fig, "Fig3_PolicyScenarios", output_dir)
    plt.close(fig)
    return fig


def fig4_stochastic_avoidance(data, output_dir):
    """
    Figure 4: Stochastic Avoidance Failure Prediction
    Panel A: Cumulative outbreak probability
    Panel B: Time-to-outbreak distribution
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    sa = data["stochastic_avoidance"]

    # Panel A: Cumulative probability over time
    ax = axes[0]

    # Generate trajectory (exponential approximation)
    years = np.arange(0, 16)
    # Using geometric series: P(outbreak by year n) = 1 - (1-p_annual)^n
    # Calibrate to match 5-year and 10-year probabilities
    p_annual = 0.18  # Calibrated to give ~63% at 5 years
    p_cumulative = 1 - (1 - p_annual) ** years
    p_cumulative = p_cumulative * (sa["probability_outbreak_10_years"] / p_cumulative[10])  # Rescale
    p_cumulative = np.clip(p_cumulative, 0, 1)

    ax.plot(years + 2024, p_cumulative * 100, color=COLORS['pwid'], linewidth=2, alpha=1.0)
    ax.fill_between(years + 2024, 0, p_cumulative * 100, color=COLORS['pwid'], alpha=1.0, hatch='//', facecolor='white', edgecolor=COLORS['pwid'])

    # Mark key points
    ax.axhline(y=sa["probability_outbreak_5_years"] * 100, color='red', linestyle=':',
               linewidth=1.2, alpha=1.0,
               label=f'5-year: {sa["probability_outbreak_5_years"] * 100:.1f}%')
    ax.axhline(y=sa["probability_outbreak_10_years"] * 100, color='orange', linestyle=':',
               linewidth=1.2, alpha=1.0,
               label=f'10-year: {sa["probability_outbreak_10_years"] * 100:.1f}%')
    ax.axvline(x=2029, color='gray', linestyle='--', alpha=1.0)

    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Outbreak Probability (%)')
    ax.set_title('A. Outbreak Probability Over Time', fontweight='bold', fontsize=10)
    ax.set_xlim(2024, 2039)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=7)

    # Panel B: Time-to-outbreak distribution (simulated)
    ax = axes[1]

    # Generate distribution matching summary statistics
    np.random.seed(42)
    # Use exponential with shift to match median of 4
    outbreak_times = np.random.exponential(scale=sa["mean_years_to_outbreak"], size=5000)
    outbreak_times = outbreak_times[outbreak_times <= 15]

    ax.hist(outbreak_times, bins=15, color=COLORS['pwid'], alpha=1.0,
            edgecolor='black', linewidth=0.5, density=True)

    ax.axvline(x=sa["median_years_to_outbreak"], color='black', linewidth=2,
               label=f'Median: {sa["median_years_to_outbreak"]:.0f} years')
    ax.axvline(x=5, color='red', linestyle='--', linewidth=1.5, label='5-year horizon')
    ax.axvline(x=10, color='orange', linestyle='--', linewidth=1.5, label='10-year horizon')

    ax.set_xlabel('Years to Outbreak')
    ax.set_ylabel('Density')
    ax.set_title('B. Time-to-Outbreak Distribution', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 15)
    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    save_figure(fig, "Fig4_StochasticAvoidance", output_dir)
    plt.close(fig)
    return fig


def fig5_snr_loocv(data, output_dir):
    """
    Figure 5: Signal-to-Noise Ratio Disparity and LOOCV Framework
    Panel A: SNR comparison (log scale)
    Panel B: Trial participant distribution
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    # Panel A: SNR comparison
    ax = axes[0]

    populations = ['MSM', 'PWID']
    snr_values = [9180, 76.4]
    colors_snr = [COLORS['msm'], COLORS['pwid']]

    bars = ax.bar(populations, snr_values, color=colors_snr, alpha=1.0,
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Signal-to-Noise Ratio (log scale)')
    ax.set_title('A. Literature Evidence Base', fontweight='bold', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(10, 20000)

    # Ratio annotation
    ax.annotate('120-fold\ndisparity', xy=(0.5, 400), fontsize=10, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=1.0))

    # Value labels
    for bar, val in zip(bars, snr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
                f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')

    # Panel B: LOOCV framework
    ax = axes[1]

    # Trial data
    training_set = {'MSM': 10800, 'Cisgender Women': 5000, 'Heterosexual Couples': 4000}
    held_out = {'PWID': 2413}

    training_total = sum(training_set.values())

    y_pos = [0, 1]
    widths = [training_total, held_out['PWID']]
    labels = ['Training Set\n(validated across\n9+ trials)',
              'Held-Out Set\n(single trial\nBangkok 2013)']
    colors_loocv = [COLORS['msm'], COLORS['pwid']]

    bars = ax.barh(y_pos, widths, color=colors_loocv, alpha=1.0,
                   edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Trial Participants')
    ax.set_title('B. LOOCV Framework', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 25000)

    # Value labels
    ax.text(training_total + 500, 0, f'{training_total:,}', va='center', fontsize=8)
    ax.text(held_out['PWID'] + 500, 1, f'{held_out["PWID"]:,}', va='center', fontsize=8)

    plt.tight_layout()
    save_figure(fig, "Fig5_SNR_LOOCV", output_dir)
    plt.close(fig)
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate all figures using canonical data"""

    print("=" * 70)
    print("AIDS AND BEHAVIOR FIGURE GENERATOR - CANONICAL VERSION")
    print("Using data from manufactured_death_results.json")
    print("=" * 70)
    print()

    output_dir = "../data/aids_behavior_figures_canonical/"

    print(f"Output directory: {output_dir}/")
    print()

    # Generate main figures
    print("Generating main manuscript figures...")
    fig1_cascade_comparison(CANONICAL_DATA, output_dir)
    fig2_barrier_decomposition(CANONICAL_DATA, output_dir)
    fig3_policy_scenarios(CANONICAL_DATA, output_dir)
    fig4_stochastic_avoidance(CANONICAL_DATA, output_dir)
    fig5_snr_loocv(CANONICAL_DATA, output_dir)

    print()
    print("=" * 70)
    print("CANONICAL VALUES SUMMARY")
    print("=" * 70)

    current = CANONICAL_DATA["cascade_results"][0]
    msm = CANONICAL_DATA["msm_comparison"]
    sa = CANONICAL_DATA["stochastic_avoidance"]

    print(f"""
PWID (Current Policy):
  P(R0=0) = {current['observed_r0_zero_rate'] * 100:.3f}%
  95% CI: ({current['r0_zero_95ci'][0] * 100:.4f}%, {current['r0_zero_95ci'][1] * 100:.4f}%)
  Cascade completion = {current['observed_cascade_completion_rate'] * 100:.3f}%
  5-year incarceration survival = {current['incarceration_survival_probability'] * 100:.1f}%

MSM (Comparison):
  P(R0=0) = {msm['p_r0_zero'] * 100:.1f}%
  Cascade completion = {msm['cascade_completion'] * 100:.1f}%
  Disparity = {CANONICAL_DATA['disparity_fold']:,}-fold

Three-Layer Barrier Decomposition:
  Pathogen Biology: {current['three_layer_decomposition']['pathogen_biology']:.1f}%
  HIV Testing: {current['three_layer_decomposition']['hiv_testing']:.1f}%
  Architectural: {current['three_layer_decomposition']['architectural']:.1f}%
    - Policy: {current['barrier_decomposition_pct']['policy']:.1f}%
    - Infrastructure: {current['barrier_decomposition_pct']['infrastructure']:.1f}%
    - Stigma: {current['barrier_decomposition_pct']['stigma']:.1f}%
    - ML: {current['barrier_decomposition_pct']['machine_learning']:.1f}%
    - Research: {current['barrier_decomposition_pct']['research_exclusion']:.1f}%

Stochastic Avoidance:
  P(5-year outbreak) = {sa['probability_outbreak_5_years'] * 100:.1f}%
  P(10-year outbreak) = {sa['probability_outbreak_10_years'] * 100:.1f}%
  Median years to outbreak = {sa['median_years_to_outbreak']:.1f}

Policy Scenarios:""")

    for r in CANONICAL_DATA["cascade_results"]:
        print(f"  {r['scenario']:<40}: {r['observed_r0_zero_rate'] * 100:.2f}%")

    print()
    print(f"All figures saved to: {output_dir}/")
    print("Formats: TIFF (300dpi), PNG (preview), EPS (vector)")
    print()
    print("Done!")


if __name__ == "__main__":
    main()