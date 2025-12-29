#!/usr/bin/env python3
"""
Manufactured Death Model - Visualization Script
Generates publication-quality figures for Lancet HIV submission

Author: AC Demidont, MD / Nyx Dynamics LLC
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import json

# Set publication-quality defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8  # Lancet uses 8-12pt
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

# Lancet dimensions (mm converted to inches)
SINGLE_COL = 75 / 25.4  # 2.95 inches
DOUBLE_COL = 154 / 25.4  # 6.06 inches
MAX_HEIGHT = 230 / 25.4 # 9.05 inches

# Color palette (Lancet-friendly, high contrast)
COLORS = {
    'pwid': '#d73027',      # Red
    'msm': '#4575b4',       # Blue
    'policy': '#fc8d59',    # Orange
    'stigma': '#fee090',    # Yellow
    'infrastructure': '#91bfdb',  # Light blue
    'research': '#e0f3f8',  # Very light blue
    'ml': '#999999',        # Gray
    'testing': '#4daf4a',   # Green
}

import os


# ... existing code ...

def load_results(filepath=None):
    """Load model results from JSON."""
    if filepath is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for the results file in the same directory
        filepath = os.path.join(script_dir, 'manufactured_death_results.json')

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Results file not found at: {filepath}\n"
            f"Please ensure 'manufactured_death_results.json' exists in the project directory.\n"
            f"You may need to run the simulation script first to generate this file."
        )

    with open(filepath, 'r') as f:
        return json.load(f)

def fig1_cascade_comparison(results, save_path=None):
    """
    Figure 1: PWID vs MSM Cascade Comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    
    pwid_current = results['cascade_results'][0]
    msm_data = results['msm_comparison']
    
    steps = ['Aware', 'Willing', 'Access', 'Disclose', 
             'Provider', 'Testing', '1st Inj', 'Sustained']
    
    pwid_probs = [pwid_current['step_probabilities'][s] 
                  for s in pwid_current['step_probabilities']]
    msm_probs = list(msm_data['cascade_steps'].values())
    
    # Panel A: Step probabilities comparison
    ax = axes[0]
    x = np.arange(len(steps))
    width = 0.35
    
    ax.bar(x - width/2, msm_probs, width, label='MSM', color=COLORS['msm'])
    ax.bar(x + width/2, pwid_probs, width, label='PWID', color=COLORS['pwid'])
    
    ax.set_ylabel('Probability')
    ax.set_title('A', loc='left', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel B: Cumulative cascade
    ax = axes[1]
    msm_cumulative = np.cumprod(msm_probs)
    pwid_cumulative = np.cumprod(pwid_probs)
    
    ax.semilogy(x, msm_cumulative, 'o-', color=COLORS['msm'], 
                linewidth=1.5, markersize=4, label='MSM')
    ax.semilogy(x, pwid_cumulative, 's-', color=COLORS['pwid'],
                linewidth=1.5, markersize=4, label='PWID')
    
    ax.set_ylabel('Cumulative P(success) [log scale]')
    ax.set_title('B', loc='left', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.set_ylim(1e-7, 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate end points
    ax.text(7, msm_cumulative[-1]*1.5, f'{msm_cumulative[-1]*100:.1f}%', 
            color=COLORS['msm'], ha='right', fontweight='bold')
    ax.text(7, pwid_cumulative[-1]*0.5, f'{pwid_cumulative[-1]*100:.4f}%', 
            color=COLORS['pwid'], ha='right', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig2_barrier_decomposition(results, save_path=None):
    """
    Figure 2: Three-Layer Barrier Decomposition
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    
    pwid_current = results['cascade_results'][0]
    decomp = pwid_current['barrier_decomposition_pct']
    
    # Panel A: Three-layer bar chart (Replacing Pie chart for publication)
    ax = axes[0]
    
    three_layer = {
        'Layer 1:\nBiology': decomp.get('pathogen_biology', 0),
        'Layer 2:\nTesting': decomp.get('hiv_testing', 0),
        'Layer 3:\nArchitectural': sum([decomp.get('policy', 0), decomp.get('stigma', 0),
                                      decomp.get('infrastructure', 0), decomp.get('research_exclusion', 0),
                                      decomp.get('machine_learning', 0)])
    }
    
    colors_3layer = [COLORS['testing'], '#f1a340', '#998ec3']
    bars = ax.bar(three_layer.keys(), three_layer.values(), color=colors_3layer)
    ax.set_ylabel('% Contribution to Failure')
    ax.set_title('A', loc='left', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # Panel B: Architectural subtypes breakdown
    ax = axes[1]
    
    arch_subtypes = {
        'Policy': decomp.get('policy', 0),
        'Stigma': decomp.get('stigma', 0),
        'Infrastructure': decomp.get('infrastructure', 0),
        'Research': decomp.get('research_exclusion', 0),
        'ML Bias': decomp.get('machine_learning', 0),
    }
    
    # Sort for better visualization
    arch_subtypes = dict(sorted(arch_subtypes.items(), key=lambda item: item[1]))
    
    labels_arch = list(arch_subtypes.keys())
    values_arch = list(arch_subtypes.values())
    colors_arch = [COLORS['policy'], COLORS['stigma'], COLORS['infrastructure'],
                   COLORS['research'], COLORS['ml']]
    
    bars = ax.barh(labels_arch, values_arch, color=colors_arch)
    ax.set_xlabel('% Attribution')
    ax.set_title('B', loc='left', fontweight='bold', fontsize=12)
    ax.set_xlim(0, max(values_arch) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig3_policy_scenarios(results, save_path=None):
    """
    Figure 3: Policy Scenario Comparison
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))
    
    scenarios = results['cascade_results']
    
    names = [s['scenario'] for s in scenarios]
    probs = [s['observed_r0_zero_rate'] * 100 for s in scenarios]
    
    # Add MSM for comparison
    names.append('MSM (Benchmark)')
    probs.append(results['msm_comparison']['p_r0_zero'] * 100)
    
    # Sort by probability
    sorted_indices = np.argsort(probs)
    names = [names[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]
    
    colors = [COLORS['pwid'] if 'MSM' not in n else COLORS['msm'] for n in names]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, probs, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('P(R0=0) - Probability of Sustained Protection (%)')
    ax.set_title('Prevention Probability by Policy Scenario', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        label = f'{prob:.4f}%' if prob < 0.1 else f'{prob:.2f}%'
        ax.text(prob + 0.2, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig4_stochastic_avoidance(results, save_path=None):
    """
    Figure 4: Stochastic Avoidance Failure Prediction
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    
    sa_data = results['stochastic_avoidance']
    
    # Panel A: Key probabilities
    ax = axes[0]
    
    metrics = {
        '5 years': sa_data.get('probability_outbreak_5_years', 0) * 100,
        '10 years': sa_data.get('probability_outbreak_10_years', 0) * 100,
        'No outbreak\n(20 yr)': sa_data.get('probability_no_outbreak', 0) * 100,
    }
    
    colors_probs = [COLORS['pwid'], COLORS['policy'], COLORS['msm']]
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors_probs)
    
    ax.set_ylabel('Probability (%)')
    ax.set_title('A', loc='left', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Panel B: Time distribution
    ax = axes[1]
    
    percentiles = {
        '10th': sa_data.get('p10_years', 0),
        '25th': sa_data.get('p25_years', 0),
        'Median': sa_data.get('median_years_to_outbreak', 0),
        '75th': sa_data.get('p75_years', 0),
        '90th': sa_data.get('p90_years', 0),
    }
    
    ax.barh(list(percentiles.keys()), list(percentiles.values()), color=COLORS['policy'])
    ax.set_xlabel('Years from 2024')
    ax.set_title('B', loc='left', fontweight='bold', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    median = sa_data.get('median_years_to_outbreak', 0)
    if median:
        ax.axvline(x=median, color='black', linestyle='--', linewidth=1)
        ax.text(median + 0.5, 2, f'Median: {median:.1f} yr', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig5_snr_disparity(results, save_path=None):
    """
    Figure 5: Signal-to-Noise Ratio Disparity
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    
    # Panel A: SNR Comparison
    ax = axes[0]
    
    populations = ['MSM\n(Training)', 'PWID\n(Held-out)', 'PWID LAI\n(Zero Data)']
    snr_values = [9180, 76.4, 0.1]
    colors_snr = [COLORS['msm'], COLORS['pwid'], '#000000']
    
    ax.bar(populations, snr_values, color=colors_snr)
    ax.set_ylabel('Signal-to-Noise Ratio (log10)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 50000)
    ax.set_title('A', loc='left', fontweight='bold', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, val in enumerate(snr_values):
        label = 'ZERO' if val == 0.1 else f'{val:,.1f}'
        ax.text(i, val * 1.2, label, ha='center', fontsize=8, fontweight='bold')
    
    # Panel B: LOOCV Structure
    ax = axes[1]
    trial_data = {
        'MSM trials': 9,
        'Women trials': 3,
        'PWID (Oral)': 1,
        'PWID (LAI)': 0,
    }
    
    colors_trials = [COLORS['msm'], '#4daf4a', COLORS['pwid'], '#000000']
    bars = ax.barh(list(trial_data.keys()), list(trial_data.values()), color=colors_trials)
    
    ax.set_xlabel('Number of Major Trials')
    ax.set_title('B', loc='left', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=0.8)
    ax.text(6, 2.7, 'TRAINING SET', fontsize=7, color='gray')
    ax.text(6, 1.3, 'HELD OUT (Zero SNR)', fontsize=7, color='black', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_figures(output_dir=None):
    """Generate all figures for manuscript."""

    if output_dir is None:
        # Use a relative path instead of hardcoded absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'outputs')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results()

    print("Generating figures...")

    fig1_cascade_comparison(results, f'{output_dir}/Fig1_CascadeComparison.png')
    fig2_barrier_decomposition(results, f'{output_dir}/Fig2_BarrierDecomposition.png')
    fig3_policy_scenarios(results, f'{output_dir}/Fig3_PolicyScenarios.png')
    fig4_stochastic_avoidance(results, f'{output_dir}/Fig4_StochasticAvoidance.png')
    fig5_snr_disparity(results, f'{output_dir}/Fig5_SNR_LOOCV.png')

    print("\nAll figures generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    generate_all_figures()
