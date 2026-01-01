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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300

# Lancet dimensions (mm converted to inches)
SINGLE_COL = 75 / 25.4  # ~2.95 inches
DOUBLE_COL = 154 / 25.4  # ~6.06 inches

# Color palette (colorblind-safe)
COLORS = {
    'pwid': '#B22222',      # Dark red - exclusion
    'msm': '#2E8B57',       # Sea green - inclusion  
    'policy': '#CD5C5C',    # Indian red
    'stigma': '#DAA520',    # Goldenrod
    'infrastructure': '#4682B4',  # Steel blue
    'research': '#9370DB',  # Medium purple
    'ml': '#708090',        # Slate gray
    'testing': '#FF6347',   # Tomato
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
    Shows the fundamental disparity in cascade completion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4))
    
    # Extract data
    pwid_current = results['cascade_results'][0]
    msm_data = results['msm_comparison']
    
    # Step names
    steps = ['Aware', 'Willing', 'Access', 'Disclose', 
             'Provider', 'Testing', '1st Inj', 'Sustained']
    
    # PWID probabilities
    pwid_probs = [pwid_current['step_probabilities'][s] 
                  for s in pwid_current['step_probabilities']]
    
    # MSM probabilities
    msm_probs = list(msm_data['cascade_steps'].values())
    
    # Panel A: Step probabilities comparison
    ax = axes[0]
    x = np.arange(len(steps))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, msm_probs, width, label='MSM', 
                   color=COLORS['msm'], alpha=0.8)
    bars2 = ax.bar(x + width/2, pwid_probs, width, label='PWID',
                   color=COLORS['pwid'], alpha=0.8)
    
    ax.set_ylabel('Step Probability')
    ax.set_xlabel('Cascade Step')
    ax.set_title('A. Step-wise Cascade Probabilities', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Panel B: Cumulative cascade
    ax = axes[1]
    
    # Calculate cumulative products
    msm_cumulative = np.cumprod(msm_probs)
    pwid_cumulative = np.cumprod(pwid_probs)
    
    ax.semilogy(x, msm_cumulative, 'o-', color=COLORS['msm'], 
                linewidth=2, markersize=6, label='MSM')
    ax.semilogy(x, pwid_cumulative, 's-', color=COLORS['pwid'],
                linewidth=2, markersize=6, label='PWID')
    
    ax.set_ylabel('Cumulative P(cascade completion)')
    ax.set_xlabel('Cascade Step')
    ax.set_title('B. Cumulative Attrition (log scale)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(1e-6, 1)
    
    # Add final values
    ax.annotate(f'MSM: {msm_cumulative[-1]*100:.1f}%',
                xy=(7, msm_cumulative[-1]), xytext=(5.5, 0.5),
                fontsize=9, color=COLORS['msm'], fontweight='bold')
    ax.annotate(f'PWID: {pwid_cumulative[-1]*100:.4f}%',
                xy=(7, pwid_cumulative[-1]), xytext=(5.5, 1e-4),
                fontsize=9, color=COLORS['pwid'], fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig2_barrier_decomposition(results, save_path=None):
    """
    Figure 2: Three-Layer Barrier Decomposition
    Shows contribution of each barrier type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4))
    
    pwid_current = results['cascade_results'][0]
    decomp = pwid_current['barrier_decomposition_pct']
    
    # Panel A: Three-layer pie chart
    ax = axes[0]
    
    three_layer = [
        decomp.get('pathogen_biology', 0),
        decomp.get('hiv_testing', 0),
        sum([decomp.get('policy', 0), decomp.get('stigma', 0),
             decomp.get('infrastructure', 0), decomp.get('research_exclusion', 0),
             decomp.get('machine_learning', 0)])
    ]
    
    labels = ['Layer 1:\nPathogen Biology', 'Layer 2:\nHIV Testing', 
              'Layer 3:\nArchitectural']
    colors_3layer = ['#FFA07A', '#FF6347', '#8B0000']
    explode = (0, 0.02, 0.05)
    
    wedges, texts, autotexts = ax.pie(
        three_layer, labels=labels, colors=colors_3layer,
        autopct='%1.1f%%', startangle=90, explode=explode,
        pctdistance=0.6
    )
    ax.set_title('A. Three-Layer Barrier Decomposition', fontweight='bold', pad=20)
    
    # Panel B: Architectural subtypes breakdown
    ax = axes[1]
    
    arch_subtypes = {
        'Policy\n(Criminalization)': decomp.get('policy', 0),
        'Stigma\n(Healthcare bias)': decomp.get('stigma', 0),
        'Infrastructure\n(MSM-centric)': decomp.get('infrastructure', 0),
        'Research\nExclusion': decomp.get('research_exclusion', 0),
        'Machine\nLearning': decomp.get('machine_learning', 0),
    }
    
    labels_arch = list(arch_subtypes.keys())
    values_arch = list(arch_subtypes.values())
    colors_arch = [COLORS['policy'], COLORS['stigma'], COLORS['infrastructure'],
                   COLORS['research'], COLORS['ml']]
    
    bars = ax.barh(labels_arch, values_arch, color=colors_arch, alpha=0.8)
    ax.set_xlabel('% of Total Barrier Attribution')
    ax.set_title('B. Architectural Failure Subtypes', fontweight='bold')
    ax.set_xlim(0, 50)
    
    # Add percentage labels
    for bar, val in zip(bars, values_arch):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig3_policy_scenarios(results, save_path=None):
    """
    Figure 3: Policy Scenario Comparison
    Shows P(R(0)=0) under different policy interventions.
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5))
    
    scenarios = results['cascade_results']
    
    names = [s['scenario'] for s in scenarios]
    probs = [s['observed_r0_zero_rate'] * 100 for s in scenarios]
    
    # Add MSM for comparison
    names.append('MSM\n(comparison)')
    probs.append(results['msm_comparison']['p_r0_zero'] * 100)
    
    # Color bars
    colors = [COLORS['pwid']] * len(scenarios) + [COLORS['msm']]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, probs, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('P(R(0)=0) - Probability of Sustained Protection (%)')
    ax.set_title('LAI-PrEP Prevention Probability by Policy Scenario\n'
                 '(Drug efficacy = 99·9% in all scenarios)', fontweight='bold')
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        if prob < 0.01:
            label = f'{prob:.4f}%'
        elif prob < 1:
            label = f'{prob:.2f}%'
        else:
            label = f'{prob:.1f}%'
        ax.text(max(prob, 0.5) + 0.5, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)
    
    # Add annotation for key insight
    ax.axvline(x=results['msm_comparison']['p_r0_zero'] * 100, 
               color=COLORS['msm'], linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig4_stochastic_avoidance(results, save_path=None):
    """
    Figure 4: Stochastic Avoidance Failure Prediction
    Shows probability of outbreak over time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4))
    
    sa_data = results['stochastic_avoidance']
    
    # Panel A: Key probabilities
    ax = axes[0]
    
    metrics = {
        'P(outbreak\nwithin 5 years)': sa_data.get('probability_outbreak_5_years', 0) * 100,
        'P(outbreak\nwithin 10 years)': sa_data.get('probability_outbreak_10_years', 0) * 100,
        'P(no outbreak\n20 years)': sa_data.get('probability_no_outbreak', 0) * 100,
    }
    
    colors_probs = ['#FF6347', '#CD5C5C', '#2E8B57']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors_probs, alpha=0.8)
    
    ax.set_ylabel('Probability (%)')
    ax.set_title('A. Stochastic Avoidance Failure Risk', fontweight='bold')
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Panel B: Time distribution
    ax = axes[1]
    
    # Create histogram-like representation
    percentiles = {
        '10th': sa_data.get('p10_years', 0),
        '25th': sa_data.get('p25_years', 0),
        'Median': sa_data.get('median_years_to_outbreak', 0),
        '75th': sa_data.get('p75_years', 0),
        '90th': sa_data.get('p90_years', 0),
    }
    
    ax.barh(list(percentiles.keys()), list(percentiles.values()), 
            color='#CD5C5C', alpha=0.8)
    ax.set_xlabel('Years from 2024')
    ax.set_title('B. Time to Major Outbreak\n(Percentile Distribution)', fontweight='bold')
    ax.axvline(x=sa_data.get('median_years_to_outbreak', 0), 
               color='darkred', linestyle='--', linewidth=2, label='Median')
    
    # Add annotation
    median = sa_data.get('median_years_to_outbreak', 0)
    if median:
        ax.annotate(f'Median: {median:.1f} years',
                   xy=(median, 2), xytext=(median + 3, 3),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='darkred'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def fig5_snr_disparity(results, save_path=None):
    """
    Figure 5: Signal-to-Noise Ratio Disparity in Training Data
    Visualizes the LOOCV framework.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4))
    
    # Panel A: SNR Comparison
    ax = axes[0]
    
    populations = ['MSM\n(In training set)', 'PWID\n(Held-out)', 'PWID LAI-PrEP\n(No data)']
    snr_values = [9180, 76.4, 0.1]  # Using 0.1 to show on log scale
    colors_snr = [COLORS['msm'], COLORS['pwid'], '#8B0000']
    
    bars = ax.bar(populations, snr_values, color=colors_snr, alpha=0.8)
    ax.set_ylabel('Signal-to-Noise Ratio (log scale)')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 20000)
    ax.set_title('A. Training Data Quality Disparity', fontweight='bold')
    
    # Add values
    for bar, val, pop in zip(bars, snr_values, populations):
        if 'No data' in pop:
            ax.text(bar.get_x() + bar.get_width()/2, 0.15, 
                    'ZERO', ha='center', fontsize=10, fontweight='bold', color='darkred')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                    f'{val:,.0f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add ratio annotation
    ax.annotate('120-fold\ndisparity', xy=(0.5, 1000), xytext=(0.5, 3000),
                ha='center', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='<->', color='gray'))
    
    # Panel B: LOOCV Framework visualization
    ax = axes[1]
    
    # Training set composition
    trial_data = {
        'MSM trials (n=9+)': 9,
        'Women trials (n=3)': 3,
        'PWID trials (n=1)': 1,
        'PWID LAI-PrEP (n=0)': 0,
    }
    
    colors_trials = [COLORS['msm'], '#9370DB', COLORS['pwid'], '#8B0000']
    
    # Create stacked bar representation
    bars = ax.barh(list(trial_data.keys()), list(trial_data.values()),
                   color=colors_trials, alpha=0.8)
    ax.set_xlabel('Number of Major Trials')
    ax.set_title('B. Leave-One-Out Structure\n(PWID = Held-out Test Set)', fontweight='bold')
    ax.set_xlim(0, 12)
    
    # Highlight the exclusion
    ax.axhline(y=2.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(6, 2.7, 'TRAINING SET', fontsize=9, color='gray')
    ax.text(6, 1.3, 'HELD OUT', fontsize=9, color='darkred', fontweight='bold')
    
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
