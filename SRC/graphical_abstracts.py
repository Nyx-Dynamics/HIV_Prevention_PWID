#!/usr/bin/env python3
"""
Graphical Abstracts for Preprints.org Submissions
- Prevention Theorem
- PWID HIV Prevention
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arrow, Circle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


def create_prevention_theorem_abstract():
    """
    Graphical abstract for Prevention Theorem manuscript.
    Shows time-dependent PEP efficacy and the irreversible integration threshold.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('The Prevention Theorem: Time-Dependent Constraints on HIV PEP',
                 fontsize=14, fontweight='bold', y=1.02)

    # Panel A: Prevention condition
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('A. Prevention Condition', fontweight='bold', fontsize=12)

    # Main equation box
    eq_box = FancyBboxPatch((1, 4), 8, 3, boxstyle="round,pad=0.1",
                            facecolor='#E8F4FD', edgecolor='#1976D2', linewidth=2)
    ax1.add_patch(eq_box)
    ax1.text(5, 5.5, r'$R_0(e) = 0$', fontsize=24, ha='center', va='center',
             fontweight='bold', color='#1976D2')
    ax1.text(5, 2, 'Zero probability of\nproductive infection',
             fontsize=11, ha='center', va='center', style='italic')
    ax1.text(5, 8.5, 'TRUE PREVENTION', fontsize=12, ha='center',
             fontweight='bold', color='#2E7D32')

    # Panel B: Time-dependent efficacy
    ax2 = axes[1]
    t = np.linspace(0, 96, 100)

    # Mucosal exposure (72h window)
    E_mucosal = np.where(t < 72, 0.99 * np.exp(-0.02 * t), 0.99 * np.exp(-0.02 * 72) * np.exp(-0.5 * (t - 72)))

    # Parenteral exposure (24h window)
    E_parenteral = np.where(t < 24, 0.99 * np.exp(-0.08 * t), 0.99 * np.exp(-0.08 * 24) * np.exp(-0.8 * (t - 24)))

    ax2.fill_between(t, E_mucosal, alpha=0.3, color='#4CAF50', label='Mucosal window')
    ax2.fill_between(t, E_parenteral, alpha=0.3, color='#F44336', label='Parenteral window')
    ax2.plot(t, E_mucosal, color='#2E7D32', linewidth=2.5, label='Mucosal (72h)')
    ax2.plot(t, E_parenteral, color='#C62828', linewidth=2.5, label='Parenteral (12-24h)')

    ax2.axvline(x=72, color='#2E7D32', linestyle='--', alpha=0.7)
    ax2.axvline(x=24, color='#C62828', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Hours Post-Exposure', fontsize=11)
    ax2.set_ylabel('PEP Efficacy', fontsize=11)
    ax2.set_title('B. Time-Dependent Efficacy', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 96)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.text(48, 0.05, 'Prevention impossible', fontsize=9, ha='center',
             style='italic', color='gray')

    # Panel C: Phase transition
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('C. Irreversible Transition', fontweight='bold', fontsize=12)

    # Pre-integration state
    pre_box = FancyBboxPatch((0.5, 6), 3.5, 2.5, boxstyle="round,pad=0.1",
                             facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
    ax3.add_patch(pre_box)
    ax3.text(2.25, 7.25, 'Pre-Integration', fontsize=10, ha='center', fontweight='bold')
    ax3.text(2.25, 6.5, r'$R_0(e)=0$ possible', fontsize=9, ha='center', color='#2E7D32')

    # Arrow
    ax3.annotate('', xy=(6, 7.25), xytext=(4.2, 7.25),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=3))
    ax3.text(5.1, 8, 'Proviral\nIntegration', fontsize=9, ha='center', color='#E65100')

    # Post-integration state
    post_box = FancyBboxPatch((6, 6), 3.5, 2.5, boxstyle="round,pad=0.1",
                              facecolor='#FFEBEE', edgecolor='#F44336', linewidth=2)
    ax3.add_patch(post_box)
    ax3.text(7.75, 7.25, 'Post-Integration', fontsize=10, ha='center', fontweight='bold')
    ax3.text(7.75, 6.5, r'$R_0(e)>0$ fixed', fontsize=9, ha='center', color='#C62828')

    # Key message
    msg_box = FancyBboxPatch((1.5, 1), 7, 2.5, boxstyle="round,pad=0.1",
                             facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2)
    ax3.add_patch(msg_box)
    ax3.text(5, 2.25, 'Parenteral exposures compress\nthe prevention window to 12-24h',
             fontsize=10, ha='center', va='center', fontweight='bold', color='#E65100')

    plt.tight_layout()
    plt.savefig('/Users/acdmbpmax/Desktop/preprints/graphical_abstract_prevention_theorem.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/acdmbpmax/Desktop/preprints/graphical_abstract_prevention_theorem.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Prevention Theorem graphical abstract saved.")


def create_pwid_abstract():
    """
    Graphical abstract for PWID HIV Prevention manuscript.
    Shows cascade failure, barrier decomposition, and outbreak prediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Structural Barriers and Outbreak Risk in HIV Prevention for PWID',
                 fontsize=14, fontweight='bold', y=1.02)

    # Panel A: Cascade comparison
    ax1 = axes[0]
    categories = ['MSM', 'PWID']
    values = [16.3, 0.003]
    colors = ['#4CAF50', '#F44336']

    bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel(r'P($R_0$=0) %', fontsize=11)
    ax1.set_title('A. Prevention Disparity', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 20)

    # Add value labels
    ax1.text(0, 17, '16.3%', ha='center', fontsize=12, fontweight='bold')
    ax1.text(1, 1.5, '0.003%', ha='center', fontsize=12, fontweight='bold', color='white')

    # Disparity annotation
    ax1.annotate('', xy=(1, 16.3), xytext=(0, 16.3),
                arrowprops=dict(arrowstyle='<->', color='#1976D2', lw=2))
    ax1.text(0.5, 18.5, '5,434-fold\ndisparity', ha='center', fontsize=10,
             fontweight='bold', color='#1976D2')

    # Panel B: Barrier decomposition (pie chart)
    ax2 = axes[1]
    barrier_labels = ['Policy/\nCriminalization\n38.4%', 'Infrastructure\n21.9%',
                      'Stigma\n20.6%', 'ML/Algorithm\n8.2%',
                      'Research\nExclusion\n4.1%', 'HIV Testing\n6.9%']
    barrier_sizes = [38.4, 21.9, 20.6, 8.2, 4.1, 6.9]
    barrier_colors = ['#D32F2F', '#F57C00', '#FBC02D', '#7B1FA2', '#512DA8', '#1976D2']

    wedges, texts = ax2.pie(barrier_sizes, colors=barrier_colors,
                            startangle=90, counterclock=False,
                            wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2))

    # Add labels manually for better positioning
    ax2.text(0, 0, 'Architectural\nFailures\n93.1%', ha='center', va='center',
             fontsize=11, fontweight='bold')
    ax2.set_title('B. Barrier Decomposition', fontweight='bold', fontsize=12)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=c, label=l.replace('\n', ' '))
                       for c, l in zip(barrier_colors,
                                      ['Policy 38.4%', 'Infrastructure 21.9%',
                                       'Stigma 20.6%', 'ML/Algorithm 8.2%',
                                       'Research 4.1%', 'HIV Testing 6.9%'])]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize=8)

    # Panel C: Outbreak prediction
    ax3 = axes[2]
    years = np.arange(0, 11)
    outbreak_prob = 1 - np.exp(-0.25 * years)  # Simplified cumulative hazard
    outbreak_prob = [0, 0.22, 0.40, 0.55, 0.65, 0.738, 0.80, 0.85, 0.89, 0.91, 0.927]

    ax3.fill_between(years, outbreak_prob, alpha=0.3, color='#F44336')
    ax3.plot(years, outbreak_prob, color='#C62828', linewidth=3, marker='o', markersize=6)

    ax3.axhline(y=0.738, color='#C62828', linestyle='--', alpha=0.7)
    ax3.axvline(x=5, color='gray', linestyle=':', alpha=0.7)

    ax3.text(5.2, 0.75, '73.8% at 5 years', fontsize=10, fontweight='bold', color='#C62828')
    ax3.text(7, 0.927, '92.7%', fontsize=10, fontweight='bold', color='#C62828')

    ax3.set_xlabel('Years', fontsize=11)
    ax3.set_ylabel('Cumulative Outbreak Probability', fontsize=11)
    ax3.set_title('C. Outbreak Prediction', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 1)

    # Key message box
    props = dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#F44336', alpha=0.9)
    ax3.text(5, 0.15, 'Predictable system\nfailure, not randomness',
             fontsize=9, ha='center', va='center', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/Users/acdmbpmax/Desktop/preprints/graphical_abstract_pwid.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/acdmbpmax/Desktop/preprints/graphical_abstract_pwid.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("PWID graphical abstract saved.")


if __name__ == "__main__":
    create_prevention_theorem_abstract()
    create_pwid_abstract()
    print("\nGraphical abstracts saved to /Users/acdmbpmax/Desktop/preprints/")
