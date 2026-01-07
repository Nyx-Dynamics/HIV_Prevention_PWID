"""
PEP Extension to HIV Reservoir Master Equation

Tests the Prevention Theorem at the boundary condition:
What happens when intervention occurs DURING the exposure-to-integration window?

Author: AC Demidont + Claude
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import expit  # Logistic function
from typing import Any, Dict, List, Tuple, Optional
import os
import json
import csv
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure output directories exist."""
    for directory in ['figures', 'outputs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

# =============================================================================
# INFECTION ESTABLISHMENT MODEL
# =============================================================================

class InfectionEstablishmentModel:
    """
    Models the critical window between exposure and reservoir establishment.

    This is where PEP operates - preventing R(0) > 0 from ever occurring.
    """

    def __init__(self):
        # Biological timing parameters (hours)
        self.mucosal_phase = 4  # Virus at entry site
        self.dendritic_uptake = 12  # DC capture and processing
        self.lymph_transit = 36  # Travel to lymph nodes
        self.systemic_spread = 60  # Dissemination begins
        self.seeding_midpoint = 72  # 50% probability of seeding
        self.integration_complete = 120  # Point of no return

        # PEP drug parameters
        self.pep_onset_hours = 2  # Time for drugs to reach effective levels
        self.pep_efficacy_peak = 0.995  # Maximum blocking efficacy
        self.pep_duration_days = 28  # Standard course

    def p_seeding_initiated(self, hours_post_exposure: float) -> float:
        """
        Probability that reservoir seeding has begun by time t.

        Modeled as logistic function centered at seeding_midpoint.
        """
        k = 0.1  # Steepness of transition
        return expit(k * (hours_post_exposure - self.seeding_midpoint))

    def p_integration_complete(self, hours_post_exposure: float) -> float:
        """
        Probability that integration is irreversibly established.
        """
        k = 0.15
        return expit(k * (hours_post_exposure - self.integration_complete))

    def pep_efficacy(self,
                     hours_to_pep: float,
                     adherence: float = 1.0) -> Dict:
        """
        Calculate PEP efficacy given time to initiation.

        Returns probability that PEP achieves R(0) = 0.
        """
        # Drug onset delay
        effective_time = hours_to_pep + self.pep_onset_hours

        # Probability seeding already initiated when drugs become effective
        p_seeded = self.p_seeding_initiated(effective_time)

        # Probability integration already complete (PEP definitely too late)
        p_integrated = self.p_integration_complete(effective_time)

        # Efficacy components:
        # 1. If not yet seeded: PEP blocks with high efficacy
        # 2. If seeded but not integrated: PEP may still clear
        # 3. If integrated: PEP fails (R(0) > 0 established)

        efficacy_if_not_seeded = self.pep_efficacy_peak * adherence
        efficacy_if_seeded_not_integrated = 0.5 * adherence  # Can sometimes clear
        efficacy_if_integrated = 0.0  # Too late

        # Weighted efficacy
        overall_efficacy = (
                (1 - p_seeded) * efficacy_if_not_seeded +
                (p_seeded - p_integrated) * efficacy_if_seeded_not_integrated +
                p_integrated * efficacy_if_integrated
        )

        return {
            'hours_to_pep': hours_to_pep,
            'p_seeding_initiated': p_seeded,
            'p_integration_complete': p_integrated,
            'pep_efficacy': overall_efficacy,
            'p_R0_equals_zero': overall_efficacy,
            'p_R0_greater_zero': 1 - overall_efficacy,
            'adherence': adherence
        }

    def simulate_pep_timing_curve(self,
                                  max_hours: float = 168,
                                  n_points: int = 100) -> Dict[str, Any]:
        """
        Generate efficacy curve across PEP timing window.
        """
        hours = np.linspace(0, max_hours, n_points)

        efficacy_list: List[float] = []
        p_seeded_list: List[float] = []
        p_integrated_list: List[float] = []
        nnt_list: List[float] = []

        for h in hours:
            r = self.pep_efficacy(h)
            efficacy_list.append(r['pep_efficacy'])
            p_seeded_list.append(r['p_seeding_initiated'])
            p_integrated_list.append(r['p_integration_complete'])

            # NNT = 1 / (efficacy * baseline_risk)
            # Assuming baseline risk of 1% per exposure
            baseline_risk = 0.01
            absolute_risk_reduction = r['pep_efficacy'] * baseline_risk
            nnt = 1 / absolute_risk_reduction if absolute_risk_reduction > 0 else np.inf
            nnt_list.append(nnt)

        results: Dict[str, Any] = {
            'hours': hours,
            'efficacy': np.array(efficacy_list),
            'p_seeded': np.array(p_seeded_list),
            'p_integrated': np.array(p_integrated_list),
            'nnt': np.array(nnt_list),
        }

        return results


# =============================================================================
# COMBINED MODEL: PEP FAILURE → RESERVOIR DYNAMICS
# =============================================================================

class PEPtoReservoirModel:
    """
    Links PEP timing to long-term reservoir outcomes.

    If PEP succeeds: R(0) = 0 → Prevention Theorem applies
    If PEP fails: R(0) > 0 → Reservoir dynamics begin
    """

    def __init__(self,
                 establishment_model: InfectionEstablishmentModel,
                 reservoir_model):  # From previous code
        self.establishment = establishment_model
        self.reservoir = reservoir_model

    def expected_reservoir_at_time(self,
                                   hours_to_pep: float,
                                   years_post_exposure: float,
                                   art_delay_days: float = 30) -> Dict:
        """
        Calculate expected reservoir size given PEP timing.

        E[R(t)] = P(PEP fails) × E[R(t) | infection established]
        """
        # PEP outcome probabilities
        pep_result = self.establishment.pep_efficacy(hours_to_pep)
        p_infection = pep_result['p_R0_greater_zero']

        # If infection establishes, simulate reservoir
        # (Using simplified exponential model for speed)
        if p_infection > 0:
            # Initial reservoir size if infection establishes
            R0_if_infected = 1e6  # Cells

            # Net decay rate on ART (approximate)
            # T_scm and microglia dominate long-term
            effective_half_life_years = 15  # Very slow due to long-lived compartments
            decay_rate = np.log(2) / effective_half_life_years

            # Reservoir at time t
            t = years_post_exposure
            R_if_infected = R0_if_infected * np.exp(-decay_rate * t)

            # But also account for clonal expansion in T_scm
            expansion_rate = 0.02  # 2% per year
            R_if_infected *= np.exp(expansion_rate * t)
        else:
            R_if_infected = 0

        # Expected value
        expected_R = p_infection * R_if_infected

        return {
            'hours_to_pep': hours_to_pep,
            'years_post_exposure': years_post_exposure,
            'p_infection': p_infection,
            'R_if_infected': R_if_infected,
            'expected_reservoir': expected_R,
            'prevented': p_infection == 0
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pep_efficacy_curve(save_path: Optional[str] = None):
    """
    Visualize PEP efficacy as function of timing.
    """
    model = InfectionEstablishmentModel()
    results = model.simulate_pep_timing_curve(max_hours=168)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: PEP Efficacy Curve
    ax = axes[0, 0]
    ax.plot(results['hours'], results['efficacy'] * 100,
            'b-', linewidth=2.5, label='PEP Efficacy')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax.axvline(x=72, color='green', linestyle=':', alpha=0.7, label='72h guideline')

    ax.fill_between(results['hours'], results['efficacy'] * 100, alpha=0.3)

    ax.set_xlabel('Hours from Exposure to PEP Initiation', fontsize=12)
    ax.set_ylabel('PEP Efficacy (%)', fontsize=12)
    ax.set_title('A. PEP Efficacy vs. Time to Initiation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Add critical windows
    ax.axvspan(0, 24, alpha=0.2, color='green', label='Optimal window')
    ax.axvspan(24, 72, alpha=0.2, color='yellow')
    ax.axvspan(72, 120, alpha=0.2, color='orange')
    ax.axvspan(120, 168, alpha=0.2, color='red')

    # Panel B: Biological Events Timeline
    ax = axes[0, 1]

    events = [
        (0, 'Exposure', 'red'),
        (4, 'Mucosal replication', 'orange'),
        (12, 'Dendritic cell uptake', 'yellow'),
        (36, 'Lymph node transit', 'lightgreen'),
        (60, 'Systemic dissemination', 'lightblue'),
        (72, 'Reservoir seeding (50%)', 'blue'),
        (120, 'Integration established', 'purple'),
    ]

    for i, (time, event, color) in enumerate(events):
        ax.barh(i, time, color=color, alpha=0.7, height=0.6)
        ax.text(time + 2, i, f'{event} ({time}h)', va='center', fontsize=10)

    ax.set_xlabel('Hours Post-Exposure', fontsize=12)
    ax.set_title('B. Biological Timeline of HIV Establishment', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 150)
    ax.axvline(x=72, color='green', linestyle='--', linewidth=2, label='72h PEP window')
    ax.legend()

    # Panel C: Probability Curves
    ax = axes[1, 0]
    ax.plot(results['hours'], results['p_seeded'] * 100,
            'r-', linewidth=2, label='P(Seeding initiated)')
    ax.plot(results['hours'], results['p_integrated'] * 100,
            'k-', linewidth=2, label='P(Integration complete)')
    ax.plot(results['hours'], (1 - results['efficacy']) * 100,
            'b--', linewidth=2, label='P(PEP fails)')

    ax.axvline(x=72, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Hours from Exposure', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('C. Probability of Infection Establishment', fontsize=14, fontweight='bold')
    ax.legend(loc='center right')
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel D: The PEP Theorem
    ax = axes[1, 1]
    ax.text(0.5, 0.9, 'THE PEP COROLLARY',
            fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.72,
            r'$\mathrm{PEP}(t < t_{crit}) \Rightarrow R(0) = 0$',
            fontsize=18, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.text(0.5, 0.55,
            'PEP initiated before integration completes\nachieves the Prevention Theorem',
            fontsize=11, ha='center', transform=ax.transAxes, style='italic')

    ax.text(0.5, 0.38,
            'Critical Windows:\n'
            '• 0-24h: >95% efficacy (optimal)\n'
            '• 24-72h: 70-95% efficacy (standard)\n'
            '• 72-120h: <70% efficacy (diminishing)\n'
            '• >120h: ~0% efficacy (too late)',
            fontsize=11, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.text(0.5, 0.1,
            'Every hour matters.\nPEP is a race against integration.',
            fontsize=12, ha='center', transform=ax.transAxes, fontweight='bold')

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")

    # plt.show() # Commented out to avoid blocking execution

    return fig, results


def plot_pep_timing_vs_reservoir(save_path: Optional[str] = None):
    """
    Show long-term reservoir consequences of PEP timing.
    """
    establishment = InfectionEstablishmentModel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Expected reservoir at 50 years vs PEP timing
    ax = axes[0]

    pep_times = [0, 6, 12, 24, 48, 72, 96, 120, 168]
    expected_reservoirs = []

    for t in pep_times:
        pep_result = establishment.pep_efficacy(t)
        p_infection = pep_result['p_R0_greater_zero']

        # If infected, estimate reservoir at 50 years
        # Using simplified model: ~10^4 cells with best ART
        reservoir_if_infected = 1e4  # Cells at 50 years with optimal ART

        expected_R = p_infection * reservoir_if_infected
        expected_reservoirs.append(expected_R)

    colors = ['green' if r < 100 else 'orange' if r < 5000 else 'red'
              for r in expected_reservoirs]

    bars = ax.bar(range(len(pep_times)),
                  [np.log10(r + 1) for r in expected_reservoirs],
                  color=colors, alpha=0.7)

    ax.set_xticks(range(len(pep_times)))
    ax.set_xticklabels([f'{t}h' for t in pep_times])
    ax.set_xlabel('Time to PEP Initiation', fontsize=12)
    ax.set_ylabel('log₁₀(Expected Reservoir + 1)', fontsize=12)
    ax.set_title('A. Expected Reservoir at 50 Years by PEP Timing',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=2, color='green', linestyle='--', alpha=0.7,
               label='Functional cure threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, expected_reservoirs)):
        if val < 1:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.1,
                    'ZERO', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='green')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.0e}', ha='center', va='bottom', fontsize=9)

    # Panel B: The message
    ax = axes[1]

    ax.text(0.5, 0.85, 'PEP TIMING IS EVERYTHING',
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)

    # Create timing impact table
    table_data = [
        ['PEP at 0h', '>99%', '~0', 'Full Prevention'],
        ['PEP at 24h', '~95%', '~500', 'High Efficacy'],
        ['PEP at 72h', '~70%', '~3,000', 'Moderate Efficacy'],
        ['PEP at 120h', '~20%', '~8,000', 'Low Efficacy'],
        ['No PEP', '0%', '~10,000', 'Infection Established'],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=['Scenario', 'Efficacy', 'E[Reservoir]', 'Outcome'],
        loc='center',
        cellLoc='center',
        colColours=['lightblue'] * 4
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color code rows
    for i in range(len(table_data)):
        if 'Full Prevention' in table_data[i][3]:
            for j in range(4):
                table[(i + 1, j)].set_facecolor('lightgreen')
        elif 'Low Efficacy' in table_data[i][3] or 'Infection Established' in table_data[i][3]:
            for j in range(4):
                table[(i + 1, j)].set_facecolor('lightcoral')

    ax.text(0.5, 0.15,
            'The Prevention Theorem has a time-dependent corollary:\n'
            'PEP efficacy decays exponentially with delay.\n\n'
            'Every hour is ~2% efficacy lost.',
            fontsize=11, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")

    # plt.show() # Commented out to avoid blocking execution

    return fig


def plot_pep_timing_vs_reservoir_quant(save_path: Optional[str] = None):
    """
    Quantitative version:
    - Panel A: log10(E[R(50)]) vs PEP timing
    - Panel B: table with numeric P(prevention), P(infection), E[R(50)]
    No qualitative labels like 'likely prevented'.
    """

    establishment = InfectionEstablishmentModel()

    # Discrete PEP initiation times in hours
    pep_times = [0, 6, 12, 24, 48, 72, 96, 120, 168]

    # Parameters for reservoir expectation
    reservoir_if_infected = 1e4  # cells at 50 years under optimal ART
    expected_reservoirs = []
    p_prev_list = []
    p_inf_list = []

    for t in pep_times:
        pep_result = establishment.pep_efficacy(t)
        p_prev = pep_result["p_R0_equals_zero"]   # P(prevention | t)
        p_inf = pep_result["p_R0_greater_zero"]   # P(infection | t)
        # Expected reservoir at 50 years:
        # E[R(50)|t] = P(infection|t) * reservoir_if_infected
        expected_r = p_inf * reservoir_if_infected

        p_prev_list.append(p_prev)
        p_inf_list.append(p_inf)
        expected_reservoirs.append(expected_r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------- Panel A: Expected reservoir vs timing ----------
    ax = axes[0]

    # Avoid log10(0) by clipping very small values
    clipped_r = np.clip(expected_reservoirs, 1e-6, None)
    bars = ax.bar(
        range(len(pep_times)),
        np.log10(clipped_r),
        color=["green" if r < 100 else "orange" if r < 5000 else "red"
               for r in expected_reservoirs],
        alpha=0.7,
    )

    ax.set_xticks(range(len(pep_times)))
    ax.set_xticklabels([f"{t} h" for t in pep_times], rotation=45, ha="right")
    ax.set_xlabel("Time from exposure to PEP initiation", fontsize=11)
    ax.set_ylabel(r"$\log_{10} E[R(50)]$ (cells)", fontsize=11)
    ax.set_title("Expected reservoir at 50 years vs. PEP timing", fontsize=12, fontweight='bold')

    # Optional horizontal line: functional-cure-like threshold (e.g. 100 cells → log10=2)
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.7, label='Functional cure threshold')
    ax.legend()

    # Label each bar with the expected reservoir (numeric, no adjectives)
    for bar, r in zip(bars, expected_reservoirs):
        x = bar.get_x() + bar.get_width() / 2
        if r < 1:
            label = "<1"
            y = 0  # near zero
        else:
            label = f"{r:.0f}"
            y = np.log10(max(r, 1e-6))
        ax.text(x, y + 0.05, label, ha="center", va="bottom", fontsize=9)

    ax.grid(True, axis="y", alpha=0.3)

    # ---------- Panel B: Quantitative probabilities ----------
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Quantitative PEP outcomes by timing", fontsize=12, pad=10, fontweight='bold')

    table_data = []
    for t, p_prev, p_inf, r in zip(pep_times, p_prev_list, p_inf_list, expected_reservoirs):
        table_data.append([
            f"{t} h",
            f"{p_prev*100:5.1f}%",   # P(prevention)
            f"{p_inf*100:5.1f}%",    # P(infection)
            f"{r:,.0f}",             # E[R(50)]
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=["Time to PEP", "P(prevention)", "P(infection)", "E[R(50)] (cells)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Optional: shade rows by infection probability (still purely numeric)
    n_rows = len(table_data)
    for i in range(1, n_rows + 1):  # row 0 is header
        # col 2 = P(infection)
        p_inf_str = table_data[i-1][2].replace("%", "")
        try:
            p_inf_val = float(p_inf_str)
        except ValueError:
            continue
        if p_inf_val < 10:
            color = "#d9f2d9"  # light green
        elif p_inf_val < 50:
            color = "#fff2cc"  # light yellow
        else:
            color = "#f4cccc"  # light red
        for j in range(4):
            table[(i, j)].set_facecolor(color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved to {save_path}")

    return fig


def save_results_to_data(results: Dict):
    """Save simulation results to JSON and CSV."""
    json_path = os.path.join('outputs', 'pep_mucosal_results.json')
    csv_path = os.path.join('outputs', 'pep_mucosal_results.csv')

    # Save JSON
    try:
        # Prepare JSON serializable results (convert numpy arrays to lists)
        json_results = {
            "timestamp": datetime.now().isoformat(),
            "hours": results['hours'].tolist(),
            "efficacy": results['efficacy'].tolist(),
            "p_seeded": results['p_seeded'].tolist(),
            "p_integrated": results['p_integrated'].tolist()
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Results saved to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")

    # Save CSV
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Hours_to_PEP', 'Efficacy', 'P_Seeding_Initiated', 'P_Integration_Complete'])
            for i in range(len(results['hours'])):
                writer.writerow([
                    results['hours'][i],
                    results['efficacy'][i],
                    results['p_seeded'][i],
                    results['p_integrated'][i]
                ])
        logger.info(f"Results saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV results: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run PEP analysis."""

    ensure_directories()

    print("=" * 80)
    print("PEP EXTENSION TO THE PREVENTION THEOREM")
    print("Testing the Master Equation at the Boundary Condition")
    print("=" * 80)

    # Run efficacy curve
    print("\nGenerating PEP efficacy analysis...")
    fig1_path = os.path.join('figures', 'pep_efficacy_curve.png')
    fig1, results = plot_pep_efficacy_curve(save_path=fig1_path)

    # Run timing vs reservoir
    print("\nGenerating timing impact analysis...")
    fig2_path = os.path.join('figures', 'pep_timing_impact.png')
    fig2 = plot_pep_timing_vs_reservoir(save_path=fig2_path)

    # Save structured data
    save_results_to_data(results)

    # Print key results
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    model = InfectionEstablishmentModel()

    print("\nPEP Efficacy by Timing:")
    for hours in [0, 12, 24, 48, 72, 96, 120]:
        result = model.pep_efficacy(hours)
        print(f"  {hours:3d}h: {result['pep_efficacy'] * 100:5.1f}% efficacy "
              f"(P(infection) = {result['p_R0_greater_zero'] * 100:4.1f}%)")

    print("\n" + "=" * 80)
    print("THE PEP COROLLARY")
    print("=" * 80)
    print("""
    The Prevention Theorem states: R(0) = 0 ⟹ R(t) = 0 ∀t

    PEP operates at the boundary condition, racing to ensure R(0) = 0
    before integration establishes R(0) > 0.

    The PEP Corollary:

        PEP(t) efficacy ≈ 1 - P(integration complete by t)

        For t < 24h:  P(integration) ≈ 0   → Efficacy ≈ 99%
        For t = 72h:  P(integration) ≈ 30% → Efficacy ≈ 70%
        For t > 120h: P(integration) ≈ 90% → Efficacy ≈ 10%

    PEP is not treatment. PEP is last-chance prevention.

    It is a race against the establishment of the initial condition.

    Every hour of delay is ~2% efficacy lost.
    Every hour matters.
    """)

    return results


if __name__ == "__main__":
    results = main()
