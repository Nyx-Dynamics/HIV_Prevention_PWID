#!/usr/bin/env python3
"""
PWID LAI-PrEP Cascade Simulation: Counterfactual Policy Analysis

Models the probability of achieving R(0)=0 (sustained HIV prevention) for
people who inject drugs under varying policy scenarios, demonstrating that
even perfect pharmacology (99% efficacy Q6M injectable) cannot overcome
structural barriers encoded in current policy.

Author: AC Demidont, MD / Nyx Dynamics LLC
Date: December 2024
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Set seed for reproducibility
random.seed(42)

@dataclass
class CascadeStep:
    """Represents a single step in the PrEP cascade"""
    name: str
    description: str
    base_probability: float
    criminalization_penalty: float  # Reduction due to criminalization
    bias_penalty: float  # Reduction due to healthcare bias
    structural_penalty: float  # Reduction due to structural barriers

@dataclass 
class PolicyScenario:
    """Represents a policy intervention scenario"""
    name: str
    description: str
    criminalization_removed: bool
    bias_reduced: float  # 0-1, proportion of bias penalty removed
    structural_barriers_reduced: float  # 0-1, proportion removed
    incarceration_rate_modifier: float  # Multiplier on base incarceration rate
    in_custody_prep_available: bool
    ssp_integrated_delivery: bool
    peer_navigation: bool

# Define cascade steps with barrier decomposition
CASCADE_STEPS = [
    CascadeStep(
        name="awareness",
        description="Aware that PrEP exists and is available for PWID",
        base_probability=0.70,
        criminalization_penalty=0.30,  # Hidden population, no outreach
        bias_penalty=0.10,  # Messaging excludes PWID
        structural_penalty=0.0
    ),
    CascadeStep(
        name="willingness",
        description="Willing to seek PrEP despite visibility risks",
        base_probability=0.80,
        criminalization_penalty=0.35,  # Seeking care = system visibility
        bias_penalty=0.05,
        structural_penalty=0.0
    ),
    CascadeStep(
        name="healthcare_access",
        description="Can physically access healthcare services",
        base_probability=0.75,
        criminalization_penalty=0.10,  # Warrants, legal concerns
        bias_penalty=0.05,
        structural_penalty=0.25  # Transport, hours, no PCP
    ),
    CascadeStep(
        name="disclosure",
        description="Willing to disclose IDU to provider",
        base_probability=0.70,
        criminalization_penalty=0.30,  # Admitting felony behavior
        bias_penalty=0.10,  # Fear of judgment
        structural_penalty=0.0
    ),
    CascadeStep(
        name="provider_willing",
        description="Provider willing to prescribe for PWID",
        base_probability=0.85,
        criminalization_penalty=0.05,
        bias_penalty=0.25,  # "Not my patient population"
        structural_penalty=0.0
    ),
    CascadeStep(
        name="affordability",
        description="Can afford/access medication",
        base_probability=0.80,
        criminalization_penalty=0.15,  # No stable address for PAP
        bias_penalty=0.05,
        structural_penalty=0.15  # Insurance gaps, ID requirements
    ),
    CascadeStep(
        name="first_injection",
        description="Returns for and receives first injection",
        base_probability=0.75,
        criminalization_penalty=0.10,
        bias_penalty=0.05,
        structural_penalty=0.15  # Appointment systems, wait times
    ),
    CascadeStep(
        name="sustained_engagement",
        description="Maintains Q6M schedule over time",
        base_probability=0.70,
        criminalization_penalty=0.20,  # Incarceration interrupts
        bias_penalty=0.10,
        structural_penalty=0.10
    )
]

# Define policy scenarios
POLICY_SCENARIOS = [
    PolicyScenario(
        name="Current Policy",
        description="Status quo: full criminalization, systemic bias, no harm reduction integration",
        criminalization_removed=False,
        bias_reduced=0.0,
        structural_barriers_reduced=0.0,
        incarceration_rate_modifier=1.0,
        in_custody_prep_available=False,
        ssp_integrated_delivery=False,
        peer_navigation=False
    ),
    PolicyScenario(
        name="Decriminalization Only",
        description="Drug use decriminalized, but healthcare system unchanged",
        criminalization_removed=True,
        bias_reduced=0.0,
        structural_barriers_reduced=0.0,
        incarceration_rate_modifier=0.3,  # Residual incarceration for other offenses
        in_custody_prep_available=False,
        ssp_integrated_delivery=False,
        peer_navigation=False
    ),
    PolicyScenario(
        name="Decrim + Bias Training",
        description="Decriminalization plus healthcare bias reduction initiatives",
        criminalization_removed=True,
        bias_reduced=0.5,
        structural_barriers_reduced=0.0,
        incarceration_rate_modifier=0.3,
        in_custody_prep_available=False,
        ssp_integrated_delivery=False,
        peer_navigation=False
    ),
    PolicyScenario(
        name="Decrim + Low-Barrier Access",
        description="Decriminalization plus structural barrier reduction (walk-in, mobile, extended hours)",
        criminalization_removed=True,
        bias_reduced=0.3,
        structural_barriers_reduced=0.6,
        incarceration_rate_modifier=0.3,
        in_custody_prep_available=False,
        ssp_integrated_delivery=False,
        peer_navigation=False
    ),
    PolicyScenario(
        name="SSP-Integrated Delivery",
        description="PrEP delivered through syringe service programs with peer support",
        criminalization_removed=True,
        bias_reduced=0.7,
        structural_barriers_reduced=0.8,
        incarceration_rate_modifier=0.3,
        in_custody_prep_available=False,
        ssp_integrated_delivery=True,
        peer_navigation=True
    ),
    PolicyScenario(
        name="Full Harm Reduction Infrastructure",
        description="Complete policy transformation: decrim, SSP integration, in-custody continuation, peer navigation",
        criminalization_removed=True,
        bias_reduced=0.8,
        structural_barriers_reduced=0.9,
        incarceration_rate_modifier=0.2,
        in_custody_prep_available=True,
        ssp_integrated_delivery=True,
        peer_navigation=True
    ),
    PolicyScenario(
        name="Theoretical Maximum",
        description="All barriers removed (theoretical ceiling for comparison)",
        criminalization_removed=True,
        bias_reduced=1.0,
        structural_barriers_reduced=1.0,
        incarceration_rate_modifier=0.0,
        in_custody_prep_available=True,
        ssp_integrated_delivery=True,
        peer_navigation=True
    )
]

def calculate_step_probability(step: CascadeStep, scenario: PolicyScenario) -> float:
    """Calculate probability of passing a cascade step under a given policy scenario"""
    
    # Start with base probability
    prob = step.base_probability
    
    # Apply criminalization penalty (removed if decriminalized)
    if not scenario.criminalization_removed:
        prob -= step.criminalization_penalty
    
    # Apply bias penalty (reduced by intervention)
    bias_impact = step.bias_penalty * (1 - scenario.bias_reduced)
    prob -= bias_impact
    
    # Apply structural penalty (reduced by intervention)
    structural_impact = step.structural_penalty * (1 - scenario.structural_barriers_reduced)
    prob -= structural_impact
    
    # SSP integration bonus for relevant steps
    if scenario.ssp_integrated_delivery:
        if step.name in ["awareness", "healthcare_access", "disclosure", "first_injection"]:
            prob += 0.15  # SSP provides trusted entry point
    
    # Peer navigation bonus
    if scenario.peer_navigation:
        if step.name in ["willingness", "first_injection", "sustained_engagement"]:
            prob += 0.10
    
    # Bound probability
    return max(0.01, min(0.99, prob))

def calculate_incarceration_disruption(
    scenario: PolicyScenario,
    years: int = 5,
    base_annual_incarceration_rate: float = 0.30
) -> float:
    """
    Calculate probability of maintaining protection given incarceration risk.
    
    Without in-custody PrEP, any incarceration resets protection to zero.
    """
    annual_rate = base_annual_incarceration_rate * scenario.incarceration_rate_modifier
    
    if scenario.in_custody_prep_available:
        # Incarceration doesn't disrupt protection
        return 1.0
    else:
        # Probability of NO incarceration over the period
        prob_no_incarceration = (1 - annual_rate) ** years
        return prob_no_incarceration

def simulate_individual(scenario: PolicyScenario, n_years: int = 5) -> Dict:
    """Simulate a single PWID's journey through the cascade"""
    
    results = {
        "completed_cascade": True,
        "failed_step": None,
        "incarceration_disrupted": False,
        "achieved_r0_zero": False,
        "step_outcomes": {}
    }
    
    # Progress through cascade
    for step in CASCADE_STEPS:
        prob = calculate_step_probability(step, scenario)
        passed = random.random() < prob
        results["step_outcomes"][step.name] = {
            "probability": prob,
            "passed": passed
        }
        
        if not passed:
            results["completed_cascade"] = False
            results["failed_step"] = step.name
            break
    
    # If completed cascade, check for incarceration disruption
    if results["completed_cascade"]:
        incarceration_survival = calculate_incarceration_disruption(scenario, n_years)
        survived = random.random() < incarceration_survival
        
        if not survived:
            results["incarceration_disrupted"] = True
            results["achieved_r0_zero"] = False
        else:
            results["achieved_r0_zero"] = True
    
    return results

def run_simulation(
    scenario: PolicyScenario,
    n_individuals: int = 100000,
    n_years: int = 5
) -> Dict:
    """Run full simulation for a policy scenario"""
    
    results = {
        "scenario": scenario.name,
        "n_individuals": n_individuals,
        "n_years": n_years,
        "achieved_r0_zero": 0,
        "completed_cascade": 0,
        "incarceration_disrupted": 0,
        "step_failure_counts": {step.name: 0 for step in CASCADE_STEPS},
        "step_probabilities": {}
    }
    
    # Calculate theoretical step probabilities
    for step in CASCADE_STEPS:
        results["step_probabilities"][step.name] = calculate_step_probability(step, scenario)
    
    # Calculate theoretical cascade probability (product of steps)
    theoretical_cascade_prob = 1.0
    for step in CASCADE_STEPS:
        theoretical_cascade_prob *= results["step_probabilities"][step.name]
    results["theoretical_cascade_probability"] = theoretical_cascade_prob
    
    # Calculate theoretical incarceration survival
    results["incarceration_survival_probability"] = calculate_incarceration_disruption(scenario, n_years)
    
    # Theoretical R(0)=0 probability
    results["theoretical_r0_zero_probability"] = (
        theoretical_cascade_prob * results["incarceration_survival_probability"]
    )
    
    # Run individual simulations
    for _ in range(n_individuals):
        individual = simulate_individual(scenario, n_years)
        
        if individual["achieved_r0_zero"]:
            results["achieved_r0_zero"] += 1
        
        if individual["completed_cascade"]:
            results["completed_cascade"] += 1
            
        if individual["incarceration_disrupted"]:
            results["incarceration_disrupted"] += 1
            
        if individual["failed_step"]:
            results["step_failure_counts"][individual["failed_step"]] += 1
    
    # Calculate observed rates
    results["observed_r0_zero_rate"] = results["achieved_r0_zero"] / n_individuals
    results["observed_cascade_completion_rate"] = results["completed_cascade"] / n_individuals
    
    # Calculate 95% CI for observed rate
    p = results["observed_r0_zero_rate"]
    n = n_individuals
    se = (p * (1-p) / n) ** 0.5
    results["r0_zero_95ci"] = (max(0, p - 1.96*se), min(1, p + 1.96*se))
    
    return results

def calculate_population_impact(
    results: Dict,
    pwid_population: int = 3_500_000,  # US PWID population estimate
    annual_hiv_incidence_no_prep: float = 0.02  # 2% annual incidence without PrEP
) -> Dict:
    """Calculate population-level impact of a policy scenario"""
    
    p_protected = results["observed_r0_zero_rate"]
    prep_efficacy = 0.99  # Assuming PURPOSE-4 level efficacy
    
    # Number achieving sustained protection
    n_protected = pwid_population * p_protected
    
    # Infections prevented among protected (assuming 99% efficacy)
    # vs baseline incidence
    annual_infections_baseline = pwid_population * annual_hiv_incidence_no_prep
    annual_infections_with_policy = (
        (pwid_population - n_protected) * annual_hiv_incidence_no_prep +
        n_protected * annual_hiv_incidence_no_prep * (1 - prep_efficacy)
    )
    
    infections_prevented = annual_infections_baseline - annual_infections_with_policy
    
    # 5-year cumulative
    five_year_infections_prevented = infections_prevented * 5
    
    # Lifetime treatment cost averted ($500K per infection)
    cost_averted = infections_prevented * 500_000
    
    return {
        "n_protected": int(n_protected),
        "percent_protected": p_protected * 100,
        "annual_infections_baseline": int(annual_infections_baseline),
        "annual_infections_with_policy": int(annual_infections_with_policy),
        "annual_infections_prevented": int(infections_prevented),
        "five_year_infections_prevented": int(five_year_infections_prevented),
        "annual_cost_averted_millions": cost_averted / 1_000_000,
        "five_year_cost_averted_billions": (cost_averted * 5) / 1_000_000_000
    }

def format_results_table(all_results: List[Dict]) -> str:
    """Format results as a text table"""
    
    lines = []
    lines.append("=" * 120)
    lines.append("PWID LAI-PrEP CASCADE SIMULATION: POLICY SCENARIO COMPARISON")
    lines.append("Assuming: 99% efficacy Q6M injectable (PURPOSE-4 success), 5-year time horizon")
    lines.append("=" * 120)
    lines.append("")
    
    # Summary table
    lines.append(f"{'Scenario':<40} {'P(R(0)=0)':<12} {'95% CI':<20} {'Cascade':<10} {'Incarc Surv':<12}")
    lines.append("-" * 100)
    
    for r in all_results:
        ci_str = f"({r['r0_zero_95ci'][0]:.4f}, {r['r0_zero_95ci'][1]:.4f})"
        lines.append(
            f"{r['scenario']:<40} "
            f"{r['observed_r0_zero_rate']:.4f}       "
            f"{ci_str:<20} "
            f"{r['observed_cascade_completion_rate']:.4f}     "
            f"{r['incarceration_survival_probability']:.4f}"
        )
    
    lines.append("")
    lines.append("=" * 120)
    lines.append("STEP-BY-STEP PROBABILITIES BY SCENARIO")
    lines.append("=" * 120)
    
    # Step probabilities
    step_names = [s.name for s in CASCADE_STEPS]
    header = f"{'Scenario':<35}" + "".join([f"{s[:8]:<10}" for s in step_names])
    lines.append(header)
    lines.append("-" * 115)
    
    for r in all_results:
        row = f"{r['scenario']:<35}"
        for step in step_names:
            row += f"{r['step_probabilities'][step]:.3f}     "
        lines.append(row)
    
    lines.append("")
    lines.append("=" * 120)
    lines.append("POPULATION IMPACT (US PWID Population: 3.5 million)")
    lines.append("=" * 120)
    
    lines.append(f"{'Scenario':<40} {'Protected':<15} {'Annual Prev':<15} {'5-Yr Prev':<15} {'5-Yr Cost Saved':<15}")
    lines.append("-" * 100)
    
    for r in all_results:
        impact = calculate_population_impact(r)
        lines.append(
            f"{r['scenario']:<40} "
            f"{impact['n_protected']:>12,}   "
            f"{impact['annual_infections_prevented']:>12,}   "
            f"{impact['five_year_infections_prevented']:>12,}   "
            f"${impact['five_year_cost_averted_billions']:>10.2f}B"
        )
    
    lines.append("")
    lines.append("=" * 120)
    lines.append("KEY FINDING: THE PHARMACOLOGY-POLICY INEQUALITY")
    lines.append("=" * 120)
    
    current = all_results[0]
    full_hr = all_results[5]
    theoretical = all_results[6]
    
    lines.append("")
    lines.append(f"Drug efficacy (PURPOSE-4 assumption):     99.0%")
    lines.append(f"P(R(0)=0 | Current Policy):               {current['observed_r0_zero_rate']*100:.2f}%")
    lines.append(f"P(R(0)=0 | Full Harm Reduction):          {full_hr['observed_r0_zero_rate']*100:.2f}%")
    lines.append(f"P(R(0)=0 | Theoretical Maximum):          {theoretical['observed_r0_zero_rate']*100:.2f}%")
    lines.append("")
    lines.append(f"Policy gap (Current vs Full HR):          {(full_hr['observed_r0_zero_rate'] - current['observed_r0_zero_rate'])*100:.2f} percentage points")
    lines.append(f"Policy gap (Current vs Theoretical):      {(theoretical['observed_r0_zero_rate'] - current['observed_r0_zero_rate'])*100:.2f} percentage points")
    lines.append("")
    
    current_impact = calculate_population_impact(current)
    full_hr_impact = calculate_population_impact(full_hr)
    
    lines.append(f"Annual infections under current policy:   {current_impact['annual_infections_with_policy']:,}")
    lines.append(f"Annual infections under full HR:          {full_hr_impact['annual_infections_with_policy']:,}")
    lines.append(f"Annual preventable infections (policy gap): {current_impact['annual_infections_with_policy'] - full_hr_impact['annual_infections_with_policy']:,}")
    lines.append("")
    lines.append("CONCLUSION: Even with 99% effective Q6M injectable PrEP,")
    lines.append(f"current policy achieves R(0)=0 for only {current['observed_r0_zero_rate']*100:.2f}% of PWID.")
    lines.append("The closed-form solution is POLICY-LOCKED, not pharmacology-locked.")
    lines.append("")
    
    return "\n".join(lines)

def generate_detailed_report(all_results: List[Dict]) -> str:
    """Generate detailed analytical report"""
    
    report = []
    report.append("DETAILED ANALYSIS: CASCADE BARRIER DECOMPOSITION")
    report.append("=" * 80)
    report.append("")
    
    current = all_results[0]
    
    report.append("CURRENT POLICY: WHERE DO PWID FALL OUT OF THE CASCADE?")
    report.append("-" * 60)
    report.append("")
    report.append("Cumulative probability at each step:")
    report.append("")
    
    cumulative = 1.0
    for step in CASCADE_STEPS:
        step_prob = current['step_probabilities'][step.name]
        cumulative *= step_prob
        attrition = (1 - step_prob) * 100
        report.append(f"  {step.name:<25}: P={step_prob:.3f}  (Cumulative: {cumulative:.4f}, Attrition: {attrition:.1f}%)")
    
    report.append("")
    report.append(f"  Cascade completion:        {current['theoretical_cascade_probability']:.4f}")
    report.append(f"  × Incarceration survival:  {current['incarceration_survival_probability']:.4f}")
    report.append(f"  = P(R(0)=0):               {current['theoretical_r0_zero_probability']:.4f}")
    report.append("")
    
    report.append("BARRIER ATTRIBUTION ANALYSIS")
    report.append("-" * 60)
    report.append("")
    report.append("Total probability loss by barrier type (under current policy):")
    report.append("")
    
    crim_loss = sum(s.criminalization_penalty for s in CASCADE_STEPS)
    bias_loss = sum(s.bias_penalty for s in CASCADE_STEPS)
    struct_loss = sum(s.structural_penalty for s in CASCADE_STEPS)
    
    report.append(f"  Criminalization penalties:  -{crim_loss:.2f} (cumulative across steps)")
    report.append(f"  Healthcare bias penalties:  -{bias_loss:.2f}")
    report.append(f"  Structural barriers:        -{struct_loss:.2f}")
    report.append("")
    
    report.append("INCREMENTAL POLICY ANALYSIS")
    report.append("-" * 60)
    report.append("")
    
    for i in range(len(all_results) - 1):
        current_scenario = all_results[i]
        next_scenario = all_results[i + 1]
        
        delta = next_scenario['observed_r0_zero_rate'] - current_scenario['observed_r0_zero_rate']
        relative = (delta / current_scenario['observed_r0_zero_rate'] * 100) if current_scenario['observed_r0_zero_rate'] > 0 else float('inf')
        
        report.append(f"  {current_scenario['scenario']:<35} → {next_scenario['scenario']}")
        report.append(f"    Δ P(R(0)=0): +{delta:.4f} ({relative:+.1f}% relative improvement)")
        report.append("")
    
    return "\n".join(report)

def main():
    """Run all simulations and generate output"""
    
    print("Running PWID LAI-PrEP Cascade Simulation...")
    print(f"Simulating {100000:,} individuals per scenario, 5-year horizon")
    print("")
    
    all_results = []
    
    for scenario in POLICY_SCENARIOS:
        print(f"  Simulating: {scenario.name}...")
        results = run_simulation(scenario, n_individuals=100000, n_years=5)
        all_results.append(results)
    
    print("")
    print("Generating reports...")
    
    # Generate main results table
    results_table = format_results_table(all_results)
    print(results_table)
    
    # Generate detailed report
    detailed_report = generate_detailed_report(all_results)
    print("")
    print(detailed_report)
    
    # Save results to JSON for further analysis
    json_results = []
    for r in all_results:
        # Convert tuples to lists for JSON serialization
        r_copy = r.copy()
        r_copy['r0_zero_95ci'] = list(r['r0_zero_95ci'])
        r_copy['impact'] = calculate_population_impact(r)
        json_results.append(r_copy)
    
    with open('/home/claude/pwid_simulation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("")
    print("Results saved to pwid_simulation_results.json")
    
    return all_results

if __name__ == "__main__":
    results = main()
