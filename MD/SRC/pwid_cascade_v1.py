#!/usr/bin/env python3
"""
PWID LAI-PrEP Cascade Simulation: Counterfactual Policy Analysis
================================================================

ANNOTATED VERSION FOR LEARNING

This simulation models the probability of achieving R(0)=0 (sustained HIV 
prevention) for people who inject drugs under varying policy scenarios.

KEY 6.00.2x CONCEPTS DEMONSTRATED:
----------------------------------
1. STOCHASTIC SIMULATION (Monte Carlo methods)
   - Using randomness to model uncertain outcomes
   - Law of large numbers: more samples → better estimates
   
2. DATA ABSTRACTION (dataclasses)
   - Bundling related data into coherent objects
   - Separating data representation from computation
   
3. PROBABILITY & STATISTICS
   - Product of independent probabilities
   - Confidence intervals
   - Standard error calculation
   
4. SIMULATION DESIGN PATTERNS
   - Parameter sweeps across scenarios
   - Reproducibility via random seeds
   - Separation of model, simulation, and analysis

Author: AC Demidont, MD / Nyx Dynamics LLC
Date: December 2024
"""

# =============================================================================
# IMPORTS
# =============================================================================

import random
# The 'random' module provides pseudo-random number generation.
# Key functions we'll use:
#   random.random()  → returns float in [0.0, 1.0)
#   random.seed(n)   → sets the random seed for reproducibility
#
# WHY PSEUDO-RANDOM? Computers can't generate true randomness. They use
# deterministic algorithms that LOOK random. Setting a seed means we get
# the SAME sequence of "random" numbers each time—crucial for reproducibility.

from dataclasses import dataclass
# Dataclasses (Python 3.7+) are a clean way to create classes that mainly
# hold data. They auto-generate __init__, __repr__, and other methods.
# 
# Without dataclass, you'd write:
#   class CascadeStep:
#       def __init__(self, name, description, base_probability, ...):
#           self.name = name
#           self.description = description
#           ...
#
# With dataclass, you just declare the fields and Python does the rest.

from typing import Any, Dict, List, Tuple
# Type hints don't change how Python runs—they're documentation that helps
# you (and tools like mypy) catch bugs. 
#
# Dict[str, float] means "a dictionary with string keys and float values"
# List[Dict] means "a list of dictionaries"
# Tuple[float, float] means "a tuple of exactly two floats"

import json
# JSON (JavaScript Object Notation) is a standard format for storing/exchanging
# structured data. We'll use it to save our results for later analysis.


# =============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# =============================================================================

random.seed(42)
# WHY 42? It's a tradition (Hitchhiker's Guide reference), but any integer works.
#
# CRITICAL CONCEPT: In stochastic simulation, you MUST be able to reproduce
# your results. Setting the seed means:
#   - Run 1: random sequence is [0.374, 0.951, 0.231, ...]
#   - Run 2: random sequence is [0.374, 0.951, 0.231, ...] (identical!)
#
# For publication, you report the seed so others can verify your results.


# =============================================================================
# DATA STRUCTURES (Data Abstraction)
# =============================================================================

@dataclass
class CascadeStep:
    """
    Represents a single step in the PrEP cascade.
    
    DATA ABSTRACTION PRINCIPLE:
    We bundle all the information about a cascade step into one object.
    This makes the code cleaner and less error-prone than passing around
    6 separate variables.
    
    DESIGN DECISION:
    We decompose barriers into THREE categories (criminalization, bias, 
    structural) because:
    1. Different policies affect different barrier types
    2. We can attribute outcomes to specific barrier categories
    3. It enables counterfactual analysis ("what if we removed criminalization?")
    
    Attributes:
        name: Short identifier (e.g., "awareness", "disclosure")
        description: Human-readable explanation
        base_probability: P(pass step) if ALL barriers removed (theoretical max)
        criminalization_penalty: Probability reduction due to drug criminalization
        bias_penalty: Probability reduction due to healthcare provider bias
        structural_penalty: Probability reduction due to structural barriers
    
    PROBABILITY MODEL:
    P(pass step | policy) = base - penalties_that_apply + bonuses_that_apply
    
    This is a simplification! Real barriers interact nonlinearly. But linear
    models are interpretable and often "good enough" for policy analysis.
    """
    name: str
    description: str
    base_probability: float
    criminalization_penalty: float
    bias_penalty: float
    structural_penalty: float


@dataclass 
class PolicyScenario:
    """
    Represents a policy intervention scenario.
    
    This is where we define the COUNTERFACTUALS—the "what if" scenarios
    we want to compare against current policy.
    
    COUNTERFACTUAL REASONING:
    "What would happen if we changed policy X?"
    We can't run real experiments on policy (ethical/practical constraints),
    so we simulate counterfactuals computationally.
    
    Attributes:
        name: Scenario identifier
        description: What this scenario represents
        criminalization_removed: Is drug use decriminalized? (bool)
        bias_reduced: Fraction of bias penalty removed (0.0 to 1.0)
        structural_barriers_reduced: Fraction of structural barriers removed
        incarceration_rate_modifier: Multiplier on base incarceration rate
        in_custody_prep_available: Can people continue PrEP while incarcerated?
        ssp_integrated_delivery: Is PrEP delivered through syringe service programs?
        peer_navigation: Are peer navigators available?
    
    DESIGN PATTERN: PARAMETERIZED SCENARIOS
    Rather than hard-coding each scenario, we define parameters that can be
    varied. This lets us easily add new scenarios or do sensitivity analysis.
    """
    name: str
    description: str
    criminalization_removed: bool
    bias_reduced: float  # 0.0 = no reduction, 1.0 = fully eliminated
    structural_barriers_reduced: float
    incarceration_rate_modifier: float  # 1.0 = baseline, 0.0 = no incarceration
    in_custody_prep_available: bool
    ssp_integrated_delivery: bool
    peer_navigation: bool


# =============================================================================
# MODEL PARAMETERS: CASCADE STEPS
# =============================================================================

# Define the 8-step cascade with barrier decomposition
# 
# IMPORTANT: These numbers come from literature synthesis and expert judgment.
# They are ESTIMATES with uncertainty. The simulation quantifies what FOLLOWS
# from these assumptions—it doesn't validate the assumptions themselves.
#
# For a real publication, you'd:
# 1. Document sources for each parameter
# 2. Run sensitivity analysis varying parameters ±25-50%
# 3. Be explicit about which parameters have strong vs weak evidence

CASCADE_STEPS = [
    CascadeStep(
        name="awareness",
        description="Aware that PrEP exists and is available for PWID",
        base_probability=0.70,
        # Base 70%: PrEP awareness in general population is ~70%, but messaging
        # rarely reaches hidden/criminalized populations
        criminalization_penalty=0.30,
        # -30%: Criminalized populations are hidden from public health outreach.
        # No safe venues for messaging. Fear of visibility.
        bias_penalty=0.10,
        # -10%: PrEP campaigns historically focused on MSM, not PWID.
        # "Not for people like me" perception.
        structural_penalty=0.0
        # No structural penalty—awareness is about information access, not services
    ),
    
    CascadeStep(
        name="willingness",
        description="Willing to seek PrEP despite visibility risks",
        base_probability=0.80,
        # Base 80%: Most people, if aware of effective prevention, would want it
        criminalization_penalty=0.35,
        # -35%: THIS IS THE KEY INSIGHT. Seeking healthcare = becoming visible
        # to a system that incarcerates you. Rational risk avoidance.
        bias_penalty=0.05,
        # -5%: Anticipation of judgment from providers
        structural_penalty=0.0
    ),
    
    CascadeStep(
        name="healthcare_access",
        description="Can physically access healthcare services",
        base_probability=0.75,
        # Base 75%: Reflects general healthcare access barriers in US
        criminalization_penalty=0.10,
        # -10%: Outstanding warrants, legal concerns about seeking care
        bias_penalty=0.05,
        # -5%: "They won't want to see me"
        structural_penalty=0.25
        # -25%: Transportation, clinic hours, no established PCP, 
        # homelessness, lack of ID/insurance documentation
    ),
    
    CascadeStep(
        name="disclosure",
        description="Willing to disclose injection drug use to provider",
        base_probability=0.70,
        # Base 70%: People generally want to be honest with doctors
        criminalization_penalty=0.30,
        # -30%: Disclosing IDU = admitting felony behavior to potential
        # mandated reporter. Child custody concerns. Documentation in
        # medical record that could be subpoenaed.
        bias_penalty=0.10,
        # -10%: Fear of being treated differently, denied services
        structural_penalty=0.0
    ),
    
    CascadeStep(
        name="provider_willing",
        description="Provider willing to prescribe PrEP for PWID",
        base_probability=0.85,
        # Base 85%: Most providers would prescribe if asked and trained
        criminalization_penalty=0.05,
        # -5%: Some providers have legal/liability concerns
        bias_penalty=0.25,
        # -25%: "Not my patient population." "They won't be adherent."
        # "I don't have time for complex patients." "They should stop
        # using drugs first." These are documented provider attitudes.
        structural_penalty=0.0
    ),
    
    CascadeStep(
        name="affordability",
        description="Can afford/access medication through PAP or insurance",
        base_probability=0.80,
        # Base 80%: Patient assistance programs exist, Medicaid covers PrEP
        criminalization_penalty=0.15,
        # -15%: PAPs require stable address for delivery. No address = no PAP.
        # Felony records can disqualify from some assistance programs.
        bias_penalty=0.05,
        # -5%: Pharmacy interactions, judgment
        structural_penalty=0.15
        # -15%: Insurance gaps, ID requirements, prior authorization delays
    ),
    
    CascadeStep(
        name="first_injection",
        description="Returns for and receives first LAI-CAB injection",
        base_probability=0.75,
        # Base 75%: ~25% of people prescribed any medication don't fill/start it
        criminalization_penalty=0.10,
        # -10%: Incarceration between prescription and appointment
        bias_penalty=0.05,
        # -5%: Clinic environment not welcoming
        structural_penalty=0.15
        # -15%: Appointment scheduling systems, wait times, inflexibility
    ),
    
    CascadeStep(
        name="sustained_engagement",
        description="Maintains Q6M schedule over time",
        base_probability=0.70,
        # Base 70%: Long-acting injectables show ~80% persistence in trials,
        # but real-world slightly lower
        criminalization_penalty=0.20,
        # -20%: Incarceration interrupts the schedule. No in-custody PrEP.
        # This is CUMULATIVE over 5 years—each year is another opportunity
        # for incarceration to disrupt continuity.
        bias_penalty=0.10,
        # -10%: Negative healthcare experiences lead to disengagement
        structural_penalty=0.10
        # -10%: Life instability, missed appointments, clinic closure
    )
]


# =============================================================================
# MODEL PARAMETERS: POLICY SCENARIOS
# =============================================================================

# We define 7 scenarios from "current policy" to "theoretical maximum"
# This creates a GRADIENT that shows incremental policy effects

POLICY_SCENARIOS = [
    PolicyScenario(
        name="Current Policy",
        description="Status quo: full criminalization, systemic bias, no harm reduction integration",
        criminalization_removed=False,
        bias_reduced=0.0,
        structural_barriers_reduced=0.0,
        incarceration_rate_modifier=1.0,  # Full baseline incarceration rate
        in_custody_prep_available=False,
        ssp_integrated_delivery=False,
        peer_navigation=False
    ),
    
    PolicyScenario(
        name="Decriminalization Only",
        description="Drug use decriminalized, but healthcare system unchanged",
        criminalization_removed=True,  # Removes all criminalization penalties!
        bias_reduced=0.0,  # But bias persists—decrim doesn't change attitudes
        structural_barriers_reduced=0.0,
        incarceration_rate_modifier=0.3,  # 70% reduction, but still some
        # (other offenses, probation violations, etc.)
        in_custody_prep_available=False,
        ssp_integrated_delivery=False,
        peer_navigation=False
    ),
    
    PolicyScenario(
        name="Decrim + Stigma Reduction",
        description="Decriminalization plus provider stigma reduction training",
        criminalization_removed=True,
        bias_reduced=0.5,  # Training reduces stigma by 50%
        structural_barriers_reduced=0.0,
        incarceration_rate_modifier=0.3,
        in_custody_prep_available=False,
        ssp_integrated_delivery=False,
        peer_navigation=False
    ),
    
    PolicyScenario(
        name="Decrim + Low-Barrier Access",
        description="Decriminalization plus structural barrier reduction",
        criminalization_removed=True,
        bias_reduced=0.3,
        structural_barriers_reduced=0.6,  # Walk-in, mobile, extended hours
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
        ssp_integrated_delivery=True,   # Adds bonus to relevant steps
        peer_navigation=True             # Adds bonus to relevant steps
    ),
    
    PolicyScenario(
        name="Full Harm Reduction Infrastructure",
        description="Complete policy transformation",
        criminalization_removed=True,
        bias_reduced=0.8,
        structural_barriers_reduced=0.9,
        incarceration_rate_modifier=0.2,
        in_custody_prep_available=True,  # KEY: Incarceration no longer disrupts!
        ssp_integrated_delivery=True,
        peer_navigation=True
    ),
    
    PolicyScenario(
        name="Theoretical Maximum",
        description="All barriers removed (ceiling for comparison)",
        criminalization_removed=True,
        bias_reduced=1.0,           # 100% bias removed
        structural_barriers_reduced=1.0,  # 100% structural barriers removed
        incarceration_rate_modifier=0.0,  # Zero incarceration
        in_custody_prep_available=True,
        ssp_integrated_delivery=True,
        peer_navigation=True
    )
]


# =============================================================================
# CORE MODEL FUNCTIONS
# =============================================================================

def calculate_step_probability(step: CascadeStep, scenario: PolicyScenario) -> float:
    """
    Calculate probability of passing a cascade step under a given policy scenario.
    
    This is the CORE MODEL EQUATION:
    
    P(pass) = base 
              - (criminalization_penalty if not decriminalized else 0)
              - (bias_penalty × (1 - bias_reduction))
              - (structural_penalty × (1 - structural_reduction))
              + SSP_bonus (if applicable)
              + peer_navigation_bonus (if applicable)
    
    MATHEMATICAL NOTE:
    This is a LINEAR model. We assume penalties and bonuses ADD/SUBTRACT.
    In reality, barriers likely INTERACT (multiply). But linear models are:
    - Easier to interpret
    - More transparent about assumptions
    - Often reasonable first approximations
    
    For a more sophisticated model, you might use:
    P(pass) = base × (1 - penalty1) × (1 - penalty2) × ...
    
    Args:
        step: The cascade step being evaluated
        scenario: The policy scenario being modeled
        
    Returns:
        Probability of passing this step (bounded to [0.01, 0.99])
    """
    
    # Start with the base probability (theoretical max without barriers)
    prob = step.base_probability
    
    # Apply criminalization penalty (completely removed if decriminalized)
    if not scenario.criminalization_removed:
        prob -= step.criminalization_penalty
    
    # Apply bias penalty (partially reduced by intervention)
    # If bias_reduced = 0.5, we apply 50% of the original penalty
    bias_impact = step.bias_penalty * (1 - scenario.bias_reduced)
    prob -= bias_impact
    
    # Apply structural penalty (partially reduced by intervention)
    structural_impact = step.structural_penalty * (1 - scenario.structural_barriers_reduced)
    prob -= structural_impact
    
    # SSP integration provides BONUS for specific steps
    # SSPs are trusted by PWID—they provide a non-judgmental entry point
    if scenario.ssp_integrated_delivery:
        if step.name in ["awareness", "healthcare_access", "disclosure", "first_injection"]:
            prob += 0.15  # +15% bonus
    
    # Peer navigation provides BONUS for engagement steps
    # Peers with lived experience help navigate the system
    if scenario.peer_navigation:
        if step.name in ["willingness", "first_injection", "sustained_engagement"]:
            prob += 0.10  # +10% bonus
    
    # BOUND the probability to valid range
    # We use [0.01, 0.99] instead of [0, 1] to avoid edge cases in simulation
    # (A step with P=0 would block everyone; P=1 would let everyone through)
    return max(0.01, min(0.99, prob))


def calculate_incarceration_disruption(
    scenario: PolicyScenario,
    years: int = 5,
    base_annual_incarceration_rate: float = 0.30
) -> float:
    """
    Calculate probability of maintaining PrEP protection given incarceration risk.
    
    KEY INSIGHT: Even if someone successfully navigates the entire cascade,
    incarceration can RESET them to zero if PrEP isn't available in custody.
    
    PROBABILITY MODEL:
    P(no incarceration in year i) = 1 - annual_rate
    P(no incarceration over N years) = (1 - annual_rate)^N
    
    This assumes incarceration events are INDEPENDENT across years, which is
    a simplification. In reality, incarceration risk is likely CORRELATED
    (prior incarceration predicts future incarceration).
    
    Args:
        scenario: Policy scenario being modeled
        years: Time horizon (default 5 years)
        base_annual_incarceration_rate: Annual probability of incarceration (default 30%)
            Note: 30% is HIGH but realistic for active PWID in many jurisdictions
            
    Returns:
        Probability of maintaining protection over the time horizon
    """
    # Adjust incarceration rate by policy scenario
    annual_rate = base_annual_incarceration_rate * scenario.incarceration_rate_modifier
    
    if scenario.in_custody_prep_available:
        # If PrEP is available in custody, incarceration doesn't disrupt protection
        return 1.0
    else:
        # Probability of avoiding incarceration for ALL years
        # P(no incarc) = (1 - rate)^years
        #
        # Example: 30% annual rate over 5 years
        # P(survive) = 0.70^5 = 0.168 (only 16.8% avoid incarceration!)
        prob_no_incarceration = (1 - annual_rate) ** years
        return prob_no_incarceration


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def simulate_individual(scenario: PolicyScenario, n_years: int = 5) -> Dict[str, Any]:
    """
    Simulate a single PWID's journey through the PrEP cascade.
    
    MONTE CARLO METHOD:
    Instead of computing exact probabilities analytically, we SIMULATE
    random outcomes and count results. With enough simulations, the
    frequencies converge to the true probabilities (Law of Large Numbers).
    
    WHY MONTE CARLO?
    1. Cascade is SEQUENTIAL: failure at step 3 means steps 4-8 never happen
    2. Incarceration is CONDITIONAL: only applies if you complete cascade
    3. Some models are too complex for analytical solutions
    4. Easy to add complexity (correlations, time-varying rates, etc.)
    
    SIMULATION LOGIC:
    For each step:
        1. Calculate probability of passing
        2. Draw random number in [0, 1)
        3. If random < probability: PASS, continue to next step
        4. If random >= probability: FAIL, exit cascade
    
    If all steps passed:
        1. Calculate incarceration survival probability
        2. Draw random number
        3. If survives: achieved R(0)=0
        4. If caught: protection lost
    
    Args:
        scenario: Policy scenario being modeled
        n_years: Time horizon for incarceration risk
        
    Returns:
        Dictionary with simulation outcomes:
        - completed_cascade: Did they pass all 8 steps?
        - failed_step: Which step did they fail (if any)?
        - incarceration_disrupted: Completed cascade but caught?
        - achieved_r0_zero: Final outcome—sustained protection?
        - step_outcomes: Details for each step
    """
    
    # Initialize result tracking
    results: Dict[str, Any] = {
        "completed_cascade": True,  # Assume success until proven otherwise
        "failed_step": None,
        "incarceration_disrupted": False,
        "achieved_r0_zero": False,
        "step_outcomes": {}
    }
    
    # Progress through cascade steps SEQUENTIALLY
    for step in CASCADE_STEPS:
        # Get probability for this step under this scenario
        prob = calculate_step_probability(step, scenario)
        
        # STOCHASTIC DECISION: Draw random number and compare to probability
        # random.random() returns uniform random in [0, 1)
        # If random < prob: event occurs (pass)
        # If random >= prob: event doesn't occur (fail)
        passed = random.random() < prob
        
        # Record outcome for this step
        results["step_outcomes"][step.name] = {
            "probability": prob,
            "passed": passed
        }
        
        # If failed, cascade stops here
        if not passed:
            results["completed_cascade"] = False
            results["failed_step"] = step.name
            break  # Exit the for loop—no point simulating remaining steps
    
    # If completed entire cascade, check for incarceration disruption
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
) -> Dict[str, Any]:
    """
    Run full Monte Carlo simulation for a policy scenario.
    
    LAW OF LARGE NUMBERS:
    As n_individuals → ∞, observed frequencies → true probabilities.
    
    How many samples do we need?
    - For proportion p, standard error SE = sqrt(p(1-p)/n)
    - With n=100,000 and p=0.25: SE = sqrt(0.25×0.75/100000) = 0.0014
    - 95% CI width = ±1.96×SE ≈ ±0.003 (very precise!)
    
    For rare events (p=0.001), we need more samples to get stable estimates.
    
    COMPUTATIONAL COST:
    100,000 individuals × 7 scenarios × (8 steps + 1 incarceration check)
    = ~6.3 million random draws
    Takes ~5-10 seconds on modern hardware.
    
    Args:
        scenario: Policy scenario to simulate
        n_individuals: Number of Monte Carlo samples
        n_years: Time horizon
        
    Returns:
        Dictionary with:
        - Theoretical probabilities (analytical)
        - Observed rates (from simulation)
        - Confidence intervals
        - Step-by-step breakdown
    """
    
    # Initialize results dictionary
    results: Dict[str, Any] = {
        "scenario": scenario.name,
        "n_individuals": n_individuals,
        "n_years": n_years,
        "achieved_r0_zero": 0,       # Counter
        "completed_cascade": 0,       # Counter
        "incarceration_disrupted": 0, # Counter
        "step_failure_counts": {step.name: 0 for step in CASCADE_STEPS},
        "step_probabilities": {}      # Will hold theoretical probs
    }
    
    # =================================
    # ANALYTICAL CALCULATION (THEORETICAL)
    # =================================
    
    # Calculate theoretical step probabilities
    for step in CASCADE_STEPS:
        results["step_probabilities"][step.name] = calculate_step_probability(step, scenario)
    
    # Theoretical cascade completion = PRODUCT of step probabilities
    # P(complete cascade) = P(step1) × P(step2) × ... × P(step8)
    #
    # This assumes INDEPENDENCE: outcome of step 3 doesn't affect
    # probability of step 4 (conditional on reaching step 4).
    # This is a simplification—in reality, there may be correlations.
    theoretical_cascade_prob = 1.0
    for step in CASCADE_STEPS:
        theoretical_cascade_prob *= results["step_probabilities"][step.name]
    results["theoretical_cascade_probability"] = theoretical_cascade_prob
    
    # Theoretical incarceration survival
    results["incarceration_survival_probability"] = calculate_incarceration_disruption(scenario, n_years)
    
    # Theoretical R(0)=0 probability = cascade × incarceration survival
    results["theoretical_r0_zero_probability"] = (
        theoretical_cascade_prob * results["incarceration_survival_probability"]
    )
    
    # =================================
    # MONTE CARLO SIMULATION
    # =================================
    
    # Run n_individuals through the simulation
    for _ in range(n_individuals):
        # The underscore '_' is Python convention for "I don't need this variable"
        # We're just counting iterations, not using the index
        
        individual = simulate_individual(scenario, n_years)
        
        # Count outcomes
        if individual["achieved_r0_zero"]:
            results["achieved_r0_zero"] += 1
        
        if individual["completed_cascade"]:
            results["completed_cascade"] += 1
            
        if individual["incarceration_disrupted"]:
            results["incarceration_disrupted"] += 1
            
        if individual["failed_step"]:
            results["step_failure_counts"][individual["failed_step"]] += 1
    
    # =================================
    # CALCULATE STATISTICS
    # =================================
    
    # Observed rates = counts / n
    results["observed_r0_zero_rate"] = results["achieved_r0_zero"] / n_individuals
    results["observed_cascade_completion_rate"] = results["completed_cascade"] / n_individuals
    
    # 95% Confidence Interval for proportion
    # 
    # For a proportion p estimated from n samples:
    #   Standard Error: SE = sqrt(p(1-p)/n)
    #   95% CI: p ± 1.96 × SE
    #
    # WHY 1.96? It's the z-score for 95% confidence in a normal distribution.
    # By Central Limit Theorem, sample proportions are approximately normal
    # for large n.
    
    p = results["observed_r0_zero_rate"]
    n = n_individuals
    se = (p * (1-p) / n) ** 0.5  # ** 0.5 is square root
    results["r0_zero_95ci"] = (max(0, p - 1.96*se), min(1, p + 1.96*se))
    
    return results


# =============================================================================
# POPULATION IMPACT CALCULATIONS
# =============================================================================

def calculate_population_impact(
    results: Dict,
    pwid_population: int = 3_500_000,  # US PWID population estimate
    annual_hiv_incidence_no_prep: float = 0.02  # 2% annual incidence without PrEP
) -> Dict:
    """
    Scale simulation results to population-level impact.
    
    This is where simulation meets POLICY RELEVANCE.
    
    We convert abstract probabilities into:
    - Number of people protected
    - Infections prevented
    - Healthcare costs averted
    
    MODEL ASSUMPTIONS:
    1. PWID population is ~3.5 million in US (CDC estimate)
    2. Annual HIV incidence without PrEP is ~2% (from outbreak data)
    3. PrEP efficacy is 99% (PURPOSE-4 assumption)
    4. Lifetime HIV treatment cost is $500,000 (CDC estimate)
    
    CAUTION: These are rough estimates for policy discussion, not precise
    forecasts. Real impact depends on many factors we're not modeling.
    
    Args:
        results: Output from run_simulation()
        pwid_population: Total PWID population
        annual_hiv_incidence_no_prep: Annual infection probability without PrEP
        
    Returns:
        Dictionary with population-level impact metrics
    """
    
    p_protected = results["observed_r0_zero_rate"]
    prep_efficacy = 0.99  # 99% efficacy assumption
    
    # Number achieving sustained protection
    n_protected = pwid_population * p_protected
    
    # INFECTION CALCULATION
    # 
    # Without any PrEP: infections = population × incidence
    # 
    # With policy:
    #   - Unprotected group: (population - n_protected) × incidence
    #   - Protected group: n_protected × incidence × (1 - efficacy)
    #     [The (1 - efficacy) term accounts for PrEP failures]
    
    annual_infections_baseline = pwid_population * annual_hiv_incidence_no_prep
    
    annual_infections_with_policy = (
        (pwid_population - n_protected) * annual_hiv_incidence_no_prep +
        n_protected * annual_hiv_incidence_no_prep * (1 - prep_efficacy)
    )
    
    infections_prevented = annual_infections_baseline - annual_infections_with_policy
    
    # 5-year cumulative (simple multiplication—ignores population dynamics)
    five_year_infections_prevented = infections_prevented * 5
    
    # Cost calculation
    # Each prevented infection saves ~$500,000 in lifetime treatment costs
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


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_results_table(all_results: List[Dict]) -> str:
    """
    Format simulation results as a text table.
    
    STRING FORMATTING IN PYTHON:
    
    f-strings (f"...") allow embedded expressions:
        name = "Alice"
        f"Hello, {name}!"  →  "Hello, Alice!"
    
    Format specifiers control display:
        f"{x:.4f}"   →  4 decimal places (0.1234)
        f"{x:.2%}"   →  percentage with 2 decimals (12.34%)
        f"{x:,}"     →  thousands separator (1,234,567)
        f"{s:<40}"   →  left-align in 40-char field
        f"{s:>10}"   →  right-align in 10-char field
    
    Args:
        all_results: List of results from run_simulation()
        
    Returns:
        Formatted string table
    """
    
    lines = []
    lines.append("=" * 120)
    lines.append("PWID LAI-PrEP CASCADE SIMULATION: POLICY SCENARIO COMPARISON")
    lines.append("Assuming: 99% efficacy Q6M injectable (PURPOSE-4 success), 5-year time horizon")
    lines.append("=" * 120)
    lines.append("")
    
    # Summary table header
    # The :<40 means "left-align in a field of width 40"
    lines.append(f"{'Scenario':<40} {'P(R(0)=0)':<12} {'95% CI':<20} {'Cascade':<10} {'Incarc Surv':<12}")
    lines.append("-" * 100)
    
    # Data rows
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
    
    # Step probabilities table
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
    
    # Extract key scenarios for comparison
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
    """
    Generate detailed analytical report with barrier attribution.
    
    This shows WHERE people fall out of the cascade—crucial for targeting
    interventions effectively.
    """
    
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
    
    # Show cascade attrition
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
    
    # Barrier attribution
    report.append("BARRIER ATTRIBUTION ANALYSIS")
    report.append("-" * 60)
    report.append("")
    report.append("Total probability loss by barrier type (under current policy):")
    report.append("")
    
    # Sum penalties across all steps
    crim_loss = sum(s.criminalization_penalty for s in CASCADE_STEPS)
    bias_loss = sum(s.bias_penalty for s in CASCADE_STEPS)
    struct_loss = sum(s.structural_penalty for s in CASCADE_STEPS)
    
    report.append(f"  Criminalization penalties:  -{crim_loss:.2f} (cumulative across steps)")
    report.append(f"  Healthcare bias penalties:  -{bias_loss:.2f}")
    report.append(f"  Structural barriers:        -{struct_loss:.2f}")
    report.append("")
    
    # Incremental policy analysis
    report.append("INCREMENTAL POLICY ANALYSIS")
    report.append("-" * 60)
    report.append("")
    
    for i in range(len(all_results) - 1):
        current_scenario = all_results[i]
        next_scenario = all_results[i + 1]
        
        delta = next_scenario['observed_r0_zero_rate'] - current_scenario['observed_r0_zero_rate']
        # Avoid division by zero
        if current_scenario['observed_r0_zero_rate'] > 0:
            relative = (delta / current_scenario['observed_r0_zero_rate'] * 100)
        else:
            relative = float('inf')  # Infinite improvement from zero
        
        report.append(f"  {current_scenario['scenario']:<35} → {next_scenario['scenario']}")
        report.append(f"    Δ P(R(0)=0): +{delta:.4f} ({relative:+.1f}% relative improvement)")
        report.append("")
    
    return "\n".join(report)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Run all simulations and generate output.
    
    PROGRAM STRUCTURE:
    1. Loop through policy scenarios
    2. Run simulation for each
    3. Collect results
    4. Generate reports
    5. Save to JSON for further analysis
    
    This is a common pattern: separate DATA GENERATION from ANALYSIS.
    We save raw results to JSON so we can do additional analysis later
    without re-running the expensive simulation.
    """
    
    print("Running PWID LAI-PrEP Cascade Simulation...")
    print(f"Simulating {100000:,} individuals per scenario, 5-year horizon")
    print("")
    
    all_results = []
    
    # Run simulation for each policy scenario
    for scenario in POLICY_SCENARIOS:
        print(f"  Simulating: {scenario.name}...")
        results = run_simulation(scenario, n_individuals=100000, n_years=5)
        all_results.append(results)
    
    print("")
    print("Generating reports...")
    
    # Generate and print main results table
    results_table = format_results_table(all_results)
    print(results_table)
    
    # Generate and print detailed report
    detailed_report = generate_detailed_report(all_results)
    print("")
    print(detailed_report)
    
    # Save results to JSON for further analysis
    json_results = []
    for r in all_results:
        # Make a copy to avoid modifying original
        r_copy = r.copy()
        # Convert tuple to list for JSON serialization (tuples become arrays)
        r_copy['r0_zero_95ci'] = list(r['r0_zero_95ci'])
        # Add impact calculations
        r_copy['impact'] = calculate_population_impact(r)
        json_results.append(r_copy)
    
    # Write to file
    with open('/home/claude/pwid_simulation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("")
    print("Results saved to pwid_simulation_results.json")
    
    return all_results


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # This block only runs when script is executed directly (not imported)
    #
    # WHY THIS PATTERN?
    # If someone does `import pwid_cascade_simulation`, they get access to
    # all the functions without automatically running the simulation.
    # This makes the code REUSABLE as a module.
    #
    # When run directly: python pwid_cascade_simulation.py
    # __name__ is set to "__main__" and this block executes.
    #
    # When imported: import pwid_cascade_simulation
    # __name__ is set to "pwid_cascade_simulation" and this block is skipped.
    
    results = main()


# =============================================================================
# KEY CONCEPTS SUMMARY FOR 6.00.2x
# =============================================================================
"""
CONCEPTS DEMONSTRATED IN THIS CODE:

1. MONTE CARLO SIMULATION
   - Use random sampling to estimate probabilities
   - Law of Large Numbers: more samples → better estimates
   - Set random seed for reproducibility
   
2. DATA ABSTRACTION
   - dataclasses bundle related data
   - Separate data representation from computation
   - Makes code more maintainable and less error-prone

3. PROBABILITY
   - Product of independent probabilities
   - Conditional probability (incarceration | cascade completion)
   - Confidence intervals via standard error

4. SIMULATION DESIGN
   - Parameterize scenarios (don't hard-code)
   - Separate theoretical (analytical) from empirical (simulated)
   - Save raw results for later analysis

5. SCIENTIFIC COMPUTING PRACTICES
   - Document assumptions explicitly
   - Report uncertainty (confidence intervals)
   - Enable reproducibility (random seed, save parameters)

6. PYTHON IDIOMS
   - f-strings for formatting
   - List comprehensions: [x.name for x in CASCADE_STEPS]
   - Dictionary comprehensions: {step.name: 0 for step in CASCADE_STEPS}
   - Context managers: with open(...) as f:
   - Type hints for documentation

7. THE BIG PICTURE
   This simulation shows that MATHEMATICAL MODELING can reveal
   policy failures that aren't obvious from descriptive statistics.
   
   The key insight: when you MULTIPLY probabilities through a cascade,
   even moderately low probabilities at each step compound to near-zero.
   
   P = 0.30 × 0.40 × 0.35 × 0.30 × 0.55 × 0.45 × 0.45 × 0.30 = 0.0004
   
   This is why current policy achieves R(0)=0 for essentially 0% of PWID.
   The cascade structure GUARANTEES failure regardless of drug efficacy.
"""
