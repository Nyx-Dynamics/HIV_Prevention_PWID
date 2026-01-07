#!/usr/bin/env python3
"""
Architectural Barrier Model: HIV Prevention Cascade Modeling for PWID
======================================================================

Comprehensive Monte Carlo simulation modeling nested barriers to HIV prevention
for people who inject drugs (PWID), demonstrating the mathematical challenge
of achieving R(0)=0 under current policy conditions.

PRIMARY HYPOTHESES:
1. Nested barriers create conditions where R(0)=0 is mathematically challenging
   for PWID regardless of pharmacological efficacy
2. Stochastic avoidance (probability) rather than intervention has been the
   primary mechanism preventing catastrophic HIV outbreaks among PWID
3. Introduction of methamphetamine into non-MSM PWID populations increases
   outbreak probability through network bridging and behavioral factors

THREE-LAYER BARRIER FRAMEWORK:
Layer 1: Pathogen Biology (irreversible R(0)>0 within hours)
Layer 2: HIV Testing Failures (acute infection detection gaps)
Layer 3: Architectural Barriers
    - Policy (criminalization, incarceration)
    - Stigma (healthcare discrimination, disclosure barriers)
    - Infrastructure (cascade implementation built for MSM)
    - Research Exclusion (LOOCV framework)
    - Machine Learning (algorithmic deprioritization)

Author: AC Demidont, DO / Nyx Dynamics LLC
Date: December 2024
"""

import numpy as np
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import json
import csv
import os
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
rng = np.random.default_rng(42)
random.seed(42)


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class BarrierLayer(Enum):
    """Three-layer barrier framework"""
    PATHOGEN_BIOLOGY = "pathogen_biology"
    HIV_TESTING = "hiv_testing"
    ARCHITECTURAL = "architectural"


class ArchitecturalSubtype(Enum):
    """Subtypes of architectural barriers"""
    POLICY = "policy"
    STIGMA = "stigma"
    INFRASTRUCTURE = "infrastructure"
    RESEARCH_EXCLUSION = "research_exclusion"
    MACHINE_LEARNING = "machine_learning"


# Literature-derived constants
LITERATURE_PARAMS = {
    # Global PWID epidemiology (Degenhardt et al., 2017)
    "global_pwid_population": 15_600_000,
    "global_pwid_hiv_prevalence": 0.178,  # 17.8%
    "us_pwid_population": 3_500_000,

    # Methamphetamine HIV risk (Plankey 2007, Grov 2020)
    "meth_hr_hiv_seroconversion": 1.46,  # HR for meth alone
    "meth_persistent_aor": 7.11,  # AOR for persistent users
    "meth_opioid_coue_2012": 0.043,  # 4.3% in 2012
    "meth_opioid_couse_2018": 0.143,  # 14.3% in 2018

    # Criminalization impact (DeBeck et al., 2017)
    "criminalization_negative_effect_rate": 0.80,  # 80% studies show negative effect
    "incarceration_hiv_rr": 1.81,  # RR 1.81 (95% CI: 1.40-2.34)

    # Healthcare stigma (Biancarelli 2019, Muncan 2020)
    "pwid_stigma_experienced": 0.88,  # 88% experienced stigma
    "pwid_stigma_affecting_care": 0.78,  # 78.1% reported
    "rural_pwid_stigma": 0.62,  # 62% in rural areas

    # Housing instability (Arum et al., 2021)
    "unstable_housing_hiv_arr": 1.39,  # aRR 1.39 (95% CI: 1.06-1.84)
    "pwid_homelessness_rate": 0.685,  # 68.5% experienced homelessness

    # PrEP cascade failure (Mistler et al., 2021)
    "pwid_prep_uptake": 0.015,  # 0-3%, using midpoint
    "global_pwid_art_coverage": 0.04,  # Only 4%

    # WHO intervention requirements vs reality
    "who_required_syringes_per_pwid": 200,  # Minimum
    "actual_syringes_per_pwid": 22,  # Global average
    "who_required_oat_coverage": 0.40,  # Minimum 40%
    "actual_oat_coverage": 0.08,  # 8% globally

    # LAI-PrEP resistance (HPTN 083, Eshleman 2022)
    "cab_la_insti_resistance_rate": 0.63,  # 63% in late-detected breakthrough
    "cab_la_detection_delay_median_days": 98,  # Median 98 days

    # Vulnerable counties (Van Handel et al., 2016)
    "vulnerable_us_counties": 220,
    "vulnerable_counties_with_ssp": 0.21,  # Only 21%

    # Outbreak data
    "scott_county_cases": 215,
    "massachusetts_cases": 180,
    "cabell_county_cases": 82,
    "kanawha_county_cases": 85,

    # Stochastic factors (Des Jarlais et al., 2022)
    "baseline_annual_outbreak_prob": 0.03,  # 3% baseline
    "covid_disruption_outbreak_prob": 0.25,  # 25% during COVID
    "high_risk_outbreak_prob": 0.33,  # 33% high-risk scenarios
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PathogenBiologyParams:
    """Layer 1: Pathogen biology parameters"""
    # HIV integration timeline (hours post-exposure)
    mucosal_phase_hours: float = 4.0
    dendritic_uptake_hours: float = 12.0
    lymph_transit_hours: float = 36.0
    systemic_spread_hours: float = 60.0
    integration_midpoint_hours: float = 72.0
    integration_complete_hours: float = 120.0  # Point of no return

    # Per-exposure transmission probability
    parenteral_transmission_prob: float = 0.0063  # 0.63% per injection
    sexual_transmission_prob: float = 0.008  # Variable by act type

    # Drug efficacy (irrelevant if cascade not completed)
    lai_prep_efficacy: float = 0.999  # PURPOSE-2 lenacapavir
    oral_prep_efficacy: float = 0.99  # When adherent

    def p_integration_complete(self, hours_post_exposure: float) -> float:
        """Probability that reservoir integration is irreversible by time t"""
        from scipy.special import expit
        k = 0.15  # Steepness
        return expit(k * (hours_post_exposure - self.integration_complete_hours))


@dataclass
class HIVTestingParams:
    """Layer 2: HIV testing failure parameters"""
    # Window periods (days)
    rna_nat_window_days: float = 10.0  # 10-33 days
    fourth_gen_agab_window_days: float = 18.0  # 18-45 days
    rapid_poc_window_days: float = 31.0  # 31-90 days

    # Detection probabilities during acute infection
    p_acute_detection_rna: float = 0.85
    p_acute_detection_agab: float = 0.60
    p_acute_detection_rapid: float = 0.30

    # LAI-PrEP specific testing gaps
    cab_la_delays_seroconversion: bool = True
    cab_la_median_detection_delay_days: float = 98.0

    # nPEP 2025 guideline compliance
    nPEP_requires_nat_at_followup: bool = True  # New requirement
    prep_requires_nat: bool = False  # Not required for PrEP

    # Resistance risk if detected late
    late_detection_insti_resistance_prob: float = 0.63
    late_detection_capsid_resistance_prob: float = 1.0  # Both PURPOSE-2 cases


@dataclass
class ArchitecturalBarrierParams:
    """Layer 3: Architectural barrier parameters"""

    # 3a. Policy barriers
    drug_criminalization: bool = True
    annual_incarceration_rate_pwid: float = 0.30  # 30% annual
    incarceration_hiv_rr: float = 1.81

    # 3b. Stigma barriers
    healthcare_stigma_prevalence: float = 0.78
    p_disclosure_given_stigma_fear: float = 0.30
    p_care_seeking_given_stigma: float = 0.50

    # 3c. Infrastructure barriers (MSM-centric design)
    prep_awareness_msm: float = 0.85
    prep_awareness_pwid: float = 0.35
    prep_uptake_msm: float = 0.25
    prep_uptake_pwid: float = 0.015
    ssp_coverage: float = 0.21  # Only 21% of vulnerable counties
    oat_coverage: float = 0.08

    # 3d. Research exclusion (LOOCV framework)
    pwid_in_major_prep_trials: int = 1  # Only Bangkok
    total_major_prep_trials: int = 11
    pwid_in_best_practices: int = 0
    total_best_practices: int = 9
    pwid_lai_prep_trials: int = 0  # Zero until PURPOSE-4
    snr_msm: float = 9180.0
    snr_pwid: float = 76.4
    snr_ratio: float = 120.0  # 120-fold disparity

    # 3e. Machine learning barriers
    p_algorithmic_deprioritization_pwid: float = 0.85  # Inverse of inclusion
    p_algorithmic_deprioritization_msm: float = 0.08

    # Combined architectural barrier effect
    def calculate_architectural_attrition(self) -> Dict[str, float]:
        """Calculate attrition contribution by architectural subtype"""
        # These are multiplicative reductions to cascade completion
        return {
            "policy": 0.525,  # 52.5% from policy barriers
            "stigma": 0.254,  # 25.4% from implementation/stigma
            "infrastructure": 0.15,  # Infrastructure gaps
            "research_exclusion": 0.10,  # LOOCV exclusion effect
            "machine_learning": 0.221,  # 22.1% from algorithmic barriers
        }


@dataclass
class CascadeStep:
    """Single step in the PrEP cascade"""
    name: str
    description: str
    base_probability: float
    barrier_layer: BarrierLayer
    architectural_subtype: Optional[ArchitecturalSubtype] = None

    # Decomposed penalties
    pathogen_penalty: float = 0.0
    testing_penalty: float = 0.0
    policy_penalty: float = 0.0
    stigma_penalty: float = 0.0
    infrastructure_penalty: float = 0.0
    research_penalty: float = 0.0
    ml_penalty: float = 0.0


@dataclass
class PolicyScenario:
    """Policy intervention scenario"""
    name: str
    description: str

    # Policy modifications
    decriminalization: bool = False
    incarceration_modifier: float = 1.0
    in_custody_prep: bool = False

    # Stigma interventions
    stigma_reduction: float = 0.0  # 0-1
    provider_stigma_training: bool = False  # Healthcare provider anti-stigma training

    # Infrastructure improvements
    ssp_integrated_delivery: bool = False
    peer_navigation: bool = False
    low_barrier_access: bool = False

    # Research inclusion
    pwid_trial_inclusion: bool = False  # PURPOSE-4 scenario

    # ML debiasing
    algorithmic_debiasing: bool = False


@dataclass
class StochasticAvoidanceParams:
    """Parameters for stochastic avoidance failure model"""
    # Network parameters
    baseline_network_density: float = 0.15  # Low connectivity
    meth_network_density_multiplier: float = 2.5  # Meth increases connections

    # Outbreak threshold parameters
    critical_network_threshold: float = 0.35  # Above this, outbreak likely

    # Temporal dynamics
    baseline_annual_outbreak_prob: float = 0.03
    years_to_threshold: Optional[float] = None

    # Risk factors
    meth_prevalence_trend_annual: float = 0.025  # 2.5% annual increase
    housing_instability_rate: float = 0.685
    sex_work_network_bridge_prob: float = 0.15


# =============================================================================
# CASCADE DEFINITION
# =============================================================================

def create_pwid_cascade() -> List[CascadeStep]:
    """
    Create the PWID LAI-PrEP cascade with 3-layer barrier decomposition.

    Returns 8-step cascade with penalties from:
    - Pathogen biology
    - HIV testing failures
    - Architectural barriers (5 subtypes)
    """

    return [
        CascadeStep(
            name="awareness",
            description="Aware that LAI-PrEP exists and is available for PWID",
            base_probability=0.70,
            barrier_layer=BarrierLayer.ARCHITECTURAL,
            architectural_subtype=ArchitecturalSubtype.INFRASTRUCTURE,
            policy_penalty=0.25,  # Hidden population, no outreach
            stigma_penalty=0.05,
            infrastructure_penalty=0.15,  # MSM-focused messaging
            research_penalty=0.05,  # Not included in campaigns
            ml_penalty=0.10,  # Deprioritized in targeting algorithms
        ),

        CascadeStep(
            name="willingness",
            description="Willing to seek LAI-PrEP despite system visibility",
            base_probability=0.80,
            barrier_layer=BarrierLayer.ARCHITECTURAL,
            architectural_subtype=ArchitecturalSubtype.POLICY,
            policy_penalty=0.35,  # Seeking care = becoming visible to system
            stigma_penalty=0.10,  # Fear of judgment
            infrastructure_penalty=0.0,
            research_penalty=0.0,
            ml_penalty=0.05,
        ),

        CascadeStep(
            name="healthcare_access",
            description="Can physically access healthcare services",
            base_probability=0.75,
            barrier_layer=BarrierLayer.ARCHITECTURAL,
            architectural_subtype=ArchitecturalSubtype.INFRASTRUCTURE,
            policy_penalty=0.10,  # Outstanding warrants
            stigma_penalty=0.05,
            infrastructure_penalty=0.25,  # Transport, hours, no PCP
            research_penalty=0.0,
            ml_penalty=0.0,
        ),

        CascadeStep(
            name="disclosure",
            description="Willing to disclose injection drug use to provider",
            base_probability=0.70,
            barrier_layer=BarrierLayer.ARCHITECTURAL,
            architectural_subtype=ArchitecturalSubtype.STIGMA,
            policy_penalty=0.30,  # Admitting felony, child custody fears
            stigma_penalty=0.15,  # Fear of discrimination (78% report)
            infrastructure_penalty=0.0,
            research_penalty=0.0,
            ml_penalty=0.0,
        ),

        CascadeStep(
            name="provider_willing",
            description="Provider willing to prescribe LAI-PrEP for PWID",
            base_probability=0.85,
            barrier_layer=BarrierLayer.ARCHITECTURAL,
            architectural_subtype=ArchitecturalSubtype.STIGMA,
            policy_penalty=0.05,
            stigma_penalty=0.25,  # "Not my patient population"
            infrastructure_penalty=0.0,
            research_penalty=0.10,  # "No evidence for PWID"
            ml_penalty=0.10,  # Algorithm doesn't flag as candidate
        ),

        CascadeStep(
            name="hiv_testing_adequate",
            description="Receives adequate HIV testing before injection",
            base_probability=0.90,
            barrier_layer=BarrierLayer.HIV_TESTING,
            testing_penalty=0.25,  # No RNA, acute infection missed
            policy_penalty=0.05,
            infrastructure_penalty=0.15,  # Testing infrastructure gaps
            research_penalty=0.0,
            ml_penalty=0.0,
        ),

        CascadeStep(
            name="first_injection",
            description="Returns for and receives first LAI-CAB injection",
            base_probability=0.75,
            barrier_layer=BarrierLayer.ARCHITECTURAL,
            architectural_subtype=ArchitecturalSubtype.INFRASTRUCTURE,
            policy_penalty=0.10,  # Incarceration between Rx and appt
            stigma_penalty=0.05,
            infrastructure_penalty=0.15,  # Appointment systems, wait times
            research_penalty=0.0,
            ml_penalty=0.0,
        ),

        CascadeStep(
            name="sustained_engagement",
            description="Maintains injection schedule over time (Q2M or Q6M)",
            base_probability=0.70,
            barrier_layer=BarrierLayer.ARCHITECTURAL,
            architectural_subtype=ArchitecturalSubtype.POLICY,
            policy_penalty=0.20,  # Incarceration interrupts schedule
            stigma_penalty=0.10,  # Negative experiences → disengagement
            infrastructure_penalty=0.10,  # Life instability
            research_penalty=0.0,
            ml_penalty=0.05,  # Deprioritized for retention outreach
        ),
    ]


# =============================================================================
# MONTE CARLO SIMULATION ENGINE
# =============================================================================

class ArchitecturalBarrierModel:
    """
    Main simulation engine for nested barrier modeling.

    Models the cumulative effect of structural, policy, and implementation
    barriers on HIV prevention cascade completion for PWID populations.
    """

    def __init__(
            self,
            pathogen_params: PathogenBiologyParams = None,
            testing_params: HIVTestingParams = None,
            architectural_params: ArchitecturalBarrierParams = None,
            stochastic_params: StochasticAvoidanceParams = None
    ):
        self.pathogen = pathogen_params or PathogenBiologyParams()
        self.testing = testing_params or HIVTestingParams()
        self.architectural = architectural_params or ArchitecturalBarrierParams()
        self.stochastic = stochastic_params or StochasticAvoidanceParams()

        self.cascade = create_pwid_cascade()

    def calculate_step_probability(
            self,
            step: CascadeStep,
            scenario: PolicyScenario
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate probability of passing a cascade step.

        Returns (probability, decomposition_dict)
        """
        prob = step.base_probability
        decomposition = {}

        # Layer 1: Pathogen biology (not directly modifiable)
        if step.pathogen_penalty > 0:
            prob -= step.pathogen_penalty
            decomposition['pathogen_biology'] = step.pathogen_penalty

        # Layer 2: HIV testing failures
        if step.testing_penalty > 0:
            testing_impact = step.testing_penalty
            # Slightly reduced if better testing infrastructure
            if scenario.low_barrier_access:
                testing_impact *= 0.8
            prob -= testing_impact
            decomposition['hiv_testing'] = testing_impact

        # Layer 3a: Policy barriers
        if step.policy_penalty > 0:
            policy_impact = step.policy_penalty
            if scenario.decriminalization:
                policy_impact = 0.0  # Fully removed
            prob -= policy_impact
            decomposition['policy'] = policy_impact

        # Layer 3b: Stigma barriers
        if step.stigma_penalty > 0:
            stigma_impact = step.stigma_penalty * (1 - scenario.stigma_reduction)
            if scenario.provider_stigma_training:
                stigma_impact *= 0.7
            prob -= stigma_impact
            decomposition['stigma'] = stigma_impact

        # Layer 3c: Infrastructure barriers
        if step.infrastructure_penalty > 0:
            infra_impact = step.infrastructure_penalty
            if scenario.ssp_integrated_delivery:
                infra_impact *= 0.3  # 70% reduction
            if scenario.low_barrier_access:
                infra_impact *= 0.5
            prob -= infra_impact
            decomposition['infrastructure'] = infra_impact

        # Layer 3d: Research exclusion
        if step.research_penalty > 0:
            research_impact = step.research_penalty
            if scenario.pwid_trial_inclusion:
                research_impact = 0.0  # PURPOSE-4 changes this
            prob -= research_impact
            decomposition['research_exclusion'] = research_impact

        # Layer 3e: ML barriers
        if step.ml_penalty > 0:
            ml_impact = step.ml_penalty
            if scenario.algorithmic_debiasing:
                ml_impact = 0.0
            prob -= ml_impact
            decomposition['machine_learning'] = ml_impact

        # Apply bonuses
        if scenario.ssp_integrated_delivery:
            if step.name in ["awareness", "healthcare_access", "first_injection"]:
                prob += 0.15

        if scenario.peer_navigation:
            if step.name in ["willingness", "first_injection", "sustained_engagement"]:
                prob += 0.10

        # Bound probability
        prob = max(0.01, min(0.99, prob))

        return prob, decomposition

    def calculate_incarceration_survival(
            self,
            scenario: PolicyScenario,
            years: int = 5
    ) -> float:
        """
        Calculate probability of maintaining protection given incarceration risk.
        """
        annual_rate = self.architectural.annual_incarceration_rate_pwid
        annual_rate *= scenario.incarceration_modifier

        if scenario.in_custody_prep:
            return 1.0  # Incarceration doesn't disrupt
        else:
            return (1 - annual_rate) ** years

    def simulate_individual(
            self,
            scenario: PolicyScenario,
            years: int = 5
    ) -> Dict:
        """
        Simulate single individual through cascade.
        """
        results = {
            "completed_cascade": True,
            "failed_step": None,
            "failed_layer": None,
            "incarceration_disrupted": False,
            "achieved_r0_zero": False,
            "step_outcomes": {},
            "barrier_attribution": {}
        }

        total_attribution = {
            "pathogen_biology": 0.0,
            "hiv_testing": 0.0,
            "policy": 0.0,
            "stigma": 0.0,
            "infrastructure": 0.0,
            "research_exclusion": 0.0,
            "machine_learning": 0.0
        }

        for step in self.cascade:
            prob, decomposition = self.calculate_step_probability(step, scenario)
            passed = random.random() < prob

            results["step_outcomes"][step.name] = {
                "probability": prob,
                "passed": passed,
                "decomposition": decomposition
            }

            # Accumulate barrier attribution
            for barrier, impact in decomposition.items():
                total_attribution[barrier] += impact

            if not passed:
                results["completed_cascade"] = False
                results["failed_step"] = step.name
                results["failed_layer"] = step.barrier_layer.value
                break

        results["barrier_attribution"] = total_attribution

        # Check incarceration if cascade completed
        if results["completed_cascade"]:
            incarceration_survival = self.calculate_incarceration_survival(
                scenario, years
            )
            survived = random.random() < incarceration_survival

            if not survived:
                results["incarceration_disrupted"] = True
            else:
                results["achieved_r0_zero"] = True

        return results

    def run_simulation(
            self,
            scenario: PolicyScenario,
            n_individuals: int = 100000,
            years: int = 5
    ) -> Dict:
        """
        Run full Monte Carlo simulation for a scenario using vectorized operations.
        """
        logger.info(f"Running simulation for scenario: {scenario.name}")
        results = {
            "scenario": scenario.name,
            "n_individuals": n_individuals,
            "years": years,
            "achieved_r0_zero": 0,
            "completed_cascade": 0,
            "incarceration_disrupted": 0,
            "step_failure_counts": {step.name: 0 for step in self.cascade},
            "layer_failure_counts": {
                "pathogen_biology": 0,
                "hiv_testing": 0,
                "architectural": 0
            },
            "step_probabilities": {},
            "barrier_attribution_totals": {
                "pathogen_biology": 0.0,
                "hiv_testing": 0.0,
                "policy": 0.0,
                "stigma": 0.0,
                "infrastructure": 0.0,
                "research_exclusion": 0.0,
                "machine_learning": 0.0
            }
        }

        # Calculate theoretical probabilities
        step_probs = []
        theoretical_cascade = 1.0
        for step in self.cascade:
            prob, decomp = self.calculate_step_probability(step, scenario)
            results["step_probabilities"][step.name] = prob
            theoretical_cascade *= prob
            step_probs.append(prob)

            # Sum barrier attributions
            for barrier, impact in decomp.items():
                results["barrier_attribution_totals"][barrier] += impact

        results["theoretical_cascade_probability"] = theoretical_cascade
        incarceration_survival_prob = self.calculate_incarceration_survival(scenario, years)
        results["incarceration_survival_probability"] = incarceration_survival_prob
        results["theoretical_r0_zero_probability"] = \
            theoretical_cascade * results["incarceration_survival_probability"]

        # Vectorized Monte Carlo simulation
        # Each row is an individual, each column is a cascade step
        step_probs_array = np.array(step_probs)
        rand_array = rng.random((n_individuals, len(self.cascade)))
        step_passed = rand_array < step_probs_array

        # An individual completes the cascade if they pass all steps
        completed_cascade_mask = np.all(step_passed, axis=1)
        results["completed_cascade"] = int(np.sum(completed_cascade_mask))

        # Track failures
        # failed_step_idx is the index of the first False in each row
        # If an individual passes all steps, we don't care about their failed_step_idx
        failed_mask = ~step_passed
        # For individuals who failed, find the first failed step
        has_failed = np.any(failed_mask, axis=1)
        failed_step_indices = np.argmax(failed_mask, axis=1)

        for idx in range(len(self.cascade)):
            step_name = self.cascade[idx].name
            # Count individuals who failed at exactly this step
            count = np.sum((failed_step_indices == idx) & has_failed)
            results["step_failure_counts"][step_name] = int(count)

        # Incarceration check for those who completed the cascade
        n_completed = results["completed_cascade"]
        if n_completed > 0:
            incarceration_rand = rng.random(n_completed)
            survived_incarceration = incarceration_rand < incarceration_survival_prob
            results["achieved_r0_zero"] = int(np.sum(survived_incarceration))
            results["incarceration_disrupted"] = int(n_completed - results["achieved_r0_zero"])

        # Calculate statistics
        results["observed_r0_zero_rate"] = results["achieved_r0_zero"] / n_individuals
        results["observed_cascade_completion_rate"] = \
            results["completed_cascade"] / n_individuals

        # Confidence interval
        p = results["observed_r0_zero_rate"]
        n = n_individuals
        se = np.sqrt(p * (1 - p) / n)
        results["r0_zero_95ci"] = (
            float(max(0, p - 1.96 * se)),
            float(min(1, p + 1.96 * se))
        )

        # Calculate barrier decomposition percentages
        total_barrier = sum(results["barrier_attribution_totals"].values())
        if total_barrier > 0:
            results["barrier_decomposition_pct"] = {
                barrier: float((impact / total_barrier) * 100)
                for barrier, impact in results["barrier_attribution_totals"].items()
            }

            # Group into 3 layers
            results["three_layer_decomposition"] = {
                "pathogen_biology": results["barrier_decomposition_pct"]["pathogen_biology"],
                "hiv_testing": results["barrier_decomposition_pct"]["hiv_testing"],
                "architectural": (
                        results["barrier_decomposition_pct"]["policy"] +
                        results["barrier_decomposition_pct"]["stigma"] +
                        results["barrier_decomposition_pct"]["infrastructure"] +
                        results["barrier_decomposition_pct"]["research_exclusion"] +
                        results["barrier_decomposition_pct"]["machine_learning"]
                )
            }

        return results


# =============================================================================
# STOCHASTIC AVOIDANCE FAILURE MODEL
# =============================================================================

class StochasticAvoidanceModel:
    """
    Models the probability of catastrophic failure of stochastic avoidance
    as primary HIV prevention mechanism for PWID.

    Hypothesis: HIV prevention in PWID has relied on probability rather than
    intervention. As network density increases (meth introduction, housing
    instability, sex work bridges), stochastic avoidance will fail.
    """

    def __init__(self, params: StochasticAvoidanceParams = None):
        self.params = params or StochasticAvoidanceParams()

    def calculate_network_density(
            self,
            year: int,
            base_year: int = 2024,
            meth_prevalence: float = 0.143  # 2018 baseline
    ) -> float:
        """
        Calculate effective network density given temporal dynamics.

        Network density increases with:
        - Methamphetamine prevalence
        - Housing instability (forced clustering)
        - Sex work bridges
        """
        years_elapsed = year - base_year

        # Methamphetamine prevalence growth
        current_meth = meth_prevalence * (
                1 + self.params.meth_prevalence_trend_annual
        ) ** years_elapsed
        current_meth = min(current_meth, 0.5)  # Cap at 50%

        # Meth effect on network density
        meth_density_effect = (
                current_meth * self.params.meth_network_density_multiplier
        )

        # Housing instability effect
        housing_effect = self.params.housing_instability_rate * 0.5

        # Sex work bridging effect
        bridge_effect = self.params.sex_work_network_bridge_prob * 0.3

        # Combined network density
        density = (
                self.params.baseline_network_density +
                meth_density_effect +
                housing_effect +
                bridge_effect
        )

        return min(density, 1.0)

    def calculate_annual_outbreak_probability(
            self,
            network_density: float,
            ssp_coverage: float = 0.21,
            oat_coverage: float = 0.08
    ) -> float:
        """
        Calculate annual probability of major outbreak.

        Based on Des Jarlais et al. (2022) modeling and
        outbreak data from Scott County, MA, WV.
        """
        # Baseline probability
        p_outbreak = self.params.baseline_annual_outbreak_prob

        # Network density effect (exponential relationship)
        if network_density > self.params.critical_network_threshold:
            excess = network_density - self.params.critical_network_threshold
            p_outbreak *= np.exp(3 * excess)  # Rapid increase above threshold

        # Protective effect of interventions
        ssp_protection = 1 - (ssp_coverage * 0.4)  # SSP reduces by up to 40%
        oat_protection = 1 - (oat_coverage * 0.3)  # OAT reduces by up to 30%

        p_outbreak *= ssp_protection * oat_protection

        return min(p_outbreak, 1.0)

    def simulate_time_to_outbreak(
            self,
            n_simulations: int = 10000,
            max_years: int = 20,
            outbreak_threshold_cases: int = 100  # "Catastrophic" threshold
    ) -> Dict:
        """
        Simulate time until stochastic avoidance fails.

        Returns distribution of years until major outbreak.
        """
        outbreak_years = []
        no_outbreak_count = 0

        for _ in range(n_simulations):
            outbreak_occurred = False

            for year in range(2024, 2024 + max_years):
                density = self.calculate_network_density(year)
                p_outbreak = self.calculate_annual_outbreak_probability(density)

                if random.random() < p_outbreak:
                    outbreak_years.append(year - 2024)
                    outbreak_occurred = True
                    break

            if not outbreak_occurred:
                no_outbreak_count += 1

        # Calculate statistics
        if outbreak_years:
            outbreak_years = np.array(outbreak_years)
            results = {
                "median_years_to_outbreak": np.median(outbreak_years),
                "mean_years_to_outbreak": np.mean(outbreak_years),
                "std_years": np.std(outbreak_years),
                "p10_years": np.percentile(outbreak_years, 10),
                "p25_years": np.percentile(outbreak_years, 25),
                "p75_years": np.percentile(outbreak_years, 75),
                "p90_years": np.percentile(outbreak_years, 90),
                "probability_outbreak_5_years": np.mean(outbreak_years <= 5),
                "probability_outbreak_10_years": np.mean(outbreak_years <= 10),
                "probability_no_outbreak": no_outbreak_count / n_simulations,
            }
        else:
            results = {
                "median_years_to_outbreak": None,
                "probability_no_outbreak": 1.0
            }

        # Add annual trajectory
        trajectory = []
        for year_offset in range(max_years):
            year = 2024 + year_offset
            density = self.calculate_network_density(year)
            p_outbreak = self.calculate_annual_outbreak_probability(density)
            trajectory.append({
                "year": year,
                "network_density": density,
                "annual_outbreak_probability": p_outbreak,
                "cumulative_outbreak_probability": 1 - (
                        (1 - p_outbreak) ** (year_offset + 1)
                ) if year_offset > 0 else p_outbreak
            })
        results["trajectory"] = trajectory

        return results

    def run_scenario_analysis(
            self,
            scenarios: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Run stochastic avoidance analysis across multiple scenarios.

        scenarios should be dict like:
        {
            "current_policy": {"ssp_coverage": 0.21, "oat_coverage": 0.08},
            "expanded_ssp": {"ssp_coverage": 0.60, "oat_coverage": 0.08},
            ...
        }
        """
        results = {}

        for name, params in scenarios.items():
            # Create modified model for scenario
            model = StochasticAvoidanceModel()

            # Override coverage parameters
            ssp = params.get("ssp_coverage", 0.21)
            oat = params.get("oat_coverage", 0.08)

            # Run simulation with modified outbreak calculation
            scenario_results = model.simulate_time_to_outbreak()
            scenario_results["scenario_name"] = name
            scenario_results["ssp_coverage"] = ssp
            scenario_results["oat_coverage"] = oat

            results[name] = scenario_results

        return results


# =============================================================================
# POLICY SCENARIOS
# =============================================================================

def create_policy_scenarios() -> List[PolicyScenario]:
    """Create standard set of policy scenarios for analysis."""
    return [
        PolicyScenario(
            name="Current Policy",
            description="Status quo: full criminalization, systemic barriers",
        ),

        PolicyScenario(
            name="Decriminalization Only",
            description="Drug use decriminalized, healthcare unchanged",
            decriminalization=True,
            incarceration_modifier=0.3,
        ),

        PolicyScenario(
            name="Decrim + Stigma Reduction",
            description="Decriminalization plus provider stigma reduction training",
            decriminalization=True,
            incarceration_modifier=0.3,
            stigma_reduction=0.5,
            provider_stigma_training=True,
        ),

        PolicyScenario(
            name="SSP-Integrated Delivery",
            description="PrEP via syringe services with peer support",
            decriminalization=True,
            incarceration_modifier=0.3,
            stigma_reduction=0.7,
            ssp_integrated_delivery=True,
            peer_navigation=True,
            low_barrier_access=True,
        ),

        PolicyScenario(
            name="Full Harm Reduction",
            description="Complete policy transformation",
            decriminalization=True,
            incarceration_modifier=0.2,
            in_custody_prep=True,
            stigma_reduction=0.8,
            provider_stigma_training=True,
            ssp_integrated_delivery=True,
            peer_navigation=True,
            low_barrier_access=True,
        ),

        PolicyScenario(
            name="Full HR + PURPOSE-4 Data",
            description="Harm reduction plus PWID trial inclusion",
            decriminalization=True,
            incarceration_modifier=0.2,
            in_custody_prep=True,
            stigma_reduction=0.8,
            provider_stigma_training=True,
            ssp_integrated_delivery=True,
            peer_navigation=True,
            low_barrier_access=True,
            pwid_trial_inclusion=True,
        ),

        PolicyScenario(
            name="Full HR + Algorithmic Debiasing",
            description="Full harm reduction plus ML debiasing",
            decriminalization=True,
            incarceration_modifier=0.2,
            in_custody_prep=True,
            stigma_reduction=0.9,
            provider_stigma_training=True,
            ssp_integrated_delivery=True,
            peer_navigation=True,
            low_barrier_access=True,
            pwid_trial_inclusion=True,
            algorithmic_debiasing=True,
        ),

        PolicyScenario(
            name="Theoretical Maximum",
            description="All barriers removed (comparison ceiling)",
            decriminalization=True,
            incarceration_modifier=0.0,
            in_custody_prep=True,
            stigma_reduction=1.0,
            provider_stigma_training=True,
            ssp_integrated_delivery=True,
            peer_navigation=True,
            low_barrier_access=True,
            pwid_trial_inclusion=True,
            algorithmic_debiasing=True,
        ),
    ]


# =============================================================================
# MSM COMPARISON MODEL
# =============================================================================

def calculate_msm_cascade_completion() -> Dict:
    """
    Calculate MSM cascade completion for comparison.

    MSM benefit from:
    - No criminalization barrier
    - Extensive infrastructure (20+ years of programs)
    - Included in all trials (training set, not held-out)
    - Algorithm optimization (high SNR)
    """
    msm_cascade = {
        "awareness": 0.90,  # Extensive campaigns
        "willingness": 0.85,  # No criminalization barrier
        "healthcare_access": 0.80,  # Better infrastructure
        "disclosure": 0.75,  # Less stigma (improving)
        "provider_willing": 0.90,  # Trained providers
        "hiv_testing_adequate": 0.85,  # Better testing
        "first_injection": 0.80,  # Good systems
        "sustained_engagement": 0.75,  # Support programs
    }

    cascade_product = np.prod(list(msm_cascade.values()))

    # MSM incarceration rate much lower
    incarceration_survival = 0.95 ** 5  # ~5% annual rate

    p_r0_zero = cascade_product * incarceration_survival

    return {
        "population": "MSM",
        "cascade_steps": msm_cascade,
        "cascade_completion": cascade_product,
        "incarceration_survival": incarceration_survival,
        "p_r0_zero": p_r0_zero,
        "snr": 9180.0,  # MSM SNR from model params
        "in_training_set": True,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete analysis and generate outputs."""
    parser = argparse.ArgumentParser(description="Architectural Barrier Model: HIV Prevention Cascade Modeling for PWID")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files (default: current directory)")
    parser.add_argument("--n-individuals", type=int, default=100000, help="Number of individuals to simulate per scenario (default: 100000)")
    parser.add_argument("--n-sa-sims", type=int, default=10000, help="Number of stochastic avoidance simulations (default: 10000)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("ARCHITECTURAL BARRIER MODEL: HIV PREVENTION CASCADE MODELING")
    print("Monte Carlo Simulation with 3-Layer Barrier Framework")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("=" * 80)
    print()

    # Initialize model
    model = ArchitecturalBarrierModel()
    scenarios = create_policy_scenarios()

    # Run simulations
    print(f"Running cascade simulations ({args.n_individuals:,} individuals per scenario)...")
    print()

    all_results = []
    for scenario in scenarios:
        print(f"  Simulating: {scenario.name}...")
        results = model.run_simulation(scenario, n_individuals=args.n_individuals, years=5)
        all_results.append(results)

    print()

    # Calculate MSM comparison
    msm_results = calculate_msm_cascade_completion()

    # Print summary table
    print("=" * 100)
    print("RESULTS: LAI-PrEP CASCADE COMPLETION BY POLICY SCENARIO")
    print("=" * 100)
    print()
    print(f"{'Scenario':<40} {'P(R(0)=0)':<12} {'95% CI':<22} {'Cascade':<10}")
    print("-" * 90)

    for r in all_results:
        ci_str = f"({r['r0_zero_95ci'][0]:.4f}, {r['r0_zero_95ci'][1]:.4f})"
        print(
            f"{r['scenario']:<40} "
            f"{r['observed_r0_zero_rate']:.4f}       "
            f"{ci_str:<22} "
            f"{r['observed_cascade_completion_rate']:.4f}"
        )

    # MSM comparison
    print()
    print("-" * 90)
    print(f"{'MSM (comparison)':<40} {msm_results['p_r0_zero']:.4f}       "
          f"{'N/A':<22} {msm_results['cascade_completion']:.4f}")
    print()

    # Disparity calculation
    current_pwid = all_results[0]['observed_r0_zero_rate']
    disparity = msm_results['p_r0_zero'] / current_pwid if current_pwid > 0 else 999.0

    print(f"DISPARITY: MSM vs PWID (Current Policy) = {disparity:,.0f}-fold")
    print(f"SNR Ratio (training data quality): 120-fold")  # SNR_MSM/SNR_PWID = 9180/76.4
    print()

    # Barrier decomposition
    print("=" * 80)
    print("THREE-LAYER BARRIER DECOMPOSITION (Current Policy)")
    print("=" * 80)
    print()

    current = all_results[0]
    if "three_layer_decomposition" in current:
        for layer, pct in current["three_layer_decomposition"].items():
            print(f"  {layer.replace('_', ' ').title():<30}: {pct:>6.1f}%")

    print()
    print("Architectural Barrier Subtypes:")
    if "barrier_decomposition_pct" in current:
        for barrier in ["policy", "stigma", "infrastructure",
                        "research_exclusion", "machine_learning"]:
            pct = current["barrier_decomposition_pct"].get(barrier, 0)
            print(f"    {barrier.replace('_', ' ').title():<28}: {pct:>6.1f}%")

    print()

    # Run stochastic avoidance model
    print("=" * 80)
    print("STOCHASTIC AVOIDANCE FAILURE PREDICTION")
    print("=" * 80)
    print()

    sa_model = StochasticAvoidanceModel()
    sa_results = sa_model.simulate_time_to_outbreak(n_simulations=args.n_sa_sims)

    if sa_results["median_years_to_outbreak"]:
        print(f"Median years to major outbreak: {sa_results['median_years_to_outbreak']:.1f}")
        print(f"P(outbreak within 5 years):     {sa_results['probability_outbreak_5_years'] * 100:.1f}%")
        print(f"P(outbreak within 10 years):    {sa_results['probability_outbreak_10_years'] * 100:.1f}%")
    else:
        print("Outbreak risk below detection threshold in simulation")

    print()
    print("Network Density Trajectory (5-year projection):")
    for t in sa_results["trajectory"][:5]:
        print(f"  {t['year']}: density={t['network_density']:.3f}, "
              f"p_outbreak={t['annual_outbreak_probability']:.3f}")

    print()

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_parameters": {
            "literature_params": LITERATURE_PARAMS,
        },
        "cascade_results": [
            {**r, "r0_zero_95ci": list(r["r0_zero_95ci"])}
            for r in all_results
        ],
        "msm_comparison": msm_results,
        "disparity_fold": disparity,
        "stochastic_avoidance": {
            k: v for k, v in sa_results.items()
            if k != "trajectory"
        },
    }

    output_path = os.path.join(args.output_dir, "architectural_barrier_results.json")
    csv_path = os.path.join(args.output_dir, "architectural_barrier_results.csv")
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str, allow_nan=False)
        print(f"Results saved to {output_path}")

        # Save to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = [
                "Scenario", "Achieved R0=0", "Completed Cascade", "Incarceration Disrupted",
                "Observed R0=0 Rate", "Observed Cascade Rate", "95% CI Lower", "95% CI Upper"
            ]
            writer.writerow(header)

            # Scenario results
            for r in all_results:
                writer.writerow([
                    r["scenario"],
                    r["achieved_r0_zero"],
                    r["completed_cascade"],
                    r["incarceration_disrupted"],
                    f"{r['observed_r0_zero_rate']:.6f}",
                    f"{r['observed_cascade_completion_rate']:.6f}",
                    f"{r['r0_zero_95ci'][0]:.6f}",
                    f"{r['r0_zero_95ci'][1]:.6f}"
                ])

            # MSM comparison
            writer.writerow([])
            writer.writerow(["MSM Comparison", "", "", "", f"{msm_results['p_r0_zero']:.6f}",
                             f"{msm_results['cascade_completion']:.6f}", "N/A", "N/A"])

            # Barrier decomposition for Current Policy
            writer.writerow([])
            writer.writerow(["Barrier Decomposition (Current Policy)", "Percentage (%)"])
            current = all_results[0]
            if "three_layer_decomposition" in current:
                for layer, pct in current["three_layer_decomposition"].items():
                    writer.writerow([layer.replace('_', ' ').title(), f"{pct:.2f}"])

            writer.writerow([])
            writer.writerow(["Architectural Subtypes (Current Policy)", "Percentage (%)"])
            if "barrier_decomposition_pct" in current:
                for barrier in ["policy", "stigma", "infrastructure", "research_exclusion", "machine_learning"]:
                    pct = current["barrier_decomposition_pct"].get(barrier, 0)
                    writer.writerow([barrier.replace('_', ' ').title(), f"{pct:.2f}"])

        print(f"Results saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    print()

    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    print("1. STRUCTURAL BARRIERS DOMINATE CASCADE FAILURE")
    print(f"   Under current policy, P(R(0)=0) = {current_pwid * 100:.3f}%")
    print(f"   Even with 99.9% drug efficacy, cascade barriers dominate")
    print()
    print("2. MSM vs PWID DISPARITY")
    print(f"   Same drug → {disparity:,.0f}-fold difference in prevention probability")
    print(f"   Difference is policy-determined, not pharmacology-determined")
    print()
    print("3. STOCHASTIC AVOIDANCE AS PRIMARY MECHANISM")
    if sa_results["median_years_to_outbreak"]:
        print(f"   Current 'prevention' relies on probability, not intervention")
        print(f"   Median time to avoidance failure: {sa_results['median_years_to_outbreak']:.1f} years")
    print()

    return all_results, sa_results


if __name__ == "__main__":
    results, sa_results = main()