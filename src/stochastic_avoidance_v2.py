#!/usr/bin/env python3
"""
Stochastic Avoidance Model V2: Multiplicative Meth×Housing Interaction
=======================================================================

Extends the enhanced stochastic avoidance model with a structural improvement
identified through Hood et al. (2018) parameter comparison:

**Multiplicative meth×housing interaction** in network density calculation.
Hood shows joint meth+unstable housing effect is ~1.5x larger than the sum
of individual effects (42% suppression vs 76% baseline = 45% reduction,
while individual effects sum to only ~31%).

Note: TasP is NOT modeled separately because the model's hiv_prevalence_pwid
inputs already reflect current viral suppression rates among PWID on ART.
Adding a separate TasP discount would double-count this effect.

Design: Subclasses original model. Original source files remain untouched.

Author: AC Demidont, MD / Nyx Dynamics LLC
Date: January 2026
"""

import sys
import os
import numpy as np
import json
import csv
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from stochastic_avoidance_enhanced import (
    ParameterWithUncertainty,
    KEY_PARAMETERS,
    REGIONAL_PROFILES,
    RegionalContextualFailureDriverProfile,
    EnhancedStochasticAvoidanceModel,
    SensitivityAnalyzer,
    rng,
    save_publication_fig,
    WIDTH_DOUBLE,
    MM_TO_INCH,
)

# =============================================================================
# V2 PARAMETERS: Original + meth×housing interaction parameter
# =============================================================================

KEY_PARAMETERS_V2 = {}

# Copy all original parameters
for _name, _param in KEY_PARAMETERS.items():
    KEY_PARAMETERS_V2[_name] = ParameterWithUncertainty(
        name=_param.name,
        point_estimate=_param.point_estimate,
        lower_bound=_param.lower_bound,
        upper_bound=_param.upper_bound,
        distribution=_param.distribution,
        source=_param.source,
    )

# --- New V2 parameter ---

KEY_PARAMETERS_V2["meth_housing_interaction"] = ParameterWithUncertainty(
    name="Meth x housing interaction coefficient",
    point_estimate=0.8,
    lower_bound=0.3,
    upper_bound=1.5,
    distribution="uniform",
    source="Hood 2018: meth x housing joint effect 1.5x > sum"
)


# =============================================================================
# V2 MODEL: Subclass with multiplicative meth×housing interaction
# =============================================================================

class EnhancedStochasticAvoidanceModelV2(EnhancedStochasticAvoidanceModel):
    """
    V2 model with multiplicative meth x housing interaction in network density.

    Hood 2018 shows meth + unstable housing jointly suppress viral suppression
    to 42% (vs 76% baseline), implying the combined network/risk effect is
    ~1.5x larger than the sum of individual effects. The V1 model treats these
    additively; V2 adds an interaction term.

    Outbreak probability uses the parent's implementation unchanged — HIV
    prevalence inputs already reflect current viral suppression among PWID
    on ART, so no separate TasP adjustment is needed.

    When meth_housing_interaction=0, reproduces V1 outputs (backward compatible).
    """

    def __init__(
        self,
        region: str = "national_average",
        params: Dict[str, ParameterWithUncertainty] = None
    ):
        super().__init__(region=region, params=params or KEY_PARAMETERS_V2)

    def calculate_network_density(
        self,
        year: int,
        meth_prevalence: float = None,
        housing_instability: float = None,
        incarceration_rate: float = None
    ) -> float:
        """
        Calculate effective network density with multiplicative meth x housing interaction.

        V2 addition: interaction term = meth_prevalence * housing_instability * interaction_coeff
        Hood 2018 shows meth + unstable housing jointly suppress viral suppression
        to 42% (vs 76% baseline), implying the combined network/risk effect is
        ~1.5x larger than the sum of individual effects.
        """
        if meth_prevalence is None:
            meth_prevalence = self.profile.prevalence_at_year(year)
        if housing_instability is None:
            housing_instability = self.params["housing_instability_rate"].point_estimate
        if incarceration_rate is None:
            incarceration_rate = self.params["incarceration_annual_rate"].point_estimate

        baseline = self.params["baseline_network_density"].point_estimate
        meth_multiplier = self.params["meth_network_multiplier"].point_estimate

        # Meth effect (same as V1)
        meth_effect = meth_prevalence * meth_multiplier * 0.5

        # Housing effect (same as V1)
        housing_effect = housing_instability * 0.3

        # Incarceration effect (same as V1)
        incarceration_effect = incarceration_rate * 0.2

        # Sex work bridging (same as V1)
        sex_work_bridge = meth_prevalence * 0.15

        # V2: Multiplicative interaction term
        interaction_coeff = self.params.get(
            "meth_housing_interaction",
            ParameterWithUncertainty("interaction", 0.0, 0.0, 0.0)
        ).point_estimate
        interaction = meth_prevalence * housing_instability * interaction_coeff

        total_density = (
            baseline + meth_effect + housing_effect
            + incarceration_effect + sex_work_bridge + interaction
        )

        return min(total_density, 1.0)

    def simulate_trajectory(
        self,
        start_year: int = 2024,
        end_year: int = 2040,
        n_simulations: int = 1000,
        include_uncertainty: bool = True
    ) -> Dict:
        """
        Simulate outbreak trajectories with V2 structural improvements.

        Changes from V1 vectorized code:
        - Uses exponential threshold logic (matching scalar calculate_outbreak_probability)
        - Adds meth x housing interaction term to vectorized density
        """
        years = list(range(start_year, end_year + 1))
        n_years = len(years)
        years_array = np.array(years)

        logger.info(f"Simulating {n_simulations} V2 trajectories for region: {self.region}")

        # Sample parameters for all simulations at once
        if include_uncertainty:
            meth_growth = self.params["meth_annual_growth_rate"].sample(n_simulations)
            baseline_density = self.params["baseline_network_density"].sample(n_simulations)
            meth_mult = self.params["meth_network_multiplier"].sample(n_simulations)
            housing = self.params["housing_instability_rate"].sample(n_simulations)
            incarc = self.params["incarceration_annual_rate"].sample(n_simulations)
            ssp = self.params["ssp_coverage"].sample(n_simulations)
            oat = self.params["oat_coverage"].sample(n_simulations)
            # V2 interaction parameter
            if "meth_housing_interaction" in self.params:
                interaction_coeff = self.params["meth_housing_interaction"].sample(n_simulations)
            else:
                interaction_coeff = np.zeros(n_simulations)
        else:
            meth_growth = np.full(n_simulations, self.params["meth_annual_growth_rate"].point_estimate)
            baseline_density = np.full(n_simulations, self.params["baseline_network_density"].point_estimate)
            meth_mult = np.full(n_simulations, self.params["meth_network_multiplier"].point_estimate)
            housing = np.full(n_simulations, self.params["housing_instability_rate"].point_estimate)
            incarc = np.full(n_simulations, self.params["incarceration_annual_rate"].point_estimate)
            ssp = np.full(n_simulations, self.params["ssp_coverage"].point_estimate)
            oat = np.full(n_simulations, self.params["oat_coverage"].point_estimate)
            # V2 interaction parameter
            if "meth_housing_interaction" in self.params:
                interaction_coeff = np.full(n_simulations, self.params["meth_housing_interaction"].point_estimate)
            else:
                interaction_coeff = np.zeros(n_simulations)

        # Vectorized calculations for meth_prevalence
        # Shape: (n_simulations, n_years)
        years_from_baseline = years_array - 2018
        meth_prev = self.profile.baseline_prevalence_2018 * (
            (1 + meth_growth[:, np.newaxis]) ** years_from_baseline
        )
        meth_prev = np.minimum(meth_prev, 0.60)

        # Calculate network density (V2: with interaction term)
        meth_effect = meth_prev * meth_mult[:, np.newaxis] * 0.5
        housing_effect = housing[:, np.newaxis] * 0.3
        incarc_effect = incarc[:, np.newaxis] * 0.2
        sex_work = meth_prev * 0.15

        # V2: Multiplicative interaction term
        interaction_term = meth_prev * housing[:, np.newaxis] * interaction_coeff[:, np.newaxis]

        density = (
            baseline_density[:, np.newaxis]
            + meth_effect + housing_effect + incarc_effect + sex_work
            + interaction_term
        )
        density = np.minimum(density, 1.0)

        # Outbreak probability (V2: exponential threshold matching scalar method)
        baseline_prob = self.params["baseline_outbreak_prob"].point_estimate
        threshold = self.params["critical_network_threshold"].point_estimate

        # Vectorized exponential threshold logic (matching scalar method)
        excess = density - threshold
        density_multiplier = np.where(
            density > threshold,
            np.exp(3 * excess),
            density / threshold
        )

        # HIV prevalence effect (already reflects current viral suppression)
        prevalence_pwid = self.profile.hiv_prevalence_pwid
        prevalence_multiplier = 1.0 + (prevalence_pwid / 0.10)

        ssp_protection = 1.0 - (ssp[:, np.newaxis] * 0.4)
        oat_protection = 1.0 - (oat[:, np.newaxis] * 0.3)

        p_outbreak_matrix = (
            baseline_prob * density_multiplier * prevalence_multiplier
            * ssp_protection * oat_protection
        )
        p_outbreak_matrix = np.minimum(p_outbreak_matrix, 1.0)

        # Cumulative probability
        no_outbreak_prob_matrix = 1.0 - p_outbreak_matrix
        cum_no_outbreak = np.cumprod(no_outbreak_prob_matrix, axis=1)
        cumulative_outbreak_prob = 1.0 - cum_no_outbreak

        # Outbreak years (Monte Carlo)
        rand_matrix = rng.random((n_simulations, n_years))
        outbreak_mask = rand_matrix < p_outbreak_matrix

        # Find first outbreak year for each simulation
        has_outbreak = np.any(outbreak_mask, axis=1)
        first_outbreak_idx = np.argmax(outbreak_mask, axis=1)

        outbreak_years = []
        for sim in range(n_simulations):
            if has_outbreak[sim]:
                outbreak_years.append(int(years_array[first_outbreak_idx[sim]]))
            else:
                outbreak_years.append(None)

        trajectories = {
            "meth_prevalence": meth_prev,
            "network_density": density,
            "outbreak_probability": p_outbreak_matrix,
            "cumulative_outbreak_prob": cumulative_outbreak_prob,
        }

        # Calculate summary statistics
        valid_outbreak_years = [y for y in outbreak_years if y is not None]

        results = {
            "region": self.region,
            "years": years,
            "trajectories": trajectories,
            "outbreak_years": outbreak_years,
            "summary": {
                "n_simulations": n_simulations,
                "n_outbreaks": len(valid_outbreak_years),
                "outbreak_rate": len(valid_outbreak_years) / n_simulations,
            }
        }

        if valid_outbreak_years:
            outbreak_times = np.array([y - start_year for y in valid_outbreak_years])
            results["summary"].update({
                "median_years_to_outbreak": float(np.median(outbreak_times)),
                "mean_years_to_outbreak": float(np.mean(outbreak_times)),
                "std_years": float(np.std(outbreak_times)),
                "p10_years": float(np.percentile(outbreak_times, 10)),
                "p25_years": float(np.percentile(outbreak_times, 25)),
                "p75_years": float(np.percentile(outbreak_times, 75)),
                "p90_years": float(np.percentile(outbreak_times, 90)),
                "p_outbreak_5yr": float(np.mean(outbreak_times <= 5)),
                "p_outbreak_10yr": float(np.mean(outbreak_times <= 10)),
            })

        # Calculate trajectory statistics
        results["trajectory_stats"] = {
            "meth_prevalence": {
                "mean": np.mean(trajectories["meth_prevalence"], axis=0).tolist(),
                "p5": np.percentile(trajectories["meth_prevalence"], 5, axis=0).tolist(),
                "p95": np.percentile(trajectories["meth_prevalence"], 95, axis=0).tolist(),
            },
            "network_density": {
                "mean": np.mean(trajectories["network_density"], axis=0).tolist(),
                "p5": np.percentile(trajectories["network_density"], 5, axis=0).tolist(),
                "p95": np.percentile(trajectories["network_density"], 95, axis=0).tolist(),
            },
            "cumulative_outbreak_prob": {
                "mean": np.mean(trajectories["cumulative_outbreak_prob"], axis=0).tolist(),
                "p5": np.percentile(trajectories["cumulative_outbreak_prob"], 5, axis=0).tolist(),
                "p95": np.percentile(trajectories["cumulative_outbreak_prob"], 95, axis=0).tolist(),
            },
        }

        return results


# =============================================================================
# V2 SENSITIVITY ANALYZER
# =============================================================================

class SensitivityAnalyzerV2(SensitivityAnalyzer):
    """
    Sensitivity analyzer using V2 model with meth x housing interaction.
    Adds the interaction parameter to tornado analysis.
    """

    def __init__(self, base_model: EnhancedStochasticAvoidanceModelV2 = None):
        self.base_model = base_model or EnhancedStochasticAvoidanceModelV2()
        self.params = self.base_model.params

    def one_way_sensitivity(
        self,
        param_name: str,
        n_points: int = 20,
        outcome: str = "p_outbreak_5yr"
    ) -> Dict:
        """One-way sensitivity using V2 model."""
        param = self.params[param_name]
        values = np.linspace(param.lower_bound, param.upper_bound, n_points)
        outcomes = []

        for val in values:
            modified_params = {k: v for k, v in self.params.items()}
            modified_params[param_name] = ParameterWithUncertainty(
                name=param.name,
                point_estimate=val,
                lower_bound=val,
                upper_bound=val,
                distribution="uniform"
            )

            model = EnhancedStochasticAvoidanceModelV2(params=modified_params)
            results = model.simulate_trajectory(n_simulations=500, include_uncertainty=False)

            if outcome in results["summary"]:
                outcomes.append(results["summary"][outcome])
            else:
                outcomes.append(np.nan)

        return {
            "parameter": param_name,
            "values": values.tolist(),
            "outcomes": outcomes,
            "outcome_metric": outcome,
            "baseline_value": param.point_estimate,
            "baseline_outcome": outcomes[n_points // 2] if len(outcomes) > n_points // 2 else None
        }

    def tornado_analysis(
        self,
        outcome: str = "p_outbreak_5yr",
        params_to_analyze: List[str] = None
    ) -> Dict:
        """Tornado analysis using V2 model."""
        if params_to_analyze is None:
            params_to_analyze = list(self.params.keys())

        results = []

        # Get baseline outcome
        baseline_results = self.base_model.simulate_trajectory(
            n_simulations=1000, include_uncertainty=False
        )
        baseline_outcome = baseline_results["summary"].get(outcome, 0)

        for param_name in params_to_analyze:
            param = self.params[param_name]

            # Low value simulation
            low_params = {k: v for k, v in self.params.items()}
            low_params[param_name] = ParameterWithUncertainty(
                name=param.name,
                point_estimate=param.lower_bound,
                lower_bound=param.lower_bound,
                upper_bound=param.lower_bound,
                distribution="uniform"
            )
            model_low = EnhancedStochasticAvoidanceModelV2(params=low_params)
            results_low = model_low.simulate_trajectory(n_simulations=500, include_uncertainty=False)
            outcome_low = results_low["summary"].get(outcome, 0)

            # High value simulation
            high_params = {k: v for k, v in self.params.items()}
            high_params[param_name] = ParameterWithUncertainty(
                name=param.name,
                point_estimate=param.upper_bound,
                lower_bound=param.upper_bound,
                upper_bound=param.upper_bound,
                distribution="uniform"
            )
            model_high = EnhancedStochasticAvoidanceModelV2(params=high_params)
            results_high = model_high.simulate_trajectory(n_simulations=500, include_uncertainty=False)
            outcome_high = results_high["summary"].get(outcome, 0)

            results.append({
                "parameter": param_name,
                "parameter_label": param.name,
                "low_value": param.lower_bound,
                "high_value": param.upper_bound,
                "baseline_value": param.point_estimate,
                "outcome_at_low": outcome_low,
                "outcome_at_high": outcome_high,
                "outcome_range": abs(outcome_high - outcome_low),
                "direction": "positive" if outcome_high > outcome_low else "negative"
            })

        results.sort(key=lambda x: x["outcome_range"], reverse=True)

        return {
            "baseline_outcome": baseline_outcome,
            "outcome_metric": outcome,
            "parameters": results
        }

    def probabilistic_sensitivity(
        self,
        n_samples: int = 2000,
        outcomes: List[str] = None
    ) -> Dict:
        """PSA using V2 model."""
        if outcomes is None:
            outcomes = ["p_outbreak_5yr", "p_outbreak_10yr", "median_years_to_outbreak"]

        param_samples = {
            name: param.sample(n_samples)
            for name, param in self.params.items()
        }

        results = {outcome: [] for outcome in outcomes}

        for i in range(n_samples):
            sample_params = {}
            for name, samples in param_samples.items():
                val = samples[i]
                sample_params[name] = ParameterWithUncertainty(
                    name=self.params[name].name,
                    point_estimate=val,
                    lower_bound=val,
                    upper_bound=val,
                    distribution="uniform"
                )

            model = EnhancedStochasticAvoidanceModelV2(params=sample_params)
            sim_results = model.simulate_trajectory(
                n_simulations=100,
                include_uncertainty=False
            )

            for outcome in outcomes:
                if outcome in sim_results["summary"]:
                    results[outcome].append(sim_results["summary"][outcome])
                else:
                    results[outcome].append(np.nan)

        summary = {}
        for outcome in outcomes:
            valid_results = [r for r in results[outcome] if not np.isnan(r)]
            if valid_results:
                summary[outcome] = {
                    "mean": np.mean(valid_results),
                    "std": np.std(valid_results),
                    "p5": np.percentile(valid_results, 5),
                    "p25": np.percentile(valid_results, 25),
                    "median": np.median(valid_results),
                    "p75": np.percentile(valid_results, 75),
                    "p95": np.percentile(valid_results, 95),
                }

        return {
            "n_samples": n_samples,
            "param_samples": {k: v.tolist() for k, v in param_samples.items()},
            "outcomes": {k: v for k, v in results.items()},
            "summary": summary
        }

    def scenario_comparison(
        self,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict:
        """Scenario comparison using V2 model."""
        results = {}

        for scenario_name, param_overrides in scenarios.items():
            modified_params = {k: v for k, v in self.params.items()}
            for param_name, value in param_overrides.items():
                if param_name in modified_params:
                    modified_params[param_name] = ParameterWithUncertainty(
                        name=self.params[param_name].name,
                        point_estimate=value,
                        lower_bound=value * 0.9,
                        upper_bound=value * 1.1,
                        distribution="uniform"
                    )

            model = EnhancedStochasticAvoidanceModelV2(params=modified_params)
            sim_results = model.simulate_trajectory(
                n_simulations=1000,
                include_uncertainty=False
            )

            results[scenario_name] = {
                "parameters": param_overrides,
                "summary": sim_results["summary"],
                "trajectory_stats": sim_results["trajectory_stats"]
            }

        return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_v2_network_density_comparison(v1_results, v2_results, output_dir):
    """Figure: V1 vs V2 network density trajectory comparison (2-panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 3.5))

    years = v1_results["years"]
    threshold = KEY_PARAMETERS_V2["critical_network_threshold"].point_estimate

    # Panel A: V1 Network Density
    ax = axes[0]
    v1_mean = v1_results["trajectory_stats"]["network_density"]["mean"]
    v1_p5 = v1_results["trajectory_stats"]["network_density"]["p5"]
    v1_p95 = v1_results["trajectory_stats"]["network_density"]["p95"]

    ax.fill_between(years, v1_p5, v1_p95, color='steelblue', hatch='///',
                    alpha=1.0, facecolor='white', edgecolor='steelblue')
    ax.plot(years, v1_mean, color='navy', linewidth=1.5, label='V1 (additive)')
    ax.axhline(y=threshold, color='black', linestyle=':', linewidth=1,
               label=f'Critical threshold ({threshold})')

    ax.set_xlabel("Year")
    ax.set_ylabel("Effective Network Density")
    ax.text(-0.1, 1.05, 'a', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.set_title("V1: Additive Model")
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    ax.set_xlim(2024, 2040)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: V2 Network Density
    ax = axes[1]
    v2_mean = v2_results["trajectory_stats"]["network_density"]["mean"]
    v2_p5 = v2_results["trajectory_stats"]["network_density"]["p5"]
    v2_p95 = v2_results["trajectory_stats"]["network_density"]["p95"]

    ax.fill_between(years, v2_p5, v2_p95, color='crimson', hatch='\\\\\\',
                    alpha=1.0, facecolor='white', edgecolor='crimson')
    ax.plot(years, v2_mean, color='darkred', linewidth=1.5, label='V2 (interaction)')
    ax.axhline(y=threshold, color='black', linestyle=':', linewidth=1,
               label=f'Critical threshold ({threshold})')

    ax.set_xlabel("Year")
    ax.set_ylabel("Effective Network Density")
    ax.text(-0.1, 1.05, 'b', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.set_title("V2: Meth x Housing Interaction")
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    ax.set_xlim(2024, 2040)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_publication_fig(fig, 'FigS5_V2_NetworkDensityComparison', output_dir)
    plt.close()
    return fig


def plot_v2_outbreak_forecast_comparison(v1_results, v2_results, output_dir):
    """Figure: V1 vs V2 outbreak probability forecast comparison (2-panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 3.5))

    years = v1_results["years"]

    # Panel A: Cumulative outbreak probability (overlaid)
    ax = axes[0]

    v1_mean = v1_results["trajectory_stats"]["cumulative_outbreak_prob"]["mean"]
    v1_p5 = v1_results["trajectory_stats"]["cumulative_outbreak_prob"]["p5"]
    v1_p95 = v1_results["trajectory_stats"]["cumulative_outbreak_prob"]["p95"]

    v2_mean = v2_results["trajectory_stats"]["cumulative_outbreak_prob"]["mean"]
    v2_p5 = v2_results["trajectory_stats"]["cumulative_outbreak_prob"]["p5"]
    v2_p95 = v2_results["trajectory_stats"]["cumulative_outbreak_prob"]["p95"]

    ax.fill_between(years, v1_p5, v1_p95, color='steelblue', hatch='///',
                    alpha=1.0, facecolor='white', edgecolor='steelblue', linewidth=0.5)
    ax.plot(years, v1_mean, color='navy', linewidth=1.5, label='V1 (additive)')

    ax.fill_between(years, v2_p5, v2_p95, color='crimson', hatch='\\\\\\',
                    alpha=0.5, facecolor='white', edgecolor='crimson', linewidth=0.5)
    ax.plot(years, v2_mean, color='darkred', linewidth=1.5, label='V2 (interaction)')

    ax.axhline(y=0.5, color='gray', linestyle=':')
    ax.axhline(y=0.9, color='gray', linestyle='--')

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative P(Major Outbreak)")
    ax.text(-0.1, 1.05, 'a', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    ax.set_xlim(2024, 2040)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Time-to-outbreak histogram (overlaid)
    ax = axes[1]

    v1_times = [y - 2024 for y in v1_results["outbreak_years"] if y is not None]
    v2_times = [y - 2024 for y in v2_results["outbreak_years"] if y is not None]

    if v1_times and v2_times:
        bins = range(0, 18)
        ax.hist(v1_times, bins=bins, color='steelblue', edgecolor='navy',
                density=True, alpha=0.6, label='V1')
        ax.hist(v2_times, bins=bins, color='crimson', edgecolor='darkred',
                density=True, alpha=0.6, label='V2')

        v1_med = np.median(v1_times)
        v2_med = np.median(v2_times)
        ax.axvline(x=v1_med, color='navy', linestyle='--', linewidth=1)
        ax.axvline(x=v2_med, color='darkred', linestyle='--', linewidth=1)
        ymax = ax.get_ylim()[1]
        ax.text(v1_med + 0.3, ymax * 0.9, f'V1: {v1_med:.0f}yr', fontsize=8, color='navy')
        ax.text(v2_med + 0.3, ymax * 0.75, f'V2: {v2_med:.0f}yr', fontsize=8, color='darkred')

    ax.set_xlabel("Years from 2024")
    ax.set_ylabel("Probability Density")
    ax.text(-0.1, 1.05, 'b', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.set_xlim(0, 16)
    ax.grid(axis='y', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_publication_fig(fig, 'FigS6_V2_OutbreakForecastComparison', output_dir)
    plt.close()
    return fig


def plot_v2_tornado_diagram(tornado_results, output_dir):
    """Figure: V2 Tornado diagram with interaction parameter highlighted."""
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 5))

    params = tornado_results["parameters"][:10]
    baseline = tornado_results["baseline_outcome"]

    for i, p in enumerate(params):
        low_diff = p["outcome_at_low"] - baseline
        high_diff = p["outcome_at_high"] - baseline

        is_new = p["parameter"] == "meth_housing_interaction"
        low_color = '#2166ac' if not is_new else '#1a9850'
        high_color = '#b2182b' if not is_new else '#d73027'

        ax.barh(i, low_diff, height=0.6, left=baseline, color=low_color)
        ax.barh(i, high_diff, height=0.6, left=baseline, color=high_color)

    ax.axvline(x=baseline, color='black', linewidth=1, linestyle='-')

    labels = []
    for p in params:
        label = p["parameter_label"]
        if p["parameter"] == "meth_housing_interaction":
            label += " [V2 NEW]"
        labels.append(label)

    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f'{tornado_results["outcome_metric"]} (Baseline = {baseline:.2%})')
    ax.grid(axis='x', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2166ac', lw=4, label='Lower value'),
        Line2D([0], [0], color='#b2182b', lw=4, label='Higher value'),
        Line2D([0], [0], color='#1a9850', lw=4, label='V2 new param (low)'),
        Line2D([0], [0], color='#d73027', lw=4, label='V2 new param (high)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=8)

    plt.tight_layout()
    save_publication_fig(fig, 'FigS7_V2_TornadoDiagram', output_dir)
    plt.close()
    return fig


def plot_v2_scenario_comparison(scenario_results, output_dir):
    """Figure: V2 Scenario comparison for policy interventions (2-panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 3.5))

    scenarios = list(scenario_results.keys())

    # Panel A: 5-year outbreak probability
    ax = axes[0]
    p5yr = [scenario_results[s]["summary"].get("p_outbreak_5yr", 0) * 100
            for s in scenarios]

    sorted_idx = np.argsort(p5yr)
    s_sorted = [scenarios[i] for i in sorted_idx]
    p_sorted = [p5yr[i] for i in sorted_idx]

    colors = ['#d73027' if p > 50 else '#fdae61' if p > 25 else '#4575b4'
              for p in p_sorted]

    ax.barh(range(len(s_sorted)), p_sorted, color=colors)
    ax.set_yticks(range(len(s_sorted)))
    ax.set_yticklabels(s_sorted, fontsize=8)
    ax.set_xlabel("P(Outbreak within 5 years) %")
    ax.text(-0.1, 1.05, 'a', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.grid(axis='x', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Median time to outbreak
    ax = axes[1]
    median_times = [scenario_results[s]["summary"].get("median_years_to_outbreak", 20)
                    for s in scenarios]
    m_sorted = [median_times[i] for i in sorted_idx]

    ax.barh(range(len(s_sorted)), m_sorted, color='#999999')
    ax.set_yticks(range(len(s_sorted)))
    ax.set_yticklabels([])
    ax.set_xlabel("Median Years to Outbreak")
    ax.text(-0.1, 1.05, 'b', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.set_xlim(0, 20)
    ax.grid(axis='x', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_publication_fig(fig, 'FigS8_V2_ScenarioComparison', output_dir)
    plt.close()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run V2 model analysis and compare with V1."""

    print("=" * 80)
    print("STOCHASTIC AVOIDANCE MODEL V2")
    print("Multiplicative Meth x Housing Interaction")
    print("=" * 80)
    print()

    n_sims = 2000

    # =========================================================================
    # 1. NEW V2 PARAMETER
    # =========================================================================
    print("1. V2 NEW PARAMETER")
    print("-" * 80)
    new_params = ["meth_housing_interaction"]
    p = KEY_PARAMETERS_V2["meth_housing_interaction"]
    print(f"  {p.name}")
    print(f"  Point estimate: {p.point_estimate}")
    print(f"  Bounds: [{p.lower_bound}, {p.upper_bound}]")
    print(f"  Distribution: {p.distribution}")
    print(f"  Source: {p.source}")
    print()

    # =========================================================================
    # 2. RUN V2 MODEL
    # =========================================================================
    print("2. RUNNING V2 MODEL...")
    print("-" * 80)

    # National forecast
    v2_model = EnhancedStochasticAvoidanceModelV2(region="national_average")
    v2_national = v2_model.simulate_trajectory(n_simulations=n_sims)

    print(f"   V2 National Summary:")
    print(f"   - P(outbreak within 5 years): {v2_national['summary'].get('p_outbreak_5yr', 0)*100:.1f}%")
    print(f"   - P(outbreak within 10 years): {v2_national['summary'].get('p_outbreak_10yr', 0)*100:.1f}%")
    print(f"   - Median years to outbreak: {v2_national['summary'].get('median_years_to_outbreak', 'N/A')}")

    # PNW forecast
    v2_pnw_model = EnhancedStochasticAvoidanceModelV2(region="pacific_northwest")
    v2_pnw = v2_pnw_model.simulate_trajectory(n_simulations=n_sims)

    print(f"\n   V2 PNW Summary:")
    print(f"   - P(outbreak within 5 years): {v2_pnw['summary'].get('p_outbreak_5yr', 0)*100:.1f}%")
    print(f"   - P(outbreak within 10 years): {v2_pnw['summary'].get('p_outbreak_10yr', 0)*100:.1f}%")
    print(f"   - Median years to outbreak: {v2_pnw['summary'].get('median_years_to_outbreak', 'N/A')}")

    # =========================================================================
    # 3. TORNADO ANALYSIS (including interaction param)
    # =========================================================================
    print("\n3. RUNNING V2 TORNADO ANALYSIS...")
    print("-" * 80)

    v2_analyzer = SensitivityAnalyzerV2()
    v2_tornado = v2_analyzer.tornado_analysis(
        outcome="p_outbreak_5yr",
        params_to_analyze=[
            "meth_annual_growth_rate", "meth_network_multiplier",
            "baseline_network_density", "critical_network_threshold",
            "housing_instability_rate", "incarceration_annual_rate",
            "ssp_coverage", "oat_coverage", "baseline_outbreak_prob",
            # V2 new parameter
            "meth_housing_interaction",
        ]
    )

    print("\n   Top Parameters by Influence:")
    for i, p in enumerate(v2_tornado["parameters"][:8]):
        marker = " [NEW]" if p["parameter"] in new_params else ""
        print(f"   {i+1}. {p['parameter_label']}: range = {p['outcome_range']*100:.1f}pp{marker}")

    # =========================================================================
    # 4. SCENARIO COMPARISON
    # =========================================================================
    print("\n4. RUNNING V2 SCENARIO COMPARISON...")
    print("-" * 80)

    policy_scenarios = {
        "Current Policy": {"ssp_coverage": 0.21, "oat_coverage": 0.08},
        "SSP Expansion (50%)": {"ssp_coverage": 0.50, "oat_coverage": 0.08},
        "OAT Expansion (40%)": {"ssp_coverage": 0.21, "oat_coverage": 0.40},
        "Combined SSP+OAT": {"ssp_coverage": 0.50, "oat_coverage": 0.40},
        "Decriminalization Effect": {
            "ssp_coverage": 0.40, "oat_coverage": 0.30,
            "incarceration_annual_rate": 0.10, "housing_instability_rate": 0.50,
        },
        "Full Harm Reduction": {
            "ssp_coverage": 0.80, "oat_coverage": 0.60,
            "incarceration_annual_rate": 0.05, "housing_instability_rate": 0.30,
        },
    }

    v2_scenarios = v2_analyzer.scenario_comparison(policy_scenarios)

    print(f"\n   {'Scenario':<30} {'P(5yr)':>10} {'Median':>10}")
    print("   " + "-" * 55)
    for name, res in v2_scenarios.items():
        p5 = res["summary"].get("p_outbreak_5yr", 0) * 100
        med = res["summary"].get("median_years_to_outbreak", float('nan'))
        med_str = f"{med:.1f}yr" if med == med else "N/A"
        print(f"   {name:<30} {p5:>9.1f}% {med_str:>10}")

    # =========================================================================
    # 5. PSA
    # =========================================================================
    print(f"\n5. RUNNING V2 PROBABILISTIC SENSITIVITY ANALYSIS (200 samples)...")
    print("-" * 80)

    v2_psa = v2_analyzer.probabilistic_sensitivity(
        n_samples=200,
        outcomes=["p_outbreak_5yr", "p_outbreak_10yr", "median_years_to_outbreak"]
    )

    print("\n   PSA Results:")
    for outcome, stats in v2_psa["summary"].items():
        print(f"   - {outcome}:")
        print(f"     Mean: {stats['mean']:.3f}, 90% CI: ({stats['p5']:.3f}, {stats['p95']:.3f})")

    # =========================================================================
    # 6. SIDE-BY-SIDE V1 vs V2 COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. V1 vs V2 COMPARISON")
    print("=" * 80)

    # Run V1 for comparison
    print("\n   Running V1 model for comparison...")
    v1_model = EnhancedStochasticAvoidanceModel(region="national_average")
    v1_national = v1_model.simulate_trajectory(n_simulations=n_sims)

    v1_pnw_model = EnhancedStochasticAvoidanceModel(region="pacific_northwest")
    v1_pnw = v1_pnw_model.simulate_trajectory(n_simulations=n_sims)

    # National comparison
    print("\n   NATIONAL FORECAST:")
    metrics = [
        ("P(outbreak within 5yr)", "p_outbreak_5yr", "%"),
        ("P(outbreak within 10yr)", "p_outbreak_10yr", "%"),
        ("Median years to outbreak", "median_years_to_outbreak", "yr"),
        ("Outbreak rate (any time)", "outbreak_rate", "%"),
    ]

    print(f"   {'Metric':<35} {'V1':>12} {'V2':>12} {'Delta':>10}")
    print("   " + "-" * 70)
    for label, key, unit in metrics:
        v1_val = v1_national["summary"].get(key, 0)
        v2_val = v2_national["summary"].get(key, 0)
        if unit == "%":
            v1_str = f"{v1_val*100:.1f}%"
            v2_str = f"{v2_val*100:.1f}%"
            delta = f"{(v2_val - v1_val)*100:+.1f}pp"
        else:
            v1_str = f"{v1_val:.2f}"
            v2_str = f"{v2_val:.2f}"
            delta = f"{v2_val - v1_val:+.2f}"
        print(f"   {label:<35} {v1_str:>12} {v2_str:>12} {delta:>10}")

    # PNW comparison
    print(f"\n   PACIFIC NORTHWEST FORECAST:")
    print(f"   {'Metric':<35} {'V1':>12} {'V2':>12} {'Delta':>10}")
    print("   " + "-" * 70)
    for label, key, unit in metrics:
        v1_val = v1_pnw["summary"].get(key, 0)
        v2_val = v2_pnw["summary"].get(key, 0)
        if unit == "%":
            v1_str = f"{v1_val*100:.1f}%"
            v2_str = f"{v2_val*100:.1f}%"
            delta = f"{(v2_val - v1_val)*100:+.1f}pp"
        else:
            v1_str = f"{v1_val:.2f}"
            v2_str = f"{v2_val:.2f}"
            delta = f"{v2_val - v1_val:+.2f}"
        print(f"   {label:<35} {v1_str:>12} {v2_str:>12} {delta:>10}")

    # Network density comparison
    print(f"\n   NETWORK DENSITY TRAJECTORY (NATIONAL, MEAN):")
    years = v1_national["years"]
    v1_density = v1_national["trajectory_stats"]["network_density"]["mean"]
    v2_density = v2_national["trajectory_stats"]["network_density"]["mean"]

    print(f"   {'Year':<8} {'V1 Density':>14} {'V2 Density':>14} {'Delta':>10}")
    print("   " + "-" * 50)
    for i in range(0, len(years), 4):
        print(f"   {years[i]:<8} {v1_density[i]:>14.4f} {v2_density[i]:>14.4f} {v2_density[i]-v1_density[i]:>+10.4f}")

    # =========================================================================
    # 7. STRUCTURAL CHANGE INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. INTERPRETATION: STRUCTURAL CHANGE IN V2")
    print("=" * 80)

    nat_v1_5yr = v1_national["summary"].get("p_outbreak_5yr", 0) * 100
    nat_v2_5yr = v2_national["summary"].get("p_outbreak_5yr", 0) * 100
    pnw_v1_5yr = v1_pnw["summary"].get("p_outbreak_5yr", 0) * 100
    pnw_v2_5yr = v2_pnw["summary"].get("p_outbreak_5yr", 0) * 100

    print(f"""
MULTIPLICATIVE METH x HOUSING INTERACTION
  The interaction coefficient (0.8) adds meth_prev * housing * 0.8 to network
  density, capturing Hood's finding that the joint effect of meth + unstable
  housing is ~1.5x larger than the sum of individual effects.

  Effect: Increases network density, especially where both meth prevalence
  and housing instability are high. This accelerates approach to the critical
  network threshold, increasing outbreak probability.

  National 5yr outbreak: {nat_v1_5yr:.1f}% (V1) -> {nat_v2_5yr:.1f}% (V2) = {nat_v2_5yr - nat_v1_5yr:+.1f}pp
  PNW 5yr outbreak:      {pnw_v1_5yr:.1f}% (V1) -> {pnw_v2_5yr:.1f}% (V2) = {pnw_v2_5yr - pnw_v1_5yr:+.1f}pp

  Note: TasP is NOT modeled separately because hiv_prevalence_pwid inputs
  already reflect current viral suppression among PWID on ART. Adding a
  separate TasP discount would double-count this protective effect.
""")

    # =========================================================================
    # 8. SAVE RESULTS
    # =========================================================================
    print("8. SAVING RESULTS...")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "csv_xlsx")
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "description": "Stochastic Avoidance Model V2 - Multiplicative Meth x Housing Interaction",
        "v2_new_parameters": {
            name: {
                "point_estimate": KEY_PARAMETERS_V2[name].point_estimate,
                "lower_bound": KEY_PARAMETERS_V2[name].lower_bound,
                "upper_bound": KEY_PARAMETERS_V2[name].upper_bound,
                "source": KEY_PARAMETERS_V2[name].source,
            }
            for name in new_params
        },
        "v2_national_forecast": {k: v for k, v in v2_national["summary"].items()},
        "v2_pnw_forecast": {k: v for k, v in v2_pnw["summary"].items()},
        "v1_vs_v2_national": {
            "v1": {k: v for k, v in v1_national["summary"].items()},
            "v2": {k: v for k, v in v2_national["summary"].items()},
        },
        "v1_vs_v2_pnw": {
            "v1": {k: v for k, v in v1_pnw["summary"].items()},
            "v2": {k: v for k, v in v2_pnw["summary"].items()},
        },
        "v2_tornado_analysis": {
            "baseline_outcome": v2_tornado["baseline_outcome"],
            "parameters": [
                {"rank": i+1, "parameter": p["parameter_label"],
                 "range_pp": p["outcome_range"]*100}
                for i, p in enumerate(v2_tornado["parameters"])
            ]
        },
        "v2_scenario_comparison": {
            name: {
                "p5yr": v2_scenarios[name]["summary"].get("p_outbreak_5yr", 0),
                "median": v2_scenarios[name]["summary"].get("median_years_to_outbreak", None),
            }
            for name in policy_scenarios
        },
        "v2_psa": v2_psa["summary"],
    }

    # --- JSON ---
    json_path = os.path.join(data_dir, "stochastic_avoidance_v2_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str, allow_nan=False)
    print(f"   JSON saved to: {json_path}")

    # --- CSV ---
    csv_path = os.path.join(data_dir, "stochastic_avoidance_v2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Section 1: V2 National Forecast
        writer.writerow(["V2 NATIONAL FORECAST SUMMARY"])
        writer.writerow(["Metric", "Value"])
        for k, v in v2_national["summary"].items():
            writer.writerow([k, v])
        writer.writerow([])

        # Section 2: V2 PNW Forecast
        writer.writerow(["V2 PACIFIC NORTHWEST FORECAST SUMMARY"])
        writer.writerow(["Metric", "Value"])
        for k, v in v2_pnw["summary"].items():
            writer.writerow([k, v])
        writer.writerow([])

        # Section 3: V1 vs V2 National Comparison
        writer.writerow(["V1 vs V2 NATIONAL COMPARISON"])
        writer.writerow(["Metric", "V1", "V2", "Delta"])
        for label, key, unit in metrics:
            v1_val = v1_national["summary"].get(key, 0)
            v2_val = v2_national["summary"].get(key, 0)
            delta = v2_val - v1_val
            writer.writerow([label, f"{v1_val:.6f}", f"{v2_val:.6f}", f"{delta:+.6f}"])
        writer.writerow([])

        # Section 4: V1 vs V2 PNW Comparison
        writer.writerow(["V1 vs V2 PNW COMPARISON"])
        writer.writerow(["Metric", "V1", "V2", "Delta"])
        for label, key, unit in metrics:
            v1_val = v1_pnw["summary"].get(key, 0)
            v2_val = v2_pnw["summary"].get(key, 0)
            delta = v2_val - v1_val
            writer.writerow([label, f"{v1_val:.6f}", f"{v2_val:.6f}", f"{delta:+.6f}"])
        writer.writerow([])

        # Section 5: V2 Tornado Analysis
        writer.writerow(["V2 TORNADO SENSITIVITY ANALYSIS"])
        writer.writerow(["Rank", "Parameter", "Low Value", "High Value",
                         "Outcome at Low", "Outcome at High", "Range (pp)"])
        for i, p in enumerate(v2_tornado["parameters"]):
            writer.writerow([
                i + 1, p["parameter_label"],
                f"{p['low_value']:.4f}", f"{p['high_value']:.4f}",
                f"{p['outcome_at_low']:.4f}", f"{p['outcome_at_high']:.4f}",
                f"{p['outcome_range']*100:.2f}"
            ])
        writer.writerow([])

        # Section 6: V2 Scenario Comparison
        writer.writerow(["V2 SCENARIO COMPARISON"])
        writer.writerow(["Scenario", "P(5yr Outbreak)", "Median Years to Outbreak"])
        for name in policy_scenarios:
            s = v2_scenarios[name]["summary"]
            writer.writerow([
                name,
                f"{s.get('p_outbreak_5yr', 0):.4f}",
                f"{s.get('median_years_to_outbreak', 0):.2f}"
            ])
        writer.writerow([])

        # Section 7: V2 PSA
        writer.writerow(["V2 PROBABILISTIC SENSITIVITY ANALYSIS"])
        writer.writerow(["Outcome", "Mean", "Std", "P5", "P25", "Median", "P75", "P95"])
        for outcome_name, stats in v2_psa["summary"].items():
            writer.writerow([
                outcome_name,
                f"{stats['mean']:.4f}", f"{stats['std']:.4f}",
                f"{stats['p5']:.4f}", f"{stats['p25']:.4f}",
                f"{stats['median']:.4f}", f"{stats['p75']:.4f}",
                f"{stats['p95']:.4f}"
            ])
        writer.writerow([])

        # Section 8: Network Density Trajectories
        writer.writerow(["NETWORK DENSITY TRAJECTORY (NATIONAL, MEAN)"])
        writer.writerow(["Year", "V1 Density", "V2 Density", "Delta"])
        years = v1_national["years"]
        v1_dens = v1_national["trajectory_stats"]["network_density"]["mean"]
        v2_dens = v2_national["trajectory_stats"]["network_density"]["mean"]
        for i in range(len(years)):
            writer.writerow([
                years[i],
                f"{v1_dens[i]:.6f}", f"{v2_dens[i]:.6f}",
                f"{v2_dens[i] - v1_dens[i]:+.6f}"
            ])

    print(f"   CSV saved to: {csv_path}")

    # --- XLSX ---
    xlsx_path = os.path.join(data_dir, "stochastic_avoidance_v2_results.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as xw:
            # V2 National Forecast
            df_nat = pd.DataFrame([
                {"Metric": k, "Value": v}
                for k, v in v2_national["summary"].items()
            ])
            df_nat.to_excel(xw, sheet_name="V2 National Forecast", index=False)

            # V2 PNW Forecast
            df_pnw = pd.DataFrame([
                {"Metric": k, "Value": v}
                for k, v in v2_pnw["summary"].items()
            ])
            df_pnw.to_excel(xw, sheet_name="V2 PNW Forecast", index=False)

            # V1 vs V2 Comparison
            comp_data = []
            for label, key, unit in metrics:
                v1_val = v1_national["summary"].get(key, 0)
                v2_val = v2_national["summary"].get(key, 0)
                comp_data.append({
                    "Metric": label,
                    "V1 National": v1_val,
                    "V2 National": v2_val,
                    "Delta": v2_val - v1_val,
                    "V1 PNW": v1_pnw["summary"].get(key, 0),
                    "V2 PNW": v2_pnw["summary"].get(key, 0),
                    "PNW Delta": v2_pnw["summary"].get(key, 0) - v1_pnw["summary"].get(key, 0),
                })
            pd.DataFrame(comp_data).to_excel(xw, sheet_name="V1 vs V2 Comparison", index=False)

            # V2 Tornado
            tornado_data = []
            for i, p in enumerate(v2_tornado["parameters"]):
                tornado_data.append({
                    "Rank": i + 1,
                    "Parameter": p["parameter_label"],
                    "Low Value": p["low_value"],
                    "High Value": p["high_value"],
                    "Outcome at Low": p["outcome_at_low"],
                    "Outcome at High": p["outcome_at_high"],
                    "Range (pp)": p["outcome_range"] * 100,
                })
            pd.DataFrame(tornado_data).to_excel(xw, sheet_name="V2 Tornado", index=False)

            # V2 Scenarios
            scenario_data = []
            for name in policy_scenarios:
                s = v2_scenarios[name]["summary"]
                scenario_data.append({
                    "Scenario": name,
                    "P(5yr Outbreak)": s.get("p_outbreak_5yr", 0),
                    "Median Years": s.get("median_years_to_outbreak", 0),
                })
            pd.DataFrame(scenario_data).to_excel(xw, sheet_name="V2 Scenarios", index=False)

            # V2 PSA
            psa_data = []
            for outcome_name, stats in v2_psa["summary"].items():
                psa_data.append({
                    "Outcome": outcome_name,
                    "Mean": stats["mean"],
                    "Std": stats["std"],
                    "P5": stats["p5"],
                    "P25": stats["p25"],
                    "Median": stats["median"],
                    "P75": stats["p75"],
                    "P95": stats["p95"],
                })
            pd.DataFrame(psa_data).to_excel(xw, sheet_name="V2 PSA", index=False)

            # Network Density Trajectories
            traj_data = []
            for i in range(len(years)):
                traj_data.append({
                    "Year": years[i],
                    "V1 Density": v1_dens[i],
                    "V2 Density": v2_dens[i],
                    "Delta": v2_dens[i] - v1_dens[i],
                })
            pd.DataFrame(traj_data).to_excel(xw, sheet_name="Network Density", index=False)

        print(f"   XLSX saved to: {xlsx_path}")
    except Exception as e:
        logger.error(f"Failed to save Excel results: {e}")

    # =========================================================================
    # 9. GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n9. GENERATING VISUALIZATIONS...")

    plot_v2_network_density_comparison(v1_national, v2_national, fig_dir)
    plot_v2_outbreak_forecast_comparison(v1_national, v2_national, fig_dir)
    plot_v2_tornado_diagram(v2_tornado, fig_dir)
    plot_v2_scenario_comparison(v2_scenarios, fig_dir)

    print("\nDone.")
    return output


if __name__ == "__main__":
    results = main()
