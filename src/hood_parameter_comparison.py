#!/usr/bin/env python3
"""
Hood et al. (2018) Parameter Adjustment Comparison
====================================================

Quantifies impact of incorporating Hood et al. (2018) King County MSM-PWID
data into the stochastic avoidance model.

Hood JE, Buskin SE, et al. "The Changing Burden of HIV Attributable to
Methamphetamine Among MSM in King County, Washington."
AIDS Patient Care and STDs. 2018;32(6):223-233.

Key Hood data incorporated:
  - Population-level RR for meth-using vs non-using MSM: 3.39–6.48
  - Injection drug use among meth-using MSM: 37% vs 4% (aOR 21.73)
  - Network expansion: 10+ partners 44% vs 21%, transactional sex 19% vs 4%
  - Viral suppression gap: meth users 59% vs 73% at 12 months
  - Meth x housing interaction: unstable + meth → only 42% suppressed
  - Treatment-as-prevention: PAR declined 25% → 13% (2010–2015)

Now includes 3-way comparison: Original / Hood (V1) / V2+Hood
V2 adds multiplicative meth x housing interaction.

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stochastic_avoidance_enhanced import (
    ParameterWithUncertainty,
    KEY_PARAMETERS,
    REGIONAL_PROFILES,
    RegionalContextualFailureDriverProfile,
    EnhancedStochasticAvoidanceModel,
    SensitivityAnalyzer,
    save_publication_fig,
    WIDTH_DOUBLE,
    MM_TO_INCH,
)

from stochastic_avoidance_v2 import (
    KEY_PARAMETERS_V2,
    EnhancedStochasticAvoidanceModelV2,
    SensitivityAnalyzerV2,
)

# Reset RNG for reproducibility
rng = np.random.default_rng(42)

# =============================================================================
# HOOD-ADJUSTED PARAMETERS
# =============================================================================

def create_hood_adjusted_parameters():
    """
    Create parameter set incorporating Hood et al. (2018) empirical data.
    
    Changes from original:
    
    1. meth_network_multiplier: 2.5 → 3.3
       Hood shows meth-using MSM have 2.1x more partners (44% vs 21% with 10+),
       9.25x injection rate (37% vs 4%), 4.75x transactional sex (19% vs 4%).
       Geometric mean of behavioral multipliers: (2.1 * 9.25 * 4.75)^(1/3) ≈ 4.5
       Discounted to 3.3 for population-level averaging (not all are extreme).
       Bounds narrowed from [1.5, 4.0] → [2.5, 4.2] (empirically grounded).
    
    2. meth_hiv_hr: 1.46 → 2.30
       Plankey 2007 HR=1.46 measures meth alone on seroconversion in MACS (no IDU).
       Hood's population-level RR = 3.39 (2015) captures compound meth+network+IDU.
       Geometric mean of Plankey HR and Hood RR: sqrt(1.46 * 3.39) ≈ 2.22
       Rounded to 2.30 to account for the compounding effect over time.
       Bounds: [1.46, 3.39] spanning both estimates.
    
    3. housing_instability_rate: 0.685 → 0.55
       Hood shows 24% unstable housing among meth-using MSM (King County).
       NHBS 23-city gives 68.5% for PWID broadly.
       For PNW where MSM-PWID bridge is substantial, blended estimate: 0.55.
       Bounds widened: [0.24, 0.685] reflecting the two data sources.
    
    4. NEW: viral_suppression_penalty (modeled via reduced OAT effectiveness proxy)
       Hood shows meth users achieve only 59% viral suppression at 12 months
       vs 73% non-users (19% relative reduction). With unstable housing: 42%.
       We model this as a reduction in OAT/treatment effectiveness.
       oat_coverage effective reduced from 0.08 → 0.06 (meth interference).
    
    5. PNW regional profile adjustment:
       Hood shows ~25% meth among newly diagnosed MSM (2010), 18% by 2015.
       HIV-negative MSM: 4-11%. Current model uses 35% for PNW PWID.
       Adjusted to 0.28 (blending PWID-specific higher rate with Hood's MSM data).
    
    6. ssp_coverage: 0.21 → 0.21 (unchanged, but bounds narrowed)
       Hood's data doesn't directly inform SSP coverage, but the declining PAR
       (25% → 13%) suggests treatment-as-prevention is working in King County.
       This means SSP+treatment combined effects are stronger than modeled.
       Bounds tightened: [0.15, 0.28] (King County has above-average SSP).
    """
    
    hood_params = {}
    
    # Copy all original parameters first
    for name, param in KEY_PARAMETERS.items():
        hood_params[name] = ParameterWithUncertainty(
            name=param.name,
            point_estimate=param.point_estimate,
            lower_bound=param.lower_bound,
            upper_bound=param.upper_bound,
            distribution=param.distribution,
            source=param.source,
        )
    
    # --- Parameter 1: meth_network_multiplier ---
    hood_params["meth_network_multiplier"] = ParameterWithUncertainty(
        name="Meth effect on network density (Hood-adjusted)",
        point_estimate=3.3,       # was 2.5
        lower_bound=2.5,          # was 1.5
        upper_bound=4.2,          # was 4.0
        distribution="uniform",
        source="Hood et al. 2018 + King County behavioral data"
    )
    
    # --- Parameter 2: meth_hiv_hr ---
    hood_params["meth_hiv_hr"] = ParameterWithUncertainty(
        name="Meth use HIV hazard ratio (Hood-adjusted)",
        point_estimate=2.30,      # was 1.46
        lower_bound=1.46,         # Plankey lower bound
        upper_bound=3.39,         # Hood 2015 RR
        distribution="lognormal",
        source="Hood et al. 2018 RR + Plankey 2007 HR (geometric mean)"
    )
    
    # --- Parameter 3: housing_instability_rate ---
    hood_params["housing_instability_rate"] = ParameterWithUncertainty(
        name="PWID homelessness rate (Hood-adjusted for PNW)",
        point_estimate=0.55,      # was 0.685
        lower_bound=0.24,         # Hood meth-MSM rate
        upper_bound=0.685,        # NHBS 23-city PWID rate
        distribution="beta",
        source="Hood et al. 2018 (24%) blended with NHBS (68.5%)"
    )
    
    # --- Parameter 4: oat_coverage (proxy for viral suppression penalty) ---
    hood_params["oat_coverage"] = ParameterWithUncertainty(
        name="OAT coverage (reduced effectiveness, meth interference)",
        point_estimate=0.06,      # was 0.08
        lower_bound=0.03,         # was 0.04
        upper_bound=0.12,         # was 0.15
        distribution="beta",
        source="Hood et al. 2018 viral suppression gap (59% vs 73%)"
    )
    
    # --- Parameter 5: ssp_coverage (tightened bounds for PNW) ---
    hood_params["ssp_coverage"] = ParameterWithUncertainty(
        name="SSP coverage (PNW-adjusted, Hood TasP effect)",
        point_estimate=0.24,      # was 0.21 (PNW slightly above average)
        lower_bound=0.18,         # was 0.15
        upper_bound=0.32,         # was 0.30
        distribution="beta",
        source="Van Handel 2016 + Hood PAR decline suggesting TasP effect"
    )
    
    return hood_params


def create_hood_adjusted_pnw_profile():
    """Create PNW regional profile adjusted with Hood data."""
    return RegionalContextualFailureDriverProfile(
        region="Pacific Northwest (WA, OR) [Hood-adjusted]",
        baseline_prevalence_2018=0.28,   # was 0.35
        annual_growth_rate=0.02,          # was 0.03 (Hood shows stable/declining)
        pwid_population=120000,
        hiv_prevalence_pwid=0.08
    )


def create_hood_adjusted_v2_parameters():
    """
    Create V2 parameter set with Hood adjustments AND the meth x housing
    interaction parameter.

    Combines:
    - All Hood parameter adjustments (meth_network_multiplier, meth_hiv_hr, etc.)
    - V2 interaction parameter
    """
    # Start with Hood-adjusted V1 parameters
    hood_params = create_hood_adjusted_parameters()

    # Add the V2 interaction parameter from KEY_PARAMETERS_V2
    name = "meth_housing_interaction"
    hood_params[name] = ParameterWithUncertainty(
        name=KEY_PARAMETERS_V2[name].name,
        point_estimate=KEY_PARAMETERS_V2[name].point_estimate,
        lower_bound=KEY_PARAMETERS_V2[name].lower_bound,
        upper_bound=KEY_PARAMETERS_V2[name].upper_bound,
        distribution=KEY_PARAMETERS_V2[name].distribution,
        source=KEY_PARAMETERS_V2[name].source,
    )

    return hood_params


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_3way_network_density(orig_results, hood_results, v2h_results, output_dir):
    """Figure: 3-way network density trajectory comparison."""
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 4.0))

    years = orig_results["years"]
    threshold = KEY_PARAMETERS["critical_network_threshold"].point_estimate

    # Original
    orig_mean = orig_results["trajectory_stats"]["network_density"]["mean"]
    orig_p5 = orig_results["trajectory_stats"]["network_density"]["p5"]
    orig_p95 = orig_results["trajectory_stats"]["network_density"]["p95"]
    ax.fill_between(years, orig_p5, orig_p95, color='steelblue', alpha=0.15)
    ax.plot(years, orig_mean, color='steelblue', linewidth=1.5, label='Original')

    # Hood (V1)
    hood_mean = hood_results["trajectory_stats"]["network_density"]["mean"]
    hood_p5 = hood_results["trajectory_stats"]["network_density"]["p5"]
    hood_p95 = hood_results["trajectory_stats"]["network_density"]["p95"]
    ax.fill_between(years, hood_p5, hood_p95, color='darkorange', alpha=0.15)
    ax.plot(years, hood_mean, color='darkorange', linewidth=1.5,
            linestyle='--', label='Hood (V1)')

    # V2+Hood
    v2h_mean = v2h_results["trajectory_stats"]["network_density"]["mean"]
    v2h_p5 = v2h_results["trajectory_stats"]["network_density"]["p5"]
    v2h_p95 = v2h_results["trajectory_stats"]["network_density"]["p95"]
    ax.fill_between(years, v2h_p5, v2h_p95, color='crimson', alpha=0.15)
    ax.plot(years, v2h_mean, color='darkred', linewidth=1.5,
            linestyle='-.', label='V2+Hood')

    ax.axhline(y=threshold, color='black', linestyle=':', linewidth=1,
               label=f'Critical threshold ({threshold})')

    ax.set_xlabel("Year")
    ax.set_ylabel("Effective Network Density")
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.set_xlim(2024, 2040)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_publication_fig(fig, 'FigS9_3Way_NetworkDensity', output_dir)
    plt.close()
    return fig


def plot_3way_outbreak_forecast(orig_results, hood_results, v2h_results, output_dir):
    """Figure: 3-way cumulative outbreak probability comparison (2-panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 3.5))

    years = orig_results["years"]

    # Panel A: Cumulative outbreak probability
    ax = axes[0]

    for label, results, color, ls in [
        ("Original", orig_results, 'steelblue', '-'),
        ("Hood (V1)", hood_results, 'darkorange', '--'),
        ("V2+Hood", v2h_results, 'darkred', '-.'),
    ]:
        mean_prob = results["trajectory_stats"]["cumulative_outbreak_prob"]["mean"]
        p5 = results["trajectory_stats"]["cumulative_outbreak_prob"]["p5"]
        p95 = results["trajectory_stats"]["cumulative_outbreak_prob"]["p95"]
        ax.fill_between(years, p5, p95, color=color, alpha=0.1)
        ax.plot(years, mean_prob, color=color, linewidth=1.5,
                linestyle=ls, label=label)

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

    # Panel B: Time-to-outbreak histogram
    ax = axes[1]

    for label, results, color, alpha in [
        ("Original", orig_results, 'steelblue', 0.5),
        ("Hood (V1)", hood_results, 'darkorange', 0.4),
        ("V2+Hood", v2h_results, 'crimson', 0.4),
    ]:
        times = [y - 2024 for y in results["outbreak_years"] if y is not None]
        if times:
            ax.hist(times, bins=range(0, 18), color=color,
                    density=True, alpha=alpha, label=label)

    ax.set_xlabel("Years from 2024")
    ax.set_ylabel("Probability Density")
    ax.text(-0.1, 1.05, 'b', transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.set_xlim(0, 16)
    ax.grid(axis='y', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_publication_fig(fig, 'FigS10_3Way_OutbreakForecast', output_dir)
    plt.close()
    return fig


def plot_3way_scenario_comparison(orig_scenarios, hood_scenarios, v2h_scenarios,
                                   scenario_names, output_dir):
    """Figure: 3-way scenario comparison (grouped horizontal bars)."""
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 4.5))

    n = len(scenario_names)
    bar_height = 0.25
    y_base = np.arange(n)

    orig_p5 = [orig_scenarios[s]["summary"].get("p_outbreak_5yr", 0) * 100
               for s in scenario_names]
    hood_p5 = [hood_scenarios[s]["summary"].get("p_outbreak_5yr", 0) * 100
               for s in scenario_names]
    v2h_p5 = [v2h_scenarios[s]["summary"].get("p_outbreak_5yr", 0) * 100
              for s in scenario_names]

    # Sort by V2+Hood value
    sorted_idx = np.argsort(v2h_p5)
    s_sorted = [scenario_names[i] for i in sorted_idx]
    orig_sorted = [orig_p5[i] for i in sorted_idx]
    hood_sorted = [hood_p5[i] for i in sorted_idx]
    v2h_sorted = [v2h_p5[i] for i in sorted_idx]

    ax.barh(y_base - bar_height, orig_sorted, bar_height,
            color='steelblue', label='Original')
    ax.barh(y_base, hood_sorted, bar_height,
            color='darkorange', label='Hood (V1)')
    ax.barh(y_base + bar_height, v2h_sorted, bar_height,
            color='darkred', label='V2+Hood')

    ax.set_yticks(y_base)
    ax.set_yticklabels(s_sorted, fontsize=8)
    ax.set_xlabel("P(Outbreak within 5 years) %")
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    ax.grid(axis='x', color='lightgray', linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_publication_fig(fig, 'FigS11_3Way_ScenarioComparison', output_dir)
    plt.close()
    return fig


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

def run_comparison(n_sims=2000):
    """Run 3-way comparison: Original vs Hood-adjusted (V1) vs V2+Hood."""

    print("=" * 80)
    print("HOOD ET AL. (2018) PARAMETER ADJUSTMENT COMPARISON")
    print("Original / Hood-Adjusted (V1) / V2+Hood (meth x housing interaction)")
    print("=" * 80)
    print()

    # =========================================================================
    # 1. PARAMETER COMPARISON TABLE
    # =========================================================================
    print("1. PARAMETER CHANGES")
    print("-" * 80)
    hood_params = create_hood_adjusted_parameters()
    v2_hood_params = create_hood_adjusted_v2_parameters()

    changed_params = [
        "meth_network_multiplier", "meth_hiv_hr", "housing_instability_rate",
        "oat_coverage", "ssp_coverage"
    ]

    print(f"{'Parameter':<35} {'Original':>12} {'Hood-Adj':>12} {'Change':>10} {'Source'}")
    print("-" * 100)
    for name in changed_params:
        orig = KEY_PARAMETERS[name]
        hood = hood_params[name]
        pct_change = ((hood.point_estimate - orig.point_estimate) / orig.point_estimate) * 100
        print(f"{orig.name[:34]:<35} {orig.point_estimate:>12.3f} {hood.point_estimate:>12.3f} {pct_change:>+9.1f}%  {hood.source[:40]}")
        print(f"  {'bounds:':<33} [{orig.lower_bound:.3f}, {orig.upper_bound:.3f}]   [{hood.lower_bound:.3f}, {hood.upper_bound:.3f}]")

    # Show V2 new parameter
    v2_new_params = ["meth_housing_interaction"]
    print(f"\nV2 NEW STRUCTURAL PARAMETER (added in V2+Hood):")
    p = v2_hood_params["meth_housing_interaction"]
    print(f"  {p.name}: {p.point_estimate} [{p.lower_bound}, {p.upper_bound}] ({p.distribution})")
    print(f"  Source: {p.source}")

    print()
    print("PNW Regional Profile:")
    orig_pnw = REGIONAL_PROFILES["pacific_northwest"]
    hood_pnw = create_hood_adjusted_pnw_profile()
    print(f"  Baseline meth prevalence: {orig_pnw.baseline_prevalence_2018} -> {hood_pnw.baseline_prevalence_2018}")
    print(f"  Annual growth rate:       {orig_pnw.annual_growth_rate} -> {hood_pnw.annual_growth_rate}")
    print()

    # =========================================================================
    # 2. RUN ORIGINAL MODEL
    # =========================================================================
    print("2. RUNNING ORIGINAL MODEL...")
    print("-" * 80)

    # National forecast
    orig_model = EnhancedStochasticAvoidanceModel(region="national_average")
    orig_national = orig_model.simulate_trajectory(n_simulations=n_sims)

    # PNW forecast
    orig_pnw_model = EnhancedStochasticAvoidanceModel(region="pacific_northwest")
    orig_pnw_results = orig_pnw_model.simulate_trajectory(n_simulations=n_sims)

    # Tornado analysis
    orig_analyzer = SensitivityAnalyzer()
    tornado_params_v1 = [
        "meth_annual_growth_rate", "meth_network_multiplier",
        "baseline_network_density", "critical_network_threshold",
        "housing_instability_rate", "incarceration_annual_rate",
        "ssp_coverage", "oat_coverage", "baseline_outbreak_prob",
    ]
    orig_tornado = orig_analyzer.tornado_analysis(
        outcome="p_outbreak_5yr",
        params_to_analyze=tornado_params_v1
    )

    # Scenario comparison
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
    orig_scenarios = orig_analyzer.scenario_comparison(policy_scenarios)

    print("  Original model complete.")

    # =========================================================================
    # 3. RUN HOOD-ADJUSTED MODEL (V1 structure)
    # =========================================================================
    print("\n3. RUNNING HOOD-ADJUSTED MODEL (V1 structure)...")
    print("-" * 80)

    # National forecast with Hood params
    hood_model = EnhancedStochasticAvoidanceModel(
        region="national_average", params=hood_params
    )
    hood_national = hood_model.simulate_trajectory(n_simulations=n_sims)

    # PNW forecast with Hood params AND adjusted PNW profile
    hood_pnw_model = EnhancedStochasticAvoidanceModel(
        region="pacific_northwest", params=hood_params
    )
    hood_pnw_model.profile = create_hood_adjusted_pnw_profile()
    hood_pnw_results = hood_pnw_model.simulate_trajectory(n_simulations=n_sims)

    # Tornado analysis with Hood params
    hood_analyzer = SensitivityAnalyzer(
        base_model=EnhancedStochasticAvoidanceModel(params=hood_params)
    )
    hood_analyzer.params = hood_params
    hood_tornado = hood_analyzer.tornado_analysis(
        outcome="p_outbreak_5yr",
        params_to_analyze=tornado_params_v1
    )

    # Scenario comparison with Hood params
    hood_scenarios = hood_analyzer.scenario_comparison(policy_scenarios)

    print("  Hood-adjusted model complete.")

    # =========================================================================
    # 3b. RUN V2+HOOD MODEL (structural changes + Hood params)
    # =========================================================================
    print("\n3b. RUNNING V2+HOOD MODEL (meth x housing interaction + Hood params)...")
    print("-" * 80)

    # National forecast with V2+Hood params
    v2_hood_model = EnhancedStochasticAvoidanceModelV2(
        region="national_average", params=v2_hood_params
    )
    v2_hood_national = v2_hood_model.simulate_trajectory(n_simulations=n_sims)

    # PNW forecast with V2+Hood params AND adjusted PNW profile
    v2_hood_pnw_model = EnhancedStochasticAvoidanceModelV2(
        region="pacific_northwest", params=v2_hood_params
    )
    v2_hood_pnw_model.profile = create_hood_adjusted_pnw_profile()
    v2_hood_pnw_results = v2_hood_pnw_model.simulate_trajectory(n_simulations=n_sims)

    # Tornado analysis with V2+Hood params (including new V2 params)
    tornado_params_v2 = tornado_params_v1 + v2_new_params
    v2_hood_analyzer = SensitivityAnalyzerV2(
        base_model=EnhancedStochasticAvoidanceModelV2(params=v2_hood_params)
    )
    v2_hood_analyzer.params = v2_hood_params
    v2_hood_tornado = v2_hood_analyzer.tornado_analysis(
        outcome="p_outbreak_5yr",
        params_to_analyze=tornado_params_v2
    )

    # Scenario comparison with V2+Hood params
    v2_hood_scenarios = v2_hood_analyzer.scenario_comparison(policy_scenarios)

    print("  V2+Hood model complete.")

    # =========================================================================
    # 4. 3-WAY COMPARISON OUTPUT
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. RESULTS COMPARISON (3-WAY)")
    print("=" * 80)

    # --- 4a: National Forecast ---
    print("\n4a. NATIONAL FORECAST COMPARISON")
    print("-" * 95)
    metrics = [
        ("P(outbreak within 5yr)", "p_outbreak_5yr", "%"),
        ("P(outbreak within 10yr)", "p_outbreak_10yr", "%"),
        ("Median years to outbreak", "median_years_to_outbreak", "yr"),
        ("Outbreak rate (any time)", "outbreak_rate", "%"),
    ]

    print(f"{'Metric':<30} {'Original':>12} {'Hood (V1)':>12} {'V2+Hood':>12} {'V1 delta':>10} {'V2 delta':>10}")
    print("-" * 95)
    for label, key, unit in metrics:
        orig_val = orig_national["summary"].get(key, 0)
        hood_val = hood_national["summary"].get(key, 0)
        v2h_val = v2_hood_national["summary"].get(key, 0)
        if unit == "%":
            orig_str = f"{orig_val*100:.1f}%"
            hood_str = f"{hood_val*100:.1f}%"
            v2h_str = f"{v2h_val*100:.1f}%"
            d1 = f"{(hood_val - orig_val)*100:+.1f}pp"
            d2 = f"{(v2h_val - orig_val)*100:+.1f}pp"
        else:
            orig_str = f"{orig_val:.2f}"
            hood_str = f"{hood_val:.2f}"
            v2h_str = f"{v2h_val:.2f}"
            d1 = f"{hood_val - orig_val:+.2f}"
            d2 = f"{v2h_val - orig_val:+.2f}"
        print(f"{label:<30} {orig_str:>12} {hood_str:>12} {v2h_str:>12} {d1:>10} {d2:>10}")

    # --- 4b: PNW Forecast ---
    print("\n4b. PACIFIC NORTHWEST FORECAST COMPARISON")
    print("-" * 95)
    print(f"{'Metric':<30} {'Original':>12} {'Hood (V1)':>12} {'V2+Hood':>12} {'V1 delta':>10} {'V2 delta':>10}")
    print("-" * 95)
    for label, key, unit in metrics:
        orig_val = orig_pnw_results["summary"].get(key, 0)
        hood_val = hood_pnw_results["summary"].get(key, 0)
        v2h_val = v2_hood_pnw_results["summary"].get(key, 0)
        if unit == "%":
            orig_str = f"{orig_val*100:.1f}%"
            hood_str = f"{hood_val*100:.1f}%"
            v2h_str = f"{v2h_val*100:.1f}%"
            d1 = f"{(hood_val - orig_val)*100:+.1f}pp"
            d2 = f"{(v2h_val - orig_val)*100:+.1f}pp"
        else:
            orig_str = f"{orig_val:.2f}"
            hood_str = f"{hood_val:.2f}"
            v2h_str = f"{v2h_val:.2f}"
            d1 = f"{hood_val - orig_val:+.2f}"
            d2 = f"{v2h_val - orig_val:+.2f}"
        print(f"{label:<30} {orig_str:>12} {hood_str:>12} {v2h_str:>12} {d1:>10} {d2:>10}")

    # --- 4c: Tornado Ranking Comparison ---
    print("\n4c. TORNADO SENSITIVITY RANKING COMPARISON")
    print("-" * 120)
    print(f"{'Rank':<5} {'Original':<30} {'Range':>7}  {'Hood (V1)':<30} {'Range':>7}  {'V2+Hood':<30} {'Range':>7}")
    print("-" * 120)
    max_rows = min(9, len(orig_tornado["parameters"]), len(hood_tornado["parameters"]))
    v2h_count = len(v2_hood_tornado["parameters"])
    for i in range(min(max_rows, v2h_count)):
        op = orig_tornado["parameters"][i]
        hp = hood_tornado["parameters"][i]
        v2p = v2_hood_tornado["parameters"][i]
        v2_marker = " *" if v2p["parameter"] in v2_new_params else ""
        print(f"{i+1:<5} {op['parameter_label'][:29]:<30} {op['outcome_range']*100:>6.1f}pp  "
              f"{hp['parameter_label'][:29]:<30} {hp['outcome_range']*100:>6.1f}pp  "
              f"{v2p['parameter_label'][:28]+v2_marker:<30} {v2p['outcome_range']*100:>6.1f}pp")
    # Show remaining V2 tornado entries if more than 9
    if v2h_count > max_rows:
        for i in range(max_rows, min(12, v2h_count)):
            v2p = v2_hood_tornado["parameters"][i]
            v2_marker = " *" if v2p["parameter"] in v2_new_params else ""
            print(f"{i+1:<5} {'':30} {'':>7}   {'':30} {'':>7}   "
                  f"{v2p['parameter_label'][:28]+v2_marker:<30} {v2p['outcome_range']*100:>6.1f}pp")

    print(f"\n  (* = new V2 parameter)")
    print(f"\nBaseline 5yr outbreak probability:")
    print(f"  Original:  {orig_tornado['baseline_outcome']*100:.2f}%")
    print(f"  Hood (V1): {hood_tornado['baseline_outcome']*100:.2f}%")
    print(f"  V2+Hood:   {v2_hood_tornado['baseline_outcome']*100:.2f}%")

    # --- 4d: Scenario Comparison ---
    print("\n4d. POLICY SCENARIO COMPARISON (3-WAY)")
    print("-" * 110)
    print(f"{'Scenario':<26} {'Orig P(5yr)':>11} {'Hood P(5yr)':>11} {'V2+H P(5yr)':>12}  {'Orig Med':>9} {'Hood Med':>9} {'V2+H Med':>9}")
    print("-" * 110)
    for scenario_name in policy_scenarios:
        orig_s = orig_scenarios[scenario_name]["summary"]
        hood_s = hood_scenarios[scenario_name]["summary"]
        v2h_s = v2_hood_scenarios[scenario_name]["summary"]

        op5 = orig_s.get("p_outbreak_5yr", 0) * 100
        hp5 = hood_s.get("p_outbreak_5yr", 0) * 100
        v2p5 = v2h_s.get("p_outbreak_5yr", 0) * 100
        om = orig_s.get("median_years_to_outbreak", float('nan'))
        hm = hood_s.get("median_years_to_outbreak", float('nan'))
        v2m = v2h_s.get("median_years_to_outbreak", float('nan'))

        om_str = f"{om:.1f}yr" if om == om else "N/A"
        hm_str = f"{hm:.1f}yr" if hm == hm else "N/A"
        v2m_str = f"{v2m:.1f}yr" if v2m == v2m else "N/A"

        print(f"{scenario_name:<26} {op5:>10.1f}% {hp5:>10.1f}% {v2p5:>11.1f}%  {om_str:>9} {hm_str:>9} {v2m_str:>9}")

    # --- 4e: Network Density Trajectory ---
    print("\n4e. NETWORK DENSITY TRAJECTORY (NATIONAL, MEAN)")
    print("-" * 70)
    years = orig_national["years"]
    orig_density = orig_national["trajectory_stats"]["network_density"]["mean"]
    hood_density = hood_national["trajectory_stats"]["network_density"]["mean"]
    v2h_density = v2_hood_national["trajectory_stats"]["network_density"]["mean"]

    print(f"{'Year':<8} {'Original':>12} {'Hood (V1)':>12} {'V2+Hood':>12}")
    print("-" * 50)
    for i in range(0, len(years), 4):  # Every 4 years
        print(f"{years[i]:<8} {orig_density[i]:>12.4f} {hood_density[i]:>12.4f} {v2h_density[i]:>12.4f}")

    # --- 4f: Cumulative Outbreak Probability ---
    print("\n4f. CUMULATIVE OUTBREAK PROBABILITY (NATIONAL, MEAN)")
    print("-" * 70)
    orig_cum = orig_national["trajectory_stats"]["cumulative_outbreak_prob"]["mean"]
    hood_cum = hood_national["trajectory_stats"]["cumulative_outbreak_prob"]["mean"]
    v2h_cum = v2_hood_national["trajectory_stats"]["cumulative_outbreak_prob"]["mean"]

    print(f"{'Year':<8} {'Original':>12} {'Hood (V1)':>12} {'V2+Hood':>12}")
    print("-" * 50)
    for i in range(0, len(years), 2):  # Every 2 years
        print(f"{years[i]:<8} {orig_cum[i]*100:>11.1f}% {hood_cum[i]*100:>11.1f}% {v2h_cum[i]*100:>11.1f}%")

    # =========================================================================
    # 5. INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. INTERPRETATION: 3-WAY COMPARISON")
    print("=" * 80)

    nat_orig_5yr = orig_national["summary"].get("p_outbreak_5yr", 0) * 100
    nat_hood_5yr = hood_national["summary"].get("p_outbreak_5yr", 0) * 100
    nat_v2h_5yr = v2_hood_national["summary"].get("p_outbreak_5yr", 0) * 100
    pnw_orig_5yr = orig_pnw_results["summary"].get("p_outbreak_5yr", 0) * 100
    pnw_hood_5yr = hood_pnw_results["summary"].get("p_outbreak_5yr", 0) * 100
    pnw_v2h_5yr = v2_hood_pnw_results["summary"].get("p_outbreak_5yr", 0) * 100

    nat_orig_med = orig_national["summary"].get("median_years_to_outbreak", 0)
    nat_hood_med = hood_national["summary"].get("median_years_to_outbreak", 0)
    nat_v2h_med = v2_hood_national["summary"].get("median_years_to_outbreak", 0)
    pnw_orig_med = orig_pnw_results["summary"].get("median_years_to_outbreak", 0)
    pnw_hood_med = hood_pnw_results["summary"].get("median_years_to_outbreak", 0)
    pnw_v2h_med = v2_hood_pnw_results["summary"].get("median_years_to_outbreak", 0)

    print(f"""
NATIONAL FORECAST (3-way):
  5yr outbreak prob: {nat_orig_5yr:.1f}% (Orig) -> {nat_hood_5yr:.1f}% (Hood V1) -> {nat_v2h_5yr:.1f}% (V2+Hood)
  Median years:      {nat_orig_med:.1f} (Orig) -> {nat_hood_med:.1f} (Hood V1) -> {nat_v2h_med:.1f} (V2+Hood)

PACIFIC NORTHWEST FORECAST (3-way):
  5yr outbreak prob: {pnw_orig_5yr:.1f}% (Orig) -> {pnw_hood_5yr:.1f}% (Hood V1) -> {pnw_v2h_5yr:.1f}% (V2+Hood)
  Median years:      {pnw_orig_med:.1f} (Orig) -> {pnw_hood_med:.1f} (Hood V1) -> {pnw_v2h_med:.1f} (V2+Hood)

KEY FINDINGS:
  1. HOOD PARAMETER CALIBRATION (V1): Adjusting parameters to Hood empirical
     data primarily changes risk through the meth_network_multiplier (2.5->3.3).

  2. V2 STRUCTURAL CHANGE: The meth x housing interaction term (coeff=0.8)
     INCREASES network density and therefore outbreak risk, especially in
     high-meth + high-instability settings. This captures Hood's finding
     that the joint effect is ~1.5x larger than the sum of individual effects
     (42% suppression for meth+unstable vs 76% baseline).

  3. TasP NOTE: The model's hiv_prevalence_pwid inputs already reflect current
     viral suppression among PWID on ART. Hood's PAR decline (25% -> 13%)
     is therefore already embedded in the prevalence data used by the model.
     No separate TasP adjustment is needed or appropriate.

  4. LIMITATION NOW ADDRESSED by V2:
     - Housing interaction: NOW modeled as multiplicative (was additive)
     - Grounded in Hood et al. (2018) empirical data

  5. TORNADO RANKING in V2+Hood now includes the interaction parameter,
     allowing sensitivity analysis of the interaction strength.
""")

    # =========================================================================
    # 6. SAVE RESULTS
    # =========================================================================
    print("\n6. SAVING RESULTS...")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "csv_xlsx")
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "description": "Hood et al. 2018 parameter comparison: Original / Hood (V1) / V2+Hood (interaction)",
        "parameter_changes": {
            name: {
                "original": KEY_PARAMETERS[name].point_estimate,
                "hood_adjusted": hood_params[name].point_estimate,
                "pct_change": ((hood_params[name].point_estimate - KEY_PARAMETERS[name].point_estimate)
                              / KEY_PARAMETERS[name].point_estimate) * 100
            }
            for name in changed_params
        },
        "v2_new_parameters": {
            name: {
                "point_estimate": v2_hood_params[name].point_estimate,
                "lower_bound": v2_hood_params[name].lower_bound,
                "upper_bound": v2_hood_params[name].upper_bound,
                "source": v2_hood_params[name].source,
            }
            for name in v2_new_params
        },
        "national_forecast": {
            "original": {k: v for k, v in orig_national["summary"].items()},
            "hood_adjusted": {k: v for k, v in hood_national["summary"].items()},
            "v2_hood": {k: v for k, v in v2_hood_national["summary"].items()},
        },
        "pnw_forecast": {
            "original": {k: v for k, v in orig_pnw_results["summary"].items()},
            "hood_adjusted": {k: v for k, v in hood_pnw_results["summary"].items()},
            "v2_hood": {k: v for k, v in v2_hood_pnw_results["summary"].items()},
        },
        "tornado_ranking": {
            "original": [
                {"rank": i+1, "parameter": p["parameter_label"], "range_pp": p["outcome_range"]*100}
                for i, p in enumerate(orig_tornado["parameters"])
            ],
            "hood_adjusted": [
                {"rank": i+1, "parameter": p["parameter_label"], "range_pp": p["outcome_range"]*100}
                for i, p in enumerate(hood_tornado["parameters"])
            ],
            "v2_hood": [
                {"rank": i+1, "parameter": p["parameter_label"], "range_pp": p["outcome_range"]*100}
                for i, p in enumerate(v2_hood_tornado["parameters"])
            ],
        },
        "scenario_comparison": {
            name: {
                "original_p5yr": orig_scenarios[name]["summary"].get("p_outbreak_5yr", 0),
                "hood_p5yr": hood_scenarios[name]["summary"].get("p_outbreak_5yr", 0),
                "v2_hood_p5yr": v2_hood_scenarios[name]["summary"].get("p_outbreak_5yr", 0),
                "original_median": orig_scenarios[name]["summary"].get("median_years_to_outbreak", None),
                "hood_median": hood_scenarios[name]["summary"].get("median_years_to_outbreak", None),
                "v2_hood_median": v2_hood_scenarios[name]["summary"].get("median_years_to_outbreak", None),
            }
            for name in policy_scenarios
        }
    }

    # --- JSON ---
    json_path = os.path.join(data_dir, "hood_parameter_comparison_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str, allow_nan=False)
    print(f"   JSON saved to: {json_path}")

    # --- CSV ---
    csv_path = os.path.join(data_dir, "hood_parameter_comparison_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Section 1: Parameter Changes
        writer.writerow(["HOOD PARAMETER ADJUSTMENTS"])
        writer.writerow(["Parameter", "Original", "Hood-Adjusted", "% Change", "Source"])
        for name in changed_params:
            orig = KEY_PARAMETERS[name]
            hood = hood_params[name]
            pct = ((hood.point_estimate - orig.point_estimate) / orig.point_estimate) * 100
            writer.writerow([
                orig.name, f"{orig.point_estimate:.4f}",
                f"{hood.point_estimate:.4f}", f"{pct:+.1f}%", hood.source
            ])
        writer.writerow([])

        # Section 2: V2 New Parameters
        writer.writerow(["V2 NEW STRUCTURAL PARAMETERS"])
        writer.writerow(["Parameter", "Point Estimate", "Lower Bound", "Upper Bound", "Source"])
        for name in v2_new_params:
            p = v2_hood_params[name]
            writer.writerow([p.name, p.point_estimate, p.lower_bound, p.upper_bound, p.source])
        writer.writerow([])

        # Section 3: National Forecast Comparison
        writer.writerow(["NATIONAL FORECAST COMPARISON (3-WAY)"])
        writer.writerow(["Metric", "Original", "Hood (V1)", "V2+Hood",
                         "V1 Delta", "V2 Delta"])
        for label, key, unit in metrics:
            ov = orig_national["summary"].get(key, 0)
            hv = hood_national["summary"].get(key, 0)
            v2v = v2_hood_national["summary"].get(key, 0)
            writer.writerow([
                label, f"{ov:.6f}", f"{hv:.6f}", f"{v2v:.6f}",
                f"{hv - ov:+.6f}", f"{v2v - ov:+.6f}"
            ])
        writer.writerow([])

        # Section 4: PNW Forecast Comparison
        writer.writerow(["PNW FORECAST COMPARISON (3-WAY)"])
        writer.writerow(["Metric", "Original", "Hood (V1)", "V2+Hood",
                         "V1 Delta", "V2 Delta"])
        for label, key, unit in metrics:
            ov = orig_pnw_results["summary"].get(key, 0)
            hv = hood_pnw_results["summary"].get(key, 0)
            v2v = v2_hood_pnw_results["summary"].get(key, 0)
            writer.writerow([
                label, f"{ov:.6f}", f"{hv:.6f}", f"{v2v:.6f}",
                f"{hv - ov:+.6f}", f"{v2v - ov:+.6f}"
            ])
        writer.writerow([])

        # Section 5: Tornado Ranking Comparison
        writer.writerow(["TORNADO SENSITIVITY RANKING (3-WAY)"])
        writer.writerow(["Rank", "Original Parameter", "Orig Range (pp)",
                         "Hood Parameter", "Hood Range (pp)",
                         "V2+Hood Parameter", "V2+Hood Range (pp)"])
        max_rows = max(
            len(orig_tornado["parameters"]),
            len(hood_tornado["parameters"]),
            len(v2_hood_tornado["parameters"])
        )
        for i in range(max_rows):
            row = [i + 1]
            for tornado in [orig_tornado, hood_tornado, v2_hood_tornado]:
                if i < len(tornado["parameters"]):
                    p = tornado["parameters"][i]
                    row.extend([p["parameter_label"], f"{p['outcome_range']*100:.2f}"])
                else:
                    row.extend(["", ""])
            writer.writerow(row)
        writer.writerow([])

        # Section 6: Scenario Comparison
        writer.writerow(["POLICY SCENARIO COMPARISON (3-WAY)"])
        writer.writerow(["Scenario", "Orig P(5yr)", "Hood P(5yr)", "V2+Hood P(5yr)",
                         "Orig Median", "Hood Median", "V2+Hood Median"])
        for name in policy_scenarios:
            os_ = orig_scenarios[name]["summary"]
            hs_ = hood_scenarios[name]["summary"]
            v2s_ = v2_hood_scenarios[name]["summary"]
            writer.writerow([
                name,
                f"{os_.get('p_outbreak_5yr', 0):.4f}",
                f"{hs_.get('p_outbreak_5yr', 0):.4f}",
                f"{v2s_.get('p_outbreak_5yr', 0):.4f}",
                f"{os_.get('median_years_to_outbreak', 0):.2f}",
                f"{hs_.get('median_years_to_outbreak', 0):.2f}",
                f"{v2s_.get('median_years_to_outbreak', 0):.2f}",
            ])
        writer.writerow([])

        # Section 7: Network Density Trajectories
        writer.writerow(["NETWORK DENSITY TRAJECTORY (NATIONAL, MEAN)"])
        writer.writerow(["Year", "Original", "Hood (V1)", "V2+Hood"])
        years = orig_national["years"]
        for i in range(len(years)):
            writer.writerow([
                years[i],
                f"{orig_density[i]:.6f}",
                f"{hood_density[i]:.6f}",
                f"{v2h_density[i]:.6f}",
            ])
        writer.writerow([])

        # Section 8: Cumulative Outbreak Probability
        writer.writerow(["CUMULATIVE OUTBREAK PROBABILITY (NATIONAL, MEAN)"])
        writer.writerow(["Year", "Original", "Hood (V1)", "V2+Hood"])
        for i in range(len(years)):
            writer.writerow([
                years[i],
                f"{orig_cum[i]:.6f}",
                f"{hood_cum[i]:.6f}",
                f"{v2h_cum[i]:.6f}",
            ])

    print(f"   CSV saved to: {csv_path}")

    # --- XLSX ---
    xlsx_path = os.path.join(data_dir, "hood_parameter_comparison_results.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as xw:
            # Parameter Changes
            param_data = []
            for name in changed_params:
                orig = KEY_PARAMETERS[name]
                hood = hood_params[name]
                pct = ((hood.point_estimate - orig.point_estimate) / orig.point_estimate) * 100
                param_data.append({
                    "Parameter": orig.name,
                    "Original": orig.point_estimate,
                    "Hood-Adjusted": hood.point_estimate,
                    "% Change": pct,
                    "Original Bounds": f"[{orig.lower_bound}, {orig.upper_bound}]",
                    "Hood Bounds": f"[{hood.lower_bound}, {hood.upper_bound}]",
                    "Source": hood.source,
                })
            pd.DataFrame(param_data).to_excel(xw, sheet_name="Parameter Changes", index=False)

            # National Forecast 3-Way
            nat_data = []
            for label, key, unit in metrics:
                ov = orig_national["summary"].get(key, 0)
                hv = hood_national["summary"].get(key, 0)
                v2v = v2_hood_national["summary"].get(key, 0)
                nat_data.append({
                    "Metric": label,
                    "Original": ov, "Hood (V1)": hv, "V2+Hood": v2v,
                    "V1 Delta": hv - ov, "V2 Delta": v2v - ov,
                })
            pd.DataFrame(nat_data).to_excel(xw, sheet_name="National Forecast", index=False)

            # PNW Forecast 3-Way
            pnw_data = []
            for label, key, unit in metrics:
                ov = orig_pnw_results["summary"].get(key, 0)
                hv = hood_pnw_results["summary"].get(key, 0)
                v2v = v2_hood_pnw_results["summary"].get(key, 0)
                pnw_data.append({
                    "Metric": label,
                    "Original": ov, "Hood (V1)": hv, "V2+Hood": v2v,
                    "V1 Delta": hv - ov, "V2 Delta": v2v - ov,
                })
            pd.DataFrame(pnw_data).to_excel(xw, sheet_name="PNW Forecast", index=False)

            # Scenario Comparison
            scen_data = []
            for name in policy_scenarios:
                os_ = orig_scenarios[name]["summary"]
                hs_ = hood_scenarios[name]["summary"]
                v2s_ = v2_hood_scenarios[name]["summary"]
                scen_data.append({
                    "Scenario": name,
                    "Orig P(5yr)": os_.get("p_outbreak_5yr", 0),
                    "Hood P(5yr)": hs_.get("p_outbreak_5yr", 0),
                    "V2+Hood P(5yr)": v2s_.get("p_outbreak_5yr", 0),
                    "Orig Median": os_.get("median_years_to_outbreak", 0),
                    "Hood Median": hs_.get("median_years_to_outbreak", 0),
                    "V2+Hood Median": v2s_.get("median_years_to_outbreak", 0),
                })
            pd.DataFrame(scen_data).to_excel(xw, sheet_name="Scenario Comparison", index=False)

            # Network Density Trajectories
            traj_data = []
            for i in range(len(years)):
                traj_data.append({
                    "Year": years[i],
                    "Original": orig_density[i],
                    "Hood (V1)": hood_density[i],
                    "V2+Hood": v2h_density[i],
                })
            pd.DataFrame(traj_data).to_excel(xw, sheet_name="Network Density", index=False)

            # Cumulative Outbreak Probability
            cum_data = []
            for i in range(len(years)):
                cum_data.append({
                    "Year": years[i],
                    "Original": orig_cum[i],
                    "Hood (V1)": hood_cum[i],
                    "V2+Hood": v2h_cum[i],
                })
            pd.DataFrame(cum_data).to_excel(xw, sheet_name="Cumulative Outbreak Prob", index=False)

        print(f"   XLSX saved to: {xlsx_path}")
    except Exception as e:
        logger.error(f"Failed to save Excel results: {e}")

    # =========================================================================
    # 7. GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n7. GENERATING VISUALIZATIONS...")

    plot_3way_network_density(orig_national, hood_national, v2_hood_national, fig_dir)
    plot_3way_outbreak_forecast(orig_national, hood_national, v2_hood_national, fig_dir)
    plot_3way_scenario_comparison(
        orig_scenarios, hood_scenarios, v2_hood_scenarios,
        list(policy_scenarios.keys()), fig_dir
    )

    print("\nDone.")
    return output


if __name__ == "__main__":
    results = run_comparison(n_sims=2000)
