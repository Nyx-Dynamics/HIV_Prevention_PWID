#!/usr/bin/env python3
"""
Generate Supplementary Results for AIDS and Behavior
====================================================

This script reproduces both the supplementary data and figures for 
Supplementary Figures 1-5 and the associated Supplemental Data tables.
"""

import os
import sys
import shutil
import logging
import json
import csv
import pandas as pd
from datetime import datetime

# Add the SRC directory to path so we can import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stochastic_avoidance_enhanced import (
        EnhancedStochasticAvoidanceModel,
        SensitivityAnalyzer,
        plot_regional_contextual_failure_driver_trajectories,
        plot_outbreak_probability_forecast,
        plot_tornado_diagram,
        plot_scenario_comparison
    )
except ImportError:
    print("Error: Could not import stochastic_avoidance_enhanced. Ensure it is in the same directory.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_data_to_supplemental(national_results, tornado_results, scenario_results, regional_results, supplemental_data_dir, backup_data_dir=None):
    """Exports all simulation results to CSV and XLSX in the supplemental data directory."""
    print(f"\n[5/6] Exporting data files...")
    
    generated_files = []

    # 1. National Forecast Data
    national_df = pd.DataFrame(national_results["summary"].items(), columns=["Metric", "Value"])
    path1 = os.path.join(supplemental_data_dir, "national_forecast_summary.csv")
    national_df.to_csv(path1, index=False)
    generated_files.append(path1)
    
    # 2. Tornado Analysis Data
    tornado_data = []
    for p in tornado_results["parameters"]:
        tornado_data.append({
            "Parameter": p["parameter_label"],
            "Lower Bound": p["low_value"],
            "Upper Bound": p["high_value"],
            "Base Value": p["baseline_value"],
            "Outcome (Low)": p["outcome_at_low"],
            "Outcome (High)": p["outcome_at_high"],
            "Outcome Range": p["outcome_range"]
        })
    tornado_df = pd.DataFrame(tornado_data)
    path2 = os.path.join(supplemental_data_dir, "tornado_analysis.csv")
    tornado_df.to_csv(path2, index=False)
    generated_files.append(path2)
    
    # 3. Scenario Comparison Data
    scenario_data = []
    for name, res in scenario_results.items():
        row = {"Scenario": name}
        row.update(res["parameters"])
        row.update(res["summary"])
        scenario_data.append(row)
    scenario_df = pd.DataFrame(scenario_data)
    path3 = os.path.join(supplemental_data_dir, "scenario_comparison.csv")
    scenario_df.to_csv(path3, index=False)
    generated_files.append(path3)
    
    # 4. Regional Comparison Data
    regional_data = []
    for region, res in regional_results.items():
        row = {"Region": region}
        row.update(res["summary"])
        regional_data.append(row)
    regional_df = pd.DataFrame(regional_data)
    path4 = os.path.join(supplemental_data_dir, "regional_comparison.csv")
    regional_df.to_csv(path4, index=False)
    generated_files.append(path4)
    
    # 5. Master XLSX file with all tabs
    xlsx_path = os.path.join(supplemental_data_dir, "supplementary_data_master.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            national_df.to_excel(writer, sheet_name='National Forecast', index=False)
            regional_df.to_excel(writer, sheet_name='Regional Comparison', index=False)
            tornado_df.to_excel(writer, sheet_name='Tornado Analysis', index=False)
            scenario_df.to_excel(writer, sheet_name='Scenario Comparison', index=False)
        generated_files.append(xlsx_path)
    except Exception as e:
        print(f"  ! Warning: Could not create Excel file: {e}")

    # Copy to backup directory if provided
    if backup_data_dir:
        os.makedirs(backup_data_dir, exist_ok=True)
        for f in generated_files:
            shutil.copy(f, backup_data_dir)
            print(f"  ✓ Exported: {os.path.basename(f)} to {backup_data_dir}")

    print(f"\n  Summary of data files created in {supplemental_data_dir}:")
    for f in generated_files:
        print(f"  ✓ {os.path.abspath(f)}")

def main():
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "Supplementary Figures")
    data_dir = os.path.join(base_dir, "data", "csv_xlsx")
    supplemental_data_dir = os.path.join(base_dir, "Supplemental Data", "data")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(supplemental_data_dir, exist_ok=True)
    
    print("=" * 80)
    print("REPRODUCING SUPPLEMENTARY RESULTS (FIGURES AND DATA)")
    print(f"Figure Directory: {os.path.abspath(output_dir)}")
    print(f"Data Directory:   {os.path.abspath(supplemental_data_dir)}")
    print("=" * 80)
    
    # Check dependencies
    try:
        import pandas as pd
        import openpyxl
    except ImportError as e:
        print(f"Error: Missing dependency. {e}")
        print("Please install required packages: pip install pandas openpyxl")
        sys.exit(1)
    
    # 1. FigS1: Contextual Stochastic Failure Driver
    print("\n[1/6] Generating FigS1: Regional Contextual Stochastic Failure Driver Trajectories...")
    plot_regional_contextual_failure_driver_trajectories()
    
    # 2. FigS2: Outbreak Forecast
    print("\n[2/6] Generating FigS2: National Outbreak Probability Forecast...")
    model = EnhancedStochasticAvoidanceModel(region="national_average")
    national_results = model.simulate_trajectory(n_simulations=2000)
    plot_outbreak_probability_forecast(national_results)
    
    # Regional analysis for data export
    print("\n[2.1/6] Running regional outbreak analysis...")
    regional_results = {}
    for region in ["appalachia", "pacific_northwest", "northeast_urban", "national_average"]:
        reg_model = EnhancedStochasticAvoidanceModel(region=region)
        regional_results[region] = reg_model.simulate_trajectory(n_simulations=1000)
    
    # 3. FigS3: Tornado Diagram
    print("\n[3/6] Generating FigS3: Tornado Sensitivity Analysis...")
    analyzer = SensitivityAnalyzer()
    tornado_results = analyzer.tornado_analysis(
        outcome="p_outbreak_5yr",
        params_to_analyze=[
            "meth_annual_growth_rate",
            "meth_network_multiplier", 
            "baseline_network_density",
            "critical_network_threshold",
            "housing_instability_rate",
            "incarceration_annual_rate",
            "ssp_coverage",
            "oat_coverage",
            "baseline_outbreak_prob",
        ]
    )
    plot_tornado_diagram(tornado_results)
    
    # 4. FigS4 & FigS5: Scenario Comparison
    print("\n[4/6] Generating FigS4: Policy Scenario Comparison...")
    policy_scenarios = {
        "Current Policy": {"ssp_coverage": 0.21, "oat_coverage": 0.08},
        "SSP Expansion (50%)": {"ssp_coverage": 0.50, "oat_coverage": 0.08},
        "OAT Expansion (40%)": {"ssp_coverage": 0.21, "oat_coverage": 0.40},
        "Combined SSP+OAT": {"ssp_coverage": 0.50, "oat_coverage": 0.40},
        "Decriminalization": {
            "ssp_coverage": 0.40, "oat_coverage": 0.30,
            "incarceration_annual_rate": 0.10, "housing_instability_rate": 0.50
        },
        "Full Harm Reduction": {
            "ssp_coverage": 0.80, "oat_coverage": 0.60,
            "incarceration_annual_rate": 0.05, "housing_instability_rate": 0.30
        },
    }
    scenario_results = analyzer.scenario_comparison(policy_scenarios)
    plot_scenario_comparison(scenario_results)
    
    # 5. Export Data to Supplemental directory
    export_data_to_supplemental(national_results, tornado_results, scenario_results, regional_results, supplemental_data_dir, backup_data_dir=data_dir)
    
    # 6. Organising and renaming for Supplementary Figures directory
    print("\n[6/6] Organizing figures into final directory...")
    
    # The plotting functions save to "MD/Data - Results/MD_figures_aids_behavior"
    src_fig_dir = "MD/Data - Results/MD_figures_aids_behavior"
    
    if os.path.exists(src_fig_dir):
        for filename in os.listdir(src_fig_dir):
            if filename.startswith("FigS"):
                shutil.copy(os.path.join(src_fig_dir, filename), os.path.join(output_dir, filename))
                print(f"  Copied {filename} to Supplementary Figures/")
        
        # FigS5 is often a PNG version of scenario comparison for detail
        s4_png = os.path.join(base_dir, "data", "figures", "scenario_comparison.png")
        if os.path.exists(s4_png):
            shutil.copy(s4_png, os.path.join(output_dir, "FigS5_ScenarioComparisonDetail.png"))
            print("  Created FigS5_ScenarioComparisonDetail.png from scenario_comparison.png")
        else:
            # If not found in default location, check if plot_scenario_comparison can be induced to save it
            pass 

    print("\n" + "=" * 80)
    print("SUCCESS: Supplementary results (figures and data) reproduced.")
    print(f"Figures: {output_dir}")
    print(f"Data: {supplemental_data_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
