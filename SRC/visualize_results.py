#!/usr/bin/env python3
"""Quick visualizer for existing PWID simulation results"""

import json
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Visualize PWID simulation results")
    parser.add_argument("--input", type=str, default="../data/csv_xlsx/architectural_barrier_results.json",
                        help="Input JSON results file (default: ../data/csv_xlsx/architectural_barrier_results.json)")
    parser.add_argument("--output", type=str, default="../data/figures/scenario_comparison.png",
                        help="Output image path (default: ../data/figures/scenario_comparison.png)")
    args = parser.parse_args()

    # Load the JSON file
    if not os.path.exists(args.input):
        # Fallback to alternative filename if default not found
        alt_input = args.input.replace("architectural_barrier_results.json", "manufactured_death_results.json")
        if os.path.exists(alt_input):
            args.input = alt_input
        else:
            print(f"Error: Input file '{args.input}' not found.")
            return

    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Handle both new and old result formats
    if isinstance(data, dict) and "cascade_results" in data:
        results = data["cascade_results"]
    elif isinstance(data, list):
        results = data
    else:
        print("Error: Unknown result format in JSON.")
        return

    print(f"✓ Loaded {len(results)} scenarios")

    # Extract data
    scenarios = [r['scenario'] for r in results]
    p_values = [r['observed_r0_zero_rate'] * 100 for r in results]

    # Create simple bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(scenarios)), p_values, color='steelblue')

    # Add values on bars
    for bar, val in zip(bars, p_values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Policy Scenario', fontsize=12, fontweight='bold')
    plt.ylabel('Probability of R(0) = 0 (%)', fontsize=12, fontweight='bold')
    plt.title('PWID LAI-PrEP Success Rate by Policy Scenario', fontsize=14, fontweight='bold')
    plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle=':', color='gray')

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {args.output}")

if __name__ == "__main__":
    main()