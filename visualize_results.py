#!/usr/bin/env python3
"""Quick visualizer for existing PWID simulation results"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load the JSON file
with open('pwid_simulation_results.json', 'r') as f:
    results = json.load(f)

print(f"✓ Loaded {len(results)} scenarios")

# Extract data
scenarios = [r['scenario'] for r in results]
p_values = [r['observed_r0_zero_rate'] * 100 for r in results]

# Create simple bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(scenarios)), p_values, color='steelblue', alpha=0.7)

# Add values on bars
for bar, val in zip(bars, p_values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Policy Scenario', fontsize=12, fontweight='bold')
plt.ylabel('Probability of R(0) = 0 (%)', fontsize=12, fontweight='bold')
plt.title('PWID LAI-PrEP Success Rate by Policy Scenario', fontsize=14, fontweight='bold')
plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)

# Save and show
Path('archieve/figures').mkdir(exist_ok=True)
plt.savefig('./figures/scenario_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ./figures/scenario_comparison.png")
plt.show()