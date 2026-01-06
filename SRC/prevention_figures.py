import matplotlib.pyplot as plt
import os
import sys

# Ensure root directory is in sys.path to find pep_mucosal.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pep_mucosal import (
    InfectionEstablishmentModel,
    plot_pep_efficacy_curve,
    plot_pep_timing_vs_reservoir,
    plot_pep_timing_vs_reservoir_quant,
)

# Set output directory
output_dir = "../data/figures"
os.makedirs(output_dir, exist_ok=True)

# ========== Figure 1: Mucosal PEP timeline (2x2 panel) ==========
fig1_path = os.path.join(output_dir, "fig1_mucosal_timeline.pdf")
fig1, results1 = plot_pep_efficacy_curve(save_path=fig1_path)
plt.close(fig1)

# ========== Figure 3: Expected reservoir vs PEP timing ==========
fig3_path = os.path.join(output_dir, "fig3_expected_reservoir.pdf")
fig3 = plot_pep_timing_vs_reservoir(save_path=fig3_path)
plt.close(fig3)

# ========== Figure 3 (Quant): Quantitative expected reservoir vs PEP timing ==========
fig3_quant_path = os.path.join(output_dir, "fig3_expected_reservoir_quant.pdf")
fig3_quant = plot_pep_timing_vs_reservoir_quant(save_path=fig3_quant_path)
plt.close(fig3_quant)

# ========== Figure 2: Parenteral vs mucosal compression ==========
# Use the same InfectionEstablishmentModel but override parenteral timing
# Parenteral exposure (e.g. IDU) bypasses mucosal barriers, leading to much faster seeding
model_mucosal = InfectionEstablishmentModel()
model_parenteral = InfectionEstablishmentModel()
model_parenteral.seeding_midpoint = 12   # hours (vs 72h mucosal)
model_parenteral.integration_complete = 24   # hours (vs 120h mucosal)

results_mucosal = model_mucosal.simulate_pep_timing_curve(max_hours=168)
results_parenteral = model_parenteral.simulate_pep_timing_curve(max_hours=168)

fig2, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    results_mucosal["hours"],
    results_mucosal["efficacy"] * 100,
    "b-",
    linewidth=2.5,
    label="Mucosal (sexual exposure)",
)
ax.plot(
    results_parenteral["hours"],
    results_parenteral["efficacy"] * 100,
    "r--",
    linewidth=2.5,
    label="Parenteral (injection exposure)",
)

# Annotate approximate critical windows
ax.axvline(x=72, color="blue", linestyle=":", label="~72 h mucosal window")
ax.axvline(x=24, color="red", linestyle=":", label="~24 h parenteral window")
ax.axhline(y=70, color="gray", linestyle=":")

ax.set_xlabel("Hours from exposure to PEP initiation", fontsize=11)
ax.set_ylabel("PEP efficacy (%)", fontsize=11)
ax.set_title("PEP efficacy: mucosal vs parenteral exposure", fontsize=12)
ax.set_xlim(0, 168)
ax.set_ylim(0, 105)
ax.grid(True, color='lightgray', linewidth=0.5)
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
fig2_path = os.path.join(output_dir, "fig2_parenteral_compression.pdf")
fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig2)

print(f"Generated figures in {output_dir}:")
print("  - fig1_mucosal_timeline.pdf")
print("  - fig2_parenteral_compression.pdf")
print("  - fig3_expected_reservoir.pdf")
print("  - fig3_expected_reservoir_quant.pdf")
