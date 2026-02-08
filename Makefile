# Makefile for BMC Public Health Source Code and Reproducibility Suite
# Author: AC Demidont, MD / Nyx Dynamics LLC
# Date: February 2026

PYTHON = python3
SRC_DIR = SRC
DATA_DIR = data
FIG_DIR = data/figures
SUPP_FIG_DIR = Supplementary Figures
SUPP_DATA_DIR = Supplemental Data/data

.PHONY: all reproduce figures supplementary clean help install-deps

all: help

help:
	@echo "BMC Public Health Reproducibility Suite"
	@echo "========================================"
	@echo "Available targets:"
	@echo "  install-deps   - Install required Python dependencies"
	@echo "  reproduce      - Run the full reproducibility suite (simulations and figures)"
	@echo "  figures        - Generate main publication figures (Fig 1-3)"
	@echo "  supplementary  - Generate supplementary figures and data"
	@echo "  clean          - Remove generated figures and temporary data"
	@echo "  help           - Show this help message"

install-deps:
	$(PYTHON) -m pip install -r ../requirements.txt

reproduce: install-deps simulate supplementary figures
	@echo "Full reproducibility suite completed."

simulate:
	@echo "Running simulations..."
	cd $(SRC_DIR) && $(PYTHON) structural_barrier_model.py --output-dir ../$(DATA_DIR)/csv_xlsx --n-individuals 100000
	cd $(SRC_DIR) && $(PYTHON) cascade_sensitivity_analysis.py --output-dir ../$(DATA_DIR)/csv_xlsx --n-samples 1000
	cd $(SRC_DIR) && $(PYTHON) stochastic_avoidance_enhanced.py --output-dir ../$(DATA_DIR)/csv_xlsx
	@echo "Simulations completed."

figures:
	@echo "Generating main publication figures..."
	cd $(SRC_DIR) && $(PYTHON) generate_outputs.py
	@echo "Main figures generated in $(FIG_DIR)"

supplementary:
	@echo "Generating supplementary figures and data..."
	cd $(SRC_DIR) && $(PYTHON) reproduce_supplementary_results.py
	@echo "Supplementary results generated in $(SUPP_FIG_DIR) and $(SUPP_DATA_DIR)"

clean:
	@echo "Cleaning up generated outputs..."
	rm -rf "$(FIG_DIR)"/*.eps "$(FIG_DIR)"/*.tif
	rm -rf "$(SUPP_FIG_DIR)"/*.eps "$(SUPP_FIG_DIR)"/*.tif "$(SUPP_FIG_DIR)"/*.png
	rm -rf "$(SUPP_DATA_DIR)"/*.csv "$(SUPP_DATA_DIR)"/*.xlsx
	@echo "Cleanup complete."
