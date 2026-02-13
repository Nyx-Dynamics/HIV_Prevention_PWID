"""
Backward-compatibility shim.

The canonical module is structural_barrier_model.py. This file re-exports
the classes under the name expected by cascade_sensitivity_analysis.py.
"""
from structural_barrier_model import (
    StructuralBarrierModel as ArchitecturalBarrierModel,
    PolicyScenario,
    create_policy_scenarios,
    create_pwid_cascade,
    CascadeStep,
)

__all__ = [
    "ArchitecturalBarrierModel",
    "PolicyScenario",
    "create_policy_scenarios",
    "create_pwid_cascade",
    "CascadeStep",
]
