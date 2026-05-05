"""Two-level theory experiment utilities for LS-DD.

This package is the clean, isolated home for simple two-level metrics used by
research scripts. It is intentionally independent of the heavier
``lsdd.alg_theory`` code path.
"""

from __future__ import annotations

from .observed import (
    ObservedTwoGridConfig,
    ObservedTwoGridResult,
    TheorySmoother,
    compute_observed_two_grid_constant,
)
from .setup import build_one_level_lsdd_context
from .threshold import TauMaxResult, compute_tau_max, extract_tau_max_from_level
from .types import TwoLevelSolverParams
from .wj import WJConfig, WJResult, compute_W_J, compute_W_J_from_level

__all__ = [
    "TheorySmoother",
    "TwoLevelSolverParams",
    "ObservedTwoGridConfig",
    "ObservedTwoGridResult",
    "compute_observed_two_grid_constant",
    "build_one_level_lsdd_context",
    "TauMaxResult",
    "extract_tau_max_from_level",
    "compute_tau_max",
    "WJConfig",
    "WJResult",
    "compute_W_J",
    "compute_W_J_from_level",
]
