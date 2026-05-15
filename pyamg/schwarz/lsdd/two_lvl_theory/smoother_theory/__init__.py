"""Smoother-side diagnostics utilities for LS-DD experiments."""

from __future__ import annotations

from .diagnostics import (
    SmootherDomainBoundDiagnostics,
    build_row_interaction_matrix,
    compute_additive_schwarz_lambda_max_from_level,
    compute_row_domain_incidence_from_level,
    compute_smoother_bound_diagnostics_from_level,
)
from .setup import build_one_level_smoother_context

__all__ = [
    "build_one_level_smoother_context",
    "SmootherDomainBoundDiagnostics",
    "compute_row_domain_incidence_from_level",
    "build_row_interaction_matrix",
    "compute_additive_schwarz_lambda_max_from_level",
    "compute_smoother_bound_diagnostics_from_level",
]

