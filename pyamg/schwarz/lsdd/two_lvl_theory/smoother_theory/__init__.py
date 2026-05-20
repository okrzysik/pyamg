"""Smoother-side diagnostics utilities for LS-DD experiments."""

from __future__ import annotations

from .diagnostics import (
    SmootherDomainDiagnostics,
    compute_additive_schwarz_lambda_max_from_level,
    compute_smoother_bound_diagnostics_from_level,
)
from .setup import build_one_level_smoother_context
from .smoother_graphs import (
    GraphColoringResult,
    SmootherInteractionGraphs,
    as_csr_pattern,
    build_C_A,
    build_C_G,
    color_count,
    compute_graph_coloring,
    compute_smoother_graphs_from_level,
    domain_incidence_from_level,
    G_row_incidence_from_level,
    graph_difference,
    interaction_matrix_to_networkx,
    simple_graph_pattern,
    validate_coloring,
)

__all__ = [
    "build_one_level_smoother_context",
    "SmootherDomainDiagnostics",
    "compute_additive_schwarz_lambda_max_from_level",
    "compute_smoother_bound_diagnostics_from_level",
    "GraphColoringResult",
    "SmootherInteractionGraphs",
    "as_csr_pattern",
    "build_C_A",
    "build_C_G",
    "color_count",
    "compute_graph_coloring",
    "compute_smoother_graphs_from_level",
    "domain_incidence_from_level",
    "G_row_incidence_from_level",
    "graph_difference",
    "interaction_matrix_to_networkx",
    "simple_graph_pattern",
    "validate_coloring",
]
