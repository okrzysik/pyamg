"""LS–AMG–DD (least-squares algebraic multigrid domain decomposition) internals.

This package contains the modularized building blocks for the experimental
LS–AMG–DD solver (`pyamg.schwarz.least_squares_dd_exp`).

Modules
-------
aggregation
    Strength-of-connection and aggregation (AggOp) construction.
subdomains
    Construction of nonoverlapping/overlapping subdomains and partition-of-unity.
local_ops
    Extraction of local principal submatrices and local outer-product terms.
eigs
    Per-aggregate generalized eigenvalue problems and eigenvector selection.
basis_scaling
    Optional post-GEP aggregate-wise basis scaling for prolongation triplets.
hierarchy
    Helpers for assembling P/R, coarsening operators, and extending the hierarchy.
smoothers
    Translation of shorthand smoother names to `change_smoothers` specs.
stats
    Per-level timing and diagnostic reporting.
alg_theory
    Theory-focused drivers and design notes.
"""

from __future__ import annotations

from . import alg_theory, aggregation, basis_scaling, eigs, hierarchy, local_ops, smoothers, stats, subdomains, two_lvl_theory

__all__ = [
    "alg_theory",
    "aggregation",
    "basis_scaling",
    "subdomains",
    "local_ops",
    "eigs",
    "hierarchy",
    "smoothers",
    "stats",
    "two_lvl_theory",
]
