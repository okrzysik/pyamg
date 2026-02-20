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
hierarchy
    Helpers for assembling P/R, coarsening operators, and extending the hierarchy.
smoothers
    Translation of shorthand smoother names to `change_smoothers` specs.
stats
    Per-level timing and diagnostic reporting.
"""

from __future__ import annotations

from . import aggregation, eigs, hierarchy, local_ops, stats, smoothers, subdomains

__all__ = [
    "aggregation",
    "subdomains",
    "local_ops",
    "eigs",
    "hierarchy",
    "smoothers",
    "stats",
]
