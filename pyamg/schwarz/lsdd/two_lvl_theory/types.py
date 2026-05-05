"""Shared configuration types for ``lsdd.two_lvl_theory``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..types import FilteringSpec


@dataclass(slots=True, frozen=True)
class TwoLevelSolverParams:
    """Two-level LS-DD setup parameters used by simplified theory experiments."""

    symmetry: Literal["symmetric", "hermitian"] = "hermitian"
    strength: Any = None
    aggregate: Any = "standard"
    agg_levels: int = 2
    kappa: float | list[float] | None = None
    nev: int | None = None
    threshold: float | None = None
    mult_threshold: float | list[float | None] | None = None
    min_coarsening: int | list[int] | None = None
    filteringA: FilteringSpec | None = (False, 0.0)
    filteringB: FilteringSpec | None = (False, 0.0)
    print_info: bool = False
    force_row_closure: bool = True
    robust_Sker_handling: bool = True
    max_levels: int = 2
    max_coarse: int = 10
    max_density: float = 1.0
    coarse_solver: Any = "splu"
