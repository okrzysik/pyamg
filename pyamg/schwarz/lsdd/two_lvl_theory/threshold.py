"""Threshold-side metrics for ``lsdd.two_lvl_theory``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .setup import build_one_level_lsdd_context
from .types import TwoLevelSolverParams


@dataclass(slots=True, frozen=True)
class TauMaxResult:
    """Threshold diagnostics from one LS-DD setup level."""

    tau: float | None
    tau_max: float | None


def extract_tau_max_from_level(level) -> TauMaxResult:
    """Extract ``tau`` and ``tau_max`` from a prepared LS-DD fine level."""
    tau = None
    if level.eigs.threshold is not None:
        tau = float(level.eigs.threshold)

    tau_max = None
    if level.eigs.first_discarded is not None:
        fd = np.asarray(level.eigs.first_discarded, dtype=float)
        fd = fd[~np.isnan(fd)]
        if fd.size:
            tau_max = float(np.max(fd))

    return TauMaxResult(tau=tau, tau_max=tau_max)


def compute_tau_max(
    *,
    B,
    A,
    BT,
    solver_params: TwoLevelSolverParams,
) -> TauMaxResult:
    """Build one LS-DD level and compute ``tau_max``."""
    level = build_one_level_lsdd_context(
        B=B,
        A=A,
        BT=BT,
        symmetry=solver_params.symmetry,
        strength=solver_params.strength,
        aggregate=solver_params.aggregate,
        agg_levels=solver_params.agg_levels,
        kappa=solver_params.kappa,
        nev=solver_params.nev,
        threshold=solver_params.threshold,
        mult_threshold=solver_params.mult_threshold,
        min_coarsening=solver_params.min_coarsening,
        filteringA=solver_params.filteringA,
        filteringB=solver_params.filteringB,
        print_info=solver_params.print_info,
        force_row_closure=solver_params.force_row_closure,
        robust_Sker_handling=solver_params.robust_Sker_handling,
        max_levels=solver_params.max_levels,
        max_coarse=solver_params.max_coarse,
        max_density=solver_params.max_density,
    )
    return extract_tau_max_from_level(level)
