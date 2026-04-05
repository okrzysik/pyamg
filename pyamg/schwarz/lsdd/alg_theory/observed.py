"""Observed-cycle diagnostics for LS-DD algebraic-theory drivers."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .models import RandomDistribution, TwoLevelSolverParams, ZERO_NORM_TOL
from .reporting import timer_add
from .spectral import draw_random_vector


def build_observed_two_level_solver(
    *,
    B,
    A,
    BT,
    solver_params: TwoLevelSolverParams,
    zeta: float,
    with_rho: bool,
    with_rho_perp: bool,
    zeta_eff: float,
):
    """Build the standalone two-level solver used for observed q/K estimates."""
    # Local import avoids circular initialization when pyamg imports
    # lsdd internals while least_squares_dd_exp is still being defined.
    from ...least_squares_dd_exp import least_squares_dd_solver_exp

    if with_rho and with_rho_perp:
        raise ValueError("with_rho and with_rho_perp cannot both be True")

    # Use the already-resolved effective damping zeta_eff directly so observed
    # two-grid quantities are evaluated at exactly the same damping used by the
    # theory-chain constants.
    smoother = ("asm", {"domain": "omega", "omega": float(zeta_eff), "withrho": False})
    kappa_value = 0.0 if solver_params.kappa is None else solver_params.kappa
    return least_squares_dd_solver_exp(
        B=B,
        BT=BT,
        A=A,
        presmoother=smoother,
        postsmoother=smoother,
        symmetry=solver_params.symmetry,
        strength=solver_params.strength,
        aggregate=solver_params.aggregate,
        agg_levels=solver_params.agg_levels,
        kappa=kappa_value,
        nev=solver_params.nev,
        threshold=solver_params.threshold,
        mult_threshold=solver_params.mult_threshold,
        min_coarsening=solver_params.min_coarsening,
        max_levels=solver_params.max_levels,
        max_coarse=solver_params.max_coarse,
        max_density=solver_params.max_density,
        filteringA=solver_params.filteringA,
        filteringB=solver_params.filteringB,
        print_info=solver_params.print_info,
        force_row_closure=solver_params.force_row_closure,
        robust_Sker_handling=solver_params.robust_Sker_handling,
        coarse_solver=solver_params.coarse_solver,
    )


def estimate_qobs_homogeneous_two_grid(
    *,
    ml,
    distribution: RandomDistribution,
    seed: int | None,
    n_samples: int,
    n_cycles: int,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[float | None, float | None]:
    """Estimate observed q/K exactly as standalone homogeneous cycle driver."""
    if n_samples <= 0 or n_cycles <= 0:
        return None, None
    if len(ml.levels) < 2:
        return None, None

    A = ml.levels[0].A
    n = int(A.shape[0])
    b0 = np.zeros(n, dtype=A.dtype)
    rng = np.random.default_rng(seed)

    x_warm = np.ones(n, dtype=A.dtype)
    _ = ml.solve(b0, x0=x_warm, tol=0.0, maxiter=1, cycle="V", accel=None)

    q_obs: float | None = None
    for _ in range(int(n_samples)):
        x = draw_random_vector(rng=rng, n=n, distribution=distribution, dtype=A.dtype)
        xAx = float(np.vdot(x, A @ x).real)
        if xAx <= 0.0:
            continue
        x /= np.sqrt(xAx)

        for _ in range(int(n_cycles)):
            xAx_old = float(np.vdot(x, A @ x).real)
            if xAx_old <= ZERO_NORM_TOL:
                break
            n_old = float(np.sqrt(max(xAx_old, 0.0)))

            timer_add(timers, f"{timer_prefix}n_cycle", 1.0)
            t_cycle = perf_counter()
            x_new = ml.solve(b0, x0=x, tol=0.0, maxiter=1, cycle="V", accel=None)
            timer_add(timers, f"{timer_prefix}cycle_sec", perf_counter() - t_cycle)

            xAx_new = float(np.vdot(x_new, A @ x_new).real)
            if xAx_new <= 0.0:
                break
            n_new = float(np.sqrt(max(xAx_new, 0.0)))
            q_k = float(n_new / max(n_old, 1.0e-300))
            q_obs = q_k if q_obs is None else max(q_obs, q_k)
            x = np.asarray(x_new).reshape(-1)

    if q_obs is None:
        return None, None
    if q_obs >= 1.0:
        return float(q_obs), float("inf")
    return float(q_obs), float(1.0 / (1.0 - q_obs))
