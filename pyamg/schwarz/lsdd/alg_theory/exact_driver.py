"""Public refined-chain diagnostics for LS-DD two-level analysis.

This module implements paper-aligned diagnostics for the refined transfer chain
using notation consistent with ``N_AJ``, ``N_YJ_perp``, ``W_Y``, and
``T0..T5 / R1..R5``.
"""

from __future__ import annotations

from time import perf_counter
from warnings import warn

import numpy as np

from .linalg import factor_dense_spd as _factor_dense_spd
from .linalg import solve_factored_dense_spd as _solve_factored_dense_spd
from .models import (
    ASolveMethod,
    ChainDiagnosticsConfig,
    ExactBlockJacobiWapResult,
    ExactSymmetricMetricWapResult,
    LocalProjectionBlock,
    RandomDistribution,
    RefinedChainResult,
    TildeMetricOps,
    TildeProjectionMode,
    TwoLevelSolverParams,
)
from .observed import build_observed_two_level_solver as _build_observed_two_level_solver
from .observed import estimate_qobs_homogeneous_two_grid as _estimate_qobs_homogeneous_two_grid
from .operators import apply_B_block_jacobi as _apply_B_block_jacobi
from .operators import apply_B_tilde as _apply_B_tilde
from .operators import apply_tildeM as _apply_tildeM
from .operators import assemble_block_jacobi_matrix as _assemble_block_jacobi_matrix
from .operators import build_block_jacobi_solver_from_matrix as _build_block_jacobi_solver_from_matrix
from .operators import build_linear_solver as _build_linear_solver
from .operators import build_tilde_metric_ops as _build_tilde_metric_ops
from .operators import build_tilde_projection_data as _build_tilde_projection_data
from .reporting import (
    fmt_sig as _fmt_sig,
    resolve_effective_damping as _resolve_effective_damping,
    safe_ratio as _safe_ratio,
    timer_add as _timer_add,
    transfer_restricted_from_N_AJ_perp_estimate as _transfer_restricted_from_N_AJ_perp_estimate,
    transfer_whole_from_N_AJ_estimate as _transfer_whole_from_N_AJ_estimate,
)
from .setup import build_one_level_lsdd as _build_one_level_lsdd
from .setup import extract_level_diagnostics as _extract_level_diagnostics
from .setup import prepare_local_projection_blocks as _prepare_local_projection_blocks
from .spectral import draw_random_vector as _draw_random_vector
from .spectral import lobpcg_MinvOp as _lobpcg_MinvOp
from .spectral import power_iteration_AinvB as _power_iteration_AinvB
from .spectral import power_MinvOp as _power_MinvOp
from .spectral import projected_power_perp_MinvOp as _projected_power_perp_MinvOp


def _rho_from_K(K: float | None) -> float | None:
    """Return ``rho = 1 - 1/K`` when meaningful, with ``K=inf`` mapped to ``rho=1``."""
    if K is None:
        return None
    Kv = float(K)
    if np.isnan(Kv):
        return None
    if np.isposinf(Kv):
        return 1.0
    if np.isneginf(Kv) or Kv == 0.0:
        return None
    return float(1.0 - 1.0 / Kv)


def compute_refined_chain(
    B,
    A=None,
    BT=None,
    *,
    cfg: ChainDiagnosticsConfig = ChainDiagnosticsConfig(),
    print_constants: bool = False,
    constant_sig_digits: int = 2,
    print_timers: bool = False,
) -> RefinedChainResult:
    """Compute refined-chain diagnostics for the LS-DD block-Jacobi baseline.

    Parameters
    ----------
    B, A, BT
        Fine-level operators passed to LS-DD setup. ``A`` may be omitted when
        it should be formed as ``BT @ B``.
    cfg
        Unified configuration carrying setup, damping, eigensolver, and
        diagnostic options.
    print_constants, constant_sig_digits
        Optional formatted constant report.
    print_timers
        Optional timer report collected from each stage.
    """
    return _compute_refined_chain_core(
        B=B,
        A=A,
        BT=BT,
        cfg=cfg,
        print_constants=print_constants,
        constant_sig_digits=constant_sig_digits,
        print_timers=print_timers,
    )


def _compute_exact_block_jacobi_wap_on_level(
    *,
    level,
    a_solve: ASolveMethod,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
    maxiter: int,
    tol: float,
    miniter: int,
    distribution: RandomDistribution,
    seed: int | None,
    local_blocks: list[LocalProjectionBlock] | None = None,
    nev_per_agg: np.ndarray | None = None,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> ExactBlockJacobiWapResult:
    """Compute exact ``W_M`` using an already prepared LS-DD fine level."""
    A_csr = level.A.tocsr()
    n_fine = int(A_csr.shape[0])
    n_aggs = int(level.n_aggs)
    n_coarse = int(level.P.shape[1])

    if local_blocks is None or nev_per_agg is None:
        blocks_built, nev_built = _prepare_local_projection_blocks(level)
        if local_blocks is None:
            local_blocks = blocks_built
        if nev_per_agg is None:
            nev_per_agg = nev_built
    assert local_blocks is not None
    assert nev_per_agg is not None
    if not local_blocks:
        raise ValueError("No local blocks were constructed; cannot compute W_M")

    def apply_B(x: np.ndarray) -> tuple[np.ndarray, float]:
        return _apply_B_block_jacobi(x=x, local_blocks=local_blocks, n_fine=n_fine)

    solve_A = _build_linear_solver(
        A=A_csr,
        method=a_solve,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )

    rng = np.random.default_rng(seed)
    x0 = _draw_random_vector(rng=rng, n=n_fine, distribution=distribution, dtype=A_csr.dtype)
    lam, rel, anorm, conv = _power_iteration_AinvB(
        A=A_csr,
        apply_B=apply_B,
        solve_A=solve_A,
        x0=x0,
        maxiter=maxiter,
        tol=tol,
        miniter=miniter,
        timers=timers,
        timer_prefix=f"{timer_prefix}power.",
    )
    if lam.size == 0:
        raise ValueError("No eigen-iterations executed")

    tau, mu_max = _extract_level_diagnostics(level)
    return ExactBlockJacobiWapResult(
        W_M=float(lam[-1]),
        rayleigh_history=lam,
        rel_change_history=rel,
        a_norm_history=anorm,
        converged=conv,
        n_iterations=int(lam.size),
        tau=tau,
        mu_max=mu_max,
        nev_per_agg=np.asarray(nev_per_agg, dtype=np.int32).copy(),
        n_fine=n_fine,
        n_aggs=n_aggs,
        n_coarse=n_coarse,
        a_solve=a_solve,
        seed=seed,
    )


def _compute_exact_symmetric_metric_wap_on_level(
    *,
    level,
    zeta_input: float,
    normalize_by_N_AJ: bool,
    N_AJ: float,
    a_solve: ASolveMethod,
    a_cg_rtol: float,
    a_cg_atol: float,
    a_cg_maxiter: int | None,
    h_solve: ASolveMethod,
    h_cg_rtol: float,
    h_cg_atol: float,
    h_cg_maxiter: int | None,
    projection_mode: TildeProjectionMode,
    maxiter: int,
    tol: float,
    miniter: int,
    distribution: RandomDistribution,
    seed: int | None,
    local_blocks: list[LocalProjectionBlock] | None = None,
    nev_per_agg: np.ndarray | None = None,
    tilde_ops: TildeMetricOps | None = None,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> ExactSymmetricMetricWapResult:
    """Compute exact ``W_Y`` with ``Y = \widetilde{J_\zeta}`` on a prepared level."""
    A_csr = level.A.tocsr()
    P = level.P.tocsr()
    P_T = P.T.tocsr()

    n_fine = int(A_csr.shape[0])
    n_aggs = int(level.n_aggs)
    n_coarse = int(P.shape[1])

    if local_blocks is None or nev_per_agg is None:
        blocks_built, nev_built = _prepare_local_projection_blocks(level)
        if local_blocks is None:
            local_blocks = blocks_built
        if nev_per_agg is None:
            nev_per_agg = nev_built
    assert local_blocks is not None
    assert nev_per_agg is not None
    if not local_blocks:
        raise ValueError("No local blocks were constructed; cannot compute W_Y")

    N_AJ_val = float(N_AJ)
    if not np.isfinite(N_AJ_val) or N_AJ_val <= 0.0:
        raise ValueError(f"N_AJ must be positive and finite, got {N_AJ_val!r}")

    zeta_eff = _resolve_effective_damping(
        zeta=float(zeta_input),
        with_rho=bool(normalize_by_N_AJ),
        with_rho_perp=False,
        N_AJ=float(N_AJ_val),
        N_AJ_perp=float(N_AJ_val),
    )

    J = _assemble_block_jacobi_matrix(local_blocks=local_blocks, n_fine=n_fine, dtype=A_csr.dtype)
    if tilde_ops is None:
        tilde_ops = _build_tilde_metric_ops(
            A_csr=A_csr,
            J=J,
            zeta_eff=zeta_eff,
            h_solve=h_solve,
            h_cg_rtol=h_cg_rtol,
            h_cg_atol=h_cg_atol,
            h_cg_maxiter=h_cg_maxiter,
        )

    def apply_tilde_metric(x: np.ndarray) -> np.ndarray:
        return _apply_tildeM(x=x, M=tilde_ops.M_damped, solve_H=tilde_ops.solve_H)

    t_proj = perf_counter()
    C_system, W = _build_tilde_projection_data(P=P, apply_tildeM=apply_tilde_metric, mode=projection_mode)
    _timer_add(timers, f"{timer_prefix}build_projection_sec", perf_counter() - t_proj)

    def apply_B(x: np.ndarray) -> tuple[np.ndarray, float]:
        return _apply_B_tilde(
            x=x,
            P=P,
            P_T=P_T,
            apply_tildeM=apply_tilde_metric,
            C_system=C_system,
            mode=projection_mode,
            W=W,
        )

    solve_A = _build_linear_solver(
        A=A_csr,
        method=a_solve,
        cg_rtol=a_cg_rtol,
        cg_atol=a_cg_atol,
        cg_maxiter=a_cg_maxiter,
    )

    rng = np.random.default_rng(seed)
    x0 = _draw_random_vector(rng=rng, n=n_fine, distribution=distribution, dtype=A_csr.dtype)
    lam, rel, anorm, conv = _power_iteration_AinvB(
        A=A_csr,
        apply_B=apply_B,
        solve_A=solve_A,
        x0=x0,
        maxiter=maxiter,
        tol=tol,
        miniter=miniter,
        timers=timers,
        timer_prefix=f"{timer_prefix}power.",
    )
    if lam.size == 0:
        raise ValueError("No eigen-iterations executed")

    tau, mu_max = _extract_level_diagnostics(level)
    W_Y = float(lam[-1])
    contraction = float(1.0 - 1.0 / W_Y) if W_Y > 0.0 else float("nan")

    return ExactSymmetricMetricWapResult(
        W_Y=W_Y,
        contraction_factor=contraction,
        rayleigh_history=lam,
        rel_change_history=rel,
        a_norm_history=anorm,
        converged=conv,
        n_iterations=int(lam.size),
        tau=tau,
        mu_max=mu_max,
        nev_per_agg=np.asarray(nev_per_agg, dtype=np.int32).copy(),
        n_fine=n_fine,
        n_aggs=n_aggs,
        n_coarse=n_coarse,
        a_solve=a_solve,
        h_solve=h_solve,
        projection_mode=projection_mode,
        N_AJ=float(N_AJ_val),
        zeta_input=float(zeta_input),
        normalize_by_N_AJ=bool(normalize_by_N_AJ),
        zeta_effective=float(zeta_eff),
        seed=seed,
    )


def _compute_refined_chain_core(
    *,
    B,
    A=None,
    BT=None,
    cfg: ChainDiagnosticsConfig,
    print_constants: bool,
    constant_sig_digits: int,
    print_timers: bool,
) -> RefinedChainResult:
    """Internal implementation of ``compute_refined_chain``.

    The routine builds one LS-DD level, estimates ``N_AJ`` and ``N_AJ_perp``,
    derives transfer constants at the configured damping, optionally computes
    exact WAP constants, and packages all chain terms and ratios.
    """
    solver_params = cfg.solver_params
    timers: dict[str, float] | None = {} if (cfg.collect_timers or print_timers) else None
    zeta_input = cfg.zeta_input
    normalize_by_N_AJ = cfg.normalize_by_N_AJ
    normalize_by_N_AJ_perp = cfg.normalize_by_N_AJ_perp
    a_solve = cfg.a_solve
    a_cg_rtol = cfg.a_cg_rtol
    a_cg_atol = cfg.a_cg_atol
    a_cg_maxiter = cfg.a_cg_maxiter
    h_solve = cfg.h_solve
    h_cg_rtol = cfg.h_cg_rtol
    h_cg_atol = cfg.h_cg_atol
    h_cg_maxiter = cfg.h_cg_maxiter
    projection_mode = cfg.projection_mode
    maxiter_W_M = cfg.maxiter_W_M
    tol_W_M = cfg.tol_W_M
    miniter_W_M = cfg.miniter_W_M
    maxiter_W_Y = cfg.maxiter_W_Y
    tol_W_Y = cfg.tol_W_Y
    miniter_W_Y = cfg.miniter_W_Y
    maxiter_W_hat = cfg.maxiter_W_hat
    tol_W_hat = cfg.tol_W_hat
    miniter_W_hat = cfg.miniter_W_hat
    N_AJ_estimator = cfg.N_AJ_estimator
    N_AJ_perp_estimator = cfg.N_AJ_perp_estimator
    N_AJ_block_size = cfg.N_AJ_block_size
    N_AJ_perp_block_size = cfg.N_AJ_perp_block_size
    maxiter_N_AJ = cfg.maxiter_N_AJ
    tol_N_AJ = cfg.tol_N_AJ
    miniter_N_AJ = cfg.miniter_N_AJ
    maxiter_N_AJ_perp = cfg.maxiter_N_AJ_perp
    tol_N_AJ_perp = cfg.tol_N_AJ_perp
    miniter_N_AJ_perp = cfg.miniter_N_AJ_perp
    distribution = cfg.distribution
    seed = cfg.seed
    estimate_q_obs_solve = cfg.estimate_q_obs_solve
    qobs_n_samples = cfg.qobs_n_samples
    qobs_n_cycles = cfg.qobs_n_cycles
    compute_exact_W_M = cfg.compute_exact_W_M
    compute_exact_W_Y = cfg.compute_exact_W_Y
    compute_exact_W_hat = cfg.compute_exact_W_hat
    t_total = perf_counter()

    t_stage = perf_counter()
    levels = _build_one_level_lsdd(
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
        return_levels=True,
    )
    level = levels[0]
    _timer_add(timers, "stage.setup_level_sec", perf_counter() - t_stage)

    t_stage = perf_counter()
    local_blocks, nev_per_agg = _prepare_local_projection_blocks(level)
    _timer_add(timers, "stage.prepare_local_blocks_sec", perf_counter() - t_stage)
    if not local_blocks:
        raise ValueError("No local blocks were constructed; cannot run block-Jacobi baseline")

    A_csr = level.A.tocsr()
    P = level.P.tocsr()
    P_T = P.T.tocsr()
    n_fine = int(A_csr.shape[0])

    t_stage = perf_counter()
    J = _assemble_block_jacobi_matrix(local_blocks=local_blocks, n_fine=n_fine, dtype=A_csr.dtype)
    MP = (J @ P).tocsr()
    C_M = (P_T @ MP).toarray()
    C_M = 0.5 * (C_M + C_M.T)
    C_M_sys = _factor_dense_spd(C_M)
    solve_J = _build_block_jacobi_solver_from_matrix(J=J, local_blocks=local_blocks)
    _timer_add(timers, "stage.build_block_jacobi_ops_sec", perf_counter() - t_stage)

    def apply_J(x: np.ndarray) -> np.ndarray:
        return np.asarray(J @ x).reshape(-1)

    def apply_QJ(x: np.ndarray) -> np.ndarray:
        y = np.asarray(J @ x).reshape(-1)
        rhs = np.asarray(P_T @ y).reshape(-1)
        alpha = _solve_factored_dense_spd(C_M_sys, rhs)
        return np.asarray(x).reshape(-1) - np.asarray(P @ alpha).reshape(-1)

    t_stage = perf_counter()
    if normalize_by_N_AJ and normalize_by_N_AJ_perp:
        raise ValueError("normalize_by_N_AJ and normalize_by_N_AJ_perp cannot both be True")

    N_AJ_seed = None if seed is None else int(seed) + 13
    N_AJ_estimator_used = str(N_AJ_estimator)
    if N_AJ_estimator == "lobpcg":
        try:
            N_AJ, N_AJ_hist, N_AJ_rel_hist, N_AJ_abs_res_hist, N_AJ_rel_res_hist, N_AJ_conv = _lobpcg_MinvOp(
                A=A_csr,
                J=J,
                solve_J=solve_J,
                n=n_fine,
                block_size=N_AJ_block_size,
                maxiter=maxiter_N_AJ,
                tol=tol_N_AJ,
                distribution=distribution,
                seed=N_AJ_seed,
                Y=None,
                apply_QM_for_residual=None,
                timers=timers,
                timer_prefix="N_AJ.",
            )
        except Exception as exc:
            warn(
                f"LOBPCG N_AJ estimation failed ({exc!r}); falling back to power iteration.",
                RuntimeWarning,
            )
            N_AJ_estimator_used = "power_fallback"
            N_AJ, N_AJ_hist, N_AJ_rel_hist, N_AJ_abs_res_hist, N_AJ_rel_res_hist, N_AJ_conv = _power_MinvOp(
                apply_op=lambda x: np.asarray(A_csr @ x).reshape(-1),
                apply_J=apply_J,
                solve_J=solve_J,
                n=n_fine,
                dtype=A_csr.dtype,
                maxiter=maxiter_N_AJ,
                tol=tol_N_AJ,
                miniter=miniter_N_AJ,
                distribution=distribution,
                seed=N_AJ_seed,
                timers=timers,
                timer_prefix="N_AJ.",
            )
    else:
        N_AJ, N_AJ_hist, N_AJ_rel_hist, N_AJ_abs_res_hist, N_AJ_rel_res_hist, N_AJ_conv = _power_MinvOp(
            apply_op=lambda x: np.asarray(A_csr @ x).reshape(-1),
            apply_J=apply_J,
            solve_J=solve_J,
            n=n_fine,
            dtype=A_csr.dtype,
            maxiter=maxiter_N_AJ,
            tol=tol_N_AJ,
            miniter=miniter_N_AJ,
            distribution=distribution,
            seed=N_AJ_seed,
            timers=timers,
            timer_prefix="N_AJ.",
        )
    _timer_add(timers, "stage.N_AJ_total_sec", perf_counter() - t_stage)

    t_stage = perf_counter()
    N_AJ_perp_method_eff = N_AJ_estimator if N_AJ_perp_estimator is None else N_AJ_perp_estimator
    N_AJ_perp_seed = None if seed is None else int(seed) + 17
    N_AJ_perp_estimator_used = str(N_AJ_perp_method_eff)
    if N_AJ_perp_method_eff == "lobpcg":
        try:
            N_AJ_perp, N_AJ_perp_hist, N_AJ_perp_rel_hist, N_AJ_perp_abs_res_hist, N_AJ_perp_rel_res_hist, N_AJ_perp_conv = _lobpcg_MinvOp(
                A=A_csr,
                J=J,
                solve_J=solve_J,
                n=n_fine,
                block_size=N_AJ_perp_block_size,
                maxiter=maxiter_N_AJ_perp,
                tol=tol_N_AJ_perp,
                distribution=distribution,
                seed=N_AJ_perp_seed,
                Y=P,
                apply_QM_for_residual=apply_QJ,
                timers=timers,
                timer_prefix="N_AJ_perp.",
            )
        except Exception as exc:
            warn(
                f"LOBPCG N_AJ_perp estimation failed ({exc!r}); falling back to projected power iteration.",
                RuntimeWarning,
            )
            N_AJ_perp_estimator_used = "power_fallback"
            N_AJ_perp, N_AJ_perp_hist, N_AJ_perp_rel_hist, N_AJ_perp_abs_res_hist, N_AJ_perp_rel_res_hist, N_AJ_perp_conv = _projected_power_perp_MinvOp(
                apply_op=lambda x: np.asarray(A_csr @ x).reshape(-1),
                apply_QM=apply_QJ,
                apply_J=apply_J,
                solve_J=solve_J,
                n=n_fine,
                dtype=A_csr.dtype,
                maxiter=maxiter_N_AJ_perp,
                tol=tol_N_AJ_perp,
                miniter=miniter_N_AJ_perp,
                distribution=distribution,
                seed=N_AJ_perp_seed,
                timers=timers,
                timer_prefix="N_AJ_perp.",
            )
    else:
        N_AJ_perp, N_AJ_perp_hist, N_AJ_perp_rel_hist, N_AJ_perp_abs_res_hist, N_AJ_perp_rel_res_hist, N_AJ_perp_conv = _projected_power_perp_MinvOp(
            apply_op=lambda x: np.asarray(A_csr @ x).reshape(-1),
            apply_QM=apply_QJ,
            apply_J=apply_J,
            solve_J=solve_J,
            n=n_fine,
            dtype=A_csr.dtype,
            maxiter=maxiter_N_AJ_perp,
            tol=tol_N_AJ_perp,
            miniter=miniter_N_AJ_perp,
            distribution=distribution,
            seed=N_AJ_perp_seed,
            timers=timers,
            timer_prefix="N_AJ_perp.",
        )
    _timer_add(timers, "stage.N_AJ_perp_total_sec", perf_counter() - t_stage)

    zeta_ref = _resolve_effective_damping(
        zeta=float(zeta_input),
        with_rho=bool(normalize_by_N_AJ),
        with_rho_perp=bool(normalize_by_N_AJ_perp),
        N_AJ=float(N_AJ),
        N_AJ_perp=float(N_AJ_perp),
    )
    N_YJ = float(_transfer_whole_from_N_AJ_estimate(
        N_AJ=float(N_AJ),
        zeta_input=float(zeta_input),
        with_rho=bool(normalize_by_N_AJ),
        with_rho_perp=bool(normalize_by_N_AJ_perp),
        N_AJ_perp_scale=float(N_AJ_perp),
    ))
    N_YJ_perp = float(
        _transfer_restricted_from_N_AJ_perp_estimate(
            N_AJ_perp=float(N_AJ_perp),
            zeta_eff=float(zeta_ref),
        )
    )

    tilde_ops: TildeMetricOps | None = None
    if compute_exact_W_Y or compute_exact_W_hat:
        t_stage = perf_counter()
        tilde_ops = _build_tilde_metric_ops(
            A_csr=A_csr,
            J=J,
            zeta_eff=float(zeta_ref),
            h_solve=h_solve,
            h_cg_rtol=h_cg_rtol,
            h_cg_atol=h_cg_atol,
            h_cg_maxiter=h_cg_maxiter,
        )
        _timer_add(timers, "stage.build_tilde_metric_sec", perf_counter() - t_stage)

    tau, mu_max = _extract_level_diagnostics(level)

    q_obs_solve = None
    K_obs_solve = None
    ml_obs = None
    if estimate_q_obs_solve:
        t_stage = perf_counter()
        q_seed = None if seed is None else int(seed) + 101
        ml_obs = _build_observed_two_level_solver(
            B=B,
            A=A,
            BT=BT,
            solver_params=solver_params,
            zeta=float(zeta_input),
            with_rho=bool(normalize_by_N_AJ),
            with_rho_perp=bool(normalize_by_N_AJ_perp),
            zeta_eff=float(zeta_ref),
        )
        q_obs_solve, K_obs_solve = _estimate_qobs_homogeneous_two_grid(
            ml=ml_obs,
            distribution=distribution,
            seed=q_seed,
            n_samples=qobs_n_samples,
            n_cycles=qobs_n_cycles,
            timers=timers,
            timer_prefix="qobs.",
        )
        _timer_add(timers, "stage.qobs_total_sec", perf_counter() - t_stage)

    exact_W_M_result: ExactBlockJacobiWapResult | None = None
    W_M: float | None = None
    if compute_exact_W_M:
        t_stage = perf_counter()
        exact_W_M_result = _compute_exact_block_jacobi_wap_on_level(
            level=level,
            a_solve=a_solve,
            cg_rtol=a_cg_rtol,
            cg_atol=a_cg_atol,
            cg_maxiter=a_cg_maxiter,
            maxiter=maxiter_W_M,
            tol=tol_W_M,
            miniter=miniter_W_M,
            distribution=distribution,
            seed=seed,
            local_blocks=local_blocks,
            nev_per_agg=nev_per_agg,
            timers=timers,
            timer_prefix="W_M.",
        )
        _timer_add(timers, "stage.W_M_total_sec", perf_counter() - t_stage)
        W_M = float(exact_W_M_result.W_M)

    exact_W_Y_result: ExactSymmetricMetricWapResult | None = None
    W_Y: float | None = None
    if compute_exact_W_Y:
        t_stage = perf_counter()
        seed_Y = None if seed is None else int(seed) + 1
        exact_W_Y_result = _compute_exact_symmetric_metric_wap_on_level(
            level=level,
            zeta_input=float(zeta_input),
            normalize_by_N_AJ=bool(normalize_by_N_AJ),
            N_AJ=float(N_AJ),
            a_solve=a_solve,
            a_cg_rtol=a_cg_rtol,
            a_cg_atol=a_cg_atol,
            a_cg_maxiter=a_cg_maxiter,
            h_solve=h_solve,
            h_cg_rtol=h_cg_rtol,
            h_cg_atol=h_cg_atol,
            h_cg_maxiter=h_cg_maxiter,
            projection_mode=projection_mode,
            maxiter=maxiter_W_Y,
            tol=tol_W_Y,
            miniter=miniter_W_Y,
            distribution=distribution,
            seed=seed_Y,
            local_blocks=local_blocks,
            nev_per_agg=nev_per_agg,
            tilde_ops=tilde_ops,
            timers=timers,
            timer_prefix="W_Y.",
        )
        _timer_add(timers, "stage.W_Y_total_sec", perf_counter() - t_stage)
        W_Y = float(exact_W_Y_result.W_Y)

    W_hat_Y: float | None = None
    W_hat_rel_hist: np.ndarray | None = None
    W_hat_converged: bool | None = None
    W_hat_n_iterations: int | None = None
    if compute_exact_W_hat:
        t_stage = perf_counter()
        if tilde_ops is None:
            raise ValueError("Internal error: missing tilde_ops for W_hat_Y")

        def apply_tilde_metric(x: np.ndarray) -> np.ndarray:
            return _apply_tildeM(x=x, M=tilde_ops.M_damped, solve_H=tilde_ops.solve_H)

        def apply_QJ_T(y: np.ndarray) -> np.ndarray:
            rhs = np.asarray(P_T @ y).reshape(-1)
            alpha = _solve_factored_dense_spd(C_M_sys, rhs)
            return np.asarray(y).reshape(-1) - np.asarray(MP @ alpha).reshape(-1)

        def apply_Bhat(x: np.ndarray) -> tuple[np.ndarray, float]:
            r = apply_QJ(x)
            y = apply_tilde_metric(r)
            z = apply_QJ_T(y)
            xBhatx = float(np.vdot(x, z).real)
            if xBhatx < 0.0 and abs(xBhatx) < 1e-12:
                xBhatx = 0.0
            return z, xBhatx

        solve_A = _build_linear_solver(
            A=A_csr,
            method=a_solve,
            cg_rtol=a_cg_rtol,
            cg_atol=a_cg_atol,
            cg_maxiter=a_cg_maxiter,
        )

        rng = np.random.default_rng(None if seed is None else int(seed) + 2)
        x0_hat = _draw_random_vector(rng=rng, n=n_fine, distribution=distribution, dtype=A_csr.dtype)
        lam_hat, _rel_hat, _an_hat, _conv_hat = _power_iteration_AinvB(
            A=A_csr,
            apply_B=apply_Bhat,
            solve_A=solve_A,
            x0=x0_hat,
            maxiter=maxiter_W_hat,
            tol=tol_W_hat,
            miniter=miniter_W_hat,
            timers=timers,
            timer_prefix="W_hat.power.",
        )
        if lam_hat.size == 0:
            raise ValueError("No iterations executed for W_hat_Y")
        W_hat_Y = float(lam_hat[-1])
        W_hat_rel_hist = np.asarray(_rel_hat, dtype=float)
        W_hat_converged = bool(_conv_hat)
        W_hat_n_iterations = int(lam_hat.size)
        _timer_add(timers, "stage.W_hat_total_sec", perf_counter() - t_stage)

    W_Mtilde = W_Y
    K_TG = W_Mtilde
    K_TG_obs = K_obs_solve
    rho_TG = _rho_from_K(K_TG)
    rho_TG_obs = _rho_from_K(K_TG_obs)

    T0 = K_TG
    T1 = W_hat_Y
    T2 = None if (W_M is None or not np.isfinite(N_YJ_perp)) else float(N_YJ_perp * W_M)
    T3 = None if (mu_max is None or not np.isfinite(N_YJ_perp)) else float(N_YJ_perp * mu_max)
    T4 = None if (tau is None or not np.isfinite(N_YJ_perp)) else float(N_YJ_perp * tau)
    T5 = None if (tau is None or not np.isfinite(N_YJ)) else float(N_YJ * tau)

    R1 = _safe_ratio(T1, T0)
    R2 = _safe_ratio(T2, T1)
    R3 = _safe_ratio(T3, T2)
    R4 = _safe_ratio(T4, T3)
    R5 = _safe_ratio(T5, T4)

    _timer_add(timers, "stage.total_sec", perf_counter() - t_total)

    if print_constants:
        sd = int(constant_sig_digits)
        print("Refined chain diagnostics:")
        print("Smoother-related constants:")
        print(f"  normalize_by_N_AJ       : {bool(normalize_by_N_AJ)}")
        print(f"  normalize_by_N_AJ_perp  : {bool(normalize_by_N_AJ_perp)}")
        print(f"  N_AJ                    : {_fmt_sig(N_AJ, sd)}")
        print(f"  N_AJ_perp               : {_fmt_sig(N_AJ_perp, sd)}")
        print(f"  zeta_ref                : {_fmt_sig(zeta_ref, sd)}")
        print(f"  N_MtildeJ               : {_fmt_sig(N_YJ, sd)}")
        print(f"  N_MtildeJ_perp          : {_fmt_sig(N_YJ_perp, sd)}")
        print("Threshold-related constants:")
        print(f"  tau                     : {_fmt_sig(tau, sd)}")
        print(f"  mu_max                  : {_fmt_sig(mu_max, sd)}")
        print("WAP constants:")
        print(f"  W_Mtilde                : {_fmt_sig(W_Mtilde, sd)}")
        print(f"  W_hat_Mtilde_from_J     : {_fmt_sig(W_hat_Y, sd)}")
        print(f"  W_J                     : {_fmt_sig(W_M, sd)}")
        print("Chain terms:")
        print(f"  T0                      : {_fmt_sig(T0, sd)}")
        print(f"  T1                      : {_fmt_sig(T1, sd)}")
        print(f"  T2                      : {_fmt_sig(T2, sd)}")
        print(f"  T3                      : {_fmt_sig(T3, sd)}")
        print(f"  T4                      : {_fmt_sig(T4, sd)}")
        print(f"  T5                      : {_fmt_sig(T5, sd)}")
        print("Chain ratios:")
        print(f"  R1                      : {_fmt_sig(R1, sd)}")
        print(f"  R2                      : {_fmt_sig(R2, sd)}")
        print(f"  R3                      : {_fmt_sig(R3, sd)}")
        print(f"  R4                      : {_fmt_sig(R4, sd)}")
        print(f"  R5                      : {_fmt_sig(R5, sd)}")
        print("Two-grid constants:")
        print(f"  K_TG                    : {_fmt_sig(K_TG, sd)}")
        print(f"  K_TG_obs                : {_fmt_sig(K_TG_obs, sd)}")
        print(f"  rho_TG                  : {_fmt_sig(rho_TG, sd)}")
        print(f"  rho_TG_obs              : {_fmt_sig(rho_TG_obs, sd)}")
        if ml_obs is not None:
            print("Observed two-grid hierarchy:")
            print(ml_obs)
        print("Eigen solver diagnostics:")
        print(
            f"  N_AJ                    : method={N_AJ_estimator_used}, "
            f"conv={bool(N_AJ_conv)}, iters={int(np.asarray(N_AJ_hist).size)}, "
            f"last_rel_change={_fmt_sig(None if np.asarray(N_AJ_rel_hist).size == 0 else float(np.asarray(N_AJ_rel_hist)[-1]), sd)}, "
            f"last_abs_res={_fmt_sig(None if np.asarray(N_AJ_abs_res_hist).size == 0 else float(np.asarray(N_AJ_abs_res_hist)[-1]), sd)}, "
            f"last_rel_res={_fmt_sig(None if np.asarray(N_AJ_rel_res_hist).size == 0 else float(np.asarray(N_AJ_rel_res_hist)[-1]), sd)}"
        )
        print(
            f"  N_AJ_perp               : method={N_AJ_perp_estimator_used}, "
            f"conv={bool(N_AJ_perp_conv)}, iters={int(np.asarray(N_AJ_perp_hist).size)}, "
            f"last_rel_change={_fmt_sig(None if np.asarray(N_AJ_perp_rel_hist).size == 0 else float(np.asarray(N_AJ_perp_rel_hist)[-1]), sd)}, "
            f"last_abs_res={_fmt_sig(None if np.asarray(N_AJ_perp_abs_res_hist).size == 0 else float(np.asarray(N_AJ_perp_abs_res_hist)[-1]), sd)}, "
            f"last_rel_res={_fmt_sig(None if np.asarray(N_AJ_perp_rel_res_hist).size == 0 else float(np.asarray(N_AJ_perp_rel_res_hist)[-1]), sd)}"
        )
        print(
            f"  W_J                     : conv={None if exact_W_M_result is None else bool(exact_W_M_result.converged)}, "
            f"iters={None if exact_W_M_result is None else int(exact_W_M_result.n_iterations)}, "
            f"last_rel_change={_fmt_sig(None if (exact_W_M_result is None or exact_W_M_result.rel_change_history.size == 0) else float(exact_W_M_result.rel_change_history[-1]), sd)}"
        )
        print(
            f"  W_Mtilde                : conv={None if exact_W_Y_result is None else bool(exact_W_Y_result.converged)}, "
            f"iters={None if exact_W_Y_result is None else int(exact_W_Y_result.n_iterations)}, "
            f"last_rel_change={_fmt_sig(None if (exact_W_Y_result is None or exact_W_Y_result.rel_change_history.size == 0) else float(exact_W_Y_result.rel_change_history[-1]), sd)}"
        )
        print(
            f"  W_hat_Mtilde_from_J     : conv={W_hat_converged}, "
            f"iters={W_hat_n_iterations}, "
            f"last_rel_change={_fmt_sig(None if (W_hat_rel_hist is None or W_hat_rel_hist.size == 0) else float(W_hat_rel_hist[-1]), sd)}"
        )

    if print_timers and timers is not None:
        print("Timing breakdown (seconds):")
        for key in sorted(timers, key=timers.get, reverse=True):
            val = timers[key]
            if key.endswith("n_iter") or key.endswith("n_cycle"):
                print(f"  {key}: {int(round(val))}")
            else:
                print(f"  {key}: {val:.6g}")

    return RefinedChainResult(
        tau=tau,
        mu_max=mu_max,
        N_AJ=float(N_AJ),
        N_AJ_perp=float(N_AJ_perp),
        zeta_ref=float(zeta_ref),
        N_YJ=float(N_YJ),
        N_YJ_perp=float(N_YJ_perp),
        W_M=W_M,
        W_Y=W_Y,
        W_hat_Y=W_hat_Y,
        T0=T0,
        T1=T1,
        T2=T2,
        T3=T3,
        T4=T4,
        T5=T5,
        R1=R1,
        R2=R2,
        R3=R3,
        R4=R4,
        R5=R5,
        q_obs_solve=q_obs_solve,
        K_obs_solve=K_obs_solve,
        N_AJ_history=np.asarray(N_AJ_hist, dtype=float),
        N_AJ_rel_change_history=np.asarray(N_AJ_rel_hist, dtype=float),
        N_AJ_abs_residual_history=np.asarray(N_AJ_abs_res_hist, dtype=float),
        N_AJ_rel_residual_history=np.asarray(N_AJ_rel_res_hist, dtype=float),
        N_AJ_converged=bool(N_AJ_conv),
        N_AJ_n_iterations=int(np.asarray(N_AJ_hist).size),
        N_AJ_estimator_used=str(N_AJ_estimator_used),
        N_AJ_perp_history=np.asarray(N_AJ_perp_hist, dtype=float),
        N_AJ_perp_rel_change_history=np.asarray(N_AJ_perp_rel_hist, dtype=float),
        N_AJ_perp_abs_residual_history=np.asarray(N_AJ_perp_abs_res_hist, dtype=float),
        N_AJ_perp_rel_residual_history=np.asarray(N_AJ_perp_rel_res_hist, dtype=float),
        N_AJ_perp_converged=bool(N_AJ_perp_conv),
        N_AJ_perp_n_iterations=int(np.asarray(N_AJ_perp_hist).size),
        N_AJ_perp_estimator_used=str(N_AJ_perp_estimator_used),
        exact_W_M_result=exact_W_M_result,
        exact_W_Y_result=exact_W_Y_result,
        timings=timers,
    )


__all__ = [
    "TwoLevelSolverParams",
    "ChainDiagnosticsConfig",
    "RefinedChainResult",
    "ExactBlockJacobiWapResult",
    "ExactSymmetricMetricWapResult",
    "compute_refined_chain",
]
