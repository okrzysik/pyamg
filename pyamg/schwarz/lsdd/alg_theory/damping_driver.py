"""Block-Jacobi damping sweep diagnostics for LS-DD algebraic theory.

This module implements the numerical stage focused on damping dependence for
block Jacobi, using a *normalized* damping coordinate ``zeta`` in ``(0, 2)``
defined by

``zeta = zeta_eff * N_AJ``.

Equivalently, the effective damping used by the smoother is

``zeta_eff = zeta / N_AJ``.

For a fixed coarse space and a sweep of ``zeta``, the driver computes only the
quantities needed for damping optimization studies:

- Whole-space and restricted spectral edges: ``N_AJ`` and ``N_AJ_perp``.
- Scalar sandwich models as functions of the normalized ``zeta``:
  - ``L_J(zeta) = 1 / (zeta_eff * (2 - zeta_eff * N_AJ_perp))``
  - ``U_J(zeta) = 1 / (zeta_eff * (2 - zeta_eff * N_AJ))``
    with ``zeta_eff = zeta / N_AJ``.
- Sampled restricted sharp target:
  - ``Phi_J(zeta) = N_{tilde(J_zeta),J}^perp``
- Observed two-grid quantities from actual solver runs:
  - ``q_obs(zeta)``, ``K_obs(zeta)``, ``rho_obs(zeta)``.

The implementation intentionally avoids computing unrelated refined-chain terms
for performance.
"""

from __future__ import annotations

from time import perf_counter
from warnings import warn

import numpy as np

from .linalg import factor_dense_spd as _factor_dense_spd
from .linalg import solve_factored_dense_spd as _solve_factored_dense_spd
from .models import DampingSweepConfig, DampingSweepResult
from .observed import build_observed_two_level_solver as _build_observed_two_level_solver
from .observed import estimate_qobs_homogeneous_two_grid as _estimate_qobs_homogeneous_two_grid
from .operators import apply_B_block_jacobi as _apply_B_block_jacobi
from .operators import apply_tildeM as _apply_tildeM
from .operators import assemble_block_jacobi_matrix as _assemble_block_jacobi_matrix
from .operators import build_block_jacobi_solver_from_matrix as _build_block_jacobi_solver_from_matrix
from .operators import build_linear_solver as _build_linear_solver
from .operators import build_tilde_metric_ops as _build_tilde_metric_ops
from .reporting import fmt_sig as _fmt_sig
from .reporting import timer_add as _timer_add
from .setup import build_one_level_lsdd as _build_one_level_lsdd
from .setup import extract_level_diagnostics as _extract_level_diagnostics
from .setup import prepare_local_projection_blocks as _prepare_local_projection_blocks
from .spectral import draw_random_vector as _draw_random_vector
from .spectral import lobpcg_MinvOp as _lobpcg_MinvOp
from .spectral import power_iteration_AinvB as _power_iteration_AinvB
from .spectral import power_MinvOp as _power_MinvOp
from .spectral import projected_power_perp_MinvOp as _projected_power_perp_MinvOp


def _zeta_grid_from_config(cfg: DampingSweepConfig) -> np.ndarray:
    """Return validated normalized damping grid in ``(0, 2)``."""
    if cfg.zeta_values is not None:
        zeta = np.asarray(cfg.zeta_values, dtype=float).reshape(-1)
        if zeta.size == 0:
            raise ValueError("zeta_values must contain at least one value")
    else:
        n = int(cfg.n_zeta)
        if n <= 0:
            raise ValueError("n_zeta must be positive")
        zmin = float(cfg.zeta_min)
        zmax = float(cfg.zeta_max)
        if not (0.0 < zmin < 2.0 and 0.0 < zmax < 2.0):
            raise ValueError("zeta_min and zeta_max must lie strictly in (0, 2)")
        if zmax <= zmin:
            raise ValueError("zeta_max must be greater than zeta_min")
        zeta = np.linspace(zmin, zmax, n, dtype=float)

    if np.any(~np.isfinite(zeta)):
        raise ValueError("zeta grid contains non-finite values")
    if np.any(zeta <= 0.0) or np.any(zeta >= 2.0):
        raise ValueError("all zeta values must lie strictly in (0, 2)")
    return zeta


def _inv_positive(x: float) -> float | None:
    """Return ``1/x`` for finite positive ``x``, else ``None``."""
    xv = float(x)
    if not np.isfinite(xv) or xv <= 0.0:
        return None
    return float(1.0 / xv)


def _argmin_finite(zeta: np.ndarray, values: np.ndarray) -> float | None:
    """Return ``zeta[argmin(values)]`` over finite entries, else ``None``."""
    vv = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(vv)
    if not np.any(mask):
        return None
    zz = np.asarray(zeta, dtype=float).reshape(-1)
    idx_local = int(np.argmin(vv[mask]))
    return float(zz[mask][idx_local])


def compute_block_jacobi_damping_sweep(
    B,
    A=None,
    BT=None,
    *,
    cfg: DampingSweepConfig = DampingSweepConfig(),
    print_constants: bool = False,
    constant_sig_digits: int = 3,
    print_timers: bool = False,
) -> DampingSweepResult:
    """Run the block-Jacobi damping sweep used in numerical optimization studies.

    Parameters
    ----------
    B, A, BT
        Fine-level operators passed to LS-DD setup. If ``A`` is ``None``, the
        LS-DD setup path forms it from ``B`` and ``BT``.
    cfg
        Sweep configuration. The grid is interpreted as normalized damping
        values in ``(0, 2)`` with ``zeta = zeta_eff * N_AJ``.
    print_constants, constant_sig_digits
        Optional formatted diagnostics printout.
    print_timers
        Optional timing summary.

    Returns
    -------
    DampingSweepResult
        Arrays for ``L_J/U_J/Phi_J`` and observed ``q/K/rho`` over the zeta
        grid, along with one-shot spectral-edge estimates and optimizer
        summaries.
    """
    timers: dict[str, float] | None = {} if (cfg.collect_timers or print_timers) else None
    t_total = perf_counter()

    zeta_vals = _zeta_grid_from_config(cfg)
    sp = cfg.solver_params

    t_stage = perf_counter()
    levels = _build_one_level_lsdd(
        B=B,
        A=A,
        BT=BT,
        symmetry=sp.symmetry,
        strength=sp.strength,
        aggregate=sp.aggregate,
        agg_levels=sp.agg_levels,
        kappa=sp.kappa,
        nev=sp.nev,
        threshold=sp.threshold,
        mult_threshold=sp.mult_threshold,
        min_coarsening=sp.min_coarsening,
        filteringA=sp.filteringA,
        filteringB=sp.filteringB,
        print_info=sp.print_info,
        force_row_closure=sp.force_row_closure,
        robust_Sker_handling=sp.robust_Sker_handling,
        max_levels=sp.max_levels,
        max_coarse=sp.max_coarse,
        max_density=sp.max_density,
        return_levels=True,
    )
    level = levels[0]
    _timer_add(timers, "stage.setup_level_sec", perf_counter() - t_stage)

    A_csr = level.A.tocsr()
    P = level.P.tocsr()
    P_T = P.T.tocsr()
    n_fine = int(A_csr.shape[0])

    t_stage = perf_counter()
    local_blocks, _ = _prepare_local_projection_blocks(level)
    if not local_blocks:
        raise ValueError("No local blocks were constructed; cannot run block-Jacobi sweep")
    _timer_add(timers, "stage.prepare_local_blocks_sec", perf_counter() - t_stage)

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
    seed = cfg.seed
    N_AJ_seed = None if seed is None else int(seed) + 13
    N_AJ_estimator_used = str(cfg.N_AJ_estimator)
    if cfg.N_AJ_estimator == "lobpcg":
        try:
            N_AJ, N_AJ_hist, N_AJ_rel_hist, N_AJ_abs_res_hist, N_AJ_rel_res_hist, N_AJ_conv = _lobpcg_MinvOp(
                A=A_csr,
                J=J,
                solve_J=solve_J,
                n=n_fine,
                block_size=int(cfg.N_AJ_block_size),
                maxiter=int(cfg.maxiter_N_AJ),
                tol=float(cfg.tol_N_AJ),
                distribution=cfg.distribution,
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
                maxiter=int(cfg.maxiter_N_AJ),
                tol=float(cfg.tol_N_AJ),
                miniter=int(cfg.miniter_N_AJ),
                distribution=cfg.distribution,
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
            maxiter=int(cfg.maxiter_N_AJ),
            tol=float(cfg.tol_N_AJ),
            miniter=int(cfg.miniter_N_AJ),
            distribution=cfg.distribution,
            seed=N_AJ_seed,
            timers=timers,
            timer_prefix="N_AJ.",
        )
    _timer_add(timers, "stage.N_AJ_total_sec", perf_counter() - t_stage)

    t_stage = perf_counter()
    N_AJ_perp_seed = None if seed is None else int(seed) + 17
    N_AJ_perp_estimator_used = str(cfg.N_AJ_perp_estimator)
    if cfg.N_AJ_perp_estimator == "lobpcg":
        try:
            N_AJ_perp, N_AJ_perp_hist, N_AJ_perp_rel_hist, N_AJ_perp_abs_res_hist, N_AJ_perp_rel_res_hist, N_AJ_perp_conv = _lobpcg_MinvOp(
                A=A_csr,
                J=J,
                solve_J=solve_J,
                n=n_fine,
                block_size=int(cfg.N_AJ_perp_block_size),
                maxiter=int(cfg.maxiter_N_AJ_perp),
                tol=float(cfg.tol_N_AJ_perp),
                distribution=cfg.distribution,
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
                maxiter=int(cfg.maxiter_N_AJ_perp),
                tol=float(cfg.tol_N_AJ_perp),
                miniter=int(cfg.miniter_N_AJ_perp),
                distribution=cfg.distribution,
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
            maxiter=int(cfg.maxiter_N_AJ_perp),
            tol=float(cfg.tol_N_AJ_perp),
            miniter=int(cfg.miniter_N_AJ_perp),
            distribution=cfg.distribution,
            seed=N_AJ_perp_seed,
            timers=timers,
            timer_prefix="N_AJ_perp.",
        )
    _timer_add(timers, "stage.N_AJ_perp_total_sec", perf_counter() - t_stage)

    tau, mu_max = _extract_level_diagnostics(level)

    W_J = None
    W_J_conv = None
    W_J_iters = None
    if bool(cfg.compute_W_J):
        t_stage = perf_counter()

        def apply_BJ(x: np.ndarray) -> tuple[np.ndarray, float]:
            return _apply_B_block_jacobi(
                x=x,
                local_blocks=local_blocks,
                n_fine=n_fine,
            )

        solve_A = _build_linear_solver(
            A=A_csr,
            method=cfg.a_solve,
            cg_rtol=float(cfg.a_cg_rtol),
            cg_atol=float(cfg.a_cg_atol),
            cg_maxiter=cfg.a_cg_maxiter,
        )
        rng = np.random.default_rng(None if seed is None else int(seed) + 29)
        x0 = _draw_random_vector(
            rng=rng,
            n=n_fine,
            distribution=cfg.distribution,
            dtype=A_csr.dtype,
        )
        lam_wj, _rel_wj, _an_wj, conv_wj = _power_iteration_AinvB(
            A=A_csr,
            apply_B=apply_BJ,
            solve_A=solve_A,
            x0=x0,
            maxiter=int(cfg.maxiter_W_J),
            tol=float(cfg.tol_W_J),
            miniter=int(cfg.miniter_W_J),
            timers=timers,
            timer_prefix="W_J.power.",
        )
        if lam_wj.size > 0:
            W_J = float(lam_wj[-1])
            W_J_conv = bool(conv_wj)
            W_J_iters = int(lam_wj.size)
        _timer_add(timers, "stage.W_J_total_sec", perf_counter() - t_stage)

    zeta_eff_vals = np.asarray(zeta_vals / float(N_AJ), dtype=float)

    zeta_low_raw = None
    if float(N_AJ_perp) > 0.0 and np.isfinite(float(N_AJ_perp)):
        zeta_low_raw = float(float(N_AJ) / float(N_AJ_perp))
    zeta_max_sampled = float(np.max(zeta_vals))
    zeta_low_clip = None if zeta_low_raw is None else float(min(zeta_low_raw, zeta_max_sampled))

    zeta_low = zeta_low_raw
    zeta_up = 1.0

    z = zeta_eff_vals
    den_L = z * (2.0 - z * float(N_AJ_perp))
    den_U = z * (2.0 - z * float(N_AJ))
    L_J = np.where(den_L > 0.0, 1.0 / den_L, np.inf)
    U_J = np.where(den_U > 0.0, 1.0 / den_U, np.inf)

    Phi_J = np.full(zeta_vals.shape, np.nan, dtype=float)
    phi_conv = np.zeros(zeta_vals.shape, dtype=bool)
    phi_iters = np.zeros(zeta_vals.shape, dtype=np.int32)
    phi_last_rel = np.full(zeta_vals.shape, np.nan, dtype=float)
    phi_last_abs_res = np.full(zeta_vals.shape, np.nan, dtype=float)
    phi_last_rel_res = np.full(zeta_vals.shape, np.nan, dtype=float)

    if bool(cfg.estimate_phi):
        t_stage = perf_counter()
        for i, zeta_eff in enumerate(zeta_eff_vals):
            if float(zeta_vals[i]) >= 2.0:
                continue
            try:
                tilde_ops = _build_tilde_metric_ops(
                    A_csr=A_csr,
                    J=J,
                    zeta_eff=float(zeta_eff),
                    h_solve=cfg.h_solve,
                    h_cg_rtol=float(cfg.h_cg_rtol),
                    h_cg_atol=float(cfg.h_cg_atol),
                    h_cg_maxiter=cfg.h_cg_maxiter,
                )

                def apply_tilde_metric(x: np.ndarray) -> np.ndarray:
                    return _apply_tildeM(x=x, M=tilde_ops.M_damped, solve_H=tilde_ops.solve_H)

                phi, hist, rel_hist, abs_res_hist, rel_res_hist, conv = _projected_power_perp_MinvOp(
                    apply_op=apply_tilde_metric,
                    apply_QM=apply_QJ,
                    apply_J=apply_J,
                    solve_J=solve_J,
                    n=n_fine,
                    dtype=A_csr.dtype,
                    maxiter=int(cfg.maxiter_Phi),
                    tol=float(cfg.tol_Phi),
                    miniter=int(cfg.miniter_Phi),
                    distribution=cfg.distribution,
                    seed=None if seed is None else int(seed) + 10000 + i,
                    timers=timers,
                    timer_prefix=f"Phi[{i}].",
                )
                Phi_J[i] = float(phi)
                phi_conv[i] = bool(conv)
                phi_iters[i] = int(hist.size)
                if rel_hist.size:
                    phi_last_rel[i] = float(rel_hist[-1])
                if abs_res_hist.size:
                    phi_last_abs_res[i] = float(abs_res_hist[-1])
                if rel_res_hist.size:
                    phi_last_rel_res[i] = float(rel_res_hist[-1])
            except Exception as exc:
                warn(
                    f"Phi_J estimation failed at normalized zeta={float(zeta_vals[i]):.6g} ({exc!r}); storing NaN.",
                    RuntimeWarning,
                )
        _timer_add(timers, "stage.Phi_total_sec", perf_counter() - t_stage)

    q_obs = np.full(zeta_vals.shape, np.nan, dtype=float)
    K_obs = np.full(zeta_vals.shape, np.nan, dtype=float)
    rho_obs = np.full(zeta_vals.shape, np.nan, dtype=float)

    if bool(cfg.estimate_observed):
        t_stage = perf_counter()
        for i, zeta_eff in enumerate(zeta_eff_vals):
            try:
                ml_obs = _build_observed_two_level_solver(
                    B=B,
                    A=A,
                    BT=BT,
                    solver_params=sp,
                    zeta=float(zeta_eff),
                    with_rho=False,
                    with_rho_perp=False,
                    zeta_eff=float(zeta_eff),
                )
                q_i, K_i = _estimate_qobs_homogeneous_two_grid(
                    ml=ml_obs,
                    distribution=cfg.distribution,
                    seed=None if seed is None else int(seed) + 20000 + i,
                    n_samples=int(cfg.observed_n_samples),
                    n_cycles=int(cfg.observed_n_cycles),
                    timers=timers,
                    timer_prefix=f"obs[{i}].",
                )
                if q_i is not None:
                    q_obs[i] = float(q_i)
                    rho_obs[i] = float(q_i)
                if K_i is not None:
                    K_obs[i] = float(K_i)
            except Exception as exc:
                warn(
                    f"Observed solve failed at normalized zeta={float(zeta_vals[i]):.6g} ({exc!r}); storing NaN.",
                    RuntimeWarning,
                )
        _timer_add(timers, "stage.observed_total_sec", perf_counter() - t_stage)

    zeta_exact = _argmin_finite(zeta_vals, Phi_J)
    zeta_best_K_obs = _argmin_finite(zeta_vals, K_obs)

    _timer_add(timers, "stage.total_sec", perf_counter() - t_total)

    if print_constants:
        sd = int(constant_sig_digits)
        print("Block-Jacobi damping sweep diagnostics:")
        print("Setup spectral edges:")
        print(f"  N_AJ                    : {_fmt_sig(float(N_AJ), sd)}")
        print(f"  N_AJ_perp               : {_fmt_sig(float(N_AJ_perp), sd)}")
        print(f"  W_J                     : {_fmt_sig(W_J, sd)}")
        print(f"  mu_max                  : {_fmt_sig(mu_max, sd)}")
        print("Damping normalization:")
        print("  N_AJ definition         : rho(J^{-1} A)")
        print("  zeta_eff definition     : zeta / N_AJ = zeta / rho(J^{-1} A)")
        print("  stable range            : zeta in (0, 2)")
        print("Predicted/sampled optima:")
        print(f"  zeta_low_raw            : {_fmt_sig(zeta_low_raw, sd)}")
        print(f"  zeta_low_clip           : {_fmt_sig(zeta_low_clip, sd)}")
        print(f"  zeta_up                 : {_fmt_sig(zeta_up, sd)}")
        print(f"  zeta_exact (argmin Phi) : {_fmt_sig(zeta_exact, sd)}")
        print(f"  zeta_best_K_obs         : {_fmt_sig(zeta_best_K_obs, sd)}")
        print("Effective optima (for smoother omega):")
        print(
            f"  zeta_low_eff            : "
            f"{_fmt_sig(None if zeta_low is None else zeta_low / float(N_AJ), sd)}"
        )
        print(
            f"  zeta_up_eff             : "
            f"{_fmt_sig(None if zeta_up is None else zeta_up / float(N_AJ), sd)}"
        )
        print(
            f"  zeta_exact_eff          : "
            f"{_fmt_sig(None if zeta_exact is None else zeta_exact / float(N_AJ), sd)}"
        )
        print(
            f"  zeta_best_K_obs_eff     : "
            f"{_fmt_sig(None if zeta_best_K_obs is None else zeta_best_K_obs / float(N_AJ), sd)}"
        )
        print("Spectral-edge solver diagnostics:")
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
        if bool(cfg.estimate_phi):
            n_phi_ok = int(np.isfinite(Phi_J).sum())
            n_phi_conv = int(phi_conv.sum())
            print("Phi_J sweep diagnostics:")
            print(f"  samples (finite)        : {n_phi_ok}/{zeta_vals.size}")
            print(f"  converged               : {n_phi_conv}/{zeta_vals.size}")

    if print_timers and timers is not None:
        print("Timing breakdown (seconds):")
        for key in sorted(timers, key=timers.get, reverse=True):
            val = timers[key]
            if key.endswith("n_iter") or key.endswith("n_cycle"):
                print(f"  {key}: {int(round(val))}")
            else:
                print(f"  {key}: {val:.6g}")

    return DampingSweepResult(
        zeta_values=np.asarray(zeta_vals, dtype=float),
        zeta_effective_values=np.asarray(zeta_eff_vals, dtype=float),
        N_AJ=float(N_AJ),
        N_AJ_perp=float(N_AJ_perp),
        tau=tau,
        mu_max=mu_max,
        W_J=W_J,
        W_J_converged=W_J_conv,
        W_J_n_iterations=W_J_iters,
        zeta_max_sampled=zeta_max_sampled,
        zeta_low_raw=zeta_low_raw,
        zeta_low_clip=zeta_low_clip,
        zeta_low=zeta_low,
        zeta_up=zeta_up,
        zeta_exact=zeta_exact,
        zeta_best_K_obs=zeta_best_K_obs,
        L_J=np.asarray(L_J, dtype=float),
        U_J=np.asarray(U_J, dtype=float),
        Phi_J=np.asarray(Phi_J, dtype=float),
        q_obs=np.asarray(q_obs, dtype=float),
        K_obs=np.asarray(K_obs, dtype=float),
        rho_obs=np.asarray(rho_obs, dtype=float),
        Phi_converged=np.asarray(phi_conv, dtype=bool),
        Phi_n_iterations=np.asarray(phi_iters, dtype=np.int32),
        Phi_last_rel_change=np.asarray(phi_last_rel, dtype=float),
        Phi_last_abs_residual=np.asarray(phi_last_abs_res, dtype=float),
        Phi_last_rel_residual=np.asarray(phi_last_rel_res, dtype=float),
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
        timings=timers,
    )


__all__ = [
    "DampingSweepConfig",
    "DampingSweepResult",
    "compute_block_jacobi_damping_sweep",
]
