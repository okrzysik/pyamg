"""Block-Jacobi damping sweep diagnostics for LS-DD algebraic theory.

This module implements the numerical stage focused on damping dependence for
block Jacobi, using a normalized damping coordinate ``zeta_norm`` in ``(0, 2)``
defined by

``zeta_norm = zeta_raw / N_AJ``.

Here ``zeta_raw`` is the raw scalar passed to ASM as ``omega``.

For a fixed coarse space and a sweep of ``zeta_norm``, the driver computes only the
quantities needed for damping optimization studies:

- Whole-space and restricted spectral edges: ``N_AJ`` and ``N_AJ_perp``.
- Scalar sandwich models as functions of ``zeta_norm``:
  - ``L_J(zeta_norm) = 1 / (zeta_raw * (2 - zeta_raw * N_AJ_perp))``
  - ``U_J(zeta_norm) = 1 / (zeta_raw * (2 - zeta_raw * N_AJ))``
    with ``zeta_raw = zeta_norm / N_AJ``.
- Sampled restricted sharp target:
  - ``Phi_J(zeta_norm) = N_{tilde(J_zeta),J}^perp``
- Observed two-grid quantities from actual solver runs:
  - ``q_obs(zeta_norm)``, ``K_obs(zeta_norm)``, ``rho_obs(zeta_norm)``.

The implementation intentionally avoids computing unrelated refined-chain terms
for performance.
"""

from __future__ import annotations

from functools import partial
from time import perf_counter
from warnings import warn

import numpy as np

from .models import DampingSweepConfig, DampingSweepResult
from .observed import build_observed_two_level_solver as _build_observed_two_level_solver
from .observed import estimate_qobs_homogeneous_two_grid as _estimate_qobs_homogeneous_two_grid
from .operators import apply_QJ as _apply_QJ
from .operators import apply_tildeM as _apply_tildeM
from .operators import apply_WJ_numerator_operator as _apply_WJ_numerator_operator
from .operators import assemble_block_jacobi_matrix as _assemble_block_jacobi_matrix
from .operators import assemble_block_jacobi_solver as _assemble_block_jacobi_solver
from .operators import build_linear_solver as _build_linear_solver
from .operators import build_tilde_metric_ops as _build_tilde_metric_ops
from .reporting import fmt_sig as _fmt_sig
from .reporting import timer_add as _timer_add
from .setup import build_one_level_lsdd as _build_one_level_lsdd
from .setup import extract_level_diagnostics as _extract_level_diagnostics
from .setup import prepare_local_projection_blocks as _prepare_local_projection_blocks
from .spectral import draw_random_vector as _draw_random_vector
from .spectral import lobpcg_metric_inverse_operator as _lobpcg_metric_inverse_operator
from .spectral import power_metric_inverse_operator as _power_metric_inverse_operator
from .spectral import power_generalized_eigen_matrix_free as _power_generalized_eigen_matrix_free
from .spectral import power_metric_inverse_operator_projected as _power_metric_inverse_operator_projected


def _zeta_norm_grid_from_config(cfg: DampingSweepConfig) -> np.ndarray:
    """Return validated normalized damping grid ``zeta_norm`` in ``(0, 2)``."""
    if cfg.zeta_norm_values is not None:
        zeta = np.asarray(cfg.zeta_norm_values, dtype=float).reshape(-1)
        if zeta.size == 0:
            raise ValueError("zeta_norm_values must contain at least one value")
    else:
        n = int(cfg.n_zeta)
        if n <= 0:
            raise ValueError("n_zeta must be positive")
        zmin = float(cfg.zeta_norm_min)
        zmax = float(cfg.zeta_norm_max)
        if not (0.0 < zmin < 2.0 and 0.0 < zmax < 2.0):
            raise ValueError("zeta_norm_min and zeta_norm_max must lie strictly in (0, 2)")
        if zmax <= zmin:
            raise ValueError("zeta_norm_max must be greater than zeta_norm_min")
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
        values ``zeta_norm`` in ``(0, 2)`` with ``zeta_raw = zeta_norm / N_AJ``.
    print_constants, constant_sig_digits
        Optional formatted diagnostics printout.
    print_timers
        Optional timing summary.

    Returns
    -------
    DampingSweepResult
        Arrays for ``L_J/U_J/Phi_J`` and observed ``q/K/rho`` over the
        normalized damping grid, along with one-shot spectral-edge estimates and optimizer
        summaries.
    """
    timers: dict[str, float] | None = {} if (cfg.collect_timers or print_timers) else None
    t_total = perf_counter()

    zeta_norm_vals = _zeta_norm_grid_from_config(cfg)
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
    n_fine = int(A_csr.shape[0])

    t_stage = perf_counter()
    local_blocks, _ = _prepare_local_projection_blocks(level)
    if not local_blocks:
        raise ValueError("No local blocks were constructed; cannot run block-Jacobi sweep")
    _timer_add(timers, "stage.prepare_local_blocks_sec", perf_counter() - t_stage)

    t_stage = perf_counter()
    J = _assemble_block_jacobi_matrix(local_blocks=local_blocks, n_fine=n_fine, dtype=A_csr.dtype)
    solve_J = _assemble_block_jacobi_solver(local_blocks=local_blocks, n_fine=n_fine)
    _timer_add(timers, "stage.build_block_jacobi_ops_sec", perf_counter() - t_stage)

    def apply_J(x: np.ndarray) -> np.ndarray:
        return np.asarray(J @ x).reshape(-1)
    apply_QJ = partial(_apply_QJ, local_blocks=local_blocks, n_fine=n_fine)

    t_stage = perf_counter()
    seed = cfg.seed
    N_AJ_seed = None if seed is None else int(seed) + 13
    N_AJ_estimator_used = str(cfg.N_AJ_estimator)
    if cfg.N_AJ_estimator == "lobpcg":
        try:
            N_AJ, N_AJ_hist, N_AJ_rel_hist, N_AJ_abs_res_hist, N_AJ_rel_res_hist, N_AJ_conv = _lobpcg_metric_inverse_operator(
                operator_matrix=A_csr,
                metric_matrix=J,
                solve_metric=solve_J,
                n=n_fine,
                block_size=int(cfg.N_AJ_block_size),
                maxiter=int(cfg.maxiter_N_AJ),
                tol=float(cfg.tol_N_AJ),
                distribution=cfg.distribution,
                seed=N_AJ_seed,
                timers=timers,
                timer_prefix="N_AJ.",
            )
        except Exception as exc:
            warn(
                f"LOBPCG N_AJ estimation failed ({exc!r}); falling back to power iteration.",
                RuntimeWarning,
            )
            N_AJ_estimator_used = "power_fallback"
            N_AJ, N_AJ_hist, N_AJ_rel_hist, N_AJ_abs_res_hist, N_AJ_rel_res_hist, N_AJ_conv = _power_metric_inverse_operator(
                apply_operator=lambda x: np.asarray(A_csr @ x).reshape(-1),
                apply_metric=apply_J,
                solve_metric=solve_J,
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
        N_AJ, N_AJ_hist, N_AJ_rel_hist, N_AJ_abs_res_hist, N_AJ_rel_res_hist, N_AJ_conv = _power_metric_inverse_operator(
            apply_operator=lambda x: np.asarray(A_csr @ x).reshape(-1),
            apply_metric=apply_J,
            solve_metric=solve_J,
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
    N_AJ_perp_vec: np.ndarray | None = None
    if cfg.N_AJ_perp_estimator == "lobpcg":
        warn(
            "N_AJ_perp_estimator='lobpcg' is unsupported with sparse coarse constraints; "
            "using projected power iteration instead.",
            RuntimeWarning,
        )
        N_AJ_perp_estimator_used = "power_projected_forced"
    N_AJ_perp, N_AJ_perp_hist, N_AJ_perp_rel_hist, N_AJ_perp_abs_res_hist, N_AJ_perp_rel_res_hist, N_AJ_perp_conv, N_AJ_perp_vec = _power_metric_inverse_operator_projected(
        apply_operator=lambda x: np.asarray(A_csr @ x).reshape(-1),
        apply_projector=apply_QJ,
        apply_metric=apply_J,
        solve_metric=solve_J,
        n=n_fine,
        dtype=A_csr.dtype,
        maxiter=int(cfg.maxiter_N_AJ_perp),
        tol=float(cfg.tol_N_AJ_perp),
        miniter=int(cfg.miniter_N_AJ_perp),
        distribution=cfg.distribution,
        seed=N_AJ_perp_seed,
        return_vector=True,
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
            return _apply_WJ_numerator_operator(
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
        lam_wj, _rel_wj, _an_wj, conv_wj = _power_generalized_eigen_matrix_free(
            metric_matrix=A_csr,
            apply_numerator=apply_BJ,
            solve_metric=solve_A,
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

    zeta_raw_vals = np.asarray(zeta_norm_vals / float(N_AJ), dtype=float)

    zeta_norm_low_raw = None
    if float(N_AJ_perp) > 0.0 and np.isfinite(float(N_AJ_perp)):
        zeta_norm_low_raw = float(float(N_AJ) / float(N_AJ_perp))
    zeta_norm_max_sampled = float(np.max(zeta_norm_vals))
    zeta_norm_low_clip = None if zeta_norm_low_raw is None else float(min(zeta_norm_low_raw, zeta_norm_max_sampled))

    zeta_norm_low = zeta_norm_low_raw
    zeta_norm_up = 1.0

    z = zeta_raw_vals
    den_L = z * (2.0 - z * float(N_AJ_perp))
    den_U = z * (2.0 - z * float(N_AJ))
    L_J = np.where(den_L > 0.0, 1.0 / den_L, np.inf)
    U_J = np.where(den_U > 0.0, 1.0 / den_U, np.inf)

    Phi_J = np.full(zeta_norm_vals.shape, np.nan, dtype=float)
    phi_conv = np.zeros(zeta_norm_vals.shape, dtype=bool)
    phi_iters = np.zeros(zeta_norm_vals.shape, dtype=np.int32)
    phi_last_rel = np.full(zeta_norm_vals.shape, np.nan, dtype=float)
    phi_last_abs_res = np.full(zeta_norm_vals.shape, np.nan, dtype=float)
    phi_last_rel_res = np.full(zeta_norm_vals.shape, np.nan, dtype=float)

    if bool(cfg.estimate_phi):
        t_stage = perf_counter()
        for i, zeta_raw in enumerate(zeta_raw_vals):
            if float(zeta_norm_vals[i]) >= 2.0:
                continue
            try:
                tilde_ops = _build_tilde_metric_ops(
                    A_csr=A_csr,
                    J=J,
                    zeta_eff=float(zeta_raw),
                    h_solve=cfg.h_solve,
                    h_cg_rtol=float(cfg.h_cg_rtol),
                    h_cg_atol=float(cfg.h_cg_atol),
                    h_cg_maxiter=cfg.h_cg_maxiter,
                )

                def apply_tilde_metric(x: np.ndarray) -> np.ndarray:
                    return _apply_tildeM(x=x, M=tilde_ops.M_damped, solve_H=tilde_ops.solve_H)

                phi, hist, rel_hist, abs_res_hist, rel_res_hist, conv = _power_metric_inverse_operator_projected(
                    apply_operator=apply_tilde_metric,
                    apply_projector=apply_QJ,
                    apply_metric=apply_J,
                    solve_metric=solve_J,
                    n=n_fine,
                    dtype=A_csr.dtype,
                    maxiter=int(cfg.maxiter_Phi),
                    tol=float(cfg.tol_Phi),
                    miniter=int(cfg.miniter_Phi),
                    distribution=cfg.distribution,
                    seed=None if seed is None else int(seed) + 10000 + i,
                    x0=N_AJ_perp_vec,
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
                    f"Phi_J estimation failed at zeta_norm={float(zeta_norm_vals[i]):.6g} ({exc!r}); storing NaN.",
                    RuntimeWarning,
                )
        _timer_add(timers, "stage.Phi_total_sec", perf_counter() - t_stage)

    q_obs = np.full(zeta_norm_vals.shape, np.nan, dtype=float)
    K_obs = np.full(zeta_norm_vals.shape, np.nan, dtype=float)
    rho_obs = np.full(zeta_norm_vals.shape, np.nan, dtype=float)

    if bool(cfg.estimate_observed):
        t_stage = perf_counter()
        for i, zeta_raw in enumerate(zeta_raw_vals):
            try:
                ml_obs = _build_observed_two_level_solver(
                    B=B,
                    A=A,
                    BT=BT,
                    solver_params=sp,
                    zeta_raw=float(zeta_raw),
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
                    f"Observed solve failed at zeta_norm={float(zeta_norm_vals[i]):.6g} ({exc!r}); storing NaN.",
                    RuntimeWarning,
                )
        _timer_add(timers, "stage.observed_total_sec", perf_counter() - t_stage)

    zeta_norm_exact = _argmin_finite(zeta_norm_vals, Phi_J)
    zeta_norm_best_K_obs = _argmin_finite(zeta_norm_vals, K_obs)

    zeta_raw_low = None if zeta_norm_low is None else float(zeta_norm_low / float(N_AJ))
    zeta_raw_up = None if zeta_norm_up is None else float(zeta_norm_up / float(N_AJ))
    zeta_raw_exact = None if zeta_norm_exact is None else float(zeta_norm_exact / float(N_AJ))
    zeta_raw_best_K_obs = (
        None if zeta_norm_best_K_obs is None else float(zeta_norm_best_K_obs / float(N_AJ))
    )

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
        print("  zeta_raw definition     : ASM omega (raw parsed value)")
        print("  zeta_norm definition    : zeta_raw / N_AJ")
        print("  stable range            : zeta_norm in (0, 2)")
        print("Predicted/sampled optima (normalized):")
        print(f"  zeta_norm_low_raw       : {_fmt_sig(zeta_norm_low_raw, sd)}")
        print(f"  zeta_norm_low_clip      : {_fmt_sig(zeta_norm_low_clip, sd)}")
        print(f"  zeta_norm_up            : {_fmt_sig(zeta_norm_up, sd)}")
        print(f"  zeta_norm_exact         : {_fmt_sig(zeta_norm_exact, sd)}")
        print(f"  zeta_norm_best_K_obs    : {_fmt_sig(zeta_norm_best_K_obs, sd)}")
        print("Predicted/sampled optima (raw ASM omega):")
        print(f"  zeta_raw_low            : {_fmt_sig(zeta_raw_low, sd)}")
        print(f"  zeta_raw_up             : {_fmt_sig(zeta_raw_up, sd)}")
        print(f"  zeta_raw_exact          : {_fmt_sig(zeta_raw_exact, sd)}")
        print(f"  zeta_raw_best_K_obs     : {_fmt_sig(zeta_raw_best_K_obs, sd)}")
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
            print(f"  samples (finite)        : {n_phi_ok}/{zeta_norm_vals.size}")
            print(f"  converged               : {n_phi_conv}/{zeta_norm_vals.size}")

    if print_timers and timers is not None:
        sec_items = {k: float(v) for k, v in timers.items() if k.endswith("_sec")}
        count_items = {
            k: int(round(float(v)))
            for k, v in timers.items()
            if k.endswith("n_iter") or k.endswith("n_cycle")
        }

        def _aggregate_indexed(items: dict[str, float], prefix: str) -> dict[str, float]:
            out: dict[str, float] = {}
            pfx = f"{prefix}["
            for k, v in items.items():
                if not k.startswith(pfx):
                    continue
                marker = "]."
                j = k.find(marker)
                if j < 0:
                    continue
                tail = k[j + len(marker) :]
                out[tail] = float(out.get(tail, 0.0) + float(v))
            return out

        def _indexed_counts(items: dict[str, int], prefix: str, suffix: str) -> np.ndarray:
            vals = [int(v) for k, v in items.items() if k.startswith(f"{prefix}[") and k.endswith(suffix)]
            return np.asarray(vals, dtype=np.int32)

        print("Timing summary:")
        total = sec_items.get("stage.total_sec", None)
        if total is not None:
            print(f"  total_sec: {total:.6g}")

        stage_items = {
            k: v
            for k, v in sec_items.items()
            if k.startswith("stage.") and k != "stage.total_sec"
        }
        if stage_items:
            print("Stage times (seconds):")
            for key in sorted(stage_items, key=stage_items.get, reverse=True):
                print(f"  {key}: {stage_items[key]:.6g}")

        phi_sec = _aggregate_indexed(sec_items, "Phi")
        obs_sec = _aggregate_indexed(sec_items, "obs")
        other_sec = {
            k: v
            for k, v in sec_items.items()
            if not k.startswith("stage.") and not k.startswith("Phi[") and not k.startswith("obs[")
        }

        if phi_sec or obs_sec or other_sec:
            print("Kernel totals (seconds):")
            for key in sorted(phi_sec, key=phi_sec.get, reverse=True):
                print(f"  Phi.{key}: {phi_sec[key]:.6g}")
            for key in sorted(obs_sec, key=obs_sec.get, reverse=True):
                print(f"  obs.{key}: {obs_sec[key]:.6g}")
            for key in sorted(other_sec, key=other_sec.get, reverse=True):
                print(f"  {key}: {other_sec[key]:.6g}")

        if count_items:
            print("Work counters:")
            for key in ("N_AJ.n_iter", "N_AJ_perp.n_iter", "W_J.power.n_iter"):
                if key in count_items:
                    print(f"  {key}: {count_items[key]}")

            phi_iters = _indexed_counts(count_items, "Phi", ".n_iter")
            if phi_iters.size:
                print(
                    "  Phi.n_iter (per-zeta): "
                    f"total={int(phi_iters.sum())}, "
                    f"min={int(np.min(phi_iters))}, "
                    f"med={int(np.median(phi_iters))}, "
                    f"max={int(np.max(phi_iters))}"
                )

            obs_cycles = _indexed_counts(count_items, "obs", ".n_cycle")
            if obs_cycles.size:
                print(
                    "  obs.n_cycle (per-zeta): "
                    f"total={int(obs_cycles.sum())}, "
                    f"min={int(np.min(obs_cycles))}, "
                    f"med={int(np.median(obs_cycles))}, "
                    f"max={int(np.max(obs_cycles))}"
                )

    return DampingSweepResult(
        zeta_norm_values=np.asarray(zeta_norm_vals, dtype=float),
        zeta_raw_values=np.asarray(zeta_raw_vals, dtype=float),
        N_AJ=float(N_AJ),
        N_AJ_perp=float(N_AJ_perp),
        tau=tau,
        mu_max=mu_max,
        W_J=W_J,
        W_J_converged=W_J_conv,
        W_J_n_iterations=W_J_iters,
        zeta_norm_max_sampled=zeta_norm_max_sampled,
        zeta_norm_low_raw=zeta_norm_low_raw,
        zeta_norm_low_clip=zeta_norm_low_clip,
        zeta_norm_low=zeta_norm_low,
        zeta_norm_up=zeta_norm_up,
        zeta_norm_exact=zeta_norm_exact,
        zeta_norm_best_K_obs=zeta_norm_best_K_obs,
        zeta_raw_low=zeta_raw_low,
        zeta_raw_up=zeta_raw_up,
        zeta_raw_exact=zeta_raw_exact,
        zeta_raw_best_K_obs=zeta_raw_best_K_obs,
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
