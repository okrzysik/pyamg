"""Iterative spectral-estimation routines used by algebraic-theory drivers."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg

from .models import RandomDistribution
from .reporting import timer_add


def draw_random_vector(
    *,
    rng: np.random.Generator,
    n: int,
    distribution: RandomDistribution,
    dtype,
) -> np.ndarray:
    """Draw a random dense test vector."""
    if distribution == "gaussian":
        v = rng.standard_normal(n)
    elif distribution == "rademacher":
        v = 2 * rng.integers(0, 2, size=n) - 1
    else:  # pragma: no cover
        raise ValueError(f"Unsupported distribution {distribution!r}")
    return np.asarray(v, dtype=dtype)


def _power_metric_inverse_core(
    *,
    apply_operator: Callable[[np.ndarray], np.ndarray],
    apply_metric: Callable[[np.ndarray], np.ndarray],
    solve_metric: Callable[[np.ndarray], np.ndarray],
    n: int,
    dtype,
    maxiter: int,
    tol: float,
    miniter: int,
    distribution: RandomDistribution,
    seed: int | None,
    apply_projector: Callable[[np.ndarray], np.ndarray] | None = None,
    x0: np.ndarray | None = None,
    return_vector: bool = False,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray
]:
    """Shared core for metric-inverse power iteration with optional projection."""
    if maxiter <= 0:
        raise ValueError("Expected maxiter > 0 for metric-inverse iteration")
    if miniter <= 0:
        raise ValueError("Expected miniter > 0 for metric-inverse iteration")
    if tol < 0.0:
        raise ValueError("Expected nonnegative tol for metric-inverse iteration")

    rng = np.random.default_rng(seed)
    if x0 is None:
        x = draw_random_vector(rng=rng, n=n, distribution=distribution, dtype=dtype)
    else:
        x = np.asarray(x0, dtype=dtype).reshape(-1)
        if int(x.size) != int(n):
            raise ValueError(f"Warm-start x0 has incompatible size {x.size} != {n}")

    if apply_projector is not None:
        x = apply_projector(x)

    for _ in range(5):
        den0 = float(np.vdot(x, apply_metric(x)).real)
        if den0 > 0.0:
            x /= np.sqrt(den0)
            break
        x = draw_random_vector(rng=rng, n=n, distribution=distribution, dtype=dtype)
        if apply_projector is not None:
            x = apply_projector(x)
    else:
        raise ValueError("Failed to initialize nonzero vector in metric norm")

    lam_hist: list[float] = []
    rel_hist: list[float] = []
    abs_res_hist: list[float] = []
    rel_res_hist: list[float] = []
    converged = False
    prev = None

    for _k in range(maxiter):
        timer_add(timers, f"{timer_prefix}n_iter", 1.0)

        t_apply_operator = perf_counter()
        y = apply_operator(x)
        timer_add(timers, f"{timer_prefix}apply_operator_sec", perf_counter() - t_apply_operator)

        mx = apply_metric(x)
        den = float(np.vdot(x, mx).real)
        if den <= 0.0:
            raise ValueError("Encountered non-positive metric norm in iteration")
        lam = float(np.vdot(x, y).real / den)
        if lam < 0.0 and abs(lam) < 1e-12:
            lam = 0.0
        lam_hist.append(lam)

        res = np.asarray(y - lam * mx).reshape(-1)
        if apply_projector is not None:
            t_proj_res = perf_counter()
            res = apply_projector(res)
            timer_add(timers, f"{timer_prefix}apply_projector_res_sec", perf_counter() - t_proj_res)
        t_res_solve_metric = perf_counter()
        zres = solve_metric(res)
        timer_add(timers, f"{timer_prefix}solve_metric_res_sec", perf_counter() - t_res_solve_metric)
        m_inv_norm_sq = float(np.vdot(res, zres).real)
        if m_inv_norm_sq < 0.0 and abs(m_inv_norm_sq) < 1e-12:
            m_inv_norm_sq = 0.0
        abs_res = float(np.sqrt(max(m_inv_norm_sq, 0.0)))
        rel_res = float(abs_res / max(abs(lam) * np.sqrt(max(den, 0.0)), 1e-300))
        abs_res_hist.append(abs_res)
        rel_res_hist.append(rel_res)

        if prev is None:
            rel = np.inf
        else:
            rel = abs(lam - prev) / max(abs(prev), 1.0)
        rel_hist.append(float(rel))
        prev = lam

        if len(lam_hist) >= miniter and rel <= tol:
            converged = True
            break

        t_solve_metric = perf_counter()
        z = solve_metric(y)
        timer_add(timers, f"{timer_prefix}solve_metric_sec", perf_counter() - t_solve_metric)

        if apply_projector is not None:
            t_proj = perf_counter()
            z = apply_projector(z)
            timer_add(timers, f"{timer_prefix}apply_projector_sec", perf_counter() - t_proj)

        t_apply_metric = perf_counter()
        denz = float(np.vdot(z, apply_metric(z)).real)
        timer_add(timers, f"{timer_prefix}apply_metric_sec", perf_counter() - t_apply_metric)
        if denz <= 0.0:
            converged = True
            break
        x = z / np.sqrt(denz)

    arr = np.asarray(lam_hist, dtype=float)
    rel_arr = np.asarray(rel_hist, dtype=float)
    abs_res_arr = np.asarray(abs_res_hist, dtype=float)
    rel_res_arr = np.asarray(rel_res_hist, dtype=float)
    if arr.size == 0:
        raise ValueError("No iterations executed in metric-inverse iteration")
    x_out = np.asarray(x, dtype=dtype).reshape(-1)
    if return_vector:
        return float(arr[-1]), arr, rel_arr, abs_res_arr, rel_res_arr, bool(converged), x_out
    return float(arr[-1]), arr, rel_arr, abs_res_arr, rel_res_arr, bool(converged)


def power_metric_inverse_operator(
    *,
    apply_operator: Callable[[np.ndarray], np.ndarray],
    apply_metric: Callable[[np.ndarray], np.ndarray],
    solve_metric: Callable[[np.ndarray], np.ndarray],
    n: int,
    dtype,
    maxiter: int,
    tol: float,
    miniter: int,
    distribution: RandomDistribution,
    seed: int | None,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Estimate ``lambda_max(metric^{-1} operator)`` by power iteration.

    Parameters
    ----------
    apply_operator
        Callable returning ``operator @ x``.
    apply_metric
        Callable returning ``metric @ x`` for the SPD metric used in Rayleigh
        quotients and normalization.
    solve_metric
        Callable approximately/exactly solving ``metric z = rhs``.
    n, dtype
        Vector dimension and working dtype.
    maxiter, tol, miniter
        Iteration controls; stopping is based on relative eigenvalue change.
    distribution, seed
        Random initialization settings.
    """
    return _power_metric_inverse_core(
        apply_operator=apply_operator,
        apply_metric=apply_metric,
        solve_metric=solve_metric,
        n=n,
        dtype=dtype,
        maxiter=maxiter,
        tol=tol,
        miniter=miniter,
        distribution=distribution,
        seed=seed,
        apply_projector=None,
        x0=None,
        return_vector=False,
        timers=timers,
        timer_prefix=timer_prefix,
    )


def power_metric_inverse_operator_projected(
    *,
    apply_operator: Callable[[np.ndarray], np.ndarray],
    apply_projector: Callable[[np.ndarray], np.ndarray],
    apply_metric: Callable[[np.ndarray], np.ndarray],
    solve_metric: Callable[[np.ndarray], np.ndarray],
    n: int,
    dtype,
    maxiter: int,
    tol: float,
    miniter: int,
    distribution: RandomDistribution,
    seed: int | None,
    x0: np.ndarray | None = None,
    return_vector: bool = False,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray
]:
    """Estimate the dominant projected eigenvalue of ``metric^{-1} operator``.

    This computes the largest eigenvalue of the projected iteration
    ``P metric^{-1} operator P``, where ``P`` is represented by
    ``apply_projector``.

    Non-trivial inputs
    ------------------
    apply_projector
        Projection/filter map applied to vectors, residuals, and iterates. In
        this codebase it is typically ``Q_J``.
    apply_metric / solve_metric
        Metric action and metric solve defining the generalized eigenproblem.
    x0
        Optional warm start. If provided, it is projected and metric-normalized
        internally before iterations.
    return_vector
        When ``True``, also return the final normalized iterate.
    """
    return _power_metric_inverse_core(
        apply_operator=apply_operator,
        apply_metric=apply_metric,
        solve_metric=solve_metric,
        n=n,
        dtype=dtype,
        maxiter=maxiter,
        tol=tol,
        miniter=miniter,
        distribution=distribution,
        seed=seed,
        apply_projector=apply_projector,
        x0=x0,
        return_vector=return_vector,
        timers=timers,
        timer_prefix=timer_prefix,
    )


def lobpcg_metric_inverse_operator(
    *,
    operator_matrix,
    metric_matrix,
    solve_metric: Callable[[np.ndarray], np.ndarray],
    n: int,
    block_size: int,
    maxiter: int,
    tol: float,
    distribution: RandomDistribution,
    seed: int | None,
    return_vector: bool = False,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray
]:
    """Estimate top generalized eigenvalue of ``operator x = lambda metric x`` using LOBPCG.

    Notes
    -----
    This wrapper returns final residual diagnostics only. It intentionally does
    not request or process per-iteration LOBPCG histories.
    """
    if block_size <= 0:
        raise ValueError("Expected positive block_size for LOBPCG")
    if maxiter <= 0:
        raise ValueError("Expected positive maxiter for LOBPCG")
    if tol < 0.0:
        raise ValueError("Expected nonnegative tol for LOBPCG")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, int(block_size)))
    if distribution == "rademacher":
        X = np.sign(X)
        X[X == 0.0] = 1.0
    X = np.asarray(X, dtype=operator_matrix.dtype)

    def m_solve(v):
        arr = np.asarray(v)
        if arr.ndim == 1:
            return solve_metric(arr)
        out = np.column_stack([solve_metric(arr[:, j]) for j in range(arr.shape[1])])
        return out

    M_prec = LinearOperator(shape=operator_matrix.shape, matvec=lambda v: m_solve(v), dtype=operator_matrix.dtype)

    t_lobpcg = perf_counter()
    vals, vecs = lobpcg(
        operator_matrix,
        X,
        B=metric_matrix,
        M=M_prec,
        tol=tol,
        maxiter=maxiter,
        largest=True,
    )
    timer_add(timers, f"{timer_prefix}solve_lobpcg_sec", perf_counter() - t_lobpcg)

    vals = np.asarray(vals, dtype=float).reshape(-1)
    j = int(np.argmax(vals))
    lam = float(vals[j])
    u = np.asarray(vecs[:, j]).reshape(-1)

    mu = np.asarray(metric_matrix @ u).reshape(-1)
    r = np.asarray(operator_matrix @ u - lam * mu).reshape(-1)
    zres = solve_metric(r)
    m_inv_norm_sq = float(np.vdot(r, zres).real)
    if m_inv_norm_sq < 0.0 and abs(m_inv_norm_sq) < 1e-12:
        m_inv_norm_sq = 0.0
    abs_res = float(np.sqrt(max(m_inv_norm_sq, 0.0)))
    den = float(np.vdot(u, mu).real)
    rel_res = float(abs_res / max(abs(lam) * np.sqrt(max(den, 0.0)), 1e-300))

    hist = np.asarray([], dtype=float)
    rel_hist = np.asarray([], dtype=float)
    abs_res_hist = np.asarray([abs_res], dtype=float)
    rel_res_hist = np.asarray([rel_res], dtype=float)
    converged = bool(np.isfinite(rel_res) and rel_res <= tol)
    if return_vector:
        return (
            lam,
            hist,
            rel_hist,
            abs_res_hist,
            rel_res_hist,
            converged,
            np.asarray(u, dtype=operator_matrix.dtype),
        )
    return lam, hist, rel_hist, abs_res_hist, rel_res_hist, converged


def power_generalized_eigen_matrix_free(
    *,
    metric_matrix,
    apply_numerator: Callable[[np.ndarray], tuple[np.ndarray, float]],
    solve_metric: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    maxiter: int,
    tol: float,
    miniter: int,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Estimate dominant ``lambda`` in ``numerator x = lambda metric x``.

    The numerator action is matrix-free via ``apply_numerator(x)``, which
    returns both ``numerator @ x`` and the Rayleigh numerator scalar
    ``x^* numerator x``.
    """
    x = np.asarray(x0).reshape(-1).copy()
    x_metric_norm_sq = float(np.vdot(x, metric_matrix @ x).real)
    if x_metric_norm_sq <= 0.0:
        raise ValueError(f"Initial vector has non-positive metric-norm squared: {x_metric_norm_sq}")
    x /= np.sqrt(x_metric_norm_sq)

    lambda_hist: list[float] = []
    rel_hist: list[float] = []
    anorm_hist: list[float] = []

    lam_prev = None
    converged = False

    for _k in range(maxiter):
        timer_add(timers, f"{timer_prefix}n_iter", 1.0)

        t_apply_numerator = perf_counter()
        y, num = apply_numerator(x)
        timer_add(timers, f"{timer_prefix}apply_numerator_sec", perf_counter() - t_apply_numerator)
        den = float(np.vdot(x, metric_matrix @ x).real)
        if den <= 0.0:
            raise ValueError(f"Encountered non-positive denominator in Rayleigh quotient: {den}")
        lam = float(num / den)
        if lam < 0.0 and abs(lam) < 1e-12:
            lam = 0.0

        lambda_hist.append(lam)
        anorm_hist.append(float(np.sqrt(den)))

        if lam_prev is None:
            rel = np.inf
        else:
            rel = abs(lam - lam_prev) / max(abs(lam_prev), 1.0)
        rel_hist.append(float(rel))
        lam_prev = lam

        if len(lambda_hist) >= miniter and rel <= tol:
            converged = True
            break

        if float(np.linalg.norm(y)) == 0.0:
            converged = True
            break

        t_solve_metric = perf_counter()
        z = solve_metric(y)
        timer_add(timers, f"{timer_prefix}solve_metric_sec", perf_counter() - t_solve_metric)
        z_metric_norm_sq = float(np.vdot(z, metric_matrix @ z).real)
        if z_metric_norm_sq <= 0.0:
            raise ValueError(f"Encountered non-positive metric-norm squared during iteration: {z_metric_norm_sq}")
        x = z / np.sqrt(z_metric_norm_sq)

    return (
        np.asarray(lambda_hist, dtype=float),
        np.asarray(rel_hist, dtype=float),
        np.asarray(anorm_hist, dtype=float),
        bool(converged),
    )
