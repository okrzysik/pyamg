"""Iterative spectral-estimation routines used by algebraic-theory drivers."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, lobpcg

from .models import RandomDistribution
from .reporting import rel_change_history, timer_add


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


def projected_power_perp_MinvOp(
    *,
    apply_op: Callable[[np.ndarray], np.ndarray],
    apply_QM: Callable[[np.ndarray], np.ndarray],
    apply_J: Callable[[np.ndarray], np.ndarray],
    solve_J: Callable[[np.ndarray], np.ndarray],
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
    """Estimate ``N_AJ_perp = lambda_max^\perp(M^{-1}Op)`` on ``ker(P^T M)``."""
    if maxiter <= 0:
        raise ValueError("Expected maxiter > 0 for projected-perp iteration")
    if miniter <= 0:
        raise ValueError("Expected miniter > 0 for projected-perp iteration")
    if tol < 0.0:
        raise ValueError("Expected nonnegative tol for projected-perp iteration")

    rng = np.random.default_rng(seed)
    r = draw_random_vector(rng=rng, n=n, distribution=distribution, dtype=dtype)
    r = apply_QM(r)

    for _ in range(5):
        den0 = float(np.vdot(r, apply_J(r)).real)
        if den0 > 0.0:
            r /= np.sqrt(den0)
            break
        r = apply_QM(draw_random_vector(rng=rng, n=n, distribution=distribution, dtype=dtype))
    else:
        raise ValueError("Failed to initialize nonzero vector in ker(P^T M)")

    lam_hist: list[float] = []
    rel_hist: list[float] = []
    abs_res_hist: list[float] = []
    rel_res_hist: list[float] = []
    converged = False
    prev = None

    for _k in range(maxiter):
        timer_add(timers, f"{timer_prefix}n_iter", 1.0)

        t_apply_op = perf_counter()
        y = apply_op(r)
        timer_add(timers, f"{timer_prefix}apply_op_sec", perf_counter() - t_apply_op)

        jx = apply_J(r)
        den = float(np.vdot(r, jx).real)
        if den <= 0.0:
            raise ValueError("Encountered non-positive J-norm in projected-perp N_AJ iteration")
        lam = float(np.vdot(r, y).real / den)
        if lam < 0.0 and abs(lam) < 1e-12:
            lam = 0.0
        lam_hist.append(lam)

        gres = np.asarray(y - lam * jx).reshape(-1)
        t_res = perf_counter()
        gres = apply_QM(gres)
        timer_add(timers, f"{timer_prefix}apply_QM_res_sec", perf_counter() - t_res)
        t_res_solve = perf_counter()
        zres = solve_J(gres)
        timer_add(timers, f"{timer_prefix}solve_J_res_sec", perf_counter() - t_res_solve)
        m_inv_norm_sq = float(np.vdot(gres, zres).real)
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

        t_solve_J = perf_counter()
        z = solve_J(y)
        timer_add(timers, f"{timer_prefix}solve_J_sec", perf_counter() - t_solve_J)

        t_apply_qm = perf_counter()
        z = apply_QM(z)
        timer_add(timers, f"{timer_prefix}apply_QM_sec", perf_counter() - t_apply_qm)

        t_apply_j = perf_counter()
        den = float(np.vdot(z, apply_J(z)).real)
        timer_add(timers, f"{timer_prefix}apply_J_sec", perf_counter() - t_apply_j)
        if den <= 0.0:
            converged = True
            break
        r = z / np.sqrt(den)

    arr = np.asarray(lam_hist, dtype=float)
    rel_arr = np.asarray(rel_hist, dtype=float)
    abs_res_arr = np.asarray(abs_res_hist, dtype=float)
    rel_res_arr = np.asarray(rel_res_hist, dtype=float)
    if arr.size == 0:
        raise ValueError("No iterations executed in projected-perp iteration")
    return float(arr[-1]), arr, rel_arr, abs_res_arr, rel_res_arr, bool(converged)


def power_MinvOp(
    *,
    apply_op: Callable[[np.ndarray], np.ndarray],
    apply_J: Callable[[np.ndarray], np.ndarray],
    solve_J: Callable[[np.ndarray], np.ndarray],
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
    """Estimate ``N_AJ = lambda_max(M^{-1}Op)`` by power iteration."""
    if maxiter <= 0:
        raise ValueError("Expected maxiter > 0 for N_AJ iteration")
    if miniter <= 0:
        raise ValueError("Expected miniter > 0 for N_AJ iteration")
    if tol < 0.0:
        raise ValueError("Expected nonnegative tol for N_AJ iteration")

    rng = np.random.default_rng(seed)
    x = draw_random_vector(rng=rng, n=n, distribution=distribution, dtype=dtype)
    den0 = float(np.vdot(x, apply_J(x)).real)
    if den0 <= 0.0:
        den0 = float(np.vdot(x, x).real)
        if den0 <= 0.0:
            raise ValueError("Failed to initialize nonzero vector for N_AJ iteration")
    x /= np.sqrt(den0)

    lam_hist: list[float] = []
    rel_hist: list[float] = []
    abs_res_hist: list[float] = []
    rel_res_hist: list[float] = []
    converged = False
    prev = None

    for _k in range(maxiter):
        timer_add(timers, f"{timer_prefix}n_iter", 1.0)

        t_apply_op = perf_counter()
        y = apply_op(x)
        timer_add(timers, f"{timer_prefix}apply_op_sec", perf_counter() - t_apply_op)

        jx = apply_J(x)
        den = float(np.vdot(x, jx).real)
        if den <= 0.0:
            raise ValueError("Encountered non-positive M-norm in N_AJ iteration")
        lam = float(np.vdot(x, y).real / den)
        if lam < 0.0 and abs(lam) < 1e-12:
            lam = 0.0
        lam_hist.append(lam)

        res = np.asarray(y - lam * jx).reshape(-1)
        t_res_solve = perf_counter()
        zres = solve_J(res)
        timer_add(timers, f"{timer_prefix}solve_J_res_sec", perf_counter() - t_res_solve)
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

        t_solve_J = perf_counter()
        z = solve_J(y)
        timer_add(timers, f"{timer_prefix}solve_J_sec", perf_counter() - t_solve_J)

        t_apply_j = perf_counter()
        denz = float(np.vdot(z, apply_J(z)).real)
        timer_add(timers, f"{timer_prefix}apply_J_sec", perf_counter() - t_apply_j)
        if denz <= 0.0:
            converged = True
            break
        x = z / np.sqrt(denz)

    arr = np.asarray(lam_hist, dtype=float)
    rel_arr = np.asarray(rel_hist, dtype=float)
    abs_res_arr = np.asarray(abs_res_hist, dtype=float)
    rel_res_arr = np.asarray(rel_res_hist, dtype=float)
    if arr.size == 0:
        raise ValueError("No iterations executed in N_AJ iteration")
    return float(arr[-1]), arr, rel_arr, abs_res_arr, rel_res_arr, bool(converged)


def lobpcg_MinvOp(
    *,
    A,
    J,
    solve_J: Callable[[np.ndarray], np.ndarray],
    n: int,
    block_size: int,
    maxiter: int,
    tol: float,
    distribution: RandomDistribution,
    seed: int | None,
    Y=None,
    apply_QM_for_residual: Callable[[np.ndarray], np.ndarray] | None = None,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Estimate top generalized eigenvalue of ``A x = lambda J x`` using LOBPCG."""
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
    X = np.asarray(X, dtype=A.dtype)

    Y_lobpcg = None
    if Y is not None:
        if issparse(Y):
            raise ValueError(
                "SciPy lobpcg constraints require a dense Y array. "
                "Refusing to densify sparse P; use a projected method instead."
            )
        Y_lobpcg = np.asarray(Y, dtype=A.dtype)
        if Y_lobpcg.ndim == 1:
            Y_lobpcg = Y_lobpcg.reshape(-1, 1)
        if Y_lobpcg.shape[0] != n:
            raise ValueError(
                f"Constraint matrix Y has incompatible row count {Y_lobpcg.shape[0]} != {n}"
            )

    def m_solve(v):
        arr = np.asarray(v)
        if arr.ndim == 1:
            return solve_J(arr)
        out = np.column_stack([solve_J(arr[:, j]) for j in range(arr.shape[1])])
        return out

    M_prec = LinearOperator(shape=A.shape, matvec=lambda v: m_solve(v), dtype=A.dtype)

    t_lobpcg = perf_counter()
    vals, vecs, lam_hist_raw, _res_hist_raw = lobpcg(
        A,
        X,
        B=J,
        M=M_prec,
        Y=Y_lobpcg,
        tol=tol,
        maxiter=maxiter,
        largest=True,
        retLambdaHistory=True,
        retResidualNormsHistory=True,
    )
    timer_add(timers, f"{timer_prefix}solve_lobpcg_sec", perf_counter() - t_lobpcg)

    vals = np.asarray(vals, dtype=float).reshape(-1)
    j = int(np.argmax(vals))
    lam = float(vals[j])
    u = np.asarray(vecs[:, j]).reshape(-1)

    hist_vals: list[float] = []
    for h in lam_hist_raw:
        hh = np.asarray(h, dtype=float).reshape(-1)
        if hh.size:
            hist_vals.append(float(np.max(hh)))
    if not hist_vals:
        hist_vals = [lam]
    hist = np.asarray(hist_vals, dtype=float)
    rel_hist = rel_change_history(hist)

    ju = np.asarray(J @ u).reshape(-1)
    r = np.asarray(A @ u - lam * ju).reshape(-1)
    if apply_QM_for_residual is not None:
        r = apply_QM_for_residual(r)
    zres = solve_J(r)
    m_inv_norm_sq = float(np.vdot(r, zres).real)
    if m_inv_norm_sq < 0.0 and abs(m_inv_norm_sq) < 1e-12:
        m_inv_norm_sq = 0.0
    abs_res = float(np.sqrt(max(m_inv_norm_sq, 0.0)))
    den = float(np.vdot(u, ju).real)
    rel_res = float(abs_res / max(abs(lam) * np.sqrt(max(den, 0.0)), 1e-300))

    abs_res_hist = np.full(hist.shape, abs_res, dtype=float)
    rel_res_hist = np.full(hist.shape, rel_res, dtype=float)
    converged = bool(hist.size >= 2 and np.isfinite(rel_hist[-1]) and rel_hist[-1] <= tol)
    if timers is not None:
        timers[f"{timer_prefix}n_iter"] = float(hist.size)
    return lam, hist, rel_hist, abs_res_hist, rel_res_hist, converged


def power_iteration_AinvB(
    *,
    A,
    apply_B: Callable[[np.ndarray], tuple[np.ndarray, float]],
    solve_A: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    maxiter: int,
    tol: float,
    miniter: int,
    timers: dict[str, float] | None = None,
    timer_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Run power iteration on ``T = A^{-1}B`` using matrix-free B applications."""
    x = np.asarray(x0).reshape(-1).copy()
    x_A_norm_sq = float(np.vdot(x, A @ x).real)
    if x_A_norm_sq <= 0.0:
        raise ValueError(f"Initial vector has non-positive A-norm squared: {x_A_norm_sq}")
    x /= np.sqrt(x_A_norm_sq)

    lambda_hist: list[float] = []
    rel_hist: list[float] = []
    anorm_hist: list[float] = []

    lam_prev = None
    converged = False

    for _k in range(maxiter):
        timer_add(timers, f"{timer_prefix}n_iter", 1.0)

        t_apply_b = perf_counter()
        y, num = apply_B(x)
        timer_add(timers, f"{timer_prefix}apply_B_sec", perf_counter() - t_apply_b)
        den = float(np.vdot(x, A @ x).real)
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

        t_solve_a = perf_counter()
        z = solve_A(y)
        timer_add(timers, f"{timer_prefix}solve_A_sec", perf_counter() - t_solve_a)
        z_A_norm_sq = float(np.vdot(z, A @ z).real)
        if z_A_norm_sq <= 0.0:
            raise ValueError(f"Encountered non-positive A-norm squared during iteration: {z_A_norm_sq}")
        x = z / np.sqrt(z_A_norm_sq)

    return (
        np.asarray(lambda_hist, dtype=float),
        np.asarray(rel_hist, dtype=float),
        np.asarray(anorm_hist, dtype=float),
        bool(converged),
    )
