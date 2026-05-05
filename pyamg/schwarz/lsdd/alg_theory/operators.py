"""Operator-application and solve helpers for algebraic-theory diagnostics."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import cho_solve
from scipy.sparse import csr_array
from scipy.sparse.linalg import cg, factorized

from .linalg import factor_dense_spd, solve_factored_dense_spd
from .models import ASolveMethod, DenseSPDSystem, LocalProjectionBlock, TildeMetricOps, TildeProjectionMode


def assemble_block_jacobi_solver(
    *,
    local_blocks: list[LocalProjectionBlock],
    n_fine: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Assemble a block-Jacobi solver from local blocks.

    This factorizes each aggregate-local block ``A_i`` once, then applies
    ``J^{-1}`` by independent per-block solves:

    ``(J^{-1} rhs)[omega_i] = A_i^{-1} rhs[omega_i]``.
    """
    block_systems: list[tuple[np.ndarray, DenseSPDSystem]] = []
    for block in local_blocks:
        idx = np.asarray(block.omega_rows, dtype=np.int32).ravel()
        if idx.size == 0:
            continue
        A_i = np.asarray(block.A_i)
        A_i = 0.5 * (A_i + A_i.T)
        block_systems.append((idx, factor_dense_spd(A_i)))

    def solve_J(rhs: np.ndarray) -> np.ndarray:
        rr = np.asarray(rhs).reshape(-1)
        if int(rr.size) != int(n_fine):
            raise ValueError(f"Expected rhs of length {n_fine}, got {rr.size}")
        x = np.zeros_like(rr)
        for idx, sys_i in block_systems:
            x[idx] = solve_factored_dense_spd(sys_i, rr[idx])
        return x

    return solve_J


def apply_local_QJ(block: LocalProjectionBlock, x_i: np.ndarray) -> np.ndarray:
    """Apply the local block-Jacobi coarse-complement projector ``Q_{J,i}``.

    For one aggregate-local block, this computes

    ``Q_{J,i} x_i = (I - Pi_{J,i}) x_i``

    with

    ``Pi_{J,i} = Z_i (Z_i^T A_i Z_i)^{-1} Z_i^T A_i``.
    """
    if block.Z_i.shape[1] == 0:
        return x_i

    rhs = block.AZ_i.T @ x_i
    if block.use_cholesky:
        alpha_i = cho_solve((block.chol_factor, block.chol_lower), rhs, check_finite=False)
    else:
        if block.gram is None:
            raise ValueError("Missing local Gram matrix for least-squares fallback")
        alpha_i = np.linalg.lstsq(block.gram, rhs, rcond=None)[0]
    return x_i - block.Z_i @ alpha_i


def apply_QJ(
    x: np.ndarray,
    *,
    local_blocks: list[LocalProjectionBlock],
    n_fine: int,
) -> np.ndarray:
    """Apply the blockwise global coarse-complement map ``Q_J`` by looping blocks.

    This routine mirrors the per-block loop style used by
    :func:`apply_WJ_numerator_operator`: it slices each block, applies the local
    projector complement ``Q_{J,i}``, and writes the result back to the global
    vector.

    Notes
    -----
    The write-back uses direct assignment on each block index set. This is
    exact when the block index sets are disjoint (the intended block-Jacobi
    baseline). For overlapping index sets, the assignment semantics may differ
    from an additive global projector construction.
    """
    xx = np.asarray(x).reshape(-1)
    if int(xx.size) != int(n_fine):
        raise ValueError(f"Expected x of length {n_fine}, got {xx.size}")

    y = xx.copy()
    for block in local_blocks:
        idx = block.omega_rows
        y[idx] = apply_local_QJ(block, xx[idx])
    return y


def apply_WJ_numerator_operator(
    *,
    x: np.ndarray,
    local_blocks: list[LocalProjectionBlock],
    n_fine: int,
) -> tuple[np.ndarray, float]:
    """Apply the block-Jacobi WAP numerator operator.

    For a fine-level vector ``x``, this applies

    ``y = B_J x = A (I - Pi_J) x = A Q_J x``

    using the blockwise projector-complement routine :func:`apply_QJ`, then
    computes the Rayleigh numerator scalar

    ``x^* B_J x = <Q_J x, A (Q_J x)>``.
    """
    xx = np.asarray(x).reshape(-1)
    if int(xx.size) != int(n_fine):
        raise ValueError(f"Expected x of length {n_fine}, got {xx.size}")

    qx = apply_QJ(x=xx, local_blocks=local_blocks, n_fine=n_fine)
    y = np.zeros(n_fine, dtype=xx.dtype)
    for block in local_blocks:
        idx = block.omega_rows
        y[idx] += block.A_i @ qx[idx]

    xBx = float(np.vdot(qx, y).real)
    if xBx < 0.0 and abs(xBx) < 1e-12:
        xBx = 0.0
    return y, xBx


def assemble_block_jacobi_matrix(
    *,
    local_blocks: list[LocalProjectionBlock],
    n_fine: int,
    dtype,
) -> csr_array:
    """Assemble explicit block-Jacobi matrix ``M`` from local A_i blocks."""
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []

    for blk in local_blocks:
        idx = blk.omega_rows
        m = int(idx.size)
        if m == 0:
            continue
        rr = np.repeat(idx, m)
        cc = np.tile(idx, m)
        vv = np.asarray(blk.A_i, dtype=dtype).reshape(-1)
        rows_all.append(rr)
        cols_all.append(cc)
        vals_all.append(vv)

    if not rows_all:
        return csr_array((n_fine, n_fine), dtype=dtype)

    rows = np.concatenate(rows_all).astype(np.int32, copy=False)
    cols = np.concatenate(cols_all).astype(np.int32, copy=False)
    vals = np.concatenate(vals_all).astype(dtype, copy=False)

    M = csr_array((vals, (rows, cols)), shape=(n_fine, n_fine))
    M.sum_duplicates()
    M = 0.5 * (M + M.T)
    M.eliminate_zeros()
    M.sort_indices()
    return M.tocsr()


def build_linear_solver(
    *,
    A,
    method: ASolveMethod,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a reusable linear-solver callable for repeated solves with fixed A."""
    if method == "direct":
        A_fac = factorized(A.tocsc())

        def solve(rhs: np.ndarray) -> np.ndarray:
            rhs = np.asarray(rhs).reshape(-1)
            return np.asarray(A_fac(rhs), dtype=rhs.dtype)

        return solve

    if method == "cg":

        def solve(rhs: np.ndarray) -> np.ndarray:
            rhs = np.asarray(rhs).reshape(-1)
            z, info = cg(A, rhs, rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter)
            if info != 0:
                raise RuntimeError(f"CG solve did not converge (info={info})")
            return np.asarray(z, dtype=rhs.dtype)

        return solve

    raise ValueError(f"Unsupported solve method {method!r}")


def build_tilde_metric_ops(
    *,
    A_csr,
    J,
    zeta_eff: float,
    h_solve: ASolveMethod,
    h_cg_rtol: float,
    h_cg_atol: float,
    h_cg_maxiter: int | None,
) -> TildeMetricOps:
    """Build reusable damped/symmetrized metric operators for tilde-M actions."""
    M_damped = (J * (1.0 / float(zeta_eff))).tocsr()
    H = (2.0 * M_damped - A_csr).tocsr()
    H = 0.5 * (H + H.T)
    H.eliminate_zeros()
    H.sort_indices()
    solve_H = build_linear_solver(
        A=H,
        method=h_solve,
        cg_rtol=h_cg_rtol,
        cg_atol=h_cg_atol,
        cg_maxiter=h_cg_maxiter,
    )
    return TildeMetricOps(
        M_damped=M_damped,
        H=H,
        solve_H=solve_H,
    )


def apply_tildeM(
    *,
    x: np.ndarray,
    M: csr_array,
    solve_H: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply ``y = \tilde M x`` with ``\tilde M = M (2M - A)^{-1} M``."""
    t = np.asarray(M @ x).reshape(-1)
    z = solve_H(t)
    y = np.asarray(M @ z).reshape(-1)
    return y


def build_tilde_projection_data(
    *,
    P,
    apply_tildeM: Callable[[np.ndarray], np.ndarray],
    mode: TildeProjectionMode,
) -> tuple[DenseSPDSystem, np.ndarray | None]:
    """Build coarse projection data for ``Pi_tilde``."""
    P_csr = P.tocsr()
    P_T = P_csr.T.tocsr()
    n_coarse = int(P_csr.shape[1])

    if n_coarse == 0:
        raise ValueError("Coarse space is empty; cannot build tilde projection")

    if mode == "precompute_W":
        cols: list[np.ndarray] = []
        for j in range(n_coarse):
            p_j = np.asarray(P_csr[:, j].toarray()).reshape(-1)
            cols.append(apply_tildeM(p_j))
        W = np.column_stack(cols)
        C = np.asarray(P_T @ W)
    elif mode == "recompute":
        W = None
        C_cols: list[np.ndarray] = []
        for j in range(n_coarse):
            p_j = np.asarray(P_csr[:, j].toarray()).reshape(-1)
            w_j = apply_tildeM(p_j)
            c_j = np.asarray(P_T @ w_j).reshape(-1)
            C_cols.append(c_j)
        C = np.column_stack(C_cols)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported projection mode {mode!r}")

    C = 0.5 * (C + C.T)
    C_system = factor_dense_spd(C)
    return C_system, W


def apply_B_tilde(
    *,
    x: np.ndarray,
    P,
    P_T,
    apply_tildeM: Callable[[np.ndarray], np.ndarray],
    C_system: DenseSPDSystem,
    mode: TildeProjectionMode,
    W: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """Apply ``B_tilde x`` matrix-free."""
    y = apply_tildeM(x)
    rhs = np.asarray(P_T @ y).reshape(-1)
    alpha = solve_factored_dense_spd(C_system, rhs)

    if mode == "precompute_W":
        if W is None:
            raise ValueError("W must be provided when mode='precompute_W'")
        bx = y - W @ alpha
    else:
        r = x - np.asarray(P @ alpha).reshape(-1)
        bx = apply_tildeM(r)

    xBtx = float(np.vdot(x, bx).real)
    if xBtx < 0.0 and abs(xBtx) < 1e-12:
        xBtx = 0.0
    return np.asarray(bx).reshape(-1), xBtx
