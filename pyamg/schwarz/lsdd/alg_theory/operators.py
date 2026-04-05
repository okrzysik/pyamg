"""Operator-application and solve helpers for algebraic-theory diagnostics."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import cho_solve
from scipy.sparse import csr_array
from scipy.sparse.linalg import cg, factorized

from pyamg.relaxation import relaxation as relax

from .linalg import factor_dense_spd, solve_factored_dense_spd
from .models import ASolveMethod, DenseSPDSystem, LocalProjectionBlock, TildeMetricOps, TildeProjectionMode


def flatten_subdomains_from_local_blocks(
    local_blocks: list[LocalProjectionBlock],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten omega-subdomains into Schwarz ``(subdomain, subdomain_ptr)`` arrays."""
    ptr = np.zeros(len(local_blocks) + 1, dtype=np.int32)
    chunks: list[np.ndarray] = []
    for i, blk in enumerate(local_blocks):
        idx = np.asarray(blk.omega_rows, dtype=np.int32).ravel()
        idx = np.sort(idx)
        chunks.append(idx)
        ptr[i + 1] = ptr[i] + int(idx.size)
    flat = np.concatenate(chunks).astype(np.int32, copy=False) if chunks else np.zeros(0, dtype=np.int32)
    return flat, ptr


def build_block_jacobi_solver_from_matrix(
    *,
    J,
    local_blocks: list[LocalProjectionBlock],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a fast block-Jacobi solve callable for matrix ``J``."""
    subdomain, subdomain_ptr = flatten_subdomains_from_local_blocks(local_blocks)
    _, _, inv_subblock, inv_subblock_ptr = relax.schwarz_parameters(
        J,
        subdomain=subdomain,
        subdomain_ptr=subdomain_ptr,
        inv_subblock=None,
        inv_subblock_ptr=None,
    )

    nblocks = int(subdomain_ptr.shape[0] - 1)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(nblocks):
        p0 = int(subdomain_ptr[i])
        p1 = int(subdomain_ptr[i + 1])
        q0 = int(inv_subblock_ptr[i])
        q1 = int(inv_subblock_ptr[i + 1])
        idx = subdomain[p0:p1]
        m = int(p1 - p0)
        inv_blk = inv_subblock[q0:q1].reshape((m, m))
        blocks.append((idx, inv_blk))

    def solve_J(rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs).reshape(-1)
        x = np.zeros_like(rhs)
        for idx, inv_blk in blocks:
            x[idx] = inv_blk @ rhs[idx]
        return x

    return solve_J


def local_residual(block: LocalProjectionBlock, x_i: np.ndarray) -> np.ndarray:
    """Apply local coarse residual map ``r_i = (I - Pi_i) x_i``."""
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


def apply_B_block_jacobi(
    *,
    x: np.ndarray,
    local_blocks: list[LocalProjectionBlock],
    n_fine: int,
) -> tuple[np.ndarray, float]:
    """Apply block-Jacobi WAP numerator operator ``B`` matrix-free."""
    y = np.zeros(n_fine, dtype=x.dtype)
    xBx = 0.0

    for block in local_blocks:
        x_i = x[block.omega_rows]
        r_i = local_residual(block, x_i)

        y_i = block.A_i @ r_i
        y[block.omega_rows] += y_i

        c_i = float(np.vdot(r_i, y_i).real)
        if c_i < 0.0 and abs(c_i) < 1e-12:
            c_i = 0.0
        xBx += c_i

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
