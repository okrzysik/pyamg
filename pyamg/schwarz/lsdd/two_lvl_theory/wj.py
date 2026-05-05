"""Block-Jacobi weak-approximation metric ``W_J`` for ``lsdd.two_lvl_theory``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, cg, eigsh, factorized, lobpcg

from .setup import build_one_level_lsdd_context
from .threshold import extract_tau_max_from_level
from .types import TwoLevelSolverParams


@dataclass(slots=True, frozen=True)
class WJConfig:
    """Numerical settings for ``W_J`` computation.

    ``W_J`` is computed as the largest generalized eigenvalue of
    ``B_J x = lambda A x``, where ``B_J = Q_J^T J Q_J = J Q_J`` and
    ``J=M_{omega J}`` is the aggregate block-Jacobi metric.  For
    moderate-size problems, ``eigensolver="eigsh"`` with a direct sparse
    factorization of ``A`` is usually much faster and more reliable than
    a plain power iteration.  The power method is retained as a fallback
    and for comparison with older runs.
    """

    eigensolver: Literal["eigsh", "lobpcg", "power"] = "eigsh"
    a_solve: Literal["cg", "direct"] = "direct"
    a_cg_rtol: float = 1.0e-10
    a_cg_atol: float = 0.0
    a_cg_maxiter: int | None = None
    maxiter: int = 120
    tol: float = 1.0e-10
    miniter: int = 3
    eigsh_ncv: int | None = None
    lobpcg_block_size: int = 8
    random_seed: int = 0


@dataclass(slots=True, frozen=True)
class WJResult:
    """Result container for ``W_J`` estimation."""

    W_J: float
    converged: bool
    n_iterations: int
    rel_change_history: np.ndarray
    a_norm_history: np.ndarray
    tau: float | None
    tau_max: float | None


@dataclass(slots=True)
class _DenseSPDSystem:
    """Factored dense SPD system helper with robust fallback."""

    use_cholesky: bool
    chol_factor: np.ndarray | None
    chol_lower: bool
    matrix: np.ndarray | None


@dataclass(slots=True)
class _LocalProjectionBlock:
    """Local projection data for one aggregate block on ``omega_i``."""

    omega_rows: np.ndarray
    A_i: np.ndarray
    Z_i: np.ndarray
    AZ_i: np.ndarray
    gram_system: _DenseSPDSystem | None


def _factor_dense_spd(A: np.ndarray) -> _DenseSPDSystem:
    """Factor dense SPD matrix with Cholesky fallback to dense solve."""
    M = np.asarray(A)
    M = 0.5 * (M + M.T)
    try:
        c, lower = cho_factor(M, lower=True, check_finite=False)
        return _DenseSPDSystem(
            use_cholesky=True,
            chol_factor=c,
            chol_lower=bool(lower),
            matrix=None,
        )
    except Exception:
        return _DenseSPDSystem(
            use_cholesky=False,
            chol_factor=None,
            chol_lower=True,
            matrix=M,
        )


def _solve_factored_dense_spd(sys: _DenseSPDSystem, rhs: np.ndarray) -> np.ndarray:
    """Solve factored dense SPD system."""
    b = np.asarray(rhs).reshape(-1)
    if sys.use_cholesky:
        if sys.chol_factor is None:
            raise ValueError("Missing Cholesky factor")
        return cho_solve((sys.chol_factor, sys.chol_lower), b, check_finite=False)
    if sys.matrix is None:
        raise ValueError("Missing dense matrix for fallback solve")
    return np.linalg.lstsq(sys.matrix, b, rcond=None)[0]


def _prepare_local_projection_blocks(level) -> list[_LocalProjectionBlock]:
    """Build local A_i-projector data from an LS-DD fine level."""
    P = level.P.tocsr()
    n_aggs = int(level.n_aggs)

    nev_arr = np.asarray(level.eigs.nev, dtype=np.int32)
    col_ptr = np.zeros(n_aggs + 1, dtype=np.int32)
    col_ptr[1:] = np.cumsum(nev_arr)

    blocks: list[_LocalProjectionBlock] = []
    for i in range(n_aggs):
        p0 = int(level.blocks.submatrices_ptr[i])
        p1 = int(level.blocks.submatrices_ptr[i + 1])
        a_flat = np.asarray(level.blocks.submatrices[p0:p1])
        dim = int(np.sqrt(a_flat.size))
        A_OMEGA = a_flat.reshape((dim, dim))

        pou_i = np.asarray(level.sub.PoU[i], dtype=float)
        omega_pos = np.flatnonzero(pou_i == 1)
        if omega_pos.size == 0:
            continue

        OMEGA_i = np.asarray(level.sub.OMEGA[i], dtype=np.int32)
        omega_rows = np.asarray(OMEGA_i[omega_pos], dtype=np.int32)
        A_i_raw = A_OMEGA[np.ix_(omega_pos, omega_pos)]
        A_i = 0.5 * (A_i_raw + A_i_raw.T)

        c0 = int(col_ptr[i])
        c1 = int(col_ptr[i + 1])
        if c1 > c0:
            Z_i = P[omega_rows, c0:c1].toarray()
            AZ_i = A_i @ Z_i
            G_i = Z_i.T @ AZ_i
            G_i = 0.5 * (G_i + G_i.T)
            gram_system = _factor_dense_spd(G_i)
        else:
            Z_i = np.zeros((omega_rows.size, 0), dtype=A_i.dtype)
            AZ_i = np.zeros((omega_rows.size, 0), dtype=A_i.dtype)
            gram_system = None

        blocks.append(
            _LocalProjectionBlock(
                omega_rows=omega_rows,
                A_i=A_i,
                Z_i=Z_i,
                AZ_i=AZ_i,
                gram_system=gram_system,
            )
        )

    return blocks


def _apply_local_QJ(block: _LocalProjectionBlock, x_i: np.ndarray) -> np.ndarray:
    """Apply local block-Jacobi coarse-complement projector ``Q_{J,i}``."""
    if block.Z_i.shape[1] == 0:
        return x_i
    if block.gram_system is None:
        raise ValueError("Missing local Gram system for non-empty local coarse space")
    rhs = block.AZ_i.T @ x_i
    alpha_i = _solve_factored_dense_spd(block.gram_system, rhs)
    return x_i - block.Z_i @ alpha_i


def _apply_QJ(x: np.ndarray, *, local_blocks: list[_LocalProjectionBlock], n_fine: int) -> np.ndarray:
    """Apply global blockwise coarse-complement map ``Q_J``."""
    xx = np.asarray(x).reshape(-1)
    if int(xx.size) != int(n_fine):
        raise ValueError(f"Expected x of length {n_fine}, got {xx.size}")
    y = xx.copy()
    for block in local_blocks:
        idx = block.omega_rows
        y[idx] = _apply_local_QJ(block, xx[idx])
    return y


def _apply_WJ_numerator_operator(
    *,
    x: np.ndarray,
    local_blocks: list[_LocalProjectionBlock],
    n_fine: int,
) -> tuple[np.ndarray, float]:
    """Apply ``B_J x = J Q_J x`` and return ``(B_J x, x^* B_J x)``.

    Here ``J`` is the aggregate block-Jacobi matrix assembled from the
    principal blocks ``A_{omega_i,omega_i}``.  Since ``Q_J`` is the
    ``J``-orthogonal coarse-complement projector, ``J Q_J = Q_J^T J Q_J``
    is the symmetric positive semidefinite numerator in the generalized
    eigenproblem for ``W_J``.
    """
    xx = np.asarray(x).reshape(-1)
    if int(xx.size) != int(n_fine):
        raise ValueError(f"Expected x of length {n_fine}, got {xx.size}")

    qx = _apply_QJ(x=xx, local_blocks=local_blocks, n_fine=n_fine)
    y = np.zeros(n_fine, dtype=xx.dtype)
    for block in local_blocks:
        idx = block.omega_rows
        y[idx] += block.A_i @ qx[idx]

    xBx = float(np.vdot(qx, y).real)
    if xBx < 0.0 and abs(xBx) < 1e-12:
        xBx = 0.0
    return y, xBx


def _build_linear_solver(
    *,
    A,
    method: Literal["cg", "direct"],
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build reusable linear-solver callable for repeated solves with fixed A."""
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


def _draw_gaussian_vector(*, rng: np.random.Generator, n: int, dtype) -> np.ndarray:
    """Draw one Gaussian random vector."""
    rr = rng.standard_normal(int(n))
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        ii = rng.standard_normal(int(n))
        return (rr + 1j * ii).astype(dtype, copy=False)
    return rr.astype(dtype, copy=False)



def _eigsh_generalized_eigen_matrix_free(
    *,
    metric_matrix,
    apply_numerator: Callable[[np.ndarray], tuple[np.ndarray, float]],
    solve_metric: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    maxiter: int,
    tol: float,
    ncv: int | None,
) -> tuple[float, bool]:
    """Compute the largest eigenvalue of ``B x = lambda A x`` with ARPACK.

    ``eigsh`` applies Lanczos iteration to the symmetric generalized EVP.
    The numerator is supplied as a matrix-free ``LinearOperator`` so that the
    block-Jacobi projector and local Gram solves are reused without forming the
    dense matrix ``B_J``.  ``solve_metric`` is exposed to ARPACK as ``A^{-1}``;
    for moderate problems a sparse direct factorization is usually preferable.
    """
    n = int(metric_matrix.shape[0])

    def matvec(x: np.ndarray) -> np.ndarray:
        y, _num = apply_numerator(np.asarray(x).reshape(-1))
        return y

    B_op = LinearOperator((n, n), matvec=matvec, dtype=metric_matrix.dtype)
    A_inv_op = LinearOperator(
        (n, n),
        matvec=lambda x: solve_metric(np.asarray(x).reshape(-1)),
        dtype=metric_matrix.dtype,
    )

    try:
        vals = eigsh(
            B_op,
            k=1,
            M=metric_matrix,
            Minv=A_inv_op,
            which="LA",
            v0=np.asarray(x0).reshape(-1),
            ncv=ncv,
            maxiter=maxiter,
            tol=tol,
            return_eigenvectors=False,
        )
        return float(np.max(vals).real), True
    except ArpackNoConvergence as err:
        if err.eigenvalues is None or len(err.eigenvalues) == 0:
            raise
        return float(np.max(err.eigenvalues).real), False


def _lobpcg_generalized_eigen_matrix_free(
    *,
    metric_matrix,
    apply_numerator: Callable[[np.ndarray], tuple[np.ndarray, float]],
    solve_metric: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_fine: int,
    block_size: int,
    maxiter: int,
    tol: float,
    dtype,
) -> tuple[float, bool]:
    """Compute the largest eigenvalue of ``B x = lambda A x`` with LOBPCG.

    LOBPCG can be useful when the largest eigenvalue is clustered.  The block
    size should be larger than the expected multiplicity of the top spectral
    cluster.  The preconditioner supplied here is an application of ``A^{-1}``,
    which is appropriate for this generalized problem.
    """
    n = int(n_fine)
    bs = max(1, min(int(block_size), max(1, n - 1)))

    def matvec(x: np.ndarray) -> np.ndarray:
        y, _num = apply_numerator(np.asarray(x).reshape(-1))
        return y

    B_op = LinearOperator((n, n), matvec=matvec, dtype=dtype)
    A_inv_op = LinearOperator(
        (n, n),
        matvec=lambda x: solve_metric(np.asarray(x).reshape(-1)),
        dtype=dtype,
    )
    X = np.column_stack([
        _draw_gaussian_vector(rng=rng, n=n, dtype=dtype) for _ in range(bs)
    ])

    vals, _vecs, lambda_hist, _residual_hist = lobpcg(
        B_op,
        X,
        B=metric_matrix,
        M=A_inv_op,
        largest=True,
        tol=tol,
        maxiter=maxiter,
        retLambdaHistory=True,
        retResidualNormsHistory=True,
    )
    converged = True
    if lambda_hist is not None and len(lambda_hist) >= 2:
        last = np.sort(np.asarray(lambda_hist[-1], dtype=float))[-1]
        prev = np.sort(np.asarray(lambda_hist[-2], dtype=float))[-1]
        rel = abs(last - prev) / max(abs(prev), 1.0)
        converged = bool(rel <= max(float(tol), 1.0e-14))
    return float(np.max(vals).real), converged


def _power_generalized_eigen_matrix_free(
    *,
    metric_matrix,
    apply_numerator: Callable[[np.ndarray], tuple[np.ndarray, float]],
    solve_metric: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    maxiter: int,
    tol: float,
    miniter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Estimate dominant ``lambda`` in ``numerator x = lambda metric x``."""
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

    for _k in range(int(maxiter)):
        y, num = apply_numerator(x)
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

        if len(lambda_hist) >= int(miniter) and rel <= float(tol):
            converged = True
            break

        if float(np.linalg.norm(y)) == 0.0:
            converged = True
            break

        z = solve_metric(y)
        z_metric_norm_sq = float(np.vdot(z, metric_matrix @ z).real)
        if z_metric_norm_sq <= 0.0:
            raise ValueError(
                f"Encountered non-positive metric-norm squared during iteration: {z_metric_norm_sq}"
            )
        x = z / np.sqrt(z_metric_norm_sq)

    return (
        np.asarray(lambda_hist, dtype=float),
        np.asarray(rel_hist, dtype=float),
        np.asarray(anorm_hist, dtype=float),
        bool(converged),
    )


def compute_W_J_from_level(*, level, cfg: WJConfig = WJConfig()) -> WJResult:
    """Compute ``W_J`` from an already-built LS-DD fine level."""
    local_blocks = _prepare_local_projection_blocks(level)
    if not local_blocks:
        raise ValueError("No local blocks were constructed; cannot compute W_J")

    A_csr = level.A.tocsr()
    n_fine = int(A_csr.shape[0])

    def apply_B(x: np.ndarray) -> tuple[np.ndarray, float]:
        return _apply_WJ_numerator_operator(x=x, local_blocks=local_blocks, n_fine=n_fine)

    solve_A = _build_linear_solver(
        A=A_csr,
        method=cfg.a_solve,
        cg_rtol=float(cfg.a_cg_rtol),
        cg_atol=float(cfg.a_cg_atol),
        cg_maxiter=cfg.a_cg_maxiter,
    )

    rng = np.random.default_rng(int(cfg.random_seed))
    x0 = _draw_gaussian_vector(rng=rng, n=n_fine, dtype=A_csr.dtype)

    if cfg.eigensolver == "eigsh":
        W_J, converged = _eigsh_generalized_eigen_matrix_free(
            metric_matrix=A_csr,
            apply_numerator=apply_B,
            solve_metric=solve_A,
            x0=x0,
            maxiter=int(cfg.maxiter),
            tol=float(cfg.tol),
            ncv=cfg.eigsh_ncv,
        )
        rel_hist = np.asarray([], dtype=float)
        anorm_hist = np.asarray([], dtype=float)
        n_iterations = 0
    elif cfg.eigensolver == "lobpcg":
        W_J, converged = _lobpcg_generalized_eigen_matrix_free(
            metric_matrix=A_csr,
            apply_numerator=apply_B,
            solve_metric=solve_A,
            rng=rng,
            n_fine=n_fine,
            block_size=int(cfg.lobpcg_block_size),
            maxiter=int(cfg.maxiter),
            tol=float(cfg.tol),
            dtype=A_csr.dtype,
        )
        rel_hist = np.asarray([], dtype=float)
        anorm_hist = np.asarray([], dtype=float)
        n_iterations = 0
    elif cfg.eigensolver == "power":
        lam_hist, rel_hist, anorm_hist, converged = _power_generalized_eigen_matrix_free(
            metric_matrix=A_csr,
            apply_numerator=apply_B,
            solve_metric=solve_A,
            x0=x0,
            maxiter=int(cfg.maxiter),
            tol=float(cfg.tol),
            miniter=int(cfg.miniter),
        )
        if lam_hist.size == 0:
            raise ValueError("No eigen-iterations executed for W_J")
        W_J = float(lam_hist[-1])
        n_iterations = int(lam_hist.size)
    else:
        raise ValueError(f"Unsupported W_J eigensolver {cfg.eigensolver!r}")

    if W_J < 0.0 and abs(W_J) < 1.0e-12:
        W_J = 0.0

    tau_data = extract_tau_max_from_level(level)
    return WJResult(
        W_J=float(W_J),
        converged=bool(converged),
        n_iterations=int(n_iterations),
        rel_change_history=rel_hist,
        a_norm_history=anorm_hist,
        tau=tau_data.tau,
        tau_max=tau_data.tau_max,
    )


def compute_W_J(
    *,
    B,
    A,
    BT,
    solver_params: TwoLevelSolverParams,
    cfg: WJConfig = WJConfig(),
) -> WJResult:
    """Build one LS-DD level and compute ``W_J``."""
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
    return compute_W_J_from_level(level=level, cfg=cfg)
