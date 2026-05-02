"""Fixed-subdomain Schwarz driver built from explicit overlapping subdomains.

This module provides a minimal smoothing-only wrapper around PyAMG's Schwarz
setup. The caller supplies a sparse SPD operator ``A`` and explicit overlapping
subdomains ``OMEGA`` with shape ``(n_subdomains, n_dofs_per_subdomain)``. The
driver normalizes each row of ``OMEGA`` to sorted ``int32`` indices, flattens the
rows into Schwarz ``subdomain`` / ``subdomain_ptr`` arrays, and constructs one
Schwarz smoother.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from warnings import warn

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_array, issparse
from scipy.sparse.linalg import LinearOperator

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import setup_additive_schwarz, setup_schwarz
from pyamg.util.utils import asfptype

from .lsdd.types import SparseLike


SmootherFn = Callable[[SparseLike, np.ndarray, np.ndarray], None]


def _as_csr_fptype(M: SparseLike, *, name: str):
    """Convert input sparse-like matrix to sorted CSR floating/complex type."""
    if not issparse(M) or M.format not in ("csr",):
        try:
            M = csr_array(M)
            warn(f"Implicit conversion of {name} to CSR", SparseEfficiencyWarning)
        except Exception as exc:
            raise TypeError(
                f"Argument {name} must have type csr_array/bsr_array or be convertible to csr_array"
            ) from exc
    M = asfptype(M)
    M = M.tocsr()
    M.eliminate_zeros()
    M.sort_indices()
    return M


def _normalize_omega(OMEGA: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize explicit overlapping subdomains into Schwarz array storage.

    Parameters
    ----------
    OMEGA
        Dense 2D array of shape ``(n_subdomains, n_dofs_per_subdomain)``. Each row
        contains the global DOF indices for one overlapping subdomain.

    Returns
    -------
    subdomain, subdomain_ptr
        Flattened subdomain indices and CSR-style pointers. Each row is sorted to
        satisfy the Schwarz routines' ordering requirement.
    """
    omega = np.asarray(OMEGA, dtype=np.int32)
    if omega.ndim != 2:
        raise ValueError(
            "OMEGA must be a 2D array of shape "
            "(n_subdomains, n_dofs_per_subdomain)"
        )
    if omega.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(omega.shape[0] + 1, dtype=np.int32)

    n_subdomains, n_dofs = omega.shape
    subdomain = np.empty(n_subdomains * n_dofs, dtype=np.int32)
    subdomain_ptr = np.arange(0, (n_subdomains + 1) * n_dofs, n_dofs, dtype=np.int32)

    for i in range(n_subdomains):
        row = np.asarray(omega[i], dtype=np.int32).ravel()
        if np.unique(row).size != row.size:
            raise ValueError(f"OMEGA row {i} contains duplicate DOF indices")
        subdomain[subdomain_ptr[i] : subdomain_ptr[i + 1]] = np.sort(row)

    return subdomain, subdomain_ptr


@dataclass
class SchwarzFSD:
    """Fixed-subdomain Schwarz solver with a single Schwarz smoother."""

    level: MultilevelSolver.Level
    smoother: SmootherFn | None
    subdomain: np.ndarray
    subdomain_ptr: np.ndarray
    smoother_spec: tuple[str, dict[str, Any]] | None

    def apply(self, b: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Apply one Schwarz preconditioner sweep to a right-hand side."""
        A = self.level.A
        b = np.asarray(b).reshape(-1)
        x = np.zeros_like(b) if x0 is None else np.asarray(x0).reshape(-1).copy()
        if self.smoother is not None:
            self.smoother(A, x, b)
        return x

    def solve(
        self,
        b: np.ndarray,
        x0: np.ndarray | None = None,
        tol: float = 1e-5,
        maxiter: int = 100,
        residuals: list[float] | None = None,
        callback: Callable[[np.ndarray], None] | None = None,
        return_info: bool = False,
    ):
        """Run smoothing iterations as a standalone solve."""
        A = self.level.A
        b = np.asarray(b).reshape(-1)
        x = np.zeros_like(b) if x0 is None else np.asarray(x0).reshape(-1).copy()

        normb = float(np.linalg.norm(b))
        if normb == 0.0:
            normb = 1.0

        normr0 = float(np.linalg.norm(b - A @ x))
        if residuals is not None:
            residuals[:] = [normr0]

        if callback is not None:
            callback(x)

        if normr0 < tol * normb:
            if return_info:
                return x, 0
            return x

        for _ in range(1, int(maxiter) + 1):
            x = self.apply(b, x0=x)

            normr = float(np.linalg.norm(b - A @ x))
            if residuals is not None:
                residuals.append(normr)
            if callback is not None:
                callback(x)

            if normr < tol * normb:
                if return_info:
                    return x, 0
                return x

        if return_info:
            return x, int(maxiter)
        return x

    def aspreconditioner(self) -> LinearOperator:
        """Return a linear operator that applies one Schwarz sweep."""
        n = int(self.level.A.shape[0])
        A = self.level.A

        def matvec(x):
            return self.apply(np.asarray(x).reshape(-1))

        return LinearOperator(shape=(n, n), matvec=matvec, dtype=A.dtype)


def schwarz_fsd_solver(
    A: SparseLike,
    OMEGA: np.ndarray,
    *,
    smoother: str = "msm",
    iterations: int = 1,
    sweep: str = "forward",
    omega: float = 0.5,
    withrho: bool = True,
    print_info: bool = False,
) -> SchwarzFSD:
    """Build a fixed-subdomain Schwarz smoothing driver from explicit overlaps."""
    _ = print_info  # reserved for future diagnostics

    A_csr = _as_csr_fptype(A.copy(), name="A")
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError("expected square matrix")

    A_csr.schwarz_use_cholesky = True

    subdomain, subdomain_ptr = _normalize_omega(OMEGA)

    level = MultilevelSolver.Level()
    level.A = A_csr

    smoother_kind = smoother.lower()
    if smoother_kind == "msm":
        smoother_spec = (
            "schwarz",
            {
                "subdomain": subdomain,
                "subdomain_ptr": subdomain_ptr,
                "iterations": int(iterations),
                "sweep": sweep,
            },
        )
        smoother_fn = setup_schwarz(
            level,
            iterations=int(iterations),
            subdomain=subdomain,
            subdomain_ptr=subdomain_ptr,
            sweep=sweep,
        )
    elif smoother_kind == "asm":
        smoother_spec = (
            "additive_schwarz",
            {
                "subdomain": subdomain,
                "subdomain_ptr": subdomain_ptr,
                "iterations": int(iterations),
                "omega": float(omega),
                "withrho": bool(withrho),
            },
        )
        smoother_fn = setup_additive_schwarz(
            level,
            iterations=int(iterations),
            subdomain=subdomain,
            subdomain_ptr=subdomain_ptr,
            omega=float(omega),
            withrho=bool(withrho),
        )
    else:
        raise ValueError(f"Unsupported smoother type for schwarz_fsd_solver: {smoother!r}")

    return SchwarzFSD(
        level=level,
        smoother=smoother_fn,
        subdomain=subdomain,
        subdomain_ptr=subdomain_ptr,
        smoother_spec=smoother_spec,
    )
