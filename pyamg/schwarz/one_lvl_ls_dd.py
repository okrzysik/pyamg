"""Single-level LS-DD setup with Schwarz smoothers and no hierarchy extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal
from warnings import warn

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_array, issparse
from scipy.sparse.linalg import LinearOperator

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import (
    setup_additive_schwarz,
    setup_rest_additive_schwarz,
    setup_rest_additive_schwarzT,
    setup_schwarz,
)
from pyamg.util.utils import asfptype, levelize_strength_or_aggregation

from .lsdd.aggregation import (
    _lsdd_build_aggop,
    _lsdd_build_strength,
    _lsdd_init_level_after_aggregation,
)
from .lsdd.local_ops import _lsdd_extract_local_principal_submatrices
from .lsdd.smoothers import lsdd_make_smoother_spec
from .lsdd.subdomains import _lsdd_build_overlap_and_pou
from .lsdd.types import SparseLike

Symmetry = Literal["symmetric", "hermitian"]
SmootherName = Literal["msm", "asm", "ras", "rasT"]
SmootherArg = SmootherName | tuple[SmootherName, dict[str, Any]] | None
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


def _instantiate_smoother(level, spec) -> SmootherFn | None:
    """Build a concrete smoother callable from a (name, kwargs) spec."""
    if spec is None:
        return None

    name, kwargs = spec
    if name == "schwarz":
        return setup_schwarz(level, **kwargs)
    if name == "additive_schwarz":
        return setup_additive_schwarz(level, **kwargs)
    if name == "rest_additive_schwarz":
        return setup_rest_additive_schwarz(level, **kwargs)
    if name == "rest_additive_schwarzT":
        return setup_rest_additive_schwarzT(level, **kwargs)

    raise ValueError(f"Unsupported smoother for one-level LS-DD: {name!r}")


@dataclass
class OneLevelLSDD:
    """Single-level LS-DD data with ready-to-apply Schwarz smoothers."""

    level: MultilevelSolver.Level
    presmoother: SmootherFn | None
    postsmoother: SmootherFn | None
    presmoother_spec: Any
    postsmoother_spec: Any

    @property
    def levels(self) -> list[MultilevelSolver.Level]:
        """Compatibility helper for call sites that expect a levels list."""
        return [self.level]

    def apply(self, b: np.ndarray, x0: np.ndarray | None = None, *, iterations: int = 1) -> np.ndarray:
        """Apply smoother-only preconditioner action to RHS ``b``."""
        x = np.zeros_like(b) if x0 is None else np.array(x0, copy=True)
        for _ in range(int(iterations)):
            if self.presmoother is not None:
                self.presmoother(self.level.A, x, b)
            if self.postsmoother is not None:
                self.postsmoother(self.level.A, x, b)
        return x

    def aspreconditioner(self, cycle: str | None = None, *, iterations: int = 1) -> LinearOperator:
        """Return LinearOperator that applies the one-level smoother action."""
        _ = cycle  # compatibility placeholder with MultilevelSolver.aspreconditioner
        n = int(self.level.A.shape[0])
        A = self.level.A

        def matvec(x):
            return self.apply(np.asarray(x).reshape(-1), iterations=iterations)

        return LinearOperator(shape=(n, n), matvec=matvec, dtype=A.dtype)

    def solve(
        self,
        b: np.ndarray,
        x0: np.ndarray | None = None,
        tol: float = 1e-5,
        maxiter: int = 100,
        cycle: str = "V",
        accel: Any = None,
        residuals: list[float] | None = None,
        callback: Callable[[np.ndarray], None] | None = None,
        return_info: bool = False,
    ):
        """Run a smoother-only iterative solve with a MultilevelSolver-like API."""
        if accel is not None:
            raise NotImplementedError("OneLevelLSDD.solve does not support accel; use accel=None")
        if str(cycle).upper() != "V":
            warn("OneLevelLSDD.solve ignores cycle and uses smoother-only iterations")

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

        for it in range(1, int(maxiter) + 1):
            if self.presmoother is not None:
                self.presmoother(A, x, b)
            if self.postsmoother is not None:
                self.postsmoother(A, x, b)

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


def one_level_ls_dd_solver(
    B: SparseLike,
    BT: SparseLike | None = None,
    A: SparseLike | None = None,
    *,
    presmoother: SmootherArg = "ras",
    postsmoother: SmootherArg = "rasT",
    symmetry: Symmetry = "symmetric",
    strength: Any = None,
    aggregate: Any = "standard",
    agg_levels: int = 1,
    force_row_closure: bool = False,
    print_info: bool = False,
) -> OneLevelLSDD:
    """Build a one-level LS-DD setup: aggregation -> subdomains -> Schwarz smoothers."""
    if symmetry not in ("symmetric", "hermitian"):
        raise ValueError('Expected "symmetric" or "hermitian" for the symmetry parameter')

    B_csr = _as_csr_fptype(B, name="B")

    if BT is None:
        BT_csr = B_csr.T.conjugate().tocsr()
        BT_csr.sort_indices()
    else:
        BT_csr = _as_csr_fptype(BT, name="BT")

    if A is None:
        A_csr = BT_csr @ B_csr
        A_csr = _as_csr_fptype(A_csr, name="A")
    else:
        A_csr = _as_csr_fptype(A.copy(), name="A")

    A_csr.symmetry = symmetry
    A_csr.schwarz_use_cholesky = True

    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError("expected square matrix")

    # Reuse levelized parsing and take the first (and only) level spec.
    max_levels = 2
    max_coarse = 1
    _, _, strength_lvls = levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    _, _, aggregate_lvls = levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)

    level = MultilevelSolver.Level()
    level.A = A_csr
    level.B = B_csr
    level.BT = BT_csr
    level.density = len(A_csr.data) / (A_csr.shape[0] ** 2)

    C = _lsdd_build_strength(A=A_csr, B=B_csr, strength_spec=strength_lvls[0])
    AggOp, _ = _lsdd_build_aggop(
        A=A_csr,
        C=C,
        aggregate_spec=aggregate_lvls[0],
        agg_levels=agg_levels,
        is_finest=True,
    )

    v_row_mult = _lsdd_init_level_after_aggregation(level=level, AggOp=AggOp, A=A_csr, B=B_csr)
    _lsdd_build_overlap_and_pou(
        level=level,
        A=A_csr,
        B=B_csr,
        v_row_mult=v_row_mult,
        force_row_closure=force_row_closure,
        print_info=print_info,
    )

    # Populate flattened subdomain and pointer arrays used by Schwarz smoothers.
    _lsdd_extract_local_principal_submatrices(level=level, A=A_csr)

    pre_spec = lsdd_make_smoother_spec(level=level, smoother=presmoother)
    post_spec = lsdd_make_smoother_spec(level=level, smoother=postsmoother)

    pre_fn = _instantiate_smoother(level, pre_spec)
    post_fn = _instantiate_smoother(level, post_spec)

    return OneLevelLSDD(
        level=level,
        presmoother=pre_fn,
        postsmoother=post_fn,
        presmoother_spec=pre_spec,
        postsmoother_spec=post_spec,
    )
