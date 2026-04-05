"""Dense linear-algebra helpers for algebraic-theory computations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .models import DenseSPDSystem


def factor_dense_spd(G: np.ndarray) -> DenseSPDSystem:
    """Factor a dense SPD matrix with robust least-squares fallback."""
    Gs = 0.5 * (G + G.T)
    try:
        cf = cho_factor(Gs, lower=True, check_finite=False)
        return DenseSPDSystem(
            use_cholesky=True,
            chol_factor=np.asarray(cf[0]),
            chol_lower=bool(cf[1]),
            matrix=None,
        )
    except LinAlgError:
        return DenseSPDSystem(
            use_cholesky=False,
            chol_factor=None,
            chol_lower=True,
            matrix=Gs,
        )


def solve_factored_dense_spd(system: DenseSPDSystem, rhs: np.ndarray) -> np.ndarray:
    """Solve a dense SPD system prepared by :func:`factor_dense_spd`."""
    if system.use_cholesky:
        return np.asarray(
            cho_solve((system.chol_factor, system.chol_lower), rhs, check_finite=False)
        )
    if system.matrix is None:
        raise ValueError("Dense SPD fallback matrix is missing")
    return np.asarray(np.linalg.lstsq(system.matrix, rhs, rcond=None)[0])
