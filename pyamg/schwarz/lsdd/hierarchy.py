
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

from pyamg.multilevel import MultilevelSolver


def _lsdd_assemble_P_from_triplets(*, level, n_fine: int, p_r, p_c, p_v, counter: int) -> int:
    """Assemble prolongation P (and restriction R = P^H) from accumulated triplets.

    Refactor-only extraction of the existing "assemble_P" block.

    Parameters
    ----------
    level
        Current level object (typically levels[-1]); sets level.P and level.R.
    n_fine
        Fine dimension (A.shape[0]).
    p_r, p_c, p_v
        Lists of arrays/lists accumulated in the per-aggregate loop.
    counter
        Number of coarse columns accumulated so far.

    Returns
    -------
    counter : int
        Final number of coarse columns (n_coarse).
    """
    if len(p_r) == 0:
        # Fallback coarse space: one dof at row 0
        p_r = [[0]]
        p_c = [[0]]
        p_v = [[1]]
        counter = 1

    p_r = np.concatenate(p_r, dtype=np.int32)
    p_c = np.concatenate(p_c, dtype=np.int32)
    p_v = np.concatenate(p_v, dtype=np.float64)

    level.P = csr_array((p_v, (p_r, p_c)), shape=(n_fine, counter))
    level.R = level.P.T.conjugate().tocsr()
    return counter

def _lsdd_coarsen_operators(*, B, BT, P, R):
    """Form coarse operators via B_c = B P, BT_c = R BT, A_c = BT_c B_c.

    Refactor-only extraction of the existing "coarsen" block.
    """
    B_c = B @ P
    BT_c = R @ BT
    A_c = BT_c @ B_c
    A_c.sort_indices()  # IMPORTANT (keep behavior)
    return A_c, B_c, BT_c


def _lsdd_append_next_level(*, levels, A, B, BT):
    """Append a new level and set A/B/BT/density (refactor-only)."""
    levels.append(MultilevelSolver.Level())
    nxt = levels[-1]
    nxt.A = A
    nxt.B = B
    nxt.BT = BT
    nxt.density = len(nxt.A.data) / (nxt.A.shape[0] ** 2)
    return nxt
