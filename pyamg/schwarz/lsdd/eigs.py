"""LS–AMG–DD experimental: local generalized eigenproblem routines.

Refactor-only: isolates the per-aggregate local GEP + eigenvector selection used
to build prolongation columns on each level.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _lsdd_process_one_aggregate_gep(
    *,
    i: int,
    level,
    nev,
    min_coarsening,
    counter: int,
    p_r: list,
    p_c: list,
    p_v: list,
    eigvals_kept: list | None = None,
) -> int:
    """Solve one aggregate's local GEP and append selected vectors to P triplets.

    Refactor-only extraction of the original in-loop logic from least_squares_dd_exp.py.

    Parameters
    ----------
    i
        Aggregate index.
    level
        Current level object (typically levels[-1]).
    nev
        If not None, keep exactly the largest `nev` eigenpairs (subject to min_coarsening).
    min_coarsening
        Minimum coarsening ratio (scalar or None).
    counter
        Current coarse column counter (used to build P triplets).
    p_r, p_c, p_v
        Triplet lists (mutated in place).
    eigvals_kept
        If provided, append the eigenvalues that are actually kept (used for stats only).

    Returns
    -------
    counter : int
        Updated coarse column counter.
    """
    # Separate overlapping and nonoverlapping local DOFs (indices within Omega_i ordering)
    nonoverlap = np.where(level.PoU[i] == 1)[0]
    overlap = np.where(level.PoU[i] == 0)[0]

    # Extract flattened local blocks (both are blocksize^2 vectors)
    p0 = level.submatrices_ptr[i]
    p1 = level.submatrices_ptr[i + 1]
    b_flat = level.auxiliary[p0:p1]
    a_flat = level.submatrices[p0:p1]

    bsz = int(np.sqrt(len(b_flat)))
    bb = np.reshape(b_flat, (bsz, bsz))

    asz = int(np.sqrt(len(a_flat)))
    aa = np.reshape(a_flat, (asz, asz))

    # Regularization
    normbb = np.linalg.norm(bb, ord=2)
    bb = bb + np.eye(bb.shape[0]) * (1e-10 * normbb)

    # Enforce minimum coarsening ratio on per aggregate basis
    max_ev = aa.shape[0]
    this_nev = nev
    if min_coarsening is not None:
        max_ev = len(level.nonoverlapping_subdomain[i]) // min_coarsening
    if nev is not None and min_coarsening is not None:
        this_nev = np.min([nev, max_ev])

    # Restrict A block to nonoverlap indices
    aa = aa[nonoverlap, :][:, nonoverlap]

    if max_ev <= 0:
        return counter

    # Schur complement of outer product in nonoverlapping subdomain
    S = (
        bb[nonoverlap, :][:, nonoverlap]
        - bb[nonoverlap, :][:, overlap]
        @ np.linalg.inv(bb[overlap, :][:, overlap])
        @ bb[overlap, :][:, nonoverlap]
    )

    # When possible only compute necessary eigenvalues/vectors
    try:
        if max_ev != S.shape[0] and max_ev > 0:
            E, V = eigh(aa, S, subset_by_index=[S.shape[0] - max_ev, S.shape[0] - 1])
        else:
            E, V = eigh(aa, S)
    except Exception:
        import pdb

        pdb.set_trace()

    # Precompute the fine rows associated with ω_i (global indexing)
    temp_inds = np.arange(level.subdomain_ptr[i], level.subdomain_ptr[i + 1])[nonoverlap]
    temp_rows = level.subdomain[temp_inds]

    if this_nev is not None and this_nev > 0:
        # NOTE: assumes eigenvalues in increasing order
        E = E[-this_nev:]
        if eigvals_kept is not None:
            eigvals_kept.extend(E.tolist())

        level.min_ev = min(level.min_ev, E[-this_nev])
        V = V[:, -this_nev:]
        level.nev[i] = this_nev

        for j in range(len(E)):
            p_r.append(temp_rows)
            p_c.append([counter] * len(temp_rows))
            p_v.append(V[:, j])
            counter += 1

    else:
        counter_nev = 0
        kept_here = []
        # NOTE: assumes eigenvalues in increasing order
        for j in range(len(E) - 1, len(E) - np.min([len(E), max_ev]) - 1, -1):
            if E[j] > level.threshold:
                kept_here.append(E[j])
                counter_nev += 1
                level.min_ev = min(level.min_ev, E[j])
                p_r.append(temp_rows)
                p_c.append([counter] * len(temp_rows))
                p_v.append(V[:, j])
                counter += 1
        level.nev[i] = counter_nev

        if eigvals_kept is not None:
            eigvals_kept.extend(kept_here)

    return counter

