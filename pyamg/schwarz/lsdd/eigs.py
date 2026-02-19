"""Local generalized eigenproblems used to construct prolongation columns.

Overview
--------
For each aggregate i on a given multigrid level, we work with a one-ring overlap
set OMEGA_i, which is split into:
  - omega_i : the nonoverlapping aggregate DOFs
  - GAMMA_i : the interface DOFs inside OMEGA_i \\ omega_i

The code assumes that, for each i, dense blocks have been precomputed and stored
in flattened form:
  - A_i := A[OMEGA_i, OMEGA_i]
  - Bsplit_i := \\tilde{A}_i  (an SPSD local splitting block built from rows of B)

We then form a Schur complement of Bsplit_i onto omega_i and solve a dense local
generalized eigenvalue problem on omega_i. Selected eigenvectors are injected
into the global prolongation operator P as columns supported on omega_i.

Key conventions
---------------
- `level.sub.PoU[i]` is a 0/1 mask over the ordering of OMEGA_i:
    PoU == 1  on omega_i entries
    PoU == 0  on GAMMA_i entries
- `level.blocks.*` stores flattened arrays and pointer arrays for all subdomains.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigh


def _lsdd_process_one_aggregate_gep(
    *,
    i: int,
    level: Any,
    nev: int | None,
    min_coarsening: int | None,
    counter: int,
    p_r: list,
    p_c: list,
    p_v: list,
    eigvals_kept: list[float] | None = None,
) -> int:
    """Solve aggregate i's local GEP and append selected vectors to P triplets.

    Parameters
    ----------
    i
        Aggregate index.

    level
        Current multigrid level object. Required attributes:

        Subdomain / PoU:
          - `level.sub.PoU[i]`: ndarray of shape (|OMEGA_i|,), entries in {0,1}.

        Flattened dense blocks and index pointers (via `level.blocks`):
          - `level.blocks.submatrices_ptr`: int32 array of shape (N+1,)
          - `level.blocks.submatrices`: 1D array containing concatenated A_i blocks
          - `level.blocks.auxiliary`: 1D array containing concatenated \\tilde{A}_i blocks
          - `level.blocks.subdomain_ptr`: int32 array of shape (N+1,)
          - `level.blocks.subdomain`: int32 array containing concatenated OMEGA_i indices

        Eigen-selection metadata:
          - `level.eigs.threshold` (float) and `level.eigs.min_ev` (float)
          - `level.eigs.nev` (int32 array length N)
          Optionally the same data may also be available under `level.eigs.*`.

    nev
        If not None: keep exactly the largest `nev` eigenpairs, additionally capped by
        `min_coarsening` if provided.

        If None: keep all eigenpairs with eigenvalue > `threshold`, scanning from
        largest to smaller eigenvalues.

    min_coarsening
        If not None, cap the number of kept eigenpairs by
            max_keep = floor(|omega_i| / min_coarsening).
        This is applied per aggregate.

    counter
        Current coarse column counter. Each accepted eigenvector becomes one new
        coarse column, and `counter` is incremented.

    p_r, p_c, p_v
        Lists that store triplets used later to assemble P:
          - p_r: list of int32 arrays of global row indices
          - p_c: list of int arrays (same length) with the coarse column id
          - p_v: list of float/complex arrays of values (same length as p_r entry)

        These lists are mutated in-place.

    eigvals_kept
        If provided, eigenvalues that were accepted (kept) on this aggregate are
        appended to this list. Used only for reporting/diagnostics.

    Returns
    -------
    counter
        Updated coarse column counter after inserting all accepted eigenvectors.

    Notes
    -----
    Local matrices:
      Let A_i = A[OMEGA_i, OMEGA_i] and Bsplit_i denote the local SPSD splitting block
      (same shape). Restrict A_i to omega_i:

          A_oo := A_i[omega, omega].

      Compute the Schur complement of Bsplit_i onto omega_i:

          S := Bsplit_oo - Bsplit_oG (Bsplit_GG)^{-1} Bsplit_Go.

      Solve the generalized EVP on omega_i:

          A_oo v = λ S v.

    Selection:
      - When `nev` is given: keep the largest `nev` eigenvalues/vectors.
      - Otherwise: keep eigenpairs with λ > threshold.

    The SciPy routine `scipy.linalg.eigh` returns eigenvalues in nondecreasing order.
    """
    pou = level.sub.PoU[i]
    omega = np.flatnonzero(pou == 1)
    GAMMA = np.flatnonzero(pou == 0)

    if omega.size == 0:
        return counter

    blocks = level.blocks

    # ---- unpack flattened local dense blocks ----
    p0 = blocks.submatrices_ptr[i]
    p1 = blocks.submatrices_ptr[i + 1]

    a_flat = blocks.submatrices[p0:p1]
    b_flat = blocks.auxiliary[p0:p1]

    # Both are square blocks stored flattened
    a_dim = int(np.sqrt(a_flat.size))
    b_dim = int(np.sqrt(b_flat.size))
    aa_full = a_flat.reshape((a_dim, a_dim))
    bb_full = b_flat.reshape((b_dim, b_dim))

    # ---- regularize bb to avoid breakdowns in the Schur complement ----
    normbb = float(np.linalg.norm(bb_full, ord=2))
    eps = 1e-10 * normbb if normbb != 0.0 else 1e-10
    bb_full = bb_full + eps * np.eye(bb_full.shape[0], dtype=bb_full.dtype)

    # ---- cap number of eigenpairs to keep (per aggregate) ----
    omega_size_global = int(level.sub.n_omega[i])
    max_keep = omega_size_global
    if min_coarsening is not None:
        max_keep = omega_size_global // int(min_coarsening)

    if max_keep <= 0:
        return counter

    # Restrict A to omega indices (indices are within OMEGA_i ordering)
    aa = aa_full[np.ix_(omega, omega)]
    if aa.shape[0] == 0:
        return counter

    max_keep = min(max_keep, aa.shape[0])

    # ---- Schur complement of bb onto omega ----
    if GAMMA.size == 0:
        S = bb_full[np.ix_(omega, omega)]
    else:
        bb_GG = bb_full[np.ix_(GAMMA, GAMMA)]
        bb_Go = bb_full[np.ix_(GAMMA, omega)]
        # X = (bb_GG)^{-1} bb_Go
        X = np.linalg.solve(bb_GG, bb_Go)
        S = bb_full[np.ix_(omega, omega)] - bb_full[np.ix_(omega, GAMMA)] @ X

    # ---- solve GEP; optionally compute only the largest max_keep eigenpairs ----
    if max_keep != S.shape[0]:
        lo = S.shape[0] - max_keep
        hi = S.shape[0] - 1
        E, V = eigh(aa, S, subset_by_index=[lo, hi])
    else:
        E, V = eigh(aa, S)

    # Map local omega indices -> global row indices for insertion into P
    idx0 = blocks.subdomain_ptr[i]
    idx1 = blocks.subdomain_ptr[i + 1]
    local_positions = np.arange(idx0, idx1, dtype=np.int32)[omega]
    global_rows = blocks.subdomain[local_positions]

    # Access threshold + per-aggregate nev array
    thr = float(level.eigs.threshold)

    # ---- selection ----
    if nev is not None:
        keep = min(int(nev), max_keep)
        if keep <= 0:
            return counter

        # E is increasing; keep the largest `keep`
        E_keep = E[-keep:]
        V_keep = V[:, -keep:]

        if eigvals_kept is not None:
            eigvals_kept.extend([float(x) for x in E_keep])

        # Track minimum kept eigenvalue across all aggregates
        min_kept = float(E_keep[0])
        if hasattr(level, "eigs") and level.eigs is not None:
            level.eigs.min_ev = min(level.eigs.min_ev, min_kept)
            level.eigs.nev[i] = keep
        level.eigs.min_ev = min(level.eigs.min_ev, min_kept)
        level.eigs.nev[i] = keep

        for j in range(keep):
            p_r.append(global_rows)
            p_c.append([counter] * len(global_rows))
            p_v.append(V_keep[:, j])
            counter += 1

        return counter

    # threshold-based selection from largest downwards
    kept_count = 0
    kept_vals: list[float] = []

    for j in range(E.size - 1, -1, -1):
        ev = float(E[j])
        if ev <= thr:
            break

        kept_vals.append(ev)
        kept_count += 1

        if hasattr(level, "eigs") and level.eigs is not None:
            level.eigs.min_ev = min(level.eigs.min_ev, ev)
        level.eigs.min_ev = min(level.eigs.min_ev, ev)

        p_r.append(global_rows)
        p_c.append([counter] * len(global_rows))
        p_v.append(V[:, j])
        counter += 1

    level.eigs.nev[i] = kept_count
    if hasattr(level, "eigs") and level.eigs is not None:
        level.eigs.nev[i] = kept_count

    if eigvals_kept is not None:
        eigvals_kept.extend(kept_vals)

    return counter
