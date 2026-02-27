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

from .types import LSDDLevel

import numpy as np
from scipy.linalg import eigh
from time import perf_counter
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from dataclasses import dataclass
from typing import Literal, Sequence
from scipy.linalg import cho_factor, cho_solve, LinAlgError, solve_triangular

def _lsdd_process_one_aggregate_gep(
    *,
    i: int,
    level: LSDDLevel,
    nev: int | None,
    min_coarsening: int | None,
    counter: int,
    p_r: list,
    p_c: list,
    p_v: list,
    eigvals_kept: list[float] | None = None,
    gep_timers,
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
        - `level.blocks.submatrices_ptr`: int32 array of shape (n_aggs+1,)
        - `level.blocks.submatrices`: 1D array containing concatenated A_i blocks
        - `level.blocks.auxiliary`: 1D array containing concatenated \\tilde{A}_i blocks
        - `level.blocks.subdomain_ptr`: int32 array of shape (n_aggs+1,)
        - `level.blocks.subdomain`: int32 array containing concatenated OMEGA_i indices

        Eigen-selection metadata:
        - `level.eigs.threshold` (float) and `level.eigs.min_ev` (float)
        - `level.eigs.nev` (int32 array length n_aggs)

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
        Lists that store triplets used later to assemble P. These are mutated in-place.

        - `p_r[k]`: int32 array of global row indices for the k-th prolongation column
        - `p_c[k]`: int32 array of the same length, filled with the coarse column id
        - `p_v[k]`: value array of the same length (float/complex), containing entries
            for the k-th prolongation column on rows `p_r[k]`

    eigvals_kept
        If provided, eigenvalues that were accepted (kept) on this aggregate are
        appended to this list. Used only for reporting/diagnostics.

    Returns
    -------
    counter
        Updated coarse column counter after inserting all accepted eigenvectors.

    Notes
    -----
    The SciPy routine `scipy.linalg.eigh` returns eigenvalues in nondecreasing order.
    """
    # Optional: accumulate sub-timings (seconds) across aggregates into `gep_timers`.
    # Expectation: `gep_timers` is a dict[str, float] (e.g. defaultdict(float)) or None.
    def _tadd(key: str, dt: float) -> None:
        if gep_timers is None:
            return
        gep_timers[key] = gep_timers.get(key, 0.0) + dt


    t0 = perf_counter()
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
    _tadd("gep_unpack", perf_counter() - t0)

    t0 = perf_counter()
    # ---- regularize bb to avoid breakdowns in the Schur complement ----
    # Use a cheap scale; avoid spectral norm (ord=2) which is SVD-cost.
    bb_full = bb_full.copy()  # do not mutate the flattened storage
    scale = float(np.linalg.norm(bb_full, ord="fro"))
    eps = 1e-10 * scale if scale != 0.0 else 1e-10
    #eps = 1e-3

    # add eps*I without allocating an identity matrix
    bb_full.flat[:: bb_full.shape[0] + 1] += eps
    _tadd("gep_regularize", perf_counter() - t0)

    t0 = perf_counter()
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
    _tadd("gep_cap_and_restrict", perf_counter() - t0)

    t0 = perf_counter()
    # ---- Schur complement of bb onto omega ----
    if GAMMA.size == 0:
        S = bb_full[np.ix_(omega, omega)]
    else:
        bb_GG = bb_full[np.ix_(GAMMA, GAMMA)]
        bb_Go = bb_full[np.ix_(GAMMA, omega)]
        # X = (bb_GG)^{-1} bb_Go
        try:
            c, lower = cho_factor(bb_GG, lower=True, check_finite=False)
            X = cho_solve((c, lower), bb_Go, check_finite=False)
        except LinAlgError:
            X = np.linalg.solve(bb_GG, bb_Go)

        S = bb_full[np.ix_(omega, omega)] - bb_full[np.ix_(omega, GAMMA)] @ X
    _tadd("gep_schur", perf_counter() - t0)

    t0 = perf_counter()
    # ---- solve GEP; compute only the eigenpairs we could possibly keep ----
    # `eigh(aa, S)` returns eigenvalues in nondecreasing order.
    # We select exactly one of:
    #   - subset_by_index: keep only the largest k eigenpairs (best when k is known/capped)
    #   - subset_by_value: keep only eigenpairs with lambda >= thr (best in threshold mode)
    nloc = S.shape[0]
    subset_kwargs: dict[str, object] = {}

    # If nev is set, we will keep at most `nev` eigenvectors (also capped by max_keep).
    if nev is not None:
        k = int(min(max_keep, nev))
        k = max(1, min(k, nloc))
        if k < nloc:
            lo = nloc - k
            hi = nloc - 1
            subset_kwargs["subset_by_index"] = [lo, hi]

    # Otherwise we are in threshold-selection mode.
    else:
        # If max_keep caps the number we could ever keep, subset by index is still best.
        k = int(max_keep)
        k = max(1, min(k, nloc))
        if k < nloc:
            lo = nloc - k
            hi = nloc - 1
            subset_kwargs["subset_by_index"] = [lo, hi]
        else:
            # No cap: try to avoid computing small eigenpairs below the threshold.
            subset_kwargs["subset_by_value"] = [float(thr), float("inf")]

    try:
        E, V = eigh(aa, S, **subset_kwargs) if subset_kwargs else eigh(aa, S)
    except TypeError:
        # Older SciPy may not support subset selection -> fall back to full solve.
        E, V = eigh(aa, S)
    _tadd("gep_eigh", perf_counter() - t0)

    t0 = perf_counter()
    # Map local omega indices -> global row indices for insertion into P
    idx0 = blocks.subdomain_ptr[i]
    idx1 = blocks.subdomain_ptr[i + 1]
    local_positions = np.arange(idx0, idx1, dtype=np.int32)[omega]
    global_rows = blocks.subdomain[local_positions]

    # Access threshold + per-aggregate nev array
    thr = float(level.eigs.threshold)
    _tadd("gep_map_rows", perf_counter() - t0)

    # ---- selection + triplet insertion ----
    if nev is not None:
        t0 = perf_counter()
        keep = min(int(nev), max_keep)
        if keep <= 0:
            return counter

        # E is increasing; keep the largest `keep`
        E_keep = E[-keep:]
        V_keep = V[:, -keep:]
        _tadd("gep_select", perf_counter() - t0)

        t0 = perf_counter()
        if eigvals_kept is not None:
            eigvals_kept.extend([float(x) for x in E_keep])

        # Persist per-aggregate accepted eigenvalues in the same order that
        # local basis columns are appended into P for this aggregate.
        if level.eigs.eigvals is not None:
            level.eigs.eigvals[i] = np.asarray(E_keep, dtype=float)

        # Track minimum kept eigenvalue across all aggregates
        min_kept = float(E_keep[0])
        level.eigs.min_ev = min(level.eigs.min_ev, min_kept)
        level.eigs.nev[i] = keep

        for j in range(keep):
            p_r.append(global_rows)
            p_c.append(np.full(global_rows.shape[0], counter, dtype=np.int32))
            p_v.append(V_keep[:, j])
            counter += 1
        _tadd("gep_triplets", perf_counter() - t0)

        return counter

    # threshold-based selection from largest downwards
    t0 = perf_counter()
    kept_count = 0
    kept_vals: list[float] = []

    for j in range(E.size - 1, -1, -1):
        ev = float(E[j])
        if ev <= thr:
            break

        kept_vals.append(ev)
        kept_count += 1

        level.eigs.min_ev = min(level.eigs.min_ev, ev)

        p_r.append(global_rows)
        p_c.append(np.full(global_rows.shape[0], counter, dtype=np.int32))
        p_v.append(V[:, j])
        counter += 1

    level.eigs.nev[i] = kept_count

    if eigvals_kept is not None:
        eigvals_kept.extend(kept_vals)

    # Persist per-aggregate accepted eigenvalues in the same order that
    # local basis columns are appended into P for this aggregate.
    if level.eigs.eigvals is not None:
        level.eigs.eigvals[i] = np.asarray(kept_vals, dtype=float)

    _tadd("gep_triplets", perf_counter() - t0)
    return counter




@dataclass(frozen=True, slots=True)
class LSDDTheoryAggregate:
    """Theory-aligned numerical objects for one aggregate.

    This lightweight container bridges the notation in the theory notes
    (e.g. "Low-rank structure in generalized eigenvalue problems") with the
    concrete objects stored/used by the implementation.

    All index arrays are expressed in the *OMEGA ordering* used by the dense blocks:

      - `Omega` is the global DOF index array defining the overlap OMEGA_i.
      - `omega_pos` / `Gamma_pos` are positions inside `Omega`.
      - `omega` / `Gamma` are the corresponding global DOF index arrays.

    For the row-weighting matrix W_omega in the notes, we store:

      - `M_omega = v_row_mult[R_omega]`  (row multiplicities)
      - `W_omega_diag = 1 / M_omega`     (diagonal of W_omega)

    Dense blocks
    -----------
    `A_Omega` and `Atilde_Omega_code` are the two dense |OMEGA_i|x|OMEGA_i| blocks
    used by the GEP plumbing:

      - `A_Omega`             = A[OMEGA_i, OMEGA_i]
      - `Atilde_Omega_code`   = \\tilde{A}_i, built by the local outer-product kernel
    """

    i: int

    # Global DOF index sets
    Omega: np.ndarray
    omega: np.ndarray
    Gamma: np.ndarray

    # Local positions inside the Omega ordering
    omega_pos: np.ndarray
    Gamma_pos: np.ndarray

    # Row set and weighting data (global row indices in B)
    R_omega: np.ndarray
    M_omega: np.ndarray
    W_omega_diag: np.ndarray

    # Dense blocks in the Omega ordering
    A_Omega: np.ndarray
    Atilde_Omega_code: np.ndarray

    # Useful permutation to place omega first, Gamma last
    perm_omega_Gamma: np.ndarray


def _lsdd_theory_extract_aggregate(*, i: int, level: LSDDLevel) -> LSDDTheoryAggregate:
    """Extract theory-named numerical objects for aggregate ``i``.

    Parameters
    ----------
    i
        Aggregate index.

    level
        Current LS–AMG–DD level. Required attributes:
          - `level.sub.PoU[i]` and `level.sub.R_rows[i]`
          - `level.blocks.subdomain`, `level.blocks.subdomain_ptr`
          - `level.blocks.submatrices`, `level.blocks.auxiliary`, `level.blocks.submatrices_ptr`
          - `level.v_row_mult` (row multiplicities produced by overlap construction)

    Returns
    -------
    LSDDTheoryAggregate
        Container of the aggregate's index sets, weights, and dense blocks, using
        variable names aligned with the theory notes.
    """
    print(f"Extracting theory objects for aggregate {i}...")
    blocks = level.blocks
    pou = level.sub.PoU[i]
    if pou is None:
        raise ValueError("Expected level.sub.PoU to be populated before theory extraction")

    # Omega in the exact ordering used by flattened dense blocks
    idx0 = blocks.subdomain_ptr[i]
    idx1 = blocks.subdomain_ptr[i + 1]
    Omega = blocks.subdomain[idx0:idx1]

    omega_pos = np.flatnonzero(pou == 1)
    Gamma_pos = np.flatnonzero(pou == 0)
    omega = Omega[omega_pos]
    Gamma = Omega[Gamma_pos]
    perm = np.concatenate([omega_pos, Gamma_pos]).astype(np.int32, copy=False)

    # Extract dense A_Omega and Atilde_Omega (code) blocks
    p0 = blocks.submatrices_ptr[i]
    p1 = blocks.submatrices_ptr[i + 1]
    a_flat = blocks.submatrices[p0:p1]
    b_flat = blocks.auxiliary[p0:p1]
    dim = int(np.sqrt(a_flat.size))
    if dim * dim != int(a_flat.size):
        raise ValueError(f"Malformed dense block for aggregate {i}: size={a_flat.size}")

    A_Omega = a_flat.reshape((dim, dim))
    Atilde_Omega_code = b_flat.reshape((dim, dim))

    # Row sets + multiplicities, aligned with theory notation
    R_omega = level.sub.R_rows[i]
    if R_omega is None:
        raise ValueError("Expected level.sub.R_rows to be populated before theory extraction")
    R_omega = np.asarray(R_omega, dtype=np.int32)

    M_omega = np.asarray(level.v_row_mult[R_omega], dtype=float)
    if np.any(M_omega <= 0):
        raise ValueError(
            "Encountered non-positive row multiplicity for rows in R_omega; "
            "this indicates v_row_mult was not constructed consistently."
        )
    W_omega_diag = 1.0 / M_omega

    return LSDDTheoryAggregate(
        i=i,
        Omega=Omega,
        omega=omega,
        Gamma=Gamma,
        omega_pos=omega_pos,
        Gamma_pos=Gamma_pos,
        R_omega=R_omega,
        M_omega=M_omega,
        W_omega_diag=W_omega_diag,
        A_Omega=A_Omega,
        Atilde_Omega_code=Atilde_Omega_code,
        perm_omega_Gamma=perm,
    )


def _lsdd_theory_scale_csr_rows(X, scales: np.ndarray):
    """Return a copy of ``X`` with each row scaled by ``scales``.

    Parameters
    ----------
    X
        CSR-like sparse matrix with attributes ``indptr`` and ``data``.

    scales
        1D array of length ``X.shape[0]``. Row ``r`` is multiplied by ``scales[r]``.

    Returns
    -------
    X_scaled
        A CSR matrix of the same shape/type as ``X`` with scaled row data.

    Notes
    -----
    This helper is used by theory/diagnostic routines to apply diagonal weights
    W_omega^{1/2} without relying on sparse broadcasting behavior.
    """
    X = X.tocsr(copy=True)
    if scales.shape[0] != X.shape[0]:
        raise ValueError("Row-scale vector has incompatible length")

    counts = np.diff(X.indptr)
    X.data *= np.repeat(scales, counts)
    return X


def _lsdd_theory_assemble_Atilde_Omega(
    *,
    i: int,
    level: LSDDLevel,
    weight_mode: Literal["inv", "none", "direct"] = "inv",
) -> np.ndarray:
    """Assemble a reference \\tilde{A}_Omega block from the theory definition.

    This routine explicitly forms

        \\tilde{A}_{\\Omega,\\text{manual}}
            = G(R_\\omega, \\Omega)^\\top \\; W_\\omega \\; G(R_\\omega, \\Omega),

    where ``G`` is the least-squares factor stored as ``level.B`` in the code.

    Parameters
    ----------
    i
        Aggregate index.

    level
        Current LS–AMG–DD level. Requires ``level.B`` and the overlap/row sets.

    weight_mode
        How to interpret the multiplicity array ``M = v_row_mult[R_omega]``:

        - ``"inv"``:    W_omega = diag(1 / M)   (matches the theory notes)
        - ``"none"``:   W_omega = I             (no weighting)
        - ``"direct"``: W_omega = diag(M)       (useful to detect accidental inversion)

    Returns
    -------
    Atilde_Omega_manual
        Dense array of shape (|Omega|, |Omega|).

    Notes
    -----
    This explicitly assembles a dense block (intended for exploration). It may be
    expensive if |Omega| is large and/or you sweep many aggregates.
    """
    agg = _lsdd_theory_extract_aggregate(i=i, level=level)

    # Build the tall submatrix G(R_omega, Omega)
    G = level.B
    G_R_Omega = G[agg.R_omega, :][:, agg.Omega]

    if weight_mode == "inv":
        w = agg.W_omega_diag
    elif weight_mode == "none":
        w = np.ones_like(agg.W_omega_diag)
    elif weight_mode == "direct":
        w = agg.M_omega
    else:  # pragma: no cover
        raise ValueError(f"Unknown weight_mode={weight_mode!r}")

    w_sqrt = np.sqrt(w)
    try:
        G_w = G_R_Omega.multiply(w_sqrt[:, None])
    except Exception:
        G_w = _lsdd_theory_scale_csr_rows(G_R_Omega, w_sqrt)

    return (G_w.T @ G_w).toarray()


def _lsdd_theory_outerprod_weighting_errors(
    *,
    i: int,
    level: LSDDLevel,
    weight_modes: Sequence[Literal["inv", "none", "direct"]] = ("inv", "none", "direct"),
) -> dict[str, float]:
    """Compare code \\tilde{A}_Omega against manual assemblies for different weight modes.

    Returns
    -------
    errors
        Mapping ``mode -> rel_error`` where

            rel_error = ||Atilde_code - Atilde_manual||_F / ||Atilde_manual||_F.

    Notes
    -----
    The intent is to identify which weight convention the compiled kernel uses.
    The correct convention should give errors near machine precision.
    """
    agg = _lsdd_theory_extract_aggregate(i=i, level=level)
    Atilde_code = np.asarray(agg.Atilde_Omega_code)

    out: dict[str, float] = {}
    for mode in weight_modes:
        Atilde_manual = _lsdd_theory_assemble_Atilde_Omega(i=i, level=level, weight_mode=mode)
        denom = float(np.linalg.norm(Atilde_manual, ord="fro"))
        if denom == 0.0:
            out[str(mode)] = float(np.linalg.norm(Atilde_code - Atilde_manual, ord="fro"))
        else:
            out[str(mode)] = float(np.linalg.norm(Atilde_code - Atilde_manual, ord="fro") / denom)
    return out


def _lsdd_theory_outerprod_weighting_sweep(
    *,
    level: LSDDLevel,
    agg_ids: Sequence[int],
    weight_modes: Sequence[Literal["inv", "none", "direct"]] = ("inv", "none", "direct"),
) -> dict[str, object]:
    """Run the outer-product weighting check on a set of aggregates.

    Parameters
    ----------
    level
        Current LS–AMG–DD level.

    agg_ids
        Aggregate indices to test.

    weight_modes
        Candidate weight conventions.

    Returns
    -------
    results
        Dictionary with keys:
          - ``"agg_ids"``, ``"weight_modes"``
          - ``"per_agg"``: dict ``i -> {mode -> error}``
          - ``"summary"``: dict ``mode -> {min, med, max}``
          - ``"best_mode_by_median"``

    Notes
    -----
    This is exploratory development code. It explicitly assembles dense blocks and
    can be expensive if ``agg_ids`` is large.
    """
    per_agg: dict[int, dict[str, float]] = {}
    for i in agg_ids:
        per_agg[int(i)] = _lsdd_theory_outerprod_weighting_errors(
            i=int(i),
            level=level,
            weight_modes=weight_modes,
        )

    summary: dict[str, dict[str, float]] = {}
    best_mode = None
    best_med = float("inf")
    for mode in weight_modes:
        arr = np.array([per_agg[i][str(mode)] for i in per_agg], dtype=float)
        if arr.size == 0:
            continue
        summary[str(mode)] = {
            "min": float(np.min(arr)),
            "med": float(np.median(arr)),
            "max": float(np.max(arr)),
        }
        if summary[str(mode)]["med"] < best_med:
            best_med = summary[str(mode)]["med"]
            best_mode = str(mode)

    return {
        "agg_ids": [int(i) for i in agg_ids],
        "weight_modes": [str(m) for m in weight_modes],
        "per_agg": per_agg,
        "summary": summary,
        "best_mode_by_median": best_mode,
    }

def _lsdd_theory_symmetrize_dense(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part (A + A.T)/2 as a float64 ndarray."""
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def _lsdd_theory_geigvals_spd(*, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Generalized eigenvalues of the SPD pencil (A,B), returned in descending order.

    Parameters
    ----------
    A, B
        Dense symmetric matrices of shape (n,n) with B SPD.

    Returns
    -------
    evals_desc
        Generalized eigenvalues of (A,B) sorted in descending order.

    Notes
    -----
    We compute eigenvalues of the symmetric similarity transform

        M = L^{-1} A L^{-T},

    where B = L L^T is the Cholesky factorization. The eigenvalues of M are the
    generalized eigenvalues of (A,B).
    """
    A = _lsdd_theory_symmetrize_dense(A)
    B = _lsdd_theory_symmetrize_dense(B)

    c, lower = cho_factor(B, lower=True, check_finite=False)
    L = c  # lower triangular

    Z = solve_triangular(L, A, lower=True, check_finite=False)
    M = solve_triangular(L, Z.T, lower=True, check_finite=False).T

    w = np.linalg.eigvalsh(_lsdd_theory_symmetrize_dense(M))  # ascending
    return w[::-1]


def _lsdd_theory_projection_pencil_spectrum(
    *,
    B: np.ndarray,
    S: np.ndarray,
    tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Spectrum of the pencil (B,S) on the extended real line.

    Parameters
    ----------
    B
        Dense SPD matrix of shape (n,n).
    S
        Dense SPSD matrix of shape (n,n). May be singular.
    tol
        Threshold on eigenvalues of M := L^{-1} S L^{-T} below which we treat them as 0,
        yielding +inf generalized eigenvalues. If None, a conservative default is used.

    Returns
    -------
    mu
        Eigenvalues of M = L^{-1} S L^{-T} in ascending order (mu >= 0). These are
        the eigenvalues of B^{-1/2} S B^{-1/2} in a Cholesky-based basis.
    gamma
        Generalized eigenvalues of the pencil (B,S) in descending order on the extended line,
        computed as gamma = 1/mu with +inf where mu <= tol. Because mu is sorted ascending,
        gamma is automatically sorted descending.
    n_inf
        Number of +inf generalized eigenvalues (i.e. count of mu <= tol).
    """
    B = _lsdd_theory_symmetrize_dense(B)
    S = _lsdd_theory_symmetrize_dense(S)

    c, lower = cho_factor(B, lower=True, check_finite=False)
    L = c

    Z = solve_triangular(L, S, lower=True, check_finite=False)
    M = solve_triangular(L, Z.T, lower=True, check_finite=False).T

    mu = np.linalg.eigvalsh(_lsdd_theory_symmetrize_dense(M))  # ascending
    mu = np.maximum(mu, 0.0)

    if tol is None:
        tol = float(np.max(mu)) * 1e-12 + 1e-15

    mask = mu > tol
    n_inf = int(mu.size - np.count_nonzero(mask))

    gamma = np.empty_like(mu)
    gamma[mask] = 1.0 / mu[mask]
    gamma[~mask] = np.inf

    return mu, gamma, n_inf


def _lsdd_theory_build_calligraphic_mats(
    *,
    i: int,
    level: LSDDLevel,
    weight_mode: Literal["inv", "none", "direct"] = "inv",
) -> dict[str, object]:
    """Assemble the theory matrices (calA, calB) and the weighted blocks H_omega/H_Gamma.

    Parameters
    ----------
    i
        Aggregate index.
    level
        Current LS–AMG–DD level (must expose level.B and the subdomain metadata used
        by `_lsdd_theory_extract_aggregate`).
    weight_mode
        How to interpret the multiplicity array M = v_row_mult[R_omega]:
          - "inv":    W_omega = diag(1/M)   (theory)
          - "none":   W_omega = I
          - "direct": W_omega = diag(M)     (diagnostic)

    Returns
    -------
    data
        Dictionary containing:
          - agg: LSDDTheoryAggregate
          - G_omega, G_Gamma (sparse)
          - H_omega, H_Gamma (sparse)
          - H_omega_dense, H_Gamma_dense (dense)
          - calA, calB (dense)
          - w_min, n_mult, n_Gamma (scalars)
    """
    agg = _lsdd_theory_extract_aggregate(i=i, level=level)

    G = level.B
    R_omega = agg.R_omega
    omega = agg.omega
    Gamma = agg.Gamma

    G_omega = G[R_omega, :][:, omega]
    if Gamma.size:
        G_Gamma = G[R_omega, :][:, Gamma]
    else:
        # empty (m_omega x 0) sparse block; use slicing to preserve sparse type
        G_Gamma = G_omega[:, :0]

    if weight_mode == "inv":
        w = agg.W_omega_diag
    elif weight_mode == "none":
        w = np.ones_like(agg.W_omega_diag)
    elif weight_mode == "direct":
        w = agg.M_omega
    else:
        raise ValueError(f"Unknown weight_mode={weight_mode!r}")

    w_sqrt = np.sqrt(w)
    try:
        H_omega = G_omega.multiply(w_sqrt[:, None])
        H_Gamma = G_Gamma.multiply(w_sqrt[:, None])
    except Exception:
        H_omega = _lsdd_theory_scale_csr_rows(G_omega, w_sqrt)
        H_Gamma = _lsdd_theory_scale_csr_rows(G_Gamma, w_sqrt)

    calA = (G_omega.T @ G_omega).toarray()
    calB = (H_omega.T @ H_omega).toarray()

    # diagnostics matching the theory definitions
    w_min = float(np.min(w)) if w.size else 1.0
    n_mult = int(np.count_nonzero(w < 1.0 - 1e-15))
    n_Gamma = int(Gamma.size)

    return {
        "agg": agg,
        "G_omega": G_omega,
        "G_Gamma": G_Gamma,
        "H_omega": H_omega,
        "H_Gamma": H_Gamma,
        "H_omega_dense": H_omega.toarray(),
        "H_Gamma_dense": H_Gamma.toarray(),
        "calA": _lsdd_theory_symmetrize_dense(calA),
        "calB": _lsdd_theory_symmetrize_dense(calB),
        "w_min": w_min,
        "n_mult": n_mult,
        "n_Gamma": n_Gamma,
        "weight_mode": weight_mode,
    }


def _lsdd_theory_build_projection_data(
    *,
    calB: np.ndarray,
    H_omega_dense: np.ndarray,
    H_Gamma_dense: np.ndarray,
    svd_tol: float | None = None,
) -> dict[str, object]:
    """Compute principal-angle data and assemble calD and calS.

    Returns
    -------
    data
        Dictionary containing:
          - r_Gamma, r_omega_Gamma, d_intersection
          - sigma (principal-angle cosines)
          - mu_pred, gamma_pred (predicted spectrum for (calB,calS))
          - calD, calS (dense)
    """
    calB = _lsdd_theory_symmetrize_dense(calB)

    m, n_omega = H_omega_dense.shape
    if H_Gamma_dense.shape[0] != m:
        raise ValueError("H_omega and H_Gamma must have the same number of rows")

    # --- Orthonormal basis for range(H_Gamma) via SVD ---
    if H_Gamma_dense.size == 0 or H_Gamma_dense.shape[1] == 0:
        r_Gamma = 0
        Q_Gamma = np.zeros((m, 0), dtype=float)
    else:
        U, s, _ = np.linalg.svd(H_Gamma_dense, full_matrices=False)
        if svd_tol is None:
            svd_tol = float(max(H_Gamma_dense.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0))
        r_Gamma = int(np.count_nonzero(s > svd_tol))
        Q_Gamma = U[:, :r_Gamma]

    # --- Orthonormal basis for range(H_omega): Q_omega^T = L^{-1} H_omega^T ---
    c, lower = cho_factor(calB, lower=True, check_finite=False)
    L = c
    Qt = solve_triangular(L, H_omega_dense.T, lower=True, check_finite=False)  # (n_omega, m)
    Q_omega = Qt.T  # (m, n_omega), columns orthonormal

    # --- Principal-angle cosines ---
    if r_Gamma == 0:
        sigma = np.zeros((0,), dtype=float)
    else:
        T = Q_Gamma.T @ Q_omega
        sigma = np.linalg.svd(T, compute_uv=False)

    r_omega_Gamma = int(min(n_omega, r_Gamma))
    sigma = sigma[:r_omega_Gamma]
    print("sigma (principal-angle cosines):", sigma)
    d_intersection = int(np.count_nonzero(sigma > 1.0 - 1e-12))

    # --- Predicted spectrum for the pencil (calB,calS) ---
    # mu_pred corresponds to eigenvalues of L^{-1} calS L^{-T} (ascending):
    mu_pred = np.concatenate([1.0 - sigma**2, np.ones(n_omega - r_omega_Gamma, dtype=float)])
    mu_pred = np.sort(mu_pred)

    tol_mu = float(np.max(mu_pred)) * 1e-12 + 1e-15
    mask = mu_pred > tol_mu
    gamma_pred = np.empty_like(mu_pred)
    gamma_pred[mask] = 1.0 / mu_pred[mask]
    gamma_pred[~mask] = np.inf  # sigma=1 case

    # --- Assemble calD and calS ---
    if r_Gamma == 0:
        calD = np.zeros_like(calB)
    else:
        C = Q_Gamma.T @ H_omega_dense  # (r_Gamma, n_omega)
        calD = C.T @ C

    calS = _lsdd_theory_symmetrize_dense(calB - calD)

    return {
        "r_Gamma": r_Gamma,
        "r_omega_Gamma": r_omega_Gamma,
        "d_intersection": d_intersection,
        "sigma": sigma,
        "mu_pred": mu_pred,
        "gamma_pred": gamma_pred,
        "calD": _lsdd_theory_symmetrize_dense(calD),
        "calS": calS,
        "svd_tol": svd_tol,
    }


def _lsdd_theory_schur_from_code_Atilde(
    *,
    agg: LSDDTheoryAggregate,
    eps_scale: float = 1e-10,
) -> np.ndarray:
    """Compute the code-style Schur complement S from the stored \\tilde{A}_Omega block.

    This mirrors the logic inside `_lsdd_process_one_aggregate_gep`:
      - take `bb_full = agg.Atilde_Omega_code`
      - add eps*I with eps = eps_scale * ||bb_full||_F
      - Schur complement onto omega with respect to the Gamma block, using Cholesky if possible

    Parameters
    ----------
    agg
        LSDDTheoryAggregate from `_lsdd_theory_extract_aggregate`.
    eps_scale
        Scaling for the diagonal regularization (matches the code default 1e-10).

    Returns
    -------
    S_code
        Dense Schur complement on omega (shape n_omega x n_omega) in the omega ordering.
    """
    bb_full = np.asarray(agg.Atilde_Omega_code, dtype=float).copy()

    scale = float(np.linalg.norm(bb_full, ord="fro"))
    eps = eps_scale * scale if scale != 0.0 else eps_scale
    bb_full.flat[:: bb_full.shape[0] + 1] += eps

    omega = agg.omega_pos
    GAMMA = agg.Gamma_pos

    if GAMMA.size == 0:
        return _lsdd_theory_symmetrize_dense(bb_full[np.ix_(omega, omega)])

    bb_GG = bb_full[np.ix_(GAMMA, GAMMA)]
    bb_Go = bb_full[np.ix_(GAMMA, omega)]

    try:
        c, lower = cho_factor(bb_GG, lower=True, check_finite=False)
        X = cho_solve((c, lower), bb_Go, check_finite=False)
    except LinAlgError:
        X = np.linalg.solve(bb_GG, bb_Go)

    S = bb_full[np.ix_(omega, omega)] - bb_full[np.ix_(omega, GAMMA)] @ X
    return _lsdd_theory_symmetrize_dense(S)


def _lsdd_theory_geigvals_spd_spsd(*, X: np.ndarray, Y: np.ndarray, tol: float | None = None) -> np.ndarray:
    """Generalized eigenvalues of the pencil (X,Y) on the extended real line.

    We interpret generalized eigenvalues of (X,Y) as values lambda in R ∪ {+inf}
    such that X u = lambda Y u with u != 0 and u^T Y u > 0, and vectors in ker(Y)
    correspond to lambda = +inf.

    Parameters
    ----------
    X
        Dense SPD matrix of shape (n,n).

    Y
        Dense symmetric SPSD matrix of shape (n,n). May be singular.

    tol
        Threshold used to decide which eigenvalues of X^{-1/2} Y X^{-1/2} are treated
        as zero (producing +inf). If None, a conservative default is chosen.

    Returns
    -------
    lambda_XY
        Array of length n containing generalized eigenvalues lambda(X,Y), sorted in
        descending order, with +inf values first (if any).
    """
    X = _lsdd_theory_symmetrize_dense(X)
    Y = _lsdd_theory_symmetrize_dense(Y)

    c, lower = cho_factor(X, lower=True, check_finite=False)
    L = c

    # M = L^{-1} Y L^{-T} (symmetric SPSD)
    Z = solve_triangular(L, Y, lower=True, check_finite=False)
    M = solve_triangular(L, Z.T, lower=True, check_finite=False).T
    M = _lsdd_theory_symmetrize_dense(M)

    theta = np.linalg.eigvalsh(M)  # ascending, theta >= 0 in exact arithmetic
    theta = np.maximum(theta, 0.0)

    if tol is None:
        tol = float(np.max(theta)) * 1e-12 + 1e-15

    lam = np.empty_like(theta)
    mask = theta > tol
    lam[mask] = 1.0 / theta[mask]
    lam[~mask] = np.inf

    # theta ascending => 1/theta descending, and zeros => +inf (largest)
    return lam

def _lsdd_theory_plot_local_pencils(
    *,
    i: int,
    n_aggs: int,
    agg: "LSDDTheoryAggregate",
    w_min: float,
    n_mult: int,
    rank_A_minus_B: int,
    r_omega_Gamma: int,
    d_omega_Gamma: int,
    lambda_AB: np.ndarray,
    lambda_BS: np.ndarray,
    lambda_BS_pred: np.ndarray,
    lambda_AS: np.ndarray,
    lambda_AS_bound: np.ndarray,
    block: bool = True,
) -> None:
    """Plot local pencil spectra using theory-consistent notation.

    Produces a 3-row figure (linear y-scale) showing:
      1) lambda(A,B) with bounds 1 and 1/w_min, and a vertical line at rank(A-B)
      2) lambda(B,S) and the predicted (1 - sigma_k^2)^{-1}, with vertical lines at
         d_{omega,Gamma} (if >0) and r_{omega,Gamma}
      3) lambda(A,S) with the theorem upper bound curve and relevant vertical lines

    All labels use LaTeX strings (wrapped in $...$) and larger font/marker settings.

    Notes
    -----
    This function enables TeX rendering via `text.usetex=True` so that $...$ works.
    """
    import matplotlib.pyplot as plt

    n_Omega = int(agg.Omega.size)
    n_omega = int(agg.omega.size)
    n_Gamma = int(agg.Gamma.size)

    # --- plotting style (big + readable) ---
    style = {
        #"text.usetex": True,
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    }
    lw = 3.0
    ms = 8.0
    alpha_grid = 0.25

    def _finite_max(arr: np.ndarray, fallback: float = 1.0) -> float:
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        return float(np.max(arr[finite])) if np.any(finite) else float(fallback)

    def _cap_inf(arr: np.ndarray, cap: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        out = arr.copy()
        out[~np.isfinite(out)] = cap
        return out

    # Choose caps/ylims per panel (linear scale)
    y1 = max(_finite_max(lambda_AB), (1.0 / w_min if w_min > 0 else 1.0), 1.0)
    y2 = max(_finite_max(lambda_BS), _finite_max(lambda_BS_pred), 1.0)
    y3 = max(_finite_max(lambda_AS), _finite_max(lambda_AS_bound), 1.0)

    # Add a small margin so markers aren’t on the top edge
    y1 *= 1.05
    y2 *= 1.05
    y3 *= 1.05

    lamAB_plot = _cap_inf(lambda_AB, y1)
    lamBS_plot = _cap_inf(lambda_BS, y2)
    lamBS_pred_plot = _cap_inf(lambda_BS_pred, y2)
    lamAS_plot = _cap_inf(lambda_AS, y3)
    lamAS_bound_plot = _cap_inf(lambda_AS_bound, y3)

    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 1, figsize=(11, 14), sharex=True)

        # -------------------------
        # Panel 1: lambda(A,B)
        # -------------------------
        ax = axes[0]
        k = np.arange(1, n_omega + 1)
        ax.plot(k, lamAB_plot, marker="o", markersize=ms, linewidth=lw, label="$\\lambda(A,B)$")
        ax.axhline(1.0, linewidth=lw, linestyle="-", label="$1$")
        if w_min > 0:
            ax.axhline(1.0 / w_min, linewidth=lw, linestyle="--", label="$1/w_{\\min}$")

        if 1 <= rank_A_minus_B <= n_omega:
            ax.axvline(rank_A_minus_B, linewidth=2.0, linestyle=":", alpha=0.8)

        ax.set_ylabel("$\\lambda$")
        ax.set_title(
            "$\\lambda(A,B) \\quad "
            + "\\left("
            + "w_{\\min}=" + f"{w_min:.2f}"
            + ",\\; n_{\\mathrm{mult}}=" + f"{n_mult:d}"
            + ",\\; \\mathrm{rank}(A-B)=" + f"{rank_A_minus_B:d}"
            + "\\right)$"
        )
        ax.grid(True, alpha=alpha_grid)
        ax.legend(loc="best")

        # -------------------------
        # Panel 2: lambda(B,S)
        # -------------------------
        ax = axes[1]
        ax.plot(k, lamBS_plot, marker="o", markersize=ms, linewidth=lw, label="$\\lambda(B,S)$")
        ax.plot(
            k,
            lamBS_pred_plot,
            marker="x",
            markersize=ms,
            linewidth=lw,
            label="$\\frac{1}{1-\\sigma_k^2}$",
        )
        ax.axhline(1.0, linewidth=lw, linestyle="-", label="$1$")

        if d_omega_Gamma > 0 and d_omega_Gamma <= n_omega:
            ax.axvline(d_omega_Gamma, linewidth=2.0, linestyle=":", alpha=0.8)
        if 1 <= r_omega_Gamma <= n_omega:
            ax.axvline(r_omega_Gamma, linewidth=2.0, linestyle="--", alpha=0.8)

        ax.set_ylabel("$\\lambda$")
        ax.set_title(
            "$\\lambda(B,S) \\quad "
            + "\\left("
            + "r_{\\omega,\\Gamma}=" + f"{r_omega_Gamma:d}"
            + ",\\; d_{\\omega,\\Gamma}=" + f"{d_omega_Gamma:d}"
            + "\\right)$"
        )
        ax.grid(True, alpha=alpha_grid)
        ax.legend(loc="best")

        # -------------------------
        # Panel 3: lambda(A,S) with theorem bound
        # -------------------------
        ax = axes[2]
        ax.plot(k, lamAS_plot, marker="o", markersize=ms, linewidth=lw, label="$\\lambda(A,S)$")
        ax.plot(k, lamAS_bound_plot, marker=None, linewidth=lw, linestyle="--", label="$\\mathrm{upper\\ bound}$")
        ax.axhline(1.0, linewidth=lw, linestyle="-", label="$1$")
        if w_min > 0:
            ax.axhline(1.0 / w_min, linewidth=lw, linestyle=":", label="$1/w_{\\min}$")

        if d_omega_Gamma > 0 and d_omega_Gamma <= n_omega:
            ax.axvline(d_omega_Gamma, linewidth=2.0, linestyle=":", alpha=0.8)
        if 1 <= r_omega_Gamma <= n_omega:
            ax.axvline(r_omega_Gamma, linewidth=2.0, linestyle="--", alpha=0.8)
        if 1 <= rank_A_minus_B <= n_omega:
            ax.axvline(rank_A_minus_B, linewidth=2.0, linestyle=":", alpha=0.5)

        # Also mark where the theorem says the tail is pinned at 1:
        k_tail = r_omega_Gamma + n_mult
        if 1 <= k_tail <= n_omega:
            ax.axvline(k_tail, linewidth=2.0, linestyle="-.", alpha=0.7)

        ax.set_xlabel("$k$")
        ax.set_ylabel("$\\lambda$")
        ax.set_title(
            "$\\lambda(A,S) \\quad "
            + "\\left("
            + "r_{\\omega,\\Gamma}=" + f"{r_omega_Gamma:d}"
            + ",\\; d_{\\omega,\\Gamma}=" + f"{d_omega_Gamma:d}"
            + ",\\; n_{\\mathrm{mult}}=" + f"{n_mult:d}"
            + "\\right)$"
        )
        ax.grid(True, alpha=alpha_grid)
        ax.legend(loc="best")

        # x-axis requirements
        axes[-1].set_xlim(1, n_omega)

        # Suptitle with sizes and "i out of total"
        fig.suptitle(
            "$\\mathrm{Aggregate}\\ "
            + f"{i}/{n_aggs}"
            + "\\quad |\\omega|=" + f"{n_omega}"
            + ",\\; |\\Gamma|=" + f"{n_Gamma}"
            + ",\\; |\\Omega|=" + f"{n_Omega}"
            + "$",
            fontsize=20,
        )
        fig.tight_layout()

        plt.show(block=block)
        plt.close(fig)

def _lsdd_theory_explore_gep_spectra(
    *,
    level: LSDDLevel,
    agg_ids: Sequence[int],
    print_k: int = 10,
    plot: bool = True,
    block: bool = True,
    weight_mode: Literal["inv", "none", "direct"] = "inv",
) -> list[dict[str, object]]:
    """Explore the theory spectra on selected aggregates using theory-consistent names.

    For each aggregate i we build (using theory notation):
      - A := G_omega^T G_omega
      - B := G_omega^T W_omega G_omega = H_omega^T H_omega
      - S := B - D, where D = H_omega^T Pi_Gamma H_omega (projection onto range(H_Gamma))

    and compute generalized eigenvalues (on the extended real line where relevant):
      - lambda_AB := lambda(A,B)
      - lambda_BS := lambda(B,S)
      - lambda_AS := lambda(A,S)

    plus the theory-predicted values:
      - lambda_BS_pred[k] = (1 - sigma_k^2)^{-1} for k<=r_{omega,Gamma}, else 1
      - lambda_AS_bound from Theorem (bounds for lambda(A,S)):
          * k=1..r_{omega,Gamma}:     <= (1/w_min) * (1 - sigma_k^2)^{-1}
          * k=r_{omega,Gamma}+1..r_{omega,Gamma}+n_mult: <= 1/w_min
          * k>r_{omega,Gamma}+n_mult: == 1

    Notes
    -----
    This prints a compact summary with 2 decimal digits and optionally shows an
    interactive (blocking) plot per aggregate.
    """
    results: list[dict[str, object]] = []
    n_aggs = int(level.n_aggs)

    def _fmt_arr(a: np.ndarray, k: int) -> str:
        a = np.asarray(a, dtype=float)
        k = min(k, a.size)
        out = []
        for j in range(k):
            val = a[j]
            out.append("inf" if not np.isfinite(val) else f"{val:.2f}")
        return "[" + ", ".join(out) + "]"

    for idx, i in enumerate(agg_ids, start=1):
        base = _lsdd_theory_build_calligraphic_mats(i=int(i), level=level, weight_mode=weight_mode)
        agg = base["agg"]

        A = base["calA"]
        B = base["calB"]
        w_min = float(base["w_min"])
        n_mult = int(base["n_mult"])

        proj = _lsdd_theory_build_projection_data(
            calB=B,
            H_omega_dense=base["H_omega_dense"],
            H_Gamma_dense=base["H_Gamma_dense"],
        )
        S = proj["calS"]

        r_omega_Gamma = int(proj["r_omega_Gamma"])
        d_omega_Gamma = int(proj["d_intersection"])
        sigma = np.asarray(proj["sigma"], dtype=float)

        # --- eigenvalues using consistent naming ---
        lambda_AB = _lsdd_theory_geigvals_spd(A=A, B=B)  # descending
        lambda_BS = _lsdd_theory_geigvals_spd_spsd(X=B, Y=S)  # descending, may include inf
        lambda_AS = _lsdd_theory_geigvals_spd_spsd(X=A, Y=S)  # descending, may include inf

        # rank(A-B)
        rank_A_minus_B = int(np.linalg.matrix_rank(_lsdd_theory_symmetrize_dense(A - B)))

        # --- predicted lambda(B,S) = (1 - sigma^2)^{-1} for k<=r, else 1 ---
        n_omega = int(A.shape[0])
        lambda_BS_pred = np.ones(n_omega, dtype=float)
        if r_omega_Gamma > 0:
            vals = 1.0 / (1.0 - sigma[:r_omega_Gamma] ** 2)
            # sigma==1 => inf by convention
            vals[np.abs(1.0 - sigma[:r_omega_Gamma]) < 1e-12] = np.inf
            lambda_BS_pred[:r_omega_Gamma] = vals

        # --- theorem bound for lambda(A,S) ---
        lambda_AS_bound = np.ones(n_omega, dtype=float)
        if w_min > 0:
            # First r terms: (1/w_min) * (1 - sigma^2)^{-1}
            lambda_AS_bound[:r_omega_Gamma] = (1.0 / w_min) * lambda_BS_pred[:r_omega_Gamma]
            # Next n_mult terms: <= 1/w_min
            k2 = min(r_omega_Gamma + n_mult, n_omega)
            if r_omega_Gamma < k2:
                lambda_AS_bound[r_omega_Gamma:k2] = 1.0 / w_min
            # Tail: == 1 already

        # ---- console output (2 decimals) ----
        n_Omega = int(agg.Omega.size)
        n_omega = int(agg.omega.size)
        n_Gamma = int(agg.Gamma.size)

        print("=" * 90)
        print(f"Aggregate {int(i)}/{n_aggs}  |omega|={n_omega}  |Gamma|={n_Gamma}  |Omega|={n_Omega}")
        print(f"w_min={w_min:.2f}  1/w_min={(1.0/w_min if w_min>0 else float('inf')):.2f}  n_mult={n_mult:d}  rank(A-B)={rank_A_minus_B:d}")
        print(f"r_(omega,Gamma)={r_omega_Gamma:d}  d_(omega,Gamma)={d_omega_Gamma:d}")
        print(f"lambda(A,B)[1:{min(print_k, n_omega)}] = {_fmt_arr(lambda_AB, print_k)}")
        print(f"lambda(B,S)[1:{min(print_k, n_omega)}] = {_fmt_arr(lambda_BS, print_k)}")
        print(f"(1-sigma_k^2)^(-1)[1:{min(print_k, n_omega)}] = {_fmt_arr(lambda_BS_pred, print_k)}")
        print(f"lambda(A,S)[1:{min(print_k, n_omega)}] = {_fmt_arr(lambda_AS, print_k)}")
        print(f"upper bound for lambda(A,S)[1:{min(print_k, n_omega)}] = {_fmt_arr(lambda_AS_bound, print_k)}")

        # ---- plotting ----
        if plot:
            _lsdd_theory_plot_local_pencils(
                i=int(i),
                n_aggs=n_aggs,
                agg=agg,
                w_min=w_min,
                n_mult=n_mult,
                rank_A_minus_B=rank_A_minus_B,
                r_omega_Gamma=r_omega_Gamma,
                d_omega_Gamma=d_omega_Gamma,
                lambda_AB=lambda_AB,
                lambda_BS=lambda_BS,
                lambda_BS_pred=lambda_BS_pred,
                lambda_AS=lambda_AS,
                lambda_AS_bound=lambda_AS_bound,
                block=block,
            )

        results.append(
            {
                "i": int(i),
                "A": A,
                "B": B,
                "S": S,
                "w_min": w_min,
                "n_mult": n_mult,
                "rank_A_minus_B": rank_A_minus_B,
                "r_omega_Gamma": r_omega_Gamma,
                "d_omega_Gamma": d_omega_Gamma,
                "sigma": sigma,
                "lambda_AB": lambda_AB,
                "lambda_BS": lambda_BS,
                "lambda_BS_pred": lambda_BS_pred,
                "lambda_AS": lambda_AS,
                "lambda_AS_bound": lambda_AS_bound,
                "Omega": agg.Omega,
                "omega": agg.omega,
                "Gamma": agg.Gamma,
            }
        )

    return results
    