"""Spectral Domain Decomposition - Least Squares.

Copied from least_squares_dd.py at commit ba9bfe81d645dd94739fc36604779eb70ff608da

ruff check least_squares_dd_exp.py --select F,E9

PYAMG_LSDD_PRINT_INFO=1 pytest -q -s ../tests/schwarz/test_lsdd_compare.py
"""


from __future__ import annotations

from warnings import warn
import numpy as np
from scipy.sparse import csr_array, issparse, \
    SparseEfficiencyWarning, coo_array, csc_array, hstack
from scipy.linalg import eigh

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import asfptype, \
    levelize_strength_or_aggregation, levelize_weight, filter_matrix_rows
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection, \
    energy_based_strength_of_connection, distance_strength_of_connection, \
    algebraic_distance, affinity_distance
from pyamg.aggregation.aggregate import standard_aggregation, naive_aggregation, \
    lloyd_aggregation, balanced_lloyd_aggregation, \
    metis_aggregation, pairwise_aggregation
from pyamg import amg_core

import time




from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Dict, Optional


@dataclass
class LsddLevelStats:
    """Per-level setup stats for LS–AMG–DD.

    This is *refactor-only* scaffolding: it does not change the algorithm.

    Parameters
    ----------
    level
        Level index ℓ (0 = finest).
    n_fine
        Dimension of A_ℓ (equivalently: number of columns of G_ℓ / B_ℓ).
    n_aggs
        Number of aggregates on this level (|{ω_i^{(ℓ)} }|).
    n_coarse
        Dimension of A_{ℓ+1} after coarsening.

    Attributes
    ----------
    timings
        Wall-time in seconds for named phases (e.g. "strength", "aggregate", "gep").
    extra
        Misc scalar diagnostics (operator complexity, mean overlap size, nev totals, etc.).
    """

    level: int
    n_fine: int
    n_aggs: Optional[int] = None
    n_coarse: Optional[int] = None
    timings: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def timeit(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + (time.perf_counter() - t0)


def _lsdd_print_level_summary(stats: LsddLevelStats, *, print_info: bool, prefix: str = "LS-DD") -> None:
    """Print a one-line summary for a level (only if print_info=True)."""
    if not print_info:
        return

    parts = [f"{prefix} ℓ={stats.level}", f"n={stats.n_fine}"]
    if stats.n_aggs is not None:
        parts.append(f"n_aggs={stats.n_aggs}")
    if stats.n_coarse is not None:
        parts.append(f"n_c={stats.n_coarse}")

    if stats.timings:
        order = [
            "strength",
            "aggregate",
            "overlap",
            "extract_A",
            "outerprod",
            "gep",
            "assemble_P",
            "coarsen",
        ]
        t_parts = []
        for k in order:
            if k in stats.timings:
                t_parts.append(f"{k}={stats.timings[k]:.3f}s")
        for k, v in sorted(stats.timings.items()):
            if k not in order:
                t_parts.append(f"{k}={v:.3f}s")
        parts.append("timings:[" + ", ".join(t_parts) + "]")

    if stats.extra:
        extras = ", ".join(f"{k}={v}" for k, v in stats.extra.items())
        parts.append("extra:[" + extras + "]")

    print(" | ".join(parts))



def least_squares_dd_solver_exp(B, BT=None, A=None,
                            presmoother="ras",
                            postsmoother="rasT",
                            symmetry='hermitian', 
                            strength=None,
                            aggregate='standard',
                            agg_levels=1,
                            kappa=500,
                            nev=None,
                            threshold=None,
                            min_coarsening=None,
                            max_levels=10,
                            max_coarse=100,
                            filteringA=(False,0),
                            filteringB=(False,0),
                            max_density=0.1,
                            print_info=False,
                            **kwargs):
    if A is not None:
        A_provided = True
        if not issparse(A) or A.format not in ('csr'):
            try:
                A = csr_array(A)
                warn('Implicit conversion of A to CSR', SparseEfficiencyWarning)
            except Exception as e:
                raise TypeError('Argument A must have type csr_array or bsr_array, '
                                'or be convertible to csr_array') from e
    else:
        A_provided = False

    if not issparse(B) or B.format not in ('csr'):
        try:
            B = csr_array(B)
            warn('Implicit conversion of B to CSR', SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Argument B must have type csr_array or bsr_array, '
                            'or be convertible to csr_array') from e
            
    if BT is None:
        BT = B.T.conjugate().tocsr()
        BT.sort_indices()
        BT_provided = False
    else:
        BT_provided = True
        if not issparse(BT) or BT.format not in ('csr'):
            try:
                BT = csr_array(BT)
                warn('Implicit conversion of BT to CSR', SparseEfficiencyWarning)
            except Exception as e:
                raise TypeError('Argument BT must have type csr_array or bsr_array, '
                                'or be convertible to csr_array') from e

    B = asfptype(B)
    BT = asfptype(BT)

    if A is None:
        A = BT @ B
        A.tocsr()
        A.sort_indices()
    A = asfptype(A)
    A = A.tocsr()
    A.eliminate_zeros()
    A.sort_indices()    # THIS IS IMPORTANT
    
    if symmetry not in ('symmetric', 'hermitian', 'nonsymmetric'):
        raise ValueError('Expected "symmetric", "nonsymmetric" or "hermitian" '
                         'for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    kappa =  levelize_weight(kappa,max_levels)
    min_coarsening = levelize_weight(min_coarsening,max_levels)

    # Construct multilevel structure
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A          # Normal Equation Matrix
    levels[-1].B = B          # Least Squares Matrix
    levels[-1].BT = BT        # A is supposed to be spectrally equivalent to BT B and share the same sparsity pattern
    levels[-1].BT_provided = BT_provided
    levels[-1].A_provided = A_provided
    levels[-1].density = len(levels[-1].A.data) / (levels[-1].A.shape[0] ** 2)

    lvl = 0
    pre_smooth = []
    post_smooth = []
    while len(levels) < max_levels and \
        levels[-1].A.shape[0] > max_coarse and \
        levels[-1].density < max_density:
        if print_info:
            print("N = {}, density = {:.4g}".format( \
                levels[-1].A.shape[0], levels[-1].density))
        _extend_hierarchy(levels, strength, aggregate, agg_levels,\
            kappa[lvl], nev, threshold, min_coarsening[lvl],\
            filteringA, filteringB, print_info)

        # print("Hierarchy extended")
        if presmoother == "msm":
            sm1 = ('schwarz', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,\
                'iterations':1, 'sweep':'symmetric'})
        elif presmoother == "asm":
            sm1 = ('additive_schwarz', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,'iterations':1})
        elif presmoother == "ras":
            levels[-2].PoU_flat = np.concatenate(levels[-2].PoU)
            sm1 = ('rest_additive_schwarz', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,
                'POU': levels[-2].PoU_flat, 'iterations':1})
        elif presmoother == "rasT":
            levels[-2].PoU_flat = np.concatenate(levels[-2].PoU)
            sm1 = ('rest_additive_schwarzT', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,
                'POU': levels[-2].PoU_flat, 'iterations':1})
        elif presmoother == None:
            sm1 = None
        else:
            raise ValueError("Invalid smoother type.")

        if postsmoother == "msm":
            sm2 = ('schwarz', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,\
                'iterations':1, 'sweep':'symmetric'})
        elif postsmoother == "asm":
            sm2 = ('additive_schwarz', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,'iterations':1})
        elif postsmoother == "ras":
            levels[-2].PoU_flat = np.concatenate(levels[-2].PoU)
            sm2 = ('rest_additive_schwarz', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,
                'POU': levels[-2].PoU_flat, 'iterations':1})
        elif postsmoother == "rasT":
            levels[-2].PoU_flat = np.concatenate(levels[-2].PoU)
            sm2 = ('rest_additive_schwarzT', {'subdomain': levels[-2].subdomain,\
                'subdomain_ptr': levels[-2].subdomain_ptr,
                'POU': levels[-2].PoU_flat, 'iterations':1})
        elif postsmoother == None:
            sm2 = None
        else:
            raise ValueError("Invalid smoother type.")

        pre_smooth.append(sm1)
        post_smooth.append(sm2)
        lvl += 1

    ml = MultilevelSolver(levels, **kwargs)

    t0 = time.perf_counter()

    ### DEBUG: test other pointwise smoothers
    # pre_smoother = ('gauss_seidel', {'sweep': 'symmetric'})
    # post_smoother = ('gauss_seidel', {'sweep': 'symmetric'})
    # pre_smoother = ('jacobi')
    # post_smoother = ('jacobi')

    change_smoothers(ml, pre_smooth, post_smooth)
    
    t1 = time.perf_counter()
    if print_info:
        print("Smoother setup time = ", t1 - t0)

    return ml

def _lsdd_process_one_aggregate_gep(
    *,
    i: int,
    level,
    nev: int | None,
    min_coarsening,
    counter: int,
    p_r: list,
    p_c: list,
    p_v: list,
) -> int:
    """Process one aggregate's local generalized EVP and append selected vectors to P triplets.

    This is a refactor-only extraction of the existing in-loop logic.

    Parameters
    ----------
    i
        Aggregate index.
    level
        The current multigrid level object (typically `levels[-1]`).
    nev
        If not None, keep exactly the largest `nev` eigenpairs (subject to min_coarsening).
    min_coarsening
        Per-level minimum coarsening ratio (assumed scalar at this call site).
    counter
        Current coarse column counter (used to build P triplets).
    p_r, p_c, p_v
        Lists of row indices, col indices, and values used to assemble P after the loop.
        These lists are mutated in-place.

    Returns
    -------
    int
        Updated `counter`.
    """
    # Separate overlapping and nonoverlapping local DOFs (indices *within* Omega_i ordering)
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

    # Regularization (as in original code)
    normbb = np.linalg.norm(bb, ord=2)
    bb = bb + np.eye(bb.shape[0]) * (1e-10 * normbb)

    # Enforce minimum coarsening ratio on per-aggregate basis
    max_ev = aa.shape[0]
    this_nev = nev
    if min_coarsening is not None:
        max_ev = len(level.nonoverlapping_subdomain[i]) // min_coarsening
    if nev is not None and min_coarsening is not None:
        this_nev = np.min([nev, max_ev])

    # Local principal submatrix restricted to nonoverlapping aggregate
    aa = aa[nonoverlap, :][:, nonoverlap]

    # Schur complement of outer product in nonoverlapping subdomain
    S = (
        bb[nonoverlap, :][:, nonoverlap]
        - bb[nonoverlap, :][:, overlap]
        @ np.linalg.inv(bb[overlap, :][:, overlap])
        @ bb[overlap, :][:, nonoverlap]
    )

    # When possible only compute necessary eigenvalues/vectors
    if max_ev <= 0:
        # Nothing to keep; leave level.nev[i] as-is (initialized to 0)
        return counter

    try:
        if max_ev != S.shape[0] and max_ev > 0:
            E, V = eigh(aa, S, subset_by_index=[S.shape[0] - max_ev, S.shape[0] - 1])
        elif max_ev > 0:
            E, V = eigh(aa, S)
    except Exception:
        import pdb

        pdb.set_trace()

    # Precompute the fine rows associated with ω_i (in global indexing)
    temp_inds = np.arange(level.subdomain_ptr[i], level.subdomain_ptr[i + 1])[nonoverlap]
    temp_rows = level.subdomain[temp_inds]

    if this_nev is not None and this_nev > 0:
        # NOTE: assumes eigenvalues in increasing order
        E = E[-this_nev:]
        level.min_ev = min(level.min_ev, E[-this_nev])
        V = V[:, -this_nev:]
        level.nev[i] = this_nev

        for j in range(len(E)):
            p_r.append(temp_rows)
            p_c.append([counter] * len(temp_rows))
            p_v.append(V[:, j])
            counter += 1

    elif max_ev > 0:
        counter_nev = 0
        # NOTE: assumes eigenvalues in increasing order
        for j in range(len(E) - 1, len(E) - np.min([len(E), max_ev]) - 1, -1):
            if E[j] > level.threshold:
                counter_nev += 1
                level.min_ev = min(level.min_ev, E[j])
                p_r.append(temp_rows)
                p_c.append([counter] * len(temp_rows))
                p_v.append(V[:, j])
                counter += 1
        level.nev[i] = counter_nev

    return counter


def _lsdd_build_overlap_and_pou(
    *,
    level,
    A,
    B,
    BT,
    v_mult,
    v_row_mult,
    blocksize,
    print_info: bool,
):
    """Build overlapping subdomains, overlap-row sets, and PoU masks.

    Refactor-only extraction of the existing code inside the "overlap" timing block.

    Fills/updates on `level`:
      - nonoverlapping_subdomain[i]
      - overlapping_subdomain[i]
      - overlapping_rows[i]
      - PoU[i]
      - nodes_vs_subdomains, T, number_of_colors, multiplicity
      - nIi[i], ni[i]

    Mutates the provided arrays:
      - v_mult (node multiplicity over overlaps)
      - v_row_mult (row multiplicity for BT-row sets)
      - blocksize (|Omega_i|)
    """
    nodes_vs_subdomains_r = []
    nodes_vs_subdomains_c = []
    nodes_vs_subdomains_v = []

    for i in range(level.N):
        # List of aggregate indices for nonoverlapping subdomains
        level.nonoverlapping_subdomain[i] = np.asarray(
            level.AggOpT.indices[level.AggOpT.indptr[i] : level.AggOpT.indptr[i + 1]],
            dtype=np.int32,
        )
        level.nIi[i] = len(level.nonoverlapping_subdomain[i])
        level.overlapping_subdomain[i] = []

        # Form overlapping subdomains as all fine-grid neighbors of each coarse aggregate
        for j in level.nonoverlapping_subdomain[i]:
            level.overlapping_subdomain[i].append(A.indices[A.indptr[j] : A.indptr[j + 1]])

        level.overlapping_subdomain[i] = np.concatenate(level.overlapping_subdomain[i], dtype=np.int32)
        level.overlapping_subdomain[i] = np.unique(level.overlapping_subdomain[i])
        level.ni[i] = len(level.overlapping_subdomain[i])

        # Get the overlapping rows
        level.overlapping_rows[i] = []
        for j in level.nonoverlapping_subdomain[i]:
            level.overlapping_rows[i].append(BT.indices[BT.indptr[j] : BT.indptr[j + 1]])

        level.overlapping_rows[i] = np.concatenate(level.overlapping_rows[i], dtype=np.int32)
        level.overlapping_rows[i] = np.unique(level.overlapping_rows[i])
        v_row_mult[level.overlapping_rows[i]] += 1

        blocksize[i] = len(level.overlapping_subdomain[i])

        # Loop over the subdomain and get the PoU (here PoU is a 0/1 mask: 1 on ω, 0 on Γ)
        v_mult[level.overlapping_subdomain[i]] += 1
        nodes_vs_subdomains_r.append(level.overlapping_subdomain[i])
        nodes_vs_subdomains_c.append(i * np.ones(len(level.overlapping_subdomain[i]), dtype=np.int32))
        nodes_vs_subdomains_v.append(np.ones(len(level.overlapping_subdomain[i])))

    nodes_vs_subdomains_r = np.concatenate(nodes_vs_subdomains_r, dtype=np.int32)
    nodes_vs_subdomains_c = np.concatenate(nodes_vs_subdomains_c, dtype=np.int32)
    nodes_vs_subdomains_v = np.concatenate(nodes_vs_subdomains_v, dtype=np.float64)

    level.nodes_vs_subdomains = csr_array(
        (nodes_vs_subdomains_v, (nodes_vs_subdomains_r, nodes_vs_subdomains_c)),
        shape=(A.shape[0], level.N),
    )
    level.T = level.nodes_vs_subdomains.T @ level.nodes_vs_subdomains
    level.T.data[:] = 1
    k_c = level.T @ np.ones(level.T.shape[0], dtype=level.T.data.dtype)
    level.number_of_colors = max(k_c)
    level.multiplicity = max(v_row_mult)

    if print_info:
        print("\tMean blocksize = {:.4g}".format(np.mean(blocksize)))
        print("\tMax blocksize = {:.4g}".format(np.max(blocksize)))

    # Form partition of unity vector separating overlapping and nonoverlapping domains.
    for i in range(level.N):
        level.PoU[i] = []
        for j in level.overlapping_subdomain[i]:
            if j in level.nonoverlapping_subdomain[i]:
                level.PoU[i].append(1)
            else:
                level.PoU[i].append(0.0)
        level.PoU[i] = np.array(level.PoU[i])

    # Sanity check PoU correct (should reproduce any vector exactly)
    temp = np.random.rand(A.shape[0])
    temp2 = 0 * temp
    for i in range(level.N):
        temp2[level.overlapping_subdomain[i]] += level.PoU[i] * temp[level.overlapping_subdomain[i]]
    if np.linalg.norm(temp2 - temp) > 1e-14 * np.linalg.norm(temp):
        warn(
            "Partition of unity is incorrect. This can happen if the partitioning strategy "
            "did not yield a nonoverlapping cover of the set of nodes"
        )


def _lsdd_extract_local_principal_submatrices(*, level, A, blocksize):
    """Extract local principal submatrices A(Omega_i, Omega_i) for all aggregates i.

    Refactor-only extraction of the existing "extract_A" block.

    Parameters
    ----------
    level
        The current level object (typically levels[-1]).
    A
        Global operator on this level.
    blocksize
        Array of length level.N with blocksize[i] = |Omega_i|.

    Side effects (sets on `level`)
    ------------------------------
    subdomain, subdomain_ptr
        Flattened Omega_i indices and pointers (CSR-style).
    submatrices_ptr
        Pointers into flattened storage for each dense block (BSR-style).
    submatrices
        Flattened storage of all dense A(Omega_i,Omega_i) blocks.
    auxiliary
        Flattened storage for all dense splitting blocks (filled later).
    """
    # Form sparse indptr and indices for principal submatrices over subdomains
    level.subdomain = np.zeros(np.sum(blocksize), dtype=np.int32)
    level.subdomain_ptr = np.zeros(level.N + 1, dtype=np.int32)
    level.subdomain_ptr[0] = 0

    for i in range(level.N):
        level.subdomain_ptr[i + 1] = level.subdomain_ptr[i] + blocksize[i]
        level.subdomain[level.subdomain_ptr[i] : level.subdomain_ptr[i + 1]] = level.overlapping_subdomain[i]

    # BSR-like indexing: each i has a blocksize[i] x blocksize[i] dense block
    level.submatrices_ptr = np.zeros(level.N + 1, dtype=np.int32)
    level.submatrices_ptr[0] = 0
    for i in range(level.N):
        level.submatrices_ptr[i + 1] = level.submatrices_ptr[i] + blocksize[i] * blocksize[i]

    # Extract submatrices from overlapping subdomains
    level.submatrices = np.zeros(level.submatrices_ptr[-1], dtype=A.data.dtype)
    amg_core.extract_subblocks(
        A.indptr,
        A.indices,
        A.data,
        level.submatrices,
        level.submatrices_ptr,
        level.subdomain,
        level.subdomain_ptr,
        int(level.subdomain_ptr.shape[0] - 1),
        A.shape[0],
    )

    # Allocate auxiliary storage for local outer products (filled later)
    level.auxiliary = np.zeros(level.submatrices_ptr[-1])


def _lsdd_local_outer_products_and_gep_init(
    *,
    level,
    B,
    BT,
    v_row_mult,
    kappa,
    threshold,
):
    """Compute local outer products (splitting blocks) and initialize GEP/P assembly accumulators.

    Refactor-only extraction of the existing "outerprod" block content that:
      - calls amg_core.local_outer_product to fill `level.auxiliary`
      - sets `level.threshold`
      - initializes (p_r, p_c, p_v, counter) and `level.min_ev`

    Returns
    -------
    p_r, p_c, p_v : lists
        Triplet lists for assembling P after the per-aggregate loop.
    counter : int
        Starting coarse column counter (0).
    """
    # amg_core C++ implementation of extracting local subdomain outerproducts
    BTT = BT.T.conjugate().tocsr()

    rows_indptr = np.zeros(level.N + 1, dtype=np.int32)
    cols_indptr = np.zeros(level.N + 1, dtype=np.int32)
    for i in range(level.N):
        rows_indptr[i + 1] = rows_indptr[i] + len(level.overlapping_rows[i])
        cols_indptr[i + 1] = cols_indptr[i] + len(level.overlapping_subdomain[i])

    rows_flat = np.concatenate(level.overlapping_rows).astype(np.int32, copy=False)
    cols_flat = np.concatenate(level.overlapping_subdomain).astype(np.int32, copy=False)

    amg_core.local_outer_product(
        B.shape[0],
        B.shape[1],
        B.indptr,
        B.indices,
        B.data,
        BTT.indptr,
        BTT.indices,
        BTT.data,
        v_row_mult,
        rows_flat,
        rows_indptr,
        cols_flat,
        cols_indptr,
        level.auxiliary,
        level.submatrices_ptr,
    )

    if threshold is None:
        level.threshold = max(0.1, ((kappa / level.number_of_colors) - 1) / level.multiplicity)
    else:
        level.threshold = threshold

    p_r = []
    p_c = []
    p_v = []
    counter = 0
    level.min_ev = 1e12

    return p_r, p_c, p_v, counter


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


def _extend_hierarchy(levels, strength, aggregate, agg_levels,\
    kappa, nev, threshold, min_coarsening, filteringA, \
    filteringB, print_info):
    """Extend the multigrid hierarchy.

    Service routine to implement the strength of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    smoothed_aggregation_solver.

    """
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    A = levels[-1].A
    B = levels[-1].B
    BT = levels[-1].BT

    
    # Initialize stats object for this level
    stats = LsddLevelStats(level=len(levels) - 1, n_fine=A.shape[0])

    # -----------------------
    # --- Filter operator ---
    # -----------------------
    if len(levels) > 1:
        if (filteringB is not None) and (filteringB[1] != 0):
            if print_info:
                print("B NNZ before filtering", len(B.data))
                print("BT NNZ before filtering", len(BT.data))
            filter_matrix_rows(B, filteringB[1], diagonal=True, lump=filteringB[0])
            filter_matrix_rows(BT, filteringB[1], diagonal=True, lump=filteringB[0])
            if print_info:
                print("B NNZ after filtering", len(B.data))
                print("BT NNZ after filtering", len(BT.data))

        if (filteringA is not None) and (filteringA[1] != 0):
            if print_info:
                print("A NNZ before filtering", len(A.data))
            filter_matrix_rows(A, filteringA[1], diagonal=True, lump=filteringA[0])
            if print_info:
                print("B NNZ after filtering", len(B.data))


    # --------------------------------------
    # --- Compute strength of connection ---
    # --------------------------------------
    with stats.timeit("strength"):
        # Compute the strength-of-connection matrix C, where larger
        # C[i,j] denote stronger couplings between i and j.
        fn, kwargs = unpack_arg(strength[len(levels)-1])
        if fn == 'symmetric':
            C = symmetric_strength_of_connection(A, **kwargs)
        elif fn == 'classical':
            C = classical_strength_of_connection(A, **kwargs)
            # test = abs(A).tocsr()
            # if (np.max(test.indptr-C.indptr) != 0) or (np.max(test.indices-C.indices) != 0):
            #     import pdb; pdb.set_trace()
        elif fn == 'distance':
            C = distance_strength_of_connection(A, **kwargs)
        elif fn in ('ode', 'evolution'):
            if 'B' in kwargs:
                C = evolution_strength_of_connection(A, **kwargs)
            else:
                C = evolution_strength_of_connection(A, B, **kwargs)
        elif fn == 'energy_based':
            C = energy_based_strength_of_connection(A, **kwargs)
        elif fn == 'predefined':
            C = kwargs['C'].tocsr()
        elif fn == 'algebraic_distance':
            C = algebraic_distance(A, **kwargs)
        elif fn == 'affinity':
            C = affinity_distance(A, **kwargs)
        ### Hussam's original implementation
        elif fn is None:
            C = abs(A.copy()).tocsr()
        else:
            raise ValueError(f'Unrecognized strength of connection method: {fn!s}')


    # ----------------------------------------
    # --- Compute aggregation matrix AggOp ---
    # ----------------------------------------
    with stats.timeit("aggregate"):
        # Compute the aggregation matrix AggOp (i.e., the nodal coarsening of A).
        # AggOp is a boolean matrix, where the sparsity pattern for the k-th column
        # denotes the fine-grid nodes agglomerated into k-th coarse-grid node.
        fn, kwargs = unpack_arg(aggregate[len(levels)-1])
        C.eliminate_zeros()
        Cnodes = None
        Aggs = []
        for i in range(0,agg_levels):
            if fn == 'standard':
                AggOp, Cnodes = standard_aggregation(C, **kwargs)
            elif fn == 'd2C':
                C = C @ C
                AggOp, Cnodes = standard_aggregation(C, **kwargs)
            elif fn == 'd3C':
                C = C @ C @ C
                AggOp, Cnodes = standard_aggregation(C, **kwargs)
            elif fn == 'naive':
                AggOp, Cnodes = naive_aggregation(C, **kwargs)
            elif fn == 'lloyd':
                AggOp, Cnodes = lloyd_aggregation(C, **kwargs)
            elif fn == 'balanced lloyd':
                if 'pad' in kwargs:
                    kwargs['A'] = A
                AggOp, Cnodes = balanced_lloyd_aggregation(C, **kwargs)
            elif fn == 'metis':
                C.data[:] = 1.0
                if(len(levels) == 1):
                    AggOp = metis_aggregation(C, **kwargs)
                else:
                    #ratio = levels[-2].N/16/levels[-1].A.shape[0]
                    # ratio = max(levels[-2].nev)*4/levels[-1].A.shape[0]
                    # AggOp = metis_aggregation(C, ratio=ratio)
                    AggOp = metis_aggregation(C, **kwargs)
            elif fn == 'pairwise':
                AggOp = pairwise_aggregation(A, **kwargs)[0]
            elif fn == 'predefined':
                AggOp = kwargs['AggOp'].tocsr()
            else:
                raise ValueError(f'Unrecognized aggregation method {fn!s}')

            Aggs.append(AggOp)
            if i < agg_levels-1:
                C = (AggOp.T @ C @ AggOp).tocsr()

        # Create aggregation matrix as product of levels
        AggOp = Aggs[0]
        for i in range(1,agg_levels):
            AggOp = AggOp @ Aggs[i]

        # AggOp = AggOp.tocsc()
        # AggOp = _remove_empty_columns(AggOp)
        # AggOp = _add_columns_containing_isolated_nodes(AggOp)
        # AggOp = AggOp.tocsr()
        nc_temp = AggOp.shape[1]
        print("\tNum aggregates = {:.4g}".format(nc_temp))
        print("\tAv. aggregate size = {:.4g}".format(AggOp.shape[0]/AggOp.shape[1]))
        AggOp = _fill_unaggregated_by_neighbors(A, AggOp, make_singletons=True)
        print("\tNum singletons = {:.4g}".format(AggOp.shape[1]-nc_temp))

        levels[-1].AggOp = AggOp
        levels[-1].AggOpT = AggOp.T.tocsr()
        levels[-1].N = AggOp.shape[1]  # number of coarse grid points
        levels[-1].nonoverlapping_subdomain = [None]*levels[-1].N
        levels[-1].overlapping_subdomain = [None]*levels[-1].N
        levels[-1].PoU = [None]*levels[-1].N
        levels[-1].overlapping_rows = [None]*levels[-1].N
        levels[-1].nIi = np.zeros(levels[-1].N, dtype=np.int32)
        levels[-1].ni = np.zeros(levels[-1].N, dtype=np.int32)
        levels[-1].nev = np.zeros(levels[-1].N, dtype=np.int32)
        v_mult = np.zeros(AggOp.shape[0])
        blocksize = np.zeros(levels[-1].N, dtype=np.int32)
        v_row_mult = np.zeros(B.shape[0])
        stats.n_aggs = levels[-1].N


    
    # ----------------------------------------------------------
    # --- Form overlapping subdomains and partition of unity ---
    # ----------------------------------------------------------
    with stats.timeit("overlap"):
        _lsdd_build_overlap_and_pou(
            level=levels[-1],
            A=A,
            B=B,
            BT=BT,
            v_mult=v_mult,
            v_row_mult=v_row_mult,
            blocksize=blocksize,
            print_info=print_info,
        )
    
    # ---------------------------------------------------------------------
    # --- Extract local principle submatrices on overlapping subdomains ---
    # ---------------------------------------------------------------------
    with stats.timeit("extract_A"):
        _lsdd_extract_local_principal_submatrices(level=levels[-1], A=A, blocksize=blocksize)

    # -----------------------------------------------------------
    # --- Form local outer products on overlapping subdomains ---
    # -----------------------------------------------------------
    with stats.timeit("outerprod"):
        p_r, p_c, p_v, counter = _lsdd_local_outer_products_and_gep_init(
            level=levels[-1],
            B=B,
            BT=BT,
            v_row_mult=v_row_mult,
            kappa=kappa,
            threshold=threshold,
        )

    # ---------------------------------------------------
    # --- Solve local generalized eigenvalue problems ---
    # ---------------------------------------------------
    with stats.timeit("gep"):
        level = levels[-1]
        for i in range(level.N):
            counter = _lsdd_process_one_aggregate_gep(
                i=i,
                level=level,
                nev=nev,
                min_coarsening=min_coarsening,
                counter=counter,
                p_r=p_r,
                p_c=p_c,
                p_v=p_v,
            )

    # ----------------------------------------------
    # --- Assemble P from the local eigenvectors ---
    # ----------------------------------------------
    with stats.timeit("assemble_P"):
        counter = _lsdd_assemble_P_from_triplets(
            level=levels[-1],
            n_fine=A.shape[0],
            p_r=p_r,
            p_c=p_c,
            p_v=p_v,
            counter=counter,
        )

    # -------------------------------------------
    # --- Form coarse grid operator A = R A P ---
    # -------------------------------------------
    with stats.timeit("coarsen"):
        A, B, BT = _lsdd_coarsen_operators(B=B, BT=BT, P=levels[-1].P, R=levels[-1].R)


    stats.n_coarse = A.shape[0]
    stats.extra["mean_nev"] = float(np.mean(levels[-1].nev))
    stats.extra["min_ev"] = float(levels[-1].min_ev)

    _lsdd_append_next_level(levels=levels, A=A, B=B, BT=BT)
    _lsdd_print_level_summary(stats, print_info=print_info)


    


  
def _remove_empty_columns(A):
    """Remove empty columns from a sparse matrix."""
    if( not issparse(A)):
        raise TypeError('Argument A must be a sparse matrix')
    if A.format != 'csc':
        raise TypeError('Argument A must be a csc sparse matrix')
    m,n = A.shape
    ones = np.ones(m, dtype=A.dtype)
    s = ones @ A
    ptr = [0]
    for i in range(n):
        if s[i] > 0:
            ptr.append(A.indptr[i+1])
    A_new = csc_array((A.data, A.indices, ptr),[m,len(ptr)-1])
    return A_new

def _add_columns_containing_isolated_nodes(A):
    m,n = A.shape
    x = A @ np.ones((n,1))
    loc_isolated_nodes = np.where(x==0)[0]
    n_isolated_nodes = len(loc_isolated_nodes)
    if n_isolated_nodes == 0:
        return A
    else:
        x_r = loc_isolated_nodes
        x_c = np.array(list(range(n_isolated_nodes)))
        x_v = x_r*0+1
        x = csc_array((x_v,(x_r,x_c)),shape=(m,n_isolated_nodes))
        # x = 0*x
        # x[loc_isolated_nodes] += 1
        A = hstack([A,x])
        return A
              

def _fill_unaggregated_by_neighbors(C, AggOp,
    make_singletons=True, use_weights=True, iterate=False):
    """
    Assigns any unaggregated fine nodes (rows of AggOp with all zeros)
    to one of the aggregates of their neighbors in C.

    Parameters
    ----------
    C : csr_array (n_fine x n_fine)
        Strength/adjacency (can be symmetric or not).
    AggOp : csr_matrix (n_fine x n_coarse)
        Aggregation operator with one nonzero per assigned row.
    make_singletons : bool
        If True, create a new singleton aggregate for any still-unassigned node.
    use_weights : bool
        If True, use |C| as weights; if False, just use sparsity pattern (all ones).
    iterate : bool
        If True, repeat voting once with the new assignments (rarely needed).

    Returns
    -------
    AggOp_filled : csr_matrix (n_fine x n_coarse’)
        Updated aggregation; columns may increase if make_singletons=True.
    """
    if not issparse(C) or C.format != 'csr':
        raise TypeError('expected csr_array')

    if not issparse(AggOp) or AggOp.format != 'csr':
        raise TypeError('expected csr_array')

    n_fine, n_coarse = AggOp.shape

    # Use absolute weights or binary structure
    W = C.copy().tocsr()
    W.setdiag(0)  # ignore self-loops
    W.eliminate_zeros()
    if use_weights:
        W.data = np.abs(W.data)
    else:
        W.data[:] = 1.0

    def single_pass(A):
        nnz_row = A.indptr[1:] - A.indptr[0:-1]
        unassigned_rows = np.where(nnz_row==0)[0]
        if unassigned_rows.size == 0:
            return A, np.array([], dtype=int), np.array([], dtype=int)

        # Vote: for each fine node, how strongly to each aggregate?
        V = W @ A  # (n_fine x n_coarse)
        # Pick best aggregate per unassigned row
        new_rows, new_cols = [], []
        for i in unassigned_rows:
            start, end = V.indptr[i], V.indptr[i+1]
            if end > start:
                cols_i = V.indices[start:end]
                vals_i = V.data[start:end]
                j = cols_i[np.argmax(vals_i)]
                new_rows.append(i)
                new_cols.append(j)
        if new_rows:
            add = coo_array((np.ones(len(new_rows)), (new_rows, new_cols)), shape=A.shape)
            A = (A + add).tocsr()
        return A, unassigned_rows, np.array(new_rows, dtype=int)

    # First pass
    AggOp, all_unassigned, newly_assigned = single_pass(AggOp)

    # Optional second pass to let just-assigned nodes help their neighbors
    if iterate and all_unassigned.size > newly_assigned.size:
        AggOp, _, _ = single_pass(AggOp)

    # Handle any rows that remain unassigned
    nnz_row = AggOp.indptr[1:] - AggOp.indptr[0:-1]
    still_unassigned = np.where(nnz_row == 0)[0]
    if make_singletons and still_unassigned.size > 0:
        k = still_unassigned.size
        # Create one new column per unassigned node
        new_cols = np.arange(AggOp.shape[1], AggOp.shape[1] + k)
        add = coo_array((np.ones(k), (still_unassigned, new_cols)), shape=(AggOp.shape[0], AggOp.shape[1] + k))
        # Pad AggOp to the new width and add
        pad = coo_array(([], ([], [])), shape=(AggOp.shape[0], k))
        AggOp = csr_array(np.hstack((AggOp.toarray(), pad.toarray())))  # safe for modest sizes
        AggOp = (AggOp + add.tocsr()).tocsr()

    AggOp.eliminate_zeros()
    return AggOp
