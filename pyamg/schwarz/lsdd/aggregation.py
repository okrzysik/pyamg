
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array, issparse, \
    coo_array

from pyamg.util.utils import filter_matrix_rows
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection, \
    energy_based_strength_of_connection, distance_strength_of_connection, \
    algebraic_distance, affinity_distance
from pyamg.aggregation.aggregate import standard_aggregation, naive_aggregation, \
    lloyd_aggregation, balanced_lloyd_aggregation, \
    metis_aggregation, pairwise_aggregation


def _lsdd_unpack_arg(v):
    """Return (fn, kwargs) for PyAMG-style args: either 'name' or ('name', {...})."""
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


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



def _lsdd_filter_ops_inplace(*, A, B, BT, filteringA, filteringB, print_info: bool) -> None:
    """Optionally filter A/B/BT in-place (refactor-only extraction)."""
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
            print("A NNZ after filtering", len(A.data))


def _lsdd_build_strength(*, A, B, strength_spec):
    """Compute strength-of-connection matrix C from strength spec (refactor-only)."""
    fn, kwargs = _lsdd_unpack_arg(strength_spec)

    if fn == "symmetric":
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == "classical":
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == "distance":
        C = distance_strength_of_connection(A, **kwargs)
    elif fn in ("ode", "evolution"):
        if "B" in kwargs:
            C = evolution_strength_of_connection(A, **kwargs)
        else:
            C = evolution_strength_of_connection(A, B, **kwargs)
    elif fn == "energy_based":
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == "predefined":
        C = kwargs["C"].tocsr()
    elif fn == "algebraic_distance":
        C = algebraic_distance(A, **kwargs)
    elif fn == "affinity":
        C = affinity_distance(A, **kwargs)
    elif fn is None:
        # Hussam's original implementation
        C = abs(A.copy()).tocsr()
    else:
        raise ValueError(f"Unrecognized strength of connection method: {fn!s}")

    return C


def _lsdd_build_aggop(*, A, C, aggregate_spec, agg_levels: int, is_finest: bool):
    """Build aggregation operator AggOp (refactor-only extraction)."""
    fn, kwargs = _lsdd_unpack_arg(aggregate_spec)

    C = C.tocsr()
    C.eliminate_zeros()

    Aggs = []
    for _ in range(agg_levels):
        if fn == "standard":
            AggOp, _Cnodes = standard_aggregation(C, **kwargs)
        elif fn == "d2C":
            C = C @ C
            AggOp, _Cnodes = standard_aggregation(C, **kwargs)
        elif fn == "d3C":
            C = C @ C @ C
            AggOp, _Cnodes = standard_aggregation(C, **kwargs)
        elif fn == "naive":
            AggOp, _Cnodes = naive_aggregation(C, **kwargs)
        elif fn == "lloyd":
            AggOp, _Cnodes = lloyd_aggregation(C, **kwargs)
        elif fn == "balanced lloyd":
            if "pad" in kwargs:
                kwargs["A"] = A
            AggOp, _Cnodes = balanced_lloyd_aggregation(C, **kwargs)
        elif fn == "metis":
            C.data[:] = 1.0
            # currently same call either way; keep is_finest hook for future tweaks
            AggOp = metis_aggregation(C, **kwargs)
        elif fn == "pairwise":
            AggOp = pairwise_aggregation(A, **kwargs)[0]
        elif fn == "predefined":
            AggOp = kwargs["AggOp"].tocsr()
        else:
            raise ValueError(f"Unrecognized aggregation method {fn!s}")

        Aggs.append(AggOp)
        if len(Aggs) < agg_levels:
            C = (AggOp.T @ C @ AggOp).tocsr()

    # Product of agg_levels (matches your current logic)
    AggOp = Aggs[0]
    for j in range(1, agg_levels):
        AggOp = AggOp @ Aggs[j]

    nc_temp = AggOp.shape[1]
    print("\tNum aggregates = {:.4g}".format(nc_temp))
    print("\tAv. aggregate size = {:.4g}".format(AggOp.shape[0] / AggOp.shape[1]))

    AggOp = _fill_unaggregated_by_neighbors(A, AggOp, make_singletons=True)

    print("\tNum singletons = {:.4g}".format(AggOp.shape[1] - nc_temp))
    return AggOp, nc_temp


def _lsdd_init_level_after_aggregation(*, level, AggOp, A, B):
    """Populate level aggregation fields + allocate per-aggregate storage (refactor-only)."""
    level.AggOp = AggOp
    level.AggOpT = AggOp.T.tocsr()
    level.N = AggOp.shape[1]

    level.nonoverlapping_subdomain = [None] * level.N
    level.overlapping_subdomain = [None] * level.N
    level.PoU = [None] * level.N
    level.overlapping_rows = [None] * level.N

    level.nIi = np.zeros(level.N, dtype=np.int32)
    level.ni = np.zeros(level.N, dtype=np.int32)
    level.nev = np.zeros(level.N, dtype=np.int32)

    v_mult = np.zeros(AggOp.shape[0])
    blocksize = np.zeros(level.N, dtype=np.int32)
    v_row_mult = np.zeros(B.shape[0])

    return v_mult, blocksize, v_row_mult
