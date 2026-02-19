"""Strength-of-connection and aggregation utilities for LS–AMG–DD.

This module provides the setup pieces that produce an aggregation operator
    AggOp : R^{n_fine x n_aggs}
with one nonzero per fine row, defining nonoverlapping aggregates (omega_i).

Main responsibilities
---------------------
1) Optional operator filtering:
   Applies row filtering to A/B/BT using `pyamg.util.utils.filter_matrix_rows`.

2) Strength-of-connection:
   Constructs a sparse adjacency/strength matrix C from a method spec, using
   standard PyAMG strength operators.

3) Aggregation:
   Builds AggOp from the requested aggregation strategy (standard, lloyd, metis, ...),
   optionally performing multiple aggregation passes (agg_levels).

4) Cleanup of unaggregated nodes:
   If an aggregation leaves some rows unassigned (a row in AggOp with no nonzeros),
   assign those nodes to neighboring aggregates via a voting step on an adjacency
   matrix, and (optionally) create singleton aggregates for any remaining.

Level initialization
--------------------
After AggOp is constructed, `_lsdd_init_level_after_aggregation` stores AggOp and
allocates per-aggregate storage on the level (Subdomains/LocalBlocks/EigenInfo).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    # SciPy sparse arrays (preferred)
    from scipy.sparse import csr_array, coo_array, hstack, issparse  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for older SciPy that only has sparse matrices
    from scipy.sparse import csr_matrix as csr_array  # type: ignore
    from scipy.sparse import coo_matrix as coo_array  # type: ignore
    from scipy.sparse import hstack, issparse  # type: ignore

from pyamg.util.utils import filter_matrix_rows
from pyamg.strength import (
    classical_strength_of_connection,
    symmetric_strength_of_connection,
    evolution_strength_of_connection,
    energy_based_strength_of_connection,
    distance_strength_of_connection,
    algebraic_distance,
    affinity_distance,
)
from pyamg.aggregation.aggregate import (
    standard_aggregation,
    naive_aggregation,
    lloyd_aggregation,
    balanced_lloyd_aggregation,
    metis_aggregation,
    pairwise_aggregation,
)

from .types import Subdomains, LocalBlocks, EigenInfo


def _lsdd_unpack_arg(v: Any) -> tuple[Any, dict[str, Any]]:
    """Normalize a PyAMG-style method spec into (name, kwargs).

    Parameters
    ----------
    v
        Either:
          - a string name like "standard", "symmetric", ...
          - a pair (name, kwargs) like ("standard", {"theta": 0.0})
          - None

    Returns
    -------
    name, kwargs
        `name` is the method identifier, `kwargs` is a dict of keyword arguments.
    """
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


def _fill_unaggregated_by_neighbors(
    Adj,
    AggOp,
    *,
    make_singletons: bool = True,
    use_weights: bool = True,
    iterate: bool = False,
):
    """Assign unaggregated fine nodes to neighbor aggregates.

    A "fine node" is unaggregated if its row in AggOp has no nonzeros.

    The procedure is:
      1) Form a nonnegative adjacency/weight matrix W from Adj.
      2) Compute V = W @ AggOp, which gives, for each fine node, a weighted vote
         for each aggregate based on its neighbors' aggregate assignments.
      3) For each unassigned row i, assign it to the aggregate with the largest vote.
      4) Optionally repeat one pass (`iterate=True`) to let newly assigned nodes help.
      5) If requested, create singleton aggregates for any remaining unassigned nodes.

    Parameters
    ----------
    Adj
        CSR sparse matrix/array of shape (n_fine, n_fine) used as a neighbor graph.
        In the current pipeline this is typically passed as `A` (not the strength C),
        to preserve existing behavior.

    AggOp
        CSR sparse matrix/array of shape (n_fine, n_aggs) with one nonzero per
        assigned row.

    make_singletons
        If True, create a new singleton aggregate for any still-unassigned row.

    use_weights
        If True, weight votes by abs(Adj) values. If False, treat the graph as
        unweighted (all ones on nonzeros).

    iterate
        If True, perform a second voting pass after the first assignments.

    Returns
    -------
    AggOp_filled
        CSR sparse aggregation operator with all rows assigned. The number of columns
        may increase if singleton aggregates are created.
    """
    if (not issparse(Adj)) or getattr(Adj, "format", None) != "csr":
        raise TypeError("Adj must be CSR sparse")

    if (not issparse(AggOp)) or getattr(AggOp, "format", None) != "csr":
        raise TypeError("AggOp must be CSR sparse")

    n_fine, n_aggs = AggOp.shape

    # Build a nonnegative adjacency/weight matrix
    W = Adj.copy().tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    if use_weights:
        W.data = np.abs(W.data)
    else:
        W.data[:] = 1.0

    def _single_pass(A):
        nnz_row = A.indptr[1:] - A.indptr[:-1]
        unassigned = np.flatnonzero(nnz_row == 0)
        if unassigned.size == 0:
            return A, unassigned, np.array([], dtype=np.int32)

        V = W @ A  # weighted votes per aggregate

        new_rows: list[int] = []
        new_cols: list[int] = []
        for i in unassigned:
            s, e = V.indptr[i], V.indptr[i + 1]
            if e <= s:
                continue
            cols_i = V.indices[s:e]
            vals_i = V.data[s:e]
            j = int(cols_i[int(np.argmax(vals_i))])
            new_rows.append(int(i))
            new_cols.append(j)

        if new_rows:
            add = coo_array(
                (np.ones(len(new_rows)), (np.asarray(new_rows), np.asarray(new_cols))),
                shape=A.shape,
            )
            A = (A + add).tocsr()

        return A, unassigned, np.asarray(new_rows, dtype=np.int32)

    # First pass
    AggOp, all_unassigned, newly_assigned = _single_pass(AggOp)

    # Optional second pass
    if iterate and all_unassigned.size > newly_assigned.size:
        AggOp, _, _ = _single_pass(AggOp)

    # Any still unassigned?
    nnz_row = AggOp.indptr[1:] - AggOp.indptr[:-1]
    still_unassigned = np.flatnonzero(nnz_row == 0)

    if make_singletons and still_unassigned.size > 0:
        k = int(still_unassigned.size)

        # Pad AggOp with k empty columns (sparse) and then add one 1 per remaining row.
        AggOp = hstack(
            [AggOp, csr_array((n_fine, k), dtype=AggOp.dtype)],
            format="csr",
        )

        new_cols = np.arange(n_aggs, n_aggs + k, dtype=np.int32)
        add = coo_array(
            (np.ones(k, dtype=AggOp.dtype), (still_unassigned.astype(np.int32), new_cols)),
            shape=AggOp.shape,
        )
        AggOp = (AggOp + add).tocsr()

    AggOp.eliminate_zeros()
    return AggOp

def _lsdd_filter_ops_inplace(
    *,
    A,
    B,
    BT,
    filteringA: tuple[bool, float] | None,
    filteringB: tuple[bool, float] | None,
) -> dict[str, int]:

    """Optionally filter A/B/BT in-place.

    Parameters
    ----------
    A
        SPD operator on the current level (CSR-like).
    B, BT
        Least-squares factors on the current level (CSR-like).
    filteringA, filteringB
        Either None (disable) or a 2-tuple `(lump_diagonal, theta)` passed to
        `filter_matrix_rows`:
          - lump_diagonal : bool
          - theta         : float or int controlling dropping

    Returns
    -------
    diag
        Dictionary of nnz counts before and after filtering, with keys 
            if filteringA: "A_nnz_before", "A_nnz_after"  
            if filteringB: "B_nnz_before", "B_nnz_after", "BT_nnz_before", "BT_nnz_after".
    """
    diag: dict[str, int] = {}

    if filteringB is not None and filteringB[1] != 0:
        diag["B_nnz_before"] = int(len(B.data))
        diag["BT_nnz_before"] = int(len(BT.data))

        filter_matrix_rows(B, filteringB[1], diagonal=True, lump=filteringB[0])
        filter_matrix_rows(BT, filteringB[1], diagonal=True, lump=filteringB[0])

        diag["B_nnz_after"] = int(len(B.data))
        diag["BT_nnz_after"] = int(len(BT.data))

    if filteringA is not None and filteringA[1] != 0:
        diag["A_nnz_before"] = int(len(A.data))

        filter_matrix_rows(A, filteringA[1], diagonal=True, lump=filteringA[0])

        diag["A_nnz_after"] = int(len(A.data))

    return diag



def _lsdd_build_strength(*, A, B, strength_spec: Any):
    """Compute strength-of-connection matrix C from a strength spec.

    Parameters
    ----------
    A
        SPD operator on this level (CSR-like).
    B
        Least-squares factor on this level. Some strength methods may use B.
    strength_spec
        PyAMG-style spec:
          - a string method name, or
          - (name, kwargs) pair.

    Returns
    -------
    C
        CSR strength-of-connection / adjacency matrix.
    """
    name, kwargs = _lsdd_unpack_arg(strength_spec)

    if name == "symmetric":
        C = symmetric_strength_of_connection(A, **kwargs)
    elif name == "classical":
        C = classical_strength_of_connection(A, **kwargs)
    elif name == "distance":
        C = distance_strength_of_connection(A, **kwargs)
    elif name in ("ode", "evolution"):
        # Some callers pass B via kwargs; otherwise we pass B explicitly.
        if "B" in kwargs:
            C = evolution_strength_of_connection(A, **kwargs)
        else:
            C = evolution_strength_of_connection(A, B, **kwargs)
    elif name == "energy_based":
        C = energy_based_strength_of_connection(A, **kwargs)
    elif name == "predefined":
        C = kwargs["C"].tocsr()
    elif name == "algebraic_distance":
        C = algebraic_distance(A, **kwargs)
    elif name == "affinity":
        C = affinity_distance(A, **kwargs)
    elif name is None:
        # Default: absolute adjacency of A
        C = abs(A.copy()).tocsr()
    else:
        raise ValueError(f"Unrecognized strength-of-connection method: {name!r}")

    C = C.tocsr()
    C.eliminate_zeros()
    return C


def _lsdd_build_aggop(
    *,
    A,
    C,
    aggregate_spec: Any,
    agg_levels: int,
    is_finest: bool,
):
    """Build aggregation operator AggOp from an aggregation spec.

    Parameters
    ----------
    A
        SPD operator on this level. Some aggregation methods use A.
    C
        Strength-of-connection / adjacency matrix (CSR).
    aggregate_spec
        PyAMG-style aggregation spec (string or (string, kwargs)).
        Supported names include:
          - "standard", "naive", "lloyd", "balanced lloyd", "metis", "pairwise",
            "d2C", "d3C", "predefined".
    agg_levels
        Number of aggregation passes to apply. For agg_levels > 1, the strength
        matrix is coarsened between passes via AggOp.T @ C @ AggOp.
    is_finest
        True if constructing aggregates on the finest level. Kept as a hook for
        future policy differences.

    Returns
    -------
    AggOp, nc_temp
        AggOp is CSR with shape (n_fine, n_aggs_final).
        nc_temp is the number of aggregates before singleton padding.
    """
    name, kwargs = _lsdd_unpack_arg(aggregate_spec)

    C = C.tocsr()
    C.eliminate_zeros()

    Aggs = []
    for _ in range(int(agg_levels)):
        if name == "standard":
            AggOp, _ = standard_aggregation(C, **kwargs)
        elif name == "d2C":
            C = (C @ C).tocsr()
            AggOp, _ = standard_aggregation(C, **kwargs)
        elif name == "d3C":
            C = (C @ C @ C).tocsr()
            AggOp, _ = standard_aggregation(C, **kwargs)
        elif name == "naive":
            AggOp, _ = naive_aggregation(C, **kwargs)
        elif name == "lloyd":
            AggOp, _ = lloyd_aggregation(C, **kwargs)
        elif name == "balanced lloyd":
            # Some variants want A in kwargs
            if "pad" in kwargs:
                kwargs["A"] = A
            AggOp, _ = balanced_lloyd_aggregation(C, **kwargs)
        elif name == "metis":
            # Metis expects an unweighted graph
            C2 = C.copy().tocsr()
            C2.data[:] = 1.0
            AggOp = metis_aggregation(C2, **kwargs)
        elif name == "pairwise":
            AggOp = pairwise_aggregation(A, **kwargs)[0]
        elif name == "predefined":
            AggOp = kwargs["AggOp"].tocsr()
        else:
            raise ValueError(f"Unrecognized aggregation method: {name!r}")

        Aggs.append(AggOp.tocsr())
        if len(Aggs) < agg_levels:
            C = (AggOp.T @ C @ AggOp).tocsr()
            C.eliminate_zeros()

    # Multiply aggregation passes together
    AggOp = Aggs[0]
    for j in range(1, len(Aggs)):
        AggOp = (AggOp @ Aggs[j]).tocsr()

    nc_temp = int(AggOp.shape[1])

    # Assign any missing rows using adjacency voting and singleton padding if needed.
    # Note: keep existing behavior by using A (not C) as the adjacency graph.
    AggOp = _fill_unaggregated_by_neighbors(A, AggOp, make_singletons=True)

    return AggOp.tocsr(), nc_temp

def _lsdd_init_level_after_aggregation(*, level, AggOp, A, B) -> np.ndarray:
    """Initialize per-level storage after aggregation is formed.

    Parameters
    ----------
    level
        Multigrid level object (mutated in-place).
    AggOp
        CSR aggregation operator of shape (n_fine, N).
    A
        SPD operator on this level (only used for sizing).
    B
        Least-squares factor on this level (only used for sizing).

    Side effects
    ------------
    Sets:
      - level.AggOp, level.AggOpT, level.N
      - level.sub   : Subdomains container (omega/OMEGA/GAMMA/PoU/rows + size arrays)
      - level.blocks: LocalBlocks container (flattened dense block storage)
      - level.eigs  : EigenInfo container (nev per agg, threshold, min_ev)

    Returns
    -------
    v_row_mult
        Array of shape (m,), initialized to zeros. This is filled during overlap
        construction and passed to the local outer-product kernel.
    """
    level.AggOp = AggOp
    level.AggOpT = AggOp.T.tocsr()
    level.N = int(AggOp.shape[1])

    level.sub = Subdomains.allocate(level.N)
    level.blocks = LocalBlocks()
    level.eigs = EigenInfo.allocate(level.N)

    # v_row_mult is filled during overlap construction and then reused in outer products
    v_row_mult = np.zeros(B.shape[0], dtype=float)
    return v_row_mult

