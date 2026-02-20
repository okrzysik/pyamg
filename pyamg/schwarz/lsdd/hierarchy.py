"""Hierarchy extension utilities for LS–AMG–DD.

This module provides:
  - assembly of global prolongation P (and restriction R),
  - least-squares propagation of operators to the next level,
  - appending the next multigrid level,
  - the orchestration routine that builds one additional level.

The public entrypoint used by `least_squares_dd_exp.py` is `_lsdd_extend_hierarchy`.
"""

from __future__ import annotations

from typing import cast
from .types import LSDDLevel
from .types import LSDDConfig



import numpy as np

try:
    from scipy.sparse import csr_array  # type: ignore
except Exception:  # pragma: no cover
    from scipy.sparse import csr_matrix as csr_array  # type: ignore


from .types import SparseLike
from pyamg.multilevel import MultilevelSolver


def _lsdd_assemble_P_from_triplets(
    *,
    level: LSDDLevel,
    n_fine: int,
    p_r: list,
    p_c: list,
    p_v: list,
    counter: int,
) -> int:
    """Assemble prolongation P (and restriction R = P^H) from accumulated triplets.

    Parameters
    ----------
    level
        Current multigrid level object. This routine sets:
        - `level.P` as a CSR sparse array/matrix of shape (n_fine, n_coarse)
        - `level.R` as the conjugate-transpose of P in CSR form

    n_fine
        Fine dimension on this level, i.e. `A.shape[0]`.

    p_r, p_c, p_v
        Triplet lists accumulated during the per-aggregate eigenproblem loop:

        - `p_r[k]` is an int32 array of global row indices for the k-th inserted vector
        - `p_c[k]` is an int32 array of the same length filled with the coarse column id
        - `p_v[k]` is a float/complex array of the same length containing values

        Each k corresponds to one prolongation column supported on omega_i.

    counter
        Current number of coarse columns accumulated so far. This is typically the
        running counter used during the eigenproblem loop.

    Returns
    -------
    n_coarse
        The final number of coarse columns (equals `counter` after assembly).

    Notes
    -----
    If no vectors were selected (empty triplet lists), this routine falls back to
    a 1-dimensional coarse space supported on row 0, to ensure the hierarchy remains valid.
    """
    if len(p_r) == 0:
        p_r = [np.array([0], dtype=np.int32)]
        p_c = [np.array([0], dtype=np.int32)]
        p_v = [np.array([1.0])]
        counter = 1

    rows = np.concatenate(p_r).astype(np.int32, copy=False)
    cols = np.concatenate(p_c).astype(np.int32, copy=False)
    vals = np.concatenate(p_v)

    level.P = csr_array((vals, (rows, cols)), shape=(n_fine, counter))
    level.R = level.P.T.conjugate().tocsr()
    return counter


def _lsdd_coarsen_operators(*, B: SparseLike, BT: SparseLike, P: SparseLike, R: SparseLike) -> tuple[SparseLike, SparseLike, SparseLike]:
    """Form coarse operators via least-squares propagation.

    Parameters
    ----------
    B, BT
        Fine-level least-squares factors on this level, with A = BT @ B.
        Shapes: B is (m x n), BT is (n x m).

    P, R
        Prolongation and restriction on this level. Shapes: P is (n x n_c),
        R is (n_c x n). Typically R = P^H.

    Returns
    -------
    A_c, B_c, BT_c
        Coarse operators:
          - B_c  = B @ P
          - BT_c = B_c.T (CSR)  [chosen for performance; consistent for real-valued problems]
          - A_c  = BT_c @ B_c

    Notes
    -----
    The implementation currently constructs BT_c by transposing B_c rather than
    explicitly computing R @ BT. This matches the existing performance-oriented
    approach used in the experimental branch.
    """
    B_c = B @ P
    BT_c = B_c.T.tocsr()
    A_c = BT_c @ B_c
    A_c.sort_indices()
    return A_c, B_c, BT_c


def _lsdd_append_next_level(*, levels: list[MultilevelSolver.Level], A: SparseLike, B: SparseLike, BT: SparseLike) -> MultilevelSolver.Level:
    """Append a new multigrid level and store A/B/BT and density metadata.

    Parameters
    ----------
    levels
        List of MultilevelSolver levels. Mutated by appending one new Level().

    A, B, BT
        Coarse-level operators to store on the newly appended level.

    Returns
    -------
    next_level
        The newly created and appended `MultilevelSolver.Level` instance.
    """
    levels.append(MultilevelSolver.Level())
    nxt = levels[-1]
    nxt.A = A
    nxt.B = B
    nxt.BT = BT
    nxt.density = len(nxt.A.data) / (nxt.A.shape[0] ** 2)
    return nxt


def _lsdd_extend_hierarchy(
    *,
    levels: list[MultilevelSolver.Level],
    strength_spec: object,
    aggregate_spec: object,
    cfg: LSDDConfig,
) -> None:
    """Extend an LS–AMG–DD hierarchy by one level.

    This function operates on the current finest level (the last element of `levels`),
    constructs the prolongation `P`, forms the next-level operators, and appends the
    new level to `levels`.

    Parameters
    ----------
    levels : list[pyamg.multilevel.MultilevelSolver.Level]
        Existing hierarchy levels. The last entry is treated as the current level and is
        populated with per-level containers and operators; a new level is appended.
    strength_spec : object
        Strength-of-connection specification for the current level. This is the already
        level-selected entry (e.g., `strength[lvl]`) and is interpreted by the strength
        construction routine.
    aggregate_spec : object
        Aggregation specification for the current level. This is the already level-selected
        entry (e.g., `aggregate[lvl]`) and is interpreted by the aggregation routine.
    cfg : LSDDConfig
        Configuration bundle for the extension step (aggregation passes, filtering specs,
        eigen selection controls, and diagnostics flag).

    Notes
    -----
    Required attributes on the current level include `A`, `B`, `BT`.
    This routine sets/updates `sub`, `blocks`, `eigs`, `P`, `R` on the current level and
    appends the next level with coarsened operators.
    """
    from .aggregation import (
        _lsdd_build_aggop,
        _lsdd_build_strength,
        _lsdd_filter_ops_inplace,
        _lsdd_init_level_after_aggregation,
    )
    from .eigs import _lsdd_process_one_aggregate_gep
    from .local_ops import (
        _lsdd_extract_local_principal_submatrices,
        _lsdd_local_outer_products_and_gep_init,
    )
    from .stats import (
        LsddLevelStats,
        _lsdd_finalize_level_stats,
        _lsdd_print_level_summary,
    )
    from .subdomains import _lsdd_build_overlap_and_pou

    level = cast(LSDDLevel, levels[-1])
    A = level.A
    B = level.B
    BT = level.BT

    stats = LsddLevelStats(level=len(levels) - 1, n_fine=A.shape[0])

    # ---- optional filtering (not used on the finest level) ----
    if len(levels) > 1:
        with stats.timeit("filter"):
            fdiag = _lsdd_filter_ops_inplace(
                A=A,
                B=B,
                BT=BT,
                filteringA=cfg.filteringA,
                filteringB=cfg.filteringB,
            )

        # Store under stable keys for stats printing
        for k, v in fdiag.items():
            stats.extra[f"filter_{k}"] = v


    # ---- strength-of-connection ----
    with stats.timeit("strength"):
        C = _lsdd_build_strength(A=A, B=B, strength_spec=strength_spec)

    # ---- aggregation ----
    with stats.timeit("aggregate"):
        AggOp, _nc_temp = _lsdd_build_aggop(
            A=A,
            C=C,
            aggregate_spec=aggregate_spec,
            agg_levels=cfg.agg_levels,
            is_finest=(len(levels) == 1),
        )
        v_row_mult = _lsdd_init_level_after_aggregation(level=level, AggOp=AggOp, A=A, B=B)


    # ---- overlap construction + PoU ----
    with stats.timeit("overlap"):
        _lsdd_build_overlap_and_pou(
            level=level,
            A=A,
            BT=BT,
            v_row_mult=v_row_mult,
            print_info=cfg.print_info,
        )


    # ---- dense principal submatrices ----
    with stats.timeit("extract_PCM"):
        _lsdd_extract_local_principal_submatrices(level=level, A=A)


    # ---- local splitting blocks + threshold init ----
    with stats.timeit("outerprod"):
        p_r, p_c, p_v, counter = _lsdd_local_outer_products_and_gep_init(
            level=level,
            B=B,
            BT=BT,
            v_row_mult=v_row_mult,
            kappa=cfg.kappa,
            threshold=cfg.threshold,
        )

    # ---- per-aggregate dense GEP ----
    eigvals_kept: list[float] = []
    with stats.timeit("gep"):
        for i in range(level.n_aggs):
            counter = _lsdd_process_one_aggregate_gep(
                i=i,
                level=level,
                nev=cfg.nev,
                min_coarsening=cfg.min_coarsening,
                counter=counter,
                p_r=p_r,
                p_c=p_c,
                p_v=p_v,
                eigvals_kept=eigvals_kept,
            )

    # ---- assemble P ----
    with stats.timeit("assemble_P"):
        _ = _lsdd_assemble_P_from_triplets(
            level=level,
            n_fine=A.shape[0],
            p_r=p_r,
            p_c=p_c,
            p_v=p_v,
            counter=counter,
        )

    # ---- coarsen operators ----
    fine_sym = getattr(level.A, "symmetry", None)
    fine_is_spd = getattr(level.A, "is_spd", None)

    with stats.timeit("coarsen"):
        A_c, B_c, BT_c = _lsdd_coarsen_operators(B=B, BT=BT, P=level.P, R=level.R)

        if fine_sym is not None:
            A_c.symmetry = fine_sym
        if fine_is_spd is not None:
            A_c.is_spd = fine_is_spd

    _lsdd_finalize_level_stats(stats=stats, level=level, eigvals_kept=eigvals_kept, n_coarse=A_c.shape[0])
    _lsdd_print_level_summary(stats, print_info=cfg.print_info)

    _lsdd_append_next_level(levels=levels, A=A_c, B=B_c, BT=BT_c)
