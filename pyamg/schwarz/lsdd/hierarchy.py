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


def _lsdd_density(A) -> float:
    """Return operator density nnz / n^2 as a float."""
    n = int(A.shape[0])
    if n <= 0:
        return 0.0
    return float(A.nnz) / float(n * n)


def _lsdd_should_coarsen(*, n_levels: int, A, max_levels: int, max_coarse: int, max_density: float) -> bool:
    """Return True iff we should attempt to coarsen the current last level.

    This matches the wrapper condition:
        n_levels < max_levels and A.shape[0] > max_coarse and density(A) < max_density
    """
    if n_levels >= int(max_levels):
        return False
    if int(A.shape[0]) <= int(max_coarse):
        return False
    if _lsdd_density(A) >= float(max_density):
        return False
    return True


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
    level.R = level.P.T.tocsr()
    return counter


def _lsdd_coarsen_operators(*, A: SparseLike, B: SparseLike, P: SparseLike, R: SparseLike, stats, levels: list[MultilevelSolver.Level], cfg: LSDDConfig) -> tuple[SparseLike, SparseLike | None]:
    """Form coarse operators for the next level.

    This routine always propagates the least-squares factor:
        B_c = B @ P

    The coarse SPD operator is formed via the Galerkin triple product:
        A_c = R @ A @ P

    Parameters
    ----------
    A
        Fine-level SPD operator on this level (square), shape (n_fine, n_fine).
    B
        Fine-level least-squares factor on this level, shape (m_rows, n_fine).
    P, R
        Prolongation and restriction, shapes (n_fine, n_coarse) and (n_coarse, n_fine).
        Typically R = P^T for this real-only implementation.

    Returns
    -------
    A_c, B_c | None
        Coarse operators:
          - A_c  = R @ A @ P
          - B_c  = B @ P, if the hierarchy will be extended further, else None.

    Notes
    -----
    Using A_c = R @ A @ P is typically much faster than forming A_c as (B @ P)^T (B @ P),
    while being algebraically equivalent when A = B^T B and R = P^T. This often cuts coarsen time hard, because it avoids the “inflate B then Gram it” path.
    """

    with stats.timeit("coarsen_P_to_csc"):
        P_csc = P if getattr(P, "format", None) == "csc" else P.tocsc()

    with stats.timeit("coarsen_A_P"):
        AP = A @ P_csc

    with stats.timeit("coarsen_R_AP"):
        A_c = R @ AP

    with stats.timeit("coarsen_sort"):
        A_c.sort_indices()

    # Pass-on metadata of A. Note that the solver requires both a valid "symmetry" attribute, and the A on all level to carry the "schwarz_use_cholesky" attribute as True.
    A_c.symmetry = getattr(A, "symmetry")
    A_c.schwarz_use_cholesky = True

    # Decide whether the *new* level associated with A_c will be the final level or not.
    continue_coarsening = _lsdd_should_coarsen(
        n_levels=len(levels) + 1, # the number of levels once level for A_c level is created
        A=A_c,
        max_levels=cfg.max_levels,
        max_coarse=cfg.max_coarse,
        max_density=cfg.max_density,
    )

    # If we coarsen past A_c then we need a B_c
    if continue_coarsening:
        with stats.timeit("coarsen_B_P"):
            B_c = B @ P_csc

        with stats.timeit("coarsen_sort"):
            B_c.sort_indices()
    else:
        B_c = None
        
    return A_c, B_c


def _lsdd_append_next_level(
    *,
    levels: list[MultilevelSolver.Level],
    A: SparseLike,
    B: SparseLike | None,
) -> MultilevelSolver.Level:
    """Append a new multigrid level and store A/B/BT and density metadata.

    Parameters
    ----------
    levels
        List of MultilevelSolver levels. Mutated by appending one new Level().

    A, B
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
    nxt.density = nxt.A.nnz / (nxt.A.shape[0] ** 2)
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
    Required attributes on the current level include `A`, `B`
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

    stats = LsddLevelStats(level=len(levels) - 1, n_fine=A.shape[0])

    # ---- optional filtering (not used on the finest level) ----
    if len(levels) > 1:
        with stats.timeit("filter"):
            fdiag = _lsdd_filter_ops_inplace(
                A=A,
                B=B,
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
        _lsdd_build_overlap_and_pou(level=level, A=A, B=B, v_row_mult=v_row_mult, print_info=cfg.print_info)


    # ---- dense principal submatrices ----
    with stats.timeit("extract_PCM"):
        _lsdd_extract_local_principal_submatrices(level=level, A=A)


    # ---- local splitting blocks + threshold init ----
    with stats.timeit("outerprod"):
        outerprod_timers = {}
        p_r, p_c, p_v, counter = _lsdd_local_outer_products_and_gep_init(
            level=level,
            B=B,
            v_row_mult=v_row_mult,
            kappa=cfg.kappa,
            threshold=cfg.threshold,
            timers=outerprod_timers,
            stats=stats
        )

    # Record sub-timers for printing (do not include these in the overall "total" sum in stats.py)
    for k, dt in outerprod_timers.items():
        stats.timings[k] = dt

    # ---- per-aggregate dense GEP ----
    eigvals_kept: list[float] = []
    gep_timers: dict[str, float] = {}  # accumulates sub-timers across all aggregates on this level
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
                gep_timers=gep_timers,
            )

    # Record sub-timers for printing (do not include these in the overall "total" sum in stats.py)
    for k, dt in gep_timers.items():
        stats.timings[k] = dt

    # ---- optional exploratory theory hooks (development / diagnostics) ----
    if cfg.explore_theory:
        from .eigs import _lsdd_theory_outerprod_weighting_sweep

        # if cfg.explore_theory_aggs is None:
        #     # Small deterministic sample: first, middle, last.
        #     mid = level.n_aggs // 2
        #     agg_ids = tuple(sorted(set([0, mid, max(0, level.n_aggs - 1)])))
        # else:
        #     agg_ids = cfg.explore_theory_aggs

        # # Store results on the level for post-hoc interactive inspection.
        # level.theory_outerprod_weighting = _lsdd_theory_outerprod_weighting_sweep(
        #     level=level,
        #     agg_ids=agg_ids,
        # )

        # print(f"Completed theory hooks for aggregates {agg_ids} on level {len(levels) - 1}.")
        # print(f"level.theory_outerprod_weighting = {level.theory_outerprod_weighting}")


        from pyamg.schwarz.lsdd.eigs import _lsdd_theory_explore_gep_spectra
        # e.g. explore a few aggregates on level 0
        res = _lsdd_theory_explore_gep_spectra(
            level=level,
            #agg_ids=[0, 1, 2],
            #agg_ids = np.arange(0, level.n_aggs),
            agg_ids = [0, level.n_aggs // 2, max(0, level.n_aggs - 1)],
            plot=True,          # blocking per-aggregate
        )

        # import pdb
        # pdb.set_trace()


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
    with stats.timeit("coarsen"):
        A_c, B_c = _lsdd_coarsen_operators(A = A, B=B, P=level.P, R=level.R, stats=stats, levels=levels, cfg=cfg)


    _lsdd_finalize_level_stats(stats=stats, level=level, eigvals_kept=eigvals_kept, n_coarse=A_c.shape[0])
    _lsdd_print_level_summary(stats, print_info=cfg.print_info)

    _lsdd_append_next_level(levels=levels, A=A_c, B=B_c)
