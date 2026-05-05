"""Setup helpers for ``lsdd.two_lvl_theory``."""

from __future__ import annotations

from typing import Any
from warnings import warn

from scipy.sparse import SparseEfficiencyWarning, csr_array, issparse

from pyamg.multilevel import MultilevelSolver
from pyamg.util.utils import asfptype, levelize_strength_or_aggregation, levelize_weight

from ..hierarchy import _lsdd_extend_hierarchy
from ..types import LSDDConfig, FilteringSpec, SparseLike


def as_csr_fptype(M: SparseLike, *, name: str):
    """Convert sparse-like input into sorted CSR with floating/complex dtype."""
    if not issparse(M) or M.format not in ("csr",):
        try:
            M = csr_array(M)
            warn(f"Implicit conversion of {name} to CSR", SparseEfficiencyWarning)
        except Exception as exc:  # pragma: no cover
            raise TypeError(
                f"Argument {name} must have type csr_array/bsr_array or be convertible to csr_array"
            ) from exc
    M = asfptype(M)
    M = M.tocsr()
    M.eliminate_zeros()
    M.sort_indices()
    return M


def build_one_level_lsdd_context(
    *,
    B: SparseLike,
    A: SparseLike | None,
    BT: SparseLike | None,
    symmetry: str,
    strength: Any,
    aggregate: Any,
    agg_levels: int,
    kappa: float | list[float] | None,
    nev: int | None,
    threshold: float | None,
    mult_threshold: float | list[float | None] | None,
    min_coarsening: int | list[int] | None,
    filteringA: FilteringSpec | None,
    filteringB: FilteringSpec | None,
    print_info: bool,
    force_row_closure: bool,
    robust_Sker_handling: bool,
    max_levels: int = 2,
    max_coarse: int = 10,
    max_density: float = 1.0,
) -> MultilevelSolver.Level:
    """Build exactly one LS-DD setup level and return the fine level object."""
    if symmetry not in ("symmetric", "hermitian"):
        raise ValueError('Expected "symmetric" or "hermitian" for the symmetry parameter')
    if threshold is not None and mult_threshold is not None:
        raise ValueError("threshold and mult_threshold are mutually exclusive; set at most one")

    B_csr = as_csr_fptype(B, name="B")
    if BT is None:
        BT_csr = B_csr.T.conjugate().tocsr()
        BT_csr.sort_indices()
    else:
        BT_csr = as_csr_fptype(BT, name="BT")

    if A is None:
        A_csr = BT_csr @ B_csr
    else:
        A_csr = as_csr_fptype(A.copy(), name="A")

    A_csr.symmetry = symmetry
    A_csr.schwarz_use_cholesky = True
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError("expected square matrix")

    _, _, strength_lvls = levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    _, _, aggregate_lvls = levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    if kappa is None:
        kappa0 = 0.0
    else:
        kappa_lvls = levelize_weight(kappa, max_levels)
        kappa0 = float(kappa_lvls[0])
    mult_threshold_lvls = levelize_weight(mult_threshold, max_levels)
    min_coarsening_lvls = levelize_weight(min_coarsening, max_levels)

    if filteringA is not None:
        filteringA = (bool(filteringA[0]), float(filteringA[1]))
    if filteringB is not None:
        filteringB = (bool(filteringB[0]), float(filteringB[1]))

    levels = [MultilevelSolver.Level()]
    levels[0].A = A_csr
    levels[0].B = B_csr
    levels[0].BT = BT_csr
    levels[0].density = len(A_csr.data) / (A_csr.shape[0] ** 2)

    threshold_cfg = 0.0 if threshold is None else float(threshold)
    cfg = LSDDConfig(
        agg_levels=agg_levels,
        kappa=kappa0,
        nev=nev,
        threshold=threshold_cfg,
        mult_threshold=mult_threshold_lvls[0],
        min_coarsening=min_coarsening_lvls[0],
        filteringA=filteringA,
        filteringB=filteringB,
        print_info=print_info,
        max_levels=max_levels,
        max_coarse=max_coarse,
        max_density=float(max_density),
        force_row_closure=force_row_closure,
        robust_Sker_handling=robust_Sker_handling,
        explore_theory=False,
        explore_theory_aggs=None,
    )

    _lsdd_extend_hierarchy(
        levels=levels,
        strength_spec=strength_lvls[0],
        aggregate_spec=aggregate_lvls[0],
        cfg=cfg,
    )
    return levels[0]
