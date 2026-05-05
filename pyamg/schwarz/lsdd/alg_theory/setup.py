"""Hierarchy/setup helpers for LS-DD algebraic-theory drivers."""

from __future__ import annotations

from warnings import warn
from typing import Any

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_array, issparse

from pyamg.multilevel import MultilevelSolver
from pyamg.util.utils import asfptype, levelize_strength_or_aggregation, levelize_weight

from ..hierarchy import _lsdd_extend_hierarchy
from ..types import FilteringSpec, LSDDConfig, SparseLike
from .linalg import factor_dense_spd
from .models import LocalProjectionBlock


def as_csr_fptype(M: SparseLike, *, name: str):
    """Convert input sparse-like matrix to sorted CSR floating/complex type."""
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


def build_one_level_lsdd(
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
    return_levels: bool = False,
):
    """Build exactly one LS-DD setup level."""
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

    if return_levels:
        return levels
    return levels[0]


def prepare_local_projection_blocks(level) -> tuple[list[LocalProjectionBlock], np.ndarray]:
    """Build local A_i-projector data from an LS-DD fine level."""
    P = level.P.tocsr()
    n_aggs = int(level.n_aggs)

    nev_arr = np.asarray(level.eigs.nev, dtype=np.int32)
    col_ptr = np.zeros(n_aggs + 1, dtype=np.int32)
    col_ptr[1:] = np.cumsum(nev_arr)

    blocks: list[LocalProjectionBlock] = []
    for i in range(n_aggs):
        p0 = int(level.blocks.submatrices_ptr[i])
        p1 = int(level.blocks.submatrices_ptr[i + 1])
        a_flat = np.asarray(level.blocks.submatrices[p0:p1])
        dim = int(np.sqrt(a_flat.size))
        A_OMEGA = a_flat.reshape((dim, dim))

        pou_i = np.asarray(level.sub.PoU[i], dtype=float)
        omega_pos = np.flatnonzero(pou_i == 1)
        if omega_pos.size == 0:
            continue

        OMEGA_i = np.asarray(level.sub.OMEGA[i], dtype=np.int32)
        omega_rows = np.asarray(OMEGA_i[omega_pos], dtype=np.int32)
        A_i_raw = A_OMEGA[np.ix_(omega_pos, omega_pos)]
        A_i = 0.5 * (A_i_raw + A_i_raw.T)

        c0 = int(col_ptr[i])
        c1 = int(col_ptr[i + 1])
        if c1 > c0:
            Z_i = P[omega_rows, c0:c1].toarray()
            AZ_i = A_i @ Z_i
            G_i = Z_i.T @ AZ_i
            G_i = 0.5 * (G_i + G_i.T)
            fact = factor_dense_spd(G_i)
            blk = LocalProjectionBlock(
                agg_id=i,
                omega_rows=omega_rows,
                A_i=A_i,
                Z_i=Z_i,
                AZ_i=AZ_i,
                use_cholesky=fact.use_cholesky,
                chol_factor=fact.chol_factor,
                chol_lower=fact.chol_lower,
                gram=fact.matrix,
            )
        else:
            blk = LocalProjectionBlock(
                agg_id=i,
                omega_rows=omega_rows,
                A_i=A_i,
                Z_i=np.zeros((omega_rows.size, 0), dtype=A_i.dtype),
                AZ_i=np.zeros((omega_rows.size, 0), dtype=A_i.dtype),
                use_cholesky=False,
                chol_factor=None,
                chol_lower=True,
                gram=None,
            )

        blocks.append(blk)

    return blocks, nev_arr


def extract_level_diagnostics(level) -> tuple[float | None, float | None]:
    """Extract optional threshold diagnostics from an LS-DD setup level."""
    tau = None
    if level.eigs.threshold is not None:
        tau = float(level.eigs.threshold)

    max_i_mu = None
    if level.eigs.first_discarded is not None:
        fd = np.asarray(level.eigs.first_discarded, dtype=float)
        fd = fd[~np.isnan(fd)]
        if fd.size:
            max_i_mu = float(np.max(fd))

    return tau, max_i_mu
