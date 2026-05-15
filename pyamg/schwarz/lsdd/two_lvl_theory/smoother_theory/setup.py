"""Setup-only helpers for smoother-side LS-DD diagnostics."""

from __future__ import annotations

from typing import Any

from pyamg.multilevel import MultilevelSolver
from pyamg.util.utils import levelize_strength_or_aggregation

from ...aggregation import (
    _lsdd_build_aggop,
    _lsdd_build_strength,
    _lsdd_init_level_after_aggregation,
)
from ...smoothers import _lsdd_flatten_subdomains
from ...subdomains import _lsdd_build_overlap_and_pou
from ..setup import as_csr_fptype


def build_one_level_smoother_context(
    *,
    B,
    A,
    BT,
    symmetry: str,
    strength: Any,
    aggregate: Any,
    agg_levels: int,
    force_row_closure: bool,
    print_info: bool,
    max_levels: int = 2,
    max_coarse: int = 10,
):
    """Build one LS-DD level with aggregation/subdomains only.

    This helper stops after overlap construction and subdomain flattening.
    It does not build local eigenproblems, interpolation, or coarse levels.
    """
    if symmetry not in ("symmetric", "hermitian"):
        raise ValueError('Expected "symmetric" or "hermitian" for the symmetry parameter')

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
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError("expected square matrix")
    A_csr.symmetry = symmetry
    A_csr.schwarz_use_cholesky = True

    _, _, strength_lvls = levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    _, _, aggregate_lvls = levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)

    levels = [MultilevelSolver.Level()]
    level = levels[0]
    level.A = A_csr
    level.B = B_csr
    level.BT = BT_csr
    level.density = len(A_csr.data) / (A_csr.shape[0] ** 2)

    C = _lsdd_build_strength(A=A_csr, B=B_csr, strength_spec=strength_lvls[0])
    AggOp, _ = _lsdd_build_aggop(
        A=A_csr,
        C=C,
        aggregate_spec=aggregate_lvls[0],
        agg_levels=int(agg_levels),
        is_finest=True,
    )
    v_row_mult = _lsdd_init_level_after_aggregation(level=level, AggOp=AggOp, A=A_csr, B=B_csr)
    _lsdd_build_overlap_and_pou(
        level=level,
        A=A_csr,
        B=B_csr,
        v_row_mult=v_row_mult,
        force_row_closure=bool(force_row_closure),
        print_info=bool(print_info),
    )

    # Ensure smoother specs for OMEGA-domain Schwarz can be created without
    # requiring dense local principal-block extraction.
    subdomain, subdomain_ptr = _lsdd_flatten_subdomains(level.sub.OMEGA)
    level.blocks.subdomain = subdomain
    level.blocks.subdomain_ptr = subdomain_ptr

    return level

