"""Spectral Domain Decomposition - Least Squares.

Copied from least_squares_dd.py at commit ba9bfe81d645dd94739fc36604779eb70ff608da

ruff check least_squares_dd_exp.py --select F,E9

PYAMG_LSDD_PRINT_INFO=1 pytest -q -s ../tests/schwarz/test_lsdd_compare.py
"""


from __future__ import annotations

from typing import Any, Literal


from .lsdd.types import LSDDConfig

from warnings import warn
from scipy.sparse import csr_array, issparse, \
    SparseEfficiencyWarning

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import asfptype, \
    levelize_strength_or_aggregation, levelize_weight


from .lsdd.hierarchy import _lsdd_extend_hierarchy
from .lsdd.smoothers import lsdd_make_smoother_spec
from .lsdd.types import FilteringSpec, SparseLike

Symmetry = Literal["symmetric", "hermitian"]
SmootherName = Literal["msm", "asm", "ras", "rasT"] | None



def least_squares_dd_solver_exp(
    B: SparseLike,
    BT: SparseLike | None = None,
    A: SparseLike | None = None,
    *,
    presmoother: SmootherName = "ras",
    postsmoother: SmootherName = "rasT",
    symmetry: Symmetry = "symmetric",
    strength: Any = None,
    aggregate: Any = "standard",
    agg_levels: int = 1,
    kappa: float | list[float] = 500,
    nev: int | None = None,
    threshold: float | None = None,
    min_coarsening: int | list[int] | None = None,
    max_levels: int = 10,
    max_coarse: int = 100,
    filteringA: FilteringSpec | None = (False, 0.0),
    filteringB: FilteringSpec | None = (False, 0.0),
    max_density: float = 0.1,
    print_info: bool = False,
    **kwargs: Any,
) -> MultilevelSolver:
    """Build an LS–AMG–DD multilevel solver from a least-squares factor.

    This constructs a multigrid hierarchy for an SPD least-squares operator

        A = B^T B,

    where B is typically rectangular (m x n) and A is (n x n). The hierarchy is
    built by repeated:
      1) strength-of-connection + aggregation (nonoverlapping aggregates ω_i),
      2) one-ring overlap Ω_i = ω_i ∪ Γ_i and Boolean PoU D_{i,ℓ},
      3) local SPSD splitting terms \\tilde{A}_{i,ℓ} from rows of B,
      4) per-aggregate generalized eigenproblems to build block-diagonal P,
      5) least-squares propagation: B_{ℓ+1} = B_ℓ P_ℓ, A_{ℓ+1} = P_ℓ^T A_ℓ P_ℓ.

    Parameters
    ----------
    B
        Sparse least-squares factor (m x n). This is the primary input.
    BT
        Optional transpose/conjugate-transpose of B. If None, it is formed.
    A
        Optional normal-equations matrix (n x n). If None, formed as BT @ B.
    presmoother, postsmoother
        Schwarz-based smoothers on each level. Typical choice: ("ras", "rasT").
    symmetry
        "symmetric" or "hermitian". Stored as metadata on A.
    strength, aggregate
        PyAMG strength / aggregation specs (levelized internally).
    agg_levels
        Number of aggregation passes per level.
    kappa
        Row-splitting parameter controlling the local SPSD splitting weights.
        Can be scalar or per-level list.
    nev, threshold
        Eigenvector selection knobs for the local GEPs: keep either a fixed
        number (nev) or those above a threshold (threshold).
    min_coarsening
        Enforce a minimum coarsening ratio (levelized internally).
    max_levels, max_coarse, max_density
        Stopping criteria for hierarchy construction.
    filteringA, filteringB
        Optional row-filtering controls of the form (lump_diagonal, theta).
    print_info
        Print per-level timing and size stats during setup.
    kwargs
        Forwarded to `MultilevelSolver`.

    Returns
    -------
    ml : pyamg.multilevel.MultilevelSolver
        Configured multilevel solver.
    """

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
    
    if symmetry not in ('symmetric', 'hermitian'):
        raise ValueError('Expected "symmetric" or "hermitian" for the symmetry parameter ')
    A.symmetry = symmetry
    
    # Set "is_spd" flag to trigger Cholesky-based inversion of Schwarz blocks in the presmoother
    A.is_spd = True

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

    # Ensure filtering specs are in the form (bool, float)
    if filteringA is not None:
        filteringA = (bool(filteringA[0]), float(filteringA[1]))
    if filteringB is not None:
        filteringB = (bool(filteringB[0]), float(filteringB[1]))

    # Construct multilevel structure
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A          
    levels[-1].B = B          
    levels[-1].BT = BT        
    levels[-1].BT_provided = BT_provided
    levels[-1].A_provided = A_provided
    levels[-1].density = len(levels[-1].A.data) / (levels[-1].A.shape[0] ** 2)

    lvl = 0
    pre_smooth = []
    post_smooth = []
    while len(levels) < max_levels and \
        levels[-1].A.shape[0] > max_coarse and \
        levels[-1].density < max_density:
        
        # Extend the hierarchy
        cfg = LSDDConfig(
            agg_levels=agg_levels,
            kappa=float(kappa[lvl]),
            nev=nev,
            threshold=threshold,
            min_coarsening=min_coarsening[lvl],
            filteringA=filteringA,
            filteringB=filteringB,
            print_info=print_info,
        )

        _lsdd_extend_hierarchy(
            levels=levels,
            strength_spec=strength[lvl],
            aggregate_spec=aggregate[lvl],
            cfg=cfg,
        )

        # Determine the specifications for the smoother for this level 
        sm1 = lsdd_make_smoother_spec(level=levels[-2], smoother=presmoother)
        sm2 = lsdd_make_smoother_spec(level=levels[-2], smoother=postsmoother)

        pre_smooth.append(sm1)
        post_smooth.append(sm2)
        lvl += 1

    ml = MultilevelSolver(levels, **kwargs)

    # Construct the smoothers across all levels
    change_smoothers(ml, pre_smooth, post_smooth)

    return ml