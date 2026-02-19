"""Spectral Domain Decomposition - Least Squares.

Copied from least_squares_dd.py at commit ba9bfe81d645dd94739fc36604779eb70ff608da

ruff check least_squares_dd_exp.py --select F,E9

PYAMG_LSDD_PRINT_INFO=1 pytest -q -s ../tests/schwarz/test_lsdd_compare.py
"""


from __future__ import annotations

from warnings import warn
import numpy as np
from scipy.sparse import csr_array, issparse, \
    SparseEfficiencyWarning, csc_array, hstack

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import asfptype, \
    levelize_strength_or_aggregation, levelize_weight


from .lsdd.aggregation import _lsdd_build_aggop, _lsdd_init_level_after_aggregation, _lsdd_filter_ops_inplace, _lsdd_build_strength
from .lsdd.eigs import _lsdd_process_one_aggregate_gep
from .lsdd.hierarchy import _lsdd_assemble_P_from_triplets, _lsdd_coarsen_operators, _lsdd_append_next_level
from .lsdd.local_ops import _lsdd_extract_local_principal_submatrices, _lsdd_local_outer_products_and_gep_init
from .lsdd.subdomains import _lsdd_build_overlap_and_pou
from .lsdd.stats import LsddLevelStats, _lsdd_finalize_level_stats, _lsdd_print_level_summary
import time

def least_squares_dd_solver_exp(B, BT=None, A=None,
                            presmoother="ras",
                            postsmoother="rasT",
                            symmetry='symmetric', 
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


def _extend_hierarchy(levels, strength, aggregate, agg_levels,\
    kappa, nev, threshold, min_coarsening, filteringA, \
    filteringB, print_info):
    """Extend the multigrid hierarchy.

    Service routine to implement the strength of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    smoothed_aggregation_solver.

    """
    A = levels[-1].A
    B = levels[-1].B
    BT = levels[-1].BT

    
    # Initialize stats object for this level
    stats = LsddLevelStats(level=len(levels) - 1, n_fine=A.shape[0])

    # -----------------------
    # --- Filter operator ---
    # -----------------------
    if len(levels) > 1:
        with stats.timeit("filter"):
            _lsdd_filter_ops_inplace(
                A=A,
                B=B,
                BT=BT,
                filteringA=filteringA,
                filteringB=filteringB,
                print_info=print_info,
            )

    # --------------------------------------
    # --- Compute strength of connection ---
    # --------------------------------------
    with stats.timeit("strength"):
        C = _lsdd_build_strength(A=A, B=B, strength_spec=strength[len(levels) - 1])

    # ----------------------------------------
    # --- Compute aggregation matrix AggOp ---
    # ----------------------------------------
    with stats.timeit("aggregate"):
        AggOp, nc_temp = _lsdd_build_aggop(
            A=A,
            C=C,
            aggregate_spec=aggregate[len(levels) - 1],
            agg_levels=agg_levels,
            is_finest=(len(levels) == 1),
        )
        v_mult, blocksize, v_row_mult = _lsdd_init_level_after_aggregation(
            level=levels[-1],
            AggOp=AggOp,
            A=A,
            B=B,
        )
    
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
    with stats.timeit("extract_PCM"):
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
        eigvals_kept = []
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
                eigvals_kept = eigvals_kept,
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
    fine_sym = getattr(levels[-1].A, "symmetry", None)
    fine_is_spd = getattr(levels[-1].A, "is_spd", None)
    with stats.timeit("coarsen"):
        A, B, BT = _lsdd_coarsen_operators(B=B, BT=BT, P=levels[-1].P, R=levels[-1].R)

        # Propagate symmetry metadata (SciPy drops custom attrs on matmul)
        if fine_sym is not None:
            A.symmetry = fine_sym
        if fine_is_spd is not None:
            A.is_spd = fine_is_spd

    # ------------------------------------------    --
    # --- Finalize stats and append next level ---
    # --------------------------------------------
    _lsdd_finalize_level_stats(
        stats=stats,
        level=levels[-1],
        blocksize=blocksize,
        eigvals_kept=eigvals_kept,
        n_coarse=A.shape[0],
    )
    _lsdd_print_level_summary(stats, print_info=print_info)

    _lsdd_append_next_level(levels=levels, A=A, B=B, BT=BT)


    


  
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
              
