"""Local dense operator extraction and local outer-product construction.

This module builds the per-aggregate *dense* blocks needed by LS–AMG–DD:

1) Dense principal submatrices
   For each aggregate i with overlap set OMEGA_i (global DOF indices),
   we extract the dense block
       A_i = A[OMEGA_i, OMEGA_i]
   and store all A_i in one flattened 1D array with pointer offsets.

2) Local splitting blocks from rows of B
   For each aggregate i, given a row set R_rows_i (global row indices in B)
   and the same overlap set OMEGA_i (global column indices for B),
   we construct an SPSD dense block \\tilde{A}_i aligned with A_i.
   This is done by a compiled kernel (`pyamg.amg_core.local_outer_product`).

Storage layout
--------------
Dense blocks are stored in flattened arrays:

- `subdomain` / `subdomain_ptr`:
    concatenation of OMEGA_i (int32 indices) and CSR-style pointers.

- `submatrices` / `submatrices_ptr`:
    concatenation of A_i (flattened row-major) and BSR-style pointers.

- `auxiliary` (same ptr as `submatrices_ptr`):
    concatenation of \\tilde{A}_i (flattened row-major).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyamg import amg_core

def _lsdd_extract_local_principal_submatrices(*, level: Any, A) -> None:
    """Extract dense principal blocks A[OMEGA_i, OMEGA_i] for all aggregates.

    Parameters
    ----------
    level
        Current multigrid level. Requires:
          - level.N
          - level.sub.OMEGA (list of global OMEGA_i arrays)
          - level.sub.n_OMEGA (int32 array of sizes)
          - level.blocks (LocalBlocks container)

    A
        CSR-like operator on this level.

    Side effects
    ------------
    Populates `level.blocks`:
      - subdomain, subdomain_ptr
      - submatrices, submatrices_ptr
      - auxiliary (allocated, filled later by local outer products)
    """
    sub = level.sub
    blocks = level.blocks

    blocksize = sub.n_OMEGA
    n_tot = int(np.sum(blocksize))

    blocks.subdomain = np.zeros(n_tot, dtype=np.int32)
    blocks.subdomain_ptr = np.zeros(level.N + 1, dtype=np.int32)
    blocks.subdomain_ptr[0] = 0

    for i in range(level.N):
        bs = int(blocksize[i])
        blocks.subdomain_ptr[i + 1] = blocks.subdomain_ptr[i] + bs
        blocks.subdomain[blocks.subdomain_ptr[i] : blocks.subdomain_ptr[i + 1]] = sub.OMEGA[i]

    blocks.submatrices_ptr = np.zeros(level.N + 1, dtype=np.int32)
    blocks.submatrices_ptr[0] = 0
    for i in range(level.N):
        bs = int(blocksize[i])
        blocks.submatrices_ptr[i + 1] = blocks.submatrices_ptr[i] + bs * bs

    blocks.submatrices = np.zeros(blocks.submatrices_ptr[-1], dtype=A.data.dtype)
    amg_core.extract_subblocks(
        A.indptr,
        A.indices,
        A.data,
        blocks.submatrices,
        blocks.submatrices_ptr,
        blocks.subdomain,
        blocks.subdomain_ptr,
        int(blocks.subdomain_ptr.shape[0] - 1),
        A.shape[0],
    )

    blocks.auxiliary = np.zeros(blocks.submatrices_ptr[-1], dtype=blocks.submatrices.dtype)


def _lsdd_local_outer_products_and_gep_init(
    *,
    level: Any,
    B,
    BT,
    v_row_mult: np.ndarray,
    kappa: float,
    threshold: float | None,
) -> tuple[list, list, list, int]:
    """Fill local splitting blocks \\tilde{A}_i and initialize GEP/P assembly state.

    Parameters
    ----------
    level
        Current level. Requires:
          - level.N
          - level.sub.R_rows (list of per-aggregate B-row index arrays)
          - level.sub.OMEGA  (list of per-aggregate DOF index arrays)
          - level.sub.number_of_colors, level.sub.multiplicity (scalars)
          - level.blocks.auxiliary and level.blocks.submatrices_ptr
          - level.eigs (EigenInfo container)

    B, BT
        Least-squares factor B and its transpose BT (CSR-like).

    v_row_mult
        Row multiplicities array of shape (m,).

    kappa, threshold
        If threshold is None, set a default threshold using kappa/colors/multiplicity.
        Otherwise use the given threshold.

    Returns
    -------
    p_r, p_c, p_v, counter
        Empty triplet lists and initial counter=0 for P assembly.
    """
    sub = level.sub
    blocks = level.blocks
    eigs = level.eigs

    BTT = BT.T.conjugate().tocsr()

    rows_indptr = np.zeros(level.N + 1, dtype=np.int32)
    cols_indptr = np.zeros(level.N + 1, dtype=np.int32)
    for i in range(level.N):
        rows_indptr[i + 1] = rows_indptr[i] + len(sub.R_rows[i])
        cols_indptr[i + 1] = cols_indptr[i] + len(sub.OMEGA[i])

    rows_flat = np.concatenate(sub.R_rows).astype(np.int32, copy=False)
    cols_flat = np.concatenate(sub.OMEGA).astype(np.int32, copy=False)

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
        blocks.auxiliary,
        blocks.submatrices_ptr,
    )

    if threshold is None:
        thr = max(0.1, ((kappa / sub.number_of_colors) - 1) / sub.multiplicity)
    else:
        thr = float(threshold)

    eigs.threshold = thr
    eigs.min_ev = 1e12

    p_r: list = []
    p_c: list = []
    p_v: list = []
    counter = 0
    return p_r, p_c, p_v, counter
