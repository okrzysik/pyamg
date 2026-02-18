
from __future__ import annotations

import numpy as np

from pyamg import amg_core


def _lsdd_extract_local_principal_submatrices(*, level, A, blocksize):
    """Extract local principal submatrices A(Omega_i, Omega_i) for all aggregates i.

    Refactor-only extraction of the existing "extract_PCM" block.

    Parameters
    ----------
    level
        The current level object (typically levels[-1]).
    A
        Global operator on this level.
    blocksize
        Array of length level.N with blocksize[i] = |Omega_i|.

    Side effects (sets on `level`)
    ------------------------------
    subdomain, subdomain_ptr
        Flattened Omega_i indices and pointers (CSR-style).
    submatrices_ptr
        Pointers into flattened storage for each dense block (BSR-style).
    submatrices
        Flattened storage of all dense A(Omega_i,Omega_i) blocks.
    auxiliary
        Flattened storage for all dense splitting blocks (filled later).
    """
    # Form sparse indptr and indices for principal submatrices over subdomains
    level.subdomain = np.zeros(np.sum(blocksize), dtype=np.int32)
    level.subdomain_ptr = np.zeros(level.N + 1, dtype=np.int32)
    level.subdomain_ptr[0] = 0

    for i in range(level.N):
        level.subdomain_ptr[i + 1] = level.subdomain_ptr[i] + blocksize[i]
        level.subdomain[level.subdomain_ptr[i] : level.subdomain_ptr[i + 1]] = level.overlapping_subdomain[i]

    # BSR-like indexing: each i has a blocksize[i] x blocksize[i] dense block
    level.submatrices_ptr = np.zeros(level.N + 1, dtype=np.int32)
    level.submatrices_ptr[0] = 0
    for i in range(level.N):
        level.submatrices_ptr[i + 1] = level.submatrices_ptr[i] + blocksize[i] * blocksize[i]

    # Extract submatrices from overlapping subdomains
    level.submatrices = np.zeros(level.submatrices_ptr[-1], dtype=A.data.dtype)
    amg_core.extract_subblocks(
        A.indptr,
        A.indices,
        A.data,
        level.submatrices,
        level.submatrices_ptr,
        level.subdomain,
        level.subdomain_ptr,
        int(level.subdomain_ptr.shape[0] - 1),
        A.shape[0],
    )

    # Allocate auxiliary storage for local outer products (filled later)
    level.auxiliary = np.zeros(level.submatrices_ptr[-1])


def _lsdd_local_outer_products_and_gep_init(
    *,
    level,
    B,
    BT,
    v_row_mult,
    kappa,
    threshold,
):
    """Compute local outer products (splitting blocks) and initialize GEP/P assembly accumulators.

    Refactor-only extraction of the existing "outerprod" block content that:
      - calls amg_core.local_outer_product to fill `level.auxiliary`
      - sets `level.threshold`
      - initializes (p_r, p_c, p_v, counter) and `level.min_ev`

    Returns
    -------
    p_r, p_c, p_v : lists
        Triplet lists for assembling P after the per-aggregate loop.
    counter : int
        Starting coarse column counter (0).
    """
    # amg_core C++ implementation of extracting local subdomain outerproducts
    BTT = BT.T.conjugate().tocsr()

    rows_indptr = np.zeros(level.N + 1, dtype=np.int32)
    cols_indptr = np.zeros(level.N + 1, dtype=np.int32)
    for i in range(level.N):
        rows_indptr[i + 1] = rows_indptr[i] + len(level.overlapping_rows[i])
        cols_indptr[i + 1] = cols_indptr[i] + len(level.overlapping_subdomain[i])

    rows_flat = np.concatenate(level.overlapping_rows).astype(np.int32, copy=False)
    cols_flat = np.concatenate(level.overlapping_subdomain).astype(np.int32, copy=False)

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
        level.auxiliary,
        level.submatrices_ptr,
    )

    if threshold is None:
        level.threshold = max(0.1, ((kappa / level.number_of_colors) - 1) / level.multiplicity)
    else:
        level.threshold = threshold

    p_r = []
    p_c = []
    p_v = []
    counter = 0
    level.min_ev = 1e12

    return p_r, p_c, p_v, counter
