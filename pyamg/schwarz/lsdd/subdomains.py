"""Subdomain construction for LS–AMG–DD.

This module builds, for each aggregate i:
  - omega_i  : the nonoverlapping aggregate DOFs (from AggOp)
  - OMEGA_i  : a one-ring overlap of omega_i using A's adjacency
  - GAMMA_i  : the interface set OMEGA_i \\ omega_i
  - R_rows_i : a set of rows of B used to build local outer products

It also builds a simple 0/1 partition-of-unity mask `PoU[i]` over the ordering
of OMEGA_i (1 on omega_i, 0 on GAMMA_i) and some auxiliary incidence data used
for diagnostics and default threshold choices.
"""

from __future__ import annotations
from re import sub

from .types import LSDDLevel

import numpy as np
from scipy.sparse import csr_array

def _lsdd_build_overlap_and_pou(*, level: LSDDLevel, A, B, v_row_mult: np.ndarray, 
                                force_row_closure: bool,
                                print_info: bool) -> None:
    """Build omega/OMEGA/GAMMA, row sets, and PoU masks for all aggregates.

    Parameters
    ----------
    level
        Current multigrid level. Required:
          - level.n_aggs
          - level.AggOpT (CSR)
          - level.sub (Subdomains container already allocated)

    A
        CSR-like operator on this level; used for graph adjacency to define OMEGA_i.

    B
        CSR-like factor (m x n) in CSR format. Used to define row sets R_rows_i via
        adjacency in B.

    v_row_mult
        Array of shape (m,), updated in-place so that v_row_mult[r] counts how many
        aggregates include row r in their R_rows_i set.

    force_row_closure
        If True, forces the R_rows_i sets to be closed under adjacency in B, which can be helpful for robustness in some cases. This not an option in the original implementation.

    print_info
        If True, prints mean/max |OMEGA_i| and performs a lightweight PoU check.

    Side effects
    ------------
    Fills `level.sub`:
      - sub.omega[i], sub.n_omega[i]
      - sub.OMEGA[i], sub.n_OMEGA[i]
      - sub.GAMMA[i]
      - sub.R_rows[i]
      - sub.PoU[i] (0/1 mask over OMEGA_i ordering)
      - sub.nodes_vs_subdomains, sub.T, sub.number_of_colors, sub.multiplicity
      - sub.PoU_flat reset to None (cache invalidated)
    """

    # Create a CSC copy of B for efficient column slicing to define R_rows_i
    B_csc = B if getattr(B, "format", None) == "csc" else B.tocsc()
    # Note that the slicing we do below does not require sorted indices
    # if hasattr(B_csc, "sort_indices") and not B_csc.has_sorted_indices:
    #     B_csc.sort_indices()

    # Originally this function took in BT in CSR form and sliced rows of BT to define R_rows_i.
    # B_csc = B.T.tocsr() # == BT that was originally used here...
    # B_csc.sort_indices()  # ensure sorted for efficient slicing

    sub = level.sub

    nodes_r: list[np.ndarray] = []
    nodes_c: list[np.ndarray] = []
    nodes_v: list[np.ndarray] = []

    for i in range(level.n_aggs):
        # omega_i: DOFs in aggregate i (global)
        omega_i = np.asarray(
            level.AggOpT.indices[level.AggOpT.indptr[i] : level.AggOpT.indptr[i + 1]],
            dtype=np.int32,
        )
        sub.omega[i] = omega_i
        sub.n_omega[i] = omega_i.size

        # OMEGA_i: union of A-neighbors of omega_i
        neigh = []
        for j in omega_i:
            neigh.append(A.indices[A.indptr[j] : A.indptr[j + 1]])
        OMEGA_i = np.unique(np.concatenate(neigh, dtype=np.int32))
        sub.OMEGA[i] = OMEGA_i
        sub.n_OMEGA[i] = OMEGA_i.size

        # R_rows_i: union of B-row indices touching omega_i (via BT adjacency)
        rows = []
        for j in omega_i:
            rows.append(B_csc.indices[B_csc.indptr[j] : B_csc.indptr[j + 1]])
        R_rows_i = np.unique(np.concatenate(rows, dtype=np.int32))
        sub.R_rows[i] = R_rows_i
        v_row_mult[R_rows_i] += 1


        # --- IMPORTANT: ensure OMEGA_i is "row-closed" with respect to the selected B-rows ---
        #
        # What went wrong without this:
        # -----------------------------
        # We build:
        #   - OMEGA_i from the graph of A (one-ring neighbors of omega_i), and
        #   - R_rows_i as all B-rows that touch omega_i (via B^T column adjacency).
        # Later, local Gram blocks are formed by restricting B to these indices:
        #     B_loc = B[R_rows_i, :][:, OMEGA_i]
        # and then taking weighted outer products / Schur complements.
        #
        # This implicitly assumes a *closure* property:
        #     for every selected row r in R_rows_i, supp(B[r, :]) ⊆ OMEGA_i,
        # i.e. that every nonzero in any selected row is retained in the column slice.
        #
        # If this is not true, then B_loc is NOT a true submatrix of B: some selected rows
        # are silently *truncated* (columns outside OMEGA_i are dropped, i.e. treated as 0).
        # That changes the local least-squares geometry and the local Schur complement S:
        # interface variables can no longer cancel residual components that depend on the
        # dropped DOFs, and expected near-kernel directions (e.g. constants on fully interior
        # aggregates for diffusion-type Grams) disappear. Practically, this turns “infinite”
        # eigenvalues into moderate ones and makes the dominant finite modes look like the
        # next-smoothest shapes (often well fit by affine functions), which can be misleading.
        #
        # Why this can happen even when A = B^T B:
        # ---------------------------------------
        # The sparsity pattern of A is determined by column dot-products:
        #     A_ij = Σ_r B_{r i} B_{r j}.
        # Even if i and j co-occur in some row(s) of B, their dot-product can be EXACTLY zero
        # (orthogonality / cancellations), so A_ij = 0 and the A-graph omits that adjacency.
        # This is common in CG(1) diffusion assembled from element-gradient Grams: some pairs
        # of local basis gradients can be orthogonal on an element, giving a zero stiffness
        # coupling, even though the element-level B-row block touches both DOFs.
        #
        # Fix:
        # ----
        # After selecting R_rows_i, augment OMEGA_i by all columns touched by those rows:
        #     OMEGA_i ← OMEGA_i ∪ (⋃_{r∈R_rows_i} supp(B[r,:])).
        # This guarantees we never truncate selected rows and preserves the correct local
        # LS projector / Schur complement structure. After this closure, fully interior
        # aggregates recover the expected null/near-null directions; with our eps*I
        # regularization of S, those show up as ~1/eps eigenvalues (finite proxies for
        # the mathematically infinite eigenvalues of the unregularized pencil).
        #
        # --- end IMPORTANT ---

        # --- NEW: close OMEGA_i under the supports of the selected B-rows ---
        # Later we form local pieces using B[R_rows_i, :][:, OMEGA_i].
        # If any selected row r has nonzeros in columns outside OMEGA_i, that row gets
        # implicitly truncated, which can destroy expected near-kernel structure.
        if force_row_closure:
            cols = []
            for r in R_rows_i:
                cols.append(B.indices[B.indptr[r] : B.indptr[r + 1]])
            if cols:
                cols_i = np.unique(np.concatenate(cols, dtype=np.int32))
                OMEGA_i = np.unique(np.concatenate((OMEGA_i, cols_i), dtype=np.int32))

            sub.OMEGA[i] = OMEGA_i
            sub.n_OMEGA[i] = OMEGA_i.size
            # --- end NEW ---

        # incidence data for overlap diagnostics
        nodes_r.append(OMEGA_i)
        nodes_c.append(i * np.ones(OMEGA_i.size, dtype=np.int32))
        nodes_v.append(np.ones(OMEGA_i.size, dtype=float))

    rr = np.concatenate(nodes_r, dtype=np.int32)
    cc = np.concatenate(nodes_c, dtype=np.int32)
    vv = np.concatenate(nodes_v, dtype=float)

    sub.nodes_vs_subdomains = csr_array((vv, (rr, cc)), shape=(A.shape[0], level.n_aggs))

    sub.T = sub.nodes_vs_subdomains.T @ sub.nodes_vs_subdomains
    sub.T.data[:] = 1

    k_c = sub.T @ np.ones(sub.T.shape[0], dtype=sub.T.data.dtype)
    sub.number_of_colors = float(np.max(k_c))
    sub.multiplicity = float(np.max(v_row_mult))

    # PoU masks and GAMMA_i (vectorized per i)
    for i in range(level.n_aggs):
        OMEGA_i = sub.OMEGA[i]
        omega_i = sub.omega[i]
        pou = np.isin(OMEGA_i, omega_i).astype(float)
        sub.PoU[i] = pou
        sub.GAMMA[i] = OMEGA_i[pou == 0]

    # Invalidate cached flattening for RAS PoU
    sub.PoU_flat = None

    # Optional PoU consistency check: store result for stats reporting
    sub.pou_rel_error = None
    if print_info:
        x = np.random.default_rng(0).random(A.shape[0])
        y = np.zeros_like(x)
        for i in range(level.n_aggs):
            OMEGA_i = sub.OMEGA[i]
            y[OMEGA_i] += sub.PoU[i] * x[OMEGA_i]

        denom = np.linalg.norm(x)
        sub.pou_rel_error = float(np.linalg.norm(y - x) / denom) if denom != 0.0 else 0.0

