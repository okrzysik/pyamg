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

from warnings import warn
from typing import Any

import numpy as np
from scipy.sparse import csr_array

def _lsdd_build_overlap_and_pou(
    *,
    level: Any,
    A,
    BT,
    v_row_mult: np.ndarray,
    print_info: bool,
) -> None:
    """Build omega/OMEGA/GAMMA, row sets, and PoU masks for all aggregates.

    Parameters
    ----------
    level
        Current multigrid level. Required:
          - level.N
          - level.AggOpT (CSR)
          - level.sub (Subdomains container already allocated)

    A
        CSR-like operator on this level; used for graph adjacency to define OMEGA_i.

    BT
        CSR-like transpose factor (n x m). Used to define row sets R_rows_i via
        adjacency in BT.

    v_row_mult
        Array of shape (m,), updated in-place so that v_row_mult[r] counts how many
        aggregates include row r in their R_rows_i set.

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
    sub = level.sub

    nodes_r: list[np.ndarray] = []
    nodes_c: list[np.ndarray] = []
    nodes_v: list[np.ndarray] = []

    for i in range(level.N):
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
            rows.append(BT.indices[BT.indptr[j] : BT.indptr[j + 1]])
        R_rows_i = np.unique(np.concatenate(rows, dtype=np.int32))
        sub.R_rows[i] = R_rows_i
        v_row_mult[R_rows_i] += 1

        # incidence data for overlap diagnostics
        nodes_r.append(OMEGA_i)
        nodes_c.append(i * np.ones(OMEGA_i.size, dtype=np.int32))
        nodes_v.append(np.ones(OMEGA_i.size, dtype=float))

    rr = np.concatenate(nodes_r, dtype=np.int32)
    cc = np.concatenate(nodes_c, dtype=np.int32)
    vv = np.concatenate(nodes_v, dtype=float)

    sub.nodes_vs_subdomains = csr_array((vv, (rr, cc)), shape=(A.shape[0], level.N))

    sub.T = sub.nodes_vs_subdomains.T @ sub.nodes_vs_subdomains
    sub.T.data[:] = 1

    k_c = sub.T @ np.ones(sub.T.shape[0], dtype=sub.T.data.dtype)
    sub.number_of_colors = float(np.max(k_c))
    sub.multiplicity = float(np.max(v_row_mult))

    # PoU masks and GAMMA_i (vectorized per i)
    for i in range(level.N):
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
        for i in range(level.N):
            OMEGA_i = sub.OMEGA[i]
            y[OMEGA_i] += sub.PoU[i] * x[OMEGA_i]

        denom = np.linalg.norm(x)
        sub.pou_rel_error = float(np.linalg.norm(y - x) / denom) if denom != 0.0 else 0.0

