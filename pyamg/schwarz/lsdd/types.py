"""Typed per-level data containers used throughout LS–AMG–DD.

This module defines small dataclasses that group “level state” into coherent parcels.
The goal is to avoid scattering many parallel attributes across a multigrid level.

Containers
----------
Subdomains
    Stores per-aggregate index sets and sizes:
      - omega[i] : nonoverlapping aggregate DOFs (global indices)
      - OMEGA[i] : one-ring overlap DOFs (global indices)
      - GAMMA[i] : interface DOFs, defined as OMEGA[i] \\ omega[i]
      - R_rows[i]: row set in B used to build local outer products (global row indices)
      - PoU[i]   : 0/1 mask over the ordering of OMEGA[i] (1 on omega, 0 on GAMMA)

    Also stores derived diagnostics:
      - nodes_vs_subdomains : incidence matrix (n_dof x N)
      - T                  : boolean overlap-count matrix (N x N)
      - number_of_colors   : max row sum of T (a crude coloring upper bound)
      - multiplicity       : max row multiplicity of the row sets R_rows[i]
      - PoU_flat           : cached concatenation of PoU masks aligned with blocks.subdomain

LocalBlocks
    Stores flattened dense blocks aligned with the OMEGA ordering:
      - subdomain/subdomain_ptr           : concatenated OMEGA indices + segment pointers
      - submatrices/submatrices_ptr       : concatenated A[OMEGA_i,OMEGA_i] dense blocks
      - auxiliary (same ptr as submatrices_ptr)
          concatenated SPSD splitting blocks \\tilde{A}_i from local outer products

EigenInfo
    Stores per-aggregate eigen-selection metadata:
      - nev[i]      : number of coarse vectors kept on aggregate i
      - threshold   : eigenvalue threshold used when nev is None
      - min_ev      : minimum eigenvalue kept across all aggregates on this level

Invariants
----------
- All index arrays are stored as int32 numpy arrays.
- OMEGA[i], GAMMA[i], R_rows[i] are unique-sorted arrays (as constructed by np.unique).
- PoU[i] has the same length and ordering as OMEGA[i].
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.sparse import csr_array  # type: ignore
except Exception:  # pragma: no cover
    csr_array = object  # type: ignore


IndexArray = NDArray[np.int32]


@dataclass(slots=True)
class Subdomains:
    """Per-aggregate index sets, sizes, and overlap diagnostics for one level.

    Attributes
    ----------
    omega, OMEGA, GAMMA, R_rows
        Per-aggregate index sets stored as lists of int32 arrays (or None before fill).
        All indices are global indices for the current level.
    PoU
        Per-aggregate 0/1 masks over OMEGA ordering:
          PoU[i][k] == 1  iff OMEGA[i][k] is in omega[i]
          PoU[i][k] == 0  iff OMEGA[i][k] is in GAMMA[i]
    n_omega, n_OMEGA
        int32 arrays of sizes |omega_i| and |OMEGA_i|.
    nodes_vs_subdomains, T, number_of_colors, multiplicity
        Derived diagnostic objects used in threshold heuristics and reporting.
    PoU_flat
        Cached concatenation of PoU masks aligned with `LocalBlocks.subdomain`.
        This is used by RAS smoothers and is invalidated whenever PoU is rebuilt.
    pou_rel_error
        Optional scalar storing a relative partition-of-unity consistency check
        (set by subdomain construction when requested).
    """


    omega: list[IndexArray | None]
    OMEGA: list[IndexArray | None]
    GAMMA: list[IndexArray | None]
    R_rows: list[IndexArray | None]
    PoU: list[np.ndarray | None]

    n_omega: IndexArray
    n_OMEGA: IndexArray

    nodes_vs_subdomains: Optional[csr_array] = None
    T: Optional[csr_array] = None
    number_of_colors: Optional[float] = None
    multiplicity: Optional[float] = None
    PoU_flat: Optional[np.ndarray] = None
    pou_rel_error: Optional[float] = None


    @classmethod
    def allocate(cls, N: int) -> "Subdomains":
        """Allocate a Subdomains container for N aggregates with empty per-aggregate slots."""
        return cls(
            omega=[None] * N,
            OMEGA=[None] * N,
            GAMMA=[None] * N,
            R_rows=[None] * N,
            PoU=[None] * N,
            n_omega=np.zeros(N, dtype=np.int32),
            n_OMEGA=np.zeros(N, dtype=np.int32),
        )


@dataclass(slots=True)
class LocalBlocks:
    """Flattened storage for per-aggregate dense blocks on one level.

    The i-th aggregate occupies a segment in each flattened array. Segment bounds are
    given by the pointer arrays:

      - subdomain[ subdomain_ptr[i] : subdomain_ptr[i+1] ]      = OMEGA_i indices
      - submatrices[ submatrices_ptr[i] : submatrices_ptr[i+1] ] = vec(A_i)
      - auxiliary [ submatrices_ptr[i] : submatrices_ptr[i+1] ]  = vec(\\tilde{A}_i)

    where vec(·) is the row-major flattening of a square dense block.
    """

    subdomain: Optional[IndexArray] = None
    subdomain_ptr: Optional[IndexArray] = None
    submatrices: Optional[np.ndarray] = None
    submatrices_ptr: Optional[IndexArray] = None
    auxiliary: Optional[np.ndarray] = None


@dataclass(slots=True)
class EigenInfo:
    """Eigen-selection metadata for one level.

    Attributes
    ----------
    nev
        int32 array of length N. `nev[i]` is the number of coarse vectors accepted
        for aggregate i.
    threshold
        Eigenvalue threshold used for selection when a fixed `nev` is not supplied.
    min_ev
        Minimum eigenvalue accepted across all aggregates on this level (for reporting).
    """

    nev: IndexArray
    threshold: Optional[float] = None
    min_ev: float = float("inf")

    @classmethod
    def allocate(cls, N: int) -> "EigenInfo":
        """Allocate an EigenInfo container for N aggregates with `nev` initialized to zeros."""
        return cls(nev=np.zeros(N, dtype=np.int32))

