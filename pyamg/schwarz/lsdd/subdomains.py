
from __future__ import annotations

from warnings import warn
import numpy as np
from scipy.sparse import csr_array



def _lsdd_build_overlap_and_pou(
    *,
    level,
    A,
    B,
    BT,
    v_mult,
    v_row_mult,
    blocksize,
    print_info: bool,
):
    """Build overlapping subdomains, overlap-row sets, and PoU masks.

    Refactor-only extraction of the existing code inside the "overlap" timing block.

    Fills/updates on `level`:
      - nonoverlapping_subdomain[i]
      - overlapping_subdomain[i]
      - overlapping_rows[i]
      - PoU[i]
      - nodes_vs_subdomains, T, number_of_colors, multiplicity
      - nIi[i], ni[i]

    Mutates the provided arrays:
      - v_mult (node multiplicity over overlaps)
      - v_row_mult (row multiplicity for BT-row sets)
      - blocksize (|Omega_i|)
    """
    nodes_vs_subdomains_r = []
    nodes_vs_subdomains_c = []
    nodes_vs_subdomains_v = []

    for i in range(level.N):
        # List of aggregate indices for nonoverlapping subdomains
        level.nonoverlapping_subdomain[i] = np.asarray(
            level.AggOpT.indices[level.AggOpT.indptr[i] : level.AggOpT.indptr[i + 1]],
            dtype=np.int32,
        )
        level.nIi[i] = len(level.nonoverlapping_subdomain[i])
        level.overlapping_subdomain[i] = []

        # Form overlapping subdomains as all fine-grid neighbors of each coarse aggregate
        for j in level.nonoverlapping_subdomain[i]:
            level.overlapping_subdomain[i].append(A.indices[A.indptr[j] : A.indptr[j + 1]])

        level.overlapping_subdomain[i] = np.concatenate(level.overlapping_subdomain[i], dtype=np.int32)
        level.overlapping_subdomain[i] = np.unique(level.overlapping_subdomain[i])
        level.ni[i] = len(level.overlapping_subdomain[i])

        # Get the overlapping rows
        level.overlapping_rows[i] = []
        for j in level.nonoverlapping_subdomain[i]:
            level.overlapping_rows[i].append(BT.indices[BT.indptr[j] : BT.indptr[j + 1]])

        level.overlapping_rows[i] = np.concatenate(level.overlapping_rows[i], dtype=np.int32)
        level.overlapping_rows[i] = np.unique(level.overlapping_rows[i])
        v_row_mult[level.overlapping_rows[i]] += 1

        blocksize[i] = len(level.overlapping_subdomain[i])

        # Loop over the subdomain and get the PoU (here PoU is a 0/1 mask: 1 on ω, 0 on Γ)
        v_mult[level.overlapping_subdomain[i]] += 1
        nodes_vs_subdomains_r.append(level.overlapping_subdomain[i])
        nodes_vs_subdomains_c.append(i * np.ones(len(level.overlapping_subdomain[i]), dtype=np.int32))
        nodes_vs_subdomains_v.append(np.ones(len(level.overlapping_subdomain[i])))

    nodes_vs_subdomains_r = np.concatenate(nodes_vs_subdomains_r, dtype=np.int32)
    nodes_vs_subdomains_c = np.concatenate(nodes_vs_subdomains_c, dtype=np.int32)
    nodes_vs_subdomains_v = np.concatenate(nodes_vs_subdomains_v, dtype=np.float64)

    level.nodes_vs_subdomains = csr_array(
        (nodes_vs_subdomains_v, (nodes_vs_subdomains_r, nodes_vs_subdomains_c)),
        shape=(A.shape[0], level.N),
    )
    level.T = level.nodes_vs_subdomains.T @ level.nodes_vs_subdomains
    level.T.data[:] = 1
    k_c = level.T @ np.ones(level.T.shape[0], dtype=level.T.data.dtype)
    level.number_of_colors = max(k_c)
    level.multiplicity = max(v_row_mult)

    if print_info:
        print("\tMean blocksize = {:.4g}".format(np.mean(blocksize)))
        print("\tMax blocksize = {:.4g}".format(np.max(blocksize)))

    # Form partition of unity vector separating overlapping and nonoverlapping domains.
    for i in range(level.N):
        level.PoU[i] = []
        for j in level.overlapping_subdomain[i]:
            if j in level.nonoverlapping_subdomain[i]:
                level.PoU[i].append(1)
            else:
                level.PoU[i].append(0.0)
        level.PoU[i] = np.array(level.PoU[i])

    # Sanity check PoU correct (should reproduce any vector exactly)
    temp = np.random.rand(A.shape[0])
    temp2 = 0 * temp
    for i in range(level.N):
        temp2[level.overlapping_subdomain[i]] += level.PoU[i] * temp[level.overlapping_subdomain[i]]
    if np.linalg.norm(temp2 - temp) > 1e-14 * np.linalg.norm(temp):
        warn(
            "Partition of unity is incorrect. This can happen if the partitioning strategy "
            "did not yield a nonoverlapping cover of the set of nodes"
        )
