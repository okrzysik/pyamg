"""Graph construction and coloring helpers for smoother-side diagnostics.

This module contains the common sparse-graph utilities used by
``diagnostics.py``.  The high-level diagnostics code should read as a sequence
of mathematical operations; the lower-level pattern algebra and coloring calls
live here.

Notation used throughout this folder
------------------------------------
For a prepared LS-DD level, let ``A`` be the assembled SPD matrix and let ``G``
be the rectangular least-squares/Gram factor stored as ``level.B``.  The theory
is written for ``A = G.T @ G``.

For a domain layout, either ``omega`` or ``OMEGA``, let ``D`` be the Boolean
DOF-by-domain incidence matrix.  Thus ``D[p, i] = 1`` means that global DOF
``p`` belongs to domain ``i``.

The ``G``-row incidence matrix is ``E = pattern(G) @ D``, interpreted
Booleanly.  Its entry ``E[j, i]`` is one when row ``j`` of ``G`` touches at
least one DOF in domain ``i``.  The row multiplicity used in the paper is
``nu = max_j sum_i E[j, i]``.

The module builds two domain interaction graphs.

``C_G``
    ``offdiag_pattern(E.T @ E)``.  This graph connects two domains when some
    row of ``G`` touches both of them.  This is the graph naturally associated
    with ``nu`` and the row-wise Gram representation.

``C_A``
    ``offdiag_pattern(D.T @ pattern(A) @ D)``.  This graph connects two
    domains when the assembled matrix ``A`` has at least one stored nonzero
    coupling between DOFs in the two domains.  This is the graph associated
    with the standard additive Schwarz coloring argument.

If the stored matrices are consistent with ``A = G.T @ G``, then ``C_A`` should
be a subgraph of ``C_G``.  They coincide when no row-outer-product assembly
cancellations remove block-level interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.sparse import csr_array, issparse

from pyamg.graph import vertex_coloring


Domain = Literal["omega", "OMEGA"]
ColoringMethod = Literal["MIS", "JP", "LDF"]
GraphKind = Literal["G", "A"]


@dataclass(slots=True, frozen=True)
class SmootherInteractionGraphs:
    """Sparse graph data for one Schwarz domain layout.

    All sparse matrices are Boolean ``csr_array`` objects with stored data equal
    to one.  The matrices are returned because driver-side scripts may want to
    draw them using geometric aggregate positions.
    """

    domain: Domain
    D: csr_array
    E: csr_array
    C_G: csr_array
    C_A: csr_array
    row_touch_counts: np.ndarray
    A_edges_subset_of_G_edges: bool
    A_minus_G: csr_array


@dataclass(slots=True, frozen=True)
class GraphColoringResult:
    """Coloring diagnostics for one interaction graph.

    ``chi_bnd_pyamg`` is the number of colors produced by
    ``pyamg.graph.vertex_coloring``.  This is an upper bound on the chromatic
    number, not an exact value.  ``chi_exact`` is populated only when the caller
    requests exact coloring through GCol.
    """

    graph_kind: GraphKind
    chi_bnd_pyamg: int | None
    colors_bnd_pyamg: np.ndarray | None
    coloring_bnd_valid: bool | None
    coloring_method: str
    chi_exact: int | None
    colors_exact: np.ndarray | None
    exact_coloring_valid: bool | None
    exact_coloring_seconds: float | None


def as_csr_pattern(M, *, name: str) -> csr_array:
    """Return the Boolean sparse pattern of ``M`` as sorted CSR.

    The graph diagnostics depend on sparsity structure.  Numerical values are
    discarded after explicit zeros have been removed.  No numerical threshold is
    applied; if an entry is stored as a nonzero, it is considered part of the
    pattern.
    """

    if not issparse(M):
        raise ValueError(f"Expected sparse {name}, got {type(M)!r}")

    out = M.tocsr(copy=True)
    out.eliminate_zeros()
    out.sort_indices()
    if out.nnz:
        out.data[:] = 1.0
    return csr_array(out)


def simple_graph_pattern(C, *, name: str = "interaction graph") -> csr_array:
    """Return the simple undirected graph represented by sparse pattern ``C``.

    The output has zero diagonal, symmetric pattern, sorted CSR indices, and
    unit data.  This is the graph format expected by coloring and plotting
    helpers in this folder.
    """

    graph = as_csr_pattern(C, name=name)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Expected square {name}, got shape {graph.shape}")

    graph.setdiag(0)
    graph.eliminate_zeros()

    # The matrix products used below should already produce symmetric patterns,
    # but taking the maximum makes the representation robust to storage details.
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    graph.sort_indices()
    if graph.nnz:
        graph.data[:] = 1.0
    return csr_array(graph)


def domain_incidence_from_level(level, *, domain: Domain) -> csr_array:
    """Return the Boolean DOF-by-domain incidence matrix ``D``.

    ``omega`` uses ``level.AggOp``.  ``OMEGA`` uses
    ``level.sub.nodes_vs_subdomains``.  The second object is produced by the
    LS-DD overlap setup path and records the overlapping Schwarz domains.
    """

    if domain == "omega":
        D = getattr(level, "AggOp", None)
        if D is None:
            raise ValueError("Expected level.AggOp for domain='omega'")
    elif domain == "OMEGA":
        sub = getattr(level, "sub", None)
        D = getattr(sub, "nodes_vs_subdomains", None)
        if D is None:
            raise ValueError(
                "Expected level.sub.nodes_vs_subdomains for domain='OMEGA'. "
                "Build overlap data before requesting OMEGA diagnostics."
            )
    else:
        raise ValueError(f"Unsupported domain {domain!r}; expected 'omega' or 'OMEGA'")

    return as_csr_pattern(D, name=f"domain incidence D({domain})")


def G_row_incidence_from_level(level, *, domain: Domain) -> csr_array:
    """Build the row-domain incidence matrix ``E`` for the stored factor ``G``.

    The code uses ``level.B`` for the factor called ``G`` in the paper.  Sparse
    multiplication initially counts how many DOFs in a domain are touched by a
    row.  We immediately collapse those counts to a Boolean pattern because the
    diagnostics only need incidence.
    """

    G = getattr(level, "B", None)
    if G is None:
        raise ValueError("Expected level.B on the prepared LS-DD level")

    Gpat = as_csr_pattern(G, name="level.B / G")
    D = domain_incidence_from_level(level, domain=domain)

    if Gpat.shape[1] != D.shape[0]:
        raise ValueError(
            "Shape mismatch while building E=pattern(G)D: "
            f"G has shape {Gpat.shape}, D has shape {D.shape}"
        )

    E = (Gpat @ D).tocsr()
    E.eliminate_zeros()
    E.sort_indices()
    if E.nnz:
        E.data[:] = 1.0
    return csr_array(E)


def build_C_G(E: csr_array) -> csr_array:
    """Build ``C_G = offdiag_pattern(E.T @ E)``.

    Two domains are adjacent in ``C_G`` when at least one row of ``G`` touches
    both domains.  This graph is the one for which every row touching ``r``
    domains creates an ``r``-clique, hence ``nu <= chi(C_G)``.
    """

    if E.ndim != 2:
        raise ValueError("Expected a two-dimensional row-domain incidence matrix E")
    return simple_graph_pattern(E.T @ E, name="C_G")


def build_C_A(A, D: csr_array) -> csr_array:
    """Build ``C_A = offdiag_pattern(D.T @ pattern(A) @ D)``.

    An edge in ``C_A`` means that the assembled matrix ``A`` has a stored
    structural coupling between the two domains.  This graph is the direct
    graph for the usual additive Schwarz coloring estimate.
    """

    Apat = as_csr_pattern(A, name="level.A")
    Dpat = as_csr_pattern(D, name="domain incidence D")

    if Apat.shape[0] != Apat.shape[1]:
        raise ValueError(f"Expected square A, got shape {Apat.shape}")
    if Apat.shape[0] != Dpat.shape[0]:
        raise ValueError(
            "Shape mismatch while building C_A=D.T pattern(A) D: "
            f"A has shape {Apat.shape}, D has shape {Dpat.shape}"
        )

    return simple_graph_pattern(Dpat.T @ Apat @ Dpat, name="C_A")


def graph_difference(left: csr_array, right: csr_array) -> csr_array:
    """Return the graph edges present in ``left`` and absent from ``right``."""

    L = simple_graph_pattern(left, name="left graph")
    R = simple_graph_pattern(right, name="right graph")
    if L.shape != R.shape:
        raise ValueError(f"Cannot compare graphs with shapes {L.shape} and {R.shape}")

    intersection = L.multiply(R).tocsr()
    diff = (L - intersection).tocsr()
    diff.eliminate_zeros()
    diff.sort_indices()
    if diff.nnz:
        diff.data[:] = 1.0
    return csr_array(diff)


def compute_smoother_graphs_from_level(level, *, domain: Domain) -> SmootherInteractionGraphs:
    """Build ``D``, ``E``, ``C_G``, and ``C_A`` for one domain layout.

    This function does only deterministic sparse-pattern algebra.  It does not
    color graphs and it does not compute smoother eigenvalues.
    """

    D = domain_incidence_from_level(level, domain=domain)
    E = G_row_incidence_from_level(level, domain=domain)
    C_G = build_C_G(E)

    A = getattr(level, "A", None)
    if A is None:
        raise ValueError("Expected level.A on the prepared LS-DD level")
    C_A = build_C_A(A, D)

    # Since E is CSR with Boolean data, the number of stored entries in a row is
    # exactly the number of domains touched by that row of G.
    row_touch_counts = np.diff(E.indptr).astype(np.int32, copy=False)

    # For consistent A=G.T@G data, every assembled A interaction should be
    # structurally explained by at least one row of G.  The difference matrix is
    # kept for debugging and plotting if this check ever fails.
    A_minus_G = graph_difference(C_A, C_G)

    return SmootherInteractionGraphs(
        domain=domain,
        D=D,
        E=E,
        C_G=C_G,
        C_A=C_A,
        row_touch_counts=row_touch_counts,
        A_edges_subset_of_G_edges=(int(A_minus_G.nnz) == 0),
        A_minus_G=A_minus_G,
    )


def interaction_matrix_to_networkx(C: csr_array):
    """Convert a sparse adjacency matrix to a NetworkX ``Graph``.

    NetworkX is imported only when this helper is called.  The graph has nodes
    ``0, ..., n-1`` and one undirected edge for each off-diagonal nonzero.
    """

    try:
        import networkx as nx  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise ImportError("NetworkX is required to materialize interaction graphs") from exc

    graph = simple_graph_pattern(C)
    coo = graph.tocoo()

    out = nx.Graph()
    out.add_nodes_from(range(int(graph.shape[0])))
    out.add_edges_from(
        (int(i), int(j))
        for i, j in zip(coo.row, coo.col, strict=False)
        if int(i) < int(j)
    )
    return out


def color_count(colors: np.ndarray | None) -> int | None:
    """Return the number of nonnegative color labels used in ``colors``."""

    if colors is None:
        return None
    colors = np.asarray(colors, dtype=np.int64)
    if colors.size == 0:
        return 0
    used = np.unique(colors)
    used = used[used >= 0]
    return int(used.size)


def validate_coloring(C: csr_array, colors: np.ndarray | None) -> bool | None:
    """Check whether ``colors`` is a proper coloring of graph ``C``."""

    if colors is None:
        return None

    graph = simple_graph_pattern(C)
    n = int(graph.shape[0])
    colors = np.asarray(colors, dtype=np.int64)

    if colors.shape != (n,):
        return False
    if np.any(colors < 0):
        return False

    coo = graph.tocoo()
    for i, j in zip(coo.row, coo.col, strict=False):
        # The graph is symmetric.  Testing only i<j avoids checking every edge
        # twice while preserving a direct correspondence with graph edges.
        if int(i) < int(j) and int(colors[int(i)]) == int(colors[int(j)]):
            return False
    return True


def _pyamg_upper_bound_coloring(
    C: csr_array,
    *,
    graph_kind: GraphKind,
    coloring_method: ColoringMethod,
) -> GraphColoringResult:
    """Compute PyAMG's heuristic coloring upper bound for ``C``."""

    graph = simple_graph_pattern(C, name=f"C_{graph_kind}")
    if int(graph.shape[0]) == 0:
        colors = np.zeros(0, dtype=np.int32)
    else:
        colors = np.asarray(vertex_coloring(graph, method=str(coloring_method)), dtype=np.int32)

    return GraphColoringResult(
        graph_kind=graph_kind,
        chi_bnd_pyamg=color_count(colors),
        colors_bnd_pyamg=colors,
        coloring_bnd_valid=validate_coloring(graph, colors),
        coloring_method=str(coloring_method),
        chi_exact=None,
        colors_exact=None,
        exact_coloring_valid=None,
        exact_coloring_seconds=None,
    )


def _add_exact_coloring(
    base: GraphColoringResult,
    C: csr_array,
    *,
    max_vertices: int | None,
    max_edges: int | None,
) -> GraphColoringResult:
    """Augment ``base`` with an exact GCol coloring.

    Exact graph coloring is exponential-time.  The optional guards are hard
    limits: if exact coloring is requested for a graph that exceeds them, this
    function raises instead of silently falling back to a heuristic.
    """

    graph = simple_graph_pattern(C, name=f"C_{base.graph_kind}")
    n_vertices = int(graph.shape[0])
    n_edges = int(graph.nnz // 2)

    if max_vertices is not None and n_vertices > int(max_vertices):
        raise ValueError(
            f"Exact coloring requested for C_{base.graph_kind}, but it has "
            f"{n_vertices} vertices > max_vertices={int(max_vertices)}"
        )
    if max_edges is not None and n_edges > int(max_edges):
        raise ValueError(
            f"Exact coloring requested for C_{base.graph_kind}, but it has "
            f"{n_edges} edges > max_edges={int(max_edges)}"
        )

    if n_vertices == 0:
        exact_colors = np.zeros(0, dtype=np.int32)
        exact_seconds = 0.0
    else:
        try:
            import gcol  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise ImportError("Exact coloring requires optional dependency gcol") from exc

        nx_graph = interaction_matrix_to_networkx(graph)
        t0 = perf_counter()
        coloring_map = gcol.node_coloring(nx_graph, strategy="dsatur", opt_alg=1)
        exact_seconds = perf_counter() - t0

        exact_colors = np.empty(n_vertices, dtype=np.int32)
        for node in range(n_vertices):
            exact_colors[node] = int(coloring_map[node])

    return GraphColoringResult(
        graph_kind=base.graph_kind,
        chi_bnd_pyamg=base.chi_bnd_pyamg,
        colors_bnd_pyamg=base.colors_bnd_pyamg,
        coloring_bnd_valid=base.coloring_bnd_valid,
        coloring_method=base.coloring_method,
        chi_exact=color_count(exact_colors),
        colors_exact=exact_colors,
        exact_coloring_valid=validate_coloring(graph, exact_colors),
        exact_coloring_seconds=float(exact_seconds),
    )


def compute_graph_coloring(
    C: csr_array,
    *,
    graph_kind: GraphKind,
    coloring_method: ColoringMethod = "MIS",
    compute_exact: bool = False,
    exact_max_vertices: int | None = None,
    exact_max_edges: int | None = None,
) -> GraphColoringResult:
    """Compute coloring diagnostics for one graph.

    The PyAMG coloring is always computed because it is cheap and gives an
    explicit upper-bound coloring.  Exact GCol coloring is computed only when
    ``compute_exact`` is true.
    """

    result = _pyamg_upper_bound_coloring(
        C,
        graph_kind=graph_kind,
        coloring_method=coloring_method,
    )
    if not bool(compute_exact):
        return result
    return _add_exact_coloring(
        result,
        C,
        max_vertices=exact_max_vertices,
        max_edges=exact_max_edges,
    )
