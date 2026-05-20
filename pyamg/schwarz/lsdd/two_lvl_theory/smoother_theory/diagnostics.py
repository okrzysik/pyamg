"""High-level smoother-side diagnostics for LS-DD levels.

For each requested domain layout, this module packages three types of data:
interaction graphs ``C_G`` and ``C_A`` from ``smoother_graphs.py``, coloring
upper bounds and exact chromatic numbers for those graphs, and the spectral
quantity ``lambda_max(M_AS^{-1} A)`` for the exact one-level additive Schwarz
smoother.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyamg.relaxation import relaxation as _relaxation
from pyamg.relaxation.smoothing import rho_additive_schwarz_A as _rho_additive_schwarz_A

from ...smoothers import lsdd_make_smoother_spec
from .smoother_graphs import (
    ColoringMethod,
    Domain,
    GraphColoringResult,
    SmootherInteractionGraphs,
    compute_graph_coloring,
    compute_smoother_graphs_from_level,
)


@dataclass(slots=True, frozen=True)
class SmootherDomainDiagnostics:
    """Diagnostics for one smoother domain layout."""

    domain: Domain

    lambda_max_MinvA: float

    nu: int
    row_touch_counts: np.ndarray
    incidence_nnz: int
    n_G_rows: int
    n_domains: int

    C_G_nnz: int
    C_A_nnz: int
    A_edges_subset_of_G_edges: bool

    chi_bnd_pyamg_G: int | None
    chi_bnd_pyamg_A: int | None
    chi_exact_G: int | None
    chi_exact_A: int | None

    graphs: SmootherInteractionGraphs
    coloring_G: GraphColoringResult
    coloring_A: GraphColoringResult

    dof_overlap_degree_color_bound: int | None
    sanity_check_messages: tuple[str, ...]


def compute_additive_schwarz_lambda_max_from_level(level, *, domain: Domain) -> float:
    """Compute ``lambda_max(M_AS^{-1} A)`` for one exact ASM layout."""

    A = level.A.tocsr()
    A.sort_indices()

    asm_spec = lsdd_make_smoother_spec(level=level, smoother=("asm", {"domain": domain}))
    if asm_spec is None:  # pragma: no cover - defensive guard.
        raise ValueError("Failed to build ASM smoother specification")

    _, kwargs = asm_spec
    subdomain = np.asarray(kwargs["subdomain"], dtype=np.int32)
    subdomain_ptr = np.asarray(kwargs["subdomain_ptr"], dtype=np.int32)

    # PyAMG stores Schwarz setup data directly on the matrix object.  Removing
    # the cached data here ensures that each domain layout gets its own local
    # Schwarz blocks.
    if hasattr(A, "schwarz_parameters"):
        delattr(A, "schwarz_parameters")

    inv_subblock = None
    inv_subblock_ptr = None
    subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr = _relaxation.schwarz_parameters(
        A,
        subdomain,
        subdomain_ptr,
        inv_subblock,
        inv_subblock_ptr,
    )

    rho = float(
        _rho_additive_schwarz_A(
            A,
            subdomain,
            subdomain_ptr,
            inv_subblock,
            inv_subblock_ptr,
        )
    )
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"Expected positive finite lambda_max(M^-1 A), got {rho!r}")
    return rho


def _stored_dof_overlap_degree_bound(level, *, domain: Domain) -> int | None:
    """Return the setup-stored DOF-overlap degree bound for ``OMEGA``."""

    if domain != "OMEGA":
        return None
    sub = getattr(level, "sub", None)
    value = getattr(sub, "number_of_colors", None)
    return None if value is None else int(value)


def _sanity_check_messages(
    *,
    graphs: SmootherInteractionGraphs,
    lambda_max_MinvA: float,
    coloring_G: GraphColoringResult,
    coloring_A: GraphColoringResult,
    tol: float,
) -> tuple[str, ...]:
    """Return failed consistency checks for one domain layout."""

    messages: list[str] = []

    if not graphs.A_edges_subset_of_G_edges:
        messages.append(
            "C_A has interactions outside C_G: "
            f"extra adjacency nnz={graphs.A_minus_G.nnz}."
        )

    if coloring_G.coloring_bnd_valid is False:
        messages.append("PyAMG coloring is invalid on C_G.")
    if coloring_A.coloring_bnd_valid is False:
        messages.append("PyAMG coloring is invalid on C_A.")
    if coloring_G.exact_coloring_valid is False:
        messages.append("Exact coloring is invalid on C_G.")
    if coloring_A.exact_coloring_valid is False:
        messages.append("Exact coloring is invalid on C_A.")

    if coloring_G.chi_exact is not None:
        nu = int(graphs.row_touch_counts.max()) if graphs.row_touch_counts.size else 0
        if nu > int(coloring_G.chi_exact):
            messages.append(f"nu={nu} exceeds chi_exact_G={int(coloring_G.chi_exact)}.")

    if coloring_G.chi_exact is not None and coloring_A.chi_exact is not None:
        if int(coloring_A.chi_exact) > int(coloring_G.chi_exact):
            messages.append(
                f"chi_exact_A={int(coloring_A.chi_exact)} exceeds "
                f"chi_exact_G={int(coloring_G.chi_exact)}."
            )

    if coloring_A.chi_exact is not None:
        if float(lambda_max_MinvA) > float(coloring_A.chi_exact) + float(tol):
            messages.append(
                "lambda_max(M^-1 A) exceeds chi_exact_A: "
                f"lambda={float(lambda_max_MinvA):.16e}, "
                f"chi_exact_A={int(coloring_A.chi_exact)}, tol={float(tol):.3e}."
            )

    return tuple(messages)


def _compute_one_domain(
    *,
    level,
    domain: Domain,
    coloring_method: ColoringMethod,
    compute_exact_coloring: bool,
    exact_coloring_max_vertices: int | None,
    exact_coloring_max_edges: int | None,
    sanity_check_tol: float,
) -> SmootherDomainDiagnostics:
    """Compute diagnostics for one domain layout."""

    graphs = compute_smoother_graphs_from_level(level, domain=domain)

    coloring_G = compute_graph_coloring(
        graphs.C_G,
        graph_kind="G",
        coloring_method=coloring_method,
        compute_exact=compute_exact_coloring,
        exact_max_vertices=exact_coloring_max_vertices,
        exact_max_edges=exact_coloring_max_edges,
    )
    coloring_A = compute_graph_coloring(
        graphs.C_A,
        graph_kind="A",
        coloring_method=coloring_method,
        compute_exact=compute_exact_coloring,
        exact_max_vertices=exact_coloring_max_vertices,
        exact_max_edges=exact_coloring_max_edges,
    )

    lambda_max = compute_additive_schwarz_lambda_max_from_level(level, domain=domain)
    nu = int(graphs.row_touch_counts.max()) if graphs.row_touch_counts.size else 0

    messages = _sanity_check_messages(
        graphs=graphs,
        lambda_max_MinvA=lambda_max,
        coloring_G=coloring_G,
        coloring_A=coloring_A,
        tol=float(sanity_check_tol),
    )

    return SmootherDomainDiagnostics(
        domain=domain,
        lambda_max_MinvA=float(lambda_max),
        nu=nu,
        row_touch_counts=graphs.row_touch_counts,
        incidence_nnz=int(graphs.E.nnz),
        n_G_rows=int(graphs.E.shape[0]),
        n_domains=int(graphs.E.shape[1]),
        C_G_nnz=int(graphs.C_G.nnz),
        C_A_nnz=int(graphs.C_A.nnz),
        A_edges_subset_of_G_edges=bool(graphs.A_edges_subset_of_G_edges),
        chi_bnd_pyamg_G=coloring_G.chi_bnd_pyamg,
        chi_bnd_pyamg_A=coloring_A.chi_bnd_pyamg,
        chi_exact_G=coloring_G.chi_exact,
        chi_exact_A=coloring_A.chi_exact,
        graphs=graphs,
        coloring_G=coloring_G,
        coloring_A=coloring_A,
        dof_overlap_degree_color_bound=_stored_dof_overlap_degree_bound(level, domain=domain),
        sanity_check_messages=messages,
    )


def compute_smoother_bound_diagnostics_from_level(
    level,
    *,
    domains: tuple[Domain, ...] = ("omega", "OMEGA"),
    coloring_method: ColoringMethod = "MIS",
    compute_exact_coloring: bool = False,
    exact_coloring_max_vertices: int | None = None,
    exact_coloring_max_edges: int | None = None,
    sanity_check_tol: float = 1.0e-8,
) -> list[SmootherDomainDiagnostics]:
    """Compute smoother-side diagnostics for a prepared one-level LS-DD level."""

    out: list[SmootherDomainDiagnostics] = []
    for domain in domains:
        out.append(
            _compute_one_domain(
                level=level,
                domain=domain,
                coloring_method=coloring_method,
                compute_exact_coloring=bool(compute_exact_coloring),
                exact_coloring_max_vertices=exact_coloring_max_vertices,
                exact_coloring_max_edges=exact_coloring_max_edges,
                sanity_check_tol=float(sanity_check_tol),
            )
        )
    return out
