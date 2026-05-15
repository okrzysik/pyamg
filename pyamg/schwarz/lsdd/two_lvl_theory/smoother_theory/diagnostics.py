"""Smoother-side bounds and coloring diagnostics from LS-DD level data."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.sparse import csr_array, issparse

from pyamg.graph import vertex_coloring
from pyamg.relaxation import relaxation as _relaxation
from pyamg.relaxation.smoothing import rho_additive_schwarz_A as _rho_additive_schwarz_A

from ...smoothers import lsdd_make_smoother_spec

Domain = Literal["omega", "OMEGA"]
ExactColoringMode = Literal["never", "if_gap", "always"]


@dataclass(slots=True, frozen=True)
class SmootherDomainBoundDiagnostics:
    """Per-domain smoother-side bounds and coloring diagnostics."""

    domain: Domain
    lambda_max_MinvA: float
    nu: int
    row_touch_counts: np.ndarray
    incidence_nnz: int
    n_rows: int
    n_aggs: int
    interaction_nnz: int
    chi_pyamg: int | None
    pyamg_coloring_method: str | None
    chi_exact: int | None
    chi_exact_status: str
    chi_exact_reason: str | None
    chi_exact_seconds: float | None
    chi_certified_by_pyamg: bool
    dof_overlap_degree_plus_one_bound: float | None


def _as_csr_pattern(M, *, name: str) -> csr_array:
    """Convert sparse-like input to a sorted CSR pattern matrix."""
    if not issparse(M):
        raise ValueError(f"Expected sparse {name}, got {type(M)!r}")
    Mcsr = M.tocsr(copy=True)
    Mcsr.eliminate_zeros()
    Mcsr.sort_indices()
    if Mcsr.nnz:
        Mcsr.data[:] = 1.0
    return Mcsr


def compute_row_domain_incidence_from_level(level, *, domain: Domain) -> csr_array:
    """Return the row-by-domain incidence matrix for one prepared LS-DD level."""
    B = getattr(level, "B", None)
    if B is None:
        raise ValueError("Expected level.B on a prepared LS-DD level")

    if domain == "omega":
        D = getattr(level, "AggOp", None)
        if D is None:
            raise ValueError("Expected level.AggOp for domain='omega'")
    elif domain == "OMEGA":
        sub = getattr(level, "sub", None)
        D = getattr(sub, "nodes_vs_subdomains", None)
        if D is None:
            raise ValueError(
                "Expected level.sub.nodes_vs_subdomains for domain='OMEGA'; "
                "build overlap data first"
            )
    else:
        raise ValueError(f"Unsupported domain {domain!r}; expected 'omega' or 'OMEGA'")

    Bpat = _as_csr_pattern(B, name="level.B")
    Dpat = _as_csr_pattern(D, name=f"domain incidence ({domain})")
    if int(Bpat.shape[1]) != int(Dpat.shape[0]):
        raise ValueError(
            "Shape mismatch for row-domain incidence: "
            f"B has shape {Bpat.shape}, domain incidence has shape {Dpat.shape}"
        )

    E = (Bpat @ Dpat).tocsr()
    E.eliminate_zeros()
    E.sort_indices()
    if E.nnz:
        E.data[:] = 1.0
    return E


def build_row_interaction_matrix(E: csr_array) -> csr_array:
    """Build the boolean off-diagonal pattern of ``E.T @ E``."""
    C = (E.T @ E).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    if C.nnz:
        C.data[:] = 1.0
    C = C.maximum(C.T).tocsr()
    C.eliminate_zeros()
    return C


def compute_additive_schwarz_lambda_max_from_level(
    level,
    *,
    domain: Domain,
) -> float:
    """Compute ``lambda_max(M^{-1}A)`` for additive Schwarz on one domain layout."""
    Acsr = level.A.tocsr()
    Acsr.sort_indices()
    asm_spec = lsdd_make_smoother_spec(level=level, smoother=("asm", {"domain": domain}))
    if asm_spec is None:  # pragma: no cover
        raise ValueError("Failed to build ASM smoother specification")
    _, kwargs = asm_spec
    subdomain = np.asarray(kwargs["subdomain"], dtype=np.int32)
    subdomain_ptr = np.asarray(kwargs["subdomain_ptr"], dtype=np.int32)

    if hasattr(Acsr, "schwarz_parameters"):
        delattr(Acsr, "schwarz_parameters")
    inv_subblock = None
    inv_subblock_ptr = None
    subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr = _relaxation.schwarz_parameters(
        Acsr, subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr
    )
    rho = float(
        _rho_additive_schwarz_A(
            Acsr, subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr
        )
    )
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"Expected positive finite lambda_max(M^-1 A), got {rho!r}")
    return rho


def _compute_exact_chromatic_number_gcol(C: csr_array) -> int:
    """Compute exact chromatic number with GCol on a NetworkX graph."""
    try:
        import networkx as nx  # type: ignore
        import gcol  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Exact coloring requires optional dependencies gcol and networkx") from exc

    coo = C.tocoo()
    G_nx = nx.Graph()
    n = int(C.shape[0])
    G_nx.add_nodes_from(range(n))
    G_nx.add_edges_from(
        (int(i), int(j)) for i, j in zip(coo.row, coo.col, strict=False) if int(i) < int(j)
    )
    return int(gcol.chromatic_number(G_nx))


def _compute_exact_coloring(
    *,
    C: csr_array,
    nu: int,
    chi_pyamg: int | None,
    exact_coloring: ExactColoringMode,
    exact_coloring_max_vertices: int | None,
    exact_coloring_max_edges: int | None,
) -> tuple[int | None, str, str | None, float | None, bool]:
    """Evaluate exact-coloring policy and return result tuple."""
    if chi_pyamg is not None and chi_pyamg < nu:
        return (
            None,
            "invalid_coloring_lower_than_nu",
            f"chi_pyamg={int(chi_pyamg)} < nu={int(nu)}",
            None,
            False,
        )

    # Policy handling:
    # - never: never run exact coloring; may still certify when chi_pyamg == nu.
    # - if_gap: certify when chi_pyamg == nu, otherwise run exact coloring.
    # - always: always run exact coloring (subject to size/dependency guards).
    if exact_coloring == "never":
        if chi_pyamg is not None and chi_pyamg == nu:
            return int(nu), "certified_by_pyamg", None, None, True
        return None, "not_requested", None, None, False
    if exact_coloring == "if_gap" and chi_pyamg is not None and chi_pyamg == nu:
        return int(nu), "certified_by_pyamg", None, None, True

    n_vertices = int(C.shape[0])
    n_edges = int(C.nnz // 2)
    if exact_coloring_max_vertices is not None and n_vertices > int(exact_coloring_max_vertices):
        return None, "skipped_size", f"vertices={n_vertices} > {int(exact_coloring_max_vertices)}", None, False
    if exact_coloring_max_edges is not None and n_edges > int(exact_coloring_max_edges):
        return None, "skipped_size", f"edges={n_edges} > {int(exact_coloring_max_edges)}", None, False

    t0 = perf_counter()
    try:
        chi = _compute_exact_chromatic_number_gcol(C)
    except ImportError as exc:
        return None, "missing_dependency", str(exc), None, False
    except Exception as exc:  # pragma: no cover
        return None, "failed", str(exc), None, False
    dt = perf_counter() - t0
    return int(chi), "computed", None, float(dt), False


def _compute_domain_diagnostics(
    *,
    level,
    domain: Domain,
    pyamg_coloring_method: Literal["MIS", "JP", "LDF"],
    exact_coloring: ExactColoringMode,
    exact_coloring_max_vertices: int | None,
    exact_coloring_max_edges: int | None,
) -> SmootherDomainBoundDiagnostics:
    """Compute smoother-side diagnostics for one domain."""
    E = compute_row_domain_incidence_from_level(level, domain=domain)
    row_touch_counts = np.diff(E.indptr).astype(np.int32, copy=False)
    nu = int(row_touch_counts.max()) if row_touch_counts.size else 0

    C = build_row_interaction_matrix(E)
    colors = vertex_coloring(C, method=str(pyamg_coloring_method))
    chi_pyamg = int(colors.max() + 1) if colors.size else 0

    chi_exact, chi_status, chi_reason, chi_seconds, certified = _compute_exact_coloring(
        C=C,
        nu=nu,
        chi_pyamg=chi_pyamg,
        exact_coloring=exact_coloring,
        exact_coloring_max_vertices=exact_coloring_max_vertices,
        exact_coloring_max_edges=exact_coloring_max_edges,
    )

    lambda_max = compute_additive_schwarz_lambda_max_from_level(level, domain=domain)

    dof_overlap_degree_plus_one_bound = None
    if domain == "OMEGA":
        color_bound = getattr(level.sub, "number_of_colors", None)
        dof_overlap_degree_plus_one_bound = (
            float(color_bound) if color_bound is not None else None
        )

    return SmootherDomainBoundDiagnostics(
        domain=domain,
        lambda_max_MinvA=float(lambda_max),
        nu=nu,
        row_touch_counts=row_touch_counts,
        incidence_nnz=int(E.nnz),
        n_rows=int(E.shape[0]),
        n_aggs=int(E.shape[1]),
        interaction_nnz=int(C.nnz),
        chi_pyamg=chi_pyamg,
        pyamg_coloring_method=str(pyamg_coloring_method),
        chi_exact=chi_exact,
        chi_exact_status=chi_status,
        chi_exact_reason=chi_reason,
        chi_exact_seconds=chi_seconds,
        chi_certified_by_pyamg=bool(certified),
        dof_overlap_degree_plus_one_bound=dof_overlap_degree_plus_one_bound,
    )


def compute_smoother_bound_diagnostics_from_level(
    level,
    *,
    domains: tuple[Domain, ...] = ("omega", "OMEGA"),
    pyamg_coloring_method: Literal["MIS", "JP", "LDF"] = "MIS",
    exact_coloring: ExactColoringMode = "if_gap",
    exact_coloring_max_vertices: int | None = None,
    exact_coloring_max_edges: int | None = None,
) -> list[SmootherDomainBoundDiagnostics]:
    """Compute smoother-side diagnostics for one prepared LS-DD level."""
    result: list[SmootherDomainBoundDiagnostics] = []
    for domain in domains:
        result.append(
            _compute_domain_diagnostics(
                level=level,
                domain=domain,
                pyamg_coloring_method=pyamg_coloring_method,
                exact_coloring=exact_coloring,
                exact_coloring_max_vertices=exact_coloring_max_vertices,
                exact_coloring_max_edges=exact_coloring_max_edges,
            )
        )
    return result

