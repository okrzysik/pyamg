# `smoother_theory`: Smoother-Side LS-DD Diagnostics

This subpackage contains **setup/sparsity/smoother diagnostics** for LS-DD levels, separated from the two-level solve diagnostics (`q_obs`, `K_obs`, `tau_max`, `W_J`).

## Purpose

Given a prepared LS-DD level (or a setup-only context), compute domain-wise quantities for:

- nonoverlapping block-Jacobi domain (`omega`)
- overlapping additive-Schwarz domain (`OMEGA`)

Core outputs include:

- `lambda_max(M^{-1}A)` (Schwarz/additive-Schwarz spectral factor)
- `nu` (row-touching multiplicity bound)
- `chi_exact` (exact chromatic number, when available/computed)
- `chi_pyamg` (PyAMG coloring count upper bound)
- incidence/interactions nnz diagnostics
- `dof_overlap_degree_color_bound` for `OMEGA` (`level.sub.number_of_colors`)

## Public API

Exported from `__init__.py`:

- `build_one_level_smoother_context(...)`
  - setup-only level builder (aggregation + overlap data + flattened subdomains)
  - does **not** build coarse interpolation or local GEP-based coarse spaces

- `compute_row_domain_incidence_from_level(level, domain=...)`
- `build_row_interaction_matrix(E)`
- `compute_additive_schwarz_lambda_max_from_level(level, domain=...)`
- `compute_smoother_bound_diagnostics_from_level(...)`
- `SmootherDomainBoundDiagnostics`

## TPL / Dependency Notes

### Required for standard diagnostics

No new hard dependency beyond the existing PyAMG LS-DD stack:

- NumPy / SciPy
- PyAMG core + LS-DD internals

These are used for:

- sparse incidence construction
- interaction matrix construction
- PyAMG coloring (`pyamg.graph.vertex_coloring`)
- Schwarz spectral-radius routine

### Optional for exact coloring

Exact coloring uses third-party libraries (TPLs):

- `networkx`
- `gcol`

They are imported **lazily** only when exact coloring is requested and needed. They are not imported at module import time.

How they are used:

1. Build sparse interaction matrix `C` from row-domain incidence.
2. Convert `C` to a NetworkX graph.
3. Call `gcol.chromatic_number(G)` for exact chromatic number.

If unavailable, exact-coloring status is reported as `missing_dependency`; the rest of the diagnostics still run.

## Exact-Coloring Policy

Supported policy values:

- `never`: do not call exact coloring
- `if_gap`: call exact coloring only when `chi_pyamg > nu`
- `always`: call exact coloring whenever size guards permit

Certification shortcut:

- If `chi_pyamg == nu`, then exact equality is certified (`nu <= chi_exact <= chi_pyamg`), so `chi_exact` is set to `nu` with status `certified_by_pyamg`.

## Installation Reminder for Optional Exact Coloring

Install optional TPLs in the same Python environment as PyAMG:

```bash
python -m pip install networkx gcol
```

## Scope Boundary

This package is intentionally smoother-side. It does **not** compute two-level LS-DD metrics (observed two-grid constants, threshold/coarse-space metrics, or `W_J`).

