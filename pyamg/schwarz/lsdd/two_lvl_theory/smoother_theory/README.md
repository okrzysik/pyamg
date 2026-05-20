# Smoother-theory diagnostics

This folder contains diagnostics for the one-level smoother quantities used in
LS-DD convergence-theory experiments.  The code is intentionally split into two
files so that the graph algebra is separate from the high-level smoother
constants.

## Files

- `smoother_graphs.py` builds the domain incidence matrices, interaction
  graphs, and graph colorings.
- `diagnostics.py` computes the smoother spectral quantity and packages the
  scalar diagnostics for each requested domain layout.
- `setup.py` remains responsible for building the one-level LS-DD context.

The intended call path is:

1. build the one-level context with `build_one_level_smoother_context`,
2. call `compute_smoother_bound_diagnostics_from_level`,
3. optionally use the returned graph matrices and color arrays for plotting.

## Domain layouts

The diagnostics support two domain layouts.

- `omega`: the nonoverlapping aggregate domains, using `level.AggOp` as the
  DOF-by-domain incidence matrix.
- `OMEGA`: the overlapping Schwarz domains, using
  `level.sub.nodes_vs_subdomains` as the DOF-by-domain incidence matrix.

In both cases the code calls this incidence matrix `D`.  The entry `D[p, i]`
is one when global degree of freedom `p` belongs to domain `i`.

## The matrices `E`, `C_G`, and `C_A`

Let `A` be the assembled SPD matrix and let `G` be the stored LS/Gram factor
`level.B`, so that the intended relation is `A = G.T @ G`.

The matrix `E` is the row-domain incidence matrix

```text
E = pattern(G) @ D,
```

interpreted Booleanly.  Thus `E[j, i] = 1` means that row `j` of `G` touches
at least one DOF in domain `i`.  The multiplicity quantity reported as `nu` is

```text
nu = max_j sum_i E[j, i].
```

The code builds two domain interaction graphs.

### `C_G`

```text
C_G = offdiag_pattern(E.T @ E).
```

This graph connects two domains when some row of `G` touches both of them.  It
is the graph naturally associated with the row-wise quantity `nu`.  If one row
of `G` touches `r` domains, those `r` domains are pairwise adjacent in `C_G`,
so `nu <= chi(C_G)`.

### `C_A`

```text
C_A = offdiag_pattern(D.T @ pattern(A) @ D).
```

This graph connects two domains when the assembled matrix `A` has a stored
nonzero coupling between DOFs in the two domains.  This is the graph used by
the standard additive Schwarz coloring argument.  For exact additive Schwarz,
`lambda_max(M_AS^{-1} A) <= chi(C_A)`.

For consistent `A = G.T @ G` data, `C_A` should be a subgraph of `C_G`.
They coincide when no row-outer-product assembly cancellations remove
block-level interactions.

## Coloring quantities

The PyAMG coloring routine gives an upper bound on the chromatic number.  The
field names therefore use `chi_bnd_pyamg_*`, not `chi_pyamg_*`.

For each graph, the diagnostics may contain:

- `chi_bnd_pyamg_G`: PyAMG upper-bound color count for `C_G`.
- `chi_bnd_pyamg_A`: PyAMG upper-bound color count for `C_A`.
- `chi_exact_G`: exact chromatic number of `C_G`, when requested.
- `chi_exact_A`: exact chromatic number of `C_A`, when requested.

Exact coloring is controlled by a boolean flag, `compute_exact_coloring`.  The
old policy distinction between `never`, `if_gap`, and `always` is intentionally
removed.  If exact coloring is requested and the graph exceeds an optional size
guard, the code raises a clear error.

## Sanity checks

The diagnostics return `sanity_check_messages`.  An empty tuple means all
checks passed.  The checks are postconditions only; they do not replace exact
coloring by shortcut certification.

The main checks are:

- `C_A` should be a subgraph of `C_G`.
- PyAMG and exact colorings, when present, should be valid graph colorings.
- If `chi_exact_G` is present, then `nu <= chi_exact_G`.
- If both exact values are present, then `chi_exact_A <= chi_exact_G`.
- If `chi_exact_A` is present, then
  `lambda_max(M_AS^{-1} A) <= chi_exact_A` up to the requested tolerance.

## Plotting

The graph matrices and coloring arrays are returned in the diagnostics object.
Plotting should usually happen in the driver repository rather than inside this
PyAMG module, because the driver side has access to mesh coordinates and
aggregate centroids.

The utility `interaction_matrix_to_networkx(C)` converts either `C_G` or `C_A`
to a NetworkX graph.  A plotting script can then draw nodes at aggregate
centroids and color them using either `colors_bnd_pyamg` or `colors_exact` from
the corresponding `GraphColoringResult`.
