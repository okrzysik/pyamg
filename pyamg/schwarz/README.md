<!-- ============================================================
FILE: docs/LSDD_REFACTOR_NEXT_CHAT.md
Purpose: copy/paste this entire section into a new chat to continue the refactor
============================================================ -->

# LS–AMG–DD refactor continuation (PyAMG `schwarz/lsdd`)

## Repo / branch context
- Working in a fork of a PyAMG branch that contains an LS–AMG–DD solver under `pyamg/schwarz/`.
- There are two entrypoints:
  - `pyamg/schwarz/least_squares_dd.py` — “reference” implementation (baseline).
  - `pyamg/schwarz/least_squares_dd_exp.py` — “experimental” implementation that has been refactored into modules under `pyamg/schwarz/lsdd/`.
- Tests + data live under:
  - `pyamg/tests/schwarz/test_lsdd_compare.py`
  - `pyamg/tests/schwarz/data/B_n4225.npz`, `B_n16641.npz`, `B_n66049.npz`
- The comparison test constructs `A = B.T @ B` and compares solve/setup metrics between `least_squares_dd` and `least_squares_dd_exp`.

### How to run the guardrail test
```bash
pytest -q -s pyamg/tests/schwarz/test_lsdd_compare.py
PYAMG_LSDD_PRINT_INFO=1 pytest -q -s pyamg/tests/schwarz/test_lsdd_compare.py -x
```

## Current code layout (post-refactor)
Directory:
- `pyamg/schwarz/`
  - `least_squares_dd.py` (baseline)
  - `least_squares_dd_exp.py` (thin wrapper; hierarchy extension happens in `lsdd/hierarchy.py`)
  - `lsdd/` (modularized implementation)
    - `types.py` — dataclasses used across the codebase (`Subdomains`, `LocalBlocks`, `EigenInfo`)
    - `aggregation.py` — filtering, strength-of-connection, aggregation operator (AggOp), and level init
    - `subdomains.py` — build `omega/OMEGA/GAMMA`, row sets `R_rows`, PoU masks; stores PoU check error
    - `local_ops.py` — extract dense `A[OMEGA_i,OMEGA_i]` blocks; build local SPSD splitting blocks from rows of `B`
    - `eigs.py` — per-aggregate dense GEP solve + eigenvector selection + triplets for P assembly
    - `smoothers.py` — map shorthand smoother names to PyAMG smoother specs, uses cached `sub.PoU_flat`
    - `stats.py` — *only* place where printing happens; accumulates timings/diagnostics and prints summaries
    - `hierarchy.py` — orchestration: filter → strength → aggregation → subdomains → dense blocks → outer products → GEP → assemble P → coarsen operators → append next level
  - `BTB.cpp` (compiled helper)
  - `REFACTOR_PLAN.md`

## Notation policy (already pushed “everywhere”)
**Capital Greek is uppercase in code:**
- `omega_i`  → `sub.omega[i]`
- `OMEGA_i`  → `sub.OMEGA[i]`
- `GAMMA_i`  → `sub.GAMMA[i]`
- `PoU[i]` is a 0/1 mask aligned with `OMEGA_i` ordering: 1 on omega, 0 on GAMMA.
- The coarse operator is built via least-squares propagation: `B_{ℓ+1} = B_ℓ P_ℓ`, `A_{ℓ+1} = (B_{ℓ+1})^T B_{ℓ+1}`.

### “One source of truth” containers on each level
Each level stores:
- `level.sub` : `Subdomains`
- `level.blocks` : `LocalBlocks`
- `level.eigs` : `EigenInfo`

## Printing / diagnostics policy (already implemented)
- No `print()` outside `lsdd/stats.py`.
- Diagnostics from filtering, PoU check, eigen selection, RAS profiling, and timings are collected into `LsddLevelStats.extra` and printed by `stats.py`.

## Known ambiguity that needs fixing next
- Plan: rename everywhere:
  - use `n_fine = A.shape[0]` for fine dimension
  - use `m_rows = B.shape[0]`, `n_cols = B.shape[1]` when discussing B/A

## Filtering option types
Filtering is currently implemented via `pyamg.util.utils.filter_matrix_rows`, so `theta` should be treated as float:
- Prefer: `filteringA: tuple[bool, float] | None`, `filteringB: tuple[bool, float] | None`
- Filtering diagnostics should be recorded (nnz before/after for A/B/BT) and printed via `stats.py`.

## Refactor goals for the next chat (no “algorithm redesign” yet)
Main objectives:
2. Tighten typing internally:
   - Introduce a `Protocol` (e.g. `LSDDLevel`) to type `level` in internal functions instead of `Any`.
   - Use concrete return types where feasible.
3. Consolidate parameter handling:
   - Consider a `LSDDConfig` dataclass (optional; can be staged).
4. Continue improving docs:
   - Ensure *every* file and *every* function has a meaningful docstring.
   - Keep docstrings factual (no “refactor-only” or “doesn’t change the math” commentary).

## How I want you (the assistant) to work with me
- Provide small, mechanical steps: “replace this block in this function” / “rename these symbols”.
- Keep changes localized; avoid full-file dumps unless requested.
- Always update docstrings + type annotations where touched.
- After each chunk, suggest the exact command(s) to run (`pytest` first; optionally `ruff`/formatters).
- If a test fails, identify the remaining old symbol and point to the exact file/line pattern to fix.

---

<!-- ============================================================
FILE: pyamg/schwarz/README.md
Purpose: README describing current code outlay
============================================================ -->

# PyAMG Schwarz: LS–AMG–DD (least-squares AMG domain decomposition)

This directory contains a least-squares algebraic multigrid + domain decomposition solver (“LS–AMG–DD”) implemented within PyAMG’s `schwarz` module. The solver builds a multilevel hierarchy for an SPD normal-equations operator
\[
A = B^\top B,
\]
where `B` is a sparse (typically rectangular) least-squares factor.

## Quick start

### Run the comparison test (baseline vs experimental)
```bash
pytest -q -s pyamg/tests/schwarz/test_lsdd_compare.py
PYAMG_LSDD_PRINT_INFO=1 pytest -q -s pyamg/tests/schwarz/test_lsdd_compare.py -x
```

### Data used by the test
The test loads `B` factors from:
- `pyamg/tests/schwarz/data/B_n4225.npz`
- `pyamg/tests/schwarz/data/B_n16641.npz`
- `pyamg/tests/schwarz/data/B_n66049.npz`

It then forms `A = B.T @ B` and solves `Ax=b` using FGMRES preconditioned by each multilevel solver, printing setup/solve metrics.

## Files

### Entrypoints
- `least_squares_dd.py`
  - Baseline/reference LS–AMG–DD implementation.
  - Useful as a stable benchmark when experimenting.

- `least_squares_dd_exp.py`
  - Experimental/refactored entrypoint.
  - Thin wrapper that delegates hierarchy construction to `lsdd/hierarchy.py`.
  - Intended for research iteration (instrumentation, alternative policies, automation).

### Modular implementation (`lsdd/`)
The experimental solver is split into focused modules:

- `lsdd/types.py`
  - Dataclasses that define the per-level state used across the pipeline:
    - `Subdomains`: stores per-aggregate index sets (`omega`, `OMEGA`, `GAMMA`), row sets (`R_rows`), PoU masks, and overlap diagnostics.
    - `LocalBlocks`: stores flattened dense blocks `A[OMEGA_i,OMEGA_i]` and local SPSD splitting blocks `\\tilde{A}_i`.
    - `EigenInfo`: stores eigen-selection parameters and per-aggregate kept counts.

- `lsdd/aggregation.py`
  - Optional filtering of operators.
  - Strength-of-connection construction.
  - Aggregation operator `AggOp` construction.
  - Level initialization: allocates `level.sub`, `level.blocks`, `level.eigs`.

- `lsdd/subdomains.py`
  - Builds per-aggregate sets:
    - `omega_i`: nonoverlapping aggregate DOFs (from `AggOp`)
    - `OMEGA_i`: one-ring overlap set (from adjacency of `A`)
    - `GAMMA_i`: interface set `OMEGA_i \\ omega_i`
    - `R_rows_i`: set of rows in `B` used to build local outer products (via adjacency of `BT`)
  - Builds partition-of-unity masks `PoU[i]` aligned with `OMEGA_i` ordering.
  - Stores a PoU consistency error for diagnostics (printed via `stats.py`).

- `lsdd/local_ops.py`
  - Extracts dense principal blocks `A[OMEGA_i, OMEGA_i]` into flattened storage.
  - Constructs local SPSD splitting blocks `\\tilde{A}_i` using a compiled kernel over rows of `B`.

- `lsdd/eigs.py`
  - For each aggregate, builds a dense generalized eigenproblem on the `omega_i` subset (using a Schur complement from the splitting block).
  - Selects eigenvectors (fixed `nev` or by threshold) and assembles prolongation `P` via triplets.

- `lsdd/smoothers.py`
  - Converts shorthand smoother names (`"ras"`, `"rasT"`, `"asm"`, `"msm"`, `None`) into PyAMG smoother specs.
  - Uses cached `sub.PoU_flat` to avoid repeated concatenation of PoU masks.

- `lsdd/stats.py`
  - Centralized timing + diagnostics collection.
  - The *only* place that prints setup summaries.
  - Reports: sizes (`|omega|`, `|GAMMA|`, `|OMEGA|`), eigen selection, threshold, PoU error, filtering nnz changes, and timed stages.

- `lsdd/hierarchy.py`
  - Orchestrates one level extension:
    1. (optional) filter ops
    2. build strength matrix
    3. build aggregates
    4. build subdomains (`omega/OMEGA/GAMMA`, `R_rows`, PoU)
    5. extract dense blocks
    6. build splitting blocks
    7. solve local GEPs / select vectors
    8. assemble prolongation `P` and restriction `R`
    9. coarsen operators via least-squares propagation
    10. append next level

## Naming / notation conventions (code ↔ math)
- `sub.omega[i]`  ↔ \( \omega_i \)
- `sub.OMEGA[i]`  ↔ \( \Omega_i \)
- `sub.GAMMA[i]`  ↔ \( \Gamma_i \)
- `sub.PoU[i]`    ↔ boolean mask over \( \Omega_i \) ordering (1 on \( \omega_i \), 0 on \( \Gamma_i \))
- `blocks.submatrices[...]` ↔ flattened dense blocks \( A_{\Omega\Omega}^{(i)} \)
- `blocks.auxiliary[...]`   ↔ flattened dense splitting blocks \( \\tilde{A}_{\Omega\Omega}^{(i)} \)
- `eigs.nev[i]`   ↔ number of coarse vectors kept on aggregate \( i \)

## Common development workflow
- Make changes only in the experimental path first (`least_squares_dd_exp.py` + `lsdd/`).
- Keep the baseline solver intact for comparison.
- Use `pyamg/tests/schwarz/test_lsdd_compare.py` as a guardrail for performance regressions and correctness.

## Planned next cleanup
- Tighten typing via a `Protocol` for the level object to reduce reliance on `Any`.
- Consider consolidating solver options into a config dataclass for reproducible experiments.
