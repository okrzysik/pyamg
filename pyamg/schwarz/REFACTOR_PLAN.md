# LS–AMG–DD (least-squares DD) refactor plan — make the implementation match the math

This document proposes a **pure restructuring / re-organization** plan for the LS–AMG–DD solver implementation in this repo, with the explicit goal of making the code:
- easier to understand and modify over weeks,
- aligned with the math notation in the notes (esp. “LS–AMG–DD: An overview”),
- modular (aggregation / Schwarz / generalized EVP isolated),
- more observable (timings, diagnostics, automation hooks),
while **not changing algorithmic behavior** at this stage.

---

## Scope and constraints

### In-scope (for this refactor phase)
- Carve the current setup pipeline into **small, named helper functions**.
- Add **verbose, math-aligned docstrings** to each helper.
- Add **structured per-level stats and timing** with consistent printing (behind flags).
- Align internal variable names (or add a consistent “notation map” layer) so you stop translating mentally.
- Keep A/B testing available and stable.

### Out-of-scope (explicitly NOT doing yet)
- No new algorithms, no changing coarsening criteria, no new eigen solvers, no changing coarse operator definitions.
- No fundamental changes to defaults or solver behavior (except trivial reordering/renaming consistent with refactor).

### Guardrails
- Keep `pyamg/schwarz/least_squares_dd.py` **unchanged** (reference).
- Do all refactor work in `pyamg/schwarz/least_squares_dd_exp.py` (experimental copy).
- After each refactor step, run the A/B test:
  ```bash
  pytest -q -s ../tests/schwarz/test_lsdd_compare.py
  ```
- Prefer **small commits**: one conceptual extraction per commit.

---

## Current repository state (baseline)

### Solver files
- **Reference**: `pyamg/schwarz/least_squares_dd.py`
- **Experimental**: `pyamg/schwarz/least_squares_dd_exp.py` (copy intended for deep changes)

### Tests
- A/B comparison test: `tests/schwarz/test_lsdd_compare.py`
- Test data: `tests/schwarz/data/B_n*.npz`

The test:
- loads `B`,
- forms `A = B.T @ B`,
- generates random RHS `b`,
- builds each solver (`ref` and `exp`),
- applies preconditioned `fgmres`,
- prints setup/solve metrics and asserts “meaningful progress”.

---

## Notation alignment: math ↔ code

### Core operators
| Math (notes) | Meaning | Code (recommended internal name) |
|---|---|---|
| \(G_\ell\) | Gram/LS factor at level ℓ | `G` (alias for input `B`) |
| \(A_\ell = G_\ell^\top G_\ell\) | SPD normal operator | `A` |
| \(P_\ell\) | prolongation / interpolation | `P` |
| \(G_{\ell+1} = G_\ell P_\ell\) | coarse Gram factor | `G_c` / `B_c` |

### Aggregation / overlap sets
| Math | Meaning | Code |
|---|---|---|
| \(\omega_i^{(\ell)}\) | nonoverlapping aggregate | `omega[i]` |
| \(\Gamma_i^{(\ell)}\) | **interface** set for aggregate i | `Gamma[i]` |
| \(\Omega_i^{(\ell)}=\omega_i\cup\Gamma_i\) | overlapping subdomain | `Omega[i]` |
| \(D_{i,\ell}\) | PoU mask: 1 on ω, 0 on Γ | `PoU[i]` |

### Schwarz / local blocks
| Math | Meaning | Code |
|---|---|---|
| \(A_i = A(\Omega_i,\Omega_i)\) | principal block for RAS/RAS-T | `A_loc[i]` |
| \(\mathcal R_{i,\ell}\) | residual row set touching \(\omega_i\) | `R_rows[i]` |
| \(M_\ell(r)\) | row multiplicity (how many ω touch row r) | `row_mult[r]` |
| \(W_\ell=\mathrm{diag}(1/M_\ell)\) | row weights | `w_row[r] = 1/row_mult[r]` |
| \(\widetilde A_i\) | SPSD splitting energy from rows | `Atilde[i]` |
| \(\widetilde S_{\omega_i}\) | reduced Schur complement used in GEP | `S_tilde_omega[i]` |

### Local generalized eigenproblems (conceptual)
- “Full” view: \(D A_i D \, u = \lambda\, \widetilde A_i\, u\)
- “Reduced” view on ω: \(H_\omega^\top H_\omega \, u_\omega = \lambda\, \widetilde S_{\omega_i}\, u_\omega\)

In code, the goal is to isolate *all* GEP-related assembly + selection logic inside a dedicated helper.

---

## Target architecture (still within one file initially)

### Top-level call graph per level
The level extension should read like your “LS–AMG–DD overview” pipeline:

1. **Strength + aggregation**
   - `C = build_strength(A, strength_spec)`
   - `(AggOp, omega) = build_aggregates(C, aggregate_spec, agg_levels)`

2. **Interface / overlap / PoU**
   - `(Omega, Gamma, PoU, subdomain_ptrs, overlap_rows, row_mult, w_row) = build_overlap_and_weights(G, omega, ...)`

3. **Schwarz blocks**
   - `A_loc = extract_principal_blocks(A, Omega, subdomain_ptrs)`

4. **SPSD splitting from Gram rows**
   - `Atilde = build_splitting_blocks(G, Omega, overlap_rows, w_row, subdomain_ptrs, ...)`

5. **Local spectral coarse space**
   - `(Zhat, nev_stats) = build_local_spectral_basis(A, A_loc, Atilde, omega, Gamma, PoU, ...)`

6. **Assemble P and coarsen**
   - `P = assemble_P(AggOp, omega, Zhat, ...)`
   - `(A_c, G_c) = coarsen_ops(A, G, P)`

7. **Append next level + smoothers**
   - `_append_level(levels, A_c, G_c)`
   - `set_level_smoothers(level, presmoother, postsmoother)`

The refactor is primarily about extracting each step into readable units with tight inputs/outputs and heavy docstrings.

---

## Phase plan: restructure first, then split into modules

### Phase 1 — “Legibility pass” (no file moves; minimal risk)
**Goal:** keep everything in `least_squares_dd_exp.py` but make it readable and instrumented.

#### 1. Add structured timing + stats
Create a `LevelStats` record (dict or dataclass) capturing:
- sizes: `n_fine`, `n_agg`, `avg_omega`, `avg_Omega`, `max_Omega`
- eigen info: `nev_total`, `nev_mean`, `nev_max`, `min_ev_kept`
- complexities: `operator_complexity`, `grid_complexity` (if available)
- timings: `t_strength`, `t_aggregate`, `t_overlap`, `t_extract_A`, `t_outerprod`, `t_gep`, `t_P`, `t_coarsen`

**Printing**
- Add a single function `print_level_summary(stats, level_index, ...)` that respects `print_info`.
- Avoid scattered prints.

#### 2. Add a “notation map” docstring block at the top of the file
A single authoritative mapping table (like above) that explains:
- what `G/B`, `A`, `omega`, `Gamma`, `Omega`, `PoU`, `A_loc`, `Atilde` mean,
- what shapes each object has.

This prevents weeks of mental translation.

#### 3. Extract the GEP + selection into one helper
Create a helper that contains **all** GEP-specific logic and returns only what the caller needs.

Recommended signature (conceptual; keep actual arg set minimal and stable):
```python
def build_local_spectral_basis(
    *,
    A: csr_matrix,
    A_loc_blocks: list[csr_matrix],
    Atilde_blocks: list[csr_matrix],
    omega: list[np.ndarray],
    Gamma: list[np.ndarray],
    PoU: list[np.ndarray],
    kappa: float,
    threshold: float | None,
    min_coarsening: list[int] | int,
    nev: int | None,
    ...
) -> tuple[list[np.ndarray], dict]:
    ...
```

The key is: everything about “reduce to ω”, “assemble Schur”, “solve local EVP”, “keep λ > τ”, “cap/ensure min coarsening” lives here and nowhere else.

#### 4. Extract overlap + PoU construction into one helper
Create `build_overlap_and_pou(...)` responsible for:
- building Γ and Ω from ω,
- building `PoU[i]` aligned with Ω,
- building flattened pointer arrays for C++ kernels (if needed).

This isolates interface bookkeeping.

#### 5. Extract Schwarz smoother wiring into one helper
Create `make_smoother_specs(level, presmoother, postsmoother, ...)` so the main driver does not intermix setup and smoother config.

**Phase 1 definition of done**
- No behavior change (A/B test still passes).
- `_extend_hierarchy` reads as a clear pipeline.
- GEP and overlap logic are each located in a single function.

---

### Phase 2 — “Small-step decomposition” (still one file; more extraction)
**Goal:** isolate your three focus areas cleanly inside the experimental file.

#### Aggregation submodule (in-file helpers)
- `build_strength(A, strength_spec) -> C`
- `build_aggregates(C, aggregate, agg_levels) -> (AggOp, omega)`
- `fill_unaggregated(AggOp, C, ...) -> AggOp` (if needed)

Docstrings should cite exactly the object being constructed:
- What does `AggOp` represent?
- What is the coarsening ratio?
- What assumptions (symmetry/hermitian) are required?

#### Schwarz blocks helpers
- `extract_principal_blocks(A, Omega, ptrs) -> A_loc`
- `build_ras_operators(A_loc, PoU, ...) -> (RAS, RAST)` (if explicit)

Even if you keep the existing smoother machinery, isolate the pieces that create local blocks and PoU.

#### SPSD splitting helpers
- `build_residual_row_sets(G, omega) -> R_rows`
- `compute_row_multiplicity(R_rows, m_rows) -> (row_mult, w_row)`
- `build_splitting_blocks(G, R_rows, Omega, w_row, ...) -> Atilde`

Make it explicit that `Atilde[i]` equals:
\[
G(R_i,\Omega_i)^\top\,W(R_i)\,G(R_i,\Omega_i).
\]

**Phase 2 definition of done**
- Aggregation, overlap/PoU, splitting, and GEP each have “one obvious home”.
- `_extend_hierarchy` is mostly orchestration and stats collection.

---

### Phase 3 — Split into separate modules (optional; after stability)
Once Phase 1–2 are stable, move helpers into dedicated files so you can work on them independently.

Suggested layout under `pyamg/schwarz/lsdd/` (or similar):
- `api.py` — public entrypoints, argument parsing, orchestration
- `aggregation.py` — strength + aggregate building
- `subdomains.py` — omega/Gamma/Omega/PoU construction + pointer formatting
- `local_ops.py` — wrappers for `amg_core.extract_subblocks` and `amg_core.local_outer_product`
- `splitting.py` — row sets + multiplicity + Atilde blocks
- `eigs.py` — reduced Schur construction, local EVP solve, selection rules
- `smoothers.py` — RAS/RAS-T configuration
- `stats.py` — timers, summaries, diagnostics

**Guideline**
- Keep `least_squares_dd_exp.py` as a thin wrapper importing `lsdd.api` so your test imports don’t need to change immediately.

---

## Documentation standards (what to write where)

### Docstring template for each helper
Each extracted helper should start with:
1. **Math definition** (1–4 lines; equations acceptable in comments/docstrings)
2. **Inputs/outputs with shapes**
3. **Side effects** (if it mutates `level` objects)
4. **Performance notes** (which C++ kernels called, expected scaling)

Example outline:
```text
Build the SPSD splitting block Atilde_i = G(R_i,Omega_i)^T W(R_i) G(R_i,Omega_i).

Inputs:
  G: (m x n) CSR, level Gram factor
  Omega[i]: indices into {0,...,n-1}
  R_rows[i]: indices into {0,...,m-1}
  w_row: length-m array with w_row[r] = 1 / multiplicity[r]

Returns:
  Atilde_blocks[i]: (|Omega_i| x |Omega_i|) CSR, SPSD
Notes:
  Uses amg_core.local_outer_product for performance.
```

### “Developer overview” markdown
Keep a short markdown file near the solver code that includes:
- pipeline diagram,
- notation map,
- list of per-level stored artifacts,
- how to run the A/B test.

(You already created a README in `schwarz/`; keep it updated once this plan is implemented.)

---

## Suggested commit sequence (low-risk, reviewable diffs)

1. **Add stats/timing scaffolding**
   - No logic changes; only timers and a summary function.
2. **Add top-of-file notation map**
   - No logic changes; documentation only.
3. **Extract overlap/PoU construction**
   - Replace inline code with `build_overlap_and_pou`.
4. **Extract GEP into `build_local_spectral_basis`**
   - Biggest clarity win; keep behavior identical.
5. **Extract aggregation helpers**
   - `build_strength`, `build_aggregates`, etc.
6. **Extract splitting helpers**
   - `R_rows`, multiplicity, `Atilde`.
7. **Optional: move helpers into modules**
   - Only after stable for at least a week of iteration.

After each commit, run:
```bash
python -m pytest -q -s tests/schwarz/test_lsdd_compare.py
```

---

## Operational notes for future automation (later phases)
Once the refactor is in place, it becomes much easier to:
- add automated parameter sweeps (e.g. `kappa`, `min_coarsening`, `aggregate`),
- add per-level profiling outputs (CSV/JSON),
- isolate and benchmark:
  - aggregation time vs coarsening ratio,
  - eigen-solve time vs block sizes,
  - Schwarz smoothing costs vs overlap size.

But these belong *after* the restructure, once the code has clean seams.

---

## Immediate next step recommendation
Start with the pieces you explicitly care about and that currently create the most cognitive load:
1) `build_overlap_and_pou(...)`
2) `build_local_spectral_basis(...)` (GEP isolation)
3) `extract_principal_blocks(...)` (Schwarz local blocks)
4) `build_splitting_blocks(...)` (SPSD splitting)

These four extractions alone will make the solver setup pipeline line up with your notes and become much easier to reason about (and later improve).

