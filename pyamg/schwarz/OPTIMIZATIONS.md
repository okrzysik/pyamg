# OPTIMIZATIONS.md — LS–AMG–DD (PyAMG `schwarz/lsdd`) performance notes

This file documents **performance-motivated implementation choices** (and a few explored-but-rejected ideas) in the refactored
LS–AMG–DD experimental path (`least_squares_dd_exp.py` + `lsdd/`). The goal is to make “non-obvious” design decisions easy to
understand when revisiting the code later.

## Scope and assumptions

- The experimental solver is designed for **real-valued** problems and a least-squares hierarchy:
  - Fine level: `A = B.T @ B` (SPD), where `B` is typically **tall** (`m_rows >> n_cols`).
  - Coarse propagation: `B_{ℓ+1} = B_ℓ P_ℓ` and the SPD operator is consistent with the chosen coarsening strategy.
- “Performance” is measured with:
  - `pyamg/tests/schwarz/bench_lsdd.py --per-level` timings
  - The guardrail test: `pyamg/tests/schwarz/test_lsdd_compare.py`

Throughout, **every optimization must preserve correctness** (via tests) and should be justified by profiling/timing data.

---

## Principles used during optimization

1. **Do not guess**: split high-level timers into sub-timers until one or two kernels dominate.
2. **Avoid touching all rows of B unless necessary**: with `m_rows >> n_cols`, operations like `B @ P` can dominate setup.
3. **Cache derived formats deliberately**: conversions like CSR↔CSC are not “free”; control where they happen so they don’t show up
   inside hot loops.
4. **Prefer numerically-appropriate dense kernels**: if a local block is SPD, use SPD factorizations (Cholesky), not generic solvers.

---

## TODOs:
-

## Optimization 1 — Form the coarse SPD operator via RAP: `A_c = R @ A @ P`

### What changed
Instead of forming the next-level SPD operator from the propagated factor:
- **Old (slow in practice at larger levels)**:
  - `B_c = B @ P`
  - `A_c = (B_c).T @ (B_c)`
- **New**:
  - `A_c = R @ A @ P`  (Galerkin triple product)
  - `B_c` is still propagated as needed (see Optimization 2).

### Why it helps
- The “Gram-from-factor” path inflates the cost because it forces large sparse-sparse multiplies involving the tall `B`.
- The RAP path tends to be faster in SciPy sparse algebra, especially once coarse dimensions shrink.

### How validated
- Re-run guardrail comparison test and `bench_lsdd.py --per-level`.
- Coarsen timings improved notably on larger problems when switching to RAP for `A_c`.

---

## Optimization 2 — Do **not** propagate `B` (or any adjacency derived from it) on the final level

### What changed
When extending the hierarchy from level `ℓ` to `ℓ+1`, we always build:
- `A_{ℓ+1}` (required for the coarse solve).

But we **only build** the tall least-squares factor on the next level:
- `B_{ℓ+1} = B_ℓ @ P_ℓ`
…if and only if we will **coarsen further** from level `ℓ+1`.

In other words: if the next iteration would stop (e.g. due to `max_levels`, `max_coarse`, or `max_density`), then `B_{ℓ+1}` is
never used and should not be formed.

### Why it helps
- Forming `B_{ℓ+1}` is a sparse-sparse multiplication that touches essentially all `nnz(B)` (because `B` is tall).
- On the last coarsening step, this cost is pure overhead: the coarse solve uses `A` only.

### How validated
- Timers showed level-2 coarsen dropping from seconds to ~tens of milliseconds once `B_{ℓ+1}` was skipped.
- Setup time reduced materially on the largest benchmark.

---

## Optimization 3 — Schwarz/PCM inversion: use Cholesky (SPD) instead of generic least-squares (`gelss`)

### What changed
Local Schwarz block inverses operate on **SPD** dense principal submatrices (PCM blocks):
- Replace any generic dense solver (`gelss` / least-squares) with **Cholesky-based** factorization and inversion/solve.

Typical dense path:
- `potrf` (Cholesky factorization)
- `potri` (invert from Cholesky factor) or use triangular solves for applications
- Symmetrization as needed (to counteract numerical drift)

### Why it helps
- `gelss` (SVD-based least-squares) is vastly more expensive than SPD-specific kernels.
- Cholesky is the correct tool for SPD blocks and improves both runtime and numerical robustness.

### How validated
- Benchmarks show “invert” time dominated by `potrf/potri`, with `gelss` eliminated.
- Guardrail tests confirm identical solver behavior.

---

## Optimization 4 — Coarsen-stage breakdown instrumentation (profiling hygiene)

### What changed
The coarse operator formation (`coarsen`) was split into sub-timers so we can see exactly where time goes:
- `A_P`, `R_AP` for RAP
- `B_P` for `B @ P` (when needed)
- `BT_build` (transpose/format conversion) and `sort`

### Why it helps
This instrumentation revealed that, on large problems, `B_P` dominated coarsen time on intermediate levels, and that attempting to
replace it with alternative propagation (e.g. via `R @ BT`) was counterproductive.

### How validated
- Sub-timers directly guided which changes helped and which were dead ends.

---

## Optimization 5 — Adjacency for overlap construction: prefer `B_csc` over storing `BT`

Since we do not compute 
Overlap construction needs: for each column/DOF `j`, the set of B-rows touching `j`.
This is **column adjacency** of `B`.

### Design options considered

**A) Store BT in CSR (legacy-style)**
- `BT = B.T.tocsr()` (often followed by `BT.sort_indices()`).
- Pros: existing code uses CSR row slicing on `BT`.
- Cons: `B.T.tocsr()` is effectively **two conversions** (transpose → CSC-like, then CSC→CSR) and showed up as a nontrivial time
  component (`BT_build`).

**B) Store `B_csc = B.tocsc()` once per level (preferred)**
- Pros:
  - One conversion (CSR→CSC).
  - Direct access to column adjacency:
    - rows touching dof `j` are `B_csc.indices[B_csc.indptr[j]:B_csc.indptr[j+1]]`.
- Cons:
  - Additional memory for a second sparse representation (similar to storing `BT`).

**C) Build `B_csc` locally inside overlap each time**
- This was tried and **rejected** because the CSC conversion cost moved into the `overlap` timer and increased total setup time.

### Current choice and guidance
- Use **Option B**: build `B_csc` **once per level** after the last mutation of `B` (i.e., after in-place filtering), then reuse it
  for overlap adjacency.
- Do **not** build `B_csc` inside overlap hot loops (avoid Option C).


- I'm not really convinced of the above. The cleanest thing seems to be to just create a local CSC copy of B in the outer-product routine... Timing was mixed on that, but not sure how it could really be worse, and it lessens out memory overhead and simplifies the code.

### Sorting
- For adjacency (set unions), **sorted indices are not required**. Sorting can be skipped unless another routine depends on it.

---

## Explored but rejected ideas (kept here to avoid re-trying)

### 1) Convert `P` to CSC inside coarsen (`P_csc = P.tocsc()` before `B @ P`)
- Result: negligible improvement, because the dominant cost was the sparse-sparse multiplication itself, not the storage mismatch.
- Kept: harmless, but not a primary lever for runtime.

### 2) Propagate `BT_c` via `BT_c = R @ BT` to avoid `B @ P`
- Result: significantly worse (large time in `R_BT` and subsequent sorting/transposes), because it forms a wide product of shape
  `n_coarse × m_rows` and incurred heavy reindex/sort overhead.
- Conclusion: for this implementation, computing `B_c = B @ P` is the least-bad way to materialize the next tall factor.

### 3) Build CSC inside overlap instead of caching per level
- Result: overlap time increased substantially (conversion cost moved into a hot stage), worsening total setup.


---

## Practical “do this first” workflow for future performance work

1. Use `bench_lsdd.py --per-level` to identify the dominant stage on the largest problem (`B_n263169`).
2. If a stage dominates, split it into sub-timers until the expensive kernel is obvious.
3. Prefer optimizations that:
   - reduce (or eliminate) `B @ P` calls,
   - reduce sparse format churn (`.T.tocsr()`, repeated `.tocsc()`),
   - or reduce per-aggregate dense work (`gep`, local factorizations).

At the current state (after the optimizations above), the large remaining hotspots are typically:
- deep-level `outerprod`
- per-aggregate `gep`
- intermediate-level `B_P = B @ P` (inherently expensive given tall `B`)

Those require either more careful sparse-kernel handling (dtype/contiguity/pointer packing) or algorithmic changes (e.g. reducing
the need to materialize `B` on intermediate levels).


## GEP performance: where the time goes, what we tried, and next steps

### What dominates runtime inside `gep`
Instrumentation of the per-aggregate generalized eigenproblem (GEP) shows that the runtime is dominated by **dense linear algebra**,
not Python bookkeeping:

- The **`eigh` call** (dense generalized symmetric EVP) is the main cost on every level:
  - Typical breakdown on `B_n263169`:
    - level 0: `eigh ≈ 1.0s` out of `gep ≈ 1.6–1.7s`
    - level 1: `eigh ≈ 1.25s` out of `gep ≈ 1.7s`
    - level 2: `eigh ≈ 1.35–1.40s` out of `gep ≈ 1.6–1.7s`
- The **Schur complement construction** (forming `S` on `omega` by eliminating `GAMMA`) is the clear #2 cost:
  - `schur ≈ 0.22–0.37s` depending on level.
- Everything else is second-order:
  - `triplets` / list appends are small (tens of ms at most on level 0 and ~0 on deep levels), which matches the fact that
    `assemble_P` is already small compared to `gep`.

**Conclusion:** optimizing Python-side list building is not a meaningful lever for `gep`; we need to reduce or accelerate the dense
`eigh` workload and/or the Schur complement work.

### Attempts made (and why they didn’t move the needle)

1) **Partial spectrum via subset selection (`subset_by_index` / `subset_by_value`)**
- Idea: since we only keep a limited number of eigenvectors per aggregate (via `nev` and/or `max_keep`, or via a threshold),
  request only the needed eigenpairs from SciPy:
  - `subset_by_index=[nloc-k, nloc-1]` for the largest `k` eigenpairs.
  - `subset_by_value=[thr, inf]` for threshold-mode selection.
- Outcome: per-level `gep` timings were essentially unchanged in the benchmark configurations tested.
- Likely reason: the selected `k` is often still large enough that the LAPACK work dominates, and/or the driver still performs
  substantial work even when a subset is requested (implementation-dependent).

- But. I also think that in the tests maybe nev was always being used rather than a threshold, such that the changed code never even executed.

2) **Reduce SciPy/LAPACK overhead (`check_finite=False`, Fortran-contiguous inputs, overwrite flags)**
- Idea: avoid hidden copies and input validation overhead in `eigh` by passing
  `np.asfortranarray(...)`, `check_finite=False`, and `overwrite_a/b=True`.
- Outcome: produced only a small improvement in some cases (e.g. modest reduction in `eigh` time on level 0), but did not change
  the overall picture; dense EVP cost still dominates.

### What could be tried next (future work)

1) **Parallelize per-aggregate GEP solves (best candidate if GEP remains a major bottleneck)**
- Each aggregate GEP is independent, so the loop over aggregates is embarrassingly parallel.
- Caveat: SciPy/LAPACK may already be multi-threaded (OpenBLAS/MKL). Parallelizing aggregates without controlling BLAS threads can
  cause oversubscription and slowdowns. Any threaded aggregate-parallel approach should cap BLAS threads to 1 during the region
  (e.g. via `threadpoolctl`).
- Implementation note: parallelism requires a two-pass assembly:
  1) compute per-aggregate results in parallel (kept eigenvectors + row indices),
  2) serial prefix-sum to assign global column indices and assemble `P` deterministically.

2) **Reduce Schur-complement cost**
- The Schur step is the second largest component and includes dense solves on `bb_GG`.
- Potential avenues:
  - reuse factorizations when possible (if repeated structure occurs),
  - reduce `GAMMA` size by tuning overlap construction (algorithmically sensitive),
  - ensure `bb_GG` operations use Cholesky fast paths consistently (already done in many cases).

3) **Alternative dense EVP strategies**
- Depending on SciPy/LAPACK backend, different drivers or formulations may be faster (still same math, different kernel).
- If we eventually accept a larger change, one can explore converting the generalized EVP into a standard EVP via Cholesky of `S`
  (when stable) and calling a faster standard symmetric eigensolver; this is more delicate numerically and should be guarded.

Overall takeaway: the GEP hotspot is a **dense-LAPACK cost**, so meaningful wins are expected from **parallelism or better dense
kernels**, not from Python-level micro-optimizations.