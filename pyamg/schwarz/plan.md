# Pathway: Optimizing `outerprod` (local B-row outer products) in LS–AMG–DD

This note is meant to be pasted into a fresh chat to drive a focused performance effort on the **`outerprod`** stage in the refactored
LS–AMG–DD experimental solver (`pyamg/schwarz/least_squares_dd_exp.py` + `pyamg/schwarz/lsdd/`).

The concrete motivation (from `bench_lsdd.py --per-level`) is that on large problems the deepest level(s) can spend **multiple seconds**
in `outerprod` (e.g. `B_n263169` level 2: `outerprod ≈ 2.7–2.8s`), often larger than `gep` and `coarsen` on that level.

---

## 0) Current code context

### Where `outerprod` lives
- Python orchestration: `pyamg/schwarz/lsdd/hierarchy.py`
- Python stage implementation: `pyamg/schwarz/lsdd/local_ops.py`
  - function typically named something like `_lsdd_local_outer_products_and_gep_init(...)`
- Compiled helper(s):
  - `pyamg/schwarz/BTB.cpp` and/or `pyamg/amg_core` entry points used by `local_ops.py`

### What `outerprod` conceptually does
Given:
- A set of aggregates `i = 0..(n_aggs-1)`
- For each aggregate:
  - Overlap DOFs `OMEGA_i` (global column indices)
  - Row-set `R_rows[i]` (rows of B that touch the aggregate; built from column→row adjacency of B)
  - Optionally PoU/multiplicity weights `v_row_mult`
It builds **dense SPSD local blocks** (stored flattened in `level.blocks.auxiliary`) via sums of row outer products:
- Roughly: `bb_full_i ≈ Σ_{r ∈ R_rows_i} w_r * (b_r|_{OMEGA_i}) (b_r|_{OMEGA_i})^T`
Those `bb_full_i` blocks are later Schur-complemented onto `omega_i` inside the per-aggregate GEP.

### Important performance fact
As levels coarsen, `B_{ℓ+1} = B_ℓ P_ℓ` tends to make **B rows less sparse**. If `nnz(row)` increases, the cost of row outer products
tends to scale like `Σ_r nnz(r)^2` (or worse), which can explode at deep levels even when `n_aggs` is small.

---

## 1) First step: instrument `outerprod` into prep/kernel/post

Before changing algorithms, split the timer the same way we did for `coarsen` and `gep`, and print sub-timers under `outerprod`
(without double-counting totals).

### Add a dict accumulator at call-site
In `hierarchy.py` around the `outerprod` stage:

- Create `outer_timers: dict[str, float] = {}`
- Pass it into the `local_ops` function that does outer products.
- After the stage, store `outer_timers` into `stats.timings` with distinct keys (e.g. `outerprod_prep`, `outerprod_kernel`, ...).

### In `local_ops.py`, add sub-timers around:
1) `outerprod_prep`
   - building/packing pointer arrays, scratch buffers, dtype casts, etc.
2) `outerprod_kernel`
   - the single compiled call (e.g. `amg_core.local_outer_product(...)`)
3) `outerprod_post`
   - any symmetrization, scaling, writing into flattened buffers, sorting, etc.

### Also record diagnostic scalars (cheap, huge value)
For each level (or at least for the hot level):
- `n_aggs`, and for each agg: `|OMEGA_i|`, `|omega_i|`, `|GAMMA_i|`
- `|R_rows_i|` distribution (min/med/max)
- **row density** statistics over the rows in `R_rows`:
  - `nnz(row)` min/med/max over all rows used on that level
  - estimate `Σ_r nnz(r)^2` over all rows used on that level
These stats often explain why deep-level `outerprod` explodes.

**Decision point after instrumentation:**
- If `outerprod_kernel` dominates → focus on C++/kernel and data layout.
- If `outerprod_prep` dominates → focus on eliminating copies / dtype churn / pointer repacking.

---

## 2) Quick wins (low-risk) to try first

### 2.1 Ensure all index arrays passed into C++ are `int32` and contiguous
A very common hidden cost is repeated `int64 → int32` copying every time we enter the compiled kernel.

Check these dtypes and contiguity once per call:
- `B.indptr.dtype`, `B.indices.dtype`
- `R_rows` concatenation buffers / ptr arrays (if passed)
- `subdomain_ptr`, `submatrices_ptr`, etc.

**Goal:** enforce `int32` *once per level* (or once per matrix build), not inside the hot stage.
- Convert `B` indices to int32 when building/coarsening `B` (or right after filtering) and keep it that way.
- Ensure pointer arrays created in Python use `dtype=np.int32`.

### 2.2 Avoid repeated format conversions inside hot path
If the kernel expects CSR arrays, make sure `B` is already CSR and stays CSR.
Avoid doing any `tocsr()/tocsc()/sort_indices()` inside the hot stage unless proven necessary.

### 2.3 Avoid repeated allocations inside the per-level call
If `outerprod_prep` shows lots of time:
- Preallocate scratch arrays (ptr arrays, work buffers) sized from `n_aggs` and reuse.
- Avoid creating large temporary Python lists/arrays repeatedly in the stage.

---

## 3) Medium effort: reduce arithmetic/memory traffic in the kernel

Assuming the kernel dominates, typical improvements include:

### 3.1 Exploit symmetry of the dense block
`bb_full_i` is SPD, so only the upper triangle needs to be accumulated, then mirrored (or call a symmetric rank-k style update).
If current kernel writes full `k×k`, halve work and improve cache by storing only triangle during accumulation.

### 3.2 Tile the dense block updates
Sparse outer products do many scattered writes into a large dense matrix.
Use **tiling** (e.g. 32×32 or 64×64 blocks) so updates hit cache lines more predictably.

### 3.3 Improve inner-loop structure for sparse row outer products
For each row `r` restricted to `OMEGA_i`:
- Let local nonzeros be `(idx[0..t-1], val[0..t-1])`
- Kernel does nested loops `for a in 0..t-1: for b in a..t-1: M[idx[a], idx[b]] += val[a]*val[b]*w`
Key knobs:
- Ensure `idx` is sorted for locality.
- Avoid branches in the inner loop.
- Use pointer arithmetic to row-major contiguous dense storage.

### 3.4 Compile flags and vectorization
Ensure compiled extension is built with:
- `-O3`
- `-march=native` (or appropriate tuning flags)
- link to a good BLAS if used (often the kernel is custom loops, but still).

---

## 4) Parallelism options (careful)

`outerprod` parallelization is tricky because multiple threads can write into the same dense block.

### 4.1 Parallelize across aggregates
Works well when `n_aggs` is large.
However the deepest-level hotspot can have very few aggregates (e.g. level 2 has 2 aggregates), so this may not help there.

### 4.2 Parallelize across rows within an aggregate (harder)
Naively parallelizing over rows causes write races into the dense matrix.
Safer strategies:
- thread-private dense accumulators + reduction (often too much memory for large `|OMEGA_i|`)
- tile-based locking (mutex per tile) or atomic adds (usually slow)
- split the dense matrix into disjoint tiles and assign rows to tiles based on support (complex)

**Practical approach:** only attempt this once we know the deep-level hotspot is truly kernel-bound and we have no other big wins.

---

## 5) Algorithmic levers (optional; higher risk, large payoff)

These are “still same method” but change computation intensity.

### 5.1 More aggressive coarse-level filtering of `B`
Since outer products scale like `Σ nnz(row)^2`, reducing row density can be huge.
Consider:
- applying `filter_matrix_rows` (or similar) more aggressively on *coarser* levels
- a level-dependent threshold for filtering `B`
Trade-off: may degrade the quality of the auxiliary blocks and hence coarse spaces.

### 5.2 Reduce `|R_rows_i|` (row-set size)
`R_rows` is derived from column adjacency.
If it can be reduced without losing essential stabilization structure (e.g. via locality restriction), outerprod cost drops.

### 5.3 Change representation: accumulate from columns instead of rows
Sometimes it is cheaper to compute `B_i^T B_i` via column-wise sparse operations (SpGEMM-style) rather than row outer products,
depending on the sparsity structure of `B` on coarse levels. This is major work but can pay off.

---

## 6) Microbenchmarking strategy (recommended)

Add a benchmark mode to isolate `outerprod` on a single level:
- Run the hierarchy build up to a target level (e.g. level 2 on `B_n263169`)
- Then repeatedly time just the `outerprod` stage (N repeats) to reduce noise
- Print:
  - `outerprod_prep/kernel/post` breakdown
  - row-density stats (`nnz(row)`, `Σ nnz(row)^2`)
This allows kernel work to be tuned without rerunning the entire setup each time.

---

## 7) “What success looks like”
- After instrumentation, we can answer:
  1) Is time mostly in the compiled kernel or Python prep?
  2) Is the deep-level explosion explained by row density (`Σ nnz(row)^2`)?
- A good first target is reducing deep-level `outerprod` by **2×** (e.g. 2.8s → ~1.4s), which would be a meaningful overall
  setup improvement on large problems.

---

## 8) Immediate next tasks for a new chat
1) Add `outerprod_prep/kernel/post` timers and print them under `outerprod`.
2) Record row-density stats and `Σ nnz(row)^2` estimate per level.
3) Based on results:
   - If `prep` dominates: eliminate dtype conversions and allocations.
   - If `kernel` dominates: focus on symmetry + tiling + loop structure + compile flags.
4) Re-run `bench_lsdd.py --per-level` on `B_n263169` and compare `outerprod` deltas.
