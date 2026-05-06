# Clean implementation plan: aggregate-wise basis scaling for `pyamg/schwarz/lsdd`

This plan is based on the uploaded `pyamg-exp-lsdd-v1.zip`, specifically the implementation under `pyamg/schwarz/lsdd`.

The important implementation fact is that the code does **not** keep explicit local basis blocks named `Z_i` after each aggregate eigenproblem. `_lsdd_process_one_aggregate_gep(...)` appends selected eigenvectors directly into the prolongation triplet lists `p_r`, `p_c`, and `p_v`. Later, `_lsdd_assemble_P_from_triplets(...)` builds `level.P` from those triplets.

So the correct implementation is a **post-GEP / pre-assembly pass** over aggregate-local triplet slices:

1. Reconstruct each aggregate-local selected basis block from `p_v`.
2. Compute an aggregate-local right scaling.
3. Write the scaled columns back into `p_v`.
4. Let the existing `P` assembly and coarsening code run normally.

---

## 1. Goal

Add an optional aggregate-wise right scaling of the selected local basis:

```text
Z_i -> Z_i C_i^{-1}
```

where:

- `Z_i` is the matrix of already-selected eigenvectors on aggregate `omega_i`,
- `k_i = level.eigs.nev[i]` is the number of selected vectors on aggregate `i`,
- `C_i` is a nonsingular `k_i x k_i` row-sampling matrix chosen from the local action of `B` on the selected basis.

The scaling must preserve:

- the number of coarse variables,
- the selected eigenvalue counts `level.eigs.nev`,
- the aggregate support of every prolongation column,
- the aggregate-by-aggregate column ordering,
- current behavior when disabled.

The first patch should be optional and default to no scaling.

---

## 2. Local construction

For aggregate `i`, reconstruct

```text
Z_i in R^{|omega_i| x k_i}
```

from the triplet columns belonging to aggregate `i`.

Use the exact row ordering already stored in the first triplet for the aggregate:

```python
omega_rows = np.asarray(p_r[base], dtype=np.int32)
Z = np.column_stack([np.asarray(p_v[j]) for j in range(base, base + k_i)])
```

Do **not** reconstruct `omega_rows` independently from `level.sub.omega[i]`; the triplet row ordering is the ordering that matches the columns stored in `p_v`.

For the recommended mode, build

```python
rows_i = np.asarray(level.sub.R_rows[i], dtype=np.int32)
Y_i = B[rows_i, :][:, omega_rows] @ Z
```

Then choose `k_i` independent rows of `Y_i`, using pivoted QR on `Y_i.T`. If the selected local row indices are `selected`, set:

```python
C_i = Y_i[selected, :]
Z_scaled = scipy.linalg.solve(C_i.T, Z.T, assume_a="gen", check_finite=False).T
```

This makes the selected local row functionals coordinate rows. That is the intended local sparsifying basis change.

---

## 3. Public API additions

In `pyamg/schwarz/least_squares_dd_exp.py`, add keyword arguments to `least_squares_dd_solver_exp(...)`:

```python
basis_scaling: str = "none",
basis_scaling_cond_max: float = 1.0e8,
basis_scaling_weight_power: float = 1.0,
basis_scaling_normalize_columns: bool = False,
```

Accepted values:

```text
basis_scaling = "none"      # current behavior
basis_scaling = "b_nodal"   # choose rows from B[:, omega_i] Z_i
basis_scaling = "p_nodal"   # choose rows from Z_i itself; debug/fallback mode
```

Validation:

```python
if basis_scaling not in ("none", "b_nodal", "p_nodal"):
    raise ValueError("basis_scaling must be one of 'none', 'b_nodal', or 'p_nodal'")
if basis_scaling_cond_max <= 0.0:
    raise ValueError("basis_scaling_cond_max must be positive")
if basis_scaling_weight_power < 0.0:
    raise ValueError("basis_scaling_weight_power must be nonnegative")
```

Pass these values into `LSDDConfig` inside the setup loop.

Do not add any new coarse assembly mode in this patch.

---

## 4. Extend `LSDDConfig`

In `pyamg/schwarz/lsdd/types.py`, add fields to `LSDDConfig`:

```python
basis_scaling: str = "none"
basis_scaling_cond_max: float = 1.0e8
basis_scaling_weight_power: float = 1.0
basis_scaling_normalize_columns: bool = False
```

Update the `LSDDConfig` docstring. Suggested text:

```text
basis_scaling
    Optional aggregate-wise basis scaling applied after local eigenvector selection
    and before prolongation assembly. "none" leaves the current algorithm unchanged.
    "b_nodal" selects rows from B[:, omega_i] Z_i. "p_nodal" selects rows from
    Z_i itself and is mainly for debugging.

basis_scaling_cond_max
    Maximum accepted condition number for the local row matrix C_i. Aggregates with
    worse conditioning are left unscaled.

basis_scaling_weight_power
    Exponent used when weighting candidate B-rows during row pivoting. Set to 0.0
    for unweighted selection.

basis_scaling_normalize_columns
    If True, normalize each scaled local basis column after scaling. Leave False
    initially.
```

---

## 5. Add `basis_scaling.py`

Create:

```text
pyamg/schwarz/lsdd/basis_scaling.py
```

Every top-level function/class needs a docstring because the existing contract tests check this.

Suggested imports:

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr, solve

from .types import LSDDLevel
```

---

## 6. Add a local diagnostics dataclass

In `basis_scaling.py`, add:

```python
@dataclass(slots=True)
class BasisScalingSummary:
    """Aggregate-wise diagnostics for optional LS-DD basis scaling."""

    method: str
    n_total: int = 0
    n_scaled: int = 0
    n_skipped_empty: int = 0
    n_skipped_too_few_rows: int = 0
    n_skipped_bad_conditioning: int = 0
    n_skipped_exception: int = 0
    cond_values: list[float] | None = None

    def __post_init__(self) -> None:
        """Allocate mutable diagnostic lists after dataclass construction."""
        if self.cond_values is None:
            self.cond_values = []
```

Use this internally, then copy scalar summaries into `stats.extra`.

---

## 7. Row-width weights

Add:

```python
def _lsdd_basis_scaling_row_widths(*, level: LSDDLevel, n_rows: int) -> np.ndarray:
    """Estimate each B-row's tentative scalar coarse width."""
```

Implementation:

```python
width = np.zeros(int(n_rows), dtype=float)
nev = np.asarray(level.eigs.nev, dtype=np.int32)
for i, k_i in enumerate(nev):
    k = int(k_i)
    if k <= 0:
        continue
    rows_i = level.sub.R_rows[i]
    if rows_i is None or len(rows_i) == 0:
        continue
    width[np.asarray(rows_i, dtype=np.int32)] += float(k)
return width
```

Rows with larger width are prioritized during pivoting. This is a heuristic only.

---

## 8. Pivot-selection helper

Add:

```python
def _lsdd_select_basis_scaling_rows(
    *,
    Y: np.ndarray,
    weights: np.ndarray | None,
    cond_max: float,
    weight_power: float,
) -> tuple[np.ndarray | None, float | None]:
    """Select a stable set of row functionals for local basis nodalization."""
```

Algorithm:

1. Let `k = Y.shape[1]`.
2. If `k == 0` or `Y.shape[0] < k`, return `(None, None)`.
3. If `weights is not None` and `weight_power > 0.0`, use weighted rows for pivoting only:

   ```python
   w = np.maximum(np.asarray(weights, dtype=float), 1.0)
   Y_select = (w ** float(weight_power))[:, None] * Y
   ```

   Otherwise use `Y_select = Y`.

4. Run column-pivoted QR on `Y_select.T`:

   ```python
   _, _, piv = qr(Y_select.T, pivoting=True, mode="economic", check_finite=False)
   selected = np.asarray(piv[:k], dtype=np.int32)
   ```

5. Build the unweighted sampled matrix:

   ```python
   C = Y[selected, :]
   ```

6. Check conditioning:

   ```python
   cond_C = float(np.linalg.cond(C))
   if not np.isfinite(cond_C) or cond_C > float(cond_max):
       return None, cond_C
   ```

7. Return `(selected, cond_C)`.

Optional fallback: if weighted pivoting fails the condition check, retry once with `weights=None`.

---

## 9. Local block scaling helper

Add:

```python
def _lsdd_scale_one_triplet_block(
    *,
    B,
    method: str,
    omega_rows: np.ndarray,
    Z: np.ndarray,
    R_rows_i: np.ndarray | None,
    row_width: np.ndarray,
    cond_max: float,
    weight_power: float,
    normalize_columns: bool,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Scale one aggregate-local basis block reconstructed from prolongation triplets."""
```

For `method == "b_nodal"`:

```python
if R_rows_i is None:
    return None, {"reason": "empty_rows"}

rows_i = np.asarray(R_rows_i, dtype=np.int32)
k = int(Z.shape[1])
if rows_i.size < k:
    return None, {"reason": "too_few_rows"}

B_i = B[rows_i, :][:, omega_rows]
Y = np.asarray(B_i @ Z)
weights = row_width[rows_i]
selected, cond_C = _lsdd_select_basis_scaling_rows(
    Y=Y,
    weights=weights,
    cond_max=cond_max,
    weight_power=weight_power,
)
```

If selection fails:

```python
return None, {"reason": "bad_conditioning", "cond": cond_C}
```

Otherwise:

```python
C = Y[selected, :]
Z_scaled = solve(C.T, Z.T, assume_a="gen", check_finite=False).T
return Z_scaled, {"reason": "scaled", "cond": cond_C, "selected": selected}
```

For `method == "p_nodal"`, use `Y = Z`, no weights, and the same solve.

Optional column normalization:

```python
if normalize_columns:
    norms = np.linalg.norm(Z_scaled, axis=0)
    mask = norms > 0.0
    Z_scaled[:, mask] /= norms[mask]
```

Keep `basis_scaling_normalize_columns=False` initially.

---

## 10. Main triplet-scaling pass

Add:

```python
def _lsdd_scale_basis_triplets_for_fill(
    *,
    level: LSDDLevel,
    B,
    p_r: list,
    p_c: list,
    p_v: list,
    method: str,
    cond_max: float,
    weight_power: float,
    normalize_columns: bool,
    stats,
) -> None:
    """Apply optional aggregate-wise right scaling to already-selected P triplets."""
```

Implementation outline:

```python
if method == "none":
    return

summary = BasisScalingSummary(method=method)
nev = np.asarray(level.eigs.nev, dtype=np.int32)
row_width = _lsdd_basis_scaling_row_widths(level=level, n_rows=B.shape[0])

base = 0
for i, k_i in enumerate(nev):
    k = int(k_i)
    summary.n_total += 1

    if k <= 0:
        summary.n_skipped_empty += 1
        continue

    omega_rows = np.asarray(p_r[base], dtype=np.int32)
    Z = np.column_stack([np.asarray(p_v[j]) for j in range(base, base + k)])

    try:
        Z_scaled, info = _lsdd_scale_one_triplet_block(
            B=B,
            method=method,
            omega_rows=omega_rows,
            Z=Z,
            R_rows_i=level.sub.R_rows[i],
            row_width=row_width,
            cond_max=cond_max,
            weight_power=weight_power,
            normalize_columns=normalize_columns,
        )
    except Exception:
        Z_scaled = None
        info = {"reason": "exception"}

    if Z_scaled is None:
        reason = str(info.get("reason", "exception"))
        if reason == "too_few_rows":
            summary.n_skipped_too_few_rows += 1
        elif reason == "bad_conditioning":
            summary.n_skipped_bad_conditioning += 1
        elif reason == "empty_rows":
            summary.n_skipped_empty += 1
        else:
            summary.n_skipped_exception += 1
    else:
        for local_col, triplet_idx in enumerate(range(base, base + k)):
            p_v[triplet_idx] = np.asarray(Z_scaled[:, local_col])
        summary.n_scaled += 1
        cond = info.get("cond")
        if cond is not None:
            summary.cond_values.append(float(cond))

    base += k

if base != len(p_v):
    raise ValueError(f"Basis scaling consumed {base} triplets but p_v has {len(p_v)} entries")
```

Store diagnostics:

```python
stats.extra["basis_scaling_method"] = method
stats.extra["basis_scaling_total"] = summary.n_total
stats.extra["basis_scaling_scaled"] = summary.n_scaled
stats.extra["basis_scaling_skipped_empty"] = summary.n_skipped_empty
stats.extra["basis_scaling_skipped_too_few_rows"] = summary.n_skipped_too_few_rows
stats.extra["basis_scaling_skipped_bad_conditioning"] = summary.n_skipped_bad_conditioning
stats.extra["basis_scaling_skipped_exception"] = summary.n_skipped_exception

if summary.cond_values:
    conds = np.asarray(summary.cond_values, dtype=float)
    stats.extra["basis_scaling_cond_min"] = float(np.min(conds))
    stats.extra["basis_scaling_cond_med"] = float(np.median(conds))
    stats.extra["basis_scaling_cond_max"] = float(np.max(conds))
```

Do not print from this module.

---

## 11. Hook into `hierarchy.py`

In `_lsdd_extend_hierarchy(...)`, insert the scaling pass after the GEP loop and before `assemble_P`.

Recommended placement:

```python
for k, dt in gep_timers.items():
    stats.timings[k] = dt

# ---- optional aggregate-local basis scaling ----
if cfg.basis_scaling != "none":
    from .basis_scaling import _lsdd_scale_basis_triplets_for_fill

    with stats.timeit("basis_scale"):
        _lsdd_scale_basis_triplets_for_fill(
            level=level,
            B=B,
            p_r=p_r,
            p_c=p_c,
            p_v=p_v,
            method=cfg.basis_scaling,
            cond_max=cfg.basis_scaling_cond_max,
            weight_power=cfg.basis_scaling_weight_power,
            normalize_columns=cfg.basis_scaling_normalize_columns,
            stats=stats,
        )
```

Do not alter `_lsdd_process_one_aggregate_gep(...)` in the first patch.

Do not alter `_lsdd_assemble_P_from_triplets(...)` except for defensive consistency checks if desired.

Leave `_lsdd_coarsen_operators(...)` algorithmically unchanged.

---

## 12. Update `stats.py`

In `_lsdd_print_level_summary(...)`, add `basis_scale` to the timing order after `gep` and before `assemble_P`:

```python
order = [
    "filter",
    "strength",
    "aggregate",
    "overlap",
    "extract_PCM",
    "extract_A",
    "outerprod",
    "gep",
    "basis_scale",
    "assemble_P",
    "prerel_stp",
    "pstrel_stp",
    "coarsen",
]
```

Add an optional diagnostics block after the coarse/eigenvalue summary:

```python
method = stats.extra.get("basis_scaling_method")
if method and method != "none":
    print(f"{indent}     basis scaling:")
    print(f"{indent}       method : {method}")
    print(f"{indent}       scaled : {stats.extra.get('basis_scaling_scaled', 0)} / {stats.extra.get('basis_scaling_total', 0)}")
    print(f"{indent}       skip empty      : {stats.extra.get('basis_scaling_skipped_empty', 0)}")
    print(f"{indent}       skip rows       : {stats.extra.get('basis_scaling_skipped_too_few_rows', 0)}")
    print(f"{indent}       skip cond       : {stats.extra.get('basis_scaling_skipped_bad_conditioning', 0)}")
    print(f"{indent}       skip exception  : {stats.extra.get('basis_scaling_skipped_exception', 0)}")
    print(f"{indent}       cond  : {_mmx(stats.extra, 'basis_scaling_cond')}")
```

Printing should stay centralized in `stats.py`.

---

## 13. Update `__init__.py`

In `pyamg/schwarz/lsdd/__init__.py`, import and export the new module.

Add `basis_scaling` to the module import list and to `__all__`.

---

## 14. Update benchmark CLI

In `pyamg/tests/schwarz/bench_lsdd.py`, add experimental solver args:

```python
parser.add_argument("--basis-scaling", choices=["none", "b_nodal", "p_nodal"], default="none")
parser.add_argument("--basis-scaling-cond-max", type=float, default=1.0e8)
parser.add_argument("--basis-scaling-weight-power", type=float, default=1.0)
parser.add_argument("--basis-scaling-normalize-columns", action="store_true")
```

Pass these only into the experimental solver kwargs.

Example:

```python
exp_kwargs.update(
    basis_scaling=args.basis_scaling,
    basis_scaling_cond_max=args.basis_scaling_cond_max,
    basis_scaling_weight_power=args.basis_scaling_weight_power,
    basis_scaling_normalize_columns=args.basis_scaling_normalize_columns,
)
```

---

## 15. Tests to add

Create:

```text
pyamg/tests/schwarz/test_lsdd_basis_scaling.py
```

### 15.1 Local selected rows become identity

Build a small synthetic block:

```python
rng = np.random.default_rng(0)
n_omega = 8
n_rows = 12
k = 3
Z = rng.standard_normal((n_omega, k))
B_i = rng.standard_normal((n_rows, n_omega))
Y = B_i @ Z
```

Call the low-level helper. Expose `selected` and `cond` in the returned info dict. Reconstruct `C = Y[selected, :]`, then check:

```python
Z_scaled = solve(C.T, Z.T, assume_a="gen", check_finite=False).T
Y_scaled = Y @ np.linalg.solve(C, np.eye(k))
np.allclose(Y_scaled[selected, :], np.eye(k), atol=1e-10, rtol=1e-10)
```

### 15.2 Triplet pass preserves column count and supports

Create synthetic triplets for two aggregates with `nev = [2, 3]`.

Run `_lsdd_scale_basis_triplets_for_fill(...)`.

Check:

```python
len(p_r) unchanged
len(p_c) unchanged
len(p_v) unchanged
p_c[j] unchanged for all j
p_r[j] unchanged for all j
```

The first implementation should not drop prolongation entries.

### 15.3 Coarse-space invariance

For a small SPD problem, assemble `P_before` and `P_after` from triplets.

Check that the coarse-space projector is unchanged up to roundoff:

```python
E_before = P_before @ np.linalg.inv(P_before.T @ A @ P_before) @ P_before.T
E_after  = P_after  @ np.linalg.inv(P_after.T  @ A @ P_after)  @ P_after.T
np.allclose(E_before, E_after, atol=1e-10, rtol=1e-10)
```

This is the key mathematical contract: the local basis changes, but the local coarse space does not.

### 15.4 End-to-end setup smoke test

Use a small existing Schwarz data case and build:

```python
basis_scaling="none"
basis_scaling="b_nodal"
basis_scaling="p_nodal"
```

Check:

```python
setup completes
ml.levels[0].P.shape[1] unchanged between none and scaled modes
all levels have A
non-final levels that continue coarsening have B
```

### 15.5 Existing contract tests

Run:

```bash
pytest -q pyamg/tests/schwarz/test_lsdd_contracts.py
```

This should catch missing docstrings or accidental `print()` calls outside approved reporting code.

---

## 16. Experiments to run

Baseline:

```bash
python pyamg/tests/schwarz/bench_lsdd.py \
  --data pyamg/tests/schwarz/data \
  --solver exp \
  --aggregate standard \
  --coarsen 8 10 \
  --robust_Sker_handling True \
  --per-level
```

Scaled:

```bash
python pyamg/tests/schwarz/bench_lsdd.py \
  --data pyamg/tests/schwarz/data \
  --solver exp \
  --aggregate standard \
  --coarsen 8 10 \
  --robust_Sker_handling True \
  --basis-scaling b_nodal \
  --basis-scaling-cond-max 1e8 \
  --basis-scaling-weight-power 1.0 \
  --per-level
```

Also test a more conservative condition cutoff:

```bash
python pyamg/tests/schwarz/bench_lsdd.py \
  --data pyamg/tests/schwarz/data \
  --solver exp \
  --aggregate standard \
  --coarsen 8 10 \
  --robust_Sker_handling True \
  --basis-scaling b_nodal \
  --basis-scaling-cond-max 1e6 \
  --basis-scaling-weight-power 1.0 \
  --per-level
```

Track:

```text
operator complexity
setup time
basis_scale time
n_coarse per level
coarse matrix nnz per level
propagated factor nnz per level when available
iteration count
convergence factor
basis_scaling_scaled / basis_scaling_total
basis_scaling skipped counts
basis_scaling_cond min/med/max
```

For lightweight diagnostics, add these stats inside `_lsdd_coarsen_operators(...)` without changing the algorithm:

```python
stats.extra["coarsen_A_nnz"] = int(A_c.nnz)
if B_c is not None:
    stats.extra["coarsen_B_nnz"] = int(B_c.nnz)
```

---

## 17. Expected outcomes and caveats

1. If most aggregates have `k_i == 1`, this will not do much. A one-dimensional local block has no meaningful internal basis choice.
2. This should not change `nnz(P)` in the first implementation. The support of each aggregate-local basis remains the same.
3. Useful metrics are coarse matrix nnz, propagated factor nnz when available, operator complexity, and setup/solve time.
4. Large `cond(C_i)` is dangerous. If selected row functionals are nearly dependent, the basis scaling can blow up coefficients and worsen the hierarchy.
5. Start with `basis_scaling_cond_max=1e6` or `1e8`.
6. Do not enable this by default until benchmark evidence supports it.

---

## 18. Minimal implementation order

1. Add config/API fields with default no-op behavior.
2. Add `basis_scaling.py` with row selection and triplet scaling.
3. Hook the scaling pass into `_lsdd_extend_hierarchy(...)` after the GEP loop and before `assemble_P`.
4. Add `basis_scale` timing and stats printing.
5. Add benchmark CLI flags.
6. Add unit tests for local identity, triplet invariants, and coarse-space invariance.
7. Run existing contract tests.
8. Run benchmark comparisons.

The first patch should leave `basis_scaling="none"` matching current behavior.
