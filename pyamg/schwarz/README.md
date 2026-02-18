# Schwarz / Least-Squares DD (LS-AMG-DD) — Experiment Track

This directory contains the **least-squares domain-decomposition AMG solver** work for the `least-squares` PyAMG branch, plus a small A/B testing harness to support **exploratory research** and **deep algorithmic changes** without touching the reference implementation.

## Goals

1. Keep a **stable reference** implementation that matches the upstream `least-squares` branch behavior.
2. Maintain an **experimental** implementation for deep modifications (new hierarchy logic, coarse construction, subdomain handling, etc.).
3. Provide a repeatable **A/B comparison test** that:
   - loads a saved Gram factor `B`,
   - forms `A = Bᵀ B`,
   - runs preconditioned Krylov solves,
   - prints setup/solve metrics and convergence rates,
   - ensures the experimental solver doesn’t regress catastrophically.

## What’s in the repo now

### Reference solver (unchanged)
- `pyamg/schwarz/least_squares_dd.py`  
  The original implementation from the `least-squares` branch.

### Experimental solver (copy for deep changes)
- `pyamg/schwarz/least_squares_dd_exp.py`  
  A copy of the reference solver intended for deep modifications.
  The public entrypoint is expected to be something like `pyamg_dd_exp(...)` or
  `least_squares_dd_solver_exp(...)` (the test tries both, via a small import fallback).

**Important:** Keep the reference file untouched; do all exploratory work in the `_exp` module.

## Testing / A-B comparison

### Test location and data
- Test: `tests/schwarz/test_lsdd_compare.py`
- Data: `tests/schwarz/data/`
  - `B_n4225.npz`
  - `B_n16641.npz`
  - `B_n66049.npz`

Each `B_n*.npz` is a SciPy sparse matrix saved via `scipy.sparse.save_npz`.
The number `n` in the filename corresponds to the **number of columns of `B`**
(i.e. the dimension of `A = Bᵀ B`).

The test:
- loads `B`,
- forms `A = B.T @ B`,
- generates a random RHS vector `b` of length `n`,
- builds both solvers (reference + experimental),
- applies each as a preconditioner inside `pyamg.krylov.fgmres`,
- prints setup time, solve time, operator complexity, residual reduction, convergence factor, etc.,
- enforces basic progress checks so the test is meaningful.

### How to run
Use `python -m pytest` to guarantee the correct interpreter/environment:

```bash
python -m pytest -q -s tests/schwarz/test_lsdd_compare.py

