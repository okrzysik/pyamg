"""
Comparison test: reference vs experimental least-squares DD solver.

Test cases are based on the B_n*.npz files from tests/schwarz/data, which contain sparse matrices
B. Each test case loads B, forms A = B.T @ B, generates a random RHS b (length n), builds the
multilevel solver, and uses it as a preconditioner for FGMRES.
See also:
    tests/schwarz/data/
        B_n4225.npz
        B_n16641.npz
        B_n66049.npz

How to run from pyamg root:
  PYAMG_LSDD_PRINT_INFO=1 pytest -q -s pyamg/tests/schwarz/test_lsdd_compare.py

Optional knobs:
  PYAMG_LSDD_PRINT_INFO=1   -> passes print_info=True into the solver
  PYAMG_LSDD_PRINT_ML=1     -> prints the MultilevelSolver object (can be long)
  PYAMG_RUN_LARGE=1         -> includes the largest test case (n=66049); skipped by default since it can be slow
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from pyamg.krylov import fgmres

# Reference vs experimental solver entrypoints.
# (Keeps this test resilient if you later rename the function.)
try:
    from pyamg.schwarz.least_squares_dd import pyamg_dd as pyamg_dd_ref  # type: ignore
except Exception:
    from pyamg.schwarz.least_squares_dd import (  # type: ignore
        least_squares_dd_solver as pyamg_dd_ref,
    )

try:
    from pyamg.schwarz.least_squares_dd_exp import pyamg_dd_exp as pyamg_dd_exp  # type: ignore
except Exception:
    from pyamg.schwarz.least_squares_dd_exp import (  # type: ignore
        least_squares_dd_solver_exp as pyamg_dd_exp,
    )


_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"
_DEFAULT_COARSEN = [8, 10]
_DEFAULT_AGGREGATE = "standard"


def _extract_n(path: Path) -> int:
    m = re.search(r"n(\d+)", path.stem)
    return int(m.group(1)) if m else -1


B_FILES = sorted(_DATA.glob("B_n*.npz"), key=_extract_n)
if not B_FILES:
    raise FileNotFoundError(f"No B_n*.npz files found in {_DATA}")


def _build_A_and_rhs(B: sparse.spmatrix, *, seed: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build A = B^T B (CSR) and a deterministic random RHS b (length n).

    The caller controls reproducibility through ``seed``.
    """
    B = B.tocsr()
    A = (B.T @ B).tocsr()

    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)

    return A, b


def _env_flag(name: str) -> bool:
    """Read a bool flag from environment where \"1\" means True."""
    return os.environ.get(name, "0") == "1"


def _convergence_metrics(residuals: np.ndarray) -> tuple[int, float, float, float]:
    """Return (iters, reduction_ratio, conv_factor, iters_to_0_1)."""
    if residuals.size < 2:
        return 0, float("nan"), float("nan"), float("nan")

    # pyamg.krylov.fgmres stores a history; in practice it includes the initial residual.
    iters = max(int(residuals.size - 1), 1)
    ratio = float(residuals[-1] / residuals[0])
    cf = float(np.exp(np.log(ratio) / iters)) if residuals[0] > 0.0 else float("nan")
    iters_to_01 = float(np.log(0.1) / np.log(cf)) if (cf > 0.0 and cf < 1.0) else float("inf")
    return iters, ratio, cf, iters_to_01


def _skip_large_case_if_needed(n: int) -> None:
    """Skip large matrix cases unless explicitly enabled by env var."""
    if (n >= 50000) and (not _env_flag("PYAMG_RUN_LARGE")):
        pytest.skip("Large case; set PYAMG_RUN_LARGE=1 to run")


def _run_one(
    solver_fn,
    *,
    B: sparse.spmatrix,
    A: sparse.spmatrix,
    b: np.ndarray,
    min_coarsening: list[int],
    aggregate: str = "standard",
):
    """Build one solver, run preconditioned FGMRES, and return benchmark metrics.

    Environment toggles:
    - PYAMG_LSDD_PRINT_INFO=1: enable solver setup logging.
    - PYAMG_LSDD_PRINT_ML=1: print the multilevel hierarchy object.
    """
    print_info = _env_flag("PYAMG_LSDD_PRINT_INFO")
    print_ml = _env_flag("PYAMG_LSDD_PRINT_ML")

    # Setup multilevel hierarchy.
    t0 = time.perf_counter()
    ml = solver_fn(
        B=B,
        BT=None,
        A=A,
        symmetry="hermitian",
        aggregate=aggregate,
        agg_levels=2,
        presmoother="ras",
        postsmoother="rasT",
        kappa=50,
        min_coarsening=min_coarsening,
        nev=None,
        threshold=None,
        max_levels=10,
        max_coarse=10,
        max_density=0.25,
        print_info=print_info,
    )
    setup_time = time.perf_counter() - t0

    if print_ml:
        print(ml)

    # Solve with V-cycle preconditioned FGMRES.
    preconditioner = ml.aspreconditioner(cycle="V")
    residuals: list[float] = []

    t1 = time.perf_counter()
    x, info = fgmres(
        A,
        b,
        tol=1e-8,
        restart=100,
        maxiter=100,
        M=preconditioner,
        residuals=residuals,
    )
    solve_time = time.perf_counter() - t1

    res_arr = np.asarray(residuals, dtype=float)
    iters, ratio, cf, iters_to_01 = _convergence_metrics(res_arr)

    oc = float(ml.operator_complexity())

    return dict(
        ml=ml,
        x=x,
        info=info,
        res=res_arr,
        setup_time=setup_time,
        solve_time=solve_time,
        iters=iters,
        ratio=ratio,
        cf=cf,
        iters_to_01=iters_to_01,
        oc=oc,
    )


def _print_summary(label: str, out: dict, *, coarsen: list[int]):
    """Print a compact per-solver summary for one test case."""
    res = out["res"]
    final_res = float(res[-1]) if res.size else float("nan")
    init_res = float(res[0]) if res.size else float("nan")

    print(f"\n--- {label} ---")
    print(f"Coarsening = {coarsen}")
    print(f"LS Setup time = {out['setup_time']:.2f} s")
    print(f"LS Solve time = {out['solve_time']:.2f} s")
    print(f"OC = {out['oc']:.2f}")
    print(f"FGMRES info = {out['info']}")
    print(f"Iters = {out['iters']}")
    print(f"Initial res = {init_res:.2e}")
    print(f"Final res   = {final_res:.2e}")
    print(f"Reduction   = {out['ratio']:.2e}")
    print(f"Conv fac    = {out['cf']:.3f}")
    print(f"Iters to 0.1 = {out['iters_to_01']:.2f}")


@pytest.mark.parametrize("b_path", B_FILES, ids=[p.stem for p in B_FILES])
def test_lsdd_ref_vs_exp_print_metrics(b_path: Path):
    """Compare convergence quality of reference vs experimental LS-DD on one matrix."""
    B = sparse.load_npz(b_path).tocsr()
    n = B.shape[1]
    _skip_large_case_if_needed(n)

    # Build a deterministic RHS for repeatable comparisons.
    A, b = _build_A_and_rhs(B, seed=n)  # deterministic per-size seed

    coarsen = _DEFAULT_COARSEN
    aggregate = _DEFAULT_AGGREGATE

    out_ref = _run_one(pyamg_dd_ref, B=B, A=A, b=b, min_coarsening=coarsen, aggregate=aggregate)
    out_exp = _run_one(pyamg_dd_exp, B=B, A=A, b=b, min_coarsening=coarsen, aggregate=aggregate)

    _print_summary("REF", out_ref, coarsen=coarsen)
    _print_summary("EXP", out_exp, coarsen=coarsen)

    # --- assertions (sanity + comparison)
    assert out_ref["res"].size >= 2
    assert out_exp["res"].size >= 2

    # Both should make meaningful progress.
    assert out_ref["ratio"] < 1e-2, (b_path.name, out_ref["ratio"], out_ref["info"])
    assert out_exp["ratio"] < 1e-2, (b_path.name, out_exp["ratio"], out_exp["info"])

    # Experimental shouldn't be catastrophically worse than reference.
    assert out_exp["ratio"] <= 100.0 * out_ref["ratio"], (
        b_path.name,
        out_ref["ratio"],
        out_exp["ratio"],
        out_ref["iters"],
        out_exp["iters"],
    )
