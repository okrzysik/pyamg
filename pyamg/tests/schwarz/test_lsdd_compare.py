"""
A/B comparison test: reference vs experimental least-squares DD solver.

Data layout (your repo):
  tests/schwarz/data/
    B_n4225.npz
    B_n16641.npz
    B_n66049.npz

Each test case loads B, forms A = B.T @ B, generates a random RHS b (length n),
builds the multilevel solver, and uses it as a preconditioner for FGMRES.

How to run (recommended, so you definitely use the active Python env):
  PYAMG_LSDD_PRINT_INFO=1 pytest -q -s tests/schwarz/test_lsdd_compare.py

Optional knobs:
  PYAMG_LSDD_PRINT_INFO=1   -> passes print_info=True into the solver
  PYAMG_LSDD_PRINT_ML=1     -> prints the MultilevelSolver object (can be long)
  PYAMG_RUN_LARGE=1         -> includes the largest test case (n=66049); skipped by default since it can be slow

NOTE: This pyamg environment was installed within a firedrake virtual env, so that's what needs to be active to run this test. If you try to run it in a different environment, you'll likely get an ImportError.
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


def _extract_n(path: Path) -> int:
    m = re.search(r"n(\d+)", path.stem)
    return int(m.group(1)) if m else -1


B_FILES = sorted(_DATA.glob("B_n*.npz"), key=_extract_n)
if not B_FILES:
    raise FileNotFoundError(f"No B_n*.npz files found in {_DATA}")


def _build_A_and_rhs(B: sparse.spmatrix, *, seed: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build A = B^T B (CSR) and a random RHS b (length n).
    Assumption (per user): A is SPD / nonsingular.
    """
    B = B.tocsr()
    A = (B.T @ B).tocsr()

    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)

    return A, b


def _run_one(
    solver_fn,
    *,
    B: sparse.spmatrix,
    A: sparse.spmatrix,
    b: np.ndarray,
    min_coarsening: list[int],
    aggregate: str = "standard",
):
    """Build solver, apply it as a preconditioner to FGMRES, return metrics."""
    print_info = os.environ.get("PYAMG_LSDD_PRINT_INFO", "0") == "1"
    print_ml = os.environ.get("PYAMG_LSDD_PRINT_ML", "0") == "1"

    # --- setup
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

    # --- solve with preconditioned FGMRES
    M = ml.aspreconditioner(cycle="V")
    res: list[float] = []

    t1 = time.perf_counter()
    x, info = fgmres(
        A,
        b,
        tol=1e-8,
        restart=100,
        maxiter=100,
        M=M,
        residuals=res,
    )
    solve_time = time.perf_counter() - t1

    res_arr = np.asarray(res, dtype=float)
    if res_arr.size >= 2:
        # pyamg.krylov.fgmres stores a history; in practice it includes the initial residual.
        iters = max(int(res_arr.size - 1), 1)
        ratio = float(res_arr[-1] / res_arr[0])
        cf = float(np.exp(np.log(ratio) / iters)) if res_arr[0] > 0.0 else float("nan")
        iters_to_01 = float(np.log(0.1) / np.log(cf)) if (cf > 0.0 and cf < 1.0) else float("inf")
    else:
        iters = 0
        ratio = float("nan")
        cf = float("nan")
        iters_to_01 = float("nan")

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
    B = sparse.load_npz(b_path).tocsr()
    n = B.shape[1]

    run_large = os.environ.get("PYAMG_RUN_LARGE", "0") == "1"
    if (n >= 50000) and (not run_large):
        pytest.skip("Large case; set PYAMG_RUN_LARGE=1 to run")

    # A is SPD (per your note), so random b is fine.
    A, b = _build_A_and_rhs(B, seed=n)  # deterministic per-size seed

    coarsen = [8, 10]
    aggregate = "standard"

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
