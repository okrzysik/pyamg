"""Benchmark LS-AMG-DD reference "ref" and experimental "exp" solver implementations on saved matrices. 

Run from pyamg root:
  python pyamg/tests/schwarz/bench_lsdd.py --data pyamg/tests/schwarz/data --aggregate standard --coarsen 8 10 --solver ref

Key options:
  --solver {ref,exp,both}   run the reference and/or experimental solver
  --robust_Sker_handling    exp only: robust local handling of kernel(S)
  --force_row_closure       exp only: enforce row-closure in aggregation
  --per-level               print per-level LS-DD timing/stats (if present)
  --csv out.csv             write a CSV summary
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from pyamg.krylov import fgmres


def _conv_factor(res: list[float]) -> tuple[int, float, float]:
    """Return (iters, conv_factor, final_res)."""
    if len(res) < 2:
        return 0, float("nan"), float("nan")
    iters = len(res) - 1
    r0 = res[0]
    r1 = res[-1]
    if r0 <= 0:
        return iters, float("nan"), float(r1)
    cf = float(np.exp(np.log(r1 / r0) / max(iters, 1)))
    return iters, cf, float(r1)


def _run_one(
    name: str,
    solver_fn,
    *,
    B,
    A,
    b,
    aggregate: str,
    coarsen: list[int],
    kappa: float,
    nev: int | None,
    max_levels: int,
    max_coarse: int,
    max_density: float,
    tol: float,
    maxiter: int,
    restart: int,
    solver_kwargs: dict | None = None,
):
    """Run one solver build+solve and return benchmark metrics.

    solver_kwargs are forwarded only to the selected solver implementation.
    Use this for solver-specific options so unsupported keywords are not
    passed to other solver variants.
    """
    if solver_kwargs is None:
        solver_kwargs = {}

    t0 = time.perf_counter()
    ml = solver_fn(
        B=B,
        BT=None,
        A=A,
        symmetry="symmetric",
        aggregate=aggregate,
        agg_levels=2,
        presmoother="ras",
        postsmoother="rasT",
        kappa=kappa,
        min_coarsening=coarsen,
        nev=nev,
        threshold=None,
        max_levels=max_levels,
        max_coarse=max_coarse,
        max_density=max_density,
        print_info=False,
        **solver_kwargs,
    )
    setup_time = time.perf_counter() - t0

    residuals: list[float] = []
    preconditioner = ml.aspreconditioner(cycle="V")

    t1 = time.perf_counter()
    _, info = fgmres(
        A,
        b,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
        M=preconditioner,
        residuals=residuals,
    )
    solve_time = time.perf_counter() - t1

    iters, cf, final_res = _conv_factor(residuals)


    return dict(
        solver=name,
        setup_time=setup_time,
        solve_time=solve_time,
        iters=iters,
        conv_factor=cf,
        final_res=final_res,
        info=info,
        oc=float(ml.operator_complexity()),
        ml=ml
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Directory containing B_n*.npz")
    parser.add_argument("--solver", choices=["ref", "exp", "both"], default="both")
    parser.add_argument("--aggregate", type=str, default="standard")
    parser.add_argument("--coarsen", type=int, nargs="+", default=[8, 10])
    parser.add_argument("--kappa", type=float, default=50.0)
    parser.add_argument("--nev", type=int, default=0, help="0 means None (threshold-based)")
    parser.add_argument("--max-levels", type=int, default=10)
    parser.add_argument("--max-coarse", type=int, default=10)
    parser.add_argument("--max-density", type=float, default=0.25)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--per-level", action="store_true")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument(
        "--robust_Sker_handling",
        type=bool,
        default=False,
        help="exp only: Force robust handling of kernel of S",
    )
    parser.add_argument(
        "--force_row_closure",
        type=bool,
        default=False,
        help="exp only: Force closure of rows in aggregation",
    )
    parser.add_argument(
        "--basis-scaling",
        choices=["none", "b_nodal", "p_nodal"],
        default="none",
        help="exp only: optional post-GEP basis scaling mode",
    )
    parser.add_argument(
        "--basis-scaling-cond-max",
        type=float,
        default=1.0e8,
        help="exp only: maximum accepted condition number for local scaling matrix",
    )
    parser.add_argument(
        "--basis-scaling-weight-power",
        type=float,
        default=1.0,
        help="exp only: row-weight exponent used in b_nodal pivot selection",
    )
    parser.add_argument(
        "--basis-scaling-normalize-columns",
        action="store_true",
        help="exp only: normalize local columns after basis scaling",
    )
    parser.add_argument(
        "--basis-scaling-drop-tol",
        type=float,
        default=0.0,
        help="exp only: relative drop tolerance for tiny scaled basis entries",
    )
    return parser.parse_args()


def _n_from_name(path: Path) -> int:
    match = re.search(r"_n(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _find_b_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("B_n*.npz"), key=_n_from_name)
    if not files:
        raise FileNotFoundError(f"No B_n*.npz found in {data_dir}")
    return files


def _build_problem(B_file: Path) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    B = sparse.load_npz(B_file).tocsr()
    A = (B.T @ B).tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(n)
    b = rng.standard_normal(n)
    return B, A, b


def _iter_solvers(which: str):
    # Import here so it uses your editable install cleanly.
    from pyamg.schwarz.least_squares_dd import least_squares_dd_solver as pyamg_dd_ref
    from pyamg.schwarz.least_squares_dd_exp import least_squares_dd_solver_exp as pyamg_dd_exp

    if which in ("ref", "both"):
        yield "ref", pyamg_dd_ref
    if which in ("exp", "both"):
        yield "exp", pyamg_dd_exp


def _print_result(name: str, out: dict) -> None:
    print(
        f"{name:>3} | setup={out['setup_time']:.2f}s "
        f"solve={out['solve_time']:.2f}s iters={out['iters']:3d} "
        f"cf={out['conv_factor']:.3f} oc={out['oc']:.2f} final_res={out['final_res']:.2e}"
    )


def _print_per_level(ml) -> None:
    from pyamg.schwarz.lsdd.stats import _lsdd_print_level_summary

    printed = False
    for level in ml.levels:
        stats = getattr(level, "lsdd_stats", None)
        if stats is None:
            continue
        if printed:
            print("-" * 72)
        printed = True
        _lsdd_print_level_summary(stats, print_info=True, prefix="", indent="")


def main() -> None:
    args = _parse_args()
    files = _find_b_files(Path(args.data))

    if args.per_level:
        print(
            "Legend: omega=|nonoverlapping aggregate|, OMEGA=|overlapping subdomain|, "
            "GAMMA=OMEGA-omega (interface size inside the overlap block). "
            "nev = # eigenvectors kept per aggregate. eig = eigenvalues kept (columns of P)."
        )

    solvers = list(_iter_solvers(args.solver))
    rows = []

    for f in files:
        B, A, b = _build_problem(f)
        n = A.shape[0]
        nev = None if args.nev == 0 else args.nev

        # Options supported only by the experimental solver.
        exp_kwargs = {
            "robust_Sker_handling": args.robust_Sker_handling,
            "force_row_closure": args.force_row_closure,
            "basis_scaling": args.basis_scaling,
            "basis_scaling_cond_max": args.basis_scaling_cond_max,
            "basis_scaling_weight_power": args.basis_scaling_weight_power,
            "basis_scaling_normalize_columns": args.basis_scaling_normalize_columns,
            "basis_scaling_drop_tol": args.basis_scaling_drop_tol,
        }

        print(f"\n=== {f.name} (n={n}) ===")
        for name, solver_fn in solvers:
            # Pass exp-only options only to the experimental solver.
            solver_kwargs = exp_kwargs if name == "exp" else None
            out = _run_one(
                name,
                solver_fn,
                B=B,
                A=A,
                b=b,
                aggregate=args.aggregate,
                coarsen=list(args.coarsen),
                kappa=args.kappa,
                nev=nev,
                max_levels=args.max_levels,
                max_coarse=args.max_coarse,
                max_density=args.max_density,
                tol=args.tol,
                maxiter=args.maxiter,
                restart=args.restart,
                solver_kwargs=solver_kwargs,
            )
            _print_result(name, out)

            if args.per_level:
                _print_per_level(out["ml"])

            rows.append(dict(case=f.stem, n=n, **out))

    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
