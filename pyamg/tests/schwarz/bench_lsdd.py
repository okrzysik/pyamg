"""Benchmark LS–AMG–DD setup/solve on saved B matrices.

Run from pyamg/schwarz/ (your current working dir):
  python ../tests/schwarz/bench_lsdd.py --data ../tests/schwarz/data --aggregate standard --coarsen 8 10

Options:
  --solver {ref,exp,both}   which solver(s) to run
  --per-level              print per-level timing summaries (if present)
  --csv out.csv            write a CSV summary
"""

from __future__ import annotations

import argparse
import csv
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
    per_level: bool,
):
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
        print_info=False,  # keep bench output clean; use --per-level if desired
    )
    setup_time = time.perf_counter() - t0

    res: list[float] = []
    M = ml.aspreconditioner(cycle="V")

    t1 = time.perf_counter()
    x, info = fgmres(A, b, tol=tol, restart=restart, maxiter=maxiter, M=M, residuals=res)
    solve_time = time.perf_counter() - t1

    iters, cf, final_res = _conv_factor(res)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Directory containing B_n*.npz")
    p.add_argument("--solver", choices=["ref", "exp", "both"], default="both")
    p.add_argument("--aggregate", type=str, default="standard")
    p.add_argument("--coarsen", type=int, nargs="+", default=[8, 10])
    p.add_argument("--kappa", type=float, default=50.0)
    p.add_argument("--nev", type=int, default=0, help="0 means None (threshold-based)")
    p.add_argument("--max-levels", type=int, default=10)
    p.add_argument("--max-coarse", type=int, default=10)
    p.add_argument("--max-density", type=float, default=0.25)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--restart", type=int, default=100)
    p.add_argument("--per-level", action="store_true")
    p.add_argument("--csv", type=str, default="")
    args = p.parse_args()

    data_dir = Path(args.data)
    import re
    def _n_from_name(p: Path) -> int:
        m = re.search(r"_n(\d+)$", p.stem)
        return int(m.group(1)) if m else 0
    files = sorted(data_dir.glob("B_n*.npz"), key=_n_from_name)


    if not files:
        raise FileNotFoundError(f"No B_n*.npz found in {data_dir}")
    
    if args.per_level:
        print(
            "Legend: omega=|nonoverlapping aggregate|, OMEGA=|overlapping subdomain|, "
            "GAMMA=OMEGA-omega (interface size inside the overlap block). "
            "nev = # eigenvectors kept per aggregate. eig = eigenvalues kept (columns of P)."
        )


    # Import here so it uses your editable install cleanly
    from pyamg.schwarz.least_squares_dd import least_squares_dd_solver as pyamg_dd_ref
    from pyamg.schwarz.least_squares_dd_exp import least_squares_dd_solver_exp as pyamg_dd_exp

    solvers = []
    if args.solver in ("ref", "both"):
        solvers.append(("ref", pyamg_dd_ref))
    if args.solver in ("exp", "both"):
        solvers.append(("exp", pyamg_dd_exp))

    rows = []
    for f in files:
        B = sparse.load_npz(f).tocsr()
        A = (B.T @ B).tocsr()
        n = A.shape[0]

        rng = np.random.default_rng(n)
        b = rng.standard_normal(n)

        nev = None if args.nev == 0 else args.nev

        print(f"\n=== {f.name} (n={n}) ===")
        for name, fn in solvers:
            out = _run_one(
                name,
                fn,
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
                per_level=args.per_level,
            )
            print(
                f"{name:>3} | setup={out['setup_time']:.2f}s "
                f"solve={out['solve_time']:.2f}s iters={out['iters']:3d} "
                f"cf={out['conv_factor']:.3f} oc={out['oc']:.2f} final_res={out['final_res']:.2e}"
            )
            # Optional: per-level summaries if you’ve attached lsdd_stats to levels
            from pyamg.schwarz.lsdd.stats import _lsdd_print_level_summary
            # if args.per_level:
            #     for lev in out['ml'].levels:
            #         s = getattr(lev, "lsdd_stats", None)
            #         if s is not None:
            #             _lsdd_print_level_summary(s, print_info=True, prefix="", indent="")

            if args.per_level:
                printed = False
                for lev in out['ml'].levels:
                    s = getattr(lev, "lsdd_stats", None)
                    if s is None:
                        continue
                    if printed or s:
                        print("" + "-" * 72)
                    printed = True
                    _lsdd_print_level_summary(s, print_info=True, prefix="", indent="")



            row = dict(case=f.stem, n=n, **out)
            rows.append(row)



    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()

