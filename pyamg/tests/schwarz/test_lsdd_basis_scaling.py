"""Unit tests for optional LS-DD aggregate-wise basis scaling.

Run from the ``pyamg`` repository root:

    python3 -m pytest -q pyamg/tests/schwarz/test_lsdd_basis_scaling.py

Run one specific test:

    python3 -m pytest -q \
      pyamg/tests/schwarz/test_lsdd_basis_scaling.py::test_scale_basis_triplets_preserves_coarse_space_projector

Show internal diagnostic values (selected rows, condition numbers, errors):

    PYAMG_LSDD_BSCALE_TEST_DEBUG=1 python3 -m pytest -q -s \
      pyamg/tests/schwarz/test_lsdd_basis_scaling.py

Run optional end-to-end saved-matrix experiment:

    PYAMG_RUN_BSCALE_E2E=1 python3 -m pytest -q -s \
      pyamg/tests/schwarz/test_lsdd_basis_scaling.py::test_basis_scaling_e2e_saved_matrix_opt_in

With drop tolerance in e2e:

    PYAMG_RUN_BSCALE_E2E=1 PYAMG_LSDD_BSCALE_DROP_TOL=1e-12 \
      python3 -m pytest -q -s \
      pyamg/tests/schwarz/test_lsdd_basis_scaling.py::test_basis_scaling_e2e_saved_matrix_opt_in

Experiment with pruning tiny entries:

    python3 -m pytest -q -s \
      pyamg/tests/schwarz/test_lsdd_basis_scaling.py::test_scale_basis_triplets_drop_tol_zeros_tiny_entries
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import csr_array

from pyamg.krylov import fgmres
from pyamg.schwarz.least_squares_dd_exp import least_squares_dd_solver_exp
from pyamg.schwarz.lsdd.basis_scaling import (
    _lsdd_scale_basis_triplets_for_fill,
    _lsdd_scale_one_triplet_block,
)
from pyamg.schwarz.lsdd.types import EigenInfo, Subdomains


def _dbg(msg: str) -> None:
    """Print debug details for this module when enabled by env var."""
    if os.environ.get("PYAMG_LSDD_BSCALE_TEST_DEBUG", "0") == "1":
        print(msg)


def _env_flag(name: str) -> bool:
    """Return True when env var is set to string "1"."""
    return os.environ.get(name, "0") == "1"


def _env_float(name: str, default: float) -> float:
    """Return float env var value or default when unset."""
    val = os.environ.get(name, "")
    if val == "":
        return float(default)
    return float(val)


def _extract_n(path: Path) -> int:
    """Extract matrix size suffix from test data filename."""
    match = re.search(r"n(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _assemble_dense_from_triplets(
    *,
    n_rows: int,
    n_cols: int,
    p_r: list[np.ndarray],
    p_c: list[np.ndarray],
    p_v: list[np.ndarray],
) -> np.ndarray:
    """Assemble a dense matrix from LS-DD-style triplets."""
    P = np.zeros((n_rows, n_cols), dtype=float)
    for rows, cols, vals in zip(p_r, p_c, p_v):
        rr = np.asarray(rows, dtype=np.int32)
        cc = np.asarray(cols, dtype=np.int32)
        vv = np.asarray(vals, dtype=float)
        P[rr, cc] = vv
    return P


def _build_test_level(
    *,
    n_aggs: int,
    nev: list[int],
    r_rows: list[np.ndarray],
):
    """Build a minimal level-like object for triplet scaling tests."""
    sub = Subdomains.allocate(n_aggs)
    for i, rows_i in enumerate(r_rows):
        sub.R_rows[i] = np.asarray(rows_i, dtype=np.int32)

    eigs = EigenInfo.allocate(n_aggs)
    eigs.nev[:] = np.asarray(nev, dtype=np.int32)

    return SimpleNamespace(sub=sub, eigs=eigs)


def test_scale_one_triplet_block_b_nodal_selected_rows_become_identity() -> None:
    """`b_nodal` scaling yields coordinate rows on the selected local functionals."""
    rng = np.random.default_rng(0)
    n_cols = 13
    n_rows = 19
    n_omega = 8
    k = 3

    omega_rows = np.array([0, 2, 4, 5, 7, 9, 10, 12], dtype=np.int32)
    R_rows_i = np.array([1, 2, 4, 8, 9, 11, 13, 14, 16, 17], dtype=np.int32)

    B = csr_array(rng.standard_normal((n_rows, n_cols)))
    Z = rng.standard_normal((n_omega, k))
    row_width = np.ones(n_rows, dtype=float)

    Z_scaled, info = _lsdd_scale_one_triplet_block(
        B=B,
        method="b_nodal",
        omega_rows=omega_rows,
        Z=Z,
        R_rows_i=R_rows_i,
        row_width=row_width,
        cond_max=1.0e12,
        weight_power=1.0,
        normalize_columns=False,
    )

    assert Z_scaled is not None
    selected = np.asarray(info["selected"], dtype=np.int32)
    cond_c = float(info.get("cond", np.nan))
    B_i = B[R_rows_i, :][:, omega_rows]
    Y = np.asarray(B_i @ Z)
    C = Y[selected, :]
    Y_scaled = Y @ np.linalg.solve(C, np.eye(k))
    err = float(np.linalg.norm(Y_scaled[selected, :] - np.eye(k)))
    _dbg(f"[identity] selected={selected.tolist()} cond(C)={cond_c:.6e} err={err:.6e}")
    assert np.allclose(Y_scaled[selected, :], np.eye(k), atol=1e-10, rtol=1e-10)


def test_scale_basis_triplets_preserves_triplet_layout() -> None:
    """Triplet scaling preserves count, support, and coarse-column indexing arrays."""
    rng = np.random.default_rng(1)
    n_cols = 12
    n_b_rows = 11
    nev = [2, 3]

    level = _build_test_level(
        n_aggs=2,
        nev=nev,
        r_rows=[
            np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.array([4, 5, 6, 7, 8, 9], dtype=np.int32),
        ],
    )

    rows0 = np.array([0, 1, 2, 3], dtype=np.int32)
    rows1 = np.array([5, 6, 7, 8, 9], dtype=np.int32)
    p_r = [
        rows0.copy(),
        rows0.copy(),
        rows1.copy(),
        rows1.copy(),
        rows1.copy(),
    ]
    p_c = [
        np.full(rows0.size, 0, dtype=np.int32),
        np.full(rows0.size, 1, dtype=np.int32),
        np.full(rows1.size, 2, dtype=np.int32),
        np.full(rows1.size, 3, dtype=np.int32),
        np.full(rows1.size, 4, dtype=np.int32),
    ]
    p_v = [
        rng.standard_normal(rows0.size),
        rng.standard_normal(rows0.size),
        rng.standard_normal(rows1.size),
        rng.standard_normal(rows1.size),
        rng.standard_normal(rows1.size),
    ]

    p_r_before = [x.copy() for x in p_r]
    p_c_before = [x.copy() for x in p_c]
    p_v_before = [x.copy() for x in p_v]
    stats = SimpleNamespace(extra={})
    B = csr_array(rng.standard_normal((n_b_rows, n_cols)))

    _lsdd_scale_basis_triplets_for_fill(
        level=level,
        B=B,
        p_r=p_r,
        p_c=p_c,
        p_v=p_v,
        method="b_nodal",
        cond_max=1.0e12,
        weight_power=1.0,
        normalize_columns=False,
        drop_tol=0.0,
        stats=stats,
    )
    _dbg(
        "[triplets] scaled="
        f"{stats.extra.get('basis_scaling_scaled', '?')}/"
        f"{stats.extra.get('basis_scaling_total', '?')} "
        f"skip_empty={stats.extra.get('basis_scaling_skipped_empty', '?')} "
        f"skip_rows={stats.extra.get('basis_scaling_skipped_too_few_rows', '?')} "
        f"skip_cond={stats.extra.get('basis_scaling_skipped_bad_conditioning', '?')} "
        f"skip_exc={stats.extra.get('basis_scaling_skipped_exception', '?')}"
    )

    assert len(p_r) == len(p_r_before)
    assert len(p_c) == len(p_c_before)
    assert len(p_v) == len(p_v_before)
    for a, b in zip(p_r, p_r_before):
        assert np.array_equal(a, b)
    for a, b in zip(p_c, p_c_before):
        assert np.array_equal(a, b)
    assert "basis_scaling_scaled" in stats.extra


def test_scale_basis_triplets_preserves_coarse_space_projector() -> None:
    """Right-scaling of local basis leaves the coarse correction operator invariant."""
    rng = np.random.default_rng(2)
    n_fine = 11
    n_b_rows = 9
    nev = [2, 2]

    level = _build_test_level(
        n_aggs=2,
        nev=nev,
        r_rows=[
            np.array([0, 1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6, 7], dtype=np.int32),
        ],
    )

    rows0 = np.array([0, 1, 2, 3], dtype=np.int32)
    rows1 = np.array([6, 7, 8, 9], dtype=np.int32)
    p_r = [
        rows0.copy(),
        rows0.copy(),
        rows1.copy(),
        rows1.copy(),
    ]
    p_c = [
        np.full(rows0.size, 0, dtype=np.int32),
        np.full(rows0.size, 1, dtype=np.int32),
        np.full(rows1.size, 2, dtype=np.int32),
        np.full(rows1.size, 3, dtype=np.int32),
    ]
    p_v = [
        rng.standard_normal(rows0.size),
        rng.standard_normal(rows0.size),
        rng.standard_normal(rows1.size),
        rng.standard_normal(rows1.size),
    ]

    P_before = _assemble_dense_from_triplets(n_rows=n_fine, n_cols=4, p_r=p_r, p_c=p_c, p_v=p_v)
    M = rng.standard_normal((n_fine, n_fine))
    A = M.T @ M + n_fine * np.eye(n_fine)

    B = csr_array(rng.standard_normal((n_b_rows, n_fine)))
    stats = SimpleNamespace(extra={})
    _lsdd_scale_basis_triplets_for_fill(
        level=level,
        B=B,
        p_r=p_r,
        p_c=p_c,
        p_v=p_v,
        method="p_nodal",
        cond_max=1.0e12,
        weight_power=0.0,
        normalize_columns=False,
        drop_tol=0.0,
        stats=stats,
    )
    P_after = _assemble_dense_from_triplets(n_rows=n_fine, n_cols=4, p_r=p_r, p_c=p_c, p_v=p_v)

    E_before = P_before @ np.linalg.inv(P_before.T @ A @ P_before) @ P_before.T
    E_after = P_after @ np.linalg.inv(P_after.T @ A @ P_after) @ P_after.T
    proj_err = float(np.linalg.norm(E_before - E_after) / max(np.linalg.norm(E_before), 1.0))
    _dbg(f"[projector] relative_diff={proj_err:.6e}")
    assert np.allclose(E_before, E_after, atol=1e-10, rtol=1e-10)


def test_basis_scaling_e2e_saved_matrix_opt_in() -> None:
    """Opt-in end-to-end check on saved Schwarz data with basis scaling on/off."""
    if not _env_flag("PYAMG_RUN_BSCALE_E2E"):
        pytest.skip("Set PYAMG_RUN_BSCALE_E2E=1 to run end-to-end saved-matrix experiment")

    data_dir = Path(__file__).resolve().parent / "data"
    b_files = sorted(data_dir.glob("B_n*.npz"), key=_extract_n)
    if not b_files:
        pytest.skip(f"No B_n*.npz files found in {data_dir}")

    b_path = b_files[0]
    B = sparse.load_npz(b_path).tocsr()
    A = (B.T @ B).tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(n)
    b = rng.standard_normal(n)
    drop_tol = _env_float("PYAMG_LSDD_BSCALE_DROP_TOL", 0.0)

    def _collect_level_coarsen_nnz(ml) -> list[tuple[int, int, int]]:
        """Return per-level tuples: (level_id, coarsen_A_nnz, coarsen_B_nnz)."""
        out: list[tuple[int, int, int]] = []
        for lev, level in enumerate(ml.levels):
            stats = getattr(level, "lsdd_stats", None)
            if stats is None:
                continue
            extra = dict(getattr(stats, "extra", {}))
            if "coarsen_A_nnz" not in extra:
                continue
            a_nnz = int(extra["coarsen_A_nnz"])
            b_nnz = int(extra["coarsen_B_nnz"]) if "coarsen_B_nnz" in extra else -1
            out.append((lev, a_nnz, b_nnz))
        return out

    def _run(mode: str, weight_power: float) -> dict[str, float | int | list[tuple[int, int, int]]]:
        ml = least_squares_dd_solver_exp(
            B=B,
            BT=None,
            A=A,
            symmetry="symmetric",
            aggregate="standard",
            agg_levels=2,
            presmoother="ras",
            postsmoother="rasT",
            kappa=50.0,
            min_coarsening=[8, 10],
            nev=None,
            threshold=None,
            max_levels=10,
            max_coarse=10,
            max_density=0.25,
            print_info=False,
            robust_Sker_handling=True,
            basis_scaling=mode,
            basis_scaling_cond_max=1.0e8,
            basis_scaling_weight_power=weight_power,
            basis_scaling_normalize_columns=False,
            basis_scaling_drop_tol=drop_tol,
        )

        M = ml.aspreconditioner(cycle="V")
        residuals: list[float] = []
        _, info = fgmres(A, b, tol=1.0e-8, restart=100, maxiter=80, M=M, residuals=residuals)
        res = np.asarray(residuals, dtype=float)
        ratio = float(res[-1] / res[0]) if res.size >= 2 and res[0] > 0.0 else float("inf")

        stats0 = getattr(ml.levels[0], "lsdd_stats", None)
        extra = {} if stats0 is None else dict(getattr(stats0, "extra", {}))
        scaled = int(extra.get("basis_scaling_scaled", 0))
        total = int(extra.get("basis_scaling_total", 0))
        nnz_a = int(extra["coarsen_A_nnz"]) if "coarsen_A_nnz" in extra else -1
        nnz_b = int(extra["coarsen_B_nnz"]) if "coarsen_B_nnz" in extra else -1
        level_nnz = _collect_level_coarsen_nnz(ml)

        return {
            "n_coarse0": int(ml.levels[0].P.shape[1]),
            "iters": int(max(res.size - 1, 0)),
            "ratio": ratio,
            "info": int(info),
            "scaled": scaled,
            "total": total,
            "coarsen_A_nnz": nnz_a,
            "coarsen_B_nnz": nnz_b,
            "oc": float(ml.operator_complexity()),
            "level_nnz": level_nnz,
        }

    runs: list[tuple[str, str, float]] = [
        ("none", "none", 1.0),
        ("p_nodal", "p_nodal", 1.0),
        ("b_nodal_unweighted", "b_nodal", 0.0),
        ("b_nodal_weighted", "b_nodal", 50.0),
    ]
    results: dict[str, dict[str, float | int | list[tuple[int, int, int]]]] = {}
    for label, mode, wpow in runs:
        results[label] = _run(mode, wpow)

    base = results["none"]
    _dbg(
        f"[e2e:{b_path.name}] baseline none: drop_tol={drop_tol:.3e} "
        f"nC={base['n_coarse0']} it={base['iters']} ratio={base['ratio']:.3e} oc={base['oc']:.3f}"
    )
    _dbg(f"[e2e:{b_path.name}] baseline level_nnz={base['level_nnz']}")
    for label in ("p_nodal", "b_nodal_unweighted", "b_nodal_weighted"):
        out = results[label]
        da = int(out["coarsen_A_nnz"]) - int(base["coarsen_A_nnz"])
        db = int(out["coarsen_B_nnz"]) - int(base["coarsen_B_nnz"])
        _dbg(
            f"[e2e:{b_path.name}] {label}: nC={out['n_coarse0']} it={out['iters']} "
            f"ratio={out['ratio']:.3e} oc={out['oc']:.3f} "
            f"A_nnz={out['coarsen_A_nnz']} (dA={da:+d}) "
            f"B_nnz={out['coarsen_B_nnz']} (dB={db:+d}) "
            f"scaled={out['scaled']}/{out['total']} level_nnz={out['level_nnz']}"
        )

    for label, out in results.items():
        assert out["iters"] > 0, label
        assert out["ratio"] < 1.0, label
        assert out["n_coarse0"] == base["n_coarse0"], label

    for label in ("p_nodal", "b_nodal_unweighted", "b_nodal_weighted"):
        out = results[label]
        assert out["total"] > 0, label
        assert out["scaled"] > 0, label


def test_scale_basis_triplets_drop_tol_zeros_tiny_entries() -> None:
    """Drop tolerance zeros tiny entries in scaled triplet blocks."""
    rng = np.random.default_rng(7)
    n_cols = 10
    n_b_rows = 10
    nev = [2]

    level = _build_test_level(
        n_aggs=1,
        nev=nev,
        r_rows=[np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)],
    )
    rows0 = np.array([0, 1, 2, 3], dtype=np.int32)
    p_r = [rows0.copy(), rows0.copy()]
    p_c = [
        np.full(rows0.size, 0, dtype=np.int32),
        np.full(rows0.size, 1, dtype=np.int32),
    ]
    p_v = [
        np.array([1.0, 1.0e-13, -1.0e-13, 1.0], dtype=float),
        np.array([1.0e-13, 1.0, 1.0, -1.0e-13], dtype=float),
    ]
    B = csr_array(rng.standard_normal((n_b_rows, n_cols)))
    stats = SimpleNamespace(extra={})

    _lsdd_scale_basis_triplets_for_fill(
        level=level,
        B=B,
        p_r=p_r,
        p_c=p_c,
        p_v=p_v,
        method="p_nodal",
        cond_max=1.0e14,
        weight_power=0.0,
        normalize_columns=False,
        drop_tol=1.0e-12,
        stats=stats,
    )

    dropped = int(stats.extra.get("basis_scaling_entries_dropped", 0))
    assert dropped > 0
