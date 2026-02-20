"""Timing and diagnostic reporting for LS–AMG–DD setup.

This module provides:
  - A small per-level timing collector (`LsddLevelStats`) that supports labeled timers.
  - Helpers to compute min/median/max summaries of per-aggregate quantities.
  - A compact, human-readable per-level summary printer.

Typical usage
-------------
Within hierarchy construction, create a `LsddLevelStats` for the current level:

    stats = LsddLevelStats(level=ell, n_fine=A.shape[0])
    with stats.timeit("aggregate"):
        ... do aggregation ...
    with stats.timeit("gep"):
        ... solve local eigenproblems ...
    _lsdd_finalize_level_stats(stats=stats, level=level, ...)
    _lsdd_print_level_summary(stats, print_info=print_info)

The caller decides which timer keys are used; this module simply stores them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
import time

from .types import LSDDLevel

import numpy as np


@dataclass(slots=True)
class LsddLevelStats:
    """Per-level setup timings and summary statistics.

    Attributes
    ----------
    level
        Multigrid level index (0 = finest).
    n_fine
        Fine dimension on this level.
    n_aggs
        Number of aggregates on this level (filled in finalize).
    n_coarse
        Coarse dimension produced by this level (filled in finalize).
    timings
        Dict mapping timer keys to elapsed seconds.
    extra
        Dict for derived metrics and summary scalars (coarsening ratio, min/med/max, etc.).
    """

    level: int
    n_fine: int
    n_aggs: int | None = None
    n_coarse: int | None = None
    timings: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def timeit(self, key: str):
        """Context manager that accumulates elapsed time under `timings[key]`."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + (time.perf_counter() - t0)


def _store_mmx(extra: dict[str, Any], base: str, arr) -> None:
    """Store min/median/max of an array-like into `extra` under `<base>_{min,med,max}`."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return
    extra[f"{base}_min"] = float(np.min(a))
    extra[f"{base}_med"] = float(np.median(a))
    extra[f"{base}_max"] = float(np.max(a))


def _lsdd_finalize_level_stats(*, stats, level: LSDDLevel, eigvals_kept, n_coarse: int) -> None:
    """Populate derived stats/diagnostics for a completed hierarchy extension.

    Parameters
    ----------
    stats
        The stats object for this level (mutated in-place).
    level
        The multigrid level object that has just been processed. Expected to contain
        either structured containers (`level.sub`, `level.eigs`) or legacy arrays.
    eigvals_kept
        List of eigenvalues accepted across aggregates on this level (may be empty).
    n_coarse
        Dimension of the coarse space produced at this level.

    Side effects
    ------------
    - Updates `stats.n_aggs`, `stats.n_coarse`, and fields in `stats.extra`.
    - Stores `stats` on the level as `level.lsdd_stats` for later inspection.
    """
    stats.n_aggs = getattr(level, "n_aggs", None)
    stats.n_coarse = int(n_coarse)
    stats.extra["cr"] = float(stats.n_fine / n_coarse) if n_coarse > 0 else float("inf")

    sub = level.sub
    eigs = level.eigs
    if getattr(sub, "pou_rel_error", None) is not None:
        stats.extra["pou_rel_error"] = float(sub.pou_rel_error)


    omega_sizes = sub.n_omega
    OMEGA_sizes = sub.n_OMEGA
    nev_arr = eigs.nev


    if omega_sizes is not None:
        _store_mmx(stats.extra, "omega", omega_sizes)

    if OMEGA_sizes is not None:
        _store_mmx(stats.extra, "OMEGA", OMEGA_sizes)

    if omega_sizes is not None and OMEGA_sizes is not None:
        Gamma = np.asarray(OMEGA_sizes, dtype=float) - np.asarray(omega_sizes, dtype=float)
        _store_mmx(stats.extra, "GAMMA", Gamma)

    if nev_arr is not None:
        _store_mmx(stats.extra, "nev", nev_arr)

    if eigvals_kept:
        ev = np.asarray(eigvals_kept, dtype=float)
        stats.extra["eig_min"] = float(np.min(ev))
        stats.extra["eig_med"] = float(np.median(ev))
        stats.extra["eig_max"] = float(np.max(ev))

    stats.extra["thr"] = float(eigs.threshold)
    level.lsdd_stats = stats


def _fmt(x) -> str:
    """Format a scalar for compact printing."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-2 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.3g}"


def _mmx(extra: dict[str, Any], base: str) -> str:
    """Return `min/med/max` string for `base` as stored in `extra`."""
    a = extra.get(f"{base}_min")
    b = extra.get(f"{base}_med")
    c = extra.get(f"{base}_max")
    if a is None or b is None or c is None:
        return "n/a"
    return f"{_fmt(a)}/{_fmt(b)}/{_fmt(c)}"


def _fmt_ms(t: float) -> str:
    """Format a duration in seconds as either milliseconds or seconds."""
    return f"{t*1e3:7.1f}ms" if t < 1.0 else f"{t:7.2f}s"


def _lsdd_print_level_summary(
    stats: LsddLevelStats,
    *,
    print_info: bool,
    prefix: str = "LS-DD",
    indent: str = "",
) -> None:
    """Print a compact per-level summary of setup diagnostics and timings.

    Parameters
    ----------
    stats
        Per-level stats object that has already been finalized.
    print_info
        If False, does nothing.
    prefix
        Short label prefix printed per level.
    indent
        Optional indentation string (useful if caller nests printing).
    """
    if not print_info:
        return

    n_c = stats.n_coarse if stats.n_coarse is not None else "?"
    cr = _fmt(stats.extra.get("cr", "n/a"))
    print(f"{indent}{prefix:<3}  level={stats.level:<2d}  n={stats.n_fine:<7d} -> {n_c:<7}  cr={cr}")

    print(f"{indent}     subdomains (min/med/max):")
    print(f"{indent}       omega : {_mmx(stats.extra, 'omega')}")
    print(f"{indent}       GAMMA : {_mmx(stats.extra, 'GAMMA')}")
    print(f"{indent}       OMEGA : {_mmx(stats.extra, 'OMEGA')}")

    if "pou_rel_error" in stats.extra:
        print(f"{indent}       PoU err: {_fmt(stats.extra['pou_rel_error'])}")


    eig = "n/a"
    if all(k in stats.extra for k in ("eig_min", "eig_med", "eig_max")):
        eig = f"{_fmt(stats.extra['eig_min'])}/{_fmt(stats.extra['eig_med'])}/{_fmt(stats.extra['eig_max'])}"

    print(f"{indent}     coarse:")
    print(f"{indent}       nev   : {_mmx(stats.extra, 'nev')}")
    print(f"{indent}       eig   : {eig}")
    print(f"{indent}       thr   : {_fmt(stats.extra.get('thr', 'n/a'))}")

    # Optional RAS profile block (only if timing keys exist)
    ras_keys = [k for k in stats.timings if k.startswith("ras_")]
    if ras_keys:
        nb = stats.extra.get("ras_nblocks", None)
        print(f"{indent}     RAS profile  #blocks: {nb}. min/med/max: {_mmx(stats.extra, 'ras')}:")
        for k in (
            "ras_extract",
            "ras_invert",
            "ras_potrf",
            "ras_potri",
            "ras_symmetrize",
            "ras_gelss",
            "ras_total",
        ):
            if k in stats.timings:
                print(f"{indent}       {k[4:]:<12} {_fmt_ms(stats.timings[k])}")


    # Filtering diagnostics (if enabled)
    fb = stats.extra.get("filter_B_nnz_before")
    fa = stats.extra.get("filter_B_nnz_after")
    if fb is not None and fa is not None and fb > 0:
        drop = 1.0 - (fa / fb)
        print(f"{indent}     filter:")
        print(f"{indent}       B   nnz: {fb} -> {fa}  drop={_fmt(drop)}")

    fbtb = stats.extra.get("filter_BT_nnz_before")
    fbta = stats.extra.get("filter_BT_nnz_after")
    if fbtb is not None and fbta is not None and fbtb > 0:
        drop = 1.0 - (fbta / fbtb)
        print(f"{indent}       BT  nnz: {fbtb} -> {fbta}  drop={_fmt(drop)}")

    fab = stats.extra.get("filter_A_nnz_before")
    faa = stats.extra.get("filter_A_nnz_after")
    if fab is not None and faa is not None and fab > 0:
        drop = 1.0 - (faa / fab)
        print(f"{indent}       A   nnz: {fab} -> {faa}  drop={_fmt(drop)}")

    # Timing summary block (only if timing keys exist)
    order = [
        "filter",
        "strength",
        "aggregate",
        "overlap",
        "extract_PCM",
        "extract_A",
        "outerprod",
        "gep",
        "assemble_P",
        "prerel_stp",
        "pstrel_stp",
        "coarsen",
    ]
    total = 0.0
    print(f"{indent}     timing:")
    for k in order:
        if k in stats.timings:
            v = stats.timings[k]
            total += v
            print(f"{indent}       {k:<11} {_fmt_ms(v)}")
    print(f"{indent}       {'total':<11} {_fmt_ms(total)}")


def _lsdd_print_setup_summary(*, smoother_setup_time: float, print_info: bool, indent: str = "") -> None:
    """Print non-level-specific setup timings.

    Parameters
    ----------
    smoother_setup_time
        Total wall time spent constructing / attaching smoothers (seconds).
    print_info
        If False, does nothing.
    indent
        Optional indentation prefix.
    """
    if not print_info:
        return
    print(f"{indent}LS-DD  smoother_setup  {_fmt_ms(float(smoother_setup_time))}")
