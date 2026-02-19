"""LS–AMG–DD experimental: stats/timing utilities (refactor-only scaffolding)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time

import numpy as np


@dataclass
class LsddLevelStats:
    """Per-level setup stats for LS–AMG–DD (no algorithmic effect)."""

    level: int
    n_fine: int
    n_aggs: Optional[int] = None
    n_coarse: Optional[int] = None
    timings: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def timeit(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + (time.perf_counter() - t0)


def _min_med_max(x):
    x = np.asarray(x, dtype=float)
    return float(np.min(x)), float(np.median(x)), float(np.max(x))



def _store_mmx(extra: dict, base: str, arr) -> None:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return
    extra[f"{base}_min"] = float(np.min(a))
    extra[f"{base}_med"] = float(np.median(a))
    extra[f"{base}_max"] = float(np.max(a))


def _lsdd_finalize_level_stats(*, stats, level, blocksize, eigvals_kept, n_coarse: int) -> None:
    stats.n_aggs = getattr(level, "N", None)
    stats.n_coarse = int(n_coarse)
    stats.extra["cr"] = float(stats.n_fine / n_coarse) if n_coarse > 0 else float("inf")

    # omega = |nonoverlapping_subdomain[i]|
    if hasattr(level, "nIi"):
        _store_mmx(stats.extra, "omega", level.nIi)

    # OMEGA = |overlapping_subdomain[i]|
    if blocksize is not None:
        _store_mmx(stats.extra, "OMEGA", blocksize)

    # GAMMA = OMEGA - omega
    if blocksize is not None and hasattr(level, "nIi"):
        Gamma = np.asarray(blocksize, dtype=float) - np.asarray(level.nIi, dtype=float)
        _store_mmx(stats.extra, "GAMMA", Gamma)

    # nev stats
    if hasattr(level, "nev"):
        _store_mmx(stats.extra, "nev", level.nev)

    # kept eigenvalue stats
    if eigvals_kept:
        ev = np.asarray(eigvals_kept, dtype=float)
        stats.extra["eig_min"] = float(np.min(ev))
        stats.extra["eig_med"] = float(np.median(ev))
        stats.extra["eig_max"] = float(np.max(ev))

    # threshold
    if hasattr(level, "threshold"):
        stats.extra["thr"] = float(level.threshold)

    # keep it on the level
    level.lsdd_stats = stats


def _fmt_time_ms(t: float) -> str:
    return f"{1e3*t:.1f}ms" if t < 1.0 else f"{t:.2f}s"

def _fmt(x) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-2 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.3g}"


def _mmx(extra: dict, base: str) -> str:
    a = extra.get(f"{base}_min")
    b = extra.get(f"{base}_med")
    c = extra.get(f"{base}_max")
    if a is None or b is None or c is None:
        return "n/a"
    return f"{_fmt(a)}/{_fmt(b)}/{_fmt(c)}"


def _fmt_ms(t: float) -> str:
    return f"{t*1e3:7.1f}ms" if t < 1.0 else f"{t:7.2f}s"


def _lsdd_print_level_summary(stats, *, print_info: bool, prefix: str = "LS-DD", indent: str = "") -> None:
    if not print_info:
        return

    n_c = stats.n_coarse if stats.n_coarse is not None else "?"
    cr = _fmt(stats.extra.get("cr", "n/a"))
    print(f"{indent}{prefix:<3}  level={stats.level:<2d}  n={stats.n_fine:<7d} -> {n_c:<7}  cr={cr}")

    print(f"{indent}     subdomains (min/med/max):")
    print(f"{indent}       omega : {_mmx(stats.extra, 'omega')}")
    print(f"{indent}       GAMMA : {_mmx(stats.extra, 'GAMMA')}")
    print(f"{indent}       OMEGA : {_mmx(stats.extra, 'OMEGA')}")

    eig = "n/a"
    if all(k in stats.extra for k in ("eig_min", "eig_med", "eig_max")):
        eig = f"{_fmt(stats.extra['eig_min'])}/{_fmt(stats.extra['eig_med'])}/{_fmt(stats.extra['eig_max'])}"
    print(f"{indent}     coarse:")
    print(f"{indent}       nev   : {_mmx(stats.extra, 'nev')}")
    print(f"{indent}       eig   : {eig}")
    print(f"{indent}       thr   : {_fmt(stats.extra.get('thr', 'n/a'))}")


    # ---- RAS profile (if present) ----
    ras_keys = [k for k in stats.timings if k.startswith("ras_")]
    if ras_keys:
        nb = stats.extra.get("ras_nblocks", None)
        print(f"{indent}     RAS profile  #blocks: {nb}. min/med/max: {_mmx(stats.extra, 'ras')}:")
        for k in ("ras_extract", "ras_invert", "ras_potrf", "ras_potri", "ras_symmetrize", "ras_gelss", "ras_total"):
            if k in stats.timings:
                print(f"{indent}       {k[4:]:<12} {_fmt_ms(stats.timings[k])}")

        

    order = ["filter", "strength", "aggregate", "overlap", "extract_PCM", "extract_A", "outerprod", "gep", "assemble_P", "prerel_stp", "pstrel_stp", "coarsen"]
    total = 0.0
    print(f"{indent}     timing:")
    for k in order:
        if k in stats.timings:
            v = stats.timings[k]
            total += v
            print(f"{indent}       {k:<11} {_fmt_ms(v)}")
    print(f"{indent}       {'total':<11} {_fmt_ms(total)}")
