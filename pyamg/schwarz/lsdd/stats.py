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


def _lsdd_print_level_summary(stats: LsddLevelStats, *, print_info: bool, prefix: str = "LS-DD") -> None:
    """Print a one-line summary for a level (only if print_info=True)."""
    if not print_info:
        return

    parts = [f"{prefix} ℓ={stats.level}", f"n={stats.n_fine}"]
    if stats.n_aggs is not None:
        parts.append(f"n_aggs={stats.n_aggs}")
    if stats.n_coarse is not None:
        parts.append(f"n_c={stats.n_coarse}")

    if stats.timings:
        order = [
            "filter",
            "strength",
            "aggregate",
            "overlap",
            "extract_A",
            "outerprod",
            "gep",
            "assemble_P",
            "coarsen",
        ]
        t_parts = []
        for k in order:
            if k in stats.timings:
                t_parts.append(f"{k}={stats.timings[k]:.3f}s")
        for k, v in sorted(stats.timings.items()):
            if k not in order:
                t_parts.append(f"{k}={v:.3f}s")
        parts.append("timings:[" + ", ".join(t_parts) + "]")

    if stats.extra:
        extras = ", ".join(f"{k}={v}" for k, v in stats.extra.items())
        parts.append("extra:[" + extras + "]")

    print(" | ".join(parts))


def _lsdd_finalize_level_stats(
    *,
    stats: LsddLevelStats,
    level,
    blocksize,
    eigvals_kept,
    n_coarse: int,
) -> None:
    """Populate stats.{n_aggs,n_coarse,extra} from the current level's artifacts."""
    stats.n_aggs = getattr(level, "N", None)
    stats.n_coarse = int(n_coarse)

    # Coarsening ratio (fine / coarse)
    stats.extra["cr"] = float(stats.n_fine / n_coarse) if n_coarse > 0 else float("inf")

    # Aggregate size stats (ω)
    if hasattr(level, "nIi"):
        omega = np.asarray(level.nIi, dtype=float)
        stats.extra["omega_mean"] = float(np.mean(omega))
        stats.extra["omega_med"] = float(np.median(omega))
        stats.extra["omega_max"] = int(np.max(omega))

    # Overlap size stats (Ω)
    if blocksize is not None:
        Omega = np.asarray(blocksize, dtype=float)
        stats.extra["Omega_mean"] = float(np.mean(Omega))
        stats.extra["Omega_med"] = float(np.median(Omega))
        stats.extra["Omega_max"] = int(np.max(Omega))

    # Eigenvectors per aggregate
    if hasattr(level, "nev"):
        nev = np.asarray(level.nev, dtype=float)
        stats.extra["nev_mean"] = float(np.mean(nev))
        stats.extra["nev_med"] = float(np.median(nev))
        stats.extra["nev_max"] = int(np.max(nev))

    # Eigenvalue stats over kept eigenvalues
    if eigvals_kept:
        ev = np.asarray(eigvals_kept, dtype=float)
        stats.extra["eig_min"] = float(np.min(ev))
        stats.extra["eig_med"] = float(np.median(ev))
        stats.extra["eig_mean"] = float(np.mean(ev))
        stats.extra["eig_max"] = float(np.max(ev))

    # Other scalars
    if hasattr(level, "threshold"):
        stats.extra["thr"] = float(level.threshold)
    if hasattr(level, "min_ev"):
        stats.extra["min_ev"] = float(level.min_ev)

    # Optional: stash stats on the level for later inspection/debug
    level.lsdd_stats = stats

