"""Observed two-level damping sweeps for a fixed Schwarz smoother choice.

This module provides an observed-only sweep driver that:

1. builds the two-level hierarchy once,
2. fixes one Schwarz subdomain layout (overlap or non-overlap),
3. computes ``rho(M^{-1}A)`` once for that fixed layout,
4. sweeps normalized damping ``zeta_norm`` in ``(0, 2)`` with
   ``zeta_raw = zeta_norm / rho(M^{-1}A)``,
5. measures observed two-grid quantities ``q_obs``, ``K_obs``, ``rho_obs``.

The implementation reuses existing LS-DD infrastructure:
- ``build_observed_two_level_solver`` for hierarchy construction,
- ``estimate_qobs_homogeneous_two_grid`` for observed-cycle estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np

from ....relaxation import relaxation as _relaxation
from ....relaxation.smoothing import change_smoothers as _change_smoothers
from ....relaxation.smoothing import rho_additive_schwarz_A as _rho_additive_schwarz_A
from .models import RandomDistribution, TwoLevelSolverParams
from .observed import (
    build_observed_two_level_solver as _build_observed_two_level_solver,
)
from .observed import (
    estimate_qobs_homogeneous_two_grid as _estimate_qobs_homogeneous_two_grid,
)
from .reporting import fmt_sig as _fmt_sig
from .reporting import timer_add as _timer_add

ObservedSmootherType = Literal["block_jacobi", "asm_overlap", "msm", "ras"]
FixedSmootherMode = Literal["symmetric", "forward"]


@dataclass(slots=True, frozen=True)
class ObservedDampingSweepConfig:
    """Configuration for observed damping sweeps with a fixed smoother type."""

    solver_params: TwoLevelSolverParams = field(default_factory=TwoLevelSolverParams)
    smoother_type: ObservedSmootherType = "block_jacobi"
    fixed_smoother_mode: FixedSmootherMode = "symmetric"

    zeta_norm_values: np.ndarray | None = None
    zeta_norm_min: float = 0.05
    zeta_norm_max: float = 1.95
    n_zeta: int = 19

    distribution: RandomDistribution = "gaussian"
    seed: int | None = 0
    observed_n_samples: int = 10
    observed_n_cycles: int = 15

    collect_timers: bool = False


@dataclass(slots=True, frozen=True)
class ObservedDampingSweepResult:
    """Observed two-level sweep results for one fixed smoother type."""

    smoother_type: ObservedSmootherType
    rho_MinvA: float
    zeta_norm_values: np.ndarray
    zeta_raw_values: np.ndarray
    q_obs: np.ndarray
    K_obs: np.ndarray
    rho_obs: np.ndarray
    timings: dict[str, float] | None = None


def _zeta_norm_grid(cfg: ObservedDampingSweepConfig) -> np.ndarray:
    """Return validated normalized damping grid ``zeta_norm`` in ``(0, 2)``."""
    if cfg.zeta_norm_values is not None:
        vals = np.asarray(cfg.zeta_norm_values, dtype=float).reshape(-1)
        if vals.size == 0:
            raise ValueError("zeta_norm_values must contain at least one value")
    else:
        n = int(cfg.n_zeta)
        if n <= 0:
            raise ValueError("n_zeta must be positive")
        zmin = float(cfg.zeta_norm_min)
        zmax = float(cfg.zeta_norm_max)
        if not (0.0 < zmin < 2.0 and 0.0 < zmax < 2.0):
            raise ValueError("zeta_norm_min and zeta_norm_max must lie strictly in (0, 2)")
        if zmax <= zmin:
            raise ValueError("zeta_norm_max must be greater than zeta_norm_min")
        vals = np.linspace(zmin, zmax, n, dtype=float)

    if np.any(~np.isfinite(vals)):
        raise ValueError("zeta_norm grid contains non-finite values")
    if np.any(vals <= 0.0) or np.any(vals >= 2.0):
        raise ValueError("all zeta_norm values must lie strictly in (0, 2)")
    return vals


def _flatten_nonoverlap_subdomains(domains) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``omega_i`` lists into Schwarz arrays ``(subdomain, subdomain_ptr)``."""
    ptr = np.zeros(len(domains) + 1, dtype=np.int32)
    chunks: list[np.ndarray] = []
    for i, dom in enumerate(domains):
        if dom is None:
            raise ValueError("Expected non-overlap subdomain indices to be populated")
        arr = np.sort(np.asarray(dom, dtype=np.int32).ravel())
        chunks.append(arr)
        ptr[i + 1] = ptr[i] + int(arr.size)
    flat = np.concatenate(chunks).astype(np.int32, copy=False) if chunks else np.zeros(0, dtype=np.int32)
    return flat, ptr


def _fixed_schwarz_data(level, smoother_type: ObservedSmootherType):
    """Return fixed Schwarz data for the requested smoother type on fine level."""
    if smoother_type == "asm_overlap":
        subdomain = np.asarray(level.blocks.subdomain, dtype=np.int32)
        subdomain_ptr = np.asarray(level.blocks.subdomain_ptr, dtype=np.int32)
    elif smoother_type == "block_jacobi":
        subdomain, subdomain_ptr = _flatten_nonoverlap_subdomains(level.sub.omega)

    # elif smoother_type == "ras":
    #     blocks = level.blocks
    #     sub = level.sub
    #     pou_flat = sub.PoU_flat
    #     if pou_flat is None:
    #         pou_flat = np.concatenate(sub.PoU)
    #         sub.PoU_flat = pou_flat
    #     spec_pre = (
    #         "rest_additive_schwarz",
    #         {
    #             "subdomain": blocks.subdomain,
    #             "subdomain_ptr": blocks.subdomain_ptr,
    #             "POU": pou_flat,
    #             "iterations": 1,
    #         },
    #     )

    else:  # pragma: no cover
        raise ValueError(f"Unsupported smoother_type: {smoother_type!r}")

    Acsr = level.A.tocsr()
    Acsr.sort_indices()
    # The observed builder may have already cached Schwarz data on A for a
    # different subdomain layout. Clear it so we can force recomputation for
    # the requested smoother type (omega vs OMEGA).
    if hasattr(Acsr, "schwarz_parameters"):
        delattr(Acsr, "schwarz_parameters")
    inv_subblock = None
    inv_subblock_ptr = None
    subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr = _relaxation.schwarz_parameters(
        Acsr, subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr
    )
    return Acsr, subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr


def _retune_level0_asm(
    *,
    ml,
    subdomain: np.ndarray,
    subdomain_ptr: np.ndarray,
    inv_subblock: np.ndarray,
    inv_subblock_ptr: np.ndarray,
    omega_raw: float,
) -> None:
    """Rebuild level-0 pre/post smoothers with fixed Schwarz data and new omega."""
    spec = (
        "additive_schwarz",
        {
            "subdomain": subdomain,
            "subdomain_ptr": subdomain_ptr,
            "inv_subblock": inv_subblock,
            "inv_subblock_ptr": inv_subblock_ptr,
            "omega": float(omega_raw),
            "withrho": False,
            "iterations": 1,
        },
    )
    _change_smoothers(ml, presmoother=[spec], postsmoother=[spec])


def _retune_level0_ras(
    *,
    ml,
    subdomain: np.ndarray,
    subdomain_ptr: np.ndarray,
    inv_subblock: np.ndarray,
    inv_subblock_ptr: np.ndarray,
    omega_raw: float,
) -> None:
    """Rebuild level-0 pre/post smoothers with fixed Schwarz data and new omega."""
    spec = (
        "ras",
        {
            "subdomain": subdomain,
            "subdomain_ptr": subdomain_ptr,
            "inv_subblock": inv_subblock,
            "inv_subblock_ptr": inv_subblock_ptr,
            "omega": float(omega_raw),
            "withrho": False,
            "iterations": 1,
            "sweep": "forward",
        },
    )
    _change_smoothers(ml, presmoother=[spec], postsmoother=[spec])


def _retune_level0_fixed_smoother(
    *,
    ml,
    smoother_type: ObservedSmootherType,
    mode: FixedSmootherMode,
) -> None:
    """Install a fixed non-damped smoother on level 0 (msm or ras).

    Parameters
    ----------
    mode
        ``"symmetric"`` uses forward pre and backward/transposed post.
        ``"forward"`` uses forward pre and forward post.
    """
    lvl = ml.levels[0]
    # Clear cached Schwarz block data so setup can rebuild with the subdomain
    # layout required by the selected fixed smoother.
    if hasattr(lvl.A, "schwarz_parameters"):
        delattr(lvl.A, "schwarz_parameters")
    if hasattr(lvl, "Acsr") and hasattr(lvl.Acsr, "schwarz_parameters"):
        delattr(lvl.Acsr, "schwarz_parameters")

    blocks = lvl.blocks
    sub = lvl.sub

    if mode not in ("symmetric", "forward"):
        raise ValueError(f"Unsupported fixed_smoother_mode: {mode!r}")

    if smoother_type == "msm":
        spec_pre = (
            "schwarz",
            {
                "subdomain": blocks.subdomain,
                "subdomain_ptr": blocks.subdomain_ptr,
                "iterations": 1,
                "sweep": "forward",
            },
        )
        if mode == "symmetric":
            spec_post = (
                "schwarz",
                {
                    "subdomain": blocks.subdomain,
                    "subdomain_ptr": blocks.subdomain_ptr,
                    "iterations": 1,
                    "sweep": "backward",
                },
            )
        else:
            spec_post = spec_pre
    elif smoother_type == "ras":
        pou_flat = sub.PoU_flat
        if pou_flat is None:
            pou_flat = np.concatenate(sub.PoU)
            sub.PoU_flat = pou_flat
        #omega = 1.05 # Improves convergence for tau=2 on h-div problem
        #omega = 1.5 # imporoves convergence for tau=7 on h-div problem
        # somehow doing one forward pass per iter is better than symmetric and two forward passes...
        omega = 1.0
        spec_pre = (
            "rest_additive_schwarz",
            {
                "subdomain": blocks.subdomain,
                "subdomain_ptr": blocks.subdomain_ptr,
                "POU": pou_flat,
                "iterations": 1,
                "omega": float(omega),
            },
        )
        if mode == "symmetric":
            spec_post = (
                "rest_additive_schwarzT",
                {
                    "subdomain": blocks.subdomain,
                    "subdomain_ptr": blocks.subdomain_ptr,
                    "POU": pou_flat,
                    "iterations": 1,
                    "omega": float(omega),
                },
            )
            spec_post = None
        else:
            spec_post = spec_pre
    else:  # pragma: no cover
        raise ValueError(f"Unsupported fixed smoother type: {smoother_type!r}")

    _change_smoothers(ml, presmoother=[spec_pre], postsmoother=[spec_post])


def compute_observed_damping_sweep(
    B,
    A=None,
    BT=None,
    *,
    cfg: ObservedDampingSweepConfig = ObservedDampingSweepConfig(),
    print_constants: bool = False,
    constant_sig_digits: int = 3,
    print_timers: bool = False,
) -> ObservedDampingSweepResult:
    """Run observed two-level damping sweep for one fixed smoother type."""
    timers: dict[str, float] | None = {} if (cfg.collect_timers or print_timers) else None
    t_total = perf_counter()

    zeta_norm = _zeta_norm_grid(cfg)

    t_setup = perf_counter()
    ml = _build_observed_two_level_solver(
        B=B,
        A=A,
        BT=BT,
        solver_params=cfg.solver_params,
        zeta=1.0,
        with_rho=False,
        with_rho_perp=False,
        zeta_eff=1.0,
    )
    if len(ml.levels) < 2:
        raise ValueError("Observed damping sweep requires at least two levels")
    level0 = ml.levels[0]
    _timer_add(timers, "stage.setup_once_sec", perf_counter() - t_setup)

    n = int(zeta_norm.size)
    q_obs = np.full(n, np.nan, dtype=float)
    K_obs = np.full(n, np.nan, dtype=float)
    rho_obs = np.full(n, np.nan, dtype=float)

    if cfg.smoother_type in ("block_jacobi", "asm_overlap"):
        Acsr, subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr = _fixed_schwarz_data(
            level0, cfg.smoother_type
        )

        t_rho = perf_counter()
        rho = float(
            _rho_additive_schwarz_A(
                Acsr,
                subdomain,
                subdomain_ptr,
                inv_subblock,
                inv_subblock_ptr,
            )
        )
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError(f"Expected positive finite rho(M^-1 A), got {rho!r}")
        _timer_add(timers, "stage.compute_rho_sec", perf_counter() - t_rho)

        zeta_raw = np.asarray(zeta_norm / rho, dtype=float)
        t_sweep = perf_counter()
        for i, omega_raw in enumerate(zeta_raw):
            _retune_level0_asm(
                ml=ml,
                subdomain=subdomain,
                subdomain_ptr=subdomain_ptr,
                inv_subblock=inv_subblock,
                inv_subblock_ptr=inv_subblock_ptr,
                omega_raw=float(omega_raw),
            )
            q_i, K_i = _estimate_qobs_homogeneous_two_grid(
                ml=ml,
                distribution=cfg.distribution,
                seed=None if cfg.seed is None else int(cfg.seed) + 10000 + i,
                n_samples=int(cfg.observed_n_samples),
                n_cycles=int(cfg.observed_n_cycles),
                timers=timers,
                timer_prefix=f"obs[{i}].",
            )
            if q_i is not None:
                q_obs[i] = float(q_i)
                rho_obs[i] = float(q_i)
            if K_i is not None:
                K_obs[i] = float(K_i)
        _timer_add(timers, "stage.observed_sweep_sec", perf_counter() - t_sweep)
    else:
        _retune_level0_fixed_smoother(
            ml=ml,
            smoother_type=cfg.smoother_type,
            mode=cfg.fixed_smoother_mode,
        )
        rho = float("nan")
        zeta_raw = np.full_like(zeta_norm, np.nan, dtype=float)

        t_obs = perf_counter()
        q_i, K_i = _estimate_qobs_homogeneous_two_grid(
            ml=ml,
            distribution=cfg.distribution,
            seed=cfg.seed,
            n_samples=int(cfg.observed_n_samples),
            n_cycles=int(cfg.observed_n_cycles),
            timers=timers,
            timer_prefix="obs_fixed.",
        )
        _timer_add(timers, "stage.observed_fixed_sec", perf_counter() - t_obs)
        qv = np.nan if q_i is None else float(q_i)
        kv = np.nan if K_i is None else float(K_i)
        rv = qv
        q_obs[:] = qv
        K_obs[:] = kv
        rho_obs[:] = rv

    _timer_add(timers, "stage.total_sec", perf_counter() - t_total)

    if print_constants:
        sd = int(constant_sig_digits)
        print("Observed damping sweep diagnostics:")
        print(f"  smoother_type           : {cfg.smoother_type}")
        if cfg.smoother_type in ("block_jacobi", "asm_overlap"):
            print(f"  rho(M^-1 A)             : {_fmt_sig(rho, sd)}")
            print("  zeta_norm definition    : rho(M^-1 A) * zeta_raw")
            print("  stable range            : zeta_norm in (0, 2)")
        else:
            print("  fixed smoother mode     : no damping parameter")
            print(f"  fixed sweep mode        : {cfg.fixed_smoother_mode}")
        print("Per-zeta observed table:")
        print("  zeta_norm zeta_raw q_obs    K_obs    rho_obs")
        for i in range(n):
            print(
                f"  {_fmt_sig(float(zeta_norm[i]), sd):<9} "
                f"{_fmt_sig(float(zeta_raw[i]), sd):<8} "
                f"{_fmt_sig(float(q_obs[i]), sd):<8} "
                f"{_fmt_sig(float(K_obs[i]), sd):<8} "
                f"{_fmt_sig(float(rho_obs[i]), sd):<8}"
            )

    if print_timers and timers is not None:
        print("Timing breakdown (seconds):")
        for key in sorted(timers, key=timers.get, reverse=True):
            val = timers[key]
            if key.endswith("n_iter") or key.endswith("n_cycle"):
                print(f"  {key}: {int(round(val))}")
            else:
                print(f"  {key}: {val:.6g}")

    return ObservedDampingSweepResult(
        smoother_type=cfg.smoother_type,
        rho_MinvA=float(rho),
        zeta_norm_values=np.asarray(zeta_norm, dtype=float),
        zeta_raw_values=np.asarray(zeta_raw, dtype=float),
        q_obs=np.asarray(q_obs, dtype=float),
        K_obs=np.asarray(K_obs, dtype=float),
        rho_obs=np.asarray(rho_obs, dtype=float),
        timings=timers,
    )


__all__ = [
    "ObservedDampingSweepConfig",
    "ObservedDampingSweepResult",
    "ObservedSmootherType",
    "FixedSmootherMode",
    "compute_observed_damping_sweep",
]
