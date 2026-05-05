"""Observed two-grid metrics for ``lsdd.two_lvl_theory``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyamg.relaxation import relaxation as _relaxation
from pyamg.relaxation.smoothing import rho_additive_schwarz_A as _rho_additive_schwarz_A

from ..smoothers import lsdd_make_smoother_spec
from ..types import LSDDLevel
from .setup import build_one_level_lsdd_context
from .types import TwoLevelSolverParams

TheorySmoother = Literal["block_jacobi", "asm_overlap", "ras", "msm"]


@dataclass(slots=True, frozen=True)
class ObservedTwoGridConfig:
    """Observed two-grid settings for one smoother."""

    smoother: TheorySmoother = "block_jacobi"
    zeta: float = 1.0
    with_rho: bool = False
    n_samples: int = 10
    n_cycles: int = 100


@dataclass(slots=True, frozen=True)
class ObservedTwoGridResult:
    """Observed two-grid outputs for one experiment point."""

    smoother: TheorySmoother
    zeta: float
    with_rho: bool
    rho_MinvA: float | None
    q_obs: float | None
    K_obs: float | None


def _draw_gaussian_vector(*, rng: np.random.Generator, n: int, dtype) -> np.ndarray:
    """Draw a Gaussian random vector of length ``n``."""
    rr = rng.standard_normal(int(n))
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        ii = rng.standard_normal(int(n))
        return (rr + 1j * ii).astype(dtype, copy=False)
    return rr.astype(dtype, copy=False)


def _estimate_qobs_homogeneous_two_grid(
    *,
    ml,
    n_samples: int,
    n_cycles: int,
) -> tuple[float | None, float | None]:
    """Estimate observed ``q_obs`` and ``K_obs`` from homogeneous one-cycle runs."""
    if n_samples <= 0 or n_cycles <= 0 or len(ml.levels) < 2:
        return None, None

    A = ml.levels[0].A
    n = int(A.shape[0])
    b0 = np.zeros(n, dtype=A.dtype)
    rng = np.random.default_rng(0)

    x_warm = np.ones(n, dtype=A.dtype)
    _ = ml.solve(b0, x0=x_warm, tol=0.0, maxiter=1, cycle="V", accel=None)

    zero_tol = 1.0e-14
    q_obs: float | None = None
    for _ in range(int(n_samples)):
        x = _draw_gaussian_vector(rng=rng, n=n, dtype=A.dtype)
        xAx = float(np.vdot(x, A @ x).real)
        if xAx <= 0.0:
            continue
        x /= np.sqrt(xAx)

        for _ in range(int(n_cycles)):
            xAx_old = float(np.vdot(x, A @ x).real)
            if xAx_old <= zero_tol:
                break
            n_old = float(np.sqrt(max(xAx_old, 0.0)))

            x_new = ml.solve(b0, x0=x, tol=0.0, maxiter=1, cycle="V", accel=None)
            xAx_new = float(np.vdot(x_new, A @ x_new).real)
            if xAx_new <= 0.0:
                break
            n_new = float(np.sqrt(max(xAx_new, 0.0)))
            q_k = float(n_new / max(n_old, 1.0e-300))
            q_obs = q_k if q_obs is None else max(q_obs, q_k)
            x = np.asarray(x_new).reshape(-1)

    if q_obs is None:
        return None, None
    if q_obs >= 1.0:
        return float(q_obs), float("inf")
    return float(q_obs), float(1.0 / (1.0 - q_obs))


def _compute_additive_schwarz_rho(
    *,
    level: LSDDLevel,
    domain: Literal["omega", "OMEGA"],
) -> float:
    """Compute ``rho(M^{-1}A)`` for additive Schwarz on one fixed subdomain layout."""
    Acsr = level.A.tocsr()
    Acsr.sort_indices()

    if domain == "omega":
        asm_spec = lsdd_make_smoother_spec(level=level, smoother=("asm", {"domain": "omega"}))
    else:
        asm_spec = lsdd_make_smoother_spec(level=level, smoother=("asm", {"domain": "OMEGA"}))
    if asm_spec is None:  # pragma: no cover
        raise ValueError("Failed to build ASM smoother specification")
    _, kwargs = asm_spec
    subdomain = np.asarray(kwargs["subdomain"], dtype=np.int32)
    subdomain_ptr = np.asarray(kwargs["subdomain_ptr"], dtype=np.int32)

    if hasattr(Acsr, "schwarz_parameters"):
        delattr(Acsr, "schwarz_parameters")

    inv_subblock = None
    inv_subblock_ptr = None
    subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr = _relaxation.schwarz_parameters(
        Acsr, subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr
    )
    rho = float(
        _rho_additive_schwarz_A(
            Acsr, subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr
        )
    )
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"Expected positive finite rho(M^-1 A), got {rho!r}")
    return rho


def _build_smoother_specs(
    *,
    smoother: TheorySmoother,
    zeta: float,
    with_rho: bool,
) -> tuple[object, object]:
    """Map smoother choices to LS-DD pre/post smoother specifications."""
    if smoother == "block_jacobi":
        spec = ("asm", {"domain": "omega", "omega": float(zeta), "withrho": bool(with_rho)})
        return spec, spec
    if smoother == "asm_overlap":
        spec = ("asm", {"domain": "OMEGA", "omega": float(zeta), "withrho": bool(with_rho)})
        return spec, spec
    if smoother == "ras":
        return "ras", "rasT"
    if smoother == "msm":
        return "msm", "msmT"
    raise ValueError(f"Unsupported smoother {smoother!r}")


def compute_observed_two_grid_constant(
    *,
    B,
    A,
    BT,
    solver_params: TwoLevelSolverParams,
    level_for_rho: LSDDLevel | None = None,
    cfg: ObservedTwoGridConfig = ObservedTwoGridConfig(),
) -> ObservedTwoGridResult:
    """Compute observed two-grid constant for one smoother and damping choice."""
    # Local import avoids circular initialization when pyamg imports
    # lsdd internals while least_squares_dd_exp is still being defined.
    from ...least_squares_dd_exp import least_squares_dd_solver_exp

    pre, post = _build_smoother_specs(
        smoother=cfg.smoother,
        zeta=float(cfg.zeta),
        with_rho=bool(cfg.with_rho),
    )
    kappa_value = 0.0 if solver_params.kappa is None else solver_params.kappa
    ml = least_squares_dd_solver_exp(
        B=B,
        BT=BT,
        A=A,
        presmoother=pre,
        postsmoother=post,
        symmetry=solver_params.symmetry,
        strength=solver_params.strength,
        aggregate=solver_params.aggregate,
        agg_levels=solver_params.agg_levels,
        kappa=kappa_value,
        nev=solver_params.nev,
        threshold=solver_params.threshold,
        mult_threshold=solver_params.mult_threshold,
        min_coarsening=solver_params.min_coarsening,
        max_levels=solver_params.max_levels,
        max_coarse=solver_params.max_coarse,
        max_density=solver_params.max_density,
        filteringA=solver_params.filteringA,
        filteringB=solver_params.filteringB,
        print_info=solver_params.print_info,
        force_row_closure=solver_params.force_row_closure,
        robust_Sker_handling=solver_params.robust_Sker_handling,
        coarse_solver=solver_params.coarse_solver,
    )
    q_obs, K_obs = _estimate_qobs_homogeneous_two_grid(
        ml=ml,
        n_samples=int(cfg.n_samples),
        n_cycles=int(cfg.n_cycles),
    )

    rho_MinvA: float | None = None
    if bool(cfg.with_rho) and cfg.smoother in {"block_jacobi", "asm_overlap"}:
        level = level_for_rho
        if level is None:
            level = build_one_level_lsdd_context(
                B=B,
                A=A,
                BT=BT,
                symmetry=solver_params.symmetry,
                strength=solver_params.strength,
                aggregate=solver_params.aggregate,
                agg_levels=solver_params.agg_levels,
                kappa=solver_params.kappa,
                nev=solver_params.nev,
                threshold=solver_params.threshold,
                mult_threshold=solver_params.mult_threshold,
                min_coarsening=solver_params.min_coarsening,
                filteringA=solver_params.filteringA,
                filteringB=solver_params.filteringB,
                print_info=solver_params.print_info,
                force_row_closure=solver_params.force_row_closure,
                robust_Sker_handling=solver_params.robust_Sker_handling,
                max_levels=solver_params.max_levels,
                max_coarse=solver_params.max_coarse,
                max_density=solver_params.max_density,
            )
        domain = "omega" if cfg.smoother == "block_jacobi" else "OMEGA"
        rho_MinvA = _compute_additive_schwarz_rho(level=level, domain=domain)

    return ObservedTwoGridResult(
        smoother=cfg.smoother,
        zeta=float(cfg.zeta),
        with_rho=bool(cfg.with_rho),
        rho_MinvA=rho_MinvA,
        q_obs=q_obs,
        K_obs=K_obs,
    )
