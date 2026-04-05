"""Data models and shared type aliases for LS-DD algebraic-theory diagnostics.

The names in this module follow the notation used in the algebraic-theory
paper (e.g. ``N_AJ``, ``N_YJ_perp``, ``W_Y``) so the code is directly mappable
to the written derivations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from scipy.sparse import csr_array

from ..types import FilteringSpec


ASolveMethod = Literal["direct", "cg"]
RandomDistribution = Literal["gaussian", "rademacher"]
TildeProjectionMode = Literal["precompute_W", "recompute"]
SpectralEstimator = Literal["power", "lobpcg"]
ZERO_NORM_TOL = 1.0e-14


@dataclass(slots=True, frozen=True)
class TwoLevelSolverParams:
    """Shared two-level LS-DD setup parameters for theory and observed runs."""

    symmetry: Literal["symmetric", "hermitian"] = "hermitian"
    strength: Any = None
    aggregate: Any = "standard"
    agg_levels: int = 2
    kappa: float | list[float] | None = None
    nev: int | None = None
    threshold: float | None = None
    mult_threshold: float | list[float | None] | None = None
    min_coarsening: int | list[int] | None = None
    filteringA: FilteringSpec | None = (False, 0.0)
    filteringB: FilteringSpec | None = (False, 0.0)
    print_info: bool = False
    force_row_closure: bool = True
    robust_Sker_handling: bool = True
    max_levels: int = 2
    max_coarse: int = 10
    max_density: float = 1.0
    coarse_solver: Any = "splu"


@dataclass(slots=True, frozen=True)
class ExactBlockJacobiWapResult:
    """Result container for exact block-Jacobi WAP computation."""

    W_M: float
    rayleigh_history: np.ndarray
    rel_change_history: np.ndarray
    a_norm_history: np.ndarray
    converged: bool
    n_iterations: int
    tau: float | None
    mu_max: float | None
    nev_per_agg: np.ndarray
    n_fine: int
    n_aggs: int
    n_coarse: int
    a_solve: ASolveMethod
    seed: int | None


@dataclass(slots=True, frozen=True)
class ExactSymmetricMetricWapResult:
    """Result container for exact ``W_Y`` with ``Y = \widetilde{J_\zeta}``."""

    W_Y: float
    contraction_factor: float
    rayleigh_history: np.ndarray
    rel_change_history: np.ndarray
    a_norm_history: np.ndarray
    converged: bool
    n_iterations: int
    tau: float | None
    mu_max: float | None
    nev_per_agg: np.ndarray
    n_fine: int
    n_aggs: int
    n_coarse: int
    a_solve: ASolveMethod
    h_solve: ASolveMethod
    projection_mode: TildeProjectionMode
    N_AJ: float
    zeta_input: float
    normalize_by_N_AJ: bool
    zeta_effective: float
    seed: int | None


@dataclass(slots=True, frozen=True)
class ChainDiagnosticsConfig:
    """Configuration for refined-chain diagnostics of the block-Jacobi baseline."""

    solver_params: TwoLevelSolverParams = field(default_factory=TwoLevelSolverParams)
    zeta_input: float = 1.0
    normalize_by_N_AJ: bool = True
    normalize_by_N_AJ_perp: bool = False

    a_solve: ASolveMethod = "cg"
    a_cg_rtol: float = 1e-10
    a_cg_atol: float = 0.0
    a_cg_maxiter: int | None = None

    h_solve: ASolveMethod = "cg"
    h_cg_rtol: float = 1e-10
    h_cg_atol: float = 0.0
    h_cg_maxiter: int | None = None

    projection_mode: TildeProjectionMode = "precompute_W"

    maxiter_W_M: int = 120
    tol_W_M: float = 1e-10
    miniter_W_M: int = 3

    maxiter_W_Y: int = 120
    tol_W_Y: float = 1e-10
    miniter_W_Y: int = 3

    maxiter_W_hat: int = 120
    tol_W_hat: float = 1e-10
    miniter_W_hat: int = 3

    N_AJ_estimator: SpectralEstimator = "power"
    N_AJ_perp_estimator: SpectralEstimator = "power"
    N_AJ_block_size: int = 3
    N_AJ_perp_block_size: int = 3
    maxiter_N_AJ: int = 120
    tol_N_AJ: float = 1e-10
    miniter_N_AJ: int = 3
    maxiter_N_AJ_perp: int = 120
    tol_N_AJ_perp: float = 1e-10
    miniter_N_AJ_perp: int = 3

    distribution: RandomDistribution = "gaussian"
    seed: int | None = 0

    estimate_q_obs_solve: bool = True
    qobs_n_samples: int = 10
    qobs_n_cycles: int = 100

    compute_exact_W_M: bool = True
    compute_exact_W_Y: bool = True
    compute_exact_W_hat: bool = True

    collect_timers: bool = False


@dataclass(slots=True, frozen=True)
class DampingSweepConfig:
    """Configuration for block-Jacobi damping sweep diagnostics.

    The sweep uses a *normalized* damping coordinate ``zeta`` in ``(0, 2)``,
    where ``zeta = 1`` corresponds to the whole-space model optimizer
    ``zeta_eff = 1 / N_AJ``. Internally, the effective damping applied to the
    smoother is

    ``zeta_eff = zeta / N_AJ``.

    The driver focuses only on damping-sensitive quantities: scalar sandwich
    models, sampled restricted target ``Phi_J(zeta)``, and observed two-grid
    ``q/K/rho`` from standalone solver runs.
    """

    solver_params: TwoLevelSolverParams = field(default_factory=TwoLevelSolverParams)

    zeta_values: np.ndarray | None = None
    zeta_min: float = 0.05
    zeta_max: float = 1.95
    n_zeta: int = 21

    N_AJ_estimator: SpectralEstimator = "power"
    N_AJ_perp_estimator: SpectralEstimator = "power"
    N_AJ_block_size: int = 1
    N_AJ_perp_block_size: int = 1
    maxiter_N_AJ: int = 120
    tol_N_AJ: float = 1e-10
    miniter_N_AJ: int = 3
    maxiter_N_AJ_perp: int = 120
    tol_N_AJ_perp: float = 1e-10
    miniter_N_AJ_perp: int = 3

    maxiter_Phi: int = 120
    tol_Phi: float = 1e-10
    miniter_Phi: int = 3

    compute_W_J: bool = True
    maxiter_W_J: int = 120
    tol_W_J: float = 1e-10
    miniter_W_J: int = 3
    a_solve: ASolveMethod = "cg"
    a_cg_rtol: float = 1e-10
    a_cg_atol: float = 0.0
    a_cg_maxiter: int | None = None

    h_solve: ASolveMethod = "cg"
    h_cg_rtol: float = 1e-10
    h_cg_atol: float = 0.0
    h_cg_maxiter: int | None = None

    distribution: RandomDistribution = "gaussian"
    seed: int | None = 0

    estimate_phi: bool = True
    estimate_observed: bool = True
    observed_n_samples: int = 10
    observed_n_cycles: int = 100

    collect_timers: bool = False


@dataclass(slots=True, frozen=True)
class DampingSweepResult:
    """Result container for block-Jacobi damping sweep diagnostics."""

    # Normalized sweep coordinate in (0, 2), with zeta=1 at 1/N_AJ effective damping.
    zeta_values: np.ndarray
    # Effective damping actually passed to the smoother, zeta_eff = zeta_values / N_AJ.
    zeta_effective_values: np.ndarray

    N_AJ: float
    N_AJ_perp: float
    tau: float | None
    mu_max: float | None
    W_J: float | None
    W_J_converged: bool | None
    W_J_n_iterations: int | None
    zeta_max_sampled: float
    zeta_low_raw: float | None
    zeta_low_clip: float | None
    # Reported in normalized zeta coordinates.
    zeta_low: float | None
    zeta_up: float | None
    zeta_exact: float | None
    zeta_best_K_obs: float | None

    L_J: np.ndarray
    U_J: np.ndarray
    Phi_J: np.ndarray
    q_obs: np.ndarray
    K_obs: np.ndarray
    rho_obs: np.ndarray

    Phi_converged: np.ndarray
    Phi_n_iterations: np.ndarray
    Phi_last_rel_change: np.ndarray
    Phi_last_abs_residual: np.ndarray
    Phi_last_rel_residual: np.ndarray

    N_AJ_history: np.ndarray
    N_AJ_rel_change_history: np.ndarray
    N_AJ_abs_residual_history: np.ndarray
    N_AJ_rel_residual_history: np.ndarray
    N_AJ_converged: bool
    N_AJ_n_iterations: int
    N_AJ_estimator_used: str

    N_AJ_perp_history: np.ndarray
    N_AJ_perp_rel_change_history: np.ndarray
    N_AJ_perp_abs_residual_history: np.ndarray
    N_AJ_perp_rel_residual_history: np.ndarray
    N_AJ_perp_converged: bool
    N_AJ_perp_n_iterations: int
    N_AJ_perp_estimator_used: str

    timings: dict[str, float] | None = None


@dataclass(slots=True, frozen=True)
class RefinedChainResult:
    """Primary diagnostics for the refined transfer chain."""

    tau: float | None
    mu_max: float | None

    N_AJ: float
    N_AJ_perp: float
    zeta_ref: float
    N_YJ: float
    N_YJ_perp: float

    W_M: float | None
    W_Y: float | None
    W_hat_Y: float | None

    T0: float | None
    T1: float | None
    T2: float | None
    T3: float | None
    T4: float | None
    T5: float | None

    R1: float | None
    R2: float | None
    R3: float | None
    R4: float | None
    R5: float | None

    q_obs_solve: float | None
    K_obs_solve: float | None

    N_AJ_history: np.ndarray
    N_AJ_rel_change_history: np.ndarray
    N_AJ_abs_residual_history: np.ndarray
    N_AJ_rel_residual_history: np.ndarray
    N_AJ_converged: bool
    N_AJ_n_iterations: int
    N_AJ_estimator_used: str

    N_AJ_perp_history: np.ndarray
    N_AJ_perp_rel_change_history: np.ndarray
    N_AJ_perp_abs_residual_history: np.ndarray
    N_AJ_perp_rel_residual_history: np.ndarray
    N_AJ_perp_converged: bool
    N_AJ_perp_n_iterations: int
    N_AJ_perp_estimator_used: str

    exact_W_M_result: ExactBlockJacobiWapResult | None
    exact_W_Y_result: ExactSymmetricMetricWapResult | None
    timings: dict[str, float] | None = None


@dataclass(slots=True)
class LocalProjectionBlock:
    """Local projection data for one aggregate block on ``omega_i``."""

    agg_id: int
    omega_rows: np.ndarray
    A_i: np.ndarray
    Z_i: np.ndarray
    AZ_i: np.ndarray
    use_cholesky: bool
    chol_factor: np.ndarray | None
    chol_lower: bool
    gram: np.ndarray | None


@dataclass(slots=True)
class DenseSPDSystem:
    """Factored dense SPD system helper with robust fallback."""

    use_cholesky: bool
    chol_factor: np.ndarray | None
    chol_lower: bool
    matrix: np.ndarray | None


@dataclass(slots=True)
class TildeMetricOps:
    """Reusable data for damped/symmetrized metric actions."""

    M_damped: csr_array
    H: csr_array
    solve_H: Callable[[np.ndarray], np.ndarray]
