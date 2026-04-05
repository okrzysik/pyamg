"""Small reporting and scalar-utility helpers for algebraic-theory drivers."""

from __future__ import annotations

import numpy as np


def safe_ratio(num: float | None, den: float | None) -> float | None:
    """Return num/den when meaningful, else None."""
    if num is None or den is None:
        return None
    if not np.isfinite(den) or den == 0.0:
        return None
    return float(num / den)


def rel_change_history(vals: np.ndarray) -> np.ndarray:
    """Compute relative-change history with first entry set to inf."""
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    out = np.empty_like(arr)
    out[0] = np.inf
    for i in range(1, arr.size):
        prev = float(arr[i - 1])
        out[i] = float(abs(float(arr[i]) - prev) / max(abs(prev), 1.0))
    return out


def timer_add(timers: dict[str, float] | None, key: str, dt: float) -> None:
    """Accumulate elapsed seconds into ``timers[key]`` when timing is enabled."""
    if timers is None:
        return
    timers[key] = float(timers.get(key, 0.0) + float(dt))


def fmt_sig(x: float | None, sig_digits: int = 2) -> str:
    """Format scalar with fixed significant digits, preserving None/inf/nan."""
    if x is None:
        return "None"
    xv = float(x)
    if np.isnan(xv):
        return "nan"
    if np.isposinf(xv):
        return "inf"
    if np.isneginf(xv):
        return "-inf"
    return f"{xv:.{int(sig_digits)}g}"


def transfer_whole_from_N_AJ_estimate(
    *,
    N_AJ: float,
    zeta_input: float,
    with_rho: bool,
    with_rho_perp: bool,
    N_AJ_perp_scale: float | None = None,
) -> float:
    """Compute ``N_{Y,J}`` from an ``N_{A,J}`` estimate and damping semantics."""
    b = float(N_AJ)
    if with_rho and with_rho_perp:
        raise ValueError("with_rho and with_rho_perp cannot both be True")
    if with_rho:
        zeta_eff = float(zeta_input) / b
    elif with_rho_perp:
        if N_AJ_perp_scale is None:
            raise ValueError("N_AJ_perp_scale is required when with_rho_perp=True")
        zeta_eff = float(zeta_input) / float(N_AJ_perp_scale)
    else:
        zeta_eff = float(zeta_input)
    den = zeta_eff * (2.0 - zeta_eff * b)
    if den <= 0.0:
        return float("inf")
    return float(1.0 / den)


def transfer_restricted_from_N_AJ_perp_estimate(*, N_AJ_perp: float, zeta_eff: float) -> float:
    """Compute ``N_{Y,J}^perp`` from ``N_{A,J}^perp`` and fixed ``zeta_eff``."""
    den = float(zeta_eff) * (2.0 - float(zeta_eff) * float(N_AJ_perp))
    if den <= 0.0:
        return float("inf")
    return float(1.0 / den)


def resolve_effective_damping(
    *,
    zeta: float,
    with_rho: bool,
    with_rho_perp: bool,
    N_AJ: float,
    N_AJ_perp: float,
) -> float:
    """Resolve effective damping with optional normalization by ``N_AJ`` or ``N_AJ_perp``."""
    if with_rho and with_rho_perp:
        raise ValueError("with_rho and with_rho_perp cannot both be True")

    z = float(zeta)
    if with_rho:
        z /= float(N_AJ)
    elif with_rho_perp:
        z /= float(N_AJ_perp)

    if not np.isfinite(z) or z <= 0.0:
        raise ValueError(f"Effective zeta must be positive and finite, got {z!r}")
    stab = z * float(N_AJ)
    if stab >= 2.0:
        raise ValueError(
            "Effective zeta violates admissibility for symmetrized metric: "
            f"zeta_eff * N_AJ = {stab:.6g} >= 2"
        )
    return z
