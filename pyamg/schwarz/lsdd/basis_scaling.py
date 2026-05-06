"""Optional aggregate-wise basis scaling for LS–AMG–DD prolongation triplets.

This module applies a local right basis transformation on already-selected
aggregate-local prolongation columns (stored as triplets) before global P
assembly. The transformation is optional and preserves local coarse-space span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr, solve

from .types import LSDDLevel


@dataclass(slots=True)
class BasisScalingSummary:
    """Aggregate-wise diagnostics for optional LS-DD basis scaling."""

    method: str
    n_total: int = 0
    n_scaled: int = 0
    n_skipped_empty: int = 0
    n_skipped_too_few_rows: int = 0
    n_skipped_bad_conditioning: int = 0
    n_skipped_exception: int = 0
    cond_values: list[float] | None = None

    def __post_init__(self) -> None:
        """Allocate mutable diagnostic lists after dataclass construction."""
        if self.cond_values is None:
            self.cond_values = []


def _lsdd_basis_scaling_row_widths(*, level: LSDDLevel, n_rows: int) -> np.ndarray:
    """Estimate each B-row's tentative scalar coarse width.

    Width is the sum, over aggregates touching that row, of selected local basis
    size k_i on that aggregate.
    """
    width = np.zeros(int(n_rows), dtype=float)
    nev = np.asarray(level.eigs.nev, dtype=np.int32)
    for i, k_i in enumerate(nev):
        k = int(k_i)
        if k <= 0:
            continue
        rows_i = level.sub.R_rows[i]
        if rows_i is None or len(rows_i) == 0:
            continue
        width[np.asarray(rows_i, dtype=np.int32)] += float(k)
    return width


def _lsdd_select_basis_scaling_rows(
    *,
    Y: np.ndarray,
    weights: np.ndarray | None,
    cond_max: float,
    weight_power: float,
) -> tuple[np.ndarray | None, float | None]:
    """Select a stable set of row functionals for local basis nodalization."""
    k = int(Y.shape[1])
    if k == 0 or int(Y.shape[0]) < k:
        return None, None

    def _pivot_and_check(Y_select: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        try:
            _, _, piv = qr(Y_select.T, pivoting=True, mode="economic", check_finite=False)
            selected = np.asarray(piv[:k], dtype=np.int32)
        except Exception:
            return None, None

        C = Y[selected, :]
        cond_C = float(np.linalg.cond(C))
        if not np.isfinite(cond_C) or cond_C > float(cond_max):
            return None, cond_C
        return selected, cond_C

    use_weighted = weights is not None and float(weight_power) > 0.0
    if use_weighted:
        w = np.maximum(np.asarray(weights, dtype=float), 1.0)
        Y_weighted = (w ** float(weight_power))[:, None] * Y
        selected, cond_C = _pivot_and_check(Y_weighted)
        if selected is not None:
            return selected, cond_C
        # Optional fallback: retry once without weights.
        selected_u, cond_C_u = _pivot_and_check(Y)
        return selected_u, cond_C_u if cond_C_u is not None else cond_C

    return _pivot_and_check(Y)


def _lsdd_scale_one_triplet_block(
    *,
    B,
    method: str,
    omega_rows: np.ndarray,
    Z: np.ndarray,
    R_rows_i: np.ndarray | None,
    row_width: np.ndarray,
    cond_max: float,
    weight_power: float,
    normalize_columns: bool,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Scale one aggregate-local basis block reconstructed from prolongation triplets."""
    k = int(Z.shape[1])
    if k <= 0:
        return None, {"reason": "empty_rows"}

    if method == "b_nodal":
        if R_rows_i is None:
            return None, {"reason": "empty_rows"}
        rows_i = np.asarray(R_rows_i, dtype=np.int32)
        if rows_i.size < k:
            return None, {"reason": "too_few_rows"}

        B_i = B[rows_i, :][:, omega_rows]
        Y = np.asarray(B_i @ Z)
        weights = row_width[rows_i]
        selected, cond_C = _lsdd_select_basis_scaling_rows(
            Y=Y,
            weights=weights,
            cond_max=cond_max,
            weight_power=weight_power,
        )
        if selected is None:
            return None, {"reason": "bad_conditioning", "cond": cond_C}
        C = Y[selected, :]

    elif method == "p_nodal":
        Y = Z
        selected, cond_C = _lsdd_select_basis_scaling_rows(
            Y=Y,
            weights=None,
            cond_max=cond_max,
            weight_power=0.0,
        )
        if selected is None:
            return None, {"reason": "bad_conditioning", "cond": cond_C}
        C = Y[selected, :]

    else:
        raise ValueError(f"Unsupported basis scaling method: {method!r}")

    Z_scaled = solve(C.T, Z.T, assume_a="gen", check_finite=False).T
    if normalize_columns:
        norms = np.linalg.norm(Z_scaled, axis=0)
        mask = norms > 0.0
        Z_scaled[:, mask] /= norms[mask]

    return Z_scaled, {"reason": "scaled", "cond": cond_C, "selected": selected}


def _lsdd_scale_basis_triplets_for_fill(
    *,
    level: LSDDLevel,
    B,
    p_r: list,
    p_c: list,
    p_v: list,
    method: str,
    cond_max: float,
    weight_power: float,
    normalize_columns: bool,
    drop_tol: float,
    stats,
) -> None:
    """Apply optional aggregate-wise right scaling to already-selected P triplets."""
    if method == "none":
        return
    if method not in ("b_nodal", "p_nodal"):
        raise ValueError(f"Unsupported basis scaling method: {method!r}")

    summary = BasisScalingSummary(method=method)
    dropped_entries_total = 0
    nev = np.asarray(level.eigs.nev, dtype=np.int32)
    row_width = _lsdd_basis_scaling_row_widths(level=level, n_rows=B.shape[0])

    base = 0
    for i, k_i in enumerate(nev):
        k = int(k_i)
        summary.n_total += 1
        if k <= 0:
            summary.n_skipped_empty += 1
            continue

        if base + k > len(p_r) or base + k > len(p_c) or base + k > len(p_v):
            raise ValueError("Malformed triplet arrays: aggregate slice exceeds triplet list length")

        omega_rows = np.asarray(p_r[base], dtype=np.int32)
        if omega_rows.ndim != 1:
            raise ValueError("Malformed triplet rows: expected 1D row index arrays")

        block_cols: list[np.ndarray] = []
        for triplet_idx in range(base, base + k):
            rows_j = np.asarray(p_r[triplet_idx], dtype=np.int32)
            if rows_j.shape != omega_rows.shape or not np.array_equal(rows_j, omega_rows):
                raise ValueError("Malformed triplets: aggregate-local columns do not share identical row support")

            col_j = np.asarray(p_c[triplet_idx], dtype=np.int32)
            if col_j.shape != omega_rows.shape:
                raise ValueError("Malformed triplets: p_c entry shape mismatch with p_r")
            if col_j.size > 0 and np.unique(col_j).size != 1:
                raise ValueError("Malformed triplets: each p_c entry must contain one coarse column id")

            vals_j = np.asarray(p_v[triplet_idx])
            if vals_j.ndim != 1 or vals_j.shape != omega_rows.shape:
                raise ValueError("Malformed triplets: p_v entry shape mismatch with p_r")
            block_cols.append(vals_j)

        Z = np.column_stack(block_cols)

        try:
            Z_scaled, info = _lsdd_scale_one_triplet_block(
                B=B,
                method=method,
                omega_rows=omega_rows,
                Z=Z,
                R_rows_i=level.sub.R_rows[i],
                row_width=row_width,
                cond_max=cond_max,
                weight_power=weight_power,
                normalize_columns=normalize_columns,
            )
        except Exception:
            Z_scaled = None
            info = {"reason": "exception"}

        if Z_scaled is None:
            reason = str(info.get("reason", "exception"))
            if reason == "too_few_rows":
                summary.n_skipped_too_few_rows += 1
            elif reason == "bad_conditioning":
                summary.n_skipped_bad_conditioning += 1
            elif reason == "empty_rows":
                summary.n_skipped_empty += 1
            else:
                summary.n_skipped_exception += 1
        else:
            if Z_scaled.shape != Z.shape:
                raise ValueError("Internal error: scaled block has unexpected shape")
            if drop_tol > 0.0 and Z_scaled.size > 0:
                max_abs = float(np.max(np.abs(Z_scaled)))
                if max_abs > 0.0:
                    thresh = float(drop_tol) * max_abs
                    mask_drop = np.abs(Z_scaled) < thresh
                    dropped_entries_total += int(np.count_nonzero(mask_drop))
                    if np.any(mask_drop):
                        Z_scaled = Z_scaled.copy()
                        Z_scaled[mask_drop] = 0.0
            for local_col, triplet_idx in enumerate(range(base, base + k)):
                p_v[triplet_idx] = np.asarray(Z_scaled[:, local_col])
            summary.n_scaled += 1
            cond = info.get("cond")
            if cond is not None:
                summary.cond_values.append(float(cond))

        base += k

    if base != len(p_v):
        raise ValueError(f"Basis scaling consumed {base} triplets but p_v has {len(p_v)} entries")

    stats.extra["basis_scaling_method"] = method
    stats.extra["basis_scaling_total"] = summary.n_total
    stats.extra["basis_scaling_scaled"] = summary.n_scaled
    stats.extra["basis_scaling_skipped_empty"] = summary.n_skipped_empty
    stats.extra["basis_scaling_skipped_too_few_rows"] = summary.n_skipped_too_few_rows
    stats.extra["basis_scaling_skipped_bad_conditioning"] = summary.n_skipped_bad_conditioning
    stats.extra["basis_scaling_skipped_exception"] = summary.n_skipped_exception
    stats.extra["basis_scaling_drop_tol"] = float(drop_tol)
    stats.extra["basis_scaling_entries_dropped"] = int(dropped_entries_total)

    if summary.cond_values:
        conds = np.asarray(summary.cond_values, dtype=float)
        stats.extra["basis_scaling_cond_min"] = float(np.min(conds))
        stats.extra["basis_scaling_cond_med"] = float(np.median(conds))
        stats.extra["basis_scaling_cond_max"] = float(np.max(conds))
