"""Smoother specification helpers for LS–AMG–DD.

This module maps short-hand smoother names to `(name, kwargs)` specifications
consumed by `pyamg.relaxation.smoothing.change_smoothers`.

Supported smoothers
-------------------
- "msm"  : multiplicative Schwarz ("schwarz", symmetric sweep)
- "asm"  : additive Schwarz ("additive_schwarz")
- "ras"  : restricted additive Schwarz ("rest_additive_schwarz")
- "rasT" : transpose variant ("rest_additive_schwarzT")
- None   : disable smoothing on this level

Required level fields
---------------------
All smoothers here require:
  - `level.blocks.subdomain` and `level.blocks.subdomain_ptr` (flattened OMEGA_i and pointers)

The RAS variants additionally require a flattened PoU vector `POU` aligned with
`level.blocks.subdomain`. This is constructed by concatenating `level.sub.PoU` and cached
as `level.sub.PoU_flat` to avoid repeated concatenations.
"""

from __future__ import annotations

from typing import Any

from .types import LSDDLevel
import numpy as np

SmootherSpec = tuple[str, dict[str, Any]]
SmootherChoice = str | tuple[str, dict[str, Any]] | None


def _lsdd_flatten_subdomains(domains: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-aggregate index arrays into (subdomain, subdomain_ptr)."""
    ptr = np.zeros(len(domains) + 1, dtype=np.int32)
    chunks: list[np.ndarray] = []

    for i, dom in enumerate(domains):
        if dom is None:
            raise ValueError("Expected subdomain index sets to be populated before smoother setup")
        arr = np.asarray(dom, dtype=np.int32).ravel()
        # Schwarz kernels expect sorted subdomain indices per block.
        arr = np.sort(arr)
        chunks.append(arr)
        ptr[i + 1] = ptr[i] + int(arr.size)

    flat = np.concatenate(chunks).astype(np.int32, copy=False) if chunks else np.zeros(0, dtype=np.int32)
    return flat, ptr


def _lsdd_unpack_smoother_arg(v: SmootherChoice) -> tuple[str | None, dict[str, Any]]:
    """Normalize a smoother option into ``(name, kwargs)`` form."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(f"Expected smoother tuple (name, kwargs), got: {v!r}")
        name, kwargs = v
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError(f"Expected smoother kwargs to be a dict, got: {type(kwargs).__name__}")
        return name, dict(kwargs)
    return v, {}


def lsdd_make_smoother_spec(*, level: LSDDLevel, smoother: SmootherChoice) -> SmootherSpec | None:
    """Return a PyAMG smoother specification for one multigrid level.

    Parameters
    ----------
    level
        Multigrid level object. Required attributes:
          - `level.blocks.subdomain`, `level.blocks.subdomain_ptr`
        Additionally for RAS/RAS^T:
          - `level.sub.PoU` (list of per-subdomain 0/1 masks), or cached `level.sub.PoU_flat`.

    smoother
        One of {"msm", "asm", "ras", "rasT"} or None.
        You may also pass ``(name, kwargs)`` to provide method-specific options
        (for example ``("asm", {"omega": 0.8, "withrho": True, "domain": "omega"})``).

    Returns
    -------
    spec
        Either:
          - None (disable smoothing), or
          - `(name, kwargs)` where `name` is a PyAMG smoother identifier and
            `kwargs` contains Schwarz subdomain data and (for RAS variants) PoU weights.

    Raises
    ------
    ValueError
        If an unsupported smoother name is provided.
    """
    smoother, extra = _lsdd_unpack_smoother_arg(smoother)
    if smoother is None:
        return None

    blocks = level.blocks
    sub = level.sub

    if smoother == "msm":
        kwargs = {
            "subdomain": blocks.subdomain,
            "subdomain_ptr": blocks.subdomain_ptr,
            "iterations": 1,
            "sweep": "forward",
        }
        kwargs.update(extra)
        return (
            "schwarz",
            kwargs,
        )

    if smoother == "msmT":
        kwargs = {
            "subdomain": blocks.subdomain,
            "subdomain_ptr": blocks.subdomain_ptr,
            "iterations": 1,
            "sweep": "backward",
        }
        kwargs.update(extra)
        return (
            "schwarz",
            kwargs,
        )

    if smoother == "asm":
        domain = extra.pop("domain", "OMEGA")
        if not isinstance(domain, str):
            raise ValueError(f"Expected asm domain to be a string, got: {type(domain).__name__}")

        if domain in ("OMEGA", "Omega", "overlap", "overlapping"):
            subdomain = blocks.subdomain
            subdomain_ptr = blocks.subdomain_ptr
        elif domain in ("omega", "nonoverlap", "non-overlap", "nonoverlapping"):
            subdomain, subdomain_ptr = _lsdd_flatten_subdomains(sub.omega)
        else:
            raise ValueError(
                f"Invalid asm domain {domain!r}. Expected one of "
                "'OMEGA'/'overlap' or 'omega'/'nonoverlap'."
            )

        kwargs = {
            "subdomain": subdomain,
            "subdomain_ptr": subdomain_ptr,
        }
        kwargs.update(extra)
        return (
            "additive_schwarz",
            kwargs,
        )

    if smoother in ("ras", "rasT"):
        pou_flat = sub.PoU_flat
        if pou_flat is None:
            pou_flat = np.concatenate(sub.PoU)
            sub.PoU_flat = pou_flat

        name = "rest_additive_schwarz" if smoother == "ras" else "rest_additive_schwarzT"
        kwargs = {
            "subdomain": blocks.subdomain,
            "subdomain_ptr": blocks.subdomain_ptr,
            "POU": pou_flat,
            "iterations": 1,
        }
        kwargs.update(extra)
        return (
            name,
            kwargs,
        )

    raise ValueError(f"Invalid smoother type: {smoother!r}")
