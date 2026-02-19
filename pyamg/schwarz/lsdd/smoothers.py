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

import numpy as np

SmootherSpec = tuple[str, dict[str, Any]]


def lsdd_make_smoother_spec(*, level: Any, smoother: str | None) -> SmootherSpec | None:
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
    if smoother is None:
        return None

    blocks = level.blocks
    sub = level.sub

    if smoother == "msm":
        return (
            "schwarz",
            {
                "subdomain": blocks.subdomain,
                "subdomain_ptr": blocks.subdomain_ptr,
                "iterations": 1,
                "sweep": "symmetric",
            },
        )

    if smoother == "asm":
        return (
            "additive_schwarz",
            {
                "subdomain": blocks.subdomain,
                "subdomain_ptr": blocks.subdomain_ptr,
                "iterations": 1,
            },
        )

    if smoother in ("ras", "rasT"):
        pou_flat = sub.PoU_flat
        if pou_flat is None:
            pou_flat = np.concatenate(sub.PoU)
            sub.PoU_flat = pou_flat

        name = "rest_additive_schwarz" if smoother == "ras" else "rest_additive_schwarzT"
        return (
            name,
            {
                "subdomain": blocks.subdomain,
                "subdomain_ptr": blocks.subdomain_ptr,
                "POU": pou_flat,
                "iterations": 1,
            },
        )

    raise ValueError(f"Invalid smoother type: {smoother!r}")
