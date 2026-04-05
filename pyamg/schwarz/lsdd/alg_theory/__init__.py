"""Algebraic-theory package for LS-DD.

The package is split by responsibility:
- `exact_driver`: public high-level driver entry points.
- `damping_driver`: focused block-Jacobi damping sweep diagnostics.
- `models`: shared dataclasses and type aliases.
- `setup`: hierarchy construction and local-block extraction.
- `operators`: matrix-free/operator application helpers.
- `spectral`: iterative eigenvalue and power-iteration routines.
- `observed`: observed-cycle q/K diagnostics.
- `linalg` and `reporting`: small utility helpers.
"""

from __future__ import annotations

from . import damping_driver, exact_driver, linalg, models, observed, operators, reporting, setup, spectral

__all__ = [
    "damping_driver",
    "exact_driver",
    "models",
    "setup",
    "operators",
    "spectral",
    "observed",
    "linalg",
    "reporting",
]
