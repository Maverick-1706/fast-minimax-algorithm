"""Precision configuration for Minimax-AIPE.

All numerical guard constants are defined here so they can be tuned in one
place when switching between FP32 (GPU) and FP64 (CPU) computation.

JAX defaults to FP32 on all hardware.  FP64 is disabled here explicitly so
that any accidental double-precision literals are caught and cast down rather
than silently promoting arrays.  Set the environment variable
    JAX_ENABLE_X64=1
before importing this package if you need FP64 for research / debugging.

FP32 machine epsilon  ≈ 1.2e-7
FP32 min normal       ≈ 1.2e-38
"""

from __future__ import annotations

import jax

# ── Disable FP64 globally (GPU-safe) ────────────────────────────────────────
# This must happen before any JAX array is created.  It is a no-op when
# JAX_ENABLE_X64=1 is set in the environment.
jax.config.update("jax_enable_x64", False)

# ── Guard constants scaled for FP32 ─────────────────────────────────────────

# Floor for denominators / norms to prevent division by zero.
# In FP32 we need at least 1e-7 (≈ machine epsilon); use 1e-6 for headroom.
ABS_TOL: float = 1e-6

# Threshold below which a cubic Hessian contribution is treated as zero.
# In FP64 this was 1e-15; in FP32 anything below ~1e-7 is indistinguishable
# from zero, so we use 1e-7.
CUBIC_ZERO: float = 1e-7

# Smallest "tiny" constant added to regularisation to keep SPD systems
# well-conditioned.  Must be larger than FP32 machine epsilon.
TINY: float = 1e-6

# Floor applied inside projection operations to avoid 0/0.
PROJ_EPS: float = 1e-6

# Minimum gap value considered non-trivial (below this we call it zero).
GAP_FLOOR: float = 1e-5

# Minimum regularisation strength (prevents the secular-equation from
# collapsing with zero regularisation).
REG_MIN: float = 1e-5

# Default absolute tolerance for test assertions under FP32.
# Tests that used 1e-12 / 1e-14 in FP64 should use this instead.
TEST_ATOL: float = 1e-4

__all__ = [
    "ABS_TOL",
    "CUBIC_ZERO",
    "TINY",
    "PROJ_EPS",
    "GAP_FLOOR",
    "REG_MIN",
    "TEST_ATOL",
]
