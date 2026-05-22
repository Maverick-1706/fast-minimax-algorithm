"""Centralized benchmark configuration.

Single source of truth for seeds, precision, repeat counts, tolerances,
ε grids, bootstrap parameters, and outlier policy.  Every benchmark
module imports from here instead of hardcoding magic numbers.

Override at runtime via environment variables or by mutating the module
attributes before calling any benchmark function.
"""

from __future__ import annotations

import os

# ── Seed ───────────────────────────────────────────────────────────────

BENCHMARK_SEED: int | None = (
    int(os.environ["BENCHMARK_SEED"])
    if "BENCHMARK_SEED" in os.environ
    else 42
)
"""Global RNG seed.  Set env ``BENCHMARK_SEED`` to override."""

# ── Precision ──────────────────────────────────────────────────────────

ENABLE_X64: bool = False
"""Whether to enable JAX float64.  Default False (float32) for speed."""

# ── Repeat counts ──────────────────────────────────────────────────────

N_REPEATS_FULL: int = 5
"""Timed repeats for full (non-quick) benchmark runs."""

N_REPEATS_QUICK: int = 1
"""Timed repeats in --quick mode."""

N_REPEATS_SCALING: int = 3
"""Timed repeats for scaling sweeps (lighter than full)."""

N_WARMUP: int = 2
"""JIT warmup runs before timed iterations."""

# ── Tolerances / ε ─────────────────────────────────────────────────────

EPSILON_DEFAULT: float = 0.01
"""Default target duality gap for solver runs."""

EPSILON_GRID: list[float] = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
"""Default ε sweep for convergence analysis."""

TOLERANCE_LEVELS: list[float] = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
"""Tolerance levels used in solver convergence tests."""

# ── Bootstrap CI ───────────────────────────────────────────────────────

BOOTSTRAP_CI_LEVEL: float = 0.95
"""Confidence level for bootstrap CIs."""

BOOTSTRAP_N_RESAMPLES: int = 10_000
"""Number of bootstrap resamples."""

BCA_ENABLED: bool = True
"""Use bias-corrected and accelerated (BCa) bootstrap.  Falls back to
percentile method when the sample is too small for jackknife (< 3)."""

# ── Outlier policy ─────────────────────────────────────────────────────

OUTLIER_IQR_K: float = 1.5
"""IQR multiplier for Tukey fences.  1.5 = standard, 3.0 = extremes only."""

OUTLIER_REMOVE: bool = True
"""Whether to remove outliers before computing summary statistics."""

OUTLIER_MAX_FRACTION: float = 0.4
"""Maximum fraction of observations that can be flagged as outliers.
If more than this fraction are flagged, none are removed (the data is
legitimately spread, not polluted by outliers)."""

# ── Repeat-policy automation ───────────────────────────────────────────

AUTO_REPEAT: bool = True
"""If True, automatically re-run when outlier fraction exceeds threshold."""

AUTO_REPEAT_MAX_EXTRA: int = 3
"""Maximum number of additional repeat rounds if outliers persist."""

AUTO_REPEAT_N: int = 2
"""Number of extra repetitions per auto-repeat round."""

CV_THRESHOLD: float = 0.15
"""Coefficient-of-variation threshold.  If CV > this after outlier
removal, trigger an auto-repeat round."""
