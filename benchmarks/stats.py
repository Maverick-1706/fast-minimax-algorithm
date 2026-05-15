"""Statistical utilities for benchmarks.

Bootstrap confidence intervals, IQR-based outlier detection, and summary
statistics.  Most quantities are computed on plain Python lists,
but bootstrap CI uses JAX for robust quantile estimation.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass
class Summary:
    """Summary statistics for a sample."""

    n: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    ci: tuple[float, float]
    outliers: list[float] = field(default_factory=list)
    n_outliers: int = 0


def bootstrap_ci(
    data: list[float],
    ci: float = 0.95,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : list[float]
        Observed sample.
    ci : float
        Confidence level (default 0.95).
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Lower and upper bounds of the CI.
    """
    n = len(data)
    if n < 2:
        m = data[0] if data else 0.0
        return (m, m)

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))

    alpha = 1.0 - ci
    means_arr = jnp.asarray(means)
    lo = float(jnp.quantile(means_arr, alpha / 2.0))
    hi = float(jnp.quantile(means_arr, 1.0 - alpha / 2.0))
    return (lo, hi)


def detect_outliers_iqr(data: list[float], k: float = 1.5) -> list[float]:
    """Flag outliers using the IQR method.

    A value is an outlier if it lies below Q1 − k·IQR or above Q3 + k·IQR.

    Parameters
    ----------
    data : list[float]
        Observed sample.
    k : float
        IQR multiplier (default 1.5 — standard Tukey fences).

    Returns
    -------
    list[float]
        The outlier values (preserving original order).
    """
    if len(data) < 4:
        return []

    sorted_d = sorted(data)
    qs = statistics.quantiles(sorted_d, n=4, method="inclusive")
    q1, q3 = qs[0], qs[2]
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return [x for x in data if x < lo or x > hi]


def summarise(data: list[float], ci: float = 0.95, n_boot: int = 10_000) -> Summary:
    """Compute summary statistics with bootstrap CI and outlier detection.

    Parameters
    ----------
    data : list[float]
        Observed sample.
    ci : float
        Confidence level for bootstrap CI.
    n_boot : int
        Number of bootstrap resamples.

    Returns
    -------
    Summary
    """
    if not data:
        return Summary(
            n=0, mean=0.0, std=0.0, median=0.0,
            min=0.0, max=0.0, ci=(0.0, 0.0),
        )

    outliers = detect_outliers_iqr(data)
    ci_lo, ci_hi = bootstrap_ci(data, ci=ci, n_boot=n_boot)

    return Summary(
        n=len(data),
        mean=statistics.mean(data),
        std=statistics.stdev(data) if len(data) > 1 else 0.0,
        median=statistics.median(data),
        min=min(data),
        max=max(data),
        ci=(ci_lo, ci_hi),
        outliers=outliers,
        n_outliers=len(outliers),
    )


def format_ci(lo: float, hi: float, precision: int = 4) -> str:
    """Format a CI as '[lo, hi]'."""
    return f"[{lo:.{precision}f},{hi:.{precision}f}]"


def format_summary(s: Summary, precision: int = 4) -> str:
    """Format a Summary as 'mean ± std [ci_lo, ci_hi] (n=X, outliers=Y)'."""
    p = precision
    return (
        f"{s.mean:.{p}f} ± {s.std:.{p}f} "
        f"{format_ci(*s.ci, precision=p)} "
        f"(n={s.n}, outliers={s.n_outliers})"
    )
