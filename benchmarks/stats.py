"""Statistical utilities for benchmarks.

BCa bootstrap confidence intervals, IQR-based outlier detection/removal,
and automated repeat-policy decisions.

Key functions
-------------
- ``bootstrap_ci`` — percentile or BCa bootstrap CI for the mean
- ``detect_outliers_iqr`` — flag outliers via Tukey fences
- ``remove_outliers`` — filter outliers (respecting max-fraction guard)
- ``summarise`` — full summary with outlier removal and BCa CI
- ``should_repeat`` — automated repeat-policy decision
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
import numpy as np

import jax.numpy as jnp

from benchmarks import config


# ── Data classes ────────────────────────────────────────────────────────


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
    n_original: int = 0
    cv: float = 0.0


@dataclass
class RepeatDecision:
    """Result of :func:`should_repeat`."""

    should_repeat: bool
    reason: str
    outlier_fraction: float
    cv: float


# ── Outlier detection ──────────────────────────────────────────────────


def detect_outliers_iqr(
    data: list[float],
    k: float | None = None,
) -> list[float]:
    """Flag outliers using the IQR method.

    A value is an outlier if it lies below Q1 − k·IQR or above Q3 + k·IQR.

    Parameters
    ----------
    data : list[float]
        Observed sample.
    k : float or None
        IQR multiplier.  Defaults to ``config.OUTLIER_IQR_K``.

    Returns
    -------
    list[float]
        The outlier values (preserving original order).
    """
    if k is None:
        k = config.OUTLIER_IQR_K
    if len(data) < 4:
        return []

    sorted_d = sorted(data)
    qs = statistics.quantiles(sorted_d, n=4, method="inclusive")
    q1, q3 = qs[0], qs[2]
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return [x for x in data if x < lo or x > hi]


def remove_outliers(
    data: list[float],
    k: float | None = None,
) -> tuple[list[float], list[float]]:
    """Remove IQR outliers, respecting the max-fraction guard.

    Parameters
    ----------
    data : list[float]
        Observed sample.
    k : float or None
        IQR multiplier.  Defaults to ``config.OUTLIER_IQR_K``.

    Returns
    -------
    clean : list[float]
        Data with outliers removed (or unchanged if guard triggered).
    outliers : list[float]
        The flagged outlier values.
    """
    outliers = detect_outliers_iqr(data, k=k)
    if not outliers:
        return list(data), []

    frac = len(outliers) / len(data)
    if frac > config.OUTLIER_MAX_FRACTION:
        return list(data), []

    # O(N) removal via a multiset — handles duplicate values correctly
    # without fragile floating-point equality scans.
    from collections import Counter
    outlier_counts = Counter(outliers)
    clean = []
    for x in data:
        if outlier_counts.get(x, 0) > 0:
            outlier_counts[x] -= 1
        else:
            clean.append(x)
    return clean, outliers


# ── Bootstrap CI ───────────────────────────────────────────────────────


def _jackknife_mean(data: list[float]) -> list[float]:
    """Return jackknife leave-one-out means."""
    n = len(data)
    total = sum(data)
    return [(total - x) / (n - 1) for x in data]


def _bca_correction(
    data: list[float],
    theta_hat: float,
    jackknife_values: list[float],
) -> float:
    """Compute BCa acceleration factor 'a' from the jackknife values.

    Returns
    -------
    a : float
        The acceleration factor: (1/6) * (Σ (θ̄ − θ₍₋ᵢ₎)³) / (Σ (θ̄ − θ₍₋ᵢ₎)²)^(3/2)
    """
    n = len(jackknife_values)
    theta_bar = sum(jackknife_values) / n

    # Acceleration factor
    num = sum((theta_bar - ji) ** 3 for ji in jackknife_values)
    den = sum((theta_bar - ji) ** 2 for ji in jackknife_values) ** 1.5
    a = num / (6.0 * den) if den > 1e-15 else 0.0

    return a

def bootstrap_ci(
    data: list[float],
    ci: float | None = None,
    n_boot: int | None = None,
    seed: int | None = None,
    method: str | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : list[float]
        Observed sample.
    ci : float or None
        Confidence level.  Defaults to ``config.BOOTSTRAP_CI_LEVEL``.
    n_boot : int or None
        Number of bootstrap resamples.  Defaults to ``config.BOOTSTRAP_N_RESAMPLES``.
    seed : int
        RNG seed for reproducibility.
    method : str or None
        ``"bca"`` for bias-corrected and accelerated, ``"percentile"``
        for the basic method.  Defaults to ``"bca"`` if
        ``config.BCA_ENABLED`` is True and sample is large enough.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Lower and upper bounds of the CI.
    """
    if ci is None:
        ci = config.BOOTSTRAP_CI_LEVEL
    if n_boot is None:
        n_boot = config.BOOTSTRAP_N_RESAMPLES

    n = len(data)
    if n < 2:
        m = data[0] if data else 0.0
        return (m, m)

    if method is None:
        method = "bca" if config.BCA_ENABLED and n >= 3 else "percentile"

    rng = np.random.default_rng(seed)

    # ── Bootstrap resampling ──────────────────────────────────────────
    data_arr = np.asarray(data)
    indices = rng.choice(n, size=(n_boot, n))
    means = data_arr[indices].mean(axis=1).tolist()

    theta_hat = statistics.mean(data)

    if method == "bca" and n >= 3:
        return _bca_interval(means, theta_hat, data, ci)
    else:
        return _percentile_interval(means, ci)


def _percentile_interval(
    means: list[float],
    ci: float,
) -> tuple[float, float]:
    """Basic percentile bootstrap CI."""
    alpha = 1.0 - ci
    means_arr = jnp.asarray(means)
    lo = float(jnp.quantile(means_arr, alpha / 2.0))
    hi = float(jnp.quantile(means_arr, 1.0 - alpha / 2.0))
    return (lo, hi)


def _bca_interval(
    means: list[float],
    theta_hat: float,
    data: list[float],
    ci: float,
) -> tuple[float, float]:
    """BCa bootstrap CI.

    Adjusts the percentile endpoints by bias-correction z0 and
    acceleration factor a computed from the jackknife.
    """
    from scipy.stats import norm as _norm

    means_arr = jnp.asarray(means)

    # ── Bias correction z0 ────────────────────────────────────────────
    frac_le = float(jnp.mean(means_arr <= theta_hat))
    # Clamp to avoid Φ⁻¹(0) = -∞ or Φ⁻¹(1) = +∞
    frac_le = float(jnp.clip(frac_le, 1.0 / len(means), 1.0 - 1.0 / len(means)))
    z0 = float(_norm.ppf(frac_le))

    # ── Acceleration factor a ─────────────────────────────────────────
    jackknife_values = _jackknife_mean(data)
    a = _bca_correction(data, theta_hat, jackknife_values)

    # ── Adjusted percentiles ──────────────────────────────────────────
    alpha = 1.0 - ci
    z_lo = float(_norm.ppf(alpha / 2.0))
    z_hi = float(_norm.ppf(1.0 - alpha / 2.0))

    denom_lo = 1.0 - a * (z_lo + z0)
    denom_hi = 1.0 - a * (z_hi + z0)

    # If denominator <= 1e-6, the BCa mapping breaks down or becomes unstable.
    # Fall back safely to the unadjusted percentile interval bounds.
    adj_lo = z0 + (z_lo + z0) / denom_lo if jnp.abs(denom_lo) > 1e-6 else z_lo
    adj_hi = z0 + (z_hi + z0) / denom_hi if jnp.abs(denom_hi) > 1e-6 else z_hi

    p_lo = float(_norm.cdf(adj_lo))
    p_hi = float(_norm.cdf(adj_hi))

    # Clamp to valid quantile range
    p_lo = max(1.0 / len(means), min(p_lo, 1.0 - 1.0 / len(means)))
    p_hi = max(1.0 / len(means), min(p_hi, 1.0 - 1.0 / len(means)))

    lo = float(jnp.quantile(means_arr, p_lo))
    hi = float(jnp.quantile(means_arr, p_hi))
    return (lo, hi)

# ── Summary ────────────────────────────────────────────────────────────


def summarise(
    data: list[float],
    ci: float | None = None,
    n_boot: int | None = None,
    remove: bool | None = None,
) -> Summary:
    """Compute summary statistics with outlier removal and BCa CI.

    Parameters
    ----------
    data : list[float]
        Observed sample.
    ci : float or None
        Confidence level.  Defaults to ``config.BOOTSTRAP_CI_LEVEL``.
    n_boot : int or None
        Number of bootstrap resamples.
    remove : bool or None
        Whether to remove outliers before computing CI.  Defaults to
        ``config.OUTLIER_REMOVE``.

    Returns
    -------
    Summary
    """
    if remove is None:
        remove = config.OUTLIER_REMOVE

    if not data:
        return Summary(
            n=0, mean=0.0, std=0.0, median=0.0,
            min=0.0, max=0.0, ci=(0.0, 0.0),
        )

    outliers = detect_outliers_iqr(data)
    n_original = len(data)

    if remove and outliers:
        clean, _ = remove_outliers(data)
    else:
        clean = list(data)

    ci_lo, ci_hi = bootstrap_ci(clean, ci=ci, n_boot=n_boot)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    mean = statistics.mean(clean) if clean else 0.0
    cv = std / abs(mean) if abs(mean) > 1e-15 else 0.0

    return Summary(
        n=len(clean),
        mean=mean,
        std=std,
        median=statistics.median(clean) if clean else 0.0,
        min=min(clean) if clean else 0.0,
        max=max(clean) if clean else 0.0,
        ci=(ci_lo, ci_hi),
        outliers=outliers,
        n_outliers=len(outliers),
        n_original=n_original,
        cv=cv,
    )


# ── Repeat-policy automation ───────────────────────────────────────────


def should_repeat(
    data: list[float],
    outlier_fraction_threshold: float | None = None,
    cv_threshold: float | None = None,
) -> RepeatDecision:
    """Decide whether to re-run a benchmark based on observed data quality.

    Triggers a repeat if:
    1. Outlier fraction exceeds *outlier_fraction_threshold*, OR
    2. Coefficient of variation (after outlier removal) exceeds *cv_threshold*.

    Parameters
    ----------
    data : list[float]
        Observed wall-clock times.
    outlier_fraction_threshold : float or None
        Defaults to ``config.OUTLIER_MAX_FRACTION``.
    cv_threshold : float or None
        Defaults to ``config.CV_THRESHOLD``.

    Returns
    -------
    RepeatDecision
    """
    if outlier_fraction_threshold is None:
        outlier_fraction_threshold = config.OUTLIER_MAX_FRACTION
    if cv_threshold is None:
        cv_threshold = config.CV_THRESHOLD

    outliers = detect_outliers_iqr(data)
    outlier_frac = len(outliers) / len(data) if data else 0.0

    clean, _ = remove_outliers(data)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    mean = statistics.mean(clean) if clean else 0.0
    cv = std / abs(mean) if abs(mean) > 1e-15 else 0.0

    if outlier_frac > outlier_fraction_threshold and outlier_frac > 0.1:
        return RepeatDecision(
            should_repeat=True,
            reason=f"outlier fraction {outlier_frac:.0%} exceeds threshold",
            outlier_fraction=outlier_frac,
            cv=cv,
        )

    if cv > cv_threshold:
        return RepeatDecision(
            should_repeat=True,
            reason=f"CV {cv:.3f} exceeds threshold {cv_threshold:.3f}",
            outlier_fraction=outlier_frac,
            cv=cv,
        )

    return RepeatDecision(
        should_repeat=False,
        reason="within tolerance",
        outlier_fraction=outlier_frac,
        cv=cv,
    )


# ── Formatting ─────────────────────────────────────────────────────────


def format_ci(lo: float, hi: float, precision: int = 4) -> str:
    """Format a CI as '[lo, hi]'."""
    return f"[{lo:.{precision}f},{hi:.{precision}f}]"


def format_summary(s: Summary, precision: int = 4) -> str:
    """Format a Summary as 'mean ± std [ci_lo, ci_hi] (n=X, outliers=Y)'."""
    p = precision
    return (
        f"{s.mean:.{p}f} ± {s.std:.{p}f} "
        f"{format_ci(*s.ci, precision=p)} "
        f"(n={s.n}/{s.n_original}, outliers={s.n_outliers})"
    )
