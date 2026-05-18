"""Convergence-rate estimation from ε-sweep benchmark data.

Fits log-log slopes to (1/ε, oracle_calls) data and compares empirical
convergence rates against theoretical predictions.  Fully self-contained —
reads ``BenchmarkResult`` objects produced by :mod:`benchmarks.convergence`.

Usage::

    from benchmarks.rates import fit_from_convergence_rows, format_rates_table

    fits = fit_from_convergence_rows(convergence_rows)
    print(format_rates_table(fits))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from typing import Optional

import jax.numpy as jnp
from scipy.stats import t as t_dist

from benchmarks.results import BenchmarkResult


# ── Data class ────────────────────────────────────────────────────────────


@dataclass
class RateFit:
    """Fitted convergence rate from ε-sweep data."""

    solver: str
    problem: str
    dim: int
    slope: float                    # empirical exponent
    intercept: float                # log(constant factor)
    r_squared: float
    slope_ci: tuple[float, float]   # 95 % jackknife CI
    n_points: int
    theoretical_slope: Optional[float] = None

    THEORY: dict = field(default_factory=lambda: {
        "aipe_npe": -4 / 7,        # ≈ -0.5714
        "aipe_len": -4 / 7,
        "eg": -1.0,
        "gda": -1.0,
    }, init=False, repr=False)

    def __post_init__(self):
        self.theoretical_slope = self.THEORY.get(self.solver)

    @property
    def matches_theory(self) -> Optional[bool]:
        """True if theoretical slope falls inside the 95 % CI."""
        if self.theoretical_slope is None:
            return None
        return self.slope_ci[0] <= self.theoretical_slope <= self.slope_ci[1]


# ── Private helpers ───────────────────────────────────────────────────────


def _ols(x: jnp.ndarray, y: jnp.ndarray) -> tuple[float, float, float]:
    """Standard OLS: y ~ slope · x + intercept.

    Returns
    -------
    slope, intercept, r_squared
    """
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    dx = x - x_mean
    dy = y - y_mean
    slope = jnp.sum(dx * dy) / jnp.sum(dx * dx)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = jnp.sum((y - y_pred) ** 2)
    ss_tot = jnp.sum((y - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r_squared)


def _jackknife_slope_ci(
    x: jnp.ndarray,
    y: jnp.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Leave-one-out jackknife CI for the OLS slope.

    Uses :func:`scipy.stats.t.ppf` for the critical value.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Lower and upper bounds of the CI on the slope.

    Notes
    -----
    If ``len(x) < 3`` the CI is degenerate: ``(slope, slope)``.
    """
    n = len(x)
    full_slope, _, _ = _ols(x, y)

    if n < 3:
        return (full_slope, full_slope)

    # Compute leave-one-out slopes
    loo_slopes: list[float] = []
    for i in range(n):
        mask = jnp.arange(n) != i
        si, _, _ = _ols(x[mask], y[mask])
        loo_slopes.append(si)

    slopes_arr = jnp.array(loo_slopes)
    theta_bar = jnp.mean(slopes_arr)
    se = jnp.sqrt((n - 1) / n * jnp.sum((slopes_arr - theta_bar) ** 2))

    df = n - 1
    t_crit = float(t_dist.ppf((1 + confidence) / 2, df))

    lo = float(full_slope - t_crit * se)
    hi = float(full_slope + t_crit * se)
    return (lo, hi)


def _group_by_solver_problem_dim(
    rows: list[BenchmarkResult],
) -> list[tuple[tuple[str, str, int], list[BenchmarkResult]]]:
    """Thin wrapper around :func:`itertools.groupby` with sorting.

    Key: ``(r.solver, r.problem, r.dim)``.
    """
    sorted_rows = sorted(rows, key=lambda r: (r.solver, r.problem, r.dim))
    return [
        ((solver, problem, dim), list(group))
        for (solver, problem, dim), group in groupby(
            sorted_rows, key=lambda r: (r.solver, r.problem, r.dim)
        )
    ]


# ── Public API ────────────────────────────────────────────────────────────


def fit_loglog_slope(
    epsilons: list[float],
    oracle_calls: list[int],
    solver: str = "",
    problem: str = "",
    dim: int = 0,
) -> RateFit:
    """Fit a log-log slope to (1/ε, oracle_calls) data.

    Parameters
    ----------
    epsilons
        Target duality gaps (ε values).
    oracle_calls
        Total oracle calls required to reach each ε.
    solver, problem, dim
        Identity fields stored in the returned :class:`RateFit`.

    Returns
    -------
    RateFit

    Raises
    ------
    ValueError
        If fewer than 3 data points are provided.
    """
    if len(epsilons) < 3:
        raise ValueError(
            f"Need ≥ 3 data points for log-log fit, got {len(epsilons)}"
        )

    x = jnp.log(1.0 / jnp.array(epsilons))
    y = jnp.log(jnp.array(oracle_calls, dtype=jnp.float64))

    slope, intercept, r_squared = _ols(x, y)
    ci_lo, ci_hi = _jackknife_slope_ci(x, y)

    return RateFit(
        solver=solver,
        problem=problem,
        dim=dim,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        slope_ci=(ci_lo, ci_hi),
        n_points=len(epsilons),
    )


def fit_from_convergence_rows(
    rows: list[BenchmarkResult],
    require_converged: bool = True,
) -> list[RateFit]:
    """Fit log-log slopes from convergence benchmark rows.

    Groups rows by *(solver, problem, dim)*, optionally filters to
    converged entries, and fits one :class:`RateFit` per group.

    Parameters
    ----------
    rows
        Convergence sweep results (e.g. from ``sweep_epsilon``).
    require_converged
        If True, keep only rows where ``gap_achieved`` is True.

    Returns
    -------
    list[RateFit]
        One fit per group that has ≥ 3 usable data points.
    """
    groups = _group_by_solver_problem_dim(rows)
    fits: list[RateFit] = []

    for (solver, problem, dim), group_rows in groups:
        # Sort by epsilon descending (largest → smallest)
        group_rows.sort(key=lambda r: r.epsilon, reverse=True)

        # Filter to converged rows if requested
        if require_converged:
            group_rows = [r for r in group_rows if r.gap_achieved]

        # Need at least 3 points for a meaningful log-log fit
        if len(group_rows) < 3:
            continue

        epsilons = [r.epsilon for r in group_rows]
        calls = [r.oracle_stats.oracle_calls for r in group_rows]

        fit = fit_loglog_slope(
            epsilons=epsilons,
            oracle_calls=calls,
            solver=solver,
            problem=problem,
            dim=dim,
        )
        fits.append(fit)

    return fits


def format_rates_table(fits: list[RateFit]) -> str:
    """Format a human-readable table of fitted convergence rates.

    Columns::

        Solver | Problem | Dim | Slope | CI [lo, hi] | R² | Theory | Match?

    *Match?* is ✓ when the theoretical slope falls inside the 95 % CI,
    ✗ when it does not, and — when no theoretical prediction exists.

    Parameters
    ----------
    fits
        Fitted rates to display.

    Returns
    -------
    str
        Formatted multi-line table.
    """
    if not fits:
        return "(no fits)"

    headers = [
        "Solver", "Problem", "Dim",
        "Slope", "CI [lo, hi]", "R²",
        "Theory", "Match?",
    ]

    # ── Build cell rows ─────────────────────────────────────────────
    cell_rows: list[list[str]] = []
    for f in fits:
        if f.matches_theory is True:
            match_str = "✓"
        elif f.matches_theory is False:
            match_str = "✗"
        else:
            match_str = "—"

        theory_str = (
            f"{f.theoretical_slope:.4f}"
            if f.theoretical_slope is not None
            else "—"
        )

        cell_rows.append([
            f.solver,
            f.problem,
            str(f.dim),
            f"{f.slope:.4f}",
            f"[{f.slope_ci[0]:.4f}, {f.slope_ci[1]:.4f}]",
            f"{f.r_squared:.4f}",
            theory_str,
            match_str,
        ])

    # ── Column widths ───────────────────────────────────────────────
    widths = [len(h) for h in headers]
    for row in cell_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # ── Render ──────────────────────────────────────────────────────
    # Numeric columns: right-align.  Everything else: left-align.
    _NUMERIC = {2, 3, 5}  # Dim, Slope, R²

    def _fmt(cells: list[str]) -> str:
        parts: list[str] = []
        for i, cell in enumerate(cells):
            if i in _NUMERIC:
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "  ".join(parts)

    sep = "  ".join("─" * w for w in widths)
    lines: list[str] = [_fmt(headers), sep]
    for row in cell_rows:
        lines.append(_fmt(row))

    return "\n".join(lines)
