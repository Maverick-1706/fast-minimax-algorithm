"""Convergence-rate estimation from ε-sweep benchmark data.

Fits log-log slopes to (1/ε, normalized_cost) data and compares empirical
convergence rates against theoretical predictions.  Fully self-contained —
reads ``BenchmarkResult`` objects produced by :mod:`benchmarks.convergence`.

All cost values use :meth:`OracleStats.normalized_cost` (gradient-equivalent
FLOP units) so that CRN and gradient-based solvers are commensurable.

Usage::

    from benchmarks.rates import fit_from_convergence_rows, format_rates_table

    fits = fit_from_convergence_rows(convergence_rows)
    print(format_rates_table(fits))
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from itertools import groupby
from typing import Optional
import numpy as np
from scipy.stats import t as t_dist
from benchmarks.results import BenchmarkResult

# ── Problem classification for theoretical rates ─────────────────────────
_SCSC_PROBLEMS = {
    "quadratic", "quadratic_saddle", "ill_quadratic", "box_constrained_quadratic",
    "box_quadratic", "adversarial_training", "diagonal_saddle", "scalable_diagonal",
    "offset_quadratic", "nonzero_rho", "random_cubic",
}

_GENERAL_CVC_PROBLEMS = {
    "bilinear", "ill_bilinear", "sparse_bilinear", "bilinear_polytope",
    "logsumexp_saddle", "separable",
}

def _get_theoretical_slope(solver: str, problem: str) -> Optional[float]:
    """Return the theoretical log-log convergence slope for a solver/problem pair.
    
    Returns None if the rate is sub-polynomial (e.g. logarithmic) or unknown,
    preventing false mismatches in polynomial rate fitting.
    """
    solver = solver.lower()
    is_scsc = problem in _SCSC_PROBLEMS
    is_general = problem in _GENERAL_CVC_PROBLEMS
    
    if solver in ("eg", "gda"):
        if is_scsc:
            return 0.0  # Linear convergence -> O(log(1/ε)) -> log-log slope ≈ 0
        if is_general:
            return 1.0  # Sublinear -> O(1/ε) -> log-log slope = 1.0
        return None
        
    if solver in ("aipe_npe", "aipe_len", "npe_restart", "minimax_aipe"):
        if is_general:
            return 4.0 / 7.0
        # For SCSC, AIPE's rate may differ or not be strictly polynomial 4/7.
        # Return None to avoid false mismatches.
        return None
        
    return None


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
    n_dropped: int = 0              # Number of non-converged points excluded

    def __post_init__(self):
        if self.theoretical_slope is None:
            self.theoretical_slope = _get_theoretical_slope(self.solver, self.problem)

    @property
    def matches_theory(self) -> Optional[bool]:
        """True if theoretical slope falls inside the 95 % CI."""
        if self.theoretical_slope is None:
            return None
        return self.slope_ci[0] <= self.theoretical_slope <= self.slope_ci[1]


# ── Private helpers ───────────────────────────────────────────────────────


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Standard OLS: y ~ slope · x + intercept.

    Returns
    -------
    slope, intercept, r_squared
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    dx = x - x_mean
    dy = y - y_mean
    slope = np.sum(dx * dy) / np.sum(dx * dx)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r_squared)


def _jackknife_slope_ci(
    x: np.ndarray,
    y: np.ndarray,
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
        mask = np.arange(n) != i
        si, _, _ = _ols(x[mask], y[mask])
        loo_slopes.append(si)

    slopes_arr = np.array(loo_slopes)
    theta_bar = np.mean(slopes_arr)
    se = np.sqrt((n - 1) / n * np.sum((slopes_arr - theta_bar) ** 2))

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
    oracle_calls: list[float],
    solver: str = "",
    problem: str = "",
    dim: int = 0,
) -> RateFit:
    """Fit a log-log slope to (1/ε, cost) data.
    
    Parameters
    ----------
    epsilons : list[float]
        Target duality gaps.
    oracle_calls : list[float]
        Normalized cost values (gradient-equivalent FLOP units from
        ``normalized_cost(d)``) or raw oracle calls.  The name is kept
        for backward compatibility with existing call sites.
    solver, problem, dim :标签
        Metadata attached to the returned :class:`RateFit`.
    """
    # Filter out degenerate cases where the solver took <= 1 cost unit
    valid_indices = [i for i, c in enumerate(oracle_calls) if c > 1.0]
    if len(valid_indices) < 3:
        raise ValueError(
            f"Need ≥ 3 valid data points with >1.0 cost, got {len(valid_indices)}"
        )
    # Rebuild clean lists using only valid indices
    epsilons = [epsilons[i] for i in valid_indices]
    oracle_calls = [oracle_calls[i] for i in valid_indices]
    x = np.log(1.0 / np.array(epsilons))
    y = np.log(np.array(oracle_calls, dtype=np.float64))

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
    require_converged: bool = False,
) -> list[RateFit]:
    """Fit log-log slopes from convergence benchmark rows.
    
    Parameters
    ----------
    require_converged : bool, default=False
        If True, drops non-converged points. WARNING: This systematically
        biases the slope optimistically (making the solver appear faster)
        because the hardest (tightest ε) instances are excluded. 
        If False (default), non-converged points are included using their
        exhausted oracle budget as a lower-bound on the true cost.
    """
    groups = _group_by_solver_problem_dim(rows)
    fits: list[RateFit] = []

    for (solver, problem, dim), group_rows in groups:
        # Sort by epsilon descending (largest → smallest)
        group_rows.sort(key=lambda r: r.epsilon, reverse=True)
        n_total = len(group_rows)
        n_dropped = 0

        # Filter to converged rows if requested
        if require_converged:
            converged_rows = [r for r in group_rows if r.gap_achieved]
            n_dropped = n_total - len(converged_rows)
            if n_dropped > 0:
                warnings.warn(
                    f"Dropped {n_dropped} non-converged points for {solver} on {problem} (dim={dim}). "
                    "This biases the slope optimistically. Consider require_converged=False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            group_rows = converged_rows

        # Guard against None stats and degenerate 0 or 1 call iterations
        valid_rows = [
            r for r in group_rows
            if r.oracle_stats is not None and r.oracle_stats.oracle_calls > 1
        ]

        # Gracefully skip the group if we don't have enough valid points
        if len(valid_rows) < 3:
            continue

        epsilons = [r.epsilon for r in valid_rows]
        
        # Use normalized_cost(d) for all cost computations.
        # This produces gradient-equivalent FLOP units that are
        # commensurable across CRN and gradient-based solvers.
        calls = []
        for r in valid_rows:
            stats = r.oracle_stats
            if stats is None:
                calls.append(0.0)
                continue
            d = (r.dim or 1) * 2
            try:
                cost = float(stats.normalized_cost(d))
            except Exception:
                cost = 0.0
            calls.append(cost)

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
        "Theory", "Match?", "Drop",
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
            str(f.n_dropped) if f.n_dropped > 0 else "0",
        ])

    # ── Column widths ───────────────────────────────────────────────
    widths = [len(h) for h in headers]
    for row in cell_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # ── Render ──────────────────────────────────────────────────────
    # Numeric columns: right-align.  Everything else: left-align.
    _NUMERIC = {2, 3, 5, 8}  # Dim, Slope, R², Drop

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
