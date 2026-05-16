"""Oracle call counting for benchmarks.

Uses OracleStats from minimax_aipe.oracles for standardized accounting.
"""

from __future__ import annotations

from minimax_aipe import OracleStats


def count_eg_oracles(n_iters: int) -> OracleStats:
    """Oracle cost of *n_iters* extragradient steps.

    Each EG step evaluates F(z) and F(z_half).  Each F evaluation requires
    one gradient call per player (2 total).  So per iteration: 2 F-evals
    = 4 gradient calls.  2 projections per iteration.
    """
    return OracleStats(
        grad_calls=4 * n_iters,
        projection_calls=2 * n_iters,
        oracle_calls=n_iters,
        call_type="gradient",
    )


def count_gda_oracles(n_iters: int) -> OracleStats:
    """Oracle cost of *n_iters* GDA steps.

    Each GDA step evaluates ∇_x f and ∇_y f once = 2 gradient calls.
    2 projections per iteration.
    """
    return OracleStats(
        grad_calls=2 * n_iters,
        projection_calls=2 * n_iters,
        oracle_calls=n_iters,
        call_type="gradient",
    )


def count_solver_oracles(result) -> OracleStats:
    """Extract oracle stats from a SolverResult."""
    return getattr(result, "oracle_stats", OracleStats())
