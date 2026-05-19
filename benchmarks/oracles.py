"""Oracle call counting for benchmarks.

Uses OracleStats from minimax_aipe.oracles for standardized accounting.
"""

from __future__ import annotations

from minimax_aipe import OracleStats


def count_eg_oracles(n_iters: int) -> OracleStats:
    """Oracle cost of *n_iters* extragradient steps.

    Each EG step evaluates F(z) and F(z_half).  One evaluation of the
    operator F(z) = 1 gradient oracle call (regardless of internal
    component count).  So per iteration: 2 F-evals = 2 gradient calls.
    2 projections per iteration.
    """
    return OracleStats(
        grad_calls=2 * n_iters,
        projection_calls=2 * n_iters,
        oracle_calls=2 * n_iters,
        call_type="gradient",
    )


def count_gda_oracles(n_iters: int) -> OracleStats:
    """Oracle cost of *n_iters* GDA steps.

    Each GDA step evaluates F(z) once = 1 gradient oracle call.
    2 projections per iteration.
    """
    return OracleStats(
        grad_calls=n_iters,
        projection_calls=2 * n_iters,
        oracle_calls=n_iters,
        call_type="gradient",
    )


def count_npe_oracles(n_iters: int) -> OracleStats:
    """Oracle cost of *n_iters* standalone NPE steps."""
    return OracleStats(
        crn_calls=n_iters,
        grad_calls=n_iters,
        hessian_calls=n_iters,
        projection_calls=2 * n_iters,
        # Default to base calls; the normalized_cost(d) method will handle scaling
        oracle_calls=n_iters, 
        call_type="crn",
    )


def count_solver_oracles(result) -> OracleStats:
    """Extract oracle stats from a SolverResult."""
    return getattr(result, "oracle_stats", OracleStats())
