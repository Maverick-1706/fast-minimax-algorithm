"""Shared reporting helpers for benchmark harnesses."""

from __future__ import annotations

from minimax_aipe.problem import BenchmarkProblem, MinimaxProblem, OracleStats, has_exact_gap


def _as_problem(problem_or_benchmark: MinimaxProblem | BenchmarkProblem) -> MinimaxProblem:
    if isinstance(problem_or_benchmark, BenchmarkProblem):
        return problem_or_benchmark.problem
    return problem_or_benchmark
def gap_source(problem_or_benchmark: MinimaxProblem | BenchmarkProblem) -> str:
    return "exact" if has_exact_gap(_as_problem(problem_or_benchmark)) else "estimated"


def sync_result(result) -> None:
    if result is None:
        return
    for attr in ("x", "y", "gap"):
        value = getattr(result, attr, None)
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def normalized_cost(
    problem_or_benchmark: MinimaxProblem | BenchmarkProblem,
    oracle_stats: OracleStats | None,
) -> float | None:
    if oracle_stats is None:
        return None

    problem = _as_problem(problem_or_benchmark)
    total_dim = problem.dim_x + problem.dim_y
    projection_weight = getattr(problem, "projection_cost_weight", None)
    return float(
        oracle_stats.normalized_cost(
            total_dim,
            projection_weight=projection_weight,
        )
    )


def row_normalized_cost(row) -> float | None:
    value = getattr(row, "normalized_cost", None)
    if value is not None:
        return float(value)
    oracle_stats = getattr(row, "oracle_stats", None)
    if oracle_stats is None:
        return None
    dim = max(int(getattr(row, "dim", 0) or 0), 1)
    return float(oracle_stats.normalized_cost(dim * 2))
