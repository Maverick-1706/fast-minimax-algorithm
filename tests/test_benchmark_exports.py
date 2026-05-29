"""Regression tests for benchmark export shape and ablation gating."""

from benchmarks.ablation import ablation_no_cubic
from benchmarks.export import flatten_ablation_rows
from benchmarks.problems import get_problem
from benchmarks.results import BenchmarkResult
from minimax_aipe import OracleStats


def test_ablation_compare_rows_are_structured_records():
    row = BenchmarkResult(
        solver="aipe_npe",
        problem="bilinear",
        dim=2,
        epsilon=0.1,
        wall_time_mean=0.01,
        wall_time_std=0.0,
        ci=(0.01, 0.01),
        oracle_stats=OracleStats(crn_calls=3, oracle_calls=3, call_type="crn"),
        converged=True,
        gap_achieved=True,
        final_gap=0.0,
        iterations=1,
        gap_source="exact",
    )

    flat = flatten_ablation_rows([row])

    assert isinstance(flat[0], dict)
    assert flat[0]["solver"] == "aipe_npe"
    assert flat[0]["gap_source"] == "exact"
    assert flat[0]["oracle_crn_calls"] == 3
    assert not isinstance(flat[0], str)


def test_no_cubic_skips_unsupported_family_without_solving():
    problem = get_problem("logsumexp_saddle", 2, seed=0)

    rows = ablation_no_cubic(problem, epsilon=0.1, n_repeats=1)

    assert rows == []
