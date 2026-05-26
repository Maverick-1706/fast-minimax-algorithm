"""Focused tests for benchmark timing and solver-comparison smoke paths."""

from __future__ import annotations

import math

from minimax_aipe import solve
from benchmarks import config
from benchmarks.export import flatten_speed_rows
from benchmarks.problems import get_problem
from benchmarks.time_solve import _time_callable, benchmark_solver_comparison


def test_time_callable_does_not_add_extra_repeats_for_stable_sample(monkeypatch):
    monkeypatch.setattr(config, "AUTO_REPEAT", True)
    monkeypatch.setattr(config, "AUTO_REPEAT_MAX_EXTRA", 3)
    monkeypatch.setattr(config, "AUTO_REPEAT_N", 2)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return 0

    out = _time_callable(fn, n_warmup=0, n_repeats=1)

    assert len(out["raw"]) == 1
    assert calls["n"] == 1


def test_time_callable_only_adds_extra_repeats_when_policy_triggers(monkeypatch):
    monkeypatch.setattr(config, "AUTO_REPEAT", True)
    monkeypatch.setattr(config, "AUTO_REPEAT_MAX_EXTRA", 3)
    monkeypatch.setattr(config, "AUTO_REPEAT_N", 2)

    decisions = iter([True, False])

    def fake_should_repeat(_times):
        class Decision:
            should_repeat = next(decisions)
        return Decision()

    monkeypatch.setattr("benchmarks.time_solve.should_repeat", fake_should_repeat)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return 0

    out = _time_callable(fn, n_warmup=0, n_repeats=1)

    assert len(out["raw"]) == 3
    assert calls["n"] == 3


def test_benchmark_solver_comparison_smoke_exports_expected_keys(monkeypatch):
    monkeypatch.setattr(config, "AUTO_REPEAT", False)
    monkeypatch.setattr(config, "BOOTSTRAP_N_RESAMPLES", 64)
    monkeypatch.setattr(config, "BCA_ENABLED", False)

    problem = get_problem("bilinear", 2, seed=0)
    rows = benchmark_solver_comparison([problem], epsilon=0.01, n_repeats=1)

    assert {row.solver for row in rows} == {"aipe_npe", "aipe_len", "eg", "gda"}
    assert all(math.isfinite(row.wall_time_mean) and row.wall_time_mean > 0.0 for row in rows)
    npe_row = next(row for row in rows if row.solver == "aipe_npe")
    assert npe_row.gap_achieved

    flat_rows = flatten_speed_rows(rows)
    required_keys = {"solver", "problem", "ci_lo", "ci_hi", "oracle_calls", "normalized_cost"}
    for row in flat_rows:
        assert required_keys.issubset(row)


def test_solve_handles_zero_rho_bilinear_with_or_without_fallback():
    problem = get_problem("bilinear", 2, seed=0)

    result = solve(problem.problem, epsilon=1e-3, M_saddle="npe", z0=problem.z0)
    baseline = solve(
        problem.problem,
        epsilon=1e-3,
        M_saddle="npe",
        z0=problem.z0,
        no_acceleration=True,
    )

    assert result.converged
    assert float(result.gap) <= 1e-3
    assert baseline.converged
    assert float(baseline.gap) <= 1e-3
    assert result.history is not None
    fallback_used = result.history.get("fallback_from_accelerated") is True
    if fallback_used:
        assert result.history["accelerated_gap"] >= float(result.gap)
