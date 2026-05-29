from __future__ import annotations

import time

import jax.numpy as jnp

from benchmarks.baselines import BaselineResult, run_eg_jit_benchmark
from benchmarks.convergence import _gap_source as convergence_gap_source, sweep_epsilon
from benchmarks.problems import get_problem
from minimax_aipe import OracleStats
from minimax_aipe.gap import estimate_gap
from minimax_aipe._framework.api import _maybe_recover_failed_result
from minimax_aipe._framework.params import _safe_gap
from minimax_aipe.problem import MinimaxProblem, SolverResult


def _estimated_gap_problem() -> MinimaxProblem:
    def f(x, y):
        return jnp.dot(x, x) - jnp.dot(y, y)

    return MinimaxProblem(
        f=f,
        dim_x=1,
        dim_y=1,
        D_x=2.0,
        D_y=2.0,
        ell=1.0,
        rho=1.0,
    )


def test_recovery_skips_estimated_gap_problems(monkeypatch):
    problem = _estimated_gap_problem()
    result = SolverResult(
        x=jnp.zeros(1),
        y=jnp.zeros(1),
        gap=1.0,
        iterations=1,
        oracle_calls=0,
        oracle_stats=OracleStats(),
        converged=False,
        history={},
    )
    calls = {"n": 0}

    def fake_solve(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("recovery should not launch extra solves for estimated gaps")

    monkeypatch.setattr("minimax_aipe._framework.api.solve", fake_solve)

    recovered = _maybe_recover_failed_result(
        result,
        problem,
        0.1,
        gamma=1.0,
        M_saddle="npe",
        m_lazy=1,
        npe_T_factor=0.5,
        z0=jnp.zeros(2),
        verbose=False,
        no_restart=False,
        no_acceleration=False,
        fixed_inner_iters=None,
        allow_recovery=True,
        max_recovery_calls=None,
    )

    assert recovered is result
    assert calls["n"] == 0


def test_run_eg_benchmark_uses_safe_gap(monkeypatch):
    problem = get_problem("bilinear", 2, seed=0).problem

    monkeypatch.setattr(
        "benchmarks.baselines.run_eg_jit",
        lambda *args, **kwargs: (
            jnp.zeros(problem.dim_x + problem.dim_y),
            0.0,
            0.01,
            200,
        ),
    )
    monkeypatch.setattr("benchmarks.baselines._safe_gap", lambda *args, **kwargs: 0.0123)

    result = run_eg_jit_benchmark(problem, epsilon=0.01)

    assert result.gap == 0.0123
    assert not result.gap_achieved


def test_sweep_epsilon_records_timeouts(monkeypatch):
    problem = get_problem("bilinear", 2, seed=0)

    def slow_solve(*args, **kwargs):
        time.sleep(0.2)
        raise AssertionError("timeout should interrupt before completion")

    def slow_baseline(*args, **kwargs):
        time.sleep(0.2)
        raise AssertionError("timeout should interrupt before completion")

    monkeypatch.setattr("benchmarks.convergence.solve", slow_solve)
    monkeypatch.setattr("benchmarks.convergence.run_npe_restart_jit_benchmark", slow_baseline)
    monkeypatch.setattr("benchmarks.convergence.run_eg_jit_benchmark", slow_baseline)

    rows = sweep_epsilon(problem, [0.1], timeout_seconds=0.05)

    assert len(rows) == 4
    assert all(row.final_gap == float("inf") for row in rows)
    assert all(not row.converged for row in rows)
    assert all(row.extra_metadata.get("timed_out") is True for row in rows)


def test_sweep_epsilon_progress_callback_receives_each_solver(monkeypatch):
    problem = get_problem("bilinear", 2, seed=0)
    seen = []

    def fake_solve(*args, **kwargs):
        return SolverResult(
            x=jnp.zeros(problem.problem.dim_x),
            y=jnp.zeros(problem.problem.dim_y),
            gap=0.0,
            iterations=1,
            oracle_calls=1,
            oracle_stats=OracleStats(oracle_calls=1, crn_calls=1),
            converged=True,
            history={},
        )

    baseline = BaselineResult(
        x=jnp.zeros(problem.problem.dim_x),
        y=jnp.zeros(problem.problem.dim_y),
        gap=0.0,
        iterations=1,
        wall_time=0.01,
        converged=True,
        gap_achieved=True,
        final_residual=0.0,
        oracle_stats=OracleStats(oracle_calls=1, grad_calls=1, call_type="gradient"),
    )

    monkeypatch.setattr("benchmarks.convergence.solve", fake_solve)
    monkeypatch.setattr("benchmarks.convergence.run_npe_restart_jit_benchmark", lambda *args, **kwargs: baseline)
    monkeypatch.setattr("benchmarks.convergence.run_eg_jit_benchmark", lambda *args, **kwargs: baseline)

    rows = sweep_epsilon(
        problem,
        [0.1],
        progress_callback=lambda row: seen.append((row.solver, row.epsilon)),
    )

    assert len(rows) == 4
    assert seen == [
        ("aipe_npe", 0.1),
        ("aipe_len", 0.1),
        ("npe_restart", 0.1),
        ("eg", 0.1),
    ]


def test_convergence_gap_source_requires_exact_duality_gap():
    problem = get_problem("nonzero_rho", 5, seed=0)
    assert problem.gap_star == 0.0
    assert convergence_gap_source(problem) == "estimated"


def test_estimate_gap_nonzero_rho_at_origin_is_finite():
    problem = get_problem("nonzero_rho", 5, seed=0)
    gap = estimate_gap(
        problem.problem,
        jnp.zeros(problem.problem.dim_x),
        jnp.zeros(problem.problem.dim_y),
        num_restarts=4,
        num_steps=200,
    )
    assert jnp.isfinite(gap)
    assert float(gap) >= 0.0


def test_safe_gap_maps_nan_to_inf(monkeypatch):
    problem = _estimated_gap_problem()
    monkeypatch.setattr(
        "minimax_aipe._framework.params.estimate_gap",
        lambda *args, **kwargs: jnp.array(jnp.nan),
    )
    gap = _safe_gap(problem, jnp.zeros(problem.dim_x), jnp.zeros(problem.dim_y), 0.1)
    assert gap == float("inf")
