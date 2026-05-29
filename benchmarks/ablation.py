"""Component ablation experiments.

Measures the isolated effect of individual solver components:
  - NPE vs LEN (Hessian reuse)
  - Lazy vs fresh Hessians (m_lazy sweep)
  - Warm-starting impact
  - Early stopping impact (npe_T_factor sweep)
  - Cubic regularization (rho=0 vs original)
  - Restart schedule (single-shot vs iterative)
  - Nesterov acceleration (plain gradient vs accelerated)
  - Inner-iteration budget (fixed vs adaptive)
  - Initialization strategy (heuristic vs zero vs random)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from minimax_aipe import solve
from benchmarks import config
from benchmarks.reporting import (
    gap_source as benchmark_gap_source,
    normalized_cost,
    row_normalized_cost,
    sync_result,
)
from benchmarks.results import BenchmarkResult
from benchmarks.stats import bootstrap_ci
from benchmarks.problems import get_problem
from minimax_aipe.problem import BenchmarkProblem


def _gap_source(prob: BenchmarkProblem) -> str:
    return benchmark_gap_source(prob)


@dataclass(frozen=True)
class _AblationContext:
    benchmark: BenchmarkProblem
    problem_obj: object
    problem_name: str
    dim: int
    epsilon: float
    n_repeats: int
    gap_source: str


@dataclass(frozen=True)
class _AblationCase:
    solver: str
    kwargs: dict
    extra: dict | None = None
    problem_obj: object | None = None
    problem_name: str | None = None


@dataclass(frozen=True)
class _TableColumn:
    header: str
    width: int
    align: str
    render: Callable[[BenchmarkResult], str]


def _resolve_ablation_context(
    prob,
    *,
    epsilon: float | None,
    n_repeats: int | None,
) -> _AblationContext:
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    return _AblationContext(
        benchmark=prob,
        problem_obj=problem,
        problem_name=prob.name or "?",
        dim=prob.dim or problem.dim_x,
        epsilon=epsilon,
        n_repeats=n_repeats,
        gap_source=_gap_source(prob),
    )


def _paired_saddle_cases(
    *,
    solver_suffix: str = "",
    len_m_lazy: int = 5,
    extra_kwargs: dict | None = None,
    problem_obj=None,
    problem_name: str | None = None,
) -> list[_AblationCase]:
    base_kwargs = dict(extra_kwargs or {})
    cases: list[_AblationCase] = []
    for M_saddle in ("npe", "len"):
        kwargs = dict(base_kwargs)
        kwargs["M_saddle"] = M_saddle
        if M_saddle == "len":
            kwargs["m_lazy"] = len_m_lazy
        cases.append(_AblationCase(
            solver=f"aipe_{M_saddle}{solver_suffix}",
            kwargs=kwargs,
            problem_obj=problem_obj,
            problem_name=problem_name,
        ))
    return cases


# ──────────────────────────────────────────────────────────────────────
# Existing experiments
# ──────────────────────────────────────────────────────────────────────


def ablation_m_lazy(
    prob,
    epsilon: float | None = None,
    m_values: list[int] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time and oracle calls as m_lazy varies."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    m_values = m_values or [1, 3, 5, 10, 20]
    return _run_ablation_cases(
        ctx,
        [
            _AblationCase(
                solver="aipe_len",
                kwargs={"M_saddle": "len", "m_lazy": m},
                extra={"m_lazy": m},
            )
            for m in m_values
        ],
    )


def ablation_npe_t_factor(
    prob,
    epsilon: float | None = None,
    t_factors: list[float] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time and oracle calls as npe_T_factor varies."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    t_factors = t_factors or [0.5, 1.0, 1.5, 2.0, 3.0]
    return _run_ablation_cases(
        ctx,
        [
            _AblationCase(
                solver="aipe_npe",
                kwargs={"M_saddle": "npe", "npe_T_factor": tf},
                extra={"npe_T_factor": tf},
            )
            for tf in t_factors
        ],
    )


def ablation_npe_vs_len(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Head-to-head NPE vs LEN on a single problem."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    return _run_ablation_cases(ctx, _paired_saddle_cases())


# ──────────────────────────────────────────────────────────────────────
# Private timing helpers
# ──────────────────────────────────────────────────────────────────────


def _sync_result(result) -> None:
    """Force synchronization on JAX arrays inside result."""
    sync_result(result)


def _time_solve_loop(
    problem,
    n_repeats: int,
    kwargs: dict,
) -> tuple[list[float], object]:
    """Run ``solve(problem, **kwargs)`` with warmup + n_repeats timed calls."""
    import gc

    w_result = None
    for _ in range(config.N_WARMUP):
        w_result = solve(problem, **kwargs)
        _sync_result(w_result)

    times: list[float] = []
    result = w_result
    for _ in range(n_repeats):
        gc.collect()
        _ = jnp.zeros(1).block_until_ready()
        t0 = time.perf_counter()
        result = solve(problem, **kwargs)
        _sync_result(result)
        times.append(time.perf_counter() - t0)

    if not times:
        times = [0.0]

    if result is None:
        result = solve(problem, **kwargs)
        _sync_result(result)

    return times, result


def _build_result(
    *,
    solver: str,
    problem_obj,
    problem: str,
    dim: int,
    epsilon: float,
    times: list[float],
    result,
    extra: dict | None = None,
    gap_source: str = "unknown",
) -> BenchmarkResult:
    """Construct a BenchmarkResult from timing data."""
    ci = bootstrap_ci(times)
    kwargs = dict(
        solver=solver,
        problem=problem,
        dim=dim,
        epsilon=epsilon,
        wall_time_mean=statistics.mean(times) if times != [0.0] else 0.0,
        wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        ci=ci,
        oracle_stats=result.oracle_stats,
        converged=result.converged,
        gap_achieved=result.gap <= epsilon,
        final_gap=float(result.gap),
        iterations=result.iterations,
        normalized_cost=normalized_cost(problem_obj, result.oracle_stats),
        gap_source=gap_source,
    )
    if extra:
        extra_meta = {}
        for k, v in extra.items():
            if k in BenchmarkResult.__dataclass_fields__:
                kwargs[k] = v
            else:
                extra_meta[k] = v
        kwargs["extra_metadata"] = extra_meta
    return BenchmarkResult(**kwargs)


def _run_ablation_cases(
    ctx: _AblationContext,
    cases: list[_AblationCase],
) -> list[BenchmarkResult]:
    rows: list[BenchmarkResult] = []
    for case in cases:
        problem_obj = case.problem_obj if case.problem_obj is not None else ctx.problem_obj
        kwargs = {"epsilon": ctx.epsilon, "z0": ctx.benchmark.z0, **case.kwargs}
        times, result = _time_solve_loop(problem_obj, ctx.n_repeats, kwargs)
        rows.append(_build_result(
            solver=case.solver,
            problem_obj=problem_obj,
            problem=case.problem_name or ctx.problem_name,
            dim=ctx.dim,
            epsilon=ctx.epsilon,
            times=times,
            result=result,
            extra=case.extra,
            gap_source=ctx.gap_source,
        ))
    return rows


# ──────────────────────────────────────────────────────────────────────
# New Experiments
# ──────────────────────────────────────────────────────────────────────


def ablation_no_cubic(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: force rho=0 in solver when problem has rho > 0."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    if ctx.benchmark.name not in ("nonzero_rho", "random_cubic"):
        return []

    seed = ctx.benchmark.meta.seed if ctx.benchmark.meta is not None else None
    zero_prob = get_problem(ctx.benchmark.name, ctx.dim, seed=seed, rho=0.0)
    return _run_ablation_cases(
        ctx,
        _paired_saddle_cases(problem_obj=zero_prob.problem, problem_name=ctx.problem_name),
    )


def ablation_no_restart(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: single-shot AIPE with no restarts."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    return _run_ablation_cases(
        ctx,
        _paired_saddle_cases(
            solver_suffix="_no_restart",
            extra_kwargs={"no_restart": True},
        ),
    )


def ablation_no_acceleration(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: disable Nesterov acceleration in the outer loop."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    return _run_ablation_cases(
        ctx,
        _paired_saddle_cases(
            solver_suffix="_no_accel",
            extra_kwargs={"no_acceleration": True},
        ),
    )


def ablation_fixed_inner(
    prob,
    epsilon: float | None = None,
    inner_iters_list: list[int] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: fixed inner-iteration budget sweep."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    inner_iters_list = inner_iters_list or [1, 5, 10, 20, 50, 100]
    return _run_ablation_cases(
        ctx,
        [
            _AblationCase(
                solver="aipe_npe_fixed_inner",
                kwargs={"M_saddle": "npe", "fixed_inner_iters": n_inner},
                extra={"fixed_inner_iters": n_inner},
            )
            for n_inner in inner_iters_list
        ],
    )


def ablation_init_comparison(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Ablation: compare initialization strategies."""
    ctx = _resolve_ablation_context(prob, epsilon=epsilon, n_repeats=n_repeats)
    d = ctx.problem_obj.dim_x + ctx.problem_obj.dim_y
    key = jax.random.PRNGKey(seed)
    raw_random = jax.random.normal(key, (d,)) * 0.5
    raw_zero = jnp.zeros(d)
    dx = ctx.problem_obj.dim_x

    random_z0 = jnp.concatenate([
        ctx.problem_obj.project_x(raw_random[:dx]),
        ctx.problem_obj.project_y(raw_random[dx:]),
    ])
    zero_z0 = jnp.concatenate([
        ctx.problem_obj.project_x(raw_zero[:dx]),
        ctx.problem_obj.project_y(raw_zero[dx:]),
    ])
    init_configs = [
        ("heuristic", ctx.benchmark.z0),
        ("zero", zero_z0),
        ("random", random_z0),
    ]

    cases: list[_AblationCase] = []
    for init_name, z0 in init_configs:
        for case in _paired_saddle_cases(problem_name=f"{ctx.problem_name}_{init_name}"):
            cases.append(_AblationCase(
                solver=case.solver,
                kwargs={**case.kwargs, "z0": z0},
                problem_name=case.problem_name,
            ))
    return _run_ablation_cases(ctx, cases)


# ──────────────────────────────────────────────────────────────────────
# Resilient Formatters
# ──────────────────────────────────────────────────────────────────────


def _format_time_cell(row: BenchmarkResult) -> str:
    return f"{row.wall_time_mean:>8.4f} [{row.ci[0]:.4f},{row.ci[1]:.4f}]"


def _format_cost_cell(row: BenchmarkResult) -> str:
    return f"{float(row_normalized_cost(row) or 0.0):.2e}"


def _render_table(
    rows: list[BenchmarkResult],
    columns: list[_TableColumn],
    *,
    empty: str = "(no data)",
) -> str:
    if not rows:
        return empty

    header = "  ".join(f"{col.header:{col.align}{col.width}}" for col in columns)
    lines = [header, "─" * len(header)]
    for row in rows:
        lines.append("  ".join(
            f"{col.render(row):{col.align}{col.width}}"
            for col in columns
        ))
    return "\n".join(lines)


def format_ablation_m_table(rows: list[BenchmarkResult]) -> str:
    """Format m_lazy ablation as a text table."""
    return _render_table(rows, [
        _TableColumn("Problem", 18, "<", lambda r: r.problem),
        _TableColumn("Dim", 4, ">", lambda r: str(r.dim)),
        _TableColumn("m_lazy", 6, ">", lambda r: str(r.m_lazy or 0)),
        _TableColumn("Time (s)", 24, ">", _format_time_cell),
        _TableColumn("Cost", 11, ">", _format_cost_cell),
        _TableColumn("Gap", 10, ">", lambda r: f"{r.final_gap:.3e}"),
    ])


def format_ablation_t_table(rows: list[BenchmarkResult]) -> str:
    """Format npe_T_factor ablation as a text table."""
    return _render_table(rows, [
        _TableColumn("Problem", 18, "<", lambda r: r.problem),
        _TableColumn("Dim", 4, ">", lambda r: str(r.dim)),
        _TableColumn("T_factor", 8, ">", lambda r: f"{float(r.npe_T_factor or 0.0):.1f}"),
        _TableColumn("Time (s)", 24, ">", _format_time_cell),
        _TableColumn("Cost", 11, ">", _format_cost_cell),
        _TableColumn("Gap", 10, ">", lambda r: f"{r.final_gap:.3e}"),
    ])


def format_ablation_no_cubic_table(rows: list[BenchmarkResult]) -> str:
    """Compare with/without cubic regularization."""
    return _render_table(rows, [
        _TableColumn("Solver", 22, "<", lambda r: r.solver),
        _TableColumn("Problem", 18, "<", lambda r: r.problem),
        _TableColumn("Dim", 4, ">", lambda r: str(r.dim)),
        _TableColumn("Time (s)", 24, ">", _format_time_cell),
        _TableColumn("Cost", 11, ">", _format_cost_cell),
        _TableColumn("Iters", 6, ">", lambda r: str(r.iterations)),
        _TableColumn("Gap", 10, ">", lambda r: f"{r.final_gap:.3e}"),
    ])


def format_ablation_init_table(rows: list[BenchmarkResult]) -> str:
    """Compare initialization strategies."""
    return _render_table(rows, [
        _TableColumn("Variant", 30, "<", lambda r: r.problem),
        _TableColumn("Solver", 22, "<", lambda r: r.solver),
        _TableColumn("Dim", 4, ">", lambda r: str(r.dim)),
        _TableColumn("Time (s)", 24, ">", _format_time_cell),
        _TableColumn("Cost", 11, ">", _format_cost_cell),
        _TableColumn("Iters", 6, ">", lambda r: str(r.iterations)),
        _TableColumn("Gap", 10, ">", lambda r: f"{r.final_gap:.3e}"),
    ])


def format_ablation_fixed_inner_table(rows: list[BenchmarkResult]) -> str:
    """Fixed inner-iteration sweep table."""
    return _render_table(rows, [
        _TableColumn("Problem", 18, "<", lambda r: r.problem),
        _TableColumn("Dim", 4, ">", lambda r: str(r.dim)),
        _TableColumn("Inner", 6, ">", lambda r: str(getattr(r, "fixed_inner_iters", 0) or 0)),
        _TableColumn("Time (s)", 24, ">", _format_time_cell),
        _TableColumn("Cost", 11, ">", _format_cost_cell),
        _TableColumn("Outer", 6, ">", lambda r: str(getattr(r, "iterations", 0))),
        _TableColumn("Gap", 10, ">", lambda r: f"{r.final_gap:.3e}"),
    ])
