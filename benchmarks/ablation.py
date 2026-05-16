"""Component ablation experiments.

Measures the isolated effect of individual solver components:
  - NPE vs LEN (Hessian reuse)
  - Lazy vs fresh Hessians (m_lazy sweep)
  - Warm-starting impact
  - Early stopping impact (npe_T_factor sweep)
"""

from __future__ import annotations

import statistics
import time

from minimax_aipe import solve
from minimax_aipe.problem import BenchmarkProblem
from benchmarks import config
from benchmarks.results import BenchmarkResult
from benchmarks.stats import bootstrap_ci


def ablation_m_lazy(
    prob,
    epsilon: float | None = None,
    m_values: list[int] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time and oracle calls as m_lazy varies.

    m=1 is equivalent to fresh Hessians (NPE).  Larger m reuses more.

    Parameters
    ----------
    prob : BenchmarkProblem
        From the problem zoo.
    epsilon : float or None
        Target gap.  Defaults to ``config.EPSILON_DEFAULT``.
    m_values : list[int]
        Hessian reuse intervals to test.
    n_repeats : int or None
        Timed runs per configuration.  Defaults to ``config.N_REPEATS_SCALING``.

    Returns
    -------
    list[BenchmarkResult]
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    m_values = m_values or [1, 3, 5, 10, 20]
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    rows = []
    for m in m_values:
        # Warmup run to avoid JIT compilation overhead in the first timed run
        _ = solve(problem, epsilon=epsilon, M_saddle="len", m_lazy=m, z0=prob.z0)

        times = []
        result = None
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            r = solve(problem, epsilon=epsilon, M_saddle="len", m_lazy=m, z0=prob.z0)
            times.append(time.perf_counter() - t0)
            result = r

        ci = bootstrap_ci(times)
        rows.append(BenchmarkResult(
            solver="aipe_len",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(times),
            wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
            ci=ci,
            oracle_stats=result.oracle_stats,
            converged=result.converged,
            gap_achieved=result.gap <= epsilon,
            final_gap=float(result.gap),
            iterations=result.iterations,
            m_lazy=m,
            normalized_cost=result.oracle_stats.normalized_cost(d),
        ))

    return rows


def ablation_npe_t_factor(
    prob,
    epsilon: float | None = None,
    t_factors: list[float] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time and oracle calls as npe_T_factor varies."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    t_factors = t_factors or [0.5, 1.0, 1.5, 2.0, 3.0]
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    rows = []
    for tf in t_factors:
        times = []
        result = None
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            r = solve(
                problem, epsilon=epsilon, M_saddle="npe",
                npe_T_factor=tf, z0=prob.z0,
            )
            times.append(time.perf_counter() - t0)
            result = r

        ci = bootstrap_ci(times)
        rows.append(BenchmarkResult(
            solver="aipe_npe",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(times),
            wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
            ci=ci,
            oracle_stats=result.oracle_stats,
            converged=result.converged,
            gap_achieved=result.gap <= epsilon,
            final_gap=float(result.gap),
            iterations=result.iterations,
            npe_T_factor=tf,
            normalized_cost=result.oracle_stats.normalized_cost(d),
        ))

    return rows


def ablation_npe_vs_len(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Head-to-head NPE vs LEN on a single problem."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    results = []

    for M_saddle in ("npe", "len"):
        kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": prob.z0}
        if M_saddle == "len":
            kwargs["m_lazy"] = 5
        _ = solve(problem, **kwargs)
        times = []
        result = None
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            result = solve(problem, **kwargs)
            times.append(time.perf_counter() - t0)
        ci = bootstrap_ci(times)
        results.append(BenchmarkResult(
            solver=f"aipe_{M_saddle}",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(times),
            wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
            ci=ci,
            oracle_stats=result.oracle_stats,
            converged=result.converged,
            gap_achieved=result.gap <= epsilon,
            final_gap=float(result.gap),
            iterations=result.iterations,
            normalized_cost=result.oracle_stats.normalized_cost(d),
        ))

    return results


def format_ablation_m_table(rows: list[BenchmarkResult]) -> str:
    """Format m_lazy ablation as a text table."""
    header = f"{'Problem':<18} {'Dim':>4}  {'m_lazy':>6}  {'Time (s)':>24}  {'Calls':>6}  {'Gap':>10}"
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {r.m_lazy:>6}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  {r.oracle_stats.crn_calls:>6}  {r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_t_table(rows: list[BenchmarkResult]) -> str:
    """Format npe_T_factor ablation as a text table."""
    header = f"{'Problem':<18} {'Dim':>4}  {'T_factor':>8}  {'Time (s)':>24}  {'Calls':>6}  {'Gap':>10}"
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        tf = r.npe_T_factor or 0.0
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {tf:>8.1f}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  {r.oracle_stats.crn_calls:>6}  {r.final_gap:>10.6f}"
        )
    return "\n".join(lines)
