"""Component ablation experiments.

Measures the isolated effect of individual solver components:
  - NPE vs LEN (Hessian reuse)
  - Lazy vs fresh Hessians (m_lazy sweep)
  - Warm-starting impact
  - Early stopping impact (npe_T_factor sweep)
  - Cubic regularization (ρ=0 vs original)
  - Restart schedule (single-shot vs iterative)
  - Nesterov acceleration (plain gradient vs accelerated)
  - Inner-iteration budget (fixed vs adaptive)
  - Initialization strategy (heuristic vs zero vs random)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import replace

import jax
import jax.numpy as jnp

from minimax_aipe import solve
from minimax_aipe.problem import BenchmarkProblem
from benchmarks import config
from benchmarks.results import BenchmarkResult
from benchmarks.stats import bootstrap_ci
from benchmarks.problems import get_problem


# ──────────────────────────────────────────────────────────────────────
# Existing experiments
# ──────────────────────────────────────────────────────────────────────


def ablation_m_lazy(
    prob,
    epsilon: float | None = None,
    m_values: list[int] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time and oracle calls as m_lazy varies.

    m=1 is equivalent to fresh Hessians (NPE).  Larger m reuses more.
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
        kwargs = {"epsilon": epsilon, "M_saddle": "len", "m_lazy": m, "z0": prob.z0}
        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        rows.append(_build_result(
            solver="aipe_len",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
            extra={"m_lazy": m},
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
        kwargs = {
            "epsilon": epsilon,
            "M_saddle": "npe",
            "npe_T_factor": tf,
            "z0": prob.z0,
        }
        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        rows.append(_build_result(
            solver="aipe_npe",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
            extra={"npe_T_factor": tf},
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
        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        results.append(_build_result(
            solver=f"aipe_{M_saddle}",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────
# Private timing helpers
# ──────────────────────────────────────────────────────────────────────


def _sync_result(result) -> None:
    """Force synchronization on JAX arrays inside result."""
    if result is None:
        return
    if hasattr(result, "x") and hasattr(result.x, "block_until_ready"):
        result.x.block_until_ready()
    if hasattr(result, "y") and hasattr(result.y, "block_until_ready"):
        result.y.block_until_ready()
    if hasattr(result, "gap") and hasattr(result.gap, "block_until_ready"):
        result.gap.block_until_ready()


def _time_solve_loop(
    problem,
    n_repeats: int,
    kwargs: dict,
) -> tuple[list[float], object]:
    """Run ``solve(problem, **kwargs)`` with warmup + n_repeats timed calls."""
    import gc
    w_result = solve(problem, **kwargs)
    _sync_result(w_result)

    times: list[float] = []
    result = w_result  # Fix: Initialize with warmup as the fallback baseline
    for _ in range(n_repeats):
        gc.collect()
        # Synchronize device queue before starting timer
        _ = jnp.zeros(1).block_until_ready()
        t0 = time.perf_counter()
        result = solve(problem, **kwargs)
        _sync_result(result)
        times.append(time.perf_counter() - t0)

    if not times:
        times = [0.0]

    return times, result

def _build_result(
    *,
    solver: str,
    problem: str,
    dim: int,
    epsilon: float,
    times: list[float],
    result,
    d: int,
    extra: dict | None = None,
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
        normalized_cost=result.oracle_stats.normalized_cost(d),
    )
    if extra:
        kwargs.update(extra)
    return BenchmarkResult(**kwargs)


# ──────────────────────────────────────────────────────────────────────
# New Experiments
# ──────────────────────────────────────────────────────────────────────


def ablation_no_cubic(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: force ρ=0 in solver when problem has ρ > 0."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"

    original_problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or original_problem.dim_x
    d = original_problem.dim_x + original_problem.dim_y

    if prob.name and prob.name in ("nonzero_rho", "random_cubic"):
        zero_prob = get_problem(prob.name, dim, seed=getattr(prob.meta, "seed", 0) if prob.meta else 0, rho=0.0)
        problem_no_cubic = zero_prob.problem
    else:
        problem_no_cubic = original_problem

    results = []
    for M_saddle in ("npe", "len"):
        kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": prob.z0}
        if M_saddle == "len":
            kwargs["m_lazy"] = 5

        times, result = _time_solve_loop(problem_no_cubic, n_repeats, kwargs)
        results.append(_build_result(
            solver=f"aipe_{M_saddle}",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
        ))

    return results


def ablation_no_restart(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: single-shot AIPE with no restarts."""
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
        kwargs = {
            "epsilon": epsilon,
            "M_saddle": M_saddle,
            "z0": prob.z0,
            "no_restart": True,
        }
        if M_saddle == "len":
            kwargs["m_lazy"] = 5

        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        results.append(_build_result(
            solver=f"aipe_{M_saddle}_no_restart",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
        ))

    return results


def ablation_no_acceleration(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: disable Nesterov acceleration in the outer loop."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"

    problem = prob.problem
    name = prob.name or "?"
    dim = problem.dim_x
    d = problem.dim_x + problem.dim_y

    results = []
    for M_saddle in ("npe", "len"):
        kwargs = {
            "epsilon": epsilon,
            "M_saddle": M_saddle,
            "z0": prob.z0,
            "no_acceleration": True,
        }
        if M_saddle == "len":
            kwargs["m_lazy"] = 5

        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        results.append(_build_result(
            solver=f"aipe_{M_saddle}_no_accel",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
        ))

    return results


def ablation_fixed_inner(
    prob,
    epsilon: float | None = None,
    inner_iters_list: list[int] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: fixed inner-iteration budget sweep."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    if inner_iters_list is None:
        inner_iters_list = [1, 5, 10, 20, 50, 100]
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"

    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    rows: list[BenchmarkResult] = []
    for n_inner in inner_iters_list:
        kwargs = {
            "epsilon": epsilon,
            "M_saddle": "npe",
            "z0": prob.z0,
            "fixed_inner_iters": n_inner,
        }

        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        rows.append(_build_result(
            solver="aipe_npe_fixed_inner",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
            extra={"m_lazy": n_inner},  # Pack inner loop budget into m_lazy slot safely
        ))

    return rows


def ablation_init_comparison(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Ablation: compare initialization strategies."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"

    problem = prob.problem
    d = problem.dim_x + problem.dim_y
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

    key = jax.random.PRNGKey(seed)
    raw_random = jax.random.normal(key, (d,)) * 0.5
    raw_zero = jnp.zeros(d)

    proj_x = problem.project_x
    proj_y = problem.project_y
    dx = problem.dim_x

    random_z0 = jnp.concatenate([proj_x(raw_random[:dx]), proj_y(raw_random[dx:])])
    zero_z0 = jnp.concatenate([proj_x(raw_zero[:dx]), proj_y(raw_zero[dx:])])
    heuristic_z0 = prob.z0
    init_configs = [
        ("heuristic", heuristic_z0),
        ("zero", zero_z0),
        ("random", random_z0),
    ]

    results: list[BenchmarkResult] = []
    for init_name, z0 in init_configs:
        for M_saddle in ("npe", "len"):
            kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": z0}
            if M_saddle == "len":
                kwargs["m_lazy"] = 5

            times, result = _time_solve_loop(problem, n_repeats, kwargs)
            results.append(_build_result(
                solver=f"aipe_{M_saddle}",
                problem=f"{name}_{init_name}",
                dim=dim,
                epsilon=epsilon,
                times=times,
                result=result,
                d=d,
            ))

    return results


# ──────────────────────────────────────────────────────────────────────
# Resilient Formatters
# ──────────────────────────────────────────────────────────────────────


def format_ablation_m_table(rows: list[BenchmarkResult]) -> str:
    """Format m_lazy ablation as a text table."""
    header = f"{'Problem':<18} {'Dim':>4}  {'m_lazy':>6}  {'Time (s)':>24}  {'Calls':>6}  {'Gap':>10}"
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        # FIX: Cross-query both metric names to ensure accurate counters regardless of solver mode
        calls = getattr(r.oracle_stats, "crn_calls", 0) or getattr(r.oracle_stats, "oracle_calls", 0)
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {r.m_lazy:>6}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  {calls:>6}  {r.final_gap:>10.6f}"
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
        # FIX: Robust fallback for common random number/oracle call name splitting
        calls = getattr(r.oracle_stats, "crn_calls", 0) or getattr(r.oracle_stats, "oracle_calls", 0)
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {tf:>8.1f}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  {calls:>6}  {r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_no_cubic_table(rows: list[BenchmarkResult]) -> str:
    """Compare with/without cubic regularization."""
    if not rows:
        return "(no data)"

    header = (
        f"{'Solver':<22} {'Problem':<18} {'Dim':>4}  "
        f"{'Time (s)':>24}  {'Calls':>6}  {'Iters':>6}  {'Gap':>10}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        calls = getattr(r.oracle_stats, "oracle_calls", 0) or getattr(r.oracle_stats, "crn_calls", 0)
        lines.append(
            f"{r.solver:<22} {r.problem:<18} {r.dim:>4}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  "
            f"{calls:>6}  {r.iterations:>6}  "
            f"{r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_init_table(rows: list[BenchmarkResult]) -> str:
    """Compare initialization strategies."""
    if not rows:
        return "(no data)"

    header = (
        f"{'Variant':<30} {'Solver':<22} {'Dim':>4}  "
        f"{'Time (s)':>24}  {'Calls':>6}  {'Iters':>6}  {'Gap':>10}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        calls = getattr(r.oracle_stats, "oracle_calls", 0) or getattr(r.oracle_stats, "crn_calls", 0)
        lines.append(
            f"{r.problem:<30} {r.solver:<22} {r.dim:>4}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  "
            f"{calls:>6}  {r.iterations:>6}  "
            f"{r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_fixed_inner_table(rows: list[BenchmarkResult]) -> str:
    """Fixed inner-iteration sweep table."""
    if not rows:
        return "(no data)"

    header = (
        f"{'Problem':<18} {'Dim':>4}  {'Inner':>6}  "
        f"{'Time (s)':>24}  {'Calls':>6}  {'Outer':>6}  {'Gap':>10}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        
        # FIX: Correct column logic mapping. 
        # "Inner" pulls the budget allocated to the 'm_lazy' property via the generator.
        # "Outer" pulls the actual outer execution loop step counter.
        inner_val = getattr(r, "m_lazy", 0)
        outer_val = getattr(r, "iterations", 0)
        calls = getattr(r.oracle_stats, "oracle_calls", 0) or getattr(r.oracle_stats, "crn_calls", 0)
        
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {inner_val:>6}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  "
            f"{calls:>6}  {outer_val:>6}  "
            f"{r.final_gap:>10.6f}"
        )
    return "\n".join(lines)