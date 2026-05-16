"""Scaling analysis: dimension, condition number, and ρ.

Measures how solver performance scales with:
  - Problem dimension (diagonal_saddle is ideal — O(n) per oracle call)
  - Condition number (ill_conditioned_* problems)
  - Hessian Lipschitz constant ρ (nonzero_rho problem)
"""

from __future__ import annotations

import gc
import itertools
import statistics
import time

import jax

from minimax_aipe import solve
from benchmarks.baselines import run_eg_jit_benchmark
from benchmarks.results import BenchmarkResult
from benchmarks.oracles import count_eg_oracles
from benchmarks.stats import bootstrap_ci
from minimax_aipe.problem import BenchmarkProblem


def _time_solve(prob: BenchmarkProblem, epsilon: float, M_saddle: str, n_repeats: int) -> BenchmarkResult:
    """Time a single solve configuration and return a BenchmarkResult.

    Parameters
    ----------
    prob : BenchmarkProblem
        The problem to solve.
    epsilon : float
        Target duality gap.
    M_saddle : str
        Saddle point solver ("npe" or "len").
    n_repeats : int
        Number of timed repetitions.

    Returns
    -------
    BenchmarkResult
    """
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": prob.z0}
    if M_saddle == "len":
        kwargs["m_lazy"] = 5

    # Warmup and JIT compilation
    _ = solve(problem, **kwargs)
    times = []
    result = None
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = solve(problem, **kwargs)
        times.append(time.perf_counter() - t0)

    ci = bootstrap_ci(times)
    d = problem.dim_x + problem.dim_y
    return BenchmarkResult(
        solver=f"aipe_{M_saddle}",
        problem=prob.name or "?",
        dim=prob.dim or problem.dim_x,
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
    )


def scale_dimension(
    problem_type: str,
    dims: list[int],
    epsilon: float = 0.05,
    n_repeats: int = 3,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time vs dimension for a given problem type.

    Parameters
    ----------
    problem_type : str
        Problem name from the registry.
    dims : list[int]
        Dimensions to test.
    epsilon : float
        Target gap.
    n_repeats : int
        Timed runs per configuration.
    seed : int or None
        Seed for problem constructors.

    Returns
    -------
    list[BenchmarkResult]
        Three results per dim (npe, len, eg).
    """
    from benchmarks.problems import get_problem

    rows = []
    for i, dim in enumerate(dims):
        prob_seed = (seed + i) if seed is not None else None
        prob = get_problem(problem_type, dim, seed=prob_seed)
        problem = prob.problem

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats))

        eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
        d = dim * 2
        eg_stats = count_eg_oracles(eg_result.iterations)
        rows.append(BenchmarkResult(
            solver="eg",
            problem=problem_type,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=eg_result.wall_time,
            wall_time_std=0.0,
            ci=(eg_result.wall_time, eg_result.wall_time),
            oracle_stats=eg_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=eg_result.gap,
            iterations=eg_result.iterations,
            normalized_cost=eg_stats.normalized_cost(d),
        ))

    return rows


def scale_condition_number(
    problem_type: str,
    condition_numbers: list[float],
    dim: int = 10,
    epsilon: float = 0.05,
    n_repeats: int = 3,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time vs condition number at fixed dimension.

    Parameters
    ----------
    problem_type : str
        "ill_bilinear" or "ill_quadratic".
    condition_numbers : list[float]
        κ values to test.
    dim : int
        Fixed dimension.
    epsilon : float
        Target gap.
    n_repeats : int
        Timed runs.
    seed : int or None
        Seed for problem constructors.

    Returns
    -------
    list[BenchmarkResult]
    """
    from benchmarks.problems import get_problem

    rows = []
    for i, kappa in enumerate(condition_numbers):
        prob_seed = (seed + i) if seed is not None else None
        prob = get_problem(problem_type, dim, seed=prob_seed, condition_number=kappa)
        problem = prob.problem

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats))

        eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
        d = dim * 2
        eg_stats = count_eg_oracles(eg_result.iterations)
        rows.append(BenchmarkResult(
            solver="eg",
            problem=problem_type,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=eg_result.wall_time,
            wall_time_std=0.0,
            ci=(eg_result.wall_time, eg_result.wall_time),
            oracle_stats=eg_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=eg_result.gap,
            iterations=eg_result.iterations,
            normalized_cost=eg_stats.normalized_cost(d),
            condition_number=kappa,
        ))

    return rows


def scale_rho(
    rho_values: list[float],
    dim: int = 10,
    epsilon: float = 0.05,
    n_repeats: int = 3,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time vs Hessian Lipschitz constant ρ.

    Uses the ``nonzero_rho`` problem constructor.

    Parameters
    ----------
    rho_values : list[float]
        ρ values to test.
    dim : int
        Problem dimension.
    epsilon : float
        Target gap.
    n_repeats : int
        Timed runs per ρ.
    seed : int or None
        Seed for problem constructors.

    Returns
    -------
    list[BenchmarkResult]
        One BenchmarkResult per (solver, ρ).
    """
    from benchmarks.problems import get_problem

    rows = []
    for i, rho in enumerate(rho_values):
        prob_seed = (seed + i) if seed is not None else None
        prob = get_problem("nonzero_rho", dim, seed=prob_seed, rho=rho)
        problem = prob.problem

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats))

        eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
        d = dim * 2
        eg_stats = count_eg_oracles(eg_result.iterations)
        rows.append(BenchmarkResult(
            solver="eg",
            problem="nonzero_rho",
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=eg_result.wall_time,
            wall_time_std=0.0,
            ci=(eg_result.wall_time, eg_result.wall_time),
            oracle_stats=eg_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=eg_result.gap,
            iterations=eg_result.iterations,
            normalized_cost=eg_stats.normalized_cost(d),
            rho=rho,
        ))

    return rows


def format_scaling_table(rows: list[BenchmarkResult], key_col: str = "dim") -> str:
    """Format scaling results as a text table.

    Groups results by (problem, key_col_value) and displays wall time
    and FLOP-normalized cost (``normalized_cost``) per solver for
    apples-to-apples oracle complexity comparison.
    """
    lines = []
    header = f"{key_col:>8}  {'NPE (s)':>10}  {'NPE cost':>10}  {'LEN (s)':>10}  {'LEN cost':>10}  {'JIT-EG (s)':>10}  {'EG cost':>10}"
    lines.append(header)
    lines.append("─" * len(header))

    def _key_val(r: BenchmarkResult) -> float:
        if key_col == "dim":
            return float(r.dim)
        elif key_col == "condition_number":
            return r.condition_number or 0.0
        elif key_col == "rho":
            return r.rho or 0.0
        return 0.0

    key = lambda r: (r.problem, _key_val(r))
    for (prob_name, kv), group in itertools.groupby(
        sorted(rows, key=key), key=key
    ):
        group_list = list(group)
        npe = next((r for r in group_list if r.solver == "aipe_npe"), None)
        lnn = next((r for r in group_list if r.solver == "aipe_len"), None)
        eg = next((r for r in group_list if r.solver == "eg"), None)

        npe_time = npe.wall_time_mean if npe else 0.0
        npe_cost = npe.normalized_cost if npe else 0.0
        len_time = lnn.wall_time_mean if lnn else 0.0
        len_cost = lnn.normalized_cost if lnn else 0.0
        eg_time = eg.wall_time_mean if eg else 0.0
        eg_cost = eg.normalized_cost if eg else 0.0

        lines.append(
            f"{kv:>8.6g}  {npe_time:>10.4f}  {npe_cost:>10.2e}  "
            f"{len_time:>10.4f}  {len_cost:>10.2e}  "
            f"{eg_time:>10.4f}  {eg_cost:>10.2e}"
        )

    return "\n".join(lines)
