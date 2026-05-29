"""Scaling analysis: dimension, condition number, and ρ.

Measures how solver performance scales with:
  - Problem dimension (diagonal_saddle is ideal — O(n) per oracle call)
  - Condition number κ (ill_conditioned_* problems)
  - Hessian Lipschitz constant ρ (nonzero_rho problem)
  - Sparsity (diagonal_saddle with sparsity parameter)
"""

from __future__ import annotations

import gc
import itertools
import statistics
import time

import jax

from minimax_aipe import solve
from benchmarks import config
from benchmarks.baselines import run_eg_jit_benchmark
from benchmarks.reporting import gap_source, normalized_cost, row_normalized_cost, sync_result
from benchmarks.results import BenchmarkResult
from benchmarks.oracles import count_eg_oracles
from benchmarks.stats import bootstrap_ci
from minimax_aipe.problem import BenchmarkProblem


def _effective_repeats(n_repeats: int | None) -> int:
    return n_repeats if n_repeats is not None else config.N_REPEATS_SCALING


def _time_eg_baseline(
    prob: BenchmarkProblem,
    epsilon: float | None,
    n_repeats: int | None,
    **opt_fields,
) -> BenchmarkResult:
    problem = prob.problem
    repeats = _effective_repeats(n_repeats)

    warmup = None
    for _ in range(config.N_WARMUP):
        warmup = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
        sync_result(warmup)

    times = []
    result = warmup
    for _ in range(repeats):
        gc.collect()
        _ = jax.numpy.zeros(1).block_until_ready()
        result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
        sync_result(result)
        times.append(result.wall_time)

    if not times:
        times = [0.0]
    if result is None:
        result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
        sync_result(result)

    stats = count_eg_oracles(result.iterations)
    return BenchmarkResult(
        solver="eg",
        problem=prob.name or "?",
        dim=prob.dim or problem.dim_x,
        epsilon=epsilon,
        wall_time_mean=statistics.mean(times) if times != [0.0] else 0.0,
        wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        ci=bootstrap_ci(times),
        oracle_stats=stats,
        converged=result.converged,
        gap_achieved=result.gap_achieved,
        final_gap=result.gap,
        iterations=result.iterations,
        normalized_cost=normalized_cost(problem, stats),
        gap_source=gap_source(problem),
        **opt_fields,
    )


def _time_solve(
    prob: BenchmarkProblem,
    epsilon: float | None = None,
    M_saddle: str = "npe",
    n_repeats: int | None = None,
    **kwargs,
) -> BenchmarkResult:
    """Time a single solve configuration and return a BenchmarkResult."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    repeats = _effective_repeats(n_repeats)
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    solve_kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": prob.z0}
    if M_saddle == "len":
        solve_kwargs["m_lazy"] = 5

    # Warmup and JIT compilation
    w_res = None
    for _ in range(config.N_WARMUP):
        w_res = solve(problem, **solve_kwargs)
        sync_result(w_res)

    times = []
    last_result = w_res  # Use warmup as fallback baseline
    for _ in range(repeats):
        gc.collect()
        # Synchronize device queue before starting timer
        _ = jax.numpy.zeros(1).block_until_ready()
        t0 = time.perf_counter()
        last_result = solve(problem, **solve_kwargs)
        sync_result(last_result)
        times.append(time.perf_counter() - t0)

    if not times:
        times = [0.0]

    ci = bootstrap_ci(times)
    opt_fields = {k: v for k, v in kwargs.items() if k in (
        "m_lazy", "npe_T_factor", "condition_number", "rho", "sparsity"
    )}

    res = BenchmarkResult(
        solver=f"aipe_{M_saddle}",
        problem=prob.name or "?",
        dim=prob.dim or problem.dim_x,
        epsilon=epsilon,
        wall_time_mean=statistics.mean(times) if times != [0.0] else 0.0,
        wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        ci=ci,
        oracle_stats=last_result.oracle_stats,
        converged=last_result.converged,
        gap_achieved=last_result.gap <= epsilon,
        final_gap=float(last_result.gap),
        iterations=last_result.iterations,
        normalized_cost=normalized_cost(problem, last_result.oracle_stats),
        gap_source=gap_source(problem),
        **opt_fields,
    )
    
    return res


def scale_dimension(
    problem_type: str,
    dims: list[int],
    epsilon: float | None = None,
    n_repeats: int | None = None,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time vs dimension for a given problem type."""
    from benchmarks.problems import get_problem

    rows = []
    for dim in dims:
        prob = get_problem(problem_type, dim, seed=seed)

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats))
        rows.append(_time_eg_baseline(prob, epsilon, n_repeats))

    return rows


def scale_condition_number(
    problem_type: str,
    kappas: list[float] | None = None,
    *,
    condition_numbers: list[float] | None = None,
    dim: int = 10,
    epsilon: float | None = None,
    n_repeats: int | None = None,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time vs condition number at fixed dimension."""
    if condition_numbers is not None:
        import warnings
        warnings.warn(
            "condition_numbers is deprecated, use kappas instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if kappas is None:
            kappas = condition_numbers
    if kappas is None:
        raise TypeError("Must provide kappas (or deprecated condition_numbers)")

    from benchmarks.problems import get_problem

    rows = []
    for kappa in kappas:
        prob = get_problem(problem_type, dim, seed=seed, kappa=kappa)

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats, condition_number=kappa))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats, condition_number=kappa))
        rows.append(
            _time_eg_baseline(
                prob,
                epsilon,
                n_repeats,
                condition_number=kappa,
            )
        )

    return rows


def scale_rho(
    rho_values: list[float],
    dim: int = 10,
    epsilon: float | None = None,
    n_repeats: int | None = None,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time vs Hessian Lipschitz constant ρ."""
    from benchmarks.problems import get_problem

    rows = []
    for rho in rho_values:
        prob = get_problem("nonzero_rho", dim, seed=seed, rho=rho)

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats, rho=2 * rho))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats, rho=2 * rho))
        rows.append(_time_eg_baseline(prob, epsilon, n_repeats, rho=2 * rho))

    return rows


def scale_sparsity(
    sparsity_values: list[float],
    dim: int = 100,
    kappa: float = 1e4,
    epsilon: float | None = None,
    n_repeats: int | None = None,
    seed: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time vs coupling sparsity at fixed dimension and κ."""
    from benchmarks.problems import get_problem

    rows = []
    for sparsity in sparsity_values:
        prob = get_problem("diagonal_saddle", dim, seed=seed, kappa=kappa, sparsity=sparsity)

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats, sparsity=sparsity))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats, sparsity=sparsity))
        rows.append(_time_eg_baseline(prob, epsilon, n_repeats, sparsity=sparsity))

    return rows


def format_scaling_table(rows: list[BenchmarkResult], key_col: str = "dim") -> str:
    """Format scaling results as a text table.

    Cost columns show normalized_cost(d) (gradient-equivalent FLOP units)
    for fair comparison across CRN and gradient-based solvers.
    """
    lines = []
    header = f"{key_col:>8}  {'NPE (s)':>10}  {'NPE cost':>11}  {'LEN (s)':>10}  {'LEN cost':>11}  {'JIT-EG (s)':>10}  {'EG cost':>11}"
    lines.append(header)
    lines.append("─" * len(header))

    def _key_val(r: BenchmarkResult) -> float:
        if key_col == "dim":
            return float(r.dim)
        elif key_col in ("condition_number", "kappa"):
            return getattr(r, "condition_number", None) or 0.0
        elif key_col == "rho":
            return getattr(r, "rho", None) or 0.0
        elif key_col == "sparsity":
            return getattr(r, "sparsity", None) or 0.0
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
        npe_cost = row_normalized_cost(npe) if npe else 0.0
        len_time = lnn.wall_time_mean if lnn else 0.0
        len_cost = row_normalized_cost(lnn) if lnn else 0.0
        eg_time = eg.wall_time_mean if eg else 0.0
        eg_cost = row_normalized_cost(eg) if eg else 0.0

        lines.append(
            f"{kv:>8.6g}  {npe_time:>10.4f}  {float(npe_cost or 0.0):>11.2e}  "
            f"{len_time:>10.4f}  {float(len_cost or 0.0):>11.2e}  "
            f"{eg_time:>10.4f}  {float(eg_cost or 0.0):>11.2e}"
        )

    return "\n".join(lines)
