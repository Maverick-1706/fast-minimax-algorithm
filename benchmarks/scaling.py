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
from benchmarks.results import BenchmarkResult
from benchmarks.oracles import count_eg_oracles
from benchmarks.stats import bootstrap_ci
from minimax_aipe.problem import BenchmarkProblem


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
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    solve_kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": prob.z0}
    if M_saddle == "len":
        solve_kwargs["m_lazy"] = 5

    # Warmup and JIT compilation
    w_res = None
    for _ in range(config.N_WARMUP):
        w_res = solve(problem, **solve_kwargs)
        if hasattr(w_res, "x"): w_res.x.block_until_ready()
        if hasattr(w_res, "y"): w_res.y.block_until_ready()
        if hasattr(w_res, "gap") and hasattr(w_res.gap, "block_until_ready"):
            w_res.gap.block_until_ready()

    times = []
    last_result = w_res  # Use warmup as fallback baseline
    for _ in range(n_repeats):
        gc.collect()
        # Synchronize device queue before starting timer
        _ = jax.numpy.zeros(1).block_until_ready()
        t0 = time.perf_counter()
        last_result = solve(problem, **solve_kwargs)
        if hasattr(last_result, "x"): last_result.x.block_until_ready()
        if hasattr(last_result, "y"): last_result.y.block_until_ready()
        if hasattr(last_result, "gap") and hasattr(last_result.gap, "block_until_ready"):
            last_result.gap.block_until_ready()
            
        times.append(time.perf_counter() - t0)

    if not times:
        times = [0.0]

    ci = bootstrap_ci(times)
    d = problem.dim_x + problem.dim_y
    
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
    for i, dim in enumerate(dims):
        prob_seed = (seed + i) if seed is not None else None
        prob = get_problem(problem_type, dim, seed=prob_seed)
        problem = prob.problem

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats))

        # (a) Warmup JIT first
        w_eg = None
        for _ in range(config.N_WARMUP):
            w_eg = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(w_eg, "x") and w_eg.x is not None: w_eg.x.block_until_ready()
            if hasattr(w_eg, "y") and w_eg.y is not None: w_eg.y.block_until_ready()

        # (b) Repeated timed runs
        if n_repeats is None:
            n_repeats = config.N_REPEATS_SCALING
        eg_times = []
        eg_result = w_eg  # Fix: Initialize with warmup to prevent NoneType bugs
        for _ in range(n_repeats):
            gc.collect()
            eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(eg_result, "x") and eg_result.x is not None: eg_result.x.block_until_ready()
            if hasattr(eg_result, "y") and eg_result.y is not None: eg_result.y.block_until_ready()
            eg_times.append(eg_result.wall_time)

        if not eg_times:
            eg_times = [0.0]

        eg_ci = bootstrap_ci(eg_times)
        d = dim * 2
        eg_stats = count_eg_oracles(eg_result.iterations)
        rows.append(BenchmarkResult(
            solver="eg",
            problem=problem_type,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(eg_times) if eg_times != [0.0] else 0.0,
            wall_time_std=statistics.stdev(eg_times) if len(eg_times) > 1 else 0.0,
            ci=eg_ci,
            oracle_stats=eg_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=eg_result.gap,
            iterations=eg_result.iterations,
        ))

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
    for i, kappa in enumerate(kappas):
        prob_seed = (seed + i) if seed is not None else None
        prob = get_problem(problem_type, dim, seed=prob_seed, kappa=kappa)
        problem = prob.problem

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats, condition_number=kappa))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats, condition_number=kappa))

        w_eg = None
        for _ in range(config.N_WARMUP):
            w_eg = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(w_eg, "x") and w_eg.x is not None: w_eg.x.block_until_ready()
            if hasattr(w_eg, "y") and w_eg.y is not None: w_eg.y.block_until_ready()

        if n_repeats is None:
            n_repeats = config.N_REPEATS_SCALING
        eg_times = []
        eg_result = w_eg  # Fix: Initialize with warmup to prevent NoneType bugs
        for _ in range(n_repeats):
            gc.collect()
            eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(eg_result, "x") and eg_result.x is not None: eg_result.x.block_until_ready()
            if hasattr(eg_result, "y") and eg_result.y is not None: eg_result.y.block_until_ready()
            eg_times.append(eg_result.wall_time)

        if not eg_times:
            eg_times = [0.0]

        eg_ci = bootstrap_ci(eg_times)
        d = dim * 2
        eg_stats = count_eg_oracles(eg_result.iterations)
        
        eg_res = BenchmarkResult(
            solver="eg",
            problem=problem_type,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(eg_times) if eg_times != [0.0] else 0.0,
            wall_time_std=statistics.stdev(eg_times) if len(eg_times) > 1 else 0.0,
            ci=eg_ci,
            oracle_stats=eg_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=eg_result.gap,
            iterations=eg_result.iterations,
            condition_number=kappa,
        )
        rows.append(eg_res)

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
    for i, rho in enumerate(rho_values):
        prob_seed = (seed + i) if seed is not None else None
        prob = get_problem("nonzero_rho", dim, seed=prob_seed, rho=rho)
        problem = prob.problem

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats, rho=2 * rho))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats, rho=2 * rho))

        w_eg = None
        for _ in range(config.N_WARMUP):
            w_eg = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(w_eg, "x") and w_eg.x is not None: w_eg.x.block_until_ready()
            if hasattr(w_eg, "y") and w_eg.y is not None: w_eg.y.block_until_ready()

        if n_repeats is None:
            n_repeats = config.N_REPEATS_SCALING
        eg_times = []
        eg_result = w_eg  # Fix: Initialize with warmup to prevent NoneType bugs
        for _ in range(n_repeats):
            gc.collect()
            eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(eg_result, "x") and eg_result.x is not None: eg_result.x.block_until_ready()
            if hasattr(eg_result, "y") and eg_result.y is not None: eg_result.y.block_until_ready()
            eg_times.append(eg_result.wall_time)

        if not eg_times:
            eg_times = [0.0]

        eg_ci = bootstrap_ci(eg_times)
        d = dim * 2
        eg_stats = count_eg_oracles(eg_result.iterations)
        
        eg_res = BenchmarkResult(
            solver="eg",
            problem="nonzero_rho",
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(eg_times) if eg_times != [0.0] else 0.0,
            wall_time_std=statistics.stdev(eg_times) if len(eg_times) > 1 else 0.0,
            ci=eg_ci,
            oracle_stats=eg_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=eg_result.gap,
            iterations=eg_result.iterations,
            rho=2 * rho,
        )
        rows.append(eg_res)

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
    for i, sparsity in enumerate(sparsity_values):
        prob_seed = (seed + i) if seed is not None else None
        prob = get_problem("diagonal_saddle", dim, seed=prob_seed,
                           kappa=kappa, sparsity=sparsity)
        problem = prob.problem

        rows.append(_time_solve(prob, epsilon, "npe", n_repeats, sparsity=sparsity))
        rows.append(_time_solve(prob, epsilon, "len", n_repeats, sparsity=sparsity))

        w_eg = None
        for _ in range(config.N_WARMUP):
            w_eg = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(w_eg, "x") and w_eg.x is not None: w_eg.x.block_until_ready()
            if hasattr(w_eg, "y") and w_eg.y is not None: w_eg.y.block_until_ready()

        if n_repeats is None:
            n_repeats = config.N_REPEATS_SCALING
        eg_times = []
        eg_result = w_eg  # Fix: Initialize with warmup to prevent NoneType bugs
        for _ in range(n_repeats):
            gc.collect()
            eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob.z0)
            if hasattr(eg_result, "x") and eg_result.x is not None: eg_result.x.block_until_ready()
            if hasattr(eg_result, "y") and eg_result.y is not None: eg_result.y.block_until_ready()
            eg_times.append(eg_result.wall_time)

        if not eg_times:
            eg_times = [0.0]

        eg_ci = bootstrap_ci(eg_times)
        d = dim * 2
        eg_stats = count_eg_oracles(eg_result.iterations)
        
        eg_res = BenchmarkResult(
            solver="eg",
            problem="diagonal_saddle",
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(eg_times) if eg_times != [0.0] else 0.0,
            wall_time_std=statistics.stdev(eg_times) if len(eg_times) > 1 else 0.0,
            ci=eg_ci,
            oracle_stats=eg_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=eg_result.gap,
            iterations=eg_result.iterations,
            sparsity=sparsity,
        )
        rows.append(eg_res)

    return rows


def format_scaling_table(rows: list[BenchmarkResult], key_col: str = "dim") -> str:
    """Format scaling results as a text table."""
    lines = []
    header = f"{key_col:>8}  {'NPE (s)':>10}  {'NPE orc':>10}  {'LEN (s)':>10}  {'LEN orc':>10}  {'JIT-EG (s)':>10}  {'EG orc':>10}"
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
        npe_cost = npe.oracle_stats.oracle_calls if npe else 0.0
        len_time = lnn.wall_time_mean if lnn else 0.0
        len_cost = lnn.oracle_stats.oracle_calls if lnn else 0.0
        eg_time = eg.wall_time_mean if eg else 0.0
        eg_cost = eg.oracle_stats.oracle_calls if eg else 0.0

        lines.append(
            f"{kv:>8.6g}  {npe_time:>10.4f}  {npe_cost:>10.2e}  "
            f"{len_time:>10.4f}  {len_cost:>10.2e}  "
            f"{eg_time:>10.4f}  {eg_cost:>10.2e}"
        )

    return "\n".join(lines)