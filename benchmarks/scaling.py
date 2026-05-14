"""Scaling analysis: dimension, condition number, and ρ.

Measures how solver performance scales with:
  - Problem dimension (diagonal_saddle is ideal — O(n) per oracle call)
  - Condition number (ill_conditioned_* problems)
  - Hessian Lipschitz constant ρ (nonzero_rho problem)
"""

from __future__ import annotations

import gc
import statistics
import time

import jax

from minimax_aipe import solve
from benchmarks.baselines import run_eg_jit_benchmark
from benchmarks.stats import bootstrap_ci


def _time_solve(prob_dict, epsilon: float, M_saddle: str, n_repeats: int) -> dict:
    """Time a single solve configuration and return stats."""
    problem = prob_dict["problem"]
    kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": prob_dict["z0"]}
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
    return {
        "time_mean": statistics.mean(times),
        "time_ci": ci,
        "oracle_calls": result.oracle_calls,
        "gap": float(result.gap),
        "iterations": result.iterations,
        "converged": result.converged,
    }


def scale_dimension(
    problem_type: str,
    dims: list[int],
    epsilon: float = 0.05,
    n_repeats: int = 3,
    seed: int | None = None,
) -> list[dict]:
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
    list[dict]
        One dict per (dim) with timing, oracle calls, gap.
    """
    from benchmarks.problems import get_problem

    rows = []
    for i, dim in enumerate(dims):
        prob_seed = (seed + i) if seed is not None else None
        prob_dict = get_problem(problem_type, dim, seed=prob_seed)
        problem = prob_dict["problem"]

        npe = _time_solve(prob_dict, epsilon, "npe", n_repeats)
        lnn = _time_solve(prob_dict, epsilon, "len", n_repeats)
        eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob_dict["z0"])

        rows.append({
            "problem": problem_type,
            "dim": dim,
            "npe_time": npe["time_mean"],
            "npe_ci_lo": npe["time_ci"][0],
            "npe_ci_hi": npe["time_ci"][1],
            "npe_calls": npe["oracle_calls"],
            "npe_gap": npe["gap"],
            "len_time": lnn["time_mean"],
            "len_ci_lo": lnn["time_ci"][0],
            "len_ci_hi": lnn["time_ci"][1],
            "len_calls": lnn["oracle_calls"],
            "len_gap": lnn["gap"],
            "eg_time": eg_result.wall_time,
            "eg_iters": eg_result.iterations,
            "eg_gap": eg_result.gap,
        })

    return rows


def scale_condition_number(
    problem_type: str,
    condition_numbers: list[float],
    dim: int = 10,
    epsilon: float = 0.05,
    n_repeats: int = 3,
    seed: int | None = None,
) -> list[dict]:
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
    list[dict]
    """
    from benchmarks.problems import get_problem

    rows = []
    for i, kappa in enumerate(condition_numbers):
        prob_seed = (seed + i) if seed is not None else None
        prob_dict = get_problem(problem_type, dim, seed=prob_seed, condition_number=kappa)
        problem = prob_dict["problem"]

        npe = _time_solve(prob_dict, epsilon, "npe", n_repeats)
        lnn = _time_solve(prob_dict, epsilon, "len", n_repeats)
        eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob_dict["z0"])

        rows.append({
            "problem": problem_type,
            "dim": dim,
            "condition_number": kappa,
            "npe_time": npe["time_mean"],
            "npe_ci_lo": npe["time_ci"][0],
            "npe_ci_hi": npe["time_ci"][1],
            "npe_calls": npe["oracle_calls"],
            "npe_gap": npe["gap"],
            "len_time": lnn["time_mean"],
            "len_ci_lo": lnn["time_ci"][0],
            "len_ci_hi": lnn["time_ci"][1],
            "len_calls": lnn["oracle_calls"],
            "len_gap": lnn["gap"],
            "eg_time": eg_result.wall_time,
            "eg_iters": eg_result.iterations,
            "eg_gap": eg_result.gap,
        })

    return rows


def scale_rho(
    rho_values: list[float],
    dim: int = 10,
    epsilon: float = 0.05,
    n_repeats: int = 3,
    seed: int | None = None,
) -> list[dict]:
    """Measure solve time vs Hessian Lipschitz constant ρ.

    Uses the ``nonzero_rho`` problem constructor.
    """
    from benchmarks.problems import get_problem

    rows = []
    for i, rho in enumerate(rho_values):
        prob_seed = (seed + i) if seed is not None else None
        prob_dict = get_problem("nonzero_rho", dim, seed=prob_seed, rho=rho)
        problem = prob_dict["problem"]

        npe = _time_solve(prob_dict, epsilon, "npe", n_repeats)
        lnn = _time_solve(prob_dict, epsilon, "len", n_repeats)
        eg_result = run_eg_jit_benchmark(problem, epsilon=epsilon, z0=prob_dict["z0"])

        rows.append({
            "problem": "nonzero_rho",
            "dim": dim,
            "rho": rho,
            "npe_time": npe["time_mean"],
            "npe_ci_lo": npe["time_ci"][0],
            "npe_ci_hi": npe["time_ci"][1],
            "npe_calls": npe["oracle_calls"],
            "npe_gap": npe["gap"],
            "len_time": lnn["time_mean"],
            "len_ci_lo": lnn["time_ci"][0],
            "len_ci_hi": lnn["time_ci"][1],
            "len_calls": lnn["oracle_calls"],
            "len_gap": lnn["gap"],
            "eg_time": eg_result.wall_time,
            "eg_iters": eg_result.iterations,
            "eg_gap": eg_result.gap,
        })

    return rows


def format_scaling_table(rows: list[dict], key_col: str = "dim") -> str:
    """Format scaling results as a text table."""
    lines = []
    cols = [key_col, "npe_time", "npe_calls", "len_time", "len_calls", "eg_time", "eg_iters"]
    header = f"{key_col:>8}  {'NPE (s)':>10}  {'NPE calls':>10}  {'LEN (s)':>10}  {'LEN calls':>10}  {'JIT-EG (s)':>10}  {'EG iters':>10}"
    lines.append(header)
    lines.append("─" * len(header))

    for r in rows:
        lines.append(
            f"{r[key_col]:>8}  {r['npe_time']:>10.4f}  {r.get('npe_calls', ''):>10}  "
            f"{r.get('len_time', 0.0):>10.4f}  {r.get('len_calls', ''):>10}  "
            f"{r.get('eg_time', 0.0):>10.4f}  {r.get('eg_iters', ''):>10}"
        )

    return "\n".join(lines)
