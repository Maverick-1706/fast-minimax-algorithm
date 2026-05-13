"""Speed benchmarks for the Minimax-AIPE solver.

Measures wall-clock time for:
  - Minimax-AIPE (NPE and LEN variants)
  - Baseline solvers (extragradient, GDA)
  - JIT vs eager mode comparison
  - Scaling across problem dimensions
"""

from __future__ import annotations

import gc
import os
import statistics
import time

import jax
import jax.numpy as jnp

from minimax_aipe import solve
from benchmarks.baselines import run_extragradient, run_gda


# ── Helpers ──────────────────────────────────────────────────────────────


def _time_callable(fn, n_warmup: int = 1, n_repeats: int = 3) -> dict:
    """Time a callable with warmup runs.

    Parameters
    ----------
    fn : callable
        No-argument function to time.
    n_warmup : int
        Warmup runs (not timed).  For JIT compilation.
    n_repeats : int
        Timed runs.

    Returns
    -------
    dict with {mean, std, min, max, raw_times}.
    """
    for _ in range(n_warmup):
        fn()
        jax.clear_caches() if hasattr(jax, 'clear_caches') else None

    times = []
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "raw": times,
    }


# ── JIT vs Eager ─────────────────────────────────────────────────────────


def benchmark_jit_vs_eager(
    problem_dict: dict,
    epsilon: float = 0.01,
    n_warmup: int = 1,
    n_repeats: int = 3,
    M_saddle: str = "npe",
) -> dict:
    """Compare solve() with JAX JIT enabled vs disabled.

    Parameters
    ----------
    problem_dict : dict
        From the problem zoo (must contain 'problem' key).
    epsilon : float
        Target duality gap.
    n_warmup : int
        Warmup runs.
    n_repeats : int
        Timed runs.
    M_saddle : str
        Inner solver: "npe" or "len".

    Returns
    -------
    dict with {jit_times, eager_times, speedup}.
    """
    problem = problem_dict["problem"]

    def run_solve():
        return solve(problem, epsilon=epsilon, M_saddle=M_saddle)

    # JIT mode (default)
    jit_times = _time_callable(run_solve, n_warmup=n_warmup, n_repeats=n_repeats)

    # Eager mode
    jax.config.update("jax_disable_jit", True)
    try:
        eager_times = _time_callable(run_solve, n_warmup=0, n_repeats=n_repeats)
    finally:
        jax.config.update("jax_disable_jit", False)

    speedup = eager_times["mean"] / max(jit_times["mean"], 1e-12)

    return {
        "jit": jit_times,
        "eager": eager_times,
        "speedup": speedup,
    }


# ── Solver comparison ────────────────────────────────────────────────────


def benchmark_solver_comparison(
    problems: list[dict],
    epsilon: float = 0.01,
    n_repeats: int = 3,
) -> list[dict]:
    """Time Minimax-AIPE (NPE + LEN) vs EG vs GDA on each problem.

    Parameters
    ----------
    problems : list[dict]
        Problem dicts from the zoo.
    epsilon : float
        Target gap.
    n_repeats : int
        Timed runs per solver.

    Returns
    -------
    list[dict]
        Each dict: {name, dim, aipe_npe, aipe_len, eg, gda, ...}.
    """
    rows = []

    for prob_dict in problems:
        problem = prob_dict["problem"]
        name = prob_dict.get("name", "?")
        dim = prob_dict.get("dim", problem.dim_x)

        print(f"  Benchmarking {name} dim={dim} ...")
        row = {"name": name, "dim": dim}

        # ── AIPE-NPE ───────────────────────────────────────────────
        def run_npe():
            return solve(problem, epsilon=epsilon, M_saddle="npe")

        row["aipe_npe"] = _time_callable(run_npe, n_warmup=1, n_repeats=n_repeats)
        result_npe = run_npe()
        row["aipe_npe"]["gap"] = result_npe.gap
        row["aipe_npe"]["iterations"] = result_npe.iterations
        row["aipe_npe"]["oracle_calls"] = result_npe.oracle_calls
        row["aipe_npe"]["converged"] = result_npe.converged

        # ── AIPE-LEN ───────────────────────────────────────────────
        def run_len():
            return solve(problem, epsilon=epsilon, M_saddle="len", m_lazy=5)

        row["aipe_len"] = _time_callable(run_len, n_warmup=1, n_repeats=n_repeats)
        result_len = run_len()
        row["aipe_len"]["gap"] = result_len.gap
        row["aipe_len"]["iterations"] = result_len.iterations
        row["aipe_len"]["oracle_calls"] = result_len.oracle_calls
        row["aipe_len"]["converged"] = result_len.converged

        # ── Extragradient ──────────────────────────────────────────
        eg_times = []
        eg_result = None
        for _ in range(n_repeats):
            r = run_extragradient(problem, epsilon=epsilon)
            eg_times.append(r.wall_time)
            eg_result = r
        row["eg"] = {
            "mean": statistics.mean(eg_times),
            "std": statistics.stdev(eg_times) if len(eg_times) > 1 else 0.0,
            "min": min(eg_times),
            "max": max(eg_times),
            "raw": eg_times,
            "gap": eg_result.gap,
            "iterations": eg_result.iterations,
            "converged": eg_result.converged,
        }

        # ── GDA ────────────────────────────────────────────────────
        gda_times = []
        gda_result = None
        for _ in range(n_repeats):
            r = run_gda(problem, epsilon=epsilon)
            gda_times.append(r.wall_time)
            gda_result = r
        row["gda"] = {
            "mean": statistics.mean(gda_times),
            "std": statistics.stdev(gda_times) if len(gda_times) > 1 else 0.0,
            "min": min(gda_times),
            "max": max(gda_times),
            "raw": gda_times,
            "gap": gda_result.gap,
            "iterations": gda_result.iterations,
            "converged": gda_result.converged,
        }

        rows.append(row)

    return rows


# ── Scaling analysis ─────────────────────────────────────────────────────


def benchmark_scaling(
    dims: list[int],
    epsilon: float = 0.05,
    problem_type: str = "bilinear",
    n_repeats: int = 2,
) -> list[dict]:
    """Measure solve time vs dimension for a given problem type.

    Parameters
    ----------
    dims : list[int]
        Dimensions to test.
    epsilon : float
        Target gap.
    problem_type : str
        "bilinear" or "quadratic".
    n_repeats : int
        Timed runs.

    Returns
    -------
    list[dict]
        Each dict: {dim, npe_time, len_time, eg_time, npe_calls, len_calls}.
    """
    from benchmarks.problems import get_problem

    rows = []
    for dim in dims:
        print(f"  Scaling test: {problem_type} dim={dim} ...")
        prob_dict = get_problem(problem_type, dim)
        problem = prob_dict["problem"]

        # AIPE-NPE
        npe_result = solve(problem, epsilon=epsilon, M_saddle="npe")
        npe_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            solve(problem, epsilon=epsilon, M_saddle="npe")
            npe_times.append(time.perf_counter() - t0)

        # AIPE-LEN
        len_result = solve(problem, epsilon=epsilon, M_saddle="len", m_lazy=5)
        len_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            solve(problem, epsilon=epsilon, M_saddle="len", m_lazy=5)
            len_times.append(time.perf_counter() - t0)

        # EG
        eg_result = run_extragradient(problem, epsilon=epsilon)

        rows.append({
            "dim": dim,
            "npe_time": statistics.mean(npe_times),
            "len_time": statistics.mean(len_times),
            "eg_time": eg_result.wall_time,
            "npe_calls": npe_result.oracle_calls,
            "len_calls": len_result.oracle_calls,
            "eg_iters": eg_result.iterations,
            "npe_gap": npe_result.gap,
            "len_gap": len_result.gap,
            "eg_gap": eg_result.gap,
        })

    return rows


# ── Formatting ───────────────────────────────────────────────────────────


def format_timing(t: dict) -> str:
    """Format a timing dict as 'mean±std s'."""
    return f"{t['mean']:.4f}±{t['std']:.4f}"


def format_solver_comparison_table(rows: list[dict]) -> str:
    """Format solver comparison results as a text table."""
    header = f"{'Problem':<22} {'Dim':>4}  {'AIPE-NPE':>14}  {'AIPE-LEN':>14}  {'EG':>14}  {'GDA':>14}  {'NPE gap':>8}  {'EG gap':>8}"
    sep = "─" * len(header)
    lines = [header, sep]

    for r in rows:
        npe = format_timing(r["aipe_npe"])
        lnn = format_timing(r["aipe_len"])
        eg = format_timing(r["eg"])
        gda = format_timing(r["gda"])
        lines.append(
            f"{r['name']:<22} {r['dim']:>4}  {npe:>14}  {lnn:>14}  {eg:>14}  {gda:>14}  "
            f"{r['aipe_npe']['gap']:>8.4f}  {r['eg']['gap']:>8.4f}"
        )

    return "\n".join(lines)


def format_jit_table(rows: list[dict]) -> str:
    """Format JIT vs eager results as a text table."""
    header = f"{'Problem':<22} {'Dim':>4}  {'JIT (s)':>12}  {'Eager (s)':>12}  {'Speedup':>8}"
    sep = "─" * len(header)
    lines = [header, sep]

    for r in rows:
        jit = r["jit"]
        eager = r["eager"]
        lines.append(
            f"{r['name']:<22} {r['dim']:>4}  "
            f"{jit['mean']:>7.4f}±{jit['std']:.3f}  "
            f"{eager['mean']:>7.4f}±{eager['std']:.3f}  "
            f"{r['speedup']:>7.2f}x"
        )

    return "\n".join(lines)


def format_scaling_table(rows: list[dict]) -> str:
    """Format scaling results as a text table."""
    header = f"{'Dim':>4}  {'NPE (s)':>10}  {'LEN (s)':>10}  {'EG (s)':>10}  {'NPE calls':>10}  {'LEN calls':>10}  {'EG iters':>10}"
    sep = "─" * len(header)
    lines = [header, sep]

    for r in rows:
        lines.append(
            f"{r['dim']:>4}  {r['npe_time']:>10.4f}  {r['len_time']:>10.4f}  {r['eg_time']:>10.4f}  "
            f"{r['npe_calls']:>10}  {r['len_calls']:>10}  {r['eg_iters']:>10}"
        )

    return "\n".join(lines)
