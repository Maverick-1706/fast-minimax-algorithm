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
from benchmarks.stats import bootstrap_ci


def ablation_m_lazy(
    prob,
    epsilon: float = 0.01,
    m_values: list[int] | None = None,
    n_repeats: int = 3,
) -> list[dict]:
    """Measure solve time and oracle calls as m_lazy varies.

    m=1 is equivalent to fresh Hessians (NPE).  Larger m reuses more.

    Parameters
    ----------
    prob : BenchmarkProblem
        From the problem zoo.
    epsilon : float
        Target gap.
    m_values : list[int]
        Hessian reuse intervals to test.
    n_repeats : int
        Timed runs per configuration.

    Returns
    -------
    list[dict]
    """
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    m_values = m_values or [1, 3, 5, 10, 20]
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

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
        rows.append({
            "name": name,
            "dim": dim,
            "m_lazy": m,
            "time_mean": statistics.mean(times),
            "time_ci_lo": ci[0],
            "time_ci_hi": ci[1],
            "oracle_calls": result.oracle_calls,
            "gap": float(result.gap),
            "converged": result.converged,
        })

    return rows


def ablation_npe_t_factor(
    prob,
    epsilon: float = 0.01,
    t_factors: list[float] | None = None,
    n_repeats: int = 3,
) -> list[dict]:
    """Measure solve time and oracle calls as npe_T_factor varies.

    """
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    t_factors = t_factors or [0.5, 1.0, 1.5, 2.0, 3.0]
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

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
        rows.append({
            "name": name,
            "dim": dim,
            "npe_T_factor": tf,
            "time_mean": statistics.mean(times),
            "time_ci_lo": ci[0],
            "time_ci_hi": ci[1],
            "oracle_calls": result.oracle_calls,
            "gap": float(result.gap),
            "converged": result.converged,
        })

    return rows


def ablation_npe_vs_len(
    prob,
    epsilon: float = 0.01,
    n_repeats: int = 3,
) -> dict:
    """Head-to-head NPE vs LEN on a single problem.

    """
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

    def _run(M_saddle: str):
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
        return {
            "time_mean": statistics.mean(times),
            "time_ci_lo": ci[0],
            "time_ci_hi": ci[1],
            "oracle_calls": result.oracle_calls,
            "gap": float(result.gap),
            "iterations": result.iterations,
            "converged": result.converged,
        }

    return {
        "name": name,
        "dim": dim,
        "npe": _run("npe"),
        "len": _run("len"),
    }


def format_ablation_m_table(rows: list[dict]) -> str:
    """Format m_lazy ablation as a text table."""
    header = f"{'Problem':<18} {'Dim':>4}  {'m_lazy':>6}  {'Time (s)':>24}  {'Calls':>6}  {'Gap':>10}"
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r['time_ci_lo']:.4f},{r['time_ci_hi']:.4f}]"
        lines.append(
            f"{r['name']:<18} {r['dim']:>4}  {r['m_lazy']:>6}  "
            f"{r['time_mean']:>8.4f} {ci:>16}  {r['oracle_calls']:>6}  {r['gap']:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_t_table(rows: list[dict]) -> str:
    """Format npe_T_factor ablation as a text table."""
    header = f"{'Problem':<18} {'Dim':>4}  {'T_factor':>8}  {'Time (s)':>24}  {'Calls':>6}  {'Gap':>10}"
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r['time_ci_lo']:.4f},{r['time_ci_hi']:.4f}]"
        lines.append(
            f"{r['name']:<18} {r['dim']:>4}  {r['npe_T_factor']:>8.1f}  "
            f"{r['time_mean']:>8.4f} {ci:>16}  {r['oracle_calls']:>6}  {r['gap']:>10.6f}"
        )
    return "\n".join(lines)
