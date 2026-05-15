"""Memory benchmarks.

Measures peak memory usage during solve using:
  - Process peak resident set size (RSS) via the resource module.
  - JAX active tensor memory via jax.live_arrays().
"""

from __future__ import annotations

import gc
import sys
import resource
from dataclasses import dataclass

import jax

from minimax_aipe import solve
from minimax_aipe.problem import BenchmarkProblem


@dataclass
class MemoryResult:
    """Memory measurement result."""

    name: str
    dim: int
    solver: str
    peak_bytes: int
    jax_bytes: int


def _get_peak_rss() -> int:
    """Get peak process resident set size in bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


def _get_jax_bytes() -> int:
    """Get total size of Python-visible JAX live arrays in bytes."""
    try:
        return sum(x.nbytes for x in jax.live_arrays())
    except Exception:
        return 0


def _bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def benchmark_memory(
    prob,
    epsilon: float = 0.01,
    M_saddle: str = "npe",
) -> MemoryResult:
    """Measure memory during a single solve() call.

    Parameters
    ----------
    prob : BenchmarkProblem
        The problem to benchmark.
    epsilon : float
        Target gap.
    M_saddle : str
        Solver type ("npe" or "len").

    Returns
    -------
    MemoryResult
    """
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

    gc.collect()

    result = solve(problem, epsilon=epsilon, M_saddle=M_saddle, z0=prob.z0)

    # Force materialization BEFORE measuring memory, fixing the lazy array issue
    result.x.block_until_ready()
    result.y.block_until_ready()
    _ = float(result.gap)

    peak = _get_peak_rss()
    jax_bytes = _get_jax_bytes()

    return MemoryResult(
        name=name,
        dim=dim,
        solver=M_saddle,
        peak_bytes=peak,
        jax_bytes=jax_bytes,
    )


def benchmark_memory_scaling(
    problems: list,
    epsilon: float = 0.01,
    M_saddle: str = "npe",
) -> list[MemoryResult]:
    """Measure memory for each problem in the list."""
    results = []
    for prob in problems:
        gc.collect()
        r = benchmark_memory(prob, epsilon=epsilon, M_saddle=M_saddle)
        results.append(r)
    return results


def format_memory_table(results: list[MemoryResult]) -> str:
    """Format memory results as a text table."""
    header = f"{'Problem':<22} {'Dim':>4}  {'Solver':>6}  {'Process Peak (MB)':>18}  {'JAX Live (MB)':>14}"
    sep = "─" * len(header)
    lines = [header, sep]

    for r in results:
        lines.append(
            f"{r.name:<22} {r.dim:>4}  {r.solver:>6}  {_bytes_to_mb(r.peak_bytes):>18.2f}  {_bytes_to_mb(r.jax_bytes):>14.2f}"
        )

    return "\n".join(lines)
