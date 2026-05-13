"""Memory benchmarks for the Minimax-AIPE solver.

Measures peak memory usage during solve using:
  - Python tracemalloc for host-side allocations
  - JAX profiler for device memory (when available)
"""

from __future__ import annotations

import gc
import tracemalloc
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from minimax_aipe import solve


@dataclass
class MemoryResult:
    """Memory measurement result."""

    name: str
    dim: int
    peak_bytes: int
    current_bytes: int
    peak_mb: float
    current_mb: float
    device_memory: dict | None


def _get_device_memory() -> dict | None:
    """Try to get JAX device memory stats."""
    try:
        device = jax.local_devices()[0]
        stats = device.memory_stats()
        if stats is not None:
            return {
                "bytes_in_use": stats.get("bytes_in_use", 0),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                "bytes_limit": stats.get("bytes_limit", 0),
            }
    except Exception:
        pass
    return None


def _bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def benchmark_memory(
    problem_dict: dict,
    epsilon: float = 0.01,
    M_saddle: str = "npe",
) -> MemoryResult:
    """Measure peak memory during a single solve() call.

    Parameters
    ----------
    problem_dict : dict
        From the problem zoo.
    epsilon : float
        Target gap.
    M_saddle : str
        "npe" or "len".

    Returns
    -------
    MemoryResult
    """
    problem = problem_dict["problem"]
    name = problem_dict.get("name", "?")
    dim = problem_dict.get("dim", problem.dim_x)

    # Clear JAX caches and Python garbage
    gc.collect()

    # Start Python tracemalloc
    tracemalloc.start()

    # Snapshot before
    snap_before = tracemalloc.take_snapshot()

    # Run solve
    result = solve(problem, epsilon=epsilon, M_saddle=M_saddle)

    # Snapshot after
    snap_after = tracemalloc.take_snapshot()
    peak, current = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Force result materialization
    _ = float(result.gap)

    device_mem = _get_device_memory()

    return MemoryResult(
        name=name,
        dim=dim,
        peak_bytes=peak,
        current_bytes=current,
        peak_mb=_bytes_to_mb(peak),
        current_mb=_bytes_to_mb(current),
        device_memory=device_mem,
    )


def benchmark_memory_scaling(
    problems: list[dict],
    epsilon: float = 0.01,
    M_saddle: str = "npe",
) -> list[MemoryResult]:
    """Measure memory for each problem in the list.

    Parameters
    ----------
    problems : list[dict]
        Problem dicts from the zoo.
    epsilon : float
        Target gap.
    M_saddle : str
        "npe" or "len".

    Returns
    -------
    list[MemoryResult]
    """
    results = []
    for prob_dict in problems:
        gc.collect()
        r = benchmark_memory(prob_dict, epsilon=epsilon, M_saddle=M_saddle)
        results.append(r)
    return results


def format_memory_table(results: list[MemoryResult]) -> str:
    """Format memory results as a text table."""
    header = f"{'Problem':<22} {'Dim':>4}  {'Peak (MB)':>10}  {'Current (MB)':>12}  {'Device Peak (MB)':>16}"
    sep = "─" * len(header)
    lines = [header, sep]

    for r in results:
        dev_peak = ""
        if r.device_memory and r.device_memory.get("peak_bytes_in_use"):
            dev_peak = f"{_bytes_to_mb(r.device_memory['peak_bytes_in_use']):>16.2f}"
        else:
            dev_peak = f"{'N/A':>16}"

        lines.append(
            f"{r.name:<22} {r.dim:>4}  {r.peak_mb:>10.2f}  {r.current_mb:>12.2f}  {dev_peak}"
        )

    return "\n".join(lines)
