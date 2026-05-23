"""Memory benchmarks using JAX device memory profiling.

Uses ``device.memory_stats()`` for accurate GPU/TPU HBM tracking,
including XLA-internal buffers, pre-allocated pools, and compiled-code
caches that ``jax.live_arrays()`` completely misses.

Measurement strategy
--------------------
1. **Warmup** — one untimed ``solve()`` so JAX JIT-compiles everything
   and caches the compiled code.  After warmup, intermediates are freed
   but the JIT cache remains live.
2. **Baseline snapshot** — captures device memory state (``bytes_in_use``)
   with JIT cache in place, so the delta measures *execution* memory
   rather than compilation overhead.
3. **Measured solve** — timed run with ``block_until_ready()`` to force
   all lazy materialisation.
4. **Post snapshot** — delta (post − baseline) = incremental memory of
   the solve.  ``peak_bytes_in_use`` reports the global device peak
   (reset per-problem via ``jax.clear_caches()`` + ``gc.collect()``).

Fallback: on CPU-only backends where ``device.memory_stats()`` is
unavailable, falls back to OS-level process RSS (``resource.getrusage``)
and Python-visible ``jax.live_arrays()``.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

from minimax_aipe.framework import solve
from minimax_aipe.problem import BenchmarkProblem


# ═══════════════════════════════════════════════════════════════════════════
# Low-level measurement primitives
# ═══════════════════════════════════════════════════════════════════════════

def _get_process_rss() -> int:
    """Current peak process RSS in bytes (OS-level, all platforms)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: bytes, Linux: kilobytes
    return usage if sys.platform == "darwin" else usage * 1024


def _get_jax_live_bytes() -> int:
    """Total bytes of Python-visible JAX arrays."""
    try:
        return sum(a.nbytes for a in jax.live_arrays())
    except Exception:
        return 0


def _device_memory_info(device=None) -> dict:
    """Query ``device.memory_stats()``.  Returns ``{}`` if unavailable.

    On GPU/TPU backends this reports actual device HBM usage including
    XLA-internal buffers, pre-allocated pools, and compiled-kernel
    residency — all invisible to ``jax.live_arrays()``.

    Returns keys (GPU/TPU, when available):
        ``bytes_in_use``, ``peak_bytes_in_use``, ``bytes_limit``,
        ``peak_bytes_limit``, ``bytes_reserved``, ``peak_bytes_reserved``

    Returns ``{}`` on CPU-only backends or on error.
    """
    if device is None:
        device = jax.local_devices()[0]
    try:
        stats = device.memory_stats()
        return dict(stats) if stats else {}
    except Exception:
        return {}


def _reset_peak_tracking() -> None:
    """Best-effort reset of device peak memory counter.

    Calls ``gc.collect()`` to release Python-held references, then
    issues a no-op device request so that XLA's allocator can reclaim
    freed blocks.  Not all backends support a true peak-reset; this
    shrinks the live pool so the *next* peak measurement is meaningful.
    """
    gc.collect()
    try:
        # Nudge the runtime to reclaim freed blocks
        _ = jnp.zeros(1).block_until_ready()
    except Exception:
        pass
    gc.collect()


def _bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# Memory snapshot
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _MemorySnapshot:
    """Point-in-time memory state across all backends."""

    process_rss: int
    jax_live_bytes: int
    device_bytes_in_use: int
    device_peak_bytes_in_use: int
    device_bytes_limit: int
    device_bytes_reserved: int
    device_peak_bytes_reserved: int
    has_device_stats: bool

    @classmethod
    def capture(cls) -> _MemorySnapshot:
        """Take a memory snapshot (with preceding ``gc.collect()``)."""
        gc.collect()
        dev = _device_memory_info()
        has_dev = "bytes_in_use" in dev
        return cls(
            process_rss=_get_process_rss(),
            jax_live_bytes=_get_jax_live_bytes(),
            device_bytes_in_use=dev.get("bytes_in_use", 0),
            device_peak_bytes_in_use=dev.get("peak_bytes_in_use", 0),
            device_bytes_limit=dev.get("bytes_limit", 0),
            device_bytes_reserved=dev.get("bytes_reserved", 0),
            device_peak_bytes_reserved=dev.get("peak_bytes_reserved", 0),
            has_device_stats=has_dev,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Result type
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryResult:
    """Memory measurement result for a single solver run.

    Primary fields (``peak_bytes``, ``jax_bytes``) are kept for backward
    compatibility with downstream export / formatting code.  Detailed
    device-level fields are populated when the backend supports
    ``device.memory_stats()`` (GPU / TPU).
    """

    # ── Identity ────────────────────────────────────────────────────
    name: str
    dim: int
    solver: str
    epsilon: float = 0.01

    # ── Primary (backward-compatible) ───────────────────────────────
    peak_bytes: int = 0
    """Best available peak: device peak (GPU/TPU) or RSS delta (CPU)."""
    jax_bytes: int = 0
    """Python-visible JAX live-array bytes after solve."""

    # ── Device memory (GPU / TPU HBM) ──────────────────────────────
    device_bytes_in_use: int = 0
    """Device memory in use after solve (bytes)."""
    device_bytes_peak: int = 0
    """Device peak memory since last reset (bytes)."""
    device_bytes_delta: int = 0
    """Incremental device memory from this solve (bytes)."""
    device_bytes_limit: int = 0
    """Total device memory available (bytes)."""
    device_bytes_reserved: int = 0
    """XLA pre-reserved memory pool (bytes)."""
    device_utilization: float = 0.0
    """Peak / limit ratio (0.0–1.0).  0 if unavailable."""

    # ── Process memory (OS-level, all backends) ─────────────────────
    process_peak_rss: int = 0
    """Peak RSS from OS (bytes)."""
    process_delta_rss: int = 0
    """RSS increase attributable to this solve (bytes)."""

    # ── Metadata ────────────────────────────────────────────────────
    has_device_stats: bool = False
    """True when ``device.memory_stats()`` returned data."""
    solve_time_s: float = 0.0
    """Wall-clock time of the measured solve (seconds)."""

    def to_dict(self) -> dict:
        """Flat JSON-serialisable dict (compatible with export pipeline)."""
        d = asdict(self)
        # Convenience MB columns for downstream analysis
        d["peak_mb"] = _bytes_to_mb(self.peak_bytes)
        d["device_peak_mb"] = _bytes_to_mb(self.device_bytes_peak)
        d["device_delta_mb"] = _bytes_to_mb(self.device_bytes_delta)
        d["device_limit_mb"] = _bytes_to_mb(self.device_bytes_limit)
        d["device_reserved_mb"] = _bytes_to_mb(self.device_bytes_reserved)
        d["process_peak_rss_mb"] = _bytes_to_mb(self.process_peak_rss)
        d["process_delta_rss_mb"] = _bytes_to_mb(self.process_delta_rss)
        d["jax_live_mb"] = _bytes_to_mb(self.jax_bytes)
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Core measurement
# ═══════════════════════════════════════════════════════════════════════════

def _force_materialize(result) -> None:
    """Block until every lazy JAX array in *result* is ready."""
    result.x.block_until_ready()
    result.y.block_until_ready()
    # Consume traced scalars so XLA can't defer the computation
    _ = float(result.gap)


def benchmark_memory(
    prob: BenchmarkProblem,
    epsilon: float = 0.01,
    M_saddle: str = "npe",
    warmup: bool = True,
) -> MemoryResult:
    """Measure device and process memory during a single ``solve()``.

    Measurement steps:

    1. (optional) Warmup — one solve to JIT-compile; intermediates
       are freed but the JIT cache remains.
    2. Reset peak tracking + baseline snapshot.
    3. Timed solve with forced materialisation.
    4. Post-solve snapshot → deltas and peak.

    Methodology Note: "Device Δ" (device_delta) measures *incremental memory
    requested from the OS/Device allocator beyond the JIT-compiled steady-state pool*,
    rather than the absolute HBM footprint of the tensors. Because XLA aggressively
    holds onto memory blocks, this delta can drop to 0 MB if the timed solve
    completely reuses the warmup pool. Thus, `primary_peak` reports the absolute
    `device_peak_bytes_in_use` to avoid artificial 0 MB reports.

    Parameters
    ----------
    prob : BenchmarkProblem
        Problem to measure.
    epsilon : float
        Target duality gap.
    M_saddle : str
        Solver variant (``"npe"`` or ``"len"``).
    warmup : bool
        Pre-compile so the measured run excludes JIT time and measures
        steady-state memory only.

    Returns
    -------
    MemoryResult
    """
    assert isinstance(prob, BenchmarkProblem), (
        f"Expected BenchmarkProblem, got {type(prob)}"
    )
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

    # ── 1. Warmup: JIT-compile without measuring ────────────────────
    if warmup:
        gc.collect()
        _warm = solve(problem, epsilon=epsilon, M_saddle=M_saddle, z0=prob.z0)
        _force_materialize(_warm)
        del _warm
        gc.collect()

    # ── 2. Baseline snapshot (JIT cache stays live) ─────────────────
    _reset_peak_tracking()
    before = _MemorySnapshot.capture()

    # ── 3. Measured solve ───────────────────────────────────────────
    t0 = time.perf_counter()
    result = solve(problem, epsilon=epsilon, M_saddle=M_saddle, z0=prob.z0)
    _force_materialize(result)
    solve_time = time.perf_counter() - t0

    # ── 4. Post-solve snapshot ──────────────────────────────────────
    after = _MemorySnapshot.capture()

    # ── Derive metrics ──────────────────────────────────────────────
    has_dev = after.has_device_stats

    if has_dev:
        device_delta = max(after.device_bytes_in_use - before.device_bytes_in_use, 0)
        device_peak = after.device_peak_bytes_in_use
        device_limit = after.device_bytes_limit
        device_reserved = after.device_bytes_reserved
        utilization = (device_peak / device_limit) if device_limit > 0 else 0.0
        # FIX: Avoid XLA memory pool delta trap where reused warmup pools yield 0 MB peak
        primary_peak = device_peak
    else:
        device_delta = 0
        device_peak = 0
        device_limit = 0
        device_reserved = 0
        utilization = 0.0
        primary_peak = 0

    rss_peak = after.process_rss
    rss_delta = max(after.process_rss - before.process_rss, 0)

    # On CPU, fall back to RSS delta as the primary peak metric
    if not has_dev:
        primary_peak = rss_peak

    return MemoryResult(
        name=name,
        dim=dim,
        solver=M_saddle,
        epsilon=epsilon,
        # Primary (backward-compatible)
        peak_bytes=primary_peak,
        jax_bytes=after.jax_live_bytes,
        # Device detail
        device_bytes_in_use=after.device_bytes_in_use,
        device_bytes_peak=device_peak,
        device_bytes_delta=device_delta,
        device_bytes_limit=device_limit,
        device_bytes_reserved=device_reserved,
        device_utilization=utilization,
        # Process detail
        process_peak_rss=rss_peak,
        process_delta_rss=rss_delta,
        # Meta
        has_device_stats=has_dev,
        solve_time_s=solve_time,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scaling benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_memory_scaling(
    problems: list[BenchmarkProblem],
    epsilon: float = 0.01,
    M_saddle: str = "npe",
) -> list[MemoryResult]:
    """Measure memory for each problem with proper inter-problem isolation.

    Between problems the JIT cache and any stale device allocations are
    cleared so that the next baseline snapshot starts from a clean slate.
    """
    results: list[MemoryResult] = []
    for prob in problems:
        # Full isolation: clear JIT cache + GC before each problem
        jax.clear_caches()
        gc.collect()
        r = benchmark_memory(
            prob, epsilon=epsilon, M_saddle=M_saddle, warmup=True,
        )
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_memory_table(results: list[MemoryResult]) -> str:
    """Format memory results as a text table.

    Adapts columns based on whether device memory stats are available.
    """
    if not results:
        return "(no memory results)"

    has_device = any(r.has_device_stats for r in results)

    if has_device:
        header = (
            f"{'Problem':<22} {'Dim':>4}  {'Solver':>6}  "
            f"{'Dev Peak':>10}  {'Dev Δ':>9}  {'Reserved':>10}  "
            f"{'Util':>6}  {'RSS':>8}  {'Time':>7}"
        )
        units = (
            f"{'':22} {'':>4}  {'':>6}  "
            f"{'(MB)':>10}  {'(MB)':>9}  {'(MB)':>10}  "
            f"{'(%)':>6}  {'(MB)':>8}  {'(s)':>7}"
        )
        sep = "─" * len(header)
        lines = [header, units, sep]

        for r in results:
            if r.has_device_stats:
                lines.append(
                    f"{r.name:<22} {r.dim:>4}  {r.solver:>6}  "
                    f"{_bytes_to_mb(r.device_bytes_peak):>10.2f}  "
                    f"{_bytes_to_mb(r.device_bytes_delta):>9.2f}  "
                    f"{_bytes_to_mb(r.device_bytes_reserved):>10.2f}  "
                    f"{r.device_utilization * 100:>5.1f}%  "
                    f"{_bytes_to_mb(r.process_peak_rss):>8.2f}  "
                    f"{r.solve_time_s:>7.3f}"
                )
            else:
                lines.append(
                    f"{r.name:<22} {r.dim:>4}  {r.solver:>6}  "
                    f"{'N/A':>10}  {'N/A':>9}  {'N/A':>10}  "
                    f"{'N/A':>6}  "
                    f"{_bytes_to_mb(r.process_peak_rss):>8.2f}  "
                    f"{r.solve_time_s:>7.3f}"
                )
    else:
        header = (
            f"{'Problem':<22} {'Dim':>4}  {'Solver':>6}  "
            f"{'Peak RSS':>10}  {'RSS Δ':>9}  "
            f"{'JAX Live':>10}  {'Time':>7}"
        )
        units = (
            f"{'':22} {'':>4}  {'':>6}  "
            f"{'(MB)':>10}  {'(MB)':>9}  "
            f"{'(MB)':>10}  {'(s)':>7}"
        )
        sep = "─" * len(header)
        lines = [header, units, sep]

        for r in results:
            lines.append(
                f"{r.name:<22} {r.dim:>4}  {r.solver:>6}  "
                f"{_bytes_to_mb(r.process_peak_rss):>10.2f}  "
                f"{_bytes_to_mb(r.process_delta_rss):>9.2f}  "
                f"{_bytes_to_mb(r.jax_bytes):>10.2f}  "
                f"{r.solve_time_s:>7.3f}"
            )

    return "\n".join(lines)
