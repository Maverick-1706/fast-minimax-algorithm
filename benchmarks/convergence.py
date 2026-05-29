"""ε-sweep / convergence rate analysis.

Runs each solver at multiple ε values and reports gap vs oracle calls (the
paper's primary metric).  This is what actually validates the O(ε^{-4/7})
claim — wall-clock alone is insufficient.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import signal
import threading
import time
from contextlib import contextmanager

import jax.numpy as jnp

from minimax_aipe import OracleStats, solve
from minimax_aipe.framework import solve_outer_trace
from benchmarks import config
from benchmarks.baselines import run_eg_jit_benchmark, run_npe_restart_jit_benchmark
from benchmarks.results import BenchmarkResult
from minimax_aipe.problem import BenchmarkProblem, MinimaxProblem


def _has_exact_gap(problem: MinimaxProblem) -> bool:
    duality_gap = getattr(problem, "duality_gap", None)
    if duality_gap is None:
        return False
    duality_gap_fn = getattr(duality_gap, "__func__", duality_gap)
    return duality_gap_fn is not MinimaxProblem.duality_gap


def _gap_source(prob: BenchmarkProblem) -> str:
    return "exact" if _has_exact_gap(prob.problem) else "estimated"


def _baseline_max_iters(problem, eps: float, d: int, *, base_cap: int) -> int:
    ell = max(float(problem.ell or 1.0), 1.0)
    scaled = int(2000 * d * ell**0.5 / max(eps, 1e-6))
    return min(base_cap, max(5000, scaled))


class _SolveTimeoutError(TimeoutError):
    pass


def _supports_alarm_timeout() -> bool:
    return (
        hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


@contextmanager
def _alarm_timeout(timeout_seconds: float):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _raise_timeout(_signum, _frame):
        raise _SolveTimeoutError(f"solver timed out after {timeout_seconds:.3f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _run_with_timeout(fn, timeout_seconds: float | None):
    if timeout_seconds is None or timeout_seconds <= 0:
        return fn(), False, 0.0

    started = time.perf_counter()
    if _supports_alarm_timeout():
        try:
            with _alarm_timeout(timeout_seconds):
                return fn(), False, time.perf_counter() - started
        except _SolveTimeoutError:
            return None, True, time.perf_counter() - started

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds), False, time.perf_counter() - started
    except concurrent.futures.TimeoutError:
        future.cancel()
        return None, True, time.perf_counter() - started
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _timeout_stats(solver: str) -> OracleStats:
    call_type = "gradient" if solver == "eg" else "crn"
    return OracleStats(call_type=call_type)


def _timeout_result(
    *,
    solver: str,
    problem: str,
    dim: int,
    epsilon: float,
    elapsed: float,
    gap_source: str,
    timeout_seconds: float,
    gap_endpoints: list[float] | None = None,
    oracle_endpoints: list[float] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        solver=solver,
        problem=problem,
        dim=dim,
        epsilon=epsilon,
        wall_time_mean=elapsed,
        wall_time_std=0.0,
        ci=(elapsed, elapsed),
        oracle_stats=_timeout_stats(solver),
        converged=False,
        gap_achieved=False,
        final_gap=float("inf"),
        iterations=0,
        gap_source=gap_source,
        gap_endpoints=gap_endpoints,
        oracle_endpoints=oracle_endpoints,
        extra_metadata={"timed_out": True, "timeout_seconds": timeout_seconds},
    )


def sweep_epsilon(
    prob,
    epsilons: list[float],
    *,
    timeout_seconds: float = config.CONVERGENCE_SOLVE_TIMEOUT_SECONDS,
    progress_callback=None,
) -> list[BenchmarkResult]:
    """Run AIPE-NPE, AIPE-LEN, and JIT-EG at each ε and collect gap + oracle stats.

    Parameters
    ----------
    prob : BenchmarkProblem
        From the problem zoo.
    epsilons : list[float]
        Target gap values to sweep.

    Returns
    -------
    list[BenchmarkResult]
        Three results per ε (npe, len, eg).
    """
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y
    gap_source = _gap_source(prob)

    rows = []
    for eps in epsilons:
        # AIPE-NPE
        r_npe, timed_out, elapsed = _run_with_timeout(
            lambda: solve(problem, epsilon=eps, M_saddle="npe", z0=prob.z0),
            timeout_seconds,
        )
        if timed_out:
            row = _timeout_result(
                solver="aipe_npe", problem=name, dim=dim, epsilon=eps,
                elapsed=elapsed, gap_source=gap_source, timeout_seconds=timeout_seconds,
            )
            rows.append(row)
        else:
            row = BenchmarkResult(
                solver="aipe_npe", problem=name, dim=dim, epsilon=eps,
                wall_time_mean=elapsed, wall_time_std=0.0,
                ci=(elapsed, elapsed),
                oracle_stats=r_npe.oracle_stats,
                converged=r_npe.converged, gap_achieved=r_npe.gap <= eps,
                final_gap=float(r_npe.gap),
                iterations=r_npe.iterations,
                gap_source=gap_source,
            )
            rows.append(row)
        if progress_callback is not None:
            progress_callback(row)

        # AIPE-LEN
        r_len, timed_out, elapsed = _run_with_timeout(
            lambda: solve(problem, epsilon=eps, M_saddle="len", m_lazy=5, z0=prob.z0),
            timeout_seconds,
        )
        if timed_out:
            row = _timeout_result(
                solver="aipe_len", problem=name, dim=dim, epsilon=eps,
                elapsed=elapsed, gap_source=gap_source, timeout_seconds=timeout_seconds,
            )
            rows.append(row)
        else:
            row = BenchmarkResult(
                solver="aipe_len", problem=name, dim=dim, epsilon=eps,
                wall_time_mean=elapsed, wall_time_std=0.0,
                ci=(elapsed, elapsed),
                oracle_stats=r_len.oracle_stats,
                converged=r_len.converged, gap_achieved=r_len.gap <= eps,
                final_gap=float(r_len.gap),
                iterations=r_len.iterations,
                gap_source=gap_source,
            )
            rows.append(row)
        if progress_callback is not None:
            progress_callback(row)

        # Standalone NPE-restart
        snpe_max_iters = _baseline_max_iters(problem, eps, d, base_cap=1_000_000)
        snpe_res, timed_out, elapsed = _run_with_timeout(
            lambda: run_npe_restart_jit_benchmark(
                problem, epsilon=eps, max_iters=snpe_max_iters, z0=prob.z0,
            ),
            timeout_seconds,
        )
        if timed_out:
            row = _timeout_result(
                solver="npe_restart", problem=name, dim=dim, epsilon=eps,
                elapsed=elapsed, gap_source=gap_source, timeout_seconds=timeout_seconds,
            )
            rows.append(row)
        else:
            row = BenchmarkResult(
                solver="npe_restart", problem=name, dim=dim, epsilon=eps,
                wall_time_mean=elapsed, wall_time_std=0.0,
                ci=(elapsed, elapsed),
                oracle_stats=snpe_res.oracle_stats,
                converged=snpe_res.converged, gap_achieved=snpe_res.gap_achieved,
                final_gap=float(snpe_res.gap),
                iterations=snpe_res.iterations,
                gap_source=gap_source,
            )
            rows.append(row)
        if progress_callback is not None:
            progress_callback(row)

        # JIT-EG (early stopping enabled via tol, scaled by dim/epsilon)
        eg_max_iters = _baseline_max_iters(problem, eps, d, base_cap=2_000_000)
        eg_res, timed_out, elapsed = _run_with_timeout(
            lambda: run_eg_jit_benchmark(problem, epsilon=eps, max_iters=eg_max_iters, z0=prob.z0),
            timeout_seconds,
        )
        if timed_out:
            row = _timeout_result(
                solver="eg", problem=name, dim=dim, epsilon=eps,
                elapsed=elapsed, gap_source=gap_source, timeout_seconds=timeout_seconds,
            )
            rows.append(row)
        else:
            row = BenchmarkResult(
                solver="eg", problem=name, dim=dim, epsilon=eps,
                wall_time_mean=elapsed, wall_time_std=0.0,
                ci=(elapsed, elapsed),
                oracle_stats=eg_res.oracle_stats,
                converged=eg_res.converged, gap_achieved=eg_res.gap_achieved,
                final_gap=float(eg_res.gap),
                iterations=eg_res.iterations,
                gap_source=gap_source,
            )
            rows.append(row)
        if progress_callback is not None:
            progress_callback(row)

    return rows

def sweep_epsilon_endpoints(
    prob,
    epsilons: list[float],
    *,
    timeout_seconds: float = config.CONVERGENCE_SOLVE_TIMEOUT_SECONDS,
) -> list[BenchmarkResult]:
    """Capture real gap-vs-oracle endpoints after each outer epoch.

    This runs the outer restart loop eagerly, stopping between epochs to
    recover the current saddle estimate and evaluate the gap.  The resulting
    endpoints are an actual trajectory for the target epsilon, not independent
    cold-start solves at different epsilon levels.

    Parameters
    ----------
    prob : BenchmarkProblem
        Problem instance (wraps problem + solver).
    epsilons : list[float]
        Target duality gaps to solve for.

    Returns
    -------
    list[BenchmarkResult]
        One result per ε, with ``gap_endpoints`` and ``oracle_endpoints`` populated.
        Empty endpoints are stored as empty lists (never None) so downstream
        code can always iterate safely.
    """
    rows: list[BenchmarkResult] = []
    gap_source = _gap_source(prob)

    for eps in epsilons:
        _ = jnp.zeros(1).block_until_ready()
        res, timed_out, elapsed = _run_with_timeout(
            lambda: solve_outer_trace(prob.problem, epsilon=eps, z0=prob.z0),
            timeout_seconds,
        )
        if timed_out:
            rows.append(_timeout_result(
                solver="minimax_aipe",
                problem=prob.name,
                dim=prob.dim or prob.problem.dim_x,
                epsilon=eps,
                elapsed=elapsed,
                gap_source=gap_source,
                timeout_seconds=timeout_seconds,
                gap_endpoints=[],
                oracle_endpoints=[],
            ))
            continue

        res.x.block_until_ready()
        gap_endpoints = list((res.history or {}).get("gap_endpoints", []))
        oracle_endpoints = list((res.history or {}).get("oracle_endpoints", []))

        row = BenchmarkResult(
            solver="minimax_aipe",
            problem=prob.name,
            dim=prob.dim or prob.problem.dim_x,
            epsilon=eps,
            wall_time_mean=0.0,
            wall_time_std=0.0,
            ci=(0.0, 0.0),
            oracle_stats=res.oracle_stats,
            converged=res.converged,
            gap_achieved=res.gap <= eps,
            final_gap=float(res.gap),
            iterations=res.iterations,
            gap_endpoints=gap_endpoints,
            oracle_endpoints=oracle_endpoints,
            gap_source=gap_source,
            best_gap=(res.history or {}).get("best_gap"),
            best_gap_epoch=(res.history or {}).get("best_gap_epoch"),
            best_oracle_cost=(res.history or {}).get("best_oracle_cost"),
        )
        rows.append(row)

    return rows

def format_convergence_table(rows: list[BenchmarkResult]) -> str:
    """Format convergence sweep results as a text table.

    Columns show duality gap and primary oracle calls for each solver variant.
    ``ok`` indicates whether the termination threshold was met.
    """
    header = (
        f"{'Problem':<18} {'Dim':>4}  {'ε':>8}  "
        f"{'A-NPE gap':>10} {'A-NPE calls':>12} {'ok':>3}  "
        f"{'Base-NPE gap':>12} {'Base calls':>12} {'ok':>3}  "
        f"{'LEN gap':>10} {'LEN calls':>10} {'ok':>3}  "
        f"{'EG gap':>10} {'EG calls':>9} {'ok':>3}"
    )
    sep = "─" * len(header)
    lines = [header, sep]

    key = lambda r: (r.problem, r.dim, r.epsilon)
    for (prob_name, dim, eps), group in itertools.groupby(
        sorted(rows, key=key), key=key
    ):
        group_list = list(group)
        npe = next((r for r in group_list if r.solver == "aipe_npe"), None)
        s_npe = next((r for r in group_list if r.solver == "npe_restart"), None)
        lnn = next((r for r in group_list if r.solver == "aipe_len"), None)
        eg = next((r for r in group_list if r.solver == "eg"), None)

        def _calls(r: BenchmarkResult | None) -> int:
            if r is None or r.oracle_stats is None:
                return 0
            return int(r.oracle_stats.oracle_calls)

        npe_gap = npe.final_gap if npe else 0.0
        npe_calls = _calls(npe)
        npe_ok = "Y" if (npe and npe.gap_achieved) else "N"
        snpe_gap = s_npe.final_gap if s_npe else 0.0
        snpe_calls = _calls(s_npe)
        snpe_ok = "Y" if (s_npe and s_npe.gap_achieved) else "N"
        len_gap = lnn.final_gap if lnn else 0.0
        len_calls = _calls(lnn)
        len_ok = "Y" if (lnn and lnn.gap_achieved) else "N"
        
        eg_residual = eg.final_gap if eg else 0.0
        eg_calls = _calls(eg)
        eg_ok = "Y" if (eg and eg.gap_achieved) else "N"

        lines.append(
            f"{prob_name:<18} {dim:>4}  {eps:>8.4f}  "
            f"{npe_gap:>10.3e} {npe_calls:>12} {npe_ok:>3}  "
            f"{snpe_gap:>12.3e} {snpe_calls:>12} {snpe_ok:>3}  "
            f"{len_gap:>10.3e} {len_calls:>10} {len_ok:>3}  "
            f"{eg_residual:>10.3e} {eg_calls:>9} {eg_ok:>3}"
        )

    return "\n".join(lines)


def format_endpoints_table(results: list[BenchmarkResult]) -> str:
    """Format a single run as an endpoints convergence table.
    
    Only formats the first result with a populated gap_endpoints.
    """
    endpoints = [r for r in results if r.gap_endpoints]
    if not endpoints:
        return "(no data — solver did not return gap_endpoints)"
    
    r = endpoints[0]
    header = f"{'Outer':>5}  {'Gap':>12}  {'Norm. cost':>14}"
    sep = "─" * len(header)
    lines = [header, sep]
    for i, (gap, cost) in enumerate(zip(r.gap_endpoints, r.oracle_endpoints or [])):
        lines.append(f"{i+1:>5}  {gap:>12.6e}  {cost:>14.2e}")
    if r.best_gap is not None:
        lines.append(
            f" best  {r.best_gap:>12.6e}  {float(r.best_oracle_cost or 0.0):>14.2e}"
        )
    return "\n".join(lines)
