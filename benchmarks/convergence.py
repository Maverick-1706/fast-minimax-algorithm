"""ε-sweep / convergence rate analysis.

Runs each solver at multiple ε values and reports gap vs oracle calls (the
paper's primary metric).  This is what actually validates the O(ε^{-4/7})
claim — wall-clock alone is insufficient.
"""

from __future__ import annotations

import itertools
import time
import jax.numpy as jnp

from minimax_aipe import solve
from benchmarks import config
from benchmarks.baselines import run_eg_jit_benchmark, run_npe_restart_jit_benchmark
from benchmarks.results import BenchmarkResult
from benchmarks.oracles import count_npe_oracles, count_eg_oracles
from minimax_aipe.problem import BenchmarkProblem


def sweep_epsilon(
    prob,
    epsilons: list[float],
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

    rows = []
    for eps in epsilons:
        # AIPE-NPE
        r_npe = solve(problem, epsilon=eps, M_saddle="npe", z0=prob.z0)
        rows.append(BenchmarkResult(
            solver="aipe_npe", problem=name, dim=dim, epsilon=eps,
            wall_time_mean=0.0, wall_time_std=0.0,
            ci=(0.0, 0.0),
            oracle_stats=r_npe.oracle_stats,
            converged=r_npe.converged, gap_achieved=r_npe.gap <= eps,
            final_gap=float(r_npe.gap),
            iterations=r_npe.iterations,
        ))

        # AIPE-LEN
        r_len = solve(problem, epsilon=eps, M_saddle="len", m_lazy=5, z0=prob.z0)
        rows.append(BenchmarkResult(
            solver="aipe_len", problem=name, dim=dim, epsilon=eps,
            wall_time_mean=0.0, wall_time_std=0.0,
            ci=(0.0, 0.0),
            oracle_stats=r_len.oracle_stats,
            converged=r_len.converged, gap_achieved=r_len.gap <= eps,
            final_gap=float(r_len.gap),
            iterations=r_len.iterations,
        ))

        # Standalone NPE-restart
        snpe_max_iters = min(100_000, max(5000, int(2000 * d / max(eps, 1e-6))))
        snpe_res = run_npe_restart_jit_benchmark(problem, epsilon=eps, max_iters=snpe_max_iters, z0=prob.z0)
        snpe_stats = snpe_res.oracle_stats
        rows.append(BenchmarkResult(
            solver="npe_restart", problem=name, dim=dim, epsilon=eps,
            wall_time_mean=0.0, wall_time_std=0.0,
            ci=(0.0, 0.0),
            oracle_stats=snpe_stats,
            converged=snpe_res.converged, gap_achieved=snpe_res.gap_achieved,
            final_gap=float(snpe_res.gap),
            iterations=snpe_res.iterations,
        ))

        # JIT-EG (early stopping enabled via tol, scaled by dim/epsilon)
        eg_max_iters = min(200_000, max(5000, int(2000 * d / max(eps, 1e-6))))
        eg_res = run_eg_jit_benchmark(problem, epsilon=eps, max_iters=eg_max_iters, z0=prob.z0)
        eg_stats = eg_res.oracle_stats
        rows.append(BenchmarkResult(
            solver="eg", problem=name, dim=dim, epsilon=eps,
            wall_time_mean=0.0, wall_time_std=0.0,
            ci=(0.0, 0.0),
            oracle_stats=eg_stats,
            converged=eg_res.converged, gap_achieved=eg_res.gap_achieved,
            final_gap=float(eg_res.gap),
            iterations=eg_res.iterations,
        ))

    return rows

def sweep_epsilon_endpoints(
    prob,
    epsilons: list[float],
) -> list[BenchmarkResult]:
    """Like sweep_epsilon(), but captures independent gap-vs-oracle-call endpoints.

    Since the solver doesn't expose per-outer-iteration hooks, we build
    the convergence curve by independently running ``solve()`` at geometrically-spaced
    epsilon levels from coarse to target, cold-starting each run from the
    initial point. This produces a valid gap-vs-oracle-calls
    curve for computing convergence rates without modifying the algorithm internals.

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
    import jax.numpy as jnp
    from minimax_aipe.framework import solve

    N_TRACE_POINTS = 8  # Number of checkpoints in the convergence trace

    rows: list[BenchmarkResult] = []

    for eps in epsilons:
        # ── Build epsilon schedule: log-spaced from coarse to target ──
        coarse_eps = min(10.0 * eps, 0.5)
        eps_schedule = [
            coarse_eps * (eps / coarse_eps) ** (i / (N_TRACE_POINTS - 1))
            for i in range(N_TRACE_POINTS)
        ]

        gap_endpoints: list[float] = []
        oracle_endpoints: list[int] = []

        # Run intermediate coarser resolutions from a cold start to collect data points
        for trace_eps in eps_schedule[:-1]:
            res_trace = solve(prob.problem, epsilon=trace_eps, z0=prob.z0)
            gap_endpoints.append(float(res_trace.gap))
            oracle_endpoints.append(int(res_trace.oracle_stats.oracle_calls))

        # Force device synchronization before starting the timer for the definitive target solve
        _ = jnp.zeros(1).block_until_ready()
        t_start = time.perf_counter()
        
        # Run the definitive final target epsilon solve
        res = solve(prob.problem, epsilon=eps, z0=prob.z0)
        res.x.block_until_ready()
        elapsed = time.perf_counter() - t_start

        # Append final target metrics to complete the endpoints arrays
        gap_endpoints.append(float(res.gap))
        oracle_endpoints.append(int(res.oracle_stats.oracle_calls))

        row = BenchmarkResult(
            solver="minimax_aipe",
            problem=prob.name,
            dim=prob.dim or prob.problem.dim_x,
            epsilon=eps,
            wall_time_mean=elapsed,
            wall_time_std=0.0,
            ci=(elapsed, elapsed),
            oracle_stats=res.oracle_stats,
            converged=res.converged,
            gap_achieved=res.gap <= eps,
            final_gap=float(res.gap),
            iterations=res.iterations,
            gap_endpoints=gap_endpoints,
            oracle_endpoints=oracle_endpoints,
        )
        rows.append(row)

    return rows

def format_convergence_table(rows: list[BenchmarkResult]) -> str:
    """Format convergence sweep results as a text table.

    Columns show duality gap for NPE/LEN/EG variants along with primary oracle
    calls (oracle_calls).  ``ok`` indicates whether the termination threshold was met.
    """
    header = (
        f"{'Problem':<18} {'Dim':>4}  {'ε':>8}  "
        f"{'A-NPE gap':>10} {'A-NPE orc':>10} {'ok':>3}  "
        f"{'S-NPE gap':>10} {'S-NPE orc':>10} {'ok':>3}  "
        f"{'LEN gap':>10} {'LEN orc':>10} {'ok':>3}  "
        f"{'EG gap':>10} {'EG orc':>10} {'ok':>3}"
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

        npe_gap = npe.final_gap if npe else 0.0
        npe_cost = npe.oracle_stats.oracle_calls if npe else 0.0
        npe_ok = "Y" if (npe and npe.gap_achieved) else "N"
        snpe_gap = s_npe.final_gap if s_npe else 0.0
        snpe_cost = s_npe.oracle_stats.oracle_calls if s_npe else 0.0
        snpe_ok = "Y" if (s_npe and s_npe.gap_achieved) else "N"
        len_gap = lnn.final_gap if lnn else 0.0
        len_cost = lnn.oracle_stats.oracle_calls if lnn else 0.0
        len_ok = "Y" if (lnn and lnn.gap_achieved) else "N"
        
        eg_residual = eg.final_gap if eg else 0.0
        eg_cost = eg.oracle_stats.oracle_calls if eg else 0.0
        eg_ok = "Y" if (eg and eg.gap_achieved) else "N"

        lines.append(
            f"{prob_name:<18} {dim:>4}  {eps:>8.4f}  "
            f"{npe_gap:>10.6f} {npe_cost:>10.2e} {npe_ok:>3}  "
            f"{snpe_gap:>10.6f} {snpe_cost:>10.2e} {snpe_ok:>3}  "
            f"{len_gap:>10.6f} {len_cost:>10.2e} {len_ok:>3}  "
            f"{eg_residual:>10.6f} {eg_cost:>10.2e} {eg_ok:>3}"
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
    header = f"{'Outer':>5}  {'Gap':>12}  {'Oracle calls':>14}"
    sep = "─" * len(header)
    lines = [header, sep]
    for i, (gap, calls) in enumerate(zip(r.gap_endpoints, r.oracle_endpoints or [])):
        lines.append(f"{i+1:>5}  {gap:>12.6e}  {calls:>14}")
    return "\n".join(lines)