"""ε-sweep / convergence rate analysis.

Runs each solver at multiple ε values and reports gap vs oracle calls (the
paper's primary metric).  This is what actually validates the O(ε^{-4/7})
claim — wall-clock alone is insufficient.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp

from minimax_aipe import solve
from benchmarks import config
from benchmarks.baselines import run_eg_jit_benchmark
from benchmarks.results import BenchmarkResult
from benchmarks.oracles import count_eg_oracles
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
            normalized_cost=r_npe.oracle_stats.normalized_cost(d),
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
            normalized_cost=r_len.oracle_stats.normalized_cost(d),
        ))

        # JIT-EG (early stopping enabled via tol, scaled by dim/epsilon)
        eg_max_iters = min(200_000, max(5000, int(2000 * d / max(eps, 1e-6))))
        eg_res = run_eg_jit_benchmark(problem, epsilon=eps, max_iters=eg_max_iters, z0=prob.z0)
        eg_stats = count_eg_oracles(eg_res.iterations)
        rows.append(BenchmarkResult(
            solver="eg", problem=name, dim=dim, epsilon=eps,
            wall_time_mean=0.0, wall_time_std=0.0,
            ci=(0.0, 0.0),
            oracle_stats=eg_stats,
            converged=eg_res.converged, gap_achieved=eg_res.gap_achieved,
            final_gap=float(eg_res.gap),
            iterations=eg_res.iterations,
            normalized_cost=eg_stats.normalized_cost(d),
        ))

    return rows


def format_convergence_table(rows: list[BenchmarkResult]) -> str:
    """Format convergence sweep results as a text table.

    Columns show duality gap and FLOP-normalized cost (normalized_cost)
    per solver.  ``gap_ok`` indicates whether gap <= ε (the success criterion).
    """
    header = (
        f"{'Problem':<18} {'Dim':>4}  {'ε':>8}  "
        f"{'NPE gap':>10} {'NPE cost':>10} {'ok':>3}  "
        f"{'LEN gap':>10} {'LEN cost':>10} {'ok':>3}  "
        f"{'EG gap':>10} {'EG cost':>10} {'ok':>3}"
    )
    sep = "─" * len(header)
    lines = [header, sep]

    key = lambda r: (r.problem, r.dim, r.epsilon)
    for (prob_name, dim, eps), group in itertools.groupby(
        sorted(rows, key=key), key=key
    ):
        group_list = list(group)
        npe = next((r for r in group_list if r.solver == "aipe_npe"), None)
        lnn = next((r for r in group_list if r.solver == "aipe_len"), None)
        eg = next((r for r in group_list if r.solver == "eg"), None)

        npe_gap = npe.final_gap if npe else 0.0
        npe_cost = npe.normalized_cost if npe else 0.0
        npe_ok = "Y" if (npe and npe.gap_achieved) else "N"
        len_gap = lnn.final_gap if lnn else 0.0
        len_cost = lnn.normalized_cost if lnn else 0.0
        len_ok = "Y" if (lnn and lnn.gap_achieved) else "N"
        eg_gap = eg.final_gap if eg else 0.0
        eg_cost = eg.normalized_cost if eg else 0.0
        eg_ok = "Y" if (eg and eg.gap_achieved) else "N"

        lines.append(
            f"{prob_name:<18} {dim:>4}  {eps:>8.4f}  "
            f"{npe_gap:>10.6f} {npe_cost:>10.2e} {npe_ok:>3}  "
            f"{len_gap:>10.6f} {len_cost:>10.2e} {len_ok:>3}  "
            f"{eg_gap:>10.6f} {eg_cost:>10.2e} {eg_ok:>3}"
        )

    return "\n".join(lines)
