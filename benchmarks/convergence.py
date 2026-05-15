"""ε-sweep / convergence rate analysis.

Runs each solver at multiple ε values and reports gap vs oracle calls (the
paper's primary metric).  This is what actually validates the O(ε^{-4/7})
claim — wall-clock alone is insufficient.
"""

from __future__ import annotations

import jax.numpy as jnp

from minimax_aipe import solve
from benchmarks.baselines import run_eg_jit_benchmark
from benchmarks.oracles import count_eg_oracles
from minimax_aipe.problem import BenchmarkProblem


def sweep_epsilon(
    prob,
    epsilons: list[float],
) -> list[dict]:
    """Run AIPE-NPE, AIPE-LEN, and JIT-EG at each ε and collect gap + oracle calls.

    Parameters
    ----------
    prob : BenchmarkProblem
        From the problem zoo.
    epsilons : list[float]
        Target gap values to sweep.

    Returns
    -------
    list[dict]
        One dict per ε with keys: name, dim, epsilon,
        npe_gap, npe_oracle_calls, len_gap, len_oracle_calls,
        eg_gap, eg_grad_calls.
    """
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

    rows = []
    for eps in epsilons:
        row = {"name": name, "dim": dim, "epsilon": eps}

        # AIPE-NPE
        r_npe = solve(problem, epsilon=eps, M_saddle="npe", z0=prob.z0)
        row["npe_gap"] = float(r_npe.gap)
        row["npe_oracle_calls"] = r_npe.oracle_calls

        # AIPE-LEN
        r_len = solve(problem, epsilon=eps, M_saddle="len", m_lazy=5, z0=prob.z0)
        row["len_gap"] = float(r_len.gap)
        row["len_oracle_calls"] = r_len.oracle_calls

        # JIT-EG (early stopping enabled via tol)
        eg_res = run_eg_jit_benchmark(problem, epsilon=eps, max_iters=5000, z0=prob.z0)
        eg_counter = count_eg_oracles(eg_res.iterations)
        
        row["eg_gap"] = float(eg_res.gap)
        row["eg_grad_calls"] = eg_counter.grad_calls

        rows.append(row)

    return rows


def format_convergence_table(rows: list[dict]) -> str:
    """Format convergence sweep results as a text table."""
    header = (
        f"{'Problem':<18} {'Dim':>4}  {'ε':>8}  "
        f"{'NPE gap':>10} {'NPE calls':>10}  "
        f"{'LEN gap':>10} {'LEN calls':>10}  "
        f"{'EG gap':>10} {'EG grads':>10}"
    )
    sep = "─" * len(header)
    lines = [header, sep]

    for r in rows:
        lines.append(
            f"{r['name']:<18} {r['dim']:>4}  {r['epsilon']:>8.4f}  "
            f"{r['npe_gap']:>10.6f} {r['npe_oracle_calls']:>10}  "
            f"{r['len_gap']:>10.6f} {r['len_oracle_calls']:>10}  "
            f"{r['eg_gap']:>10.6f} {r['eg_grad_calls']:>10}"
        )

    return "\n".join(lines)
