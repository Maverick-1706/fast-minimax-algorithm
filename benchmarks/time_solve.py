"""Wall-clock benchmarks (JIT-normalized).

Every baseline is JIT-compiled via jax.lax.fori_loop for identical
compilation treatment.  All timing uses bootstrap 95% CI (default 5
repeats) and IQR outlier flagging.
"""

from __future__ import annotations

import gc
import time

import jax
import jax.numpy as jnp

from minimax_aipe import solve
from benchmarks.baselines import run_eg_jit_benchmark, run_gda_jit_benchmark
from benchmarks.oracles import count_solver_oracles, count_eg_oracles, count_gda_oracles
from benchmarks.stats import summarise
from minimax_aipe.problem import BenchmarkProblem


# ── Timing infrastructure ────────────────────────────────────────────────


def _time_callable(fn, n_warmup: int = 1, n_repeats: int = 5) -> dict:
    """Time a callable with warmup runs.

    Returns dict with {mean, std, min, max, ci, raw_times, summary}.
    """
    last_result = None
    for _ in range(n_warmup):
        last_result = fn()

    times = []
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        last_result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    s = summarise(times)

    return {
        "mean": s.mean,
        "std": s.std,
        "min": s.min,
        "max": s.max,
        "ci": s.ci,
        "raw": times,
        "n_outliers": s.n_outliers,
        "outliers": s.outliers,
        "result": last_result,
    }


# ── JIT vs Eager ─────────────────────────────────────────────────────────


def benchmark_jit_vs_eager(
    prob: BenchmarkProblem,
    epsilon: float = 0.01,
    n_warmup: int = 1,
    n_repeats: int = 5,
    M_saddle: str = "npe",
) -> dict:
    """Compare JAX JIT enabled vs disabled on the pure numerical core."""
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    
    # Pre-setup the computational core to isolate JIT numerical speedup
    from minimax_aipe.framework import RegularizedSubproblem
    from minimax_aipe.npe import npe

    gamma = 1.0
    kernel = RegularizedSubproblem(problem, gamma)
    z0 = prob.z0
    if z0 is None:
        z0 = jnp.zeros(problem.dim_x + problem.dim_y)
        
    x_bar = z0[:problem.dim_x]
    y_bar = z0[problem.dim_x:]
    npe_gamma = 2.0 * kernel.rho_h

    def F_h(z): return kernel.operator_F_h(z, x_bar, y_bar)
    def merit(z): return jnp.dot(F_h(z), F_h(z))

    if M_saddle == "npe":
        oracle = kernel.make_crn_oracle(x_bar, y_bar, npe_gamma, tol=1e-4)
        def _core():
            return npe(oracle, F_h, z0, 50, npe_gamma, project=kernel.project, fn=merit)
    else:
        from minimax_aipe.len import len_loop, make_lazy_crn_npe_oracle
        h_prob = kernel.make_h_problem(x_bar, y_bar)
        oracle = make_lazy_crn_npe_oracle(h_prob, npe_gamma, tol=1e-4)
        def _core():
            return len_loop(oracle, F_h, z0, 50, npe_gamma, m=5, project=kernel.project, fn=merit)

    core_jit = jax.jit(_core)

    def run_solve():
        z_out, _ = core_jit()
        z_out.block_until_ready()
        return z_out

    jit_times = _time_callable(run_solve, n_warmup=n_warmup, n_repeats=n_repeats)

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
    problems: list[BenchmarkProblem],
    epsilon: float = 0.01,
    n_repeats: int = 5,
) -> list[dict]:
    """Time AIPE-NPE, AIPE-LEN, JIT-EG, JIT-GDA on each problem.

    All baselines are JIT-compiled.  Reports oracle calls alongside timing.
    """
    for prob in problems:
        assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    rows = []

    for prob in problems:
        problem = prob.problem
        name = prob.name or "?"
        dim = prob.dim or problem.dim_x

        print(f"  Benchmarking {name} dim={dim} ...")
        row = {"name": name, "dim": dim}
        z0 = prob.z0

        # ── AIPE-NPE ───────────────────────────────────────────────
        def run_npe():
            return solve(problem, epsilon=epsilon, M_saddle="npe", z0=z0)

        row["aipe_npe"] = _time_callable(run_npe, n_warmup=1, n_repeats=n_repeats)
        result_npe = row["aipe_npe"]["result"]
        oc_npe = count_solver_oracles(result_npe)
        row["aipe_npe"]["gap"] = result_npe.gap
        row["aipe_npe"]["iterations"] = result_npe.iterations
        row["aipe_npe"]["oracle_calls"] = result_npe.oracle_calls
        row["aipe_npe"]["converged"] = result_npe.converged
        row["aipe_npe"]["oracles"] = oc_npe.to_dict()

        # ── AIPE-LEN ───────────────────────────────────────────────
        def run_len():
            return solve(problem, epsilon=epsilon, M_saddle="len", m_lazy=5, z0=z0)

        row["aipe_len"] = _time_callable(run_len, n_warmup=1, n_repeats=n_repeats)
        result_len = row["aipe_len"]["result"]
        oc_len = count_solver_oracles(result_len)
        row["aipe_len"]["gap"] = result_len.gap
        row["aipe_len"]["iterations"] = result_len.iterations
        row["aipe_len"]["oracle_calls"] = result_len.oracle_calls
        row["aipe_len"]["converged"] = result_len.converged
        row["aipe_len"]["oracles"] = oc_len.to_dict()

        # ── JIT-EG ─────────────────────────────────────────────────
        def run_eg():
            return run_eg_jit_benchmark(problem, epsilon=epsilon, z0=z0)

        row["eg"] = _time_callable(run_eg, n_warmup=1, n_repeats=n_repeats)
        eg_result = row["eg"]["result"]
        oc_eg = count_eg_oracles(eg_result.iterations)
        row["eg"]["gap"] = eg_result.gap
        row["eg"]["iterations"] = eg_result.iterations
        row["eg"]["converged"] = eg_result.converged
        row["eg"]["oracles"] = oc_eg.to_dict()

        # ── JIT-GDA ────────────────────────────────────────────────
        def run_gda():
            return run_gda_jit_benchmark(problem, epsilon=epsilon, z0=z0)

        row["gda"] = _time_callable(run_gda, n_warmup=1, n_repeats=n_repeats)
        gda_result = row["gda"]["result"]
        oc_gda = count_gda_oracles(gda_result.iterations)
        row["gda"]["gap"] = gda_result.gap
        row["gda"]["iterations"] = gda_result.iterations
        row["gda"]["converged"] = gda_result.converged
        row["gda"]["oracles"] = oc_gda.to_dict()

        rows.append(row)

    return rows


# ── Formatting ───────────────────────────────────────────────────────────


def format_timing(t: dict) -> str:
    """Format a timing dict as 'mean [ci_lo, ci_hi]'."""
    lo, hi = t["ci"]
    return f"{t['mean']:.4f} [{lo:.4f},{hi:.4f}]"


def format_solver_comparison_table(rows: list[dict]) -> str:
    """Format solver comparison results as a text table."""
    header = (
        f"{'Problem':<22} {'Dim':>4}  "
        f"{'AIPE-NPE (Time | CRN)':>32}  {'AIPE-LEN (Time | CRN)':>32}  "
        f"{'JIT-EG (Time | Grad)':>32}  {'JIT-GDA (Time | Grad)':>32}  "
        f"{'NPE gap':>8}  {'EG gap':>8}"
    )
    sep = "─" * len(header)
    lines = [header, sep]

    for r in rows:
        npe = format_timing(r["aipe_npe"])
        lnn = format_timing(r["aipe_len"])
        eg = format_timing(r["eg"])
        gda = format_timing(r["gda"])
        
        npe_str = f"{npe} | {r['aipe_npe']['oracles']['crn_calls']}"
        lnn_str = f"{lnn} | {r['aipe_len']['oracles']['crn_calls']}"
        eg_str = f"{eg} | {r['eg']['oracles']['grad_calls']}"
        gda_str = f"{gda} | {r['gda']['oracles']['grad_calls']}"

        lines.append(
            f"{r['name']:<22} {r['dim']:>4}  {npe_str:>32}  {lnn_str:>32}  {eg_str:>32}  {gda_str:>32}  "
            f"{r['aipe_npe']['gap']:>8.4f}  {r['eg']['gap']:>8.4f}"
        )

    return "\n".join(lines)


def format_jit_table(rows: list[dict]) -> str:
    """Format JIT vs eager results as a text table."""
    header = f"{'Problem':<22} {'Dim':>4}  {'JIT (s)':>24}  {'Eager (s)':>24}  {'Speedup':>8}"
    sep = "─" * len(header)
    lines = [header, sep]

    for r in rows:
        jit = r["jit"]
        eager = r["eager"]
        lines.append(
            f"{r['name']:<22} {r['dim']:>4}  "
            f"{format_timing(jit):>24}  {format_timing(eager):>24}  "
            f"{r['speedup']:>7.2f}x"
        )

    return "\n".join(lines)
