"""Wall-clock benchmarks (JIT-normalized).

Every baseline is JIT-compiled via jax.lax.fori_loop for identical
compilation treatment.  All timing uses BCa bootstrap 95% CI, IQR
outlier removal, and automated repeat-policy when variance is high.
"""

from __future__ import annotations

import gc
import itertools
import time

import jax
import jax.numpy as jnp

from minimax_aipe import solve
from benchmarks import config
from benchmarks.baselines import run_eg_jit_benchmark, run_gda_jit_benchmark
from benchmarks.results import BenchmarkResult
from benchmarks.stats import summarise, should_repeat
from minimax_aipe.problem import BenchmarkProblem


# ── Timing infrastructure ────────────────────────────────────────────────


def _time_callable(fn, n_warmup: int | None = None, n_repeats: int | None = None) -> dict:
    """Time a callable with warmup runs and automated repeat-policy.

    If ``config.AUTO_REPEAT`` is True, extra repetitions are added when
    the coefficient of variation is too high or too many outliers are
    detected.

    Returns dict with {mean, std, min, max, ci, raw_times, summary}.
    """
    if n_warmup is None:
        n_warmup = config.N_WARMUP
    if n_repeats is None:
        n_repeats = config.N_REPEATS_FULL

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

    # ── Automated repeat policy ───────────────────────────────────────
    if config.AUTO_REPEAT:
        for _ in range(config.AUTO_REPEAT_MAX_EXTRA):
            decision = should_repeat(times)
            if not decision.should_repeat:
                break
            for _ in range(config.AUTO_REPEAT_N):
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
    epsilon: float | None = None,
    n_warmup: int | None = None,
    n_repeats: int | None = None,
    M_saddle: str = "npe",
) -> dict:
    """Compare JAX JIT enabled vs disabled on the pure numerical core."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
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

    # 1. Measure Eager FIRST (completely avoiding JIT cache contamination)
    def run_eager():
        z_out, _ = _core()
        z_out.block_until_ready()
        return z_out

    eager_times = _time_callable(run_eager, n_warmup=0, n_repeats=n_repeats)

    # 2. Measure JIT SECOND
    core_jit = jax.jit(_core)

    def run_jit():
        z_out, _ = core_jit()
        z_out.block_until_ready()
        return z_out

    jit_times = _time_callable(run_jit, n_warmup=n_warmup, n_repeats=n_repeats)

    speedup = eager_times["mean"] / max(jit_times["mean"], 1e-12)

    return {
        "jit": jit_times,
        "eager": eager_times,
        "speedup": speedup,
    }


# ── Solver comparison ────────────────────────────────────────────────────


def benchmark_solver_comparison(
    problems: list[BenchmarkProblem],
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Time AIPE-NPE, AIPE-LEN, JIT-EG, JIT-GDA on each problem.

    All baselines are JIT-compiled.  Returns one BenchmarkResult per solver.
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_FULL
    for prob in problems:
        assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    rows = []

    for prob in problems:
        problem = prob.problem
        name = prob.name or "?"
        dim = prob.dim or problem.dim_x
        d = problem.dim_x + problem.dim_y

        print(f"  Benchmarking {name} dim={dim} ...")
        z0 = prob.z0

        # ── AIPE-NPE ───────────────────────────────────────────────
        def run_npe():
            res = solve(problem, epsilon=epsilon, M_saddle="npe", z0=z0)
            # FORCE SYNC: Prevent async dispatch illusion
            if hasattr(res, "x"): res.x.block_until_ready()
            if hasattr(res, "y"): res.y.block_until_ready()
            if hasattr(res, "gap") and hasattr(res.gap, "block_until_ready"):
                res.gap.block_until_ready()
            return res
        t_npe = _time_callable(run_npe, n_warmup=1, n_repeats=n_repeats)
        result_npe = t_npe["result"]
        rows.append(BenchmarkResult(
            solver="aipe_npe",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=t_npe["mean"],
            wall_time_std=t_npe["std"],
            ci=t_npe["ci"],
            oracle_stats=result_npe.oracle_stats,
            converged=result_npe.converged,
            gap_achieved=result_npe.gap <= epsilon,
            final_gap=float(result_npe.gap),
            iterations=result_npe.iterations,
            wall_time_min=t_npe["min"],
            wall_time_max=t_npe["max"],
            n_outliers=t_npe["n_outliers"],
            normalized_cost=result_npe.oracle_stats.normalized_cost(d),
        ))

        # ── AIPE-LEN ───────────────────────────────────────────────
        def run_len():
            res = solve(problem, epsilon=epsilon, M_saddle="len", m_lazy=5, z0=z0)
            # FORCE SYNC: Prevent async dispatch illusion
            if hasattr(res, "x"): res.x.block_until_ready()
            if hasattr(res, "y"): res.y.block_until_ready()
            if hasattr(res, "gap") and hasattr(res.gap, "block_until_ready"):
                res.gap.block_until_ready()
            return res
            
        t_len = _time_callable(run_len, n_warmup=1, n_repeats=n_repeats)
        result_len = t_len["result"]
        rows.append(BenchmarkResult(
            solver="aipe_len",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=t_len["mean"],
            wall_time_std=t_len["std"],
            ci=t_len["ci"],
            oracle_stats=result_len.oracle_stats,
            converged=result_len.converged,
            gap_achieved=result_len.gap <= epsilon,
            final_gap=float(result_len.gap),
            iterations=result_len.iterations,
            wall_time_min=t_len["min"],
            wall_time_max=t_len["max"],
            n_outliers=t_len["n_outliers"],
            normalized_cost=result_len.oracle_stats.normalized_cost(d),
        ))

        # ── JIT-EG ─────────────────────────────────────────────────
        def run_eg():
            return run_eg_jit_benchmark(problem, epsilon=epsilon, z0=z0)

        t_eg = _time_callable(run_eg, n_warmup=1, n_repeats=n_repeats)
        eg_result = t_eg["result"]
        rows.append(BenchmarkResult(
            solver="eg",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=t_eg["mean"],
            wall_time_std=t_eg["std"],
            ci=t_eg["ci"],
            oracle_stats=eg_result.oracle_stats,
            converged=eg_result.converged,
            gap_achieved=eg_result.gap_achieved,
            final_gap=float(eg_result.gap),
            iterations=eg_result.iterations,
            wall_time_min=t_eg["min"],
            wall_time_max=t_eg["max"],
            n_outliers=t_eg["n_outliers"],
            normalized_cost=eg_result.oracle_stats.normalized_cost(d) if eg_result.oracle_stats else 0.0,
        ))

        # ── JIT-GDA ────────────────────────────────────────────────
        def run_gda():
            return run_gda_jit_benchmark(problem, epsilon=epsilon, z0=z0)

        t_gda = _time_callable(run_gda, n_warmup=1, n_repeats=n_repeats)
        gda_result = t_gda["result"]
        rows.append(BenchmarkResult(
            solver="gda",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=t_gda["mean"],
            wall_time_std=t_gda["std"],
            ci=t_gda["ci"],
            oracle_stats=gda_result.oracle_stats,
            converged=gda_result.converged,
            gap_achieved=gda_result.gap_achieved,
            final_gap=float(gda_result.gap),
            iterations=gda_result.iterations,
            wall_time_min=t_gda["min"],
            wall_time_max=t_gda["max"],
            n_outliers=t_gda["n_outliers"],
            normalized_cost=gda_result.oracle_stats.normalized_cost(d) if gda_result.oracle_stats else 0.0,
        ))

    return rows


# ── Formatting ───────────────────────────────────────────────────────────


def format_solver_comparison_table(rows: list[BenchmarkResult]) -> str:
    """Format solver comparison results as a text table."""
    header = (
        f"{'Problem':<22} {'Dim':>4}  "
        f"{'AIPE-NPE (Time | NrmCost)':>34}  {'AIPE-LEN (Time | NrmCost)':>34}  "
        f"{'JIT-EG (Time | NrmCost)':>34}  {'JIT-GDA (Time | NrmCost)':>34}  "
        f"{'NPE gap':>8}  {'EG gap':>8}"
    )
    sep = "─" * len(header)
    lines = [header, sep]

    key = lambda r: (r.problem, r.dim)
    for (prob_name, dim), group in itertools.groupby(
        sorted(rows, key=key), key=key
    ):
        group_list = list(group)
        npe = next((r for r in group_list if r.solver == "aipe_npe"), None)
        lnn = next((r for r in group_list if r.solver == "aipe_len"), None)
        eg = next((r for r in group_list if r.solver == "eg"), None)
        gda = next((r for r in group_list if r.solver == "gda"), None)

        def _fmt(r: BenchmarkResult | None) -> str:
            if r is None:
                return "N/A"
            ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
            return f"{r.wall_time_mean:.4f} {ci}"

        def _norm(r: BenchmarkResult | None) -> str:
            if r is None or r.normalized_cost is None:
                return "N/A"
            return f"{r.normalized_cost:.2e}"

        npe_str = f"{_fmt(npe)} | {_norm(npe)}"
        lnn_str = f"{_fmt(lnn)} | {_norm(lnn)}"
        eg_str = f"{_fmt(eg)} | {_norm(eg)}"
        gda_str = f"{_fmt(gda)} | {_norm(gda)}"

        npe_gap = npe.final_gap if npe else 0.0
        eg_gap = eg.final_gap if eg else 0.0

        lines.append(
            f"{prob_name:<22} {dim:>4}  {npe_str:>34}  {lnn_str:>34}  {eg_str:>34}  {gda_str:>34}  "
            f"{npe_gap:>8.4f}  {eg_gap:>8.4f}"
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
        lo_j, hi_j = jit["ci"]
        lo_e, hi_e = eager["ci"]
        jit_str = f"{jit['mean']:.4f} [{lo_j:.4f},{hi_j:.4f}]"
        eager_str = f"{eager['mean']:.4f} [{lo_e:.4f},{hi_e:.4f}]"
        lines.append(
            f"{r['name']:<22} {r['dim']:>4}  "
            f"{jit_str:>24}  {eager_str:>24}  "
            f"{r['speedup']:>7.2f}x"
        )

    return "\n".join(lines)