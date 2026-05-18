"""Component ablation experiments.

Measures the isolated effect of individual solver components:
  - NPE vs LEN (Hessian reuse)
  - Lazy vs fresh Hessians (m_lazy sweep)
  - Warm-starting impact
  - Early stopping impact (npe_T_factor sweep)
  - Cubic regularization (ρ=0 vs original)
  - Restart schedule (single-shot vs iterative)
  - Nesterov acceleration (plain gradient vs accelerated)
  - Inner-iteration budget (fixed vs adaptive)
  - Initialization strategy (heuristic vs zero vs random)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import replace

import jax
import jax.numpy as jnp

from minimax_aipe import solve
from minimax_aipe.problem import BenchmarkProblem
from benchmarks import config
from benchmarks.results import BenchmarkResult
from benchmarks.stats import bootstrap_ci


# ──────────────────────────────────────────────────────────────────────
# Existing experiments (unchanged)
# ──────────────────────────────────────────────────────────────────────


def ablation_m_lazy(
    prob,
    epsilon: float | None = None,
    m_values: list[int] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time and oracle calls as m_lazy varies.

    m=1 is equivalent to fresh Hessians (NPE).  Larger m reuses more.

    Parameters
    ----------
    prob : BenchmarkProblem
        From the problem zoo.
    epsilon : float or None
        Target gap.  Defaults to ``config.EPSILON_DEFAULT``.
    m_values : list[int]
        Hessian reuse intervals to test.
    n_repeats : int or None
        Timed runs per configuration.  Defaults to ``config.N_REPEATS_SCALING``.

    Returns
    -------
    list[BenchmarkResult]
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    m_values = m_values or [1, 3, 5, 10, 20]
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

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
        rows.append(BenchmarkResult(
            solver="aipe_len",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(times),
            wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
            ci=ci,
            oracle_stats=result.oracle_stats,
            converged=result.converged,
            gap_achieved=result.gap <= epsilon,
            final_gap=float(result.gap),
            iterations=result.iterations,
            m_lazy=m,
            normalized_cost=result.oracle_stats.normalized_cost(d),
        ))

    return rows


def ablation_npe_t_factor(
    prob,
    epsilon: float | None = None,
    t_factors: list[float] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Measure solve time and oracle calls as npe_T_factor varies."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    t_factors = t_factors or [0.5, 1.0, 1.5, 2.0, 3.0]
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

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
        rows.append(BenchmarkResult(
            solver="aipe_npe",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(times),
            wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
            ci=ci,
            oracle_stats=result.oracle_stats,
            converged=result.converged,
            gap_achieved=result.gap <= epsilon,
            final_gap=float(result.gap),
            iterations=result.iterations,
            npe_T_factor=tf,
            normalized_cost=result.oracle_stats.normalized_cost(d),
        ))

    return rows


def ablation_npe_vs_len(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Head-to-head NPE vs LEN on a single problem."""
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), f"Expected BenchmarkProblem, got {type(prob)}"
    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    results = []

    for M_saddle in ("npe", "len"):
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
        results.append(BenchmarkResult(
            solver=f"aipe_{M_saddle}",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            wall_time_mean=statistics.mean(times),
            wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
            ci=ci,
            oracle_stats=result.oracle_stats,
            converged=result.converged,
            gap_achieved=result.gap <= epsilon,
            final_gap=float(result.gap),
            iterations=result.iterations,
            normalized_cost=result.oracle_stats.normalized_cost(d),
        ))

    return results


# ──────────────────────────────────────────────────────────────────────
# Existing formatters (unchanged)
# ──────────────────────────────────────────────────────────────────────


def format_ablation_m_table(rows: list[BenchmarkResult]) -> str:
    """Format m_lazy ablation as a text table."""
    header = f"{'Problem':<18} {'Dim':>4}  {'m_lazy':>6}  {'Time (s)':>24}  {'Calls':>6}  {'Gap':>10}"
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {r.m_lazy:>6}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  {r.oracle_stats.crn_calls:>6}  {r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_t_table(rows: list[BenchmarkResult]) -> str:
    """Format npe_T_factor ablation as a text table."""
    header = f"{'Problem':<18} {'Dim':>4}  {'T_factor':>8}  {'Time (s)':>24}  {'Calls':>6}  {'Gap':>10}"
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        tf = r.npe_T_factor or 0.0
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {tf:>8.1f}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  {r.oracle_stats.crn_calls:>6}  {r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# NEW: Private timing helper (DRY — used by every new experiment)
# ──────────────────────────────────────────────────────────────────────


def _time_solve_loop(
    problem,
    n_repeats: int,
    kwargs: dict,
) -> tuple[list[float], object]:
    """Run ``solve(problem, **kwargs)`` with warmup + n_repeats timed calls.

    Returns
    -------
    times : list[float]
        Wall-clock seconds per timed call.
    result : object
        The result of the *last* timed call (for stats extraction).
    """
    # Warmup — absorb JIT compilation
    _ = solve(problem, **kwargs)

    times: list[float] = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = solve(problem, **kwargs)
        times.append(time.perf_counter() - t0)

    return times, result


def _build_result(
    *,
    solver: str,
    problem: str,
    dim: int,
    epsilon: float,
    times: list[float],
    result,
    d: int,
    extra: dict | None = None,
) -> BenchmarkResult:
    """Construct a BenchmarkResult from timing data."""
    ci = bootstrap_ci(times)
    kwargs = dict(
        solver=solver,
        problem=problem,
        dim=dim,
        epsilon=epsilon,
        wall_time_mean=statistics.mean(times),
        wall_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        ci=ci,
        oracle_stats=result.oracle_stats,
        converged=result.converged,
        gap_achieved=result.gap <= epsilon,
        final_gap=float(result.gap),
        iterations=result.iterations,
        normalized_cost=result.oracle_stats.normalized_cost(d),
    )
    if extra:
        kwargs.update(extra)
    return BenchmarkResult(**kwargs)


# ──────────────────────────────────────────────────────────────────────
# 3a: No-cubic regularization (ρ = 0)
# ──────────────────────────────────────────────────────────────────────


def ablation_no_cubic(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: force ρ=0 in solver when problem has ρ > 0.

    Creates a modified problem copy with ``rho=0.0`` and runs both NPE
    and LEN.  The difference vs the original problem measures the cubic
    regularization contribution.

    No solver changes required — uses ``dataclasses.replace`` on the
    underlying :class:`MinimaxProblem`.
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), (
        f"Expected BenchmarkProblem, got {type(prob)}"
    )

    original_problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or original_problem.dim_x
    d = original_problem.dim_x + original_problem.dim_y

    # Create a rho=0 variant of the problem
    problem_no_cubic = replace(original_problem, rho=0.0)

    results = []
    for M_saddle in ("npe", "len"):
        kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": prob.z0}
        if M_saddle == "len":
            kwargs["m_lazy"] = 5

        times, result = _time_solve_loop(problem_no_cubic, n_repeats, kwargs)
        results.append(_build_result(
            solver=f"aipe_{M_saddle}",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────
# 3b: No-restart (single outer loop)
#
#   SOLVER DEPENDENCY:
#     solve(..., no_restart=True)  — runs exactly one outer iteration
# ──────────────────────────────────────────────────────────────────────


def ablation_no_restart(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: single-shot AIPE with no restarts.

    Requires ``solve(..., no_restart=True)`` support.  When set, the
    solver runs exactly one outer loop and returns.  Compare against
    the default (restarts enabled) to measure restart benefit.
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), (
        f"Expected BenchmarkProblem, got {type(prob)}"
    )

    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    results = []
    for M_saddle in ("npe", "len"):
        kwargs = {
            "epsilon": epsilon,
            "M_saddle": M_saddle,
            "z0": prob.z0,
            "no_restart": True,
        }
        if M_saddle == "len":
            kwargs["m_lazy"] = 5

        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        results.append(_build_result(
            solver=f"aipe_{M_saddle}_no_restart",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────
# 3c: No-acceleration (plain gradient instead of Nesterov)
#
#   SOLVER DEPENDENCY:
#     solve(..., no_acceleration=True)
# ──────────────────────────────────────────────────────────────────────


def ablation_no_acceleration(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: disable Nesterov acceleration in the outer loop.

    Requires ``solve(..., no_acceleration=True)`` support.  When set, the
    solver replaces the acceleration step with a plain gradient step.
    Middle and inner loops run normally.
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), (
        f"Expected BenchmarkProblem, got {type(prob)}"
    )

    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    results = []
    for M_saddle in ("npe", "len"):
        kwargs = {
            "epsilon": epsilon,
            "M_saddle": M_saddle,
            "z0": prob.z0,
            "no_acceleration": True,
        }
        if M_saddle == "len":
            kwargs["m_lazy"] = 5

        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        results.append(_build_result(
            solver=f"aipe_{M_saddle}_no_accel",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
        ))

    return results


# ──────────────────────────────────────────────────────────────────────
# 3d: Fixed inner-iteration budget sweep
#
#   SOLVER DEPENDENCY:
#     solve(..., fixed_inner_iters=N)
# ──────────────────────────────────────────────────────────────────────


def ablation_fixed_inner(
    prob,
    epsilon: float | None = None,
    inner_iters_list: list[int] | None = None,
    n_repeats: int | None = None,
) -> list[BenchmarkResult]:
    """Ablation: fixed inner-iteration budget sweep.

    Requires ``solve(..., fixed_inner_iters=N)`` support.  Varies the
    number of inner NPE iterations at a fixed ε to show the tradeoff
    between inner-loop accuracy and outer-loop progress.
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    if inner_iters_list is None:
        inner_iters_list = [1, 5, 10, 20, 50, 100]
    assert isinstance(prob, BenchmarkProblem), (
        f"Expected BenchmarkProblem, got {type(prob)}"
    )

    problem = prob.problem
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x
    d = problem.dim_x + problem.dim_y

    rows: list[BenchmarkResult] = []
    for n_inner in inner_iters_list:
        kwargs = {
            "epsilon": epsilon,
            "M_saddle": "npe",
            "z0": prob.z0,
            "fixed_inner_iters": n_inner,
        }

        times, result = _time_solve_loop(problem, n_repeats, kwargs)
        rows.append(_build_result(
            solver="aipe_npe_fixed_inner",
            problem=name,
            dim=dim,
            epsilon=epsilon,
            times=times,
            result=result,
            d=d,
            extra={"iterations": n_inner},
        ))

    return rows


# ──────────────────────────────────────────────────────────────────────
# 3e: Random-vs-zero initialization
#
#   NO SOLVER CHANGES NEEDED
# ──────────────────────────────────────────────────────────────────────


def ablation_init_comparison(
    prob,
    epsilon: float | None = None,
    n_repeats: int | None = None,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Ablation: compare initialization strategies.

    Tests three initializations:
      - heuristic: ``prob.z0`` (the default deterministic init)
      - zero: ``jnp.zeros(total_dim)``
      - random: scaled normal draw

    No solver changes required.
    """
    if epsilon is None:
        epsilon = config.EPSILON_DEFAULT
    if n_repeats is None:
        n_repeats = config.N_REPEATS_SCALING
    assert isinstance(prob, BenchmarkProblem), (
        f"Expected BenchmarkProblem, got {type(prob)}"
    )

    problem = prob.problem
    d = problem.dim_x + problem.dim_y
    name = prob.name or "?"
    dim = prob.dim or problem.dim_x

    key = jax.random.PRNGKey(seed)
    random_z0 = jax.random.normal(key, (d,)) * 0.5
    zero_z0 = jnp.zeros(d)
    heuristic_z0 = prob.z0

    init_configs = [
        ("heuristic", heuristic_z0),
        ("zero", zero_z0),
        ("random", random_z0),
    ]

    results: list[BenchmarkResult] = []
    for init_name, z0 in init_configs:
        for M_saddle in ("npe", "len"):
            kwargs = {"epsilon": epsilon, "M_saddle": M_saddle, "z0": z0}
            if M_saddle == "len":
                kwargs["m_lazy"] = 5

            times, result = _time_solve_loop(problem, n_repeats, kwargs)
            results.append(_build_result(
                solver=f"aipe_{M_saddle}",
                problem=f"{name}_{init_name}",
                dim=dim,
                epsilon=epsilon,
                times=times,
                result=result,
                d=d,
            ))

    return results


# ──────────────────────────────────────────────────────────────────────
# NEW formatters
# ──────────────────────────────────────────────────────────────────────


def format_ablation_no_cubic_table(rows: list[BenchmarkResult]) -> str:
    """Compare with/without cubic regularization.

    Groups rows by solver and shows ρ=0 performance side by side.
    """
    if not rows:
        return "(no data)"

    header = (
        f"{'Solver':<22} {'Problem':<18} {'Dim':>4}  "
        f"{'Time (s)':>24}  {'Calls':>6}  {'Iters':>6}  {'Gap':>10}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        lines.append(
            f"{r.solver:<22} {r.problem:<18} {r.dim:>4}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  "
            f"{r.oracle_stats.oracle_calls:>6}  {r.iterations:>6}  "
            f"{r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_init_table(rows: list[BenchmarkResult]) -> str:
    """Compare initialization strategies.

    Rows are grouped by init variant; solver name appears in the problem
    label as ``<name>_<init>``.
    """
    if not rows:
        return "(no data)"

    header = (
        f"{'Variant':<30} {'Solver':<22} {'Dim':>4}  "
        f"{'Time (s)':>24}  {'Calls':>6}  {'Iters':>6}  {'Gap':>10}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        lines.append(
            f"{r.problem:<30} {r.solver:<22} {r.dim:>4}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  "
            f"{r.oracle_stats.oracle_calls:>6}  {r.iterations:>6}  "
            f"{r.final_gap:>10.6f}"
        )
    return "\n".join(lines)


def format_ablation_fixed_inner_table(rows: list[BenchmarkResult]) -> str:
    """Fixed inner-iteration sweep table.

    Shows inner budget, wall time, oracle calls, outer iterations, and
    final gap to reveal the accuracy/cost tradeoff.
    """
    if not rows:
        return "(no data)"

    header = (
        f"{'Problem':<18} {'Dim':>4}  {'Inner':>6}  "
        f"{'Time (s)':>24}  {'Calls':>6}  {'Outer':>6}  {'Gap':>10}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in rows:
        ci = f"[{r.ci[0]:.4f},{r.ci[1]:.4f}]"
        # For fixed-inner rows, npe_T_factor is unused; the relevant
        # parameter is stored in iterations (overridden by extra=).
        lines.append(
            f"{r.problem:<18} {r.dim:>4}  {r.iterations:>6}  "
            f"{r.wall_time_mean:>8.4f} {ci:>16}  "
            f"{r.oracle_stats.oracle_calls:>6}  {r.iterations:>6}  "
            f"{r.final_gap:>10.6f}"
        )
    return "\n".join(lines)
