"""CLI entry point for benchmarks.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --quick
    python -m benchmarks.run_all --section speed --dims 2,5,10 --names bilinear,quadratic
    python -m benchmarks.run_all --output csv --seed 42
    python -m benchmarks.run_all --output json:results.json --seed 42 --repeats 5
    python -m benchmarks.run_all --section convergence --names bilinear --dims 5

Sections: speed, jit, scaling, memory, convergence, ablation, all.
"""

from __future__ import annotations

import argparse
import sys
import time

import jax

from benchmarks.export import (
    collect_metadata,
    export_results,
    flatten_ablation_rows,
    flatten_convergence_rows,
    flatten_jit_rows,
    flatten_memory_rows,
    flatten_scaling_rows,
    flatten_speed_rows,
    write_json,
    write_metadata,
)
from benchmarks import config
from benchmarks.problems import get_all_problems, get_problem


def _header(title: str) -> str:
    width = 72
    return f"\n{'═' * width}\n  {title}\n{'═' * width}\n"


def _platform_info() -> str:
    import platform
    return (
        f"Platform: {platform.platform()}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"JAX: {jax.__version__}\n"
        f"Devices: {', '.join(str(d) for d in jax.local_devices())}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark suite for Minimax-AIPE solver.")
    parser.add_argument("--epsilon", type=float, default=config.EPSILON_DEFAULT, help=f"Target duality gap (default: {config.EPSILON_DEFAULT}).")
    parser.add_argument("--dims", type=str, default=None, help="Comma-separated dimensions.")
    parser.add_argument("--quick", action="store_true", help="Reduced set: 1 problem, 1 repeat, dim=2.")
    parser.add_argument("--section", type=str, default="all",
                        choices=["all", "speed", "jit", "scaling", "memory", "convergence", "ablation"],
                        help="Which benchmark section to run.")
    parser.add_argument("--names", type=str, default=None, help="Comma-separated problem names.")
    parser.add_argument("--repeats", type=int, default=None, help=f"Timed repeats (default: {config.N_REPEATS_FULL}, or {config.N_REPEATS_QUICK} with --quick).")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed for all problem constructors.")
    parser.add_argument("--output", type=str, default=None, metavar="FMT[:PATH]",
                        help="Export: 'csv', 'json', 'csv:out.csv', 'json:out.json'.")
    return parser.parse_args()


def _run_speed(problems, epsilon, n_repeats, export_data):
    from benchmarks.time_solve import benchmark_solver_comparison, format_solver_comparison_table
    print(_header("Speed: Solver Comparison"))
    t0 = time.perf_counter()
    speed_rows = benchmark_solver_comparison(problems, epsilon=epsilon, n_repeats=n_repeats)
    elapsed = time.perf_counter() - t0
    print()
    print(format_solver_comparison_table(speed_rows))
    print(f"\n  Total time: {elapsed:.1f}s\n")
    export_data["speed"] = flatten_speed_rows(speed_rows)


def _run_jit(problems, epsilon, n_repeats, export_data):
    from benchmarks.time_solve import benchmark_jit_vs_eager, format_jit_table
    print(_header("JIT vs Eager"))
    jit_problems = problems[:3] if len(problems) > 3 else problems
    jit_rows = []
    for prob in jit_problems:
        name = prob.name or "?"
        dim = prob.dim or prob.problem.dim_x
        print(f"  JIT vs eager: {name} dim={dim} ...")
        r = benchmark_jit_vs_eager(prob, epsilon=epsilon, n_repeats=n_repeats)
        jit_rows.append({"name": name, "dim": dim, **r})
    print()
    print(format_jit_table(jit_rows))
    print()
    export_data["jit_vs_eager"] = flatten_jit_rows(jit_rows)


def _run_scaling(epsilon, n_repeats, seed, export_data):
    from benchmarks.scaling import scale_dimension, scale_rho, scale_condition_number, scale_sparsity, format_scaling_table
    print(_header("Scaling Analysis"))

    scale_dims = [2, 5, 10, 20]

    # Dimension scaling on diagonal_saddle
    print("  diagonal_saddle (dimension):")
    rows = scale_dimension("diagonal_saddle", scale_dims, epsilon=epsilon,
                           n_repeats=max(1, n_repeats // 2), seed=seed)
    print(format_scaling_table(rows))
    export_data.setdefault("scaling_dim", []).extend(flatten_scaling_rows(rows))
    print()

    # Condition number scaling
    print("  ill_conditioned_quadratic (condition number sweep):")
    cond_rows = scale_condition_number("ill_quadratic", kappas=[1e1, 1e2, 1e3, 1e4], dim=10, epsilon=epsilon,
                                       n_repeats=max(1, n_repeats // 2), seed=seed)
    # BUG FIX: Added explicit key_col so groupby doesn't squash the distinct condition numbers
    print(format_scaling_table(cond_rows, key_col="kappa"))
    export_data.setdefault("scaling_cond", []).extend(flatten_scaling_rows(cond_rows))
    print()

    # Dimension scaling on bilinear
    print("  bilinear (dimension):")
    rows = scale_dimension("bilinear", scale_dims, epsilon=epsilon,
                           n_repeats=max(1, n_repeats // 2), seed=seed)
    print(format_scaling_table(rows))
    export_data.setdefault("scaling_dim", []).extend(flatten_scaling_rows(rows))
    print()

    # ρ scaling
    print("  nonzero_rho (ρ sweep):")
    rho_rows = scale_rho([0.1, 0.5, 1.0, 5.0, 10.0], dim=10, epsilon=epsilon,
                         n_repeats=max(1, n_repeats // 2), seed=seed)
    npe_rows = [r for r in rho_rows if r.solver == "aipe_npe"]
    header = f"{'ρ':>8}  {'NPE (s)':>10}  {'NPE cost':>11}"
    print(header)
    print("─" * len(header))
    for r in npe_rows:
        d = r.dim * 2
        cost = float(r.oracle_stats.normalized_cost(d)) if r.oracle_stats else 0
        print(f"{getattr(r, 'rho', 0.0):>8.1f}  {r.wall_time_mean:>10.4f}  {cost:>11.2e}")
    export_data.setdefault("scaling_rho", []).extend(flatten_scaling_rows(rho_rows))
    print()

    # Sparsity scaling
    print("  diagonal_saddle (sparsity sweep):")
    sparsity_rows = scale_sparsity([0.0, 0.3, 0.6, 0.9], dim=100, kappa=1e4, epsilon=epsilon,
                                   n_repeats=max(1, n_repeats // 2), seed=seed)
    npe_rows = [r for r in sparsity_rows if r.solver == "aipe_npe"]
    header = f"{'sparsity':>8}  {'NPE (s)':>10}  {'NPE cost':>11}"
    print(header)
    print("─" * len(header))
    for r in npe_rows:
        d = r.dim * 2
        cost = float(r.oracle_stats.normalized_cost(d)) if r.oracle_stats else 0
        print(f"{getattr(r, 'sparsity', 0.0):>8.2f}  {r.wall_time_mean:>10.4f}  {cost:>11.2e}")
    export_data.setdefault("scaling_sparsity", []).extend(flatten_scaling_rows(sparsity_rows))
    print()
    
def _run_memory(problems, epsilon, export_data):
    from benchmarks.memory import benchmark_memory_scaling, format_memory_table
    print(_header("Memory Usage"))
    mem_problems = problems[:4] if len(problems) > 4 else problems
    mem_results = benchmark_memory_scaling(mem_problems, epsilon=epsilon)
    print(format_memory_table(mem_results))
    print()
    export_data["memory"] = flatten_memory_rows(mem_results)


def _run_convergence(problems, epsilon, export_data):
    from benchmarks.convergence import sweep_epsilon, format_convergence_table, sweep_epsilon_endpoints, format_endpoints_table
    print(_header("Convergence: ε-Sweep"))
    epsilons = config.EPSILON_GRID
    
    all_convergence_rows = []
    all_trace_rows = []
    
    for prob in problems:
        name = prob.name or "?"
        dim = prob.dim or prob.problem.dim_x
        print(f"  {name} dim={dim}:")
        rows = sweep_epsilon(prob, epsilons)
        print(format_convergence_table(rows))
        print()

        trace_rows = sweep_epsilon_endpoints(prob, [epsilons[-1]])
        print(format_endpoints_table(trace_rows))

        all_convergence_rows.extend(flatten_convergence_rows(rows))
        all_trace_rows.extend(flatten_convergence_rows(trace_rows))
        
    export_data.setdefault("convergence", []).extend(all_convergence_rows)
    export_data.setdefault("traces", []).extend(all_trace_rows)


def _run_ablation(problems, epsilon, n_repeats, export_data):
    from benchmarks.ablation import (
        ablation_m_lazy,
        ablation_npe_vs_len,
        ablation_npe_t_factor,
        ablation_no_cubic,
        ablation_no_restart,
        ablation_no_acceleration,
        ablation_fixed_inner,
        ablation_init_comparison,
        format_ablation_m_table,
        format_ablation_t_table,
        format_ablation_no_cubic_table,
        format_ablation_init_table,
        format_ablation_fixed_inner_table,
    )
    print(_header("Ablation"))

    # ── m_lazy sweep ────────────────────────────────────────────────
    prob = problems[0]
    name = prob.name or "?"
    dim = prob.dim or prob.problem.dim_x
    print(f"  m_lazy sweep on {name} dim={dim}:")
    m_rows = ablation_m_lazy(prob, epsilon=epsilon, n_repeats=max(1, n_repeats // 2))
    print(format_ablation_m_table(m_rows))
    export_data.setdefault("ablation_m", []).extend(flatten_ablation_rows(m_rows))
    print()

    # ── T_factor sweep ──────────────────────────────────────────────
    print(f"  T_factor sweep on {name} dim={dim}:")
    t_rows = ablation_npe_t_factor(prob, epsilon=epsilon, n_repeats=max(1, n_repeats // 2))
    print(format_ablation_t_table(t_rows))
    export_data.setdefault("ablation_t", []).extend(flatten_ablation_rows(t_rows))
    print()

    # ── NPE vs LEN head-to-head ─────────────────────────────────────
    print("  NPE vs LEN head-to-head:")
    for prob in problems:
        results = ablation_npe_vs_len(prob, epsilon=epsilon, n_repeats=max(1, n_repeats // 2))
        npe = next((r for r in results if r.solver == "aipe_npe"), None)
        lnn = next((r for r in results if r.solver == "aipe_len"), None)
        if npe and lnn:
            npe_t = npe.wall_time_mean
            len_t = lnn.wall_time_mean
            npe_calls = npe.oracle_stats.crn_calls
            len_calls = lnn.oracle_stats.crn_calls
            print(f"    {npe.problem:18s} dim={npe.dim:>4}  "
                  f"NPE: {npe_t:.4f}s ({npe_calls} calls)  "
                  f"LEN: {len_t:.4f}s ({len_calls} calls)")
            export_data.setdefault("ablation_compare", []).extend(results)
    print()

    # ── NEW: No-cubic regularization ────────────────────────────────
    # No solver changes needed — works whenever problem.rho > 0.
    rho_problem = next(
        (p for p in problems if getattr(p.problem, "rho", None) and p.problem.rho > 0),
        None,
    )
    if rho_problem is not None:
        rname = rho_problem.name or "?"
        rdim = rho_problem.dim or rho_problem.problem.dim_x
        print(f"  No-cubic (ρ=0) ablation on {rname} dim={rdim}:")
        no_cubic_rows = ablation_no_cubic(
            rho_problem, epsilon=epsilon, n_repeats=max(1, n_repeats // 2),
        )
        print(format_ablation_no_cubic_table(no_cubic_rows))
        export_data.setdefault("ablation_no_cubic", []).extend(
            flatten_ablation_rows(no_cubic_rows),
        )
        print()
    else:
        print("  [skip] no_cubic — no problems with ρ > 0 found")
        print()

    # ── NEW: Initialization comparison ──────────────────────────────
    # No solver changes needed.
    init_prob = problems[0]
    iname = init_prob.name or "?"
    idim = init_prob.dim or init_prob.problem.dim_x
    print(f"  Init comparison on {iname} dim={idim}:")
    init_rows = ablation_init_comparison(
        init_prob, epsilon=epsilon, n_repeats=max(1, n_repeats // 2),
    )
    print(format_ablation_init_table(init_rows))
    export_data.setdefault("ablation_init", []).extend(
        flatten_ablation_rows(init_rows),
    )
    print()

    # ── NEW: No-restart ─────────────────────────────────────────────
    # Requires: solve(..., no_restart=True)
    prob0 = problems[0]
    p0name = prob0.name or "?"
    p0dim = prob0.dim or prob0.problem.dim_x
    print(f"  No-restart ablation on {p0name} dim={p0dim}:")
    try:
        no_restart_rows = ablation_no_restart(
            prob0, epsilon=epsilon, n_repeats=max(1, n_repeats // 2),
        )
        for r in no_restart_rows:
            d = r.dim * 2
            cost = float(r.oracle_stats.normalized_cost(d)) if r.oracle_stats else 0
            print(f"    {r.solver:<28s} gap={r.final_gap:.6f}  "
                  f"cost={cost:.2e}  "
                  f"time={r.wall_time_mean:.4f}s")
        export_data.setdefault("ablation_no_restart", []).extend(
            flatten_ablation_rows(no_restart_rows),
        )
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise
        print(f"    [skip] solver does not support the requested ablation yet: {e}")
    print()

    # ── NEW: No-acceleration ────────────────────────────────────────
    # Requires: solve(..., no_acceleration=True)
    print(f"  No-acceleration ablation on {p0name} dim={p0dim}:")
    try:
        no_accel_rows = ablation_no_acceleration(
            prob0, epsilon=epsilon, n_repeats=max(1, n_repeats // 2),
        )
        for r in no_accel_rows:
            d = r.dim * 2
            cost = float(r.oracle_stats.normalized_cost(d)) if r.oracle_stats else 0
            print(f"    {r.solver:<28s} gap={r.final_gap:.6f}  "
                  f"cost={cost:.2e}  "
                  f"time={r.wall_time_mean:.4f}s")
        export_data.setdefault("ablation_no_accel", []).extend(
            flatten_ablation_rows(no_accel_rows),
        )
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise
        print(f"    [skip] solver does not support the requested ablation yet: {e}")
    print()

    # ── NEW: Fixed inner iterations ─────────────────────────────────
    # Requires: solve(..., fixed_inner_iters=N)
    print(f"  Fixed inner-iter sweep on {p0name} dim={p0dim}:")
    try:
        fixed_inner_rows = ablation_fixed_inner(
            prob0, epsilon=epsilon, n_repeats=max(1, n_repeats // 2),
        )
        print(format_ablation_fixed_inner_table(fixed_inner_rows))
        export_data.setdefault("ablation_fixed_inner", []).extend(
            flatten_ablation_rows(fixed_inner_rows),
        )
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise
        print(f"    [skip] solver does not support the requested ablation yet: {e}")
    print()


def main():
    args = _parse_args()

    dims = [int(d.strip()) for d in args.dims.split(",")] if args.dims else None
    names = [n.strip() for n in args.names.split(",")] if args.names else None
    n_repeats = args.repeats or (config.N_REPEATS_QUICK if args.quick else config.N_REPEATS_FULL)
    epsilon = args.epsilon

    output_fmt, output_path = None, None
    if args.output:
        parts = args.output.split(":", 1)
        output_fmt = parts[0].strip().lower()
        if len(parts) > 1 and parts[1].strip():
            output_path = parts[1].strip()
        if output_fmt not in ("csv", "json"):
            print(f"Error: --output format must be 'csv' or 'json', got {output_fmt!r}")
            sys.exit(1)

    # ── Header ──────────────────────────────────────────────────────
    print(_header("Minimax-AIPE Benchmark Suite"))
    print(_platform_info())
    print(f"Epsilon: {epsilon}")
    print(f"Repeats: {n_repeats}")
    print(f"Quick mode: {args.quick}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    if output_fmt:
        print(f"Output: {output_fmt}" + (f" → {output_path}" if output_path else " → stdout"))
    print()

    # ── Problem zoo ─────────────────────────────────────────────────
    if args.quick:
        problems = get_all_problems(dims=[2], names=["bilinear"], seed=args.seed)
    else:
        problems = get_all_problems(dims=dims, names=names, seed=args.seed)

    if not problems:
        print("No problems to benchmark. Check --dims / --names.")
        sys.exit(1)

    print(f"Problems ({len(problems)}):")
    for p in problems:
        print(f"  {p.name:22s}  dim={p.dim}")
    print()

    # ── Metadata ────────────────────────────────────────────────────
    metadata = collect_metadata(
        epsilon=epsilon, n_repeats=n_repeats, seed=args.seed,
        section=args.section, dims=args.dims, names=args.names,
        quick=args.quick, problems=problems,
    )
    write_metadata(metadata)
    print()

    # ── Collect results ─────────────────────────────────────────────
    export_data: dict[str, list[dict]] = {"metadata": [metadata]}

    # ── Sections ────────────────────────────────────────────────────
    if args.section in ("all", "speed"):
        _run_speed(problems, epsilon, n_repeats, export_data)

    if args.section in ("all", "jit"):
        _run_jit(problems, epsilon, n_repeats, export_data)

    if args.section in ("all", "scaling"):
        _run_scaling(epsilon, n_repeats, args.seed, export_data)

    if args.section in ("all", "memory"):
        _run_memory(problems, epsilon, export_data)

    if args.section in ("all", "convergence"):
        _run_convergence(problems, epsilon, export_data)

    if args.section in ("all", "ablation"):
        _run_ablation(problems, epsilon, n_repeats, export_data)

    # ── Export ──────────────────────────────────────────────────────
    if output_fmt:
        print()
        # FIX: Ensure destination directory exists before exporting to prevent open() crashes
        if output_path:
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
        export_results(export_data, output_fmt, output_path)

    print(_header("Done"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
