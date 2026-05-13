"""CLI entry point for benchmarks.

Usage:
    python -m benchmarks.run_all [--epsilon 0.01] [--dims 2,5,10] [--quick] [--section speed|memory|jit|scaling]
"""

from __future__ import annotations

import argparse
import sys
import time

import jax

from benchmarks.problems import get_all_problems, list_problems
from benchmarks.time_solve import (
    benchmark_jit_vs_eager,
    benchmark_scaling,
    benchmark_solver_comparison,
    format_jit_table,
    format_scaling_table,
    format_solver_comparison_table,
)
from benchmarks.memory_solve import (
    benchmark_memory_scaling,
    format_memory_table,
)


def _header(title: str) -> str:
    width = 72
    return f"\n{'═' * width}\n  {title}\n{'═' * width}\n"


def _platform_info() -> str:
    import platform
    jax_version = jax.__version__
    devices = [str(d) for d in jax.local_devices()]
    return (
        f"Platform: {platform.platform()}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"JAX: {jax_version}\n"
        f"Devices: {', '.join(devices)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suite for Minimax-AIPE solver."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Target duality gap (default: 0.01).",
    )
    parser.add_argument(
        "--dims", type=str, default=None,
        help="Comma-separated dimensions to test (default: per-problem defaults).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced set: 2 problems, 1 repeat, smaller dims.",
    )
    parser.add_argument(
        "--section", type=str, default="all",
        choices=["all", "speed", "jit", "scaling", "memory"],
        help="Which benchmark section to run.",
    )
    parser.add_argument(
        "--names", type=str, default=None,
        help="Comma-separated problem names (default: all).",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="Number of timed repeats (default: 3, or 1 with --quick).",
    )
    args = parser.parse_args()

    # ── Parse arguments ─────────────────────────────────────────────
    dims = None
    if args.dims:
        dims = [int(d.strip()) for d in args.dims.split(",")]

    names = None
    if args.names:
        names = [n.strip() for n in args.names.split(",")]

    n_repeats = args.repeats or (1 if args.quick else 3)
    epsilon = args.epsilon

    # ── Header ──────────────────────────────────────────────────────
    print(_header("Minimax-AIPE Benchmark Suite"))
    print(_platform_info())
    print(f"Epsilon: {epsilon}")
    print(f"Repeats: {n_repeats}")
    print(f"Quick mode: {args.quick}")
    print()

    # ── Problem zoo ─────────────────────────────────────────────────
    if args.quick:
        problems = get_all_problems(dims=[2], names=["bilinear"])
    else:
        problems = get_all_problems(dims=dims, names=names)

    if not problems:
        print("No problems to benchmark. Check --dims / --names.")
        sys.exit(1)

    print(f"Problems ({len(problems)}):")
    for p in problems:
        print(f"  {p['name']:22s}  dim={p['dim']}")
    print()

    # ── Speed benchmarks ────────────────────────────────────────────
    if args.section in ("all", "speed"):
        print(_header("Speed: Solver Comparison"))
        t0 = time.perf_counter()
        rows = benchmark_solver_comparison(problems, epsilon=epsilon, n_repeats=n_repeats)
        elapsed = time.perf_counter() - t0
        print()
        print(format_solver_comparison_table(rows))
        print(f"\n  Total time: {elapsed:.1f}s\n")

    # ── JIT vs Eager ────────────────────────────────────────────────
    if args.section in ("all", "jit"):
        print(_header("JIT vs Eager"))
        jit_problems = problems[:3] if len(problems) > 3 else problems
        jit_rows = []
        for prob_dict in jit_problems:
            name = prob_dict.get("name", "?")
            dim = prob_dict.get("dim", prob_dict["problem"].dim_x)
            print(f"  JIT vs eager: {name} dim={dim} ...")
            r = benchmark_jit_vs_eager(prob_dict, epsilon=epsilon, n_repeats=n_repeats)
            jit_rows.append({"name": name, "dim": dim, **r})
        print()
        print(format_jit_table(jit_rows))
        print()

    # ── Scaling ─────────────────────────────────────────────────────
    if args.section in ("all", "scaling"):
        print(_header("Scaling Analysis"))
        scale_dims = [2, 5] if args.quick else [2, 5, 10, 20]
        for ptype in ["bilinear", "quadratic"]:
            print(f"  {ptype}:")
            scaling_rows = benchmark_scaling(scale_dims, epsilon=epsilon, problem_type=ptype, n_repeats=max(1, n_repeats // 2))
            print(format_scaling_table(scaling_rows))
            print()

    # ── Memory ──────────────────────────────────────────────────────
    if args.section in ("all", "memory"):
        print(_header("Memory Usage"))
        mem_problems = problems[:4] if len(problems) > 4 else problems
        mem_results = benchmark_memory_scaling(mem_problems, epsilon=epsilon)
        print(format_memory_table(mem_results))
        print()

    # ── Done ────────────────────────────────────────────────────────
    print(_header("Done"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
