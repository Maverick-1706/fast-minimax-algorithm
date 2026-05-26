"""Shared cache for deterministic solve() calls used across test modules."""

from __future__ import annotations

from typing import Any

from minimax_aipe import solve


_SOLVE_CACHE: dict[tuple[Any, ...], Any] = {}


def _problem_key(problem) -> tuple[Any, ...]:
    return (
        getattr(problem, "name", None),
        getattr(problem, "dim", None),
        getattr(getattr(problem, "meta", None), "seed", None),
        float(problem.problem.D_x),
        float(problem.problem.D_y),
        float(problem.problem.rho or 0.0),
        float(problem.problem.ell or 0.0),
    )


def cached_solve(
    benchmark_problem,
    epsilon: float,
    *,
    gamma: float | None = None,
    M_saddle: str = "npe",
    m_lazy: int = -1,
    npe_T_factor: float = 1.0,
    verbose: bool = False,
):
    key = (
        _problem_key(benchmark_problem),
        float(epsilon),
        None if gamma is None else float(gamma),
        M_saddle,
        int(m_lazy),
        float(npe_T_factor),
        bool(verbose),
    )
    if key not in _SOLVE_CACHE:
        _SOLVE_CACHE[key] = solve(
            benchmark_problem.problem,
            epsilon=epsilon,
            gamma=gamma,
            M_saddle=M_saddle,
            m_lazy=m_lazy,
            npe_T_factor=npe_T_factor,
            verbose=verbose,
        )
    return _SOLVE_CACHE[key]
