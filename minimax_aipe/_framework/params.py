"""Parameter computation and common setup helpers for the framework."""

from __future__ import annotations

from dataclasses import dataclass
import math
from math import ceil, log2
from typing import Optional

import jax.numpy as jnp
from jax import Array

from minimax_aipe._precision import (
    ABS_TOL as _ABS_TOL,
    GAP_FLOOR as _GAP_FLOOR,
)
from minimax_aipe.gap import estimate_gap
from minimax_aipe.problem import MinimaxProblem


@dataclass(frozen=True)
class _LoopParams:
    T_outer: int
    S_outer: int
    T_middle: int
    S_middle: int
    T_inner: int
    S_inner: int
    zeta_1: float
    zeta_2: float
    zeta_3: float
    m_lazy: int = 5


def _validate_solver_inputs(problem: MinimaxProblem, epsilon: float, M_saddle: str) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if M_saddle not in ("npe", "len"):
        raise ValueError("M_saddle must be either 'npe' or 'len'")


def _normalize_initial_z(problem: MinimaxProblem, z0: Optional[Array]) -> Array:
    if z0 is None:
        return _initial_z(problem)
    z0_arr = jnp.asarray(z0)
    expected = problem.dim_x + problem.dim_y
    if z0_arr.shape != (expected,):
        raise ValueError(f"z0 must have shape ({expected},), got {z0_arr.shape}")
    x0, y0 = _split(problem, z0_arr)
    return jnp.concatenate([problem.project_x(x0), problem.project_y(y0)])


def _default_gamma(problem: MinimaxProblem, gamma: float | None) -> float:
    if gamma is not None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        return float(gamma)
    rho = float(problem.rho or 0.0)
    if rho > 0:
        return rho
    import math

    ell = max(float(problem.ell or 0.0), 1.0)
    return max(1.0, math.sqrt(ell))


def _resolve_gamma(
    problem: MinimaxProblem,
    gamma: float | None,
    M_saddle: str,
    m_lazy: int,
) -> float:
    if gamma is not None:
        return float(gamma)
    if M_saddle == "len":
        rho = float(problem.rho or 0.0)
        if rho > 0:
            return rho / max(m_lazy**0.5, 1.0) if m_lazy > 0 else rho
        import math

        ell = max(float(problem.ell or 0.0), 1.0)
        base_gamma = max(1.0, math.sqrt(ell))
        return base_gamma / max(m_lazy**0.5, 1.0) if m_lazy > 0 else base_gamma
    return _default_gamma(problem, None)


def _compute_loop_params(
    problem: MinimaxProblem,
    epsilon: float,
    gamma: float,
    npe_T_factor: float = 0.5,
    m_lazy: int = -1,
    no_restart: bool = False,
    fixed_inner_iters: Optional[int] = None,
) -> _LoopParams:
    """Compute iteration counts and accuracy parameters for all three loops."""
    D = max(_diameter(problem), _ABS_TOL)
    ell = max(_ell(problem), _ABS_TOL)
    rho = float(problem.rho or 0.0)

    mu_y = epsilon / (2.0 * max(_diam(problem.D_y), _ABS_TOL) ** 3)
    zeta_1_raw = mu_y * epsilon**2 / (147.0 * ell**3 * D**2 + _ABS_TOL)
    zeta_1_floor = min(epsilon, _ABS_TOL)
    zeta_1 = min(epsilon, max(zeta_1_raw, zeta_1_floor))
    zeta_2 = min(zeta_1 * 0.2, 1e-3)
    zeta_3 = min(zeta_2 * 0.2, 1e-4)

    _S_CAP = 12
    S = max(1, min(
        int(ceil(log2(max(D / max(epsilon, _ABS_TOL), 2.0)))),
        _S_CAP,
    ))
    if no_restart:
        S = 1

    mu_x = epsilon / (2.0 * max(_diam(problem.D_x), _ABS_TOL) ** 3)
    T_outer = max(1, min(200, int(ceil(
        npe_T_factor * (gamma / max(mu_x, _ABS_TOL)) ** (2.0 / 7.0)
    ))))
    T_middle = max(1, min(200, int(ceil(
        npe_T_factor * (gamma / max(mu_y, _ABS_TOL)) ** (2.0 / 7.0)
    ))))

    rho_h = rho + 2 * gamma
    npe_gamma = 2.0 * rho_h
    mu_inner = gamma / 2.0
    T_inner = max(1, min(200, int(ceil(
        npe_T_factor * (npe_gamma / max(mu_inner, _ABS_TOL)) ** (2.0 / 3.0)
    ))))

    S_inner_default = max(1, min(S, _S_CAP))
    if fixed_inner_iters is not None:
        T_inner = max(1, fixed_inner_iters)
        S_inner_default = 1
    if no_restart:
        S_inner_default = 1

    if m_lazy <= 0:
        dim_total = problem.dim_x + problem.dim_y
        eff_cond = ell / max(gamma, _ABS_TOL)
        adaptive_m = int(max(3, (dim_total ** 0.5) * max(1.0, 0.5 * log2(eff_cond + 1))))
        m_lazy = max(1, min(adaptive_m, 50))
    else:
        m_lazy = max(1, m_lazy)

    m_lazy = max(1, min(m_lazy, T_inner))

    return _LoopParams(
        T_outer=T_outer,
        S_outer=S,
        T_middle=T_middle,
        S_middle=1 if no_restart else max(1, min(S, _S_CAP)),
        T_inner=T_inner,
        S_inner=S_inner_default,
        zeta_1=zeta_1,
        zeta_2=zeta_2,
        zeta_3=zeta_3,
        m_lazy=m_lazy,
    )


def _has_exact_gap(problem: MinimaxProblem) -> bool:
    duality_gap = getattr(problem, "duality_gap", None)
    if duality_gap is None:
        return False
    duality_gap_fn = getattr(duality_gap, "__func__", duality_gap)
    return duality_gap_fn is not MinimaxProblem.duality_gap


def _estimated_gap_budget(problem: MinimaxProblem, epsilon: float) -> tuple[int, int]:
    D = _diameter(problem)
    ell = max(_ell(problem), _ABS_TOL)
    kappa_proxy = ell * D * D / max(epsilon, _GAP_FLOOR)
    # Cap the proxy to avoid exploding gap-estimation work on hard problems.
    kappa_proxy = min(kappa_proxy, 1000.0)
    num_steps = max(10000, min(20000, int(200 * kappa_proxy ** 0.5)))
    num_restarts = max(10, min(14, int(4 * (1 + D) ** 0.5)))
    return num_restarts, num_steps


def _safe_gap(problem: MinimaxProblem, x: Array, y: Array, epsilon: float) -> float:
    if _has_exact_gap(problem):
        gap = problem.duality_gap(x, y)
        if hasattr(gap, "block_until_ready"):
            gap.block_until_ready()
        gap_value = float(gap)
        if not math.isfinite(gap_value):
            return float("inf")
        return max(0.0, gap_value)

    num_restarts, num_steps = _estimated_gap_budget(problem, epsilon)
    gap = estimate_gap(
        problem, x, y,
        num_restarts=num_restarts, num_steps=num_steps, lr=None,
    )
    if hasattr(gap, "block_until_ready"):
        gap.block_until_ready()
    gap_value = float(gap)
    if not math.isfinite(gap_value):
        return float("inf")
    return max(0.0, gap_value)


def _initial_z(problem: MinimaxProblem) -> Array:
    x0 = problem.project_x(jnp.zeros(problem.dim_x))
    y0 = problem.project_y(jnp.zeros(problem.dim_y))
    return jnp.concatenate([x0, y0])


def _split(problem: MinimaxProblem, z: Array) -> tuple[Array, Array]:
    return z[: problem.dim_x], z[problem.dim_x :]


def _diam(value: float | None) -> float:
    return float(value) if value is not None and value > 0 else 1.0


def _diameter(problem: MinimaxProblem) -> float:
    return max(_diam(problem.D_x), _diam(problem.D_y))


def _ell(problem: MinimaxProblem) -> float:
    return float(problem.ell) if problem.ell is not None and problem.ell > 0 else 1.0
