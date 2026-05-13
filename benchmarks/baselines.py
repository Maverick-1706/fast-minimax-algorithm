"""Baseline solvers for benchmark comparison.

Provides simple first-order methods (extragradient, gradient descent-ascent)
that serve as reference points for the Minimax-AIPE solver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe.problem import MinimaxProblem


@dataclass
class BaselineResult:
    """Result from a baseline solver run."""

    x: Array
    y: Array
    gap: float
    iterations: int
    wall_time: float
    converged: bool
    final_residual: float


def _estimate_gap_simple(problem: MinimaxProblem, x: Array, y: Array, steps: int = 200) -> float:
    """Cheap gap estimate: gradient ascent on y, descent on x."""
    lr = 0.5 / max(float(problem.ell) if problem.ell else 1.0, 1.0)

    # max_y f(x, y)
    y_cur = problem.project_y(jnp.zeros(problem.dim_y))
    for _ in range(steps):
        _, gy_neg = problem.grad_f(x, y_cur)
        y_cur = problem.project_y(y_cur - lr * gy_neg)
    max_f = float(problem.f(x, y_cur))

    # min_x f(x, y)
    x_cur = problem.project_x(jnp.zeros(problem.dim_x))
    for _ in range(steps):
        gx, _ = problem.grad_f(x_cur, y)
        x_cur = problem.project_x(x_cur - lr * gx)
    min_f = float(problem.f(x_cur, y))

    return max(0.0, max_f - min_f)


def run_extragradient(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 10_000,
    tol: float = 1e-8,
) -> BaselineResult:
    """Vanilla extragradient method for monotone VIs.

    Algorithm:
        z_{1/2} = proj(z - η F(z))
        z_{t+1}  = proj(z - η F(z_{1/2}))

    Step size η = 1/(2ℓ).

    Parameters
    ----------
    problem : MinimaxProblem
    epsilon : float
        Target gap (used for convergence check every 100 iters).
    max_iters : int
        Maximum iterations.
    tol : float
        Residual tolerance ‖F(z)‖.

    Returns
    -------
    BaselineResult
    """
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 1.0 / (2.0 * ell)

    x0 = problem.project_x(jnp.zeros(problem.dim_x))
    y0 = problem.project_y(jnp.zeros(problem.dim_y))
    z = jnp.concatenate([x0, y0])

    def proj(z_arr):
        x, y = z_arr[: problem.dim_x], z_arr[problem.dim_x :]
        return jnp.concatenate([problem.project_x(x), problem.project_y(y)])

    F = problem.operator_F

    t0 = time.perf_counter()

    for i in range(max_iters):
        Fz = F(z)
        z_half = proj(z - eta * Fz)
        F_half = F(z_half)
        z = proj(z - eta * F_half)

        if i % 100 == 0:
            residual = float(jnp.linalg.norm(F_half))
            if residual < tol:
                break

    wall_time = time.perf_counter() - t0
    x_out, y_out = z[: problem.dim_x], z[problem.dim_x :]
    residual = float(jnp.linalg.norm(F(z)))
    gap = _estimate_gap_simple(problem, x_out, y_out)

    return BaselineResult(
        x=x_out,
        y=y_out,
        gap=gap,
        iterations=i + 1,
        wall_time=wall_time,
        converged=gap <= epsilon,
        final_residual=residual,
    )


def run_gda(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 10_000,
    tol: float = 1e-8,
) -> BaselineResult:
    """Simultaneous gradient descent-ascent (GDA).

    Algorithm:
        x_{t+1} = proj(x - η ∇_x f)
        y_{t+1} = proj(y + η ∇_y f)

    Step size η = 1/ℓ.

    Parameters
    ----------
    problem : MinimaxProblem
    epsilon : float
        Target gap.
    max_iters : int
        Maximum iterations.
    tol : float
        Residual tolerance.

    Returns
    -------
    BaselineResult
    """
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 1.0 / ell

    x = problem.project_x(jnp.zeros(problem.dim_x))
    y = problem.project_y(jnp.zeros(problem.dim_y))

    t0 = time.perf_counter()

    for i in range(max_iters):
        gx, gy_neg = problem.grad_f(x, y)
        x = problem.project_x(x - eta * gx)
        # gy_neg = -∇_y f, so gradient ascent on y: y - η * gy_neg = y + η * ∇_y f
        y = problem.project_y(y - eta * gy_neg)

        if i % 100 == 0:
            Fz = jnp.concatenate([gx, gy_neg])
            residual = float(jnp.linalg.norm(Fz))
            if residual < tol:
                break

    wall_time = time.perf_counter() - t0
    gx, gy_neg = problem.grad_f(x, y)
    residual = float(jnp.linalg.norm(jnp.concatenate([gx, gy_neg])))
    gap = _estimate_gap_simple(problem, x, y)

    return BaselineResult(
        x=x,
        y=y,
        gap=gap,
        iterations=i + 1,
        wall_time=wall_time,
        converged=gap <= epsilon,
        final_residual=residual,
    )


def run_eg_jit(
    problem: MinimaxProblem,
    max_iters: int = 5_000,
) -> tuple[Array, float, float]:
    """JIT-compiled extragradient for timing comparisons.

    Returns (z, residual, wall_time).  No convergence checking inside the
    loop (pure timing).
    """
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 1.0 / (2.0 * ell)

    x0 = problem.project_x(jnp.zeros(problem.dim_x))
    y0 = problem.project_y(jnp.zeros(problem.dim_y))
    z0 = jnp.concatenate([x0, y0])

    proj = lambda z: jnp.concatenate([
        problem.project_x(z[: problem.dim_x]),
        problem.project_y(z[problem.dim_x :]),
    ])

    @jax.jit
    def eg_loop(z_init):
        def body(i, z):
            Fz = problem.operator_F(z)
            z_half = proj(z - eta * Fz)
            F_half = problem.operator_F(z_half)
            return proj(z - eta * F_half)

        return jax.lax.fori_loop(0, max_iters, body, z_init)

    z0.block_until_ready()
    t0 = time.perf_counter()
    z_out = eg_loop(z0)
    z_out.block_until_ready()
    wall_time = time.perf_counter() - t0

    residual = float(jnp.linalg.norm(problem.operator_F(z_out)))
    return z_out, residual, wall_time
