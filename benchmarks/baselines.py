"""Fair, JIT-compiled baseline solvers.

Design principle: every baseline uses jax.lax.fori_loop for identical
compilation treatment as the Minimax-AIPE solver.  No Python loops in
timing-critical paths.
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


from minimax_aipe.gap import estimate_gap

# ── JIT-compiled extragradient ───────────────────────────────────────────


def run_eg_jit(
    problem: MinimaxProblem,
    max_iters: int = 100_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled extragradient via jax.lax.while_loop.

    Returns (z, residual, wall_time, actual_iters).
    """
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 1.0 / (2.0 * ell)

    if z0 is None:
        x0 = problem.project_x(jnp.zeros(problem.dim_x))
        y0 = problem.project_y(jnp.zeros(problem.dim_y))
        z_start = jnp.concatenate([x0, y0])
    else:
        z0_arr = jnp.asarray(z0)
        x0 = problem.project_x(z0_arr[: problem.dim_x])
        y0 = problem.project_y(z0_arr[problem.dim_x :])
        z_start = jnp.concatenate([x0, y0])

    proj = lambda z: jnp.concatenate([
        problem.project_x(z[: problem.dim_x]),
        problem.project_y(z[problem.dim_x :]),
    ])

    @jax.jit
    def eg_loop(z_init):
        tol_sq = jnp.asarray(tol ** 2, dtype=z_init.dtype)
        max_i = jnp.int32(max_iters)

        def cond(state):
            i, _z, _prev_z = state
            not_done = i < max_i
            step_sq = jnp.sum((_z - _prev_z) ** 2)
            step_big = step_sq > tol_sq
            return not_done & jnp.where(i > 0, step_big, jnp.bool_(True))

        def body(state):
            i, z, _prev_z = state
            Fz = problem.operator_F(z)
            z_half = proj(z - eta * Fz)
            F_half = problem.operator_F(z_half)
            return (i + 1, proj(z - eta * F_half), z)

        return jax.lax.while_loop(cond, body, (jnp.int32(0), z_init, z_init))

    z_start.block_until_ready()
    t0 = time.perf_counter()
    iters_out, z_out, _ = eg_loop(z_start)
    z_out.block_until_ready()
    wall_time = time.perf_counter() - t0

    residual = float(jnp.linalg.norm(problem.operator_F(z_out)))
    return z_out, residual, wall_time, int(iters_out)


def run_eg_jit_benchmark(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 100_000,
    z0: Array | None = None,
) -> BaselineResult:
    """JIT-EG benchmark wrapper returning BaselineResult with gap."""
    z_out, residual, wall_time, actual_iters = run_eg_jit(
        problem, max_iters=max_iters, z0=z0, tol=epsilon
    )
    x_out = z_out[: problem.dim_x]
    y_out = z_out[problem.dim_x :]
    gap = estimate_gap(problem, x_out, y_out)

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        converged=gap <= epsilon, final_residual=residual,
    )


# ── JIT-compiled GDA ─────────────────────────────────────────────────────


def run_gda_jit(
    problem: MinimaxProblem,
    max_iters: int = 200_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled gradient descent-ascent via jax.lax.while_loop.

    Returns (z, residual, wall_time, actual_iters).
    """
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 1.0 / ell

    if z0 is None:
        x0 = problem.project_x(jnp.zeros(problem.dim_x))
        y0 = problem.project_y(jnp.zeros(problem.dim_y))
        z_start = jnp.concatenate([x0, y0])
    else:
        z0_arr = jnp.asarray(z0)
        x0 = problem.project_x(z0_arr[: problem.dim_x])
        y0 = problem.project_y(z0_arr[problem.dim_x :])
        z_start = jnp.concatenate([x0, y0])

    @jax.jit
    def gda_loop(z_init):
        tol_sq = jnp.asarray(tol ** 2, dtype=z_init.dtype)
        max_i = jnp.int32(max_iters)

        def cond(state):
            i, _z, _prev_z = state
            not_done = i < max_i
            step_sq = jnp.sum((_z - _prev_z) ** 2)
            step_big = step_sq > tol_sq
            return not_done & jnp.where(i > 0, step_big, jnp.bool_(True))

        def body(state):
            i, z, _prev_z = state
            x, y = z[: problem.dim_x], z[problem.dim_x :]
            gx, gy_neg = problem.grad_f(x, y)
            x_new = problem.project_x(x - eta * gx)
            y_new = problem.project_y(y - eta * gy_neg)
            return (i + 1, jnp.concatenate([x_new, y_new]), z)

        return jax.lax.while_loop(cond, body, (jnp.int32(0), z_init, z_init))

    z_start.block_until_ready()
    t0 = time.perf_counter()
    iters_out, z_out, _ = gda_loop(z_start)
    z_out.block_until_ready()
    wall_time = time.perf_counter() - t0

    residual = float(jnp.linalg.norm(problem.operator_F(z_out)))
    return z_out, residual, wall_time, int(iters_out)


def run_gda_jit_benchmark(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 200_000,
    z0: Array | None = None,
) -> BaselineResult:
    """JIT-GDA benchmark wrapper returning BaselineResult with gap."""
    z_out, residual, wall_time, actual_iters = run_gda_jit(
        problem, max_iters=max_iters, z0=z0, tol=epsilon
    )
    x_out = z_out[: problem.dim_x]
    y_out = z_out[problem.dim_x :]
    gap = estimate_gap(problem, x_out, y_out)

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        converged=gap <= epsilon, final_residual=residual,
    )
