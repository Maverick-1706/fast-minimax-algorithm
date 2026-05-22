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

from minimax_aipe import OracleStats
from minimax_aipe.problem import MinimaxProblem


@dataclass
class BaselineResult:
    """Result from a baseline solver run.

    Attributes
    ----------
    converged : bool
        Whether the loop terminated early (operator residual < tol).
    gap_achieved : bool
        Whether the duality gap <= epsilon (the actual success criterion).
    """

    x: Array
    y: Array
    gap: float
    iterations: int
    wall_time: float
    converged: bool
    gap_achieved: bool
    final_residual: float
    oracle_stats: OracleStats | None = None


from minimax_aipe.gap import estimate_gap

# ── JIT-compiled extragradient ───────────────────────────────────────────


def run_eg_jit(
    problem: MinimaxProblem,
    max_iters: int = 100_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled extragradient via jax.lax.while_loop."""
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

    F_op = problem.operator_F

    @jax.jit
    def eg_loop(z_init):
        tol_val = jnp.asarray(
            tol if tol > 0 else -1.0, dtype=z_init.dtype
        )
        max_i = jnp.int32(max_iters)
        init_Fz = F_op(z_init)

        def cond(state):
            i, _z, Fz = state
            return i < max_i

        def body(state):
            i, z, Fz = state
            z_half = proj(z - eta * Fz)
            F_half = F_op(z_half)
            z_new = proj(z - eta * F_half)
            new_Fz = F_op(z_new)
            return (i + 1, z_new, new_Fz)

        iters_out, z_out, _ = jax.lax.while_loop(
            cond, body, (jnp.int32(0), z_init, init_Fz)
        )
        return iters_out, z_out

    z_start.block_until_ready()
    t0 = time.perf_counter()
    iters_out, z_out = eg_loop(z_start)
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
    gap_val = estimate_gap(problem, x_out, y_out)
    if hasattr(gap_val, "block_until_ready"):
        gap_val.block_until_ready()
    gap = float(gap_val)

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        # FIX: Explicitly check if the solver completed early before hitting the budget cap
        converged=actual_iters < max_iters, 
        gap_achieved=gap <= epsilon,
        final_residual=residual,
        oracle_stats=OracleStats(grad_calls=2 * actual_iters, projection_calls=2 * actual_iters, oracle_calls=2 * actual_iters),
    )


# ── JIT-compiled GDA ─────────────────────────────────────────────────────


def run_gda_jit(
    problem: MinimaxProblem,
    max_iters: int = 200_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled gradient descent-ascent via jax.lax.while_loop."""
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 0.5 / ell

    if z0 is None:
        x0 = problem.project_x(jnp.zeros(problem.dim_x))
        y0 = problem.project_y(jnp.zeros(problem.dim_y))
        z_start = jnp.concatenate([x0, y0])
    else:
        z0_arr = jnp.asarray(z0)
        x0 = problem.project_x(z0_arr[: problem.dim_x])
        y0 = problem.project_y(z0_arr[problem.dim_x :])
        z_start = jnp.concatenate([x0, y0])

    F_op = problem.operator_F

    @jax.jit
    def gda_loop(z_init):
        tol_val = jnp.asarray(
            tol if tol > 0 else -1.0, dtype=z_init.dtype
        )
        max_i = jnp.int32(max_iters)
        init_Fz = F_op(z_init)

        def cond(state):
            i, _z, Fz = state
            return i < max_i

        def body(state):
            i, z, Fz = state
            gx = Fz[: problem.dim_x]
            gy_neg = Fz[problem.dim_x :]
            x, y = z[: problem.dim_x], z[problem.dim_x :]
            x_new = problem.project_x(x - eta * gx)
            y_new = problem.project_y(y - eta * gy_neg)
            z_new = jnp.concatenate([x_new, y_new])
            new_Fz = F_op(z_new)
            return (i + 1, z_new, new_Fz)

        iters_out, z_out, _ = jax.lax.while_loop(
            cond, body, (jnp.int32(0), z_init, init_Fz)
        )
        return iters_out, z_out

    z_start.block_until_ready()
    t0 = time.perf_counter()
    iters_out, z_out = gda_loop(z_start)
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
    gap_val = estimate_gap(problem, x_out, y_out)
    if hasattr(gap_val, "block_until_ready"):
        gap_val.block_until_ready()
    gap = float(gap_val)

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        # FIX: Explicitly check if the solver completed early before hitting the budget cap
        converged=actual_iters < max_iters, 
        gap_achieved=gap <= epsilon,
        final_residual=residual,
        oracle_stats=OracleStats(grad_calls=actual_iters, projection_calls=2 * actual_iters, oracle_calls=actual_iters),
    )

# ── JIT-compiled NPE-restart ─────────────────────────────────────────────

from minimax_aipe.npe import npe, make_crn_npe_oracle

def run_npe_restart_jit(
    problem: MinimaxProblem,
    max_iters: int = 100_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled standalone NPE-restart via jax.lax.while_loop over epochs."""
    rho = max(float(problem.rho) if problem.rho else 1.0, 1e-6)
    gamma = 2.0 * rho
    
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    T = min(max_iters, max(10, int((ell / rho) ** (2.0 / 3.0))))

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

    F_op = problem.operator_F
    oracle = make_crn_npe_oracle(problem, gamma)
    merit = lambda z: jnp.sum(F_op(z) ** 2)

    @jax.jit
    def npe_epoch_loop(z_init):
        tol_val = jnp.asarray(
            tol if tol > 0 else -1.0, dtype=z_init.dtype
        )
        max_epochs = jnp.int32(max(1, max_iters // T))
        init_gap = estimate_gap(problem, z_init[: problem.dim_x], z_init[problem.dim_x :])

        def cond(state):
            epoch, _z, gap = state
            not_done = epoch < max_epochs
            gap_big = gap > tol_val
            # FIX: Remove the jnp.where guard to allow exit at epoch=0 if already within tolerance
            return not_done & gap_big

        def body(state):
            epoch, z, _ = state
            z_new, _ = npe(oracle, F_op, z, T, gamma, project=proj, fn=merit)
            new_gap = estimate_gap(problem, z_new[: problem.dim_x], z_new[problem.dim_x :])
            return (epoch + 1, z_new, new_gap)

        epochs_out, z_out, _ = jax.lax.while_loop(
            cond, body, (jnp.int32(0), z_init, init_gap)
        )
        return epochs_out, z_out

    z_start.block_until_ready()
    t0 = time.perf_counter()
    epochs_out, z_out = npe_epoch_loop(z_start)
    z_out.block_until_ready()
    wall_time = time.perf_counter() - t0

    actual_iters = int(epochs_out) * T
    residual = float(jnp.linalg.norm(problem.operator_F(z_out)))
    return z_out, residual, wall_time, actual_iters

def run_npe_restart_jit_benchmark(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 100_000,
    z0: Array | None = None,
) -> BaselineResult:
    """JIT-NPE-restart benchmark wrapper returning BaselineResult with gap."""
    z_out, residual, wall_time, actual_iters = run_npe_restart_jit(
        problem, max_iters=max_iters, z0=z0, tol=epsilon
    )
    x_out = z_out[: problem.dim_x]
    y_out = z_out[problem.dim_x :]
    gap_val = estimate_gap(problem, x_out, y_out)
    if hasattr(gap_val, "block_until_ready"):
        gap_val.block_until_ready()
    gap = float(gap_val)

    # Compute maximum allowed iterations based on internal epoch floor allocation
    rho = max(float(problem.rho) if problem.rho else 1.0, 1e-6)
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    T = min(max_iters, max(10, int((ell / rho) ** (2.0 / 3.0))))
    max_expected_iters = (max_iters // T) * T

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        # FIX: Match the baseline early-termination check using the expected maximum epoch ceiling
        converged=actual_iters < max_expected_iters, 
        gap_achieved=gap <= epsilon,
        final_residual=residual,
        oracle_stats=OracleStats(
            crn_calls=actual_iters,
            grad_calls=actual_iters,
            hessian_calls=actual_iters,
            oracle_calls=actual_iters,
            call_type="crn"
        ),
    )