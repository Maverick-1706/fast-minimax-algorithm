"""Fair, JIT-compiled baseline solvers.

Design principle: every baseline uses jax.lax.fori_loop for identical
compilation treatment as the Minimax-AIPE solver.  No Python loops in
timing-critical paths.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe import OracleStats
from minimax_aipe.framework import _safe_gap
from minimax_aipe.problem import MinimaxProblem
from benchmarks.oracles import count_eg_oracles, count_gda_oracles, count_npe_oracles

@dataclass
class BaselineResult:
    """Result from a baseline solver run.

    Attributes
    ----------
    converged : bool
        Whether the loop terminated early (gap-based stopping criterion met).
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

# ── JIT-compiled extragradient ───────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _get_eg_loop(problem: MinimaxProblem, max_iters: int, tol: float):
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 1.0 / (2.0 * ell)

    proj = lambda z: jnp.concatenate([
        problem.project_x(z[: problem.dim_x]),
        problem.project_y(z[problem.dim_x :]),
    ])

    F_op = problem.operator_F

    @jax.jit
    def eg_loop(z_init):
        D = max(float(problem.D_x), float(problem.D_y), 1.0)
        res_tol = tol / D if tol > 0 else -1.0
        tol_val = jnp.asarray(res_tol, dtype=z_init.dtype)
        eval_freq = jnp.int32(100)
        max_chunks = jnp.int32(max_iters // 100)
        Fz_init = F_op(z_init)

        def cond(state):
            chunk, _z, Fz = state
            res_norm = jnp.linalg.norm(Fz)
            return (chunk < max_chunks) & (res_norm > tol_val)

        def chunk_body(i, z_cur):
            z_half = proj(z_cur - eta * F_op(z_cur))
            z_new = proj(z_cur - eta * F_op(z_half))
            return z_new

        def body(state):
            chunk, z, _ = state
            z_new = jax.lax.fori_loop(0, eval_freq, chunk_body, z)
            Fz_new = F_op(z_new)
            return (chunk + 1, z_new, Fz_new)

        chunks_out, z_out, Fz_out = jax.lax.while_loop(
            cond, body, (jnp.int32(0), z_init, Fz_init)
        )
        return chunks_out * eval_freq, z_out, Fz_out

    return eg_loop

def run_eg_jit(
    problem: MinimaxProblem,
    max_iters: int = 100_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled extragradient via jax.lax.while_loop."""
    if z0 is None:
        x0 = problem.project_x(jnp.zeros(problem.dim_x))
        y0 = problem.project_y(jnp.zeros(problem.dim_y))
        z_start = jnp.concatenate([x0, y0])
    else:
        z0_arr = jnp.asarray(z0)
        x0 = problem.project_x(z0_arr[: problem.dim_x])
        y0 = problem.project_y(z0_arr[problem.dim_x :])
        z_start = jnp.concatenate([x0, y0])

    eg_loop = _get_eg_loop(problem, max_iters, tol)

    z_start.block_until_ready()
    t0 = time.perf_counter()
    iters_out, z_out, Fz_out = eg_loop(z_start)
    Fz_out.block_until_ready()
    residual = float(jnp.linalg.norm(Fz_out))
    wall_time = time.perf_counter() - t0
    return z_out, residual, wall_time, int(iters_out)

def run_eg_jit_benchmark(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 100_000,
    z0: Array | None = None,
) -> BaselineResult:
    """JIT-EG benchmark wrapper returning BaselineResult with gap."""
    z_out, residual, loop_time, actual_iters = run_eg_jit(
        problem, max_iters=max_iters, z0=z0, tol=epsilon
    )
    
    # Time the post-processing (gap estimation) to ensure fair comparison
    # with AIPE, which includes gap computation in its solve time.
    t_gap = time.perf_counter()
    x_out = z_out[: problem.dim_x]
    y_out = z_out[problem.dim_x :]
    gap = _safe_gap(problem, x_out, y_out, epsilon)
    gap_time = time.perf_counter() - t_gap
    
    wall_time = loop_time + gap_time

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        # FIX: Explicitly check if the solver completed early before hitting the budget cap
        converged=actual_iters < (max_iters // 100) * 100,
        gap_achieved=gap <= epsilon,
        final_residual=residual,
        oracle_stats=count_eg_oracles(actual_iters),  # ← Replaced manual construction
    )


# ── JIT-compiled GDA ─────────────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _get_gda_loop(problem: MinimaxProblem, max_iters: int, tol: float):
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    eta = 0.5 / ell
    F_op = problem.operator_F

    @jax.jit
    def gda_loop(z_init):
        D = max(float(problem.D_x), float(problem.D_y), 1.0)
        res_tol = tol / D if tol > 0 else -1.0
        tol_val = jnp.asarray(res_tol, dtype=z_init.dtype)
        eval_freq = jnp.int32(100)
        max_chunks = jnp.int32(max_iters // 100)
        Fz_init = F_op(z_init)

        def cond(state):
            chunk, _z, Fz = state
            res_norm = jnp.linalg.norm(Fz)
            return (chunk < max_chunks) & (res_norm > tol_val)

        def chunk_body(i, z_cur):
            Fz = F_op(z_cur)
            gx = Fz[: problem.dim_x]
            gy_neg = Fz[problem.dim_x :]
            x, y = z_cur[: problem.dim_x], z_cur[problem.dim_x :]
            x_new = problem.project_x(x - eta * gx)
            y_new = problem.project_y(y - eta * gy_neg)
            return jnp.concatenate([x_new, y_new])

        def body(state):
            chunk, z, _ = state
            z_new = jax.lax.fori_loop(0, eval_freq, chunk_body, z)
            Fz_new = F_op(z_new)
            return (chunk + 1, z_new, Fz_new)

        chunks_out, z_out, Fz_out = jax.lax.while_loop(
            cond, body, (jnp.int32(0), z_init, Fz_init)
        )
        return chunks_out * eval_freq, z_out, Fz_out

    return gda_loop


def run_gda_jit(
    problem: MinimaxProblem,
    max_iters: int = 200_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled gradient descent-ascent via jax.lax.while_loop."""
    if z0 is None:
        x0 = problem.project_x(jnp.zeros(problem.dim_x))
        y0 = problem.project_y(jnp.zeros(problem.dim_y))
        z_start = jnp.concatenate([x0, y0])
    else:
        z0_arr = jnp.asarray(z0)
        x0 = problem.project_x(z0_arr[: problem.dim_x])
        y0 = problem.project_y(z0_arr[problem.dim_x :])
        z_start = jnp.concatenate([x0, y0])

    gda_loop = _get_gda_loop(problem, max_iters, tol)

    z_start.block_until_ready()
    t0 = time.perf_counter()
    iters_out, z_out, Fz_out = gda_loop(z_start)
    Fz_out.block_until_ready()
    residual = float(jnp.linalg.norm(Fz_out))
    wall_time = time.perf_counter() - t0
    return z_out, residual, wall_time, int(iters_out)

def run_gda_jit_benchmark(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 200_000,
    z0: Array | None = None,
) -> BaselineResult:
    """JIT-GDA benchmark wrapper returning BaselineResult with gap."""
    z_out, residual, loop_time, actual_iters = run_gda_jit(
        problem, max_iters=max_iters, z0=z0, tol=epsilon
    )
    
    # Time the post-processing (gap estimation) to ensure fair comparison
    t_gap = time.perf_counter()
    x_out = z_out[: problem.dim_x]
    y_out = z_out[problem.dim_x :]
    gap = _safe_gap(problem, x_out, y_out, epsilon)
    gap_time = time.perf_counter() - t_gap
    
    wall_time = loop_time + gap_time

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        # FIX: Explicitly check if the solver completed early before hitting the budget cap
        converged=actual_iters < (max_iters // 100) * 100,
        gap_achieved=gap <= epsilon,
        final_residual=residual,
        oracle_stats=count_gda_oracles(actual_iters),  # ← Replaced manual construction
    )

# ── JIT-compiled NPE-restart ─────────────────────────────────────────────

from minimax_aipe.npe import npe, make_crn_npe_oracle

@functools.lru_cache(maxsize=None)
def _get_npe_epoch_loop(problem: MinimaxProblem, max_iters: int, tol: float):
    actual_rho = float(problem.rho) if problem.rho is not None else 0.0
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    
    if actual_rho <= 1e-10:
        # Smooth-case fallback using eps-dependent bound
        eps = max(tol, 1e-4)
        gamma = eps
        T = min(max_iters, max(10, int((ell / eps) ** 0.5)))
    else:
        # Strongly monotone case
        rho = max(actual_rho, 1e-6)
        gamma = 2.0 * rho
        T = min(max_iters, max(10, int((ell / rho) ** (2.0 / 3.0))))

    proj = lambda z: jnp.concatenate([
        problem.project_x(z[: problem.dim_x]),
        problem.project_y(z[problem.dim_x :]),
    ])

    F_op = problem.operator_F
    oracle = make_crn_npe_oracle(problem, gamma)
    merit = lambda z: jnp.sum(F_op(z) ** 2)

    @jax.jit
    def npe_epoch_loop(z_init):
        D = max(float(problem.D_x), float(problem.D_y), 1.0)
        res_tol = tol / D if tol > 0 else -1.0
        tol_val = jnp.asarray(res_tol, dtype=z_init.dtype)
        max_epochs = jnp.int32(max(1, max_iters // T))
        Fz_init = F_op(z_init)

        def cond(state):
            epoch, _z, Fz = state
            not_done = epoch < max_epochs
            res_norm = jnp.linalg.norm(Fz)
            return not_done & (res_norm > tol_val)

        def body(state):
            epoch, z, _ = state
            z_new, _ = npe(oracle, F_op, z, T, gamma, project=proj, fn=merit)
            Fz_new = F_op(z_new)
            return (epoch + 1, z_new, Fz_new)

        epochs_out, z_out, Fz_out = jax.lax.while_loop(
            cond, body, (jnp.int32(0), z_init, Fz_init)
        )
        return epochs_out, z_out, Fz_out

    return npe_epoch_loop, T


def run_npe_restart_jit(
    problem: MinimaxProblem,
    max_iters: int = 100_000,
    z0: Array | None = None,
    tol: float = 0.0,
) -> tuple[Array, float, float, int]:
    """JIT-compiled standalone NPE-restart via jax.lax.while_loop over epochs."""
    if z0 is None:
        x0 = problem.project_x(jnp.zeros(problem.dim_x))
        y0 = problem.project_y(jnp.zeros(problem.dim_y))
        z_start = jnp.concatenate([x0, y0])
    else:
        z0_arr = jnp.asarray(z0)
        x0 = problem.project_x(z0_arr[: problem.dim_x])
        y0 = problem.project_y(z0_arr[problem.dim_x :])
        z_start = jnp.concatenate([x0, y0])

    npe_epoch_loop, T = _get_npe_epoch_loop(problem, max_iters, tol)

    z_start.block_until_ready()
    t0 = time.perf_counter()
    epochs_out, z_out, Fz_out = npe_epoch_loop(z_start)
    Fz_out.block_until_ready()
    actual_iters = int(epochs_out) * T
    residual = float(jnp.linalg.norm(Fz_out))
    wall_time = time.perf_counter() - t0
    return z_out, residual, wall_time, actual_iters

def run_npe_restart_jit_benchmark(
    problem: MinimaxProblem,
    epsilon: float = 0.01,
    max_iters: int = 100_000,
    z0: Array | None = None,
) -> BaselineResult:
    """JIT-NPE-restart benchmark wrapper returning BaselineResult with gap."""
    z_out, residual, loop_time, actual_iters = run_npe_restart_jit(
        problem, max_iters=max_iters, z0=z0, tol=epsilon
    )
    
    # Time the post-processing (gap estimation) to ensure fair comparison
    t_gap = time.perf_counter()
    x_out = z_out[: problem.dim_x]
    y_out = z_out[problem.dim_x :]
    gap = _safe_gap(problem, x_out, y_out, epsilon)
    gap_time = time.perf_counter() - t_gap
    
    wall_time = loop_time + gap_time

    # Compute maximum allowed iterations based on internal epoch floor allocation
    actual_rho = float(problem.rho) if problem.rho is not None else 0.0
    ell = max(float(problem.ell) if problem.ell else 1.0, 1e-8)
    
    if actual_rho <= 1e-10:
        eps = max(epsilon, 1e-4)
        T = min(max_iters, max(10, int((ell / eps) ** 0.5)))
    else:
        rho = max(actual_rho, 1e-6)
        T = min(max_iters, max(10, int((ell / rho) ** (2.0 / 3.0))))
    
    max_expected_iters = (max_iters // T) * T

    return BaselineResult(
        x=x_out, y=y_out, gap=gap,
        iterations=actual_iters, wall_time=wall_time,
        # FIX: Match the baseline early-termination check using the expected maximum epoch ceiling
        converged=actual_iters < max_expected_iters,
        gap_achieved=gap <= epsilon,
        final_residual=residual,
        oracle_stats=count_npe_oracles(actual_iters),  # ← Replaced manual construction
    )
