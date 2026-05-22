"""Duality gap estimation for convex-concave minimax problems.

The duality gap (Definition 3.1) is:
    Gap(x̂, ŷ) = max_{y ∈ Y} f(x̂, y) - min_{x ∈ X} f(x, ŷ)
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe.problem import MinimaxProblem


def estimate_gap(
    problem: MinimaxProblem,
    x: Array,
    y: Array,
    *,
    num_restarts: int = 10,
    num_steps: int = 500,
    lr: Optional[float] = None,
    momentum: float = 0.9,
    key: Optional[Array] = None,
) -> Array:
    """Estimate the duality gap via Nesterov accelerated gradient ascent/descent.

    For small problems, computes:
        max_y f(x̂, y) via Nesterov accelerated gradient ascent on y
        min_x f(x, ŷ) via Nesterov accelerated gradient descent on x

    Three design choices improve robustness over plain GD:

    1. **Dynamic learning rate** — when *lr* is ``None`` (default), uses
       ``1 / problem.ell`` (inverse smoothness constant), which is the
       standard step size for L-smooth optimisation and prevents
       divergence on ill-conditioned problems where a fixed 0.01 rate
       would produce NaN.
    2. **Nesterov momentum** — the inner optimisation loop uses
       Nesterov's Accelerated Gradient method (NAG) with the
       *look-ahead* formulation, significantly improving convergence
       within the 500-iteration budget for ill-conditioned problems.
    3. **Current-iterate seeding** — the first restart initialisation is
       exactly the current *(x, y)*, which guarantees that
       ``max_y f(x, y') ≥ f(x, y) ≥ min_x f(x', y)``, so the
       estimated gap is always ≥ 0.

    Both the per-restart step loop and the restart loop are compiled
    into single XLA computations via ``jax.lax.fori_loop``, eliminating
    per-step Python dispatch overhead.

    Parameters
    ----------
    problem : MinimaxProblem
    x, y : Array
        Current iterate.
    num_restarts : int
        Number of random restarts to avoid local optima.
    num_steps : int
        Gradient steps per restart.
    lr : float or None
        Learning rate. When ``None`` (default), use ``1.0 / problem.ell``
        if a positive smoothness constant is available; otherwise fall
        back to the legacy default ``0.01``.
    momentum : float
        Nesterov momentum coefficient β ∈ [0, 1).  Default 0.9.
    key : Array, optional
        JAX PRNG key. If None, uses a fixed seed.

    Returns
    -------
    gap : Array
        Estimated duality gap (guaranteed ≥ 0 when the first restart
        seed is the current iterate and the optimisation does not
        diverge).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # ── Dynamic learning rate from problem smoothness ─────────────────
    if lr is None:
        ell_x = problem.ell_x
        lr_x = 1.0 / ell_x if ell_x is not None and ell_x > 0.0 else 0.01
        
        ell_y = problem.ell_y
        lr_y = 1.0 / ell_y if ell_y is not None and ell_y > 0.0 else 0.01
    else:
        lr_x = lr
        lr_y = lr

    # Keep x,y 1D even when caller passed scalar values.
    x = jnp.atleast_1d(jnp.asarray(x))
    y = jnp.atleast_1d(jnp.asarray(y))

    f = problem.f
    project_x = problem.project_x
    project_y = problem.project_y

    beta = momentum

    # Pre-generate all random initial points in one batched draw.
    key, y_key, x_key = jax.random.split(key, 3)
    y_inits_raw = jax.random.normal(
        y_key, shape=(num_restarts, *y.shape)
    ) * problem.D_y
    x_inits_raw = jax.random.normal(
        x_key, shape=(num_restarts, *x.shape)
    ) * problem.D_x
    y_inits = jax.vmap(project_y)(y_inits_raw)
    x_inits = jax.vmap(project_x)(x_inits_raw)

    # Seed the first restart with the current iterate so that
    # max_y f(x,y') ≥ f(x,y) ≥ min_x f(x',y), guaranteeing gap ≥ 0.
    y_inits = y_inits.at[0].set(y)
    x_inits = x_inits.at[0].set(x)

    # ------------------------------------------------------------------ #
    #  max_y  f(x, y)  via Nesterov accelerated gradient ascent           #
    # ------------------------------------------------------------------ #
    f_x = lambda yy: f(x, yy)
    grad_f_x = jax.grad(f_x)

    def y_step_body(_step: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        """Single NAG ascent step on y (traced inside fori_loop).
        
        Velocity form of Nesterov:
            y_ahead = y_cur + β · v_cur           (look-ahead)
            g       = ∇_y f(x, y_ahead)
            v_new   = β · v_cur + lr · g          (ascent)
            y_new   = Π(y_cur + v_new)            (project)
        """
        y_cur, v_cur, current_max = carry
        y_ahead = y_cur + beta * v_cur
        g = grad_f_x(y_ahead)
        v_new = beta * v_cur + lr_y * g  # Fixed: uses lr_y instead of joint lr
        y_new = project_y(y_cur + v_new)
        return (y_new, v_new, jnp.maximum(current_max, f_x(y_new)))

    def y_restart_body(restart_idx: int, best_val: Array) -> Array:
        """Run one full restart and fold the result into the running max."""
        y0 = y_inits[restart_idx]
        v0 = jnp.zeros_like(y0)
        _y_final, _v_final, traj_max = jax.lax.fori_loop(0, num_steps, y_step_body, (y0, v0, f_x(y0)))
        return jnp.maximum(best_val, traj_max)

    best_max = jax.lax.fori_loop(0, num_restarts, y_restart_body, -jnp.inf)

    # ------------------------------------------------------------------ #
    #  min_x  f(x, y)  via Nesterov accelerated gradient descent           #
    # ------------------------------------------------------------------ #
    f_y = lambda xx: f(xx, y)
    grad_f_y = jax.grad(f_y)

    def x_step_body(_step: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        """Single NAG descent step on x (traced inside fori_loop).
        
        Velocity form of Nesterov:
            x_ahead = x_cur + β · v_cur           (look-ahead)
            g       = ∇_x f(x_ahead, y)
            v_new   = β · v_cur − lr · g          (descent)
            x_new   = Π(x_cur + v_new)            (project)
        """
        x_cur, v_cur, current_min = carry
        x_ahead = x_cur + beta * v_cur
        g = grad_f_y(x_ahead)
        v_new = beta * v_cur - lr_x * g  # Fixed: uses lr_x instead of joint lr
        x_new = project_x(x_cur + v_new)
        return (x_new, v_new, jnp.minimum(current_min, f_y(x_new)))

    def x_restart_body(restart_idx: int, best_val: Array) -> Array:
        """Run one full restart and fold the result into the running min."""
        x0 = x_inits[restart_idx]
        v0 = jnp.zeros_like(x0)
        _x_final, _v_final, traj_min = jax.lax.fori_loop(0, num_steps, x_step_body, (x0, v0, f_y(x0)))
        return jnp.minimum(best_val, traj_min)

    best_min = jax.lax.fori_loop(0, num_restarts, x_restart_body, jnp.inf)

    return jnp.maximum(0.0, best_max - best_min)


__all__ = [
    "estimate_gap",
]
