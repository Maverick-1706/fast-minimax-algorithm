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
    lr: float = 0.01,
    key: Optional[Array] = None,
) -> float:
    """Estimate the duality gap via gradient ascent/descent.

    For small problems, computes:
        max_y f(x̂, y) via gradient ascent on y
        min_x f(x, ŷ) via gradient descent on x

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
    lr : float
        Learning rate.
    key : Array, optional
        JAX PRNG key. If None, uses a fixed seed.

    Returns
    -------
    gap : float
        Estimated duality gap.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    f = problem.f
    project_x = problem.project_x
    project_y = problem.project_y

    # Pre-generate all random initial points in one batched draw.
    key, y_key, x_key = jax.random.split(key, 3)
    y_inits = project_y(
        jax.random.normal(y_key, shape=(num_restarts, *y.shape)) * problem.D_y
    )
    x_inits = project_x(
        jax.random.normal(x_key, shape=(num_restarts, *x.shape)) * problem.D_x
    )

    # ------------------------------------------------------------------ #
    #  max_y  f(x, y)  via gradient ascent                                #
    # ------------------------------------------------------------------ #
    def y_step_body(_step: int, y_cur: Array) -> Array:
        """Single gradient ascent step on y (traced inside fori_loop)."""
        grad = grad_f_x(y_cur)
        return project_y(y_cur + lr * grad)


    def y_restart_body(restart_idx: int, best_val: Array) -> Array:
        """Run one full restart and fold the result into the running max."""
        y_final = jax.lax.fori_loop(
            0, num_steps, y_step_body, y_inits[restart_idx]
        )
        return jnp.maximum(best_val, f(x, y_final))


    f_x = lambda yy: f(x, yy)
    grad_f_x = jax.grad(f_x)
    best_max = jax.lax.fori_loop(0, num_restarts, y_restart_body, -jnp.inf)

    # ------------------------------------------------------------------ #
    #  min_x  f(x, y)  via gradient descent                               #
    # ------------------------------------------------------------------ #
    def x_step_body(_step: int, x_cur: Array) -> Array:
        """Single gradient descent step on x (traced inside fori_loop)."""
        grad = grad_f_y(x_cur)
        return project_x(x_cur - lr * grad)


    def x_restart_body(restart_idx: int, best_val: Array) -> Array:
        """Run one full restart and fold the result into the running min."""
        x_final = jax.lax.fori_loop(
            0, num_steps, x_step_body, x_inits[restart_idx]
        )
        return jnp.minimum(best_val, f(x_final, y))


    f_y = lambda xx: f(xx, y)
    grad_f_y = jax.grad(f_y)
    best_min = jax.lax.fori_loop(0, num_restarts, x_restart_body, jnp.inf)

    return float(best_max - best_min)
