"""Oracle implementations for the Minimax-AIPE algorithm.

Provides the building blocks called by the outer / middle / inner loops:
  * ``eg_step``                — extragradient (Definition 3.3, Eq. 3)
  * ``crn_oracle``             — cubic-regularised Newton for minimax (Definition 3.2)
  * ``crn_oracle_minimization``— CRN for a scalar convex objective
  * ``lazy_crn_oracle``        — CRN with a stale Hessian snapshot (Definition E.1)
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe.problem import MinimaxProblem


# ── helpers ────────────────────────────────────────────────────────────────

def _split(z: Array, dim_x: int) -> tuple[Array, Array]:
    """Partition *z* into (x, y) given the primal dimension."""
    return z[:dim_x], z[dim_x:]


def _project_z(problem: MinimaxProblem, z: Array) -> Array:
    """Euclidean projection of z = [x, y] onto Z = X × Y."""
    x, y = _split(z, problem.dim_x)
    return jnp.concatenate([problem.project_x(x), problem.project_y(y)])


def _build_jacobian(problem: MinimaxProblem, x: Array, y: Array) -> Array:
    """Assemble the Jacobian ∇F(z) from ``hessian_f``.

    ``hessian_f(x, y)`` returns ``((H_xx, H_xy), (H_yx, H_yy))``.
    The Jacobian of ``F(z) = [∇_x f, −∇_y f]`` is::

        [[  H_xx,   H_xy ],
         [ −H_yx, −H_yy ]]
    """
    (H_xx, H_xy), (H_yx, H_yy) = problem.hessian_f(x, y)
    top = jnp.concatenate([H_xx, H_xy], axis=1)
    bot = jnp.concatenate([-H_yx, -H_yy], axis=1)
    return jnp.concatenate([top, bot], axis=0)


# ── 1. extragradient step (Definition 3.3 / Eq. 3) ────────────────────────

def eg_step(
    problem: MinimaxProblem,
    z: Array,
    eta: float,
) -> tuple[Array, Array]:
    """One extragradient step on the monotone VI.

    Computes::

        z½  = proj_Z(z − η F(z))
        z¹  = proj_Z(z − η F(z½))
        c₁  = (z − z¹)/η − F(z½)

    Parameters
    ----------
    problem : MinimaxProblem
    z       : Array — current iterate [x, y]
    eta     : float — step size

    Returns
    -------
    z_new : Array — z¹
    c     : Array — residual certificate  c₁ ∈ ∂I_Z(z¹)
    """
    Fz = problem.operator_F(z)
    z_half = _project_z(problem, z - eta * Fz)
    F_half = problem.operator_F(z_half)
    z_new = _project_z(problem, z - eta * F_half)
    c = (z - z_new) / eta - F_half
    return z_new, c


# ── 2. CRN oracle — minimax (Definition 3.2) ──────────────────────────────

def crn_oracle(
    problem: MinimaxProblem,
    z_bar: Array,
    gamma: float,
    n_iters: int = 50,
    tol: float = 0.0,
) -> tuple[Array, Array]:
    """Cubic-regularised Newton oracle for a minimax problem.

    Solves the variational inequality via fixed-point iteration on the
    secular equation ``(H + λI)δ = −g`` with ``λ = (γ/2)‖δ‖``.

    Always uses ``jax.lax.fori_loop`` (static bound) so that JAX can
    unroll the loop at compile time.  The ``tol`` parameter is accepted
    for API compatibility but does not switch to ``while_loop`` — nested
    dynamic loops cause exponential compilation time inside JAX
    ``scan``/``fori_loop`` bodies.
    """
    x_bar, y_bar = _split(z_bar, problem.dim_x)
    g = problem.operator_F(z_bar)
    H = _build_jacobian(problem, x_bar, y_bar)
    d = z_bar.shape[0]
    dtype = z_bar.dtype
    eye = jnp.eye(d, dtype=dtype)
    tiny = jnp.asarray(1e-12, dtype=dtype)

    def body(i, state):
        lam, z = state
        delta = jnp.linalg.solve(H + (lam + tiny) * eye, -g)
        z_new = _project_z(problem, z_bar + delta)
        d_eff = z_new - z_bar
        return (gamma / 2.0) * jnp.linalg.norm(d_eff), z_new

    lam, z = jax.lax.fori_loop(0, n_iters, body, (jnp.zeros(()), z_bar))
    d_eff = z - z_bar
    u = -(g + H @ d_eff + lam * d_eff)
    return z, u


def crn_oracle_minimization(
    grad_fn: Callable[[Array], Array],
    hess_fn: Callable[[Array], Array],
    z_bar: Array,
    gamma: float,
    n_iters: int = 50,
    project: Optional[Callable[[Array], Array]] = None,
    tol: float = 0.0,
) -> tuple[Array, Array]:
    """CRN oracle for a scalar convex objective ``min h(z)``.

    Always uses ``jax.lax.fori_loop`` (see :func:`crn_oracle` for rationale).
    """
    g = grad_fn(z_bar)
    H = hess_fn(z_bar)
    d = z_bar.shape[0]
    dtype = z_bar.dtype
    eye = jnp.eye(d, dtype=dtype)
    tiny = jnp.asarray(1e-12, dtype=dtype)

    def body(i, state):
        lam, z = state
        delta = jnp.linalg.solve(H + (lam + tiny) * eye, -g)
        z_new = z_bar + delta
        if project is not None:
            z_new = project(z_new)
        d_eff = z_new - z_bar
        return (gamma / 2.0) * jnp.linalg.norm(d_eff), z_new

    lam, z = jax.lax.fori_loop(0, n_iters, body, (jnp.zeros(()), z_bar))
    d_eff = z - z_bar
    u = -(g + H @ d_eff + lam * d_eff)
    return z, u


def lazy_crn_oracle(
    problem: MinimaxProblem,
    z_bar: Array,
    z_snapshot: Array,
    gamma: float,
    n_iters: int = 50,
    tol: float = 0.0,
) -> tuple[Array, Array]:
    """Lazy CRN oracle — reuses a stale Hessian from *z_snapshot*.

    Always uses ``jax.lax.fori_loop`` (see :func:`crn_oracle` for rationale).
    """
    x_ss, y_ss = _split(z_snapshot, problem.dim_x)
    g = problem.operator_F(z_bar)
    H = _build_jacobian(problem, x_ss, y_ss)
    d = z_bar.shape[0]
    dtype = z_bar.dtype
    eye = jnp.eye(d, dtype=dtype)
    tiny = jnp.asarray(1e-12, dtype=dtype)

    def body(i, state):
        lam, z = state
        delta = jnp.linalg.solve(H + (lam + tiny) * eye, -g)
        z_new = _project_z(problem, z_bar + delta)
        d_eff = z_new - z_bar
        return (gamma / 2.0) * jnp.linalg.norm(d_eff), z_new

    lam, z = jax.lax.fori_loop(0, n_iters, body, (jnp.zeros(()), z_bar))
    d_eff = z - z_bar
    u = -(g + H @ d_eff + lam * d_eff)
    return z, u


def lazy_crn_oracle(
    problem: MinimaxProblem,
    z_bar: Array,
    z_snapshot: Array,
    gamma: float,
    n_iters: int = 50,
    tol: float = 0.0,
) -> tuple[Array, Array]:
    """Lazy CRN oracle — reuses a stale Hessian from *z_snapshot*.

    Parameters
    ----------
    tol : float
        Adaptive secular-equation convergence tolerance.
        See :func:`crn_oracle` for details.
    """
    x_ss, y_ss = _split(z_snapshot, problem.dim_x)
    g = problem.operator_F(z_bar)
    H = _build_jacobian(problem, x_ss, y_ss)
    d = z_bar.shape[0]
    dtype = z_bar.dtype
    eye = jnp.eye(d, dtype=dtype)
    tiny = jnp.asarray(1e-12, dtype=dtype)

    if tol > 0:
        def cond(state):
            lam, _z, i, prev_lam = state
            change = jnp.abs(lam - prev_lam)
            return (i < n_iters) & (change > jnp.maximum(tol * lam, tiny))

        def body(state):
            lam, _z, i, _prev = state
            delta = jnp.linalg.solve(H + (lam + tiny) * eye, -g)
            z_new = _project_z(problem, z_bar + delta)
            d_eff = z_new - z_bar
            return ((gamma / 2.0) * jnp.linalg.norm(d_eff), z_new, i + 1, lam)

        lam, z, _i, _p = jax.lax.while_loop(
            cond, body,
            (jnp.zeros(()), z_bar, 0, jnp.asarray(-1.0, dtype=dtype)),
        )
    else:
        def body(i, state):
            lam, z = state
            delta = jnp.linalg.solve(H + (lam + tiny) * eye, -g)
            z_new = _project_z(problem, z_bar + delta)
            d_eff = z_new - z_bar
            return (gamma / 2.0) * jnp.linalg.norm(d_eff), z_new

        lam, z = jax.lax.fori_loop(0, n_iters, body, (jnp.zeros(()), z_bar))

    d_eff = z - z_bar
    u = -(g + H @ d_eff + lam * d_eff)
    return z, u

__all__ = [
    "eg_step",
    "crn_oracle",
    "crn_oracle_minimization",
    "lazy_crn_oracle",
]
