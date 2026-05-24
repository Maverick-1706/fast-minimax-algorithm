"""Accelerated Lazy Extra Newton (ALEN).

Lazy variant of AIPE (A-NPE) that reuses Hessians across consecutive
proximal steps.  Used as M_min in the Minimax-AIPE triple loop so that
the LEN path does not negate its own computational advantage by
recomputing Hessians inside every scalar sub-solve.

Theorem E.4 (Chen et al. 2025a):
    ALEN-restart returns z with ‖z − z*‖ ≤ ε in
    O(m + m^{5/7} (ρ/μ)^{2/7} log(d₀/ε) log m) lazy CRN oracle calls.

Design
------
The existing :func:`~minimax_aipe.aipe.aipe` and
:func:`~minimax_aipe.aipe.aipe_restart` are **unchanged**.  ALEN works
entirely through the proximal-oracle abstraction:

1. :func:`make_lazy_crn_prox_oracle` returns a *factory* ``z_snap → prox``.
2. :func:`aipe_restart_lazy` calls the factory at every epoch boundary
   (Python-level ``for`` loop), producing a proximal oracle whose Hessian
   is frozen at the epoch's starting iterate.
3. Within the epoch (``T`` steps of :func:`aipe`), the Hessian is a
   compile-time constant — zero extra overhead.

This gives one Hessian computation per epoch instead of one per step.
"""

from __future__ import annotations

from math import ceil
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array

from minimax_aipe.aipe import ProxOracle, aipe
from minimax_aipe.oracles import crn_oracle_minimization


# ═══════════════════════════════════════════════════════════════════════════
# Public types
# ═══════════════════════════════════════════════════════════════════════════

#: A factory that, given a snapshot point, returns a proximal oracle.
#: ``z_snapshot → (z_bar → (z_tilde, u))``
ProxOracleFactory = Callable[[Array], ProxOracle]


# ═══════════════════════════════════════════════════════════════════════════
# Core factory
# ═══════════════════════════════════════════════════════════════════════════

def make_lazy_crn_prox_oracle(
    grad_fn: Callable[[Array], Array],
    hess_fn: Callable[[Array], Array],
    gamma: float,
    n_iters: int = 20,
    project: Optional[Callable[[Array], Array]] = None,
    tol: float = 0.0,
) -> ProxOracleFactory:
    """Create a factory for lazy CRN-based (0, γ)-proximal oracles.

    Analogous to :func:`~minimax_aipe.aipe.make_crn_prox_oracle` but
    returns a **factory** rather than a single oracle.  Each factory call
    materialises the Hessian once at the given snapshot, then returns a
    proximal oracle that reuses it for every query.

    The gradient is always evaluated at the query point ``z_bar`` (fresh).

    Parameters
    ----------
    grad_fn : Callable
        Gradient of the scalar convex objective ``h``.
    hess_fn : Callable
        Hessian of ``h``.
    gamma : float
        Proximal regularisation parameter.  The CRN oracle is called
        with cubic parameter ``2γ`` (Lemma 4.1).
    n_iters : int
        Maximum secular-equation iterations per CRN call.
    project : Callable or None
        Feasible-set projection (or ``None`` for unconstrained).
    tol : float
        Secular-equation convergence tolerance (forwarded to
        :func:`crn_oracle_minimization`).

    Returns
    -------
    factory : ProxOracleFactory
        ``z_snapshot → ProxOracle``.  The returned oracle evaluates
        ``∇²h`` at ``z_snapshot`` (cached) and ``∇h`` at the query
        point (fresh).
    """

    def factory(z_snapshot: Array) -> ProxOracle:
        # Materialise Hessian once.  This is a concrete JAX array that
        # becomes a compile-time constant inside the AIPE scan body.
        H_ss = hess_fn(z_snapshot)

        # Stale Hessian: ignores the query argument and returns the
        # cached matrix.  The ``_z`` parameter exists only for API
        # compatibility with ``crn_oracle_minimization``.
        def stale_hess(_z: Array) -> Array:
            return H_ss

        def prox(z_bar: Array) -> tuple[Array, Array, Array]:
            return crn_oracle_minimization(
                grad_fn, stale_hess, z_bar, 2.0 * gamma,
                n_iters=n_iters, project=project, tol=tol,
            )

        return prox

    return factory


# ═══════════════════════════════════════════════════════════════════════════
# ALEN-restart (Algorithm 9 analogue for scalar minimisation)
# ═══════════════════════════════════════════════════════════════════════════

def aipe_restart_lazy(
    prox_oracle_factory: ProxOracleFactory,
    grad_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    S: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], float]] = None,
) -> tuple[Array, int]:
    """ALEN-restart: AIPE with restarts and lazy Hessian updates.

    At each epoch boundary, calls ``prox_oracle_factory(z_current)`` to
    create a proximal oracle with Hessian evaluated at ``z_current``.
    Within the epoch the Hessian is reused for all ``T`` proximal steps.

    This yields one Hessian computation per epoch (``S`` total) instead
    of one per step (``S · T`` total).

    Parameters
    ----------
    prox_oracle_factory : ProxOracleFactory
        ``z_snapshot → ProxOracle``.
        Use :func:`make_lazy_crn_prox_oracle` to create one.
    grad_fn, z0, T, gamma, S, project, fn
        Forwarded to :func:`~minimax_aipe.aipe.aipe`.

    Returns
    -------
    z_out : Array
        Approximate minimiser.
    oracle_calls : int
        Total proximal oracle invocations (≈ S × (T + 1)).
    """
    z = z0
    total_stats = jnp.zeros(3, dtype=jnp.int32)
    for s in range(S):
        prox = prox_oracle_factory(z)
            
        result = aipe(prox, grad_fn, z, T, gamma,   
                       project=project, fn=fn)
        z = result[0]
        # result[3] holds total_inner_calls from aipe
        calls_arr = result[3] 
        total_stats = total_stats + calls_arr
    return z, total_stats

# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrappers for M_min usage
# ═══════════════════════════════════════════════════════════════════════════

def _alen_schedule(total_steps: int, m: int) -> tuple[int, int]:
    """Compute ``(T_epoch, S_epoch)`` so that the Hessian is refreshed
    every ``m`` proximal calls and the total work ≈ ``total_steps``.

    Caps ``S_epoch`` at 10 to limit tracing overhead when called from
    inside a JAX scan body (each epoch compiles a separate scan).
    """
    m = max(1, m)
    S_epoch = min(max(1, total_steps // m), 10)
    T_epoch = max(1, total_steps // max(S_epoch, 1))
    return T_epoch, S_epoch


def minimize_x_alen(
    problem,
    y_fixed: Array,
    *,
    steps: int,
    gamma: float,
    m: int = 5,
    x_init: Optional[Array] = None,
) -> Array:
    """Approximately minimise ``x ↦ f(x, y_fixed)`` using ALEN-restart.

    Drop-in replacement for ``_minimize_x`` that uses lazy Hessians.
    """
    def grad_fn(x: Array) -> Array:
        gx, _ = problem.grad_f(x, y_fixed)
        return gx

    def hess_fn(x: Array) -> Array:
        (H_xx, _), (_, _) = problem.hessian_f(x, y_fixed)
        return H_xx

    effective_gamma = max(float(problem.rho or gamma), 1e-8)
    factory = make_lazy_crn_prox_oracle(grad_fn, hess_fn, effective_gamma)

    x0 = problem.project_x(
        x_init if x_init is not None
        else jnp.zeros(problem.dim_x, dtype=y_fixed.dtype)
    )
    T_ep, S_ep = _alen_schedule(steps, m)

    z_out, calls = aipe_restart_lazy(
        factory, grad_fn, x0, T_ep, effective_gamma, S_ep,
        project=problem.project_x,
    )
    return z_out, calls


def maximize_y_alen(
    problem,
    x_fixed: Array,
    *,
    steps: int,
    gamma: float,
    m: int = 5,
    y_init: Optional[Array] = None,
) -> Array:
    """Approximately maximise ``y ↦ f(x_fixed, y)`` using ALEN-restart.

    Minimises the convex function ``y ↦ −f(x_fixed, y)`` via ALEN.
    Drop-in replacement for ``_maximize_y``.
    """
    # −f is convex in y (since f is concave in y).
    def grad_fn(y: Array) -> Array:
        _, gy_neg = problem.grad_f(x_fixed, y)
        return gy_neg  # = −∇_y f

    def hess_fn(y: Array) -> Array:
        (_, _), (_, H_yy) = problem.hessian_f(x_fixed, y)
        return -H_yy  # = −∇²_yy f  (PSD since f concave in y)

    effective_gamma = max(float(problem.rho or gamma), 1e-8)
    factory = make_lazy_crn_prox_oracle(grad_fn, hess_fn, effective_gamma)

    y0 = problem.project_y(
        y_init if y_init is not None
        else jnp.zeros(problem.dim_y, dtype=x_fixed.dtype)
    )
    T_ep, S_ep = _alen_schedule(steps, m)

    z_out, calls = aipe_restart_lazy(
        factory, grad_fn, y0, T_ep, effective_gamma, S_ep,
        project=problem.project_y,
    )
    return z_out, calls


__all__ = [
    "ProxOracleFactory",
    "make_lazy_crn_prox_oracle",
    "aipe_restart_lazy",
    "minimize_x_alen",
    "maximize_y_alen",
]
