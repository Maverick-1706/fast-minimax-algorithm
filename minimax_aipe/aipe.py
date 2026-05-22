"""Accelerated Inexact Proximal Extragradient (AIPE) methods.

Implements:
  * ``aipe``              — Algorithm 1: single AIPE run for scalar convex min h(z)
  * ``aipe_restart``      — Algorithm 2: AIPE with restart scheme
  * ``make_crn_prox_oracle`` — factory for CRN-based proximal oracles (Def 4.1 / Lemma 4.1)

Key design choices
──────────────────
* **Proximal oracle abstraction** — AIPE operates on *any* callable satisfying
  Definition 4.1, not on CRN specifically.  Swap in stochastic, quasi-Newton,
  or custom oracles without touching the acceleration logic.
* **JIT-native** — the main loop uses ``jax.lax.scan`` + ``jax.lax.cond``;
  ``T`` must be a static integer under ``jax.jit``.
* **State dataclass** — loop state is a ``NamedTuple`` pytree, making it
  easy to inspect, log, or convert to ``jax.lax.while_loop``.
* **Adaptive proximal accuracy** — ``make_crn_prox_oracle`` forwards a
  ``tol`` parameter to the secular-equation solver, enabling residual-based
  early exit inside the inner CRN loop.

.. note::

    **δ tolerance gap.**  The ``tol`` parameter on ``make_crn_prox_oracle``
    controls the convergence tolerance of the internal CRN secular-equation
    solver.  This is related to—but not identical to—the δ bound required
    by Definition 4.1 (``‖z̃ − ẑ‖ ≤ δ`` where ẑ is the exact proximal
    solution).  For production use with theoretical guarantees, the δ
    tolerance should be verified externally against the definition.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

from minimax_aipe._precision import TINY as _TINY_AIPE, REG_MIN as _REG_MIN_AIPE
from minimax_aipe.oracles import crn_oracle_minimization


# ── public types ───────────────────────────────────────────────────────────

#: A proximal oracle: ``z_bar ↦ (z_tilde, u)`` satisfying Definition 4.1.
#: The oracle's ``(γ, δ)`` parameters are baked into the closure.
ProxOracle = Callable[[Array], tuple[Array, Array]]

#: A warm-start proximal oracle: ``(z_bar, warm_init) ↦ (z_tilde, u, warm_out)``.
WarmStartProxOracle = Callable[[Array, Array], tuple[Array, Array, Array, ...]]


# ── loop state ─────────────────────────────────────────────────────────────

class AIPEState(NamedTuple):
    """Carry state for the AIPE scan loop (Algorithm 1)."""

    z: Array          # current iterate
    v: Array          # dual extrapolation point
    A: Array          # accumulated step weight  (scalar)
    lam_prime: Array  # target proximal accuracy  (scalar)


# ── numerical guard ─────────────────────────────────────────────────────────

# Upper bound on step size in v-update to prevent overflow when the
# quadratic root a' grows large (can happen when λ' is very small and A
# is large).  This is a pragmatic clamp; the theoretical algorithm places
# no bound on a', but in floating point the product a'·(g+u) can overflow.
_MAX_A_PRIME = 1e10


# ── proximal oracle factory ────────────────────────────────────────────────

def make_crn_prox_oracle(
    grad_fn: Callable[[Array], Array],
    hess_fn: Callable[[Array], Array],
    gamma: float,
    n_iters: int = 20,
    project: Optional[Callable[[Array], Array]] = None,
    tol: float = 0.0,
) -> ProxOracle:
    """Create a CRN-based ``(0, γ)``-proximal oracle (Lemma 4.1).

    Internally calls ``crn_oracle_minimization`` with parameter ``2·gamma``
    (the factor of 2 comes from Lemma 4.1: ``CRN(·, 2ρ)`` implements an
    ``(0, ρ)``-proximal oracle).

    Parameters
    ----------
    grad_fn  : gradient of the scalar objective h
    hess_fn  : Hessian of h
    gamma    : proximal oracle parameter γ
    n_iters  : maximum secular-equation iterations per call
    project  : feasible-set projection (or None for unconstrained)
    tol      : passed to ``crn_oracle_minimization``; when > 0 the secular
               equation exits early on convergence.

               .. warning::
                   ``tol`` bounds the *secular-equation residual*, which is
                   correlated with but not identical to the Definition 4.1
                   δ-error ``‖z̃ − ẑ‖ ≤ δ``.  For strict theoretical
                   guarantees, the δ error should be validated separately.
    """

    def prox(z_bar: Array) -> tuple[Array, Array]:
        return crn_oracle_minimization(
            grad_fn,
            hess_fn,
            z_bar,
            2.0 * gamma,
            n_iters=n_iters,
            project=project,
            tol=tol,
        )

    return prox


# ── Algorithm 1 ────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=[0,1,3,5,6])
def aipe(
    prox_oracle: Union[ProxOracle, WarmStartProxOracle],
    grad_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], float]] = None,
    warm_init: Optional[Array] = None,
) -> tuple[Array, int, Optional[Array], Array]:
    """Algorithm 1: Accelerated Inexact Proximal Extragradient.

    Returns ``(z_out, T, warm_final, total_inner_calls)`` where
    *total_inner_calls* is the accumulated inner oracle call count
    from prox_oracle invocations (0 when the prox_oracle does not
    report inner calls).
    """
    dtype = z0.dtype
    tiny = jnp.asarray(_TINY_AIPE, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    max_a = jnp.asarray(_MAX_A_PRIME, dtype=dtype)
    lam_tol = jnp.asarray(_REG_MIN_AIPE, dtype=dtype)
    int_zero = jnp.zeros((), dtype=jnp.int32)

    _has_warm = warm_init is not None

    # ── initial proximal oracle call (before loop) ────────────────
    if _has_warm:
        prox_result_0 = prox_oracle(z0, warm_init)
        z_tilde_0, u_0, warm_0 = prox_result_0[0], prox_result_0[1], prox_result_0[2]
        _has_inner_calls = len(prox_result_0) > 3
        init_inner_calls = prox_result_0[3] if _has_inner_calls else int_zero
    else:
        result = prox_oracle(z0)
        z_tilde_0, u_0 = result[0], result[1]
        warm_0 = jnp.zeros((), dtype=dtype)
        _has_inner_calls = len(result) > 3
        init_inner_calls = result[3] if _has_inner_calls else int_zero
    lam_0 = gamma * jnp.linalg.norm(z_tilde_0 - z0)
    lam_prime_init = jnp.maximum(lam_0, lam_tol)

    init = AIPEState(
        z=z0, v=z0,
        A=jnp.zeros((), dtype=dtype),
        lam_prime=lam_prime_init,
    )

    if _has_warm:
        def step(carry, t):
            s, warm, total_inner_calls = carry

            def _do_step(_):
                disc = one + 8.0 * s.lam_prime * s.A
                denom = jnp.maximum(4.0 * s.lam_prime, tiny)
                a_prime = (one + jnp.sqrt(jnp.maximum(disc, 0.0))) / denom
                a_prime = jnp.minimum(a_prime, max_a)
                A_prime = s.A + a_prime

                w_A = s.A / jnp.maximum(A_prime, tiny)
                w_a = a_prime / jnp.maximum(A_prime, tiny)
                z_bar = w_A * s.z + w_a * s.v

                def initial_prox(_):
                    if _has_inner_calls:
                        return z_tilde_0, u_0, warm_0, int_zero
                    return z_tilde_0, u_0, warm_0

                def loop_prox(_):
                    return prox_oracle(z_bar, warm)

                prox_result = jax.lax.cond(
                    t == 0, initial_prox, loop_prox, operand=None,
                )
                z_tilde, u, warm_new = prox_result[0], prox_result[1], prox_result[2]

                if _has_inner_calls:
                    step_inner_calls = prox_result[3]
                else:
                    step_inner_calls = int_zero

                lam = gamma * jnp.linalg.norm(z_tilde - z_bar)
                accept = lam <= s.lam_prime

                def accept_fn(_):
                    return (
                        A_prime,
                        z_tilde,
                        s.lam_prime / 2.0,
                        a_prime,
                    )

                def reject_fn(_):
                    gamma_t = s.lam_prime / jnp.maximum(lam, tiny)
                    a_r = gamma_t * a_prime
                    A_r = s.A + a_r
                    z_r = (
                        (one - gamma_t) * s.A / jnp.maximum(A_r, tiny) * s.z
                        + gamma_t * A_prime / jnp.maximum(A_r, tiny) * z_tilde
                    )
                    return (
                        A_r,
                        z_r,
                        2.0 * s.lam_prime,
                        a_r,
                    )

                A_new, z_new, lam_prime_new, a_step = jax.lax.cond(
                    accept, accept_fn, reject_fn, operand=None,
                )

                g = grad_fn(z_tilde)
                v_new = s.v - a_step * g
                if project is not None:
                    v_new = project(v_new)

                new_state = AIPEState(
                    z=z_new, v=v_new, A=A_new, lam_prime=lam_prime_new,
                )
                return (new_state, warm_new,
                        total_inner_calls + step_inner_calls), (z_tilde, z_new)

            return _do_step(None)

        scan_init = (init, warm_0, init_inner_calls)
        (final_state, warm_final, total_inner_calls), (all_z_tilde, all_z) = jax.lax.scan(
            step, scan_init, jnp.arange(T, dtype=jnp.int32),
        )
    else:
        def step(carry, t):
            s, total_inner_calls = carry

            def _do_step(_):
                disc = one + 8.0 * s.lam_prime * s.A
                denom = jnp.maximum(4.0 * s.lam_prime, tiny)
                a_prime = (one + jnp.sqrt(jnp.maximum(disc, 0.0))) / denom
                a_prime = jnp.minimum(a_prime, max_a)
                A_prime = s.A + a_prime

                w_A = s.A / jnp.maximum(A_prime, tiny)
                w_a = a_prime / jnp.maximum(A_prime, tiny)
                z_bar = w_A * s.z + w_a * s.v

                def initial_prox(_):
                    return z_tilde_0, u_0, int_zero

                def loop_prox(_):
                    result = prox_oracle(z_bar)
                    ic = result[3] if len(result) > 3 else int_zero
                    return result[0], result[1], ic

                prox_result = jax.lax.cond(
                    t == 0, initial_prox, loop_prox, operand=None,
                )
                z_tilde, u = prox_result[0], prox_result[1]
                step_inner_calls = prox_result[2]

                lam = gamma * jnp.linalg.norm(z_tilde - z_bar)
                accept = lam <= s.lam_prime

                def accept_fn(_):
                    return (
                        A_prime,
                        z_tilde,
                        s.lam_prime / 2.0,
                        a_prime,
                    )

                def reject_fn(_):
                    gamma_t = s.lam_prime / jnp.maximum(lam, tiny)
                    a_r = gamma_t * a_prime
                    A_r = s.A + a_r
                    z_r = (
                        (one - gamma_t) * s.A / jnp.maximum(A_r, tiny) * s.z
                        + gamma_t * A_prime / jnp.maximum(A_r, tiny) * z_tilde
                    )
                    return (
                        A_r,
                        z_r,
                        2.0 * s.lam_prime,
                        a_r,
                    )

                A_new, z_new, lam_prime_new, a_step = jax.lax.cond(
                    accept, accept_fn, reject_fn, operand=None,
                )

                g = grad_fn(z_tilde)
                v_new = s.v - a_step * g
                if project is not None:
                    v_new = project(v_new)

                new_state = AIPEState(
                    z=z_new, v=v_new, A=A_new, lam_prime=lam_prime_new,
                )
                return (new_state,
                        total_inner_calls + step_inner_calls), (z_tilde, z_new)

            return _do_step(None)

        (final_state, total_inner_calls), (all_z_tilde, all_z) = jax.lax.scan(
            step, (init, init_inner_calls), jnp.arange(T, dtype=jnp.int32),
        )
        # Bug A Fix: Return Python None when warm-start isn't being used
        warm_final = None

    # ── line 25 — output selection ────────────────────────────────
    if fn is not None:
        # Bug B Fix: Explicitly capture z_tilde_0 so early convergence doesn't drop it
        candidates = jnp.concatenate(
            [
                jnp.expand_dims(z0, 0), 
                jnp.expand_dims(z_tilde_0, 0), 
                all_z_tilde, 
                all_z
            ], 
            axis=0,
        )
        values = jax.vmap(fn)(candidates)
        z_out = candidates[jnp.argmin(values)]
    else:
        z_out = final_state.z

    return z_out, T, warm_final, total_inner_calls
    
# ── Algorithm 2 ────────────────────────────────────────────────────────────

def aipe_restart(
    prox_oracle: ProxOracle,
    grad_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    S: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], float]] = None,
) -> tuple[Array, int]:
    """Algorithm 2: AIPE with restart scheme.

    Each epoch halves ``‖z − z*‖`` (Theorem 4.1), so
    ``S = O(log(D/ε))`` epochs suffice for accuracy ε.

    Parameters
    ----------
    prox_oracle, grad_fn, gamma, project, fn
        Forwarded to :func:`aipe`.
    z0 : Array
        Initial iterate.
    T : int
        Iterations per epoch (static under JIT).
    S : int
        Number of restart epochs.

    Returns
    -------
    z_out : Array
        Approximate minimiser.
    oracle_calls : int
        Total proximal oracle calls (= S × T).
    """
    z = z0
    total_calls = 0

    for _ in range(S):
        result = aipe(
            prox_oracle, grad_fn, z, T, gamma,
            project=project, fn=fn,
        )
        z = result[0]
        calls = result[1]
        total_calls += calls

    return z, total_calls

__all__ = [
    "ProxOracle",
    "AIPEState",
    "aipe",
    "aipe_restart",
    "make_crn_prox_oracle",
]
