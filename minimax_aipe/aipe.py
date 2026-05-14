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

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

from minimax_aipe.oracles import crn_oracle_minimization


# ── public types ───────────────────────────────────────────────────────────

#: A proximal oracle: ``z_bar ↦ (z_tilde, u)`` satisfying Definition 4.1.
#: The oracle's ``(γ, δ)`` parameters are baked into the closure.
ProxOracle = Callable[[Array], tuple[Array, Array]]


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
    prox_oracle: ProxOracle,
    grad_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], float]] = None,
) -> tuple[Array, int]:
    """Algorithm 1: Accelerated Inexact Proximal Extragradient.

    Solves ``min_{z ∈ Z} h(z)`` for a convex function *h* using an
    accelerated scheme with inexact second-order proximal oracles.

    Parameters
    ----------
    prox_oracle : ProxOracle
        ``(z_bar) → (z_tilde, u)`` satisfying Definition 4.1.
        Use :func:`make_crn_prox_oracle` to create a CRN-based oracle.
    grad_fn : Callable
        Gradient of *h* (used for the v-update, line 22 of Alg 1).
    z0 : Array
        Initial iterate.
    T : int
        Number of iterations.  **Must be a concrete (static) integer
        under** ``jax.jit``.
    gamma : float
        Regularisation parameter for the proximal sub-problems.
    project : Callable or None
        Optional feasible-set projection for the v-update.
    fn : Callable or None
        Optional function-value oracle for output selection (line 25).
        When provided, the iterate with smallest ``fn`` value among
        all ``z_t``, ``z̃_t``, and ``z0`` is returned.

    Returns
    -------
    z_out : Array
        Approximate minimiser.
    oracle_calls : int
        Number of proximal oracle invocations (= T).
    """
    dtype = z0.dtype
    tiny = jnp.asarray(1e-12, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    max_a = jnp.asarray(_MAX_A_PRIME, dtype=dtype)

    # ── initial proximal oracle call (before loop) ────────────────────
    z_tilde_0, u_0 = prox_oracle(z0)
    lam_0 = gamma * jnp.linalg.norm(z_tilde_0 - z0)
    lam_prime_init = jnp.maximum(lam_0, tiny)

    # ── main loop via jax.lax.scan ────────────────────────────────────
    init = AIPEState(
        z=z0, v=z0,
        A=jnp.zeros((), dtype=dtype),        # ← dtype explicit (was float32)
        lam_prime=lam_prime_init,
    )

    def step(carry: AIPEState, t):
        s = carry

        # line 5 — solve  2λ'(a')² − a' − A = 0  (positive root)
        disc = one + 8.0 * s.lam_prime * s.A
        denom = jnp.maximum(4.0 * s.lam_prime, tiny)
        a_prime = (one + jnp.sqrt(jnp.maximum(disc, 0.0))) / denom
        # Clamp a' to prevent overflow in downstream arithmetic
        a_prime = jnp.minimum(a_prime, max_a)
        A_prime = s.A + a_prime

        # line 7 — extrapolation point  z̄_t
        w_A = s.A / jnp.maximum(A_prime, tiny)
        w_a = a_prime / jnp.maximum(A_prime, tiny)
        z_bar = w_A * s.z + w_a * s.v

        # Lines 8-10: reuse the initial prox call at t=0.  Algorithm 1
        # calls iProx before the loop, then only calls it inside the loop
        # when t > 0.
        def initial_prox(_):
            return z_tilde_0, u_0

        def loop_prox(_):
            return prox_oracle(z_bar)

        z_tilde, u = jax.lax.cond(t == 0, initial_prox, loop_prox, operand=None)
        lam = gamma * jnp.linalg.norm(z_tilde - z_bar)

        # lines 12-21 — conditional update (accept / reject)
        accept = lam <= s.lam_prime

        def accept_fn(_):
            return (
                A_prime,                # A_new = A + a'
                z_tilde,                # z_new = z̃_{t+1}
                s.lam_prime / 2.0,      # λ'_{t+1} = λ'_t / 2
                a_prime,                # a_{t+1}
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
                2.0 * s.lam_prime,      # λ'_{t+1} = 2·λ'_t
                a_r,                    # a_{t+1}
            )

        A_new, z_new, lam_prime_new, a_step = jax.lax.cond(
            accept, accept_fn, reject_fn, operand=None,
        )

        # Lines 22-23: v-update with the accepted/scaled a_{t+1}.
        g = grad_fn(z_tilde)
        v_new = s.v - a_step * (g + u)
        if project is not None:
            v_new = project(v_new)

        new_state = AIPEState(
            z=z_new, v=v_new, A=A_new, lam_prime=lam_prime_new,
        )
        # accumulate candidates for output selection
        return new_state, (z_tilde, z_new)

    final_state, (all_z_tilde, all_z) = jax.lax.scan(
        step, init, jnp.arange(T, dtype=jnp.int32)
    )

    # ── line 25 — output selection ────────────────────────────────────
    if fn is not None:
        # candidates: z0, z̃₁ … z̃_T, z₁ … z_T   (2T + 1 total)
        candidates = jnp.concatenate(
            [jnp.expand_dims(z0, 0), all_z_tilde, all_z], axis=0,
        )
        values = jax.vmap(fn)(candidates)
        z_out = candidates[jnp.argmin(values)]
    else:
        z_out = final_state.z

    return z_out, T


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
        Total proximal oracle calls (≈ S × (T + 1)).
    """
    z = z0
    total_calls = 0

    for _ in range(S):
        z, calls = aipe(
            prox_oracle, grad_fn, z, T, gamma,
            project=project, fn=fn,
        )
        total_calls += calls

    return z, total_calls

__all__ = [
    "ProxOracle",
    "AIPEState",
    "aipe",
    "aipe_restart",
    "make_crn_prox_oracle",
]
