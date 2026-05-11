"""Newton Proximal Extragradient (NPE) method.

Implements Algorithm 6 (NPE) and Algorithm 7 (NPE-restart) from:
    Chen, Liu, Luo & Zhang (2025),
    "Solving Convex-Concave Problems with Õ(ε^{-4/7}) Second-Order Oracle Complexity."

Key design choices
──────────────────
* **Oracle abstraction** — NPE operates on *any* callable satisfying
  ``NPEOracle = Callable[[Array], tuple[Array, Array]]``; swap in
  stochastic, quasi-Newton, or custom oracles without touching the
  main loop logic.
* **JIT-native** — the main loop uses ``jax.lax.scan``;
  ``T`` must be a static integer under ``jax.jit``.
* **State dataclass** — loop state is a ``NamedTuple`` pytree, making
  it easy to inspect, log, or convert to ``jax.lax.while_loop``.
* **Output selection** — an optional ``fn`` oracle returns the iterate
  with smallest function value (mirrors AIPE's output-selection logic).
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

from minimax_aipe.oracles import crn_oracle


# ── public types ───────────────────────────────────────────────────────────

#: An NPE cubic-regularised Newton oracle: ``z ↦ (z_half, u)``.
#: The oracle's ``γ`` parameter and ``problem`` are baked into the closure.
NPEOracle = Callable[[Array], tuple[Array, Array]]


# ── loop state ─────────────────────────────────────────────────────────────

class NPEState(NamedTuple):
    """Carry state for the NPE scan loop (Algorithm 6)."""

    z: Array            # current iterate
    weighted_sum: Array # accumulator for η · z_{t+1/2}
    eta_sum: Array      # accumulator for η


# ── numerical guards ────────────────────────────────────────────────────────

# Maximum allowed step size η.  The theoretical formula η = 1/(2γ·‖z−z_{t+1/2}‖)
# can produce enormous values (≥ 1e14) when the half-step is extremely close to
# the current iterate (dist ≤ 1e-15).  This cap prevents overflow in the
# subsequent z-update ``z − η·F`` while being large enough that it does not
# affect convergence on well-conditioned problems.
_MAX_ETA = 1e12


# ── helpers ────────────────────────────────────────────────────────────────

def project_z(problem, z: Array) -> Array:
    """Project z = [x, y] onto Z = X × Y (product projection).

    Wrap as ``lambda z: project_z(problem, z)`` to obtain a
    ``Callable[[Array], Array]`` suitable for the ``project``
    parameter of :func:`npe`.
    """
    x, y = z[: problem.dim_x], z[problem.dim_x :]
    return jnp.concatenate([problem.project_x(x), problem.project_y(y)])


# ── oracle factory ─────────────────────────────────────────────────────────

def make_crn_npe_oracle(
    problem,
    gamma: float,
    tol: float = 0.0,
) -> NPEOracle:
    """Create a CRN-based cubic-regularised Newton oracle for NPE.

    Parameters
    ----------
    problem : MinimaxProblem
    gamma : float — cubic regularisation (typically ≈ 2ρ)
    tol : float — secular-equation convergence tolerance.
        When > 0, the CRN solver exits early once the secular equation
        residual converges, reducing wasted iterations when the subproblem
        is easy.
    """

    def oracle(z: Array) -> tuple[Array, Array]:
        return crn_oracle(problem, z, gamma, tol=tol)

    return oracle


# ── Algorithm 6 ────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=[0,1,3,5,6])
def npe(
    oracle: NPEOracle,
    F_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], float]] = None,
) -> tuple[Array, int]:
    """Algorithm 6 — Newton Proximal Extragradient.

    Each iteration:
        1. Calls the cubic-regularised Newton oracle once  →  z_{t+1/2}
        2. Computes step size         →  η_t = 1/(2γ‖z_t − z_{t+1/2}‖)
        3. Extrapolates               →  z_{t+1} = proj(z_t − η_t F(z_{t+1/2}))

    Returns the η-weighted average of the half-step iterates
    {z_{t+1/2}}, which is the quantity whose regret is bounded in
    Theorem E.2.  When *fn* is provided, the iterate with the
    smallest *fn* value among all candidates is returned instead.

    Parameters
    ----------
    oracle : NPEOracle
        ``(z) → (z_half, u)`` — cubic-regularised Newton step.
        Use :func:`make_crn_npe_oracle` to create a CRN-based oracle.
    F_fn : Callable
        Saddle-point operator ``F(z) = [∇_x L, −∇_y L]``.
    z0 : Array
        Initial iterate ``[x0, y0]``.
    T : int
        Number of iterations.  **Must be a concrete (static) integer
        under** ``jax.jit``.
    gamma : float
        Cubic regularisation (typically ≈ 2ρ).
    project : Callable or None
        Optional projection onto the feasible set Z.
    fn : Callable or None
        Optional function-value oracle for output selection.
        When provided, the iterate with smallest ``fn`` value among
        all ``z0``, ``z_half_1 … z_half_T``, ``z_1 … z_T`` is returned.

    Returns
    -------
    z_out : Array
        Approximate saddle point.
    oracle_calls : int
        Number of oracle invocations (= T).
    """
    dtype = z0.dtype
    tiny = jnp.asarray(1e-15, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    two_gamma = jnp.asarray(2.0 * gamma, dtype=dtype)
    max_eta = jnp.asarray(_MAX_ETA, dtype=dtype)

    init = NPEState(
        z=z0,
        weighted_sum=jnp.zeros_like(z0),
        eta_sum=jnp.zeros((), dtype=dtype),
    )

    def step(carry: NPEState, _unused):
        s = carry

        # Line 2: cubic-regularised Newton step
        z_half, _u = oracle(s.z)

        # Line 3: step size  η_t = 1 / (2γ ‖z_t − z_{t+1/2}‖)
        # When dist is extremely small (≤ tiny), η_t is clamped to 0
        # so the iterate does not move (the oracle returned a point
        # indistinguishable from the query, which is fine).
        dist = jnp.linalg.norm(s.z - z_half)
        inv_dist = jnp.where(dist > tiny, one / dist, jnp.zeros((), dtype=dtype))
        eta = jnp.minimum(inv_dist / two_gamma, max_eta)

        # Line 4: projected extragradient step
        F_half = F_fn(z_half)
        z_new = s.z - eta * F_half
        if project is not None:
            z_new = project(z_new)

        # Line 6: accumulate for η-weighted average
        new_carry = NPEState(
            z=z_new,
            weighted_sum=s.weighted_sum + eta * z_half,
            eta_sum=s.eta_sum + eta,
        )
        return new_carry, (z_half, z_new)

    final_state, (all_z_half, all_z) = jax.lax.scan(step, init, length=T)

    # ── output selection ──────────────────────────────────────────────
    if fn is not None:
        # candidates: z0, z_half_1 … z_half_T, z_1 … z_T  (2T + 1 total)
        candidates = jnp.concatenate(
            [jnp.expand_dims(z0, 0), all_z_half, all_z], axis=0,
        )
        values = jax.vmap(fn)(candidates)
        z_out = candidates[jnp.argmin(values)]
    else:
        # η-weighted average (Theorem E.2 output)
        z_out = jnp.where(
            final_state.eta_sum > tiny,
            final_state.weighted_sum / final_state.eta_sum,
            final_state.z,
        )

    return z_out, T


# ── Algorithm 7 ────────────────────────────────────────────────────────────

def npe_restart(
    oracle: NPEOracle,
    F_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    S: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], float]] = None,
) -> tuple[Array, int]:
    """Algorithm 7 — NPE with epoch restarts.

    Under μ-strong monotonicity (Assumption 5.1 with μ_x = μ_y = μ),
    each epoch halves ``‖z − z*‖`` (Theorem E.1), so
    ``S = ⌈log₂(d₀/ε)⌉`` epochs suffice for ``‖z^(S) − z*‖ ≤ ε``.

    Parameters
    ----------
    oracle, F_fn, gamma, project, fn
        Forwarded to :func:`npe`.
    z0 : Array
        Initial iterate.
    T : int
        Iterations per epoch (static under JIT).
    S : int
        Number of restart epochs.

    Returns
    -------
    z_out : Array
        Approximate saddle point.
    oracle_calls : int
        Total oracle invocations (≈ S × T).
    """
    z = z0
    total_calls = 0

    for _ in range(S):
        z, calls = npe(oracle, F_fn, z, T, gamma, project=project, fn=fn)
        total_calls += calls

    return z, total_calls

__all__ = [
    "NPEOracle",
    "NPEState",
    "npe",
    "npe_restart",
    "make_crn_npe_oracle",
    "project_z",
]
