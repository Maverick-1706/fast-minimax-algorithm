# minimax_aipe/len.py
"""Lazy Extra Newton (LEN) method.

Implements Algorithm 8 (LEN) and Algorithm 9 (LEN-restart) from:
    Chen, Liu, Luo & Zhang (2025), "Solving Convex-Concave Problems with
    Õ(ε^{-4/7}) Second-Order Oracle Complexity."

Appendix E.2 — Guarantee of the LEN Subroutine.

LEN is the lazy-Hessian variant of NPE (Algorithm 6).  Instead of computing a
fresh Hessian at every iteration, LEN maintains a *snapshot* point ``z_snapshot``
whose Hessian is reused for *m* consecutive iterations.  The snapshot is
refreshed every *m* steps:

    π(t) = t − (t mod m)

which means ``z_snapshot = z_{π(t)}`` stays constant for each block of *m*
iterations, then jumps to the current iterate.

Lazy CRN oracle (Definition E.1)
--------------------------------
The core primitive is :func:`lazy_crn_oracle` (already in ``oracles.py``), which
solves the cubic-regularised Newton subproblem using ∇F evaluated at the
*snapshot* rather than the query point:

    ⟨F(z̄) + ∇F(z_ss)(z − z̄) + (γ/2)‖z − z̄‖²·I,  z′ − z⟩ ≥ 0

This trades one Hessian computation every *m* steps for a small increase in
total iteration count: from O(ε^{-2/3}) to O(m + m^{2/3}ε^{-2/3}) (Theorem E.3).
"""

from __future__ import annotations

import warnings
from typing import Callable, NamedTuple, Optional, Union
import builtins

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

from minimax_aipe.npe import project_z  # single source of truth
from minimax_aipe.oracles import lazy_crn_oracle
from minimax_aipe._compat import CallStats
from minimax_aipe.problem import MinimaxProblem

# ═══════════════════════════════════════════════════════════════════════════
# Public types
# ═══════════════════════════════════════════════════════════════════════════

#: A LEN cubic-regularised Newton oracle.
#:
#: Signature: ``(z: Array, z_snapshot: Array) → (z_half, u)``
#: or, when constructed with ``return_F=True``:
#: ``(z, z_snapshot) → (z_half, u, F_half)``.
#:
#: This is a *two-argument* callable, not single-argument — the snapshot
#: is supplied explicitly so the caller can vary it across iterations.
LENOracle = Callable[[Array, Array], Union[tuple[Array, Array],
                                            tuple[Array, Array, Array]]]


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics result
# ═══════════════════════════════════════════════════════════════════════════

class LENResult(NamedTuple):
    """Rich return value for :func:`len_loop` / :func:`len_restart`.

    Requested via ``return_full=True``.  The simple ``(z_out, oracle_calls)``
    tuple remains the default for backward compatibility.
    """
    z: Array                    #: Final iterate (or η-weighted average).
    oracle_calls: int            #: Total CRN oracle invocations.
    iterations: int              #: Total LEN iterations executed.
    snapshot_refreshes: int      #: Number of snapshot-block boundaries.
    num_rejected: int            #: Steps rejected by safety guards.
    final_gradient_norm: Array   #: ‖F(z_out)‖ (computed iff ``F_fn`` supplied).
    converged: bool              #: True if no safety guard fired.


# ═══════════════════════════════════════════════════════════════════════════
# Loop state
# ═══════════════════════════════════════════════════════════════════════════

class LENState(NamedTuple):
    """Carry state for the LEN scan loop (Algorithm 8).

    Mirrors :class:`minimax_aipe.npe.NPEState` with the addition of
    ``z_snapshot`` (the Hessian evaluation point) and ``t`` (iteration counter
    used for the snapshot schedule ``π(t) = t − t % m``).
    """
    z: Array                # current iterate
    z_snapshot: Array        # Hessian evaluation point (updated every m steps)
    t: int                   # iteration counter (scalar, static under scan)
    weighted_sum: Array      # accumulator for η · z_{t+1/2}
    eta_sum: Array           # accumulator for η
    best_z: Array            # best iterate found by ``fn`` (or z₀)
    best_fn: Array           # ``fn(best_z)`` value
    snapshot_refreshes: int  # how many times the snapshot was refreshed
    num_rejected: int        # steps rejected by safety guards
    stats: Array             # [crn_calls, linear_solves] accumulator


# ═══════════════════════════════════════════════════════════════════════════
# Numerical constants
# ═══════════════════════════════════════════════════════════════════════════

#: Hard clamp on step-size η.  Serves as a last-resort safety net;
#: the primary stabilisation is ``eta_floor`` (see :func:`len_loop`).
_MAX_ETA: float = 1e12

#: Default floor on ‖z − z_{t+1/2}‖ to keep η bounded.
_DEFAULT_ETA_FLOOR: float = 1e-8

#: Default threshold for ‖z‖ above which a step is considered diverged.
_DEFAULT_MAX_NORM: float = 1e10


# ═══════════════════════════════════════════════════════════════════════════
# Parameter validation
# ═══════════════════════════════════════════════════════════════════════════

def _validate_params(*, T: int, m: int, S: int = 1, gamma: float = 1.0) -> None:
    """Fail loud on invalid parameters (raises ``ValueError``)."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    
    import jax
    if not isinstance(gamma, jax.core.Tracer):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")


# ═══════════════════════════════════════════════════════════════════════════
# Oracle factory
# ═══════════════════════════════════════════════════════════════════════════

def make_lazy_crn_npe_oracle(
    problem: MinimaxProblem,
    gamma: float,
    n_iters: int = 15,
    *,
    return_F: bool = False,
    tol: float = 0.0,
) -> LENOracle:
    """Create a lazy-CRN oracle for LEN (Definition E.1).

    Unlike :func:`~minimax_aipe.npe.make_crn_npe_oracle` which evaluates ∇F at
    the query point, this oracle evaluates ∇F at a *snapshot* point supplied by
    the caller.

    The returned callable has signature
    ``(z, z_snapshot) → (z_half, u[, F_half])``.

    Parameters
    ----------
    problem : MinimaxProblem
    gamma : float
        Cubic regularisation (typically ≈ 2ρ for the subproblem, or
        ``2 * (ρ + γ_outer)`` inside the triple loop).
    n_iters : int
        Maximum secular-equation iterations inside each CRN call.
    return_F : bool
        If ``True``, the oracle also returns ``F_half = F(z_half)``, saving a
        separate operator evaluation inside :func:`len_loop`.
    tol : float
        Secular-equation convergence tolerance (forwarded to :func:`~minimax_aipe.oracles.lazy_crn_oracle`).

    Returns
    -------
    oracle : LENOracle
        ``(z, z_snapshot) → (z_half, u)`` or ``(z, z_snapshot) → (z_half, u, F_half)``.
    """
    def oracle(z: Array, z_snapshot: Array) -> Union[
        tuple[Array, Array], tuple[Array, Array, Array]
    ]:
        z_half, u, oracle_stats = lazy_crn_oracle(
            problem,
            z_bar=z,
            z_snapshot=z_snapshot,
            gamma=gamma,
            n_iters=n_iters,
            tol=tol,
        )
        if return_F:
            F_half = problem.operator_F(z_half)
            return z_half, u, F_half, oracle_stats
        return z_half, u, oracle_stats

    return oracle


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 8 — Lazy Extra Newton (single epoch)
# ═══════════════════════════════════════════════════════════════════════════

def _len_scan_loop(
    oracle: LENOracle,
    F_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    m: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], Array]] = None,
    *,
    adaptive_refresh: bool = False,
    staleness_threshold: float = 0.1,
    eta_floor: float = _DEFAULT_ETA_FLOOR,
    max_norm: float = _DEFAULT_MAX_NORM,
    safety_checks: bool = True,
    return_full: bool = False,
) -> Union[tuple[Array, int], LENResult]:

    """Algorithm 8 — Lazy Extra Newton (single epoch).

    Identical structure to :func:`minimax_aipe.npe.npe` except:

    1. The CRN oracle receives **two** arguments ``(z, z_snapshot)`` instead
       of one.
    2. The snapshot ``z_snapshot`` is refreshed every *m* iterations according
       to the schedule ``π(t) = t − t % m``.
    3. Numerics are hardened with ``eta_floor``, NaN guards, and norm-
       explosion detection.

    Parameters
    ----------
    oracle : LENOracle
        ``(z, z_snapshot) → (z_half, u)`` — lazy CRN oracle.
        Use :func:`make_lazy_crn_npe_oracle` to create one.  May also return
        a 3-tuple ``(z_half, u, F_half)`` to avoid recomputation.
    F_fn : callable
        Saddle-point operator ``F(z) = [∇_x L, −∇_y L]``.
    z0 : Array
        Initial iterate ``[x0, y0]``.
    T : int
        Number of iterations.  **Must be a concrete (static) integer under
        ``jax.jit``**.
    gamma : float
        Cubic regularisation parameter (must be > 0).
    m : int
        Hessian reuse interval.  The snapshot is refreshed every *m* steps.
        ``m = 1`` recovers standard NPE (fresh Hessian every step).
    project : callable or None
        Optional projection onto the feasible set Z.
    fn : callable or None
        Optional function-value oracle for output selection.  When provided,
        the iterate with smallest ``fn`` value among all candidates is
        returned (via running-best tracking, avoiding O(T) storage).
    eta_floor : float
        Floor on ``‖z − z_{t+1/2}‖`` before computing ``η``.  Prevents
        division-by-zero blow-up in a principled way.  Default ``1e-8``.
    max_norm : float
        If ``‖z_new‖`` exceeds this, the step is rejected (safety guard).
        Default ``1e10``.
    safety_checks : bool
        Enable NaN/Inf guards and norm-explosion detection.
    return_full : bool
        If ``True``, return :class:`LENResult` instead of ``(z_out, calls)``.
    adaptive_refresh : bool
        When ``True``, the Hessian snapshot is refreshed early whenever
        ``‖z_t − z_snapshot‖`` exceeds ``staleness_threshold``, in addition
        to the regular every-*m*-steps schedule.  ``m`` then serves as the
        *maximum* reuse interval (hard upper bound) rather than the fixed
        interval.  Default ``False`` (original behavior).
    staleness_threshold : float
        Threshold on ``‖z_t − z_snapshot‖`` above which the snapshot is
        refreshed early (only used when ``adaptive_refresh=True``).
        Default ``0.1``.

    Returns
    -------
    z_out : Array
        η-weighted average of half-step iterates (or best by *fn*).
    oracle_calls : int
        Number of lazy CRN oracle invocations (= T).
    (or :class:`LENResult` if ``return_full=True``)

    Notes
    -----
    **Snapshot schedule detail.**
    For iteration ``t`` (0-indexed), the snapshot used is the iterate at time
    ``π(t) = t − (t % m)``.  A new snapshot block starts at iterations
    ``0, m, 2m, 3m, ...``.  At the start of each block we replace
    ``z_snapshot ← z_current``; during the block ``z_snapshot`` is frozen.
    """
    _validate_params(T=T, m=m, gamma=gamma)

    dtype = z0.dtype
    two_gamma = jnp.asarray(2.0 * gamma, dtype=dtype)
    max_eta = jnp.asarray(_MAX_ETA, dtype=dtype)
    eta_floor_arr = jnp.asarray(eta_floor, dtype=dtype)
    max_norm_arr = jnp.asarray(max_norm, dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)

    # Initial best
    if fn is not None:
        best_fn_init = fn(z0)
    else:
        best_fn_init = jnp.asarray(jnp.inf, dtype=dtype)

    stats_zero = jnp.zeros(2, dtype=jnp.int32)
    init = LENState(
        z=z0,
        z_snapshot=z0,
        t=0,
        weighted_sum=jnp.zeros_like(z0),
        eta_sum=zero,
        best_z=z0,
        best_fn=best_fn_init,
        snapshot_refreshes=0,
        num_rejected=0,
        stats=stats_zero,
    )

    def step(carry: LENState, _unused):
        s = carry

                # ── snapshot schedule: periodic + optional adaptive refresh ──
        # Periodic: π(t) = t − (t % m).  When t % m == 0 a new block
        # starts and we replace the snapshot with the current iterate.
        # This is the hard upper bound — the snapshot is never more
        # than m steps stale.
        #
        # Adaptive: when enabled, the snapshot is also refreshed early
        # if the iterate has drifted too far from it (‖z − z_snapshot‖
        # exceeds ``staleness_threshold``).  This prevents divergence
        # in rapidly-changing landscapes while avoiding unnecessary
        # Hessian recomputations when the old one is still good.
        periodic_refresh = (s.t % m) == 0

        if adaptive_refresh:
            staleness = jnp.linalg.norm(s.z - s.z_snapshot)
            staleness_exceeded = staleness > jnp.asarray(
                staleness_threshold, dtype=dtype
            )
            refresh = periodic_refresh | staleness_exceeded
        else:
            refresh = periodic_refresh

        # Use jax.lax.cond so that only one branch is traced — important
        # when snapshot objects grow (cached factorisations, structured
        # state, distributed sharding).
        z_snapshot = jax.lax.cond(
            refresh,
            lambda _: s.z,
            lambda _: s.z_snapshot,
            operand=None,
        )
        snapshot_refreshes = s.snapshot_refreshes + jnp.where(
            refresh, 1, 0
        )


        # ── Line 2: lazy cubic-regularised Newton step ───────────────
        result = oracle(s.z, z_snapshot)
        if isinstance(result, tuple) and builtins.len(result) == 4:
            z_half, _u, F_half, oracle_stats = result   # type: ignore[misc]
        else:
            z_half, _u, oracle_stats = result[:3]       # type: ignore[misc]
            F_half = F_fn(z_half)

        # ── Line 3: step size η_t = 1 / (2γ · ‖z_t − z_{t+1/2}‖) ──
        # Mathematically η = 1/(2γ·‖Δ‖), but this explodes when ‖Δ‖→0.
        # Instead we clamp ‖Δ‖ from below by eta_floor, which is a
        # principled trust-region stabilisation rather than an arbitrary
        # hard clamp on η itself.
        dist = jnp.linalg.norm(s.z - z_half)
        dist = jnp.maximum(dist, eta_floor_arr)
        eta = jnp.minimum(one / (two_gamma * dist), max_eta)

        # ── Line 4: projected extragradient step ─────────────────────
        z_new_raw = s.z - eta * F_half
        if project is not None:
            z_new_raw = project(z_new_raw)

        # ── Safety guards ────────────────────────────────────────────
        if safety_checks:
            finite = jnp.all(jnp.isfinite(z_new_raw))
            not_exploded = jnp.linalg.norm(z_new_raw) < max_norm_arr
            step_ok = jnp.logical_and(finite, not_exploded)

            z_new = jnp.where(step_ok, z_new_raw, s.z)
            # If step was rejected, zero out η so it doesn't affect the
            # weighted average.
            eta = jnp.where(step_ok, eta, zero)
            rejected = jnp.where(step_ok, 0, 1)
        else:
            z_new = z_new_raw
            rejected = 0

        # ── Running-best tracking (fn output selection) ──────────────
        if fn is not None:
            # Evaluate fn at both half-step and full-step candidates.
            fn_half = fn(z_half)
            fn_new = fn(z_new)

            # Update running best: check z_half, z_new against current best.
            best_fn = s.best_fn
            best_z = s.best_z

            improve_half = fn_half < best_fn
            best_fn = jnp.where(improve_half, fn_half, best_fn)
            best_z = jnp.where(improve_half, z_half, best_z)

            improve_new = fn_new < best_fn
            best_fn = jnp.where(improve_new, fn_new, best_fn)
            best_z = jnp.where(improve_new, z_new, best_z)
        else:
            best_fn = s.best_fn
            best_z = s.best_z

        # ── Line 6: accumulate for η-weighted average ────────────────
        new_carry = LENState(
            z=z_new,
            z_snapshot=z_snapshot,
            t=s.t + 1,
            weighted_sum=s.weighted_sum + eta * z_half,
            eta_sum=s.eta_sum + eta,
            best_z=best_z,
            best_fn=best_fn,
            snapshot_refreshes=snapshot_refreshes,
            num_rejected=s.num_rejected + rejected,
            stats=s.stats + oracle_stats,
        )
        return new_carry, (z_half, z_new)

    final_state, (_all_z_half, _all_z) = jax.lax.scan(step, init, length=T)

    # ── output selection ────────────────────────────────────────────
    if fn is not None:
        z_out = final_state.best_z
    else:
        # For the standard η-weighted average we fall back to the current
        # iterate when η_sum ≈ 0 (can happen if all steps rejected).
        safe_denom = jnp.maximum(final_state.eta_sum, eta_floor_arr)
        z_out = jnp.where(
            final_state.eta_sum > eta_floor_arr,
            final_state.weighted_sum / safe_denom,
            final_state.z,
        )

    if return_full:
        # Compute final gradient norm (best-effort)
        final_grad = F_fn(z_out)
        final_gn = jnp.linalg.norm(final_grad)
        return LENResult(
            z=z_out,
            oracle_calls=int(final_state.stats[0]),
            iterations=T,
            snapshot_refreshes=int(final_state.snapshot_refreshes),
            num_rejected=int(final_state.num_rejected),
            final_gradient_norm=final_gn,
            converged=bool(final_state.num_rejected == 0),
        )

    return z_out, final_state.stats


# ── public alias ──────────────────────────────────────────────────────────

@partial(
    jax.jit,
    static_argnums=(0, 1, 3, 5, 6, 7),
        static_argnames=("project", "fn", "adaptive_refresh", "safety_checks", "return_full"),
)
def _len_impl(
    oracle: LENOracle,
    F_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    m: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], Array]] = None,
    *,
    adaptive_refresh: bool = False,
    staleness_threshold: float = 0.1,
    eta_floor: float = _DEFAULT_ETA_FLOOR,
    max_norm: float = _DEFAULT_MAX_NORM,
    safety_checks: bool = True,
    return_full: bool = False,
) -> Union[tuple[Array, int], LENResult]:
    """Public alias for :func:`_len_scan_loop` (Algorithm 8 — LEN).

    This is the recommended entry point.  ``len_loop`` is kept for backward
    compatibility; both call the same underlying implementation.
    
    Algorithm 8 — Lazy Extra Newton (single epoch).

    JIT-compiled by default.  ``safety_checks`` and ``return_full``
    are static keyword arguments — changing them triggers recompilation.
    ``eta_floor`` and ``max_norm`` are traced floats.
    Set ``JAX_DISABLE_JIT=1`` for eager execution during debugging.
    """
    return _len_scan_loop(
        oracle, F_fn, z0, T, gamma, m,
        project=project, fn=fn,
        adaptive_refresh=adaptive_refresh,
        staleness_threshold=staleness_threshold,
        eta_floor=eta_floor, max_norm=max_norm,
        safety_checks=safety_checks, return_full=return_full,
    )


# ── backward-compat alias ─────────────────────────────────────────────────

def len_loop(
    oracle: LENOracle,
    F_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    m: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], Array]] = None,
    *,
    adaptive_refresh: bool = False,
    staleness_threshold: float = 0.1,
    eta_floor: float = _DEFAULT_ETA_FLOOR,
    max_norm: float = _DEFAULT_MAX_NORM,
    safety_checks: bool = True,
    return_full: bool = False,
) -> Union[tuple[Array, int], LENResult]:
    """Backward-compatible wrapper — delegates to :func:`len`."""
    result = _len_scan_loop(
        oracle, F_fn, z0, T, gamma, m,
        project=project, fn=fn,
        adaptive_refresh=adaptive_refresh,
        staleness_threshold=staleness_threshold,
        eta_floor=eta_floor, max_norm=max_norm,
        safety_checks=safety_checks, return_full=return_full,
    )
    if return_full:
        return result
    z_out, stats = result
    return z_out, CallStats(stats)


def len(
    oracle: LENOracle,
    F_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    m: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], Array]] = None,
    *,
    adaptive_refresh: bool = False,
    staleness_threshold: float = 0.1,
    eta_floor: float = _DEFAULT_ETA_FLOOR,
    max_norm: float = _DEFAULT_MAX_NORM,
    safety_checks: bool = True,
    return_full: bool = False,
) -> Union[tuple[Array, CallStats], LENResult]:
    """Public Algorithm 8 wrapper with scalar-compatible call stats."""
    result = _len_impl(
        oracle, F_fn, z0, T, gamma, m,
        project=project, fn=fn,
        adaptive_refresh=adaptive_refresh,
        staleness_threshold=staleness_threshold,
        eta_floor=eta_floor, max_norm=max_norm,
        safety_checks=safety_checks, return_full=return_full,
    )
    if return_full:
        return result
    z_out, stats = result
    return z_out, CallStats(stats)


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 9 — LEN with epoch restarts
# ═══════════════════════════════════════════════════════════════════════════

def len_restart(
    oracle: LENOracle,
    F_fn: Callable[[Array], Array],
    z0: Array,
    T: int,
    gamma: float,
    m: int,
    S: int,
    project: Optional[Callable[[Array], Array]] = None,
    fn: Optional[Callable[[Array], Array]] = None,
    *,
    adaptive_refresh: bool = False,
    staleness_threshold: float = 0.1,
    eta_floor: float = _DEFAULT_ETA_FLOOR,
    max_norm: float = _DEFAULT_MAX_NORM,
    safety_checks: bool = True,
    return_full: bool = False,
) -> Union[tuple[Array, int], LENResult]:
    _validate_params(T=T, m=m, S=S, gamma=gamma)

    z = z0
    total_stats = jnp.zeros(2, dtype=jnp.int32)
    total_rejected = 0
    total_refreshes = 0
    all_converged = True

    for _ in range(S):
        result = _len_scan_loop(
            oracle, F_fn, z, T, gamma, m,
            project=project, fn=fn,
            adaptive_refresh=adaptive_refresh,
            staleness_threshold=staleness_threshold,
            eta_floor=eta_floor, max_norm=max_norm,
            safety_checks=safety_checks,
            return_full=return_full,
        )
        if return_full:
            z = result.z
            total_stats = total_stats + jnp.stack([jnp.int32(result.oracle_calls), jnp.int32(0)])
            total_rejected += result.num_rejected
            total_refreshes += result.snapshot_refreshes
            all_converged = all_converged and result.converged
        else:
            z, stats_arr = result  # type: ignore[misc]
            total_stats = total_stats + stats_arr

    if return_full:
        final_grad = F_fn(z)
        return LENResult(
            z=z,
            oracle_calls=int(total_stats[0]),
            iterations=int(total_stats[0]),
            snapshot_refreshes=total_refreshes,
            num_rejected=total_rejected,
            final_gradient_norm=jnp.linalg.norm(final_grad),
            converged=all_converged,
        )

    return z, CallStats(total_stats)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrapper for integration into the triple loop
# ═══════════════════════════════════════════════════════════════════════════

def make_len_saddle_solver(
    problem: MinimaxProblem,
    m: int,
) -> Callable[
    [MinimaxProblem, Array, float, object, str],
    tuple[Array, int],
]:
    """Create a LEN-based saddle subproblem solver.

    Compatible with ``minimax_aipe.minimax_aipe._solve_saddle_subproblem``.
    Returns a callable with the same signature as the existing ``npe_restart``
    path so the triple-loop framework can swap in LEN by passing
    ``M_saddle="len"``.

    Parameters
    ----------
    problem : MinimaxProblem
    m : int
        Hessian reuse interval (must be > 0).

    Returns
    -------
    solver : callable
        ``(h_problem, z0, gamma, params, M_saddle) → (z_hat, calls)``
    """
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")

    def solver(
        h_problem: MinimaxProblem,
        z0: Array,
        gamma: float,
        params,  # _LoopParams
        M_saddle: str,
    ) -> tuple[Array, int]:
        from minimax_aipe.npe import project_z as pz

        sub_rho = max(float(h_problem.rho or 0.0), 1e-6)
        len_gamma = 2.0 * sub_rho

        oracle = make_lazy_crn_npe_oracle(h_problem, len_gamma)

        from minimax_aipe._precision import ABS_TOL as _ABS_TOL
        # Theorem E.1: NPE-restart requires T = O((2ρ_sub / μ)^(2/3))
        # CRN oracle calls per epoch, where len_gamma = 2ρ_sub is the
        # CRN cubic parameter and gamma (outer) provides μ-strong
        # monotonicity.  Diameter D affects only the number of restart
        # epochs S via log(D/ε), not the per-epoch count T.
        condition_ratio = len_gamma / max(gamma, _ABS_TOL)
        complexity = condition_ratio ** (2.0 / 3.0)
        inner_T = max(8, min(int(round(complexity)), params.T_inner))

        return len_restart(
            oracle,
            h_problem.operator_F,
            z0,
            T=inner_T,
            gamma=len_gamma,
            m=m,
            S=params.S_inner,
            project=lambda z: pz(h_problem, z),
            fn=lambda z: jnp.dot(
                h_problem.operator_F(z), h_problem.operator_F(z)
            ),
        )

    return solver

__all__ = [
    "LENOracle",
    "LENResult",
    "len_loop",
    "len_restart",
    "make_lazy_crn_npe_oracle",
    "make_len_saddle_solver",
]
