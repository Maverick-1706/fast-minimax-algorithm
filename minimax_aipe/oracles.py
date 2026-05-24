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
import jax.scipy.linalg as jsp_linalg
from jax import Array

from minimax_aipe._precision import REG_MIN, TINY as _TINY
from minimax_aipe._compat import CRNResult
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

def _cholesky_solve(A: Array, b: Array) -> Array:
    """Solve A x = b via Cholesky (A must be SPD).

    ~2× faster and more stable than LU for symmetric positive-definite
    systems such as H + λI.
    """
    L = jnp.linalg.cholesky(A)
    y = jsp_linalg.solve_triangular(L, b, lower=True)
    x = jsp_linalg.solve_triangular(L.T, y, lower=False)
    return x

def _block_chol_solve(g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny):
    """Solve (J + λI)δ = −g for the minimax Jacobian via Schur complement.

    J = [[H_xx, H_xy], [−H_yx, −H_yy]].
    Converts to the symmetrised form and uses sign-aware block
    elimination with Cholesky on each (smaller) block.
    """
    d_x = H_xx.shape[0]
    g_x, g_y = g[:d_x], g[d_x:]

    # (2,2) block of the symmetrised system: K = H_yy − λI
    K = H_yy - lam * eye_y

    # Robust check: which sign requires a smaller Gershgorin shift to become SPD?
    dK = jnp.diag(K)
    oK = jnp.sum(jnp.abs(K), axis=1) - jnp.abs(dK)
    shift_pos = jnp.maximum(-jnp.min(dK - oK), 0.0)
    shift_neg = jnp.maximum(-jnp.min(-dK - oK), 0.0)
    neg = shift_neg < shift_pos

    def _solve(k_mat, sign_val):
        k_spd = k_mat + tiny * eye_y
        # Gershgorin safety for any residual indefiniteness
        dk = jnp.diag(k_spd)
        ok = jnp.sum(jnp.abs(k_spd), axis=1) - jnp.abs(dk)
        k_spd = k_spd + jnp.maximum(-jnp.min(dk - ok), 0.0) * eye_y

        Lk = jnp.linalg.cholesky(k_spd)

        def _sv(b):
            return jsp_linalg.solve_triangular(
                Lk.T,
                jsp_linalg.solve_triangular(Lk, b, lower=True),
                lower=False,
            )

        Ki_Hyx = sign_val * _sv(H_yx)
        Ki_gy = sign_val * _sv(g_y)

        # Schur complement: S = (H_xx + λI) − H_xy K⁻¹ H_yx
        S = (H_xx + lam * eye_x) - H_xy @ Ki_Hyx
        sd = jnp.diag(S)
        so = jnp.sum(jnp.abs(S), axis=1) - jnp.abs(sd)
        S = S + jnp.maximum(-jnp.min(sd - so) + tiny, 0.0) * eye_x

        Ls = jnp.linalg.cholesky(S)
        dx = jsp_linalg.solve_triangular(
            Ls.T,
            jsp_linalg.solve_triangular(Ls, -g_x - H_xy @ Ki_gy, lower=True),
            lower=False,
        )
        dy = Ki_gy - Ki_Hyx @ dx
        return jnp.concatenate([dx, dy])

    return jax.lax.cond(
        neg,
        lambda _: _solve(-K, -1.0),
        lambda _: _solve(K, 1.0),
        operand=None,
    )


def _safe_sym_chol_solve(A, b, tiny):
    """Solve Ax = b for a symmetric (possibly indefinite) A.

    Applies a Gershgorin-based diagonal shift to make A positive
    definite before using Cholesky.  When A is already SPD the
    shift is zero (identical to plain _cholesky_solve).
    """
    n = A.shape[0]
    dtype = A.dtype
    eye = jnp.eye(n, dtype=dtype)
    d = jnp.diag(A)
    o = jnp.sum(jnp.abs(A), axis=1) - jnp.abs(d)
    shift = jnp.maximum(-jnp.min(d - o) + tiny, 0.0)
    A_spd = A + shift * eye
    L = jnp.linalg.cholesky(A_spd)
    return jsp_linalg.solve_triangular(
        L.T,
        jsp_linalg.solve_triangular(L, b, lower=True),
        lower=False,
    )


def _stable_lam_update(lam: Array, lam_candidate: Array, i: Array) -> Array:
    """Secular-equation safeguard: clip to [0.5 λ, 2 λ] after first iter.

    Allow lambda to freely adapt initially (e.g., i < 3) before clipping.
    """
    dtype = lam.dtype
    reg_min = jnp.asarray(REG_MIN, dtype=dtype)
    safe_candidate = jnp.where(jnp.isfinite(lam_candidate), lam_candidate, 2.0 * lam)
    # FIX: Ensure lambda never collapses exactly to 0 (which causes Schur-complement precision NaNs)
    safe_candidate = jnp.maximum(safe_candidate, reg_min)
    
    return jnp.where(
        i < 3,
        safe_candidate,
        jnp.clip(safe_candidate, jnp.maximum(0.5 * lam, reg_min), 2.0 * lam),
    )


def _residual(g: Array, H: Array, d_eff: Array, lam: Array,
              out_dtype) -> Array:
    """Certificate  u = −(g + H δ + λδ)  computed in ``out_dtype`` (FP32-safe)."""
    u = -(g + H @ d_eff + lam * d_eff)
    return u.astype(out_dtype)

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
    x_bar, y_bar = _split(z_bar, problem.dim_x)
    g = problem.operator_F(z_bar)
    H = _build_jacobian(problem, x_bar, y_bar)
    (H_xx, H_xy), (H_yx, H_yy) = problem.hessian_f(x_bar, y_bar)
    d_x = problem.dim_x
    dtype = z_bar.dtype
    eye_x = jnp.eye(d_x, dtype=dtype)
    eye_y = jnp.eye(problem.dim_y, dtype=dtype)
    tiny = jnp.asarray(_TINY, dtype=dtype)

    lam_init = jnp.maximum(gamma / 2.0, jnp.asarray(REG_MIN, dtype=dtype))

    if tol > 0:
        def cond(state):
            lam, _z, i, prev_lam = state
            change = jnp.abs(lam - prev_lam)
            return (i < n_iters) & (change > jnp.maximum(tol * lam, tiny))

        def body(state):
            lam, _z, i, _prev = state
            delta = _block_chol_solve(
                g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny,
            )
            z_new = _project_z(problem, z_bar + delta)
            d_eff = z_new - z_bar
            lam_candidate = (gamma / 2.0) * jnp.linalg.norm(d_eff)
            return (_stable_lam_update(lam, lam_candidate, i), z_new, i + 1, lam)

        lam, z, n_secular, _p = jax.lax.while_loop(
            cond, body,
            (lam_init, z_bar, jnp.int32(0), jnp.asarray(-1.0, dtype=dtype)),
        )
    else:
        def body(i, state):
            lam, z = state
            delta = _block_chol_solve(
                g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny,
            )
            z_new = _project_z(problem, z_bar + delta)
            d_eff = z_new - z_bar
            lam_candidate = (gamma / 2.0) * jnp.linalg.norm(d_eff)
            return _stable_lam_update(lam, lam_candidate, i), z_new

        lam, z = jax.lax.fori_loop(0, n_iters, body, (lam_init, z_bar))
        n_secular = jnp.int32(n_iters)

    d_eff = z - z_bar
    u = _residual(g, H, d_eff, lam, dtype)
    stats = jnp.stack([jnp.int32(1), n_secular, jnp.int32(1)])
    return CRNResult(z, u, stats)

def crn_oracle_minimization(
    grad_fn: Callable[[Array], Array],
    hess_fn: Callable[[Array], Array],
    z_bar: Array,
    gamma: float,
    n_iters: int = 15,
    project: Optional[Callable[[Array], Array]] = None,
    tol: float = 0.0,
) -> tuple[Array, Array]:
    g = grad_fn(z_bar)
    H = hess_fn(z_bar)
    d = z_bar.shape[0]
    dtype = z_bar.dtype
    eye = jnp.eye(d, dtype=dtype)
    tiny = jnp.asarray(_TINY, dtype=dtype)

    lam_init = jnp.maximum(gamma / 2.0, jnp.asarray(REG_MIN, dtype=dtype))

    if tol > 0:
        def cond(state):
            lam, _z, i, prev_lam = state
            change = jnp.abs(lam - prev_lam)
            return (i < n_iters) & (change > jnp.maximum(tol * lam, tiny))

        def body(state):
            lam, _z, i, _prev = state
            delta = _safe_sym_chol_solve(H + lam * eye, -g, tiny)
            z_new = z_bar + delta
            if project is not None:
                z_new = project(z_new)
            d_eff = z_new - z_bar
            lam_candidate = (gamma / 2.0) * jnp.linalg.norm(d_eff)
            return (_stable_lam_update(lam, lam_candidate, i), z_new, i + 1, lam)

        lam, z, n_secular, _p = jax.lax.while_loop(
            cond, body,
            (lam_init, z_bar, jnp.int32(0), jnp.asarray(-1.0, dtype=dtype)),
        )
    else:
        def body(i, state):
            lam, z = state
            delta = _safe_sym_chol_solve(H + lam * eye, -g, tiny)
            z_new = z_bar + delta
            if project is not None:
                z_new = project(z_new)
            d_eff = z_new - z_bar
            lam_candidate = (gamma / 2.0) * jnp.linalg.norm(d_eff)
            return _stable_lam_update(lam, lam_candidate, i), z_new

        lam, z = jax.lax.fori_loop(0, n_iters, body, (lam_init, z_bar))
        n_secular = jnp.int32(n_iters)

    d_eff = z - z_bar
    u = _residual(g, H, d_eff, lam, dtype)
    stats = jnp.stack([jnp.int32(1), n_secular, jnp.int32(1)])
    return CRNResult(z, u, stats)

def lazy_crn_oracle(
    problem: MinimaxProblem,
    z_bar: Array,
    z_snapshot: Array,
    gamma: float,
    n_iters: int = 15,
    tol: float = 0.0,
) -> tuple[Array, Array]:
    x_ss, y_ss = _split(z_snapshot, problem.dim_x)
    g = problem.operator_F(z_bar)
    H = _build_jacobian(problem, x_ss, y_ss)
    (H_xx, H_xy), (H_yx, H_yy) = problem.hessian_f(x_ss, y_ss)
    d_x = problem.dim_x
    dtype = z_bar.dtype
    eye_x = jnp.eye(d_x, dtype=dtype)
    eye_y = jnp.eye(problem.dim_y, dtype=dtype)
    tiny = jnp.asarray(_TINY, dtype=dtype)
    lam_init = jnp.maximum(gamma / 2.0, jnp.asarray(REG_MIN, dtype=dtype))

    if tol > 0:
        def cond(state):
            lam, _z, i, prev_lam = state
            change = jnp.abs(lam - prev_lam)
            return (i < n_iters) & (change > jnp.maximum(tol * lam, tiny))

        def body(state):
            lam, _z, i, _prev = state
            delta = _block_chol_solve(
                g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny,
            )
            z_new = _project_z(problem, z_bar + delta)
            d_eff = z_new - z_bar
            lam_candidate = (gamma / 2.0) * jnp.linalg.norm(d_eff)
            return (_stable_lam_update(lam, lam_candidate, i), z_new, i + 1, lam)

        lam, z, n_secular, _p = jax.lax.while_loop(
            cond, body,
            (lam_init, z_bar, jnp.int32(0), jnp.asarray(-1.0, dtype=dtype)),
        )
    else:
        def body(i, state):
            lam, z = state
            delta = _block_chol_solve(
                g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny,
            )
            z_new = _project_z(problem, z_bar + delta)
            d_eff = z_new - z_bar
            lam_candidate = (gamma / 2.0) * jnp.linalg.norm(d_eff)
            return _stable_lam_update(lam, lam_candidate, i), z_new

        lam, z = jax.lax.fori_loop(0, n_iters, body, (lam_init, z_bar))
        n_secular = jnp.int32(n_iters)

    d_eff = z - z_bar
    u = _residual(g, H, d_eff, lam, dtype)
    stats = jnp.stack([jnp.int32(1), n_secular, jnp.int32(1)])
    return CRNResult(z, u, stats)


__all__ = [
    "eg_step",
    "crn_oracle",
    "crn_oracle_minimization",
    "lazy_crn_oracle",
]
