"""High-level Minimax-AIPE framework.

Wires the lower-level AIPE, NPE, LEN, ALEN, CRN, and EG primitives into a
usable solver entry point and exposes the surrogate-problem constructors used
by the three-loop Minimax-AIPE reduction.

Architecture (the full triple loop)
------------------------------------
Algorithm 3 (outer) — AIPE minimises Φ(x) = max_y f(x,y)
    │                  via aipe_restart + inexact proximal oracle for Φ.
    │
    └── Algorithm 4 (middle) — Inexact proximal oracle for Φ.
            Solves min_x max_y g(x,y;x_bar) by running AIPE on -Ψ:
                │      aipe_restart minimises -Ψ(y) = -min_x g(x,y;x_bar)
                │
                └── Algorithm 5 (inner) — Inexact proximal oracle for -Ψ.
                        Solves the regularised saddle subproblem
                        min_x max_y h(x,y;x_bar,y_bar) via NPE/LEN-restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil, log2
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import Array

from minimax_aipe.aipe import ProxOracle, aipe, aipe_restart
from minimax_aipe.alen import (
    aipe_restart_lazy,
    make_lazy_crn_prox_oracle,
    maximize_y_alen,
    minimize_x_alen,
)
from minimax_aipe.gap import estimate_gap
from minimax_aipe.len import len_loop, len_restart, make_lazy_crn_npe_oracle
from minimax_aipe.npe import make_crn_npe_oracle, npe, npe_restart, project_z
from minimax_aipe.oracles import _block_chol_solve, _stable_lam_update, eg_step
from minimax_aipe.problem import MinimaxProblem, OracleStats, SolverResult
from minimax_aipe._precision import (
    ABS_TOL as _ABS_TOL,
    CUBIC_ZERO as _CUBIC_ZERO,
    GAP_FLOOR as _GAP_FLOOR,
    REG_MIN as _REG_MIN,
    TINY as _TINY,
)
from minimax_aipe._compat import CRNResult, CallStats


logger = logging.getLogger(__name__)


def _stats_array(value):
    return value.stats if isinstance(value, CallStats) else value


# ═════════════════════════════════════════════════════════════════════════════
# Block Schur-complement linear solve  
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# Mutable call counter
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _CallCounter:
    """Simple mutable call counter for bookkeeping across nested loops.

    Not JIT-safe — must be used outside traced code (which is safe here
    because ``aipe_restart`` itself is an eager Python loop).
    """
    total: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# JIT-stable pipeline cache
# ═════════════════════════════════════════════════════════════════════════════

class _CachedPipeline:
    """Provides stable Python identities for closures passed to ``@jax.jit``.

    ``aipe()`` and ``npe()`` mark their callable arguments as
    ``static_argnums``.  JAX caches JIT compilations by the *identity*
    of each static arg.  When ``solve()`` is called in a tight loop
    (e.g. benchmark timing), creating fresh closures every time forces a
    full re-compilation on every call.

    ``_CachedPipeline`` sidesteps this by building the kernel, Φ-oracles,
    and the outer proximal-oracle **once** and exposing them as *bound
    methods* whose identity is fixed for the lifetime of the instance.

    Call counts are threaded through JAX return values (pure dataflow)
    rather than via ``jax.debug.callback`` side-effects, ensuring
    correct oracle-call tracking inside JIT-traced loops.
    """

    def __init__(self, problem, gamma, params, M_saddle):
        self.problem = problem
        self.gamma = gamma
        self.params = params
        self.M_saddle = M_saddle
        self.kernel = RegularizedSubproblem(problem, gamma)
        self.phi_fn, self.grad_phi_fn = _make_phi_oracle(
            problem, gamma, params,
            M_saddle=M_saddle, m_lazy=params.m_lazy,
        )

    # -- bound methods (stable Python identity) -------------------------

    def prox_phi(self, x_bar: Array, y_init: Optional[Array] = None
                 ) -> tuple[Array, Array, Array, Array]:
        """Stable-identity proximal oracle for Φ (Algorithm 4).

        Returns ``(x_out, u_out, y_hat, inner_calls)`` where *y_hat* is
        the recovered dual variable usable as warm-start for the next
        call and *inner_calls* is the accumulated inner oracle count.
        """
        x_out, u_out, y_hat, inner_calls = _iProx_Phi(
            self.problem, x_bar, self.gamma,
            zeta_2=self.params.zeta_2,
            params=self.params,
            M_saddle=self.M_saddle,
            y_init=y_init,
            kernel=self.kernel,
        )
        return x_out, u_out, y_hat, inner_calls

    def run_outer_epoch(self, x_cur: Array, warm_y: Optional[Array] = None
                        ) -> tuple[Array, int, Array, Array]:
        """Stable-identity epoch function for the outer AIPE loop.

        Returns ``(x_out, calls, warm_y_new, inner_calls)`` for
        warm-start threading and oracle-call accumulation.
        """
        result = aipe(
            self.prox_phi, self.grad_phi_fn, x_cur,
            self.params.T_outer, self.gamma,
            project=self.problem.project_x,
            warm_init=warm_y,
        )
        x_out, calls, warm_y_new, inner_calls = (
            result[0], result[1], result[2], result[3],
        )
        return x_out, calls, warm_y_new, inner_calls


import functools

@functools.lru_cache(maxsize=1)
def _get_pipeline(problem, gamma, params, M_saddle):
    """Retrieve or create a cached :class:`_CachedPipeline`."""
    return _CachedPipeline(problem, gamma, params, M_saddle)


# ═════════════════════════════════════════════════════════════════════════════
# Loop parameters
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _LoopParams:
    T_outer: int
    S_outer: int
    T_middle: int
    S_middle: int
    T_inner: int
    S_inner: int
    zeta_1: float
    zeta_2: float
    zeta_3: float
    m_lazy: int = 5


# ═════════════════════════════════════════════════════════════════════════════
# Cubic regularisation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cubic_grad(delta: Array, gamma: float) -> Array:
    """Gradient of ``(γ/3)·‖δ‖³``."""
    delta = jnp.asarray(delta)
    return gamma * jnp.linalg.norm(delta) * delta


def _cubic_hess(delta: Array, gamma: float) -> Array:
    """Hessian of ``(γ/3)·‖δ‖³`` with zero-limit branch."""
    delta = jnp.asarray(delta)
    norm = jnp.linalg.norm(delta)
    eye = jnp.eye(delta.shape[0], dtype=delta.dtype)
    safe_norm = jnp.maximum(norm, jnp.asarray(_CUBIC_ZERO, dtype=delta.dtype))
    hess = gamma * (norm * eye + jnp.outer(delta, delta) / safe_norm)
    return jnp.where(norm > _CUBIC_ZERO, hess, jnp.zeros_like(hess))


# ═════════════════════════════════════════════════════════════════════════════
# Regularised subproblem kernel  (public, reusable)
# ═════════════════════════════════════════════════════════════════════════════

class RegularizedSubproblem:
    """Reusable kernel for the regularised h-subproblem.

    Constructed **once** per ``(problem, gamma)`` pair and threaded through
    the triple loop.  The parametric methods
    (:meth:`operator_F_h`, :meth:`jacobian_F_h`, :meth:`make_crn_oracle`)
    share a *fixed Python identity* across all calls so that JAX compiles
    each underlying function exactly once, regardless of how many times the
    solver invokes them with different ``(x_bar, y_bar)`` arguments.

    Users can also construct this independently to experiment with the
    regularised saddle subproblem outside the full Minimax-AIPE pipeline::

        kernel = RegularizedSubproblem(my_problem, gamma=0.1)
        z = jnp.concatenate([x0, y0])
        F = kernel.operator_F_h(z, x_bar, y_bar)
        H = kernel.jacobian_F_h(x, y, x_bar, y_bar)

    Parameters
    ----------
    problem : MinimaxProblem
        The base (unregularised) minimax problem.
    gamma : float
        Cubic regularisation strength.
    """

    __slots__ = (
        "_problem", "_gamma",
        "dim_x", "dim_y", "D_x", "D_y",
        "rho_h", "ell_h",
        "project_x", "project_y",
    )

    def __init__(self, problem: MinimaxProblem, gamma: float) -> None:
        self._problem = problem
        self._gamma = gamma
        self.dim_x = problem.dim_x
        self.dim_y = problem.dim_y
        self.D_x = problem.D_x
        self.D_y = problem.D_y
        diameter = max(problem.D_x, problem.D_y, 1.0)
        #  Hessian Lipschitz of (γ/3)||·||³ is 2γ (Lemma 3.3),
        #          so ρ_h = ρ + 2γ.  Theorem 5.3 confirms inner loop uses
        #          T_saddle(ρ+2γ, γ/2).
        self.rho_h = (problem.rho or 0.0) + 2 * gamma
        self.ell_h = (problem.ell or 0.0) + 2 * gamma * diameter
        self.project_x = problem.project_x
        self.project_y = problem.project_y

    # ── repr ─────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"RegularizedSubproblem(dim=({self.dim_x},{self.dim_y}), "
            f"gamma={self._gamma:.4e}, rho_h={self.rho_h:.4e})"
        )

    @property
    def gamma(self) -> float:
        """The cubic regularisation strength."""
        return self._gamma

    @property
    def base_problem(self) -> MinimaxProblem:
        """The underlying unregularised minimax problem."""
        return self._problem

    # ── parametric operator F_h(z; x_bar, y_bar) ────────────────────
    def operator_F_h(self, z: Array, x_bar: Array, y_bar: Array) -> Array:
        r"""Evaluate the monotone operator of the h-subproblem.

        .. math::

            F_h(z) = \begin{bmatrix} \nabla_x h \\ -\nabla_y h \end{bmatrix}

        where
        :math:`h(x,y) = f(x,y) + \frac{\gamma}{3}\|x - \bar{x}\|^3
        - \frac{\gamma}{3}\|y - \bar{y}\|^3`.

        Parameters
        ----------
        z : Array, shape ``(dim_x + dim_y,)``
            Joint iterate.
        x_bar, y_bar : Array
            Regularisation centres.

        Returns
        -------
        Array, shape ``(dim_x + dim_y,)``
        """
        x, y = z[: self.dim_x], z[self.dim_x :]
        gx, gy_neg = self._problem.grad_f(x, y)
        gx_h = gx + _cubic_grad(x - x_bar, self._gamma)
        gy_neg_h = gy_neg + _cubic_grad(y - y_bar, self._gamma)
        return jnp.concatenate([gx_h, gy_neg_h])

    # ── parametric Jacobian of F_h ───────────────────────────────────
    def jacobian_F_h(
        self, x: Array, y: Array, x_bar: Array, y_bar: Array
    ) -> Array:
        r"""Jacobian of the monotone operator of the h-subproblem.

        Assembles the block matrix

        .. math::

            \nabla F_h = \begin{bmatrix}
                H_{xx}^h & H_{xy} \\
                -H_{yx} & -H_{yy}^h
            \end{bmatrix}

        where the diagonal blocks include the cubic-correction Hessians.

        Parameters
        ----------
        x, y : Array
            Current iterate components.
        x_bar, y_bar : Array
            Regularisation centres.

        Returns
        -------
        Array, shape ``(dim_x + dim_y, dim_x + dim_y)``
        """
        (H_xx, H_xy), (H_yx, H_yy) = self._problem.hessian_f(x, y)
        H_xx_h = H_xx + _cubic_hess(x - x_bar, self._gamma)
        H_yy_h = H_yy - _cubic_hess(y - y_bar, self._gamma)
        top = jnp.concatenate([H_xx_h, H_xy], axis=1)
        bot = jnp.concatenate([-H_yx, -H_yy_h], axis=1)
        return jnp.concatenate([top, bot], axis=0)

    # ── raw Hessian blocks for _block_chol_solve ─────────────────────
    def hessian_blocks_h(
        self, x: Array, y: Array, x_bar: Array, y_bar: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        r"""Raw Hessian blocks of the h-subproblem (no Jacobian sign flip).

        Returns ``((H_xx_h, H_xy), (H_yx, H_yy_h))`` — the *unsigned*
        second-order blocks of ``h(x,y; x_bar, y_bar)``.  These are the
        blocks that :func:`_block_chol_solve` expects: that function
        internally accounts for the ``[[-H_yx, -H_yy]]`` sign structure
        of the minimax Jacobian.

        Parameters
        ----------
        x, y : Array
            Current iterate components.
        x_bar, y_bar : Array
            Regularisation centres.

        Returns
        -------
        ((H_xx_h, H_xy), (H_yx, H_yy_h))
        """
        (H_xx, H_xy), (H_yx, H_yy) = self._problem.hessian_f(x, y)
        H_xx_h = H_xx + _cubic_hess(x - x_bar, self._gamma)
        H_yy_h = H_yy - _cubic_hess(y - y_bar, self._gamma)
        return (H_xx_h, H_xy), (H_yx, H_yy_h)

    # ── joint projection ─────────────────────────────────────────────
    def project(self, z: Array) -> Array:
        """Project a joint iterate onto ``D_x × D_y``."""
        return jnp.concatenate([
            self.project_x(z[: self.dim_x]),
            self.project_y(z[self.dim_x :]),
        ])

    # ── on-the-fly MinimaxProblem for the h-subproblem ──────────────
    def make_h_problem(
        self, x_bar: Array, y_bar: Array
    ) -> MinimaxProblem:
        """Build a :class:`MinimaxProblem` for ``h`` at fixed centres.

        This is useful for experimentation or when the full
        ``MinimaxProblem`` interface (e.g. ``operator_F``, ``project_z``)
        is needed alongside the parametric kernel.

        Parameters
        ----------
        x_bar, y_bar : Array
            Regularisation centres.

        Returns
        -------
        MinimaxProblem
        """
        return _make_h_problem(self._problem, x_bar, y_bar, self._gamma)

    # ── CRN oracle for NPE (parametric) ─────────────────────────────
    def make_crn_oracle(
        self,
        x_bar: Array,
        y_bar: Array,
        npe_gamma: float,
        n_iters: int = 15,
        tol: float = 0.0,
    ) -> Callable[[Array], tuple[Array, Array]]:
        """Return a CRN NPE oracle bound to fixed ``(x_bar, y_bar)``.

        Uses block Schur-complement solves  and initialises λ at
        ``npe_gamma / 2``  for immediate regularisation on the
        first iteration, consistent with the standalone ``crn_oracle``
        in ``oracles.py``.
        """
        dim_x = self.dim_x
        _tiny = _TINY
        gamma = self._gamma

        def _project_z_h(z: Array) -> Array:
            xz, yz = z[:dim_x], z[dim_x:]
            return jnp.concatenate([
                self.project_x(xz), self.project_y(yz),
            ])

        if tol > 0:
            def oracle(z_bar: Array) -> tuple[Array, Array]:
                g = self.operator_F_h(z_bar, x_bar, y_bar)
                xb, yb = z_bar[:dim_x], z_bar[dim_x:]
                # Jacobian (with sign flips) — used only for the residual
                H = self.jacobian_F_h(xb, yb, x_bar, y_bar)
                # Raw Hessian blocks — the format _block_chol_solve expects
                (H_xx, H_xy), (H_yx, H_yy) = self.hessian_blocks_h(
                    xb, yb, x_bar, y_bar,
                )
                
                dtype = z_bar.dtype
                tiny = jnp.asarray(_tiny, dtype=dtype)
                tol_jax = jnp.asarray(tol, dtype=dtype)
                eye_x = jnp.eye(dim_x, dtype=dtype)
                eye_y = jnp.eye(self.dim_y, dtype=dtype)
                lam0 = jnp.asarray(npe_gamma / 2.0, dtype=dtype)

                def cond(state):
                    lam, _z, i, prev_lam = state
                    change = jnp.abs(lam - prev_lam)
                    return (i < n_iters) & (change > jnp.maximum(tol_jax * lam, tiny))

                def body(state):
                    lam, _z, i, _prev = state
                    delta = _block_chol_solve(g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny)
                    z_new = _project_z_h(z_bar + delta)
                    d_eff = z_new - z_bar
                    lam_candidate = (npe_gamma / 2.0) * jnp.linalg.norm(d_eff)
                    return (
                        _stable_lam_update(lam, lam_candidate, i),
                        z_new, i + 1, lam,
                    )

                lam, z, n_secular, _p = jax.lax.while_loop(
                    cond, body,
                    (lam0, z_bar,
                     jnp.int32(0), jnp.asarray(-1.0, dtype=dtype)),
                )
                d_eff = z - z_bar
                u = -(g + H @ d_eff + lam * d_eff)
                stats = jnp.stack([jnp.int32(1), n_secular, jnp.int32(1)])
                return CRNResult(z, u, stats)
        else:
            def oracle(z_bar: Array) -> tuple[Array, Array]:
                g = self.operator_F_h(z_bar, x_bar, y_bar)
                xb, yb = z_bar[:dim_x], z_bar[dim_x:]
                # Jacobian (with sign flips) — used only for the residual
                H = self.jacobian_F_h(xb, yb, x_bar, y_bar)
                # Raw Hessian blocks — the format _block_chol_solve expects
                (H_xx, H_xy), (H_yx, H_yy) = self.hessian_blocks_h(
                    xb, yb, x_bar, y_bar,
                )
                
                dtype = z_bar.dtype
                tiny = jnp.asarray(_tiny, dtype=dtype)
                eye_x = jnp.eye(dim_x, dtype=dtype)
                eye_y = jnp.eye(self.dim_y, dtype=dtype)
                lam0 = jnp.asarray(npe_gamma / 2.0, dtype=dtype)

                def body(i, state):
                    lam, z = state
                    delta = _block_chol_solve(g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny)
                    z_new = _project_z_h(z_bar + delta)
                    d_eff = z_new - z_bar
                    lam_candidate = (npe_gamma / 2.0) * jnp.linalg.norm(d_eff)
                    return _stable_lam_update(lam, lam_candidate, i), z_new

                lam, z = jax.lax.fori_loop(
                    0, n_iters, body, (lam0, z_bar)
                )
                d_eff = z - z_bar
                u = -(g + H @ d_eff + lam * d_eff)
                stats = jnp.stack([jnp.int32(1), jnp.int32(n_iters), jnp.int32(1)])
                return CRNResult(z, u, stats)

        return oracle


# Backward-compatible alias
_HKernel = RegularizedSubproblem


# ═════════════════════════════════════════════════════════════════════════════
# Surrogate problem constructors
# ═════════════════════════════════════════════════════════════════════════════

def _make_g_problem(
    problem: MinimaxProblem, x_bar: Array, gamma: float,
) -> MinimaxProblem:
    """Construct ``g(x,y;x̄) = f(x,y) + (γ/3)·‖x−x̄‖³``."""
    x_bar = jnp.asarray(x_bar)

    def f_g(x: Array, y: Array):
        dx = x - x_bar
        return problem.f(x, y) + (gamma / 3.0) * jnp.linalg.norm(dx) ** 3

    def grad_g(x: Array, y: Array) -> tuple[Array, Array]:
        gx, gy_neg = problem.grad_f(x, y)
        return gx + _cubic_grad(x - x_bar, gamma), gy_neg

    def hess_g(x: Array, y: Array):
        (H_xx, H_xy), (H_yx, H_yy) = problem.hessian_f(x, y)
        H_xx_g = H_xx + _cubic_hess(x - x_bar, gamma)
        return (H_xx_g, H_xy), (H_yx, H_yy)

    return MinimaxProblem(
        f=f_g, grad_f=grad_g, hessian_f=hess_g,
        dim_x=problem.dim_x, dim_y=problem.dim_y,
        D_x=problem.D_x, D_y=problem.D_y,
        #  ρ_g = ρ + 2γ (Hessian Lipschitz of cubic is 2γ)
        rho=(problem.rho or 0.0) + 2 * gamma,
        #  ℓ_g = ℓ + 2γ·D_x (gradient Lipschitz of cubic is 2γD)
        ell=(problem.ell or 0.0) + 2 * gamma * max(problem.D_x, 1.0),
        ell_x=(problem.ell_x or 0.0) + 2 * gamma * max(problem.D_x, 1.0),
        ell_y=problem.ell_y,
        L=problem.L,
        project_x=problem.project_x, project_y=problem.project_y,
    )


def _make_h_problem(
    problem: MinimaxProblem,
    x_bar: Array,
    y_bar: Array,
    gamma: float,
) -> MinimaxProblem:
    """Construct ``h = f + (γ/3)·‖x−x̄‖³ − (γ/3)·‖y−ȳ‖³``."""
    x_bar = jnp.asarray(x_bar)
    y_bar = jnp.asarray(y_bar)

    def f_h(x: Array, y: Array):
        dx = x - x_bar
        dy = y - y_bar
        return (
            problem.f(x, y)
            + (gamma / 3.0) * jnp.linalg.norm(dx) ** 3
            - (gamma / 3.0) * jnp.linalg.norm(dy) ** 3
        )

    def grad_h(x: Array, y: Array) -> tuple[Array, Array]:
        gx, gy_neg = problem.grad_f(x, y)
        return (
            gx + _cubic_grad(x - x_bar, gamma),
            gy_neg + _cubic_grad(y - y_bar, gamma),
        )

    def hess_h(x: Array, y: Array):
        (H_xx, H_xy), (H_yx, H_yy) = problem.hessian_f(x, y)
        H_xx_h = H_xx + _cubic_hess(x - x_bar, gamma)
        H_yy_h = H_yy - _cubic_hess(y - y_bar, gamma)
        return (H_xx_h, H_xy), (H_yx, H_yy_h)

    diameter = max(problem.D_x, problem.D_y, 1.0)
    return MinimaxProblem(
        f=f_h, grad_f=grad_h, hessian_f=hess_h,
        dim_x=problem.dim_x, dim_y=problem.dim_y,
        D_x=problem.D_x, D_y=problem.D_y,
        #  ρ_h = ρ + 2γ
        rho=(problem.rho or 0.0) + 2 * gamma,
        #  ℓ_h = ℓ + 2γ·D
        ell=(problem.ell or 0.0) + 2 * gamma * diameter,
        ell_x=(problem.ell_x or 0.0) + 2 * gamma * max(problem.D_x, 1.0),
        ell_y=(problem.ell_y or 0.0) + 2 * gamma * max(problem.D_y, 1.0),
        L=problem.L,
        project_x=problem.project_x, project_y=problem.project_y,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Early-stopping restart helper
# ═════════════════════════════════════════════════════════════════════════════

def _restart_with_early_stop(
    run_epoch: Callable[[Array], tuple[Array, int]],
    z0: Array,
    S: int,
    *,
    residual_fn: Optional[Callable[[Array], float]] = None,
    residual_tol: float = 0.0,
    step_tol: float = 0.0,
) -> tuple[Array, int, int]:
    """Generic restart loop with between-epoch convergence checks."""
    z = z0
    total_calls = 0
    epochs_used = 0

    for s in range(S):
        z_new, calls = run_epoch(z)
        total_calls += calls
        epochs_used = s + 1

        if step_tol > 0:
            step = float(jnp.linalg.norm(z_new - z))
            if step < step_tol:
                return z_new, total_calls, epochs_used

        if residual_fn is not None and residual_tol > 0:
            res = float(residual_fn(z_new))
            if res < residual_tol:
                return z_new, total_calls, epochs_used

        z = z_new

    return z, total_calls, epochs_used


def _restart_jax(
    epoch_fn: Callable,
    z0: Array,
    S: int,
    *,
    step_tol: float = 0.0,
    warm: Optional[Array] = None,
    stats_init: Optional[Array] = None,
) -> tuple[Array, int, Optional[Array], Array]:
    """JAX-compatible restart with early stopping via ``jax.lax.while_loop``.

    Returns ``(z_final, epochs, warm_out, total_inner_calls)`` where
    *total_inner_calls* is the sum of the 4th element from each
    ``epoch_fn`` invocation (defaults to 0 when ``epoch_fn`` returns
    only 3 elements).

    Parameters
    ----------
    stats_init : Array or None
        Initial value for the inner-calls accumulator.  When ``None``,
        defaults to ``jnp.int32(0)`` (scalar).  Pass
        ``jnp.zeros(2, jnp.int32)`` to accumulate ``[crn_calls,
        linear_solves]`` as a 2-element vector.
    """
    dtype = z0.dtype
    tol_sq = jnp.asarray(
        step_tol ** 2 if step_tol > 0 else -1.0, dtype=dtype
    )
    S_jax = jnp.int32(S)
    tol_sq_cast = tol_sq.astype(dtype)
    int_zero = stats_init if stats_init is not None else jnp.int32(0)

    if warm is not None:
        def cond(carry):
            _z, prev_z, _w, epoch, _tic = carry
            not_done = epoch < S_jax
            diff = (_z - prev_z).astype(dtype)
            step_sq = jnp.dot(diff, diff)
            step_big = step_sq > tol_sq_cast
            return not_done & jnp.where(epoch > 0, step_big, jnp.bool_(True))

        def body(carry):
            z, _prev_z, w, epoch, total_inner_calls = carry
            result = epoch_fn(z, w)
            z_new, _calls, w_new = result[0], result[1], result[2]
            epoch_inner = _stats_array(result[3] if len(result) > 3 else int_zero)
            return (z_new, z, w_new, epoch + 1,
                    total_inner_calls + epoch_inner)

        z_final, _, warm_out, epochs, total_inner_calls = jax.lax.while_loop(
            cond, body, (z0, z0, warm, jnp.int32(0), int_zero),
        )
        return z_final, epochs, warm_out, total_inner_calls
    else:
        def cond(carry):
            _z, prev_z, epoch, _tic = carry
            not_done = epoch < S_jax
            diff = (_z - prev_z).astype(dtype)
            step_sq = jnp.dot(diff, diff)
            step_big = step_sq > tol_sq_cast
            return not_done & jnp.where(epoch > 0, step_big, jnp.bool_(True))

        def body(carry):
            z, _prev_z, epoch, total_inner_calls = carry
            result = epoch_fn(z, None)
            z_new = result[0]
            epoch_inner = _stats_array(result[3] if len(result) > 3 else int_zero)
            return (z_new, z, epoch + 1,
                    total_inner_calls + epoch_inner)

        z_final, _, epochs, total_inner_calls = jax.lax.while_loop(
            cond, body, (z0, z0, jnp.int32(0), int_zero),
        )
        return z_final, epochs, None, total_inner_calls


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 4 — Inexact proximal oracle for Φ  (middle loop)
# ═════════════════════════════════════════════════════════════════════════════

def _iProx_Phi(
    problem: MinimaxProblem,
    x_bar: Array,
    gamma: float,
    zeta_2: float = 1e-4,
    *,
    params: Optional[_LoopParams] = None,
    M_saddle: str = "npe",
    y_init: Optional[Array] = None,
    kernel: Optional[RegularizedSubproblem] = None,
) -> tuple[Array, Array, Array, Array]:
    """Algorithm 4: Inexact proximal oracle for ``Φ(x) = max_y f(x, y)``.

    Returns ``(x_out, u_out, y_hat, total_inner_calls)`` where
    *total_inner_calls* is the accumulated inner oracle count.
    """
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)

    g_problem = _make_g_problem(problem, x_bar, gamma)

    if kernel is None:
        kernel = RegularizedSubproblem(problem, gamma)

    inner_zeta_3 = min(params.zeta_3, zeta_2 * 0.1)

    neg_psi_fn, grad_neg_psi_fn = _make_psi_oracle(
        problem, x_bar, gamma, params,
        M_saddle=M_saddle, m_lazy=params.m_lazy,
    )

    def _prox_psi(y_bar: Array, warm_z: Optional[Array] = None
                  ) -> tuple[Array, Array, Array, Array]:
        y_out, v_out, z_hat, inner_calls = _iProx_Psi(
            problem, x_bar, y_bar, gamma,
            zeta_3=inner_zeta_3,
            params=params,
            M_saddle=M_saddle,
            kernel=kernel,
            z_init=warm_z,
        )
        return y_out, v_out, z_hat, inner_calls

    if y_init is not None:
        y0 = problem.project_y(y_init)
    else:
        y0 = problem.project_y(jnp.zeros(problem.dim_y))

    def _run_middle_epoch(y_cur: Array, warm_z: Optional[Array] = None
                          ) -> tuple[Array, int, Array]:
        return aipe(
            _prox_psi, grad_neg_psi_fn, y_cur,
            params.T_middle, gamma,
            project=problem.project_y,
            warm_init=warm_z,
        )

    z0_init = jnp.concatenate([problem.project_x(x_bar), y0])
    y_hat, _, _z_hat_out, total_inner_calls = _restart_jax(
        _run_middle_epoch, y0, params.S_middle,
        step_tol=params.zeta_2,
        warm=z0_init,
        stats_init=jnp.zeros(3, dtype=jnp.int32),
    )

    x_hat, min_x_crn = _minimize_x_auto(
        g_problem, y_hat,
        steps=max(20, params.T_inner * params.S_inner),
        M_saddle=M_saddle,
        gamma=gamma,
        m_lazy=params.m_lazy,
    )
    total_inner_calls = total_inner_calls + min_x_crn

    ell_g = max(_ell(g_problem), _ABS_TOL)
    eta_g = 1.0 / (2.0 * max(ell_g, _ABS_TOL))
    
    gx_bar, _ = g_problem.grad_f(x_bar, y_hat)
    x_half = g_problem.project_x(x_bar - eta_g * gx_bar)
    gx_half, _ = g_problem.grad_f(x_half, y_hat)
    x_out = g_problem.project_x(x_bar - eta_g * gx_half)
    u_out = (x_bar - x_out) / eta_g - gx_half
        
    return x_out, u_out, y_hat, total_inner_calls


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 5 — Inexact proximal oracle for -Ψ  (inner loop)
# ═════════════════════════════════════════════════════════════════════════════

def _iProx_Psi(
    problem: MinimaxProblem,
    x_bar: Array,
    y_bar: Array,
    gamma: float,
    zeta_3: float = 1e-4,
    *,
    params: Optional[_LoopParams] = None,
    M_saddle: str = "npe",
    kernel: Optional[RegularizedSubproblem] = None,
    z_init: Optional[Array] = None,
) -> tuple[Array, Array, Array, Array]:
    """Algorithm 5: Inexact proximal oracle for ``-Ψ(y; x̄)``.

    Returns ``(y_out, v_out, z_hat, inner_calls)`` where *inner_calls*
    is a JAX scalar counting the total inner oracle invocations.
    """
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)

    # Build the kernel if not supplied (backward-compatible path).
    if kernel is None:
        kernel = RegularizedSubproblem(problem, gamma)

    sub_rho = max(kernel.rho_h, _REG_MIN)
    npe_gamma = 2.0 * sub_rho
    D = max(_diam(kernel.D_x), _diam(kernel.D_y), _ABS_TOL)

    inner_T = params.T_inner

    if z_init is not None:
        z0 = jnp.concatenate([
            kernel.project_x(z_init[: kernel.dim_x]),
            kernel.project_y(z_init[kernel.dim_x :]),
        ])
    else:
        # Cleaned up redundant double-projection
        z0 = jnp.concatenate([
            kernel.project_x(x_bar), kernel.project_y(y_bar),
        ])

    def _F_h(z: Array) -> Array:
        return kernel.operator_F_h(z, x_bar, y_bar)

    proj = lambda z: jnp.concatenate([
        kernel.project_x(z[: kernel.dim_x]),
        kernel.project_y(z[kernel.dim_x :]),
    ])
    merit = lambda z: jnp.dot(_F_h(z), _F_h(z))

    if M_saddle == "npe":
        crn_oracle_fn = kernel.make_crn_oracle(
            x_bar, y_bar, npe_gamma, tol=zeta_3
        )

        def _run_inner(z: Array) -> tuple[Array, int]:
            return npe(
                crn_oracle_fn, _F_h, z,
                inner_T, npe_gamma, project=proj, fn=merit,
            )
    elif M_saddle == "len":
        def _run_inner(z: Array) -> tuple[Array, int]:
            def _crn_with_cached_hessian(
                z_bar: Array, H_snapshot: Array,
            ) -> tuple[Array, Array]:
                g = _F_h(z_bar)
                dtype_local = z_bar.dtype
                tiny_local = jnp.asarray(_TINY, dtype=dtype_local)
                tol_jax = jnp.asarray(zeta_3, dtype=dtype_local)
                lam0 = jnp.asarray(npe_gamma / 2.0, dtype=dtype_local)
                dim_x_local = kernel.dim_x

                J_xx = H_snapshot[:dim_x_local, :dim_x_local]
                J_xy = H_snapshot[:dim_x_local, dim_x_local:]
                # H_snapshot uses Jacobian sign convention:
                #   bottom row = [-H_yx, -H_yy]
                # _block_chol_solve expects raw Hessian blocks, so negate.
                H_yx = -H_snapshot[dim_x_local:, :dim_x_local]
                H_yy = -H_snapshot[dim_x_local:, dim_x_local:]
                eye_x = jnp.eye(dim_x_local, dtype=dtype_local)
                eye_y = jnp.eye(kernel.dim_y, dtype=dtype_local)

                def cond(state):
                    lam, _z, i, prev_lam = state
                    change = jnp.abs(lam - prev_lam)
                    return (i < 50) & (change > jnp.maximum(tol_jax * lam, tiny_local))

                def body(state):
                    lam, _z, i, _prev = state
                    delta = _block_chol_solve(g, J_xx, J_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny_local)
                    z_new = proj(z_bar + delta)
                    d_eff = z_new - z_bar
                    lam_candidate = (npe_gamma / 2.0) * jnp.linalg.norm(d_eff)
                    return (
                        _stable_lam_update(lam, lam_candidate, i),
                        z_new,
                        i + 1,
                        lam,
                    )

                lam, z_half, n_secular, _prev = jax.lax.while_loop(
                    cond, body,
                    (lam0, z_bar, jnp.int32(0),
                     jnp.asarray(-1.0, dtype=dtype_local)),
                )
                d_eff = z_half - z_bar
                u = -(g + H_snapshot @ d_eff + lam * d_eff)
                stats = jnp.stack([jnp.int32(1), n_secular, jnp.int32(1)])
                return z_half, u, stats

            def _len_oracle(
                z_bar: Array, z_snapshot: Array,
            ) -> tuple[Array, Array, Array]:
                xs = z_snapshot[: kernel.dim_x]
                ys = z_snapshot[kernel.dim_x :]
                H = kernel.jacobian_F_h(xs, ys, x_bar, y_bar)
                return _crn_with_cached_hessian(z_bar, H)

            max_norm_val = 100.0 * max(D, 1.0)
            z_out, epoch_stats = len_loop(
                _len_oracle, _F_h, z, inner_T, npe_gamma,
                m=params.m_lazy, project=proj, fn=merit,
                eta_floor=float(_ABS_TOL),
                max_norm=float(max_norm_val),
            )
            return z_out, epoch_stats
    else:
        raise ValueError(f"Unknown M_saddle={M_saddle!r}; expected 'npe' or 'len'.")

    def _run_inner_warm(z: Array, _warm):
        z_new, inner_stats = _run_inner(z)
        return z_new, inner_stats, None, inner_stats

    stats_init = jnp.zeros(3, dtype=jnp.int32)
    z_hat, epochs, _, calls = _restart_jax(
        _run_inner_warm, z0, params.S_inner,
        step_tol=max(zeta_3 * 0.01, _ABS_TOL),
        stats_init=stats_init,
    )

    # ── EG refinement ────────────────────────────────────────────────
    ell_h = max(kernel.ell_h, _ABS_TOL)
    eta = 1.0 / (2.0 * max(ell_h, _ABS_TOL))

    z_bar_joint = jnp.concatenate([x_bar, y_bar])
    F_bar = _F_h(z_bar_joint)
    z_half = proj(z_bar_joint - eta * F_bar)
    F_half = _F_h(z_half)
    z_out = proj(z_bar_joint - eta * F_half)

    c_out = (z_bar_joint - z_out) / eta - F_half
    _x_out, y_out = z_out[: kernel.dim_x], z_out[kernel.dim_x :]
    v_out = c_out[kernel.dim_x :]

    inner_calls = calls

    return y_out, v_out, z_hat, inner_calls


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def solve(
    problem: MinimaxProblem,
    epsilon: float,
    *,
    gamma: float | None = None,
    M_saddle: str = "npe",
    m_lazy: int = -1,  #  -1 = auto-adapt (was 5, silently overriding users)
    npe_T_factor: float = 1.0,
    z0: Optional[Array] = None,
    verbose: bool = False,
    no_restart: bool = False,
    no_acceleration: bool = False,
    fixed_inner_iters: Optional[int] = None,
) -> SolverResult:
    """Solve ``min_x max_y f(x, y)`` to approximately ``epsilon`` gap."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if M_saddle not in ("npe", "len"):
        raise ValueError("M_saddle must be either 'npe' or 'len'")

    if gamma is not None:
        gamma = float(gamma)
    elif M_saddle == "len":
        rho = float(problem.rho or 1.0)
        gamma = rho / max(m_lazy ** 0.5, 1.0) if m_lazy > 0 else rho
    else:
        gamma = _default_gamma(problem, None)

    mu_x = epsilon / (2.0 * max(_diam(problem.D_x), _ABS_TOL) ** 3)
    mu_y = epsilon / (2.0 * max(_diam(problem.D_y), _ABS_TOL) ** 3)
    params = _compute_loop_params(
        problem, epsilon, gamma, npe_T_factor, m_lazy=m_lazy,
        no_restart=no_restart, fixed_inner_iters=fixed_inner_iters,
    )

    if verbose:
        logger.setLevel(logging.DEBUG)

    if z0 is None:
        z0_start = _initial_z(problem)
    else:
        z0_arr = jnp.asarray(z0)
        expected = problem.dim_x + problem.dim_y
        if z0_arr.shape != (expected,):
            raise ValueError(f"z0 must have shape ({expected},), got {z0_arr.shape}")
        x0, y0 = _split(problem, z0_arr)
        z0_start = jnp.concatenate([problem.project_x(x0), problem.project_y(y0)])

    z_hat, stats_array, outer_epochs, final_y_calls = _algorithm_3(
        problem, gamma, mu_x, mu_y, params.zeta_1,
        params=params, M_saddle=M_saddle, z0=z0_start, verbose=verbose,
        no_acceleration=no_acceleration,
    )

    eta = 1.0 / (2.0 * max(_ell(problem), _ABS_TOL))
    z_out, _cert = eg_step(problem, z_hat, eta)
    x_out, y_out = _split(problem, z_out)

    gap = _safe_gap(problem, x_out, y_out, epsilon)
    if hasattr(gap, "block_until_ready"):
        gap.block_until_ready()

    history = {
        "gamma": gamma,
        "mu_x": mu_x,
        "mu_y": mu_y,
        "zeta_1": params.zeta_1,
        "zeta_2": params.zeta_2,
        "zeta_3": params.zeta_3,
        "T_outer": params.T_outer,
        "S_outer": params.S_outer,
        "T_middle": params.T_middle,
        "S_middle": params.S_middle,
        "T_inner": params.T_inner,
        "S_inner": params.S_inner,
        "M_saddle": M_saddle,
    }

    inner_crn = int(stats_array[0])
    inner_linear = int(stats_array[1])
    inner_grad = int(stats_array[2])
    d = problem.dim_x + problem.dim_y

    if M_saddle == "npe":
        inner_hessians = inner_crn
    else:
        inner_hessians = inner_crn // max(params.m_lazy, 1)

    inner_proj = inner_linear + inner_crn + 2

    # Safe extraction to Python int to prevent JAX array leakage into OracleStats
    actual_outer = int(jnp.maximum(1, outer_epochs).item())
    
    outer_grad = actual_outer * params.T_outer
    middle_grad = actual_outer * params.T_outer * params.S_middle * params.T_middle

    # EG refinements still happen outside the JAX-traced stats pipeline:
    # 2 grad evaluations per _iProx_Phi EG + 2 per _iProx_Psi EG
    hidden_iprox_phi_eg_grad = actual_outer * params.T_outer * 2
    hidden_iprox_psi_eg_grad = middle_grad * 2

    # Final _maximize_y in _algorithm_3: gradient calls tracked via final_y_calls[2]
    final_maximize_y_grad = int(final_y_calls[2].item()) if final_y_calls.shape[0] > 2 else max(20, params.T_middle * params.S_middle)

    # Final EG step in solve()
    final_eg_grad = 2

    total_hidden_grad = hidden_iprox_phi_eg_grad + hidden_iprox_psi_eg_grad + final_maximize_y_grad
    
    total_hidden_proj = total_hidden_grad  # EG steps match proj 1:1 with grads

    final_eg_proj = 2

    total_oracle_calls = inner_crn + int(final_y_calls[0].item())

    oracle_stats = OracleStats(
        grad_calls=inner_grad + total_hidden_grad + final_eg_grad,
        hessian_calls=inner_hessians,
        hvp_calls=0,
        crn_calls=inner_crn + int(final_y_calls[0].item()),
        projection_calls=inner_proj + final_eg_proj + total_hidden_proj,
        linear_solves=inner_linear + int(final_y_calls[1].item()),
        oracle_calls=total_oracle_calls,
        call_type="crn",
        fn_evals=0,
    )

    return SolverResult(
        x=x_out,
        y=y_out,
        gap=gap,
        iterations=actual_outer,
        oracle_calls=total_oracle_calls,
        oracle_stats=oracle_stats,
        converged=gap <= epsilon,
        history=history,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 3 — Full triple-loop Minimax-AIPE reduction  (outer loop)
# ══════════════════════════════════════════════════════════════════════════════════

def _algorithm_3(
    problem: MinimaxProblem,
    gamma: float,
    mu_x: float,
    mu_y: float,
    zeta_1: float,
    *,
    params: Optional[_LoopParams] = None,
    M_saddle: str = "npe",
    z0: Optional[Array] = None,
    verbose: bool = False,
    no_acceleration: bool = False,
) -> tuple[Array, int, int]:
    """Algorithm 3: Full three-loop Minimax-AIPE reduction."""
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)
    if z0 is None:
        z0 = _initial_z(problem)

    x0, _y0 = _split(problem, z0)

    # ── Reuse (or build) a JIT-stable pipeline ──────────────────────
    pipeline = _get_pipeline(problem, gamma, params, M_saddle)

    logger.debug("Pipeline kernel: %r", pipeline.kernel)

    # ── Outer AIPE with restart + early stopping ─────────────────────
    if no_acceleration:
        def _non_accel_epoch(x: Array, w: Optional[Array] = None) -> tuple[Array, int, Optional[Array], Array]:
            """Non-accelerated proximal point epoch (ablation baseline)."""
            if w is not None:
                x_new, _u, y_new, inner_calls = pipeline.prox_phi(x, w)
            else:
                x_new, _u, y_new, inner_calls = pipeline.prox_phi(x)
            return x_new, 1, y_new, inner_calls

        epoch_fn = _non_accel_epoch
    else:
        epoch_fn = pipeline.run_outer_epoch

    x_hat, outer_epochs, _warm_y_out, total_inner_calls = _restart_jax(
        epoch_fn, x0, params.S_outer,
        step_tol=params.zeta_1,
        stats_init=jnp.zeros(3, dtype=jnp.int32),
    )

    # ── Recover y ≈ argmax_y f(x_hat, y) ────────────────────────────
    y_hat, final_y_calls = _maximize_y_auto(
        problem, x_hat,
        steps=max(20, params.T_middle * params.S_middle),
        M_saddle=M_saddle,
        gamma=gamma,
        m_lazy=params.m_lazy,
    )

    inner_crn = int(total_inner_calls[0])
    inner_linear = int(total_inner_calls[1])

    grad_norm = float(jnp.linalg.norm(pipeline.grad_phi_fn(x_hat)[0]))
    phi_val = float(pipeline.phi_fn(x_hat))
    logger.info(
        "Algorithm 3: φ=%.4e  |∇φ|=%.3e  inner_crn=%d  inner_linear=%d  "
        "outer_epochs=%d/%d",
        phi_val, grad_norm, inner_crn, inner_linear,
        int(outer_epochs), params.S_outer,
    )

    z_hat = jnp.concatenate([x_hat, y_hat])
    return z_hat, CallStats(total_inner_calls), outer_epochs, final_y_calls


# ═════════════════════════════════════════════════════════════════════════════
# Auxiliary oracles for Φ and Ψ
# ═════════════════════════════════════════════════════════════════════════════

def _make_phi_oracle(
    problem: MinimaxProblem,
    gamma: float,
    params: _LoopParams,
    M_saddle: str = "npe",
    m_lazy: int = 5,
) -> tuple[
    Callable[[Array], Array],
    Callable[[Array], Array],
]:
    """Build approximate value, gradient, and Hessian oracles for Φ."""
    def _solve_y(x: Array) -> tuple[Array, Array]:
        return _maximize_y_auto(
            problem, x,
            steps=max(20, params.T_middle * params.S_middle),
            M_saddle=M_saddle,
            gamma=gamma,
            m_lazy=m_lazy,
        )

    def phi(x: Array):
        y, _calls = _solve_y(x)
        return problem.f(x, y)

    def grad_phi(x: Array) -> tuple[Array, Array]:
        y, calls = _solve_y(x)
        gx, gy_neg = problem.grad_f(x, y)
        # Implicit gradient correction to remove first-order bias from inexact y.
        # ∇Φ(x) ≈ ∇_x f - ∇_{xy} f (∇_{yy} f)^{-1} ∇_y f
        (_, H_xy), (_, H_yy) = problem.hessian_f(x, y)
        damping = 1e-5 * jnp.eye(H_yy.shape[0], dtype=H_yy.dtype)
        # Solve (-H_yy + damping) v = -gy_neg  (which is mathematically H_yy v = gy_neg)
        correction = H_xy @ jsp_linalg.solve(-H_yy + damping, -gy_neg)
        grad_call = jnp.array([jnp.int32(0), jnp.int32(0), jnp.int32(1)], dtype=calls.dtype)
        return gx + correction, calls + grad_call

    return phi, grad_phi


def _make_psi_oracle(
    problem: MinimaxProblem,
    x_bar: Array,
    gamma: float,
    params: _LoopParams,
    M_saddle: str = "npe",
    m_lazy: int = 5,
) -> tuple[
    Callable[[Array], Array],
    Callable[[Array], Array],
]:
    """Build approximate oracles for the convex function ``-Ψ(y; x̄)``."""
    g_problem = _make_g_problem(problem, x_bar, gamma)

    def _solve_x(y: Array) -> tuple[Array, Array]:
        return _minimize_x_auto(
            g_problem, y,
            steps=max(20, params.T_inner * params.S_inner),
            M_saddle=M_saddle,
            gamma=gamma,
            m_lazy=m_lazy,
            x_init=x_bar,
        )

    def neg_psi(y: Array):
        x, _calls = _solve_x(y)
        return -g_problem.f(x, y)

    def grad_neg_psi(y: Array) -> tuple[Array, Array]:
        x, calls = _solve_x(y)
        gx, gy_neg = g_problem.grad_f(x, y)
        # Implicit gradient correction to remove first-order bias from inexact x.
        # ∇(-Ψ)(y) ≈ -∇_y g + ∇_{xy} g (∇_{xx} g)^{-1} ∇_x g
        (H_xx, _), (H_yx, _) = g_problem.hessian_f(x, y)
        damping = 1e-5 * jnp.eye(H_xx.shape[0], dtype=H_xx.dtype)
        correction = H_yx @ jsp_linalg.solve(H_xx + damping, gx)
        grad_call = jnp.array([jnp.int32(0), jnp.int32(0), jnp.int32(1)], dtype=calls.dtype)
        return gy_neg + correction, calls + grad_call

    return neg_psi, grad_neg_psi

# ═════════════════════════════════════════════════════════════════════════════
# Saddle subproblem solver (innermost layer)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_saddle_subproblem(
    problem: MinimaxProblem,
    z0: Array,
    gamma: float,
    params: _LoopParams,
    M_saddle: str,
    tolerance: float = 0.0,
    kernel: Optional[RegularizedSubproblem] = None,
) -> tuple[Array, int]:
    """Solve a saddle subproblem via NPE-restart or LEN-restart."""
    sub_rho = max(float(problem.rho or 0.0), _REG_MIN)
    npe_gamma = 2.0 * sub_rho

    mu_inner = gamma / 2.0
    inner_T = max(1, min(200, int(ceil(
        (npe_gamma / max(mu_inner, _ABS_TOL)) ** (2.0 / 3.0)
    ))))
    inner_T = min(inner_T, params.T_inner)

    proj = lambda z: project_z(problem, z)
    merit = lambda z: jnp.dot(problem.operator_F(z), problem.operator_F(z))

    if M_saddle == "npe":
        oracle = make_crn_npe_oracle(problem, npe_gamma, tol=tolerance)

        def _run_inner(z: Array) -> tuple[Array, int]:
            return npe(
                oracle, problem.operator_F, z,
                inner_T, npe_gamma, project=proj, fn=merit,
            )

    elif M_saddle == "len":
        oracle = make_lazy_crn_npe_oracle(problem, npe_gamma, tol=tolerance)

        def _run_inner(z: Array) -> tuple[Array, int]:
            return len_loop(
                oracle, problem.operator_F, z,
                inner_T, npe_gamma, m=params.m_lazy,
                project=proj, fn=merit,
            )
    else:
        raise ValueError(f"Unknown M_saddle={M_saddle!r}; expected 'npe' or 'len'.")

    def _run_inner_warm(z: Array, _warm):
        z_new, calls = _run_inner(z)
        return z_new, calls, None, calls

    z_hat, epochs, _, calls = _restart_jax(
        _run_inner_warm, z0, params.S_inner,
        step_tol=max(tolerance * 0.01, _ABS_TOL),
        stats_init=jnp.zeros(3, dtype=jnp.int32),
    )
    return z_hat, calls

# ═════════════════════════════════════════════════════════════════════════════
# ALEN-aware sub-solvers (dispatched by M_saddle)
# ═════════════════════════════════════════════════════════════════════════════

def _minimize_x_auto(
    problem: MinimaxProblem,
    y: Array,
    *,
    steps: int,
    x_init: Optional[Array] = None,
    M_saddle: str = "npe",
    gamma: float = 1.0,
    m_lazy: int = 5,
) -> tuple[Array, Array]:
    """Approximately minimise ``x ↦ f(x, y)``."""
    if M_saddle == "len":
        return minimize_x_alen(
            problem, y, steps=steps, gamma=gamma,
            m=m_lazy, x_init=x_init,
        )
    return _minimize_x(problem, y, steps=steps, x_init=x_init)


def _maximize_y_auto(
    problem: MinimaxProblem,
    x: Array,
    *,
    steps: int,
    y_init: Optional[Array] = None,
    M_saddle: str = "npe",
    gamma: float = 1.0,
    m_lazy: int = 5,
) -> tuple[Array, Array]:
    """Approximately maximise ``y ↦ f(x, y)``."""
    if M_saddle == "len":
        return maximize_y_alen(
            problem, x, steps=steps, gamma=gamma,
            m=m_lazy, y_init=y_init,
        )
    return _maximize_y(problem, x, steps=steps, y_init=y_init)


# ═════════════════════════════════════════════════════════════════════════════
# Simple gradient ascent / descent helpers
# ═════════════════════════════════════════════════════════════════════════════

def _maximize_y(
    problem: MinimaxProblem,
    x: Array,
    *,
    steps: int,
    y_init: Optional[Array] = None,
) -> tuple[Array, Array]:
    """Approximately maximise ``y ↦ f(x, y)`` by gradient ascent."""
    dtype = x.dtype
    if y_init is not None:
        y = problem.project_y(y_init)
    else:
        y = problem.project_y(jnp.zeros(problem.dim_y, dtype=dtype))
    lr = 1.0 / max(_ell(problem), _ABS_TOL)

    def body(_i, cur):
        _gx, gy_neg = problem.grad_f(x, cur)
        return problem.project_y(cur - lr * gy_neg)

    return jax.lax.fori_loop(0, int(steps), body, y), jnp.stack([jnp.int32(0), jnp.int32(0), jnp.int32(steps)])


def _minimize_x(
    problem: MinimaxProblem,
    y: Array,
    *,
    steps: int,
    x_init: Optional[Array] = None,
) -> tuple[Array, Array]:
    """Approximately minimise ``x ↦ f(x, y)`` by gradient descent."""
    dtype = y.dtype
    if x_init is not None:
        x = problem.project_x(x_init)
    else:
        x = problem.project_x(jnp.zeros(problem.dim_x, dtype=dtype))
    lr = 1.0 / max(_ell(problem), _ABS_TOL)

    def body(_i, cur):
        gx, _gy_neg = problem.grad_f(cur, y)
        return problem.project_x(cur - lr * gx)

    return jax.lax.fori_loop(0, int(steps), body, x), jnp.stack([jnp.int32(0), jnp.int32(0), jnp.int32(steps)])


# ═════════════════════════════════════════════════════════════════════════════
# Parameter computation
# ═════════════════════════════════════════════════════════════════════════════

def _compute_loop_params(
    problem: MinimaxProblem,
    epsilon: float,
    gamma: float,
    npe_T_factor: float = 0.5,
    m_lazy: int = -1,  # -1 = auto-adapt
    no_restart: bool = False,
    fixed_inner_iters: Optional[int] = None,
) -> _LoopParams:
    """Compute iteration counts and accuracy parameters for all three loops."""
    D = max(_diameter(problem), _ABS_TOL)
    ell = max(_ell(problem), _ABS_TOL)
    rho = float(problem.rho or 0.0)

    # ── Accuracy scheduling ──────────────────────────────────────────
    mu_y = epsilon / (2.0 * max(_diam(problem.D_y), _ABS_TOL) ** 3)
    mu_x = epsilon / (2.0 * max(_diam(problem.D_x), _ABS_TOL) ** 3)
    zeta_1_raw = mu_y * epsilon**2 / (147.0 * ell**3 * D**2 + _ABS_TOL)
    zeta_1 = min(epsilon, max(zeta_1_raw, 1e-30))
    zeta_2 = min(zeta_1 * 0.2, 1e-3)
    zeta_3 = min(zeta_2 * 0.2, 1e-4)

    # ── Restart counts: epsilon-based with practical cap ──────────────
    _S_CAP = 12
    S = max(1, min(
        int(ceil(log2(max(D / max(epsilon, _ABS_TOL), 2.0)))),
        _S_CAP,
    ))

    if no_restart:
        S = 1

    # ── Outer loop: AIPE on Φ (Theorem 4.1) ───────────────────────────
    T_outer = max(1, min(200, int(ceil(
        npe_T_factor * (gamma / max(mu_x, _ABS_TOL)) ** (2.0 / 7.0)
    ))))

    # ── Middle loop: AIPE on -Ψ (Theorem 4.1) ───────────────────────
    T_middle = max(1, min(200, int(ceil(
        npe_T_factor * (gamma / max(mu_y, _ABS_TOL)) ** (2.0 / 7.0)
    ))))

    # ── Inner loop: NPE/LEN on h-subproblem (Theorem E.1) ────────────
    rho_h = rho + 2 * gamma
    npe_gamma = 2.0 * rho_h
    mu_inner = gamma / 2.0
    T_inner = max(1, min(200, int(ceil(
        npe_T_factor * (npe_gamma / max(mu_inner, _ABS_TOL)) ** (2.0 / 3.0)
    ))))

    S_inner_default = max(1, min(S, _S_CAP))

    if fixed_inner_iters is not None:
        T_inner = max(1, fixed_inner_iters)
        S_inner_default = 1

    if no_restart:
        S_inner_default = 1

    # ── Adaptive m_lazy heuristic ─────────────────────────────────────
    if m_lazy <= 0:
        dim_total = problem.dim_x + problem.dim_y
        eff_cond = ell / max(gamma, _ABS_TOL)
        adaptive_m = int(max(3, (dim_total ** 0.5) * max(1.0, 0.5 * log2(eff_cond + 1))))
        m_lazy = max(1, min(adaptive_m, 50))
    else:
        m_lazy = max(1, m_lazy)

    return _LoopParams(
        T_outer=T_outer,
        S_outer=S,
        T_middle=T_middle,
        S_middle=1 if no_restart else max(1, min(S, _S_CAP)),
        T_inner=T_inner,
        S_inner=S_inner_default,
        zeta_1=zeta_1,
        zeta_2=zeta_2,
        zeta_3=zeta_3,
        m_lazy=m_lazy,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def _default_gamma(problem: MinimaxProblem, gamma: float | None) -> float:
    if gamma is not None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        return float(gamma)
    rho = float(problem.rho or 0.0)
    return rho if rho > 0 else 1.0


def _safe_gap(problem: MinimaxProblem, x: Array, y: Array, epsilon: float) -> float:
    try:
        gap = problem.duality_gap(x, y)
        if hasattr(gap, "block_until_ready"):
            gap.block_until_ready()
        return max(0.0, float(gap))
    except NotImplementedError:
        pass

    D = _diameter(problem)
    num_steps = max(5000, min(15000, int(200 * D / max(epsilon, _GAP_FLOOR))))
    gap = estimate_gap(
        problem, x, y,
        num_restarts=8, num_steps=num_steps, lr=None,
    )
    if hasattr(gap, "block_until_ready"):
        gap.block_until_ready()
    # Ensure a pure Python float is returned so that `converged` in SolverResult 
    # resolves to a Python bool rather than a JAX boolean array.
    return max(0.0, float(gap))


def _initial_z(problem: MinimaxProblem) -> Array:
    x0 = problem.project_x(jnp.zeros(problem.dim_x))
    y0 = problem.project_y(jnp.zeros(problem.dim_y))
    return jnp.concatenate([x0, y0])


def _split(problem: MinimaxProblem, z: Array) -> tuple[Array, Array]:
    return z[: problem.dim_x], z[problem.dim_x :]


def _diam(value: float | None) -> float:
    return float(value) if value is not None and value > 0 else 1.0


def _diameter(problem: MinimaxProblem) -> float:
    return max(_diam(problem.D_x), _diam(problem.D_y))


def _ell(problem: MinimaxProblem) -> float:
    return float(problem.ell) if problem.ell is not None and problem.ell > 0 else 1.0


__all__ = [
    "solve",
    "RegularizedSubproblem",
    # Internal building blocks — exported for advanced usage
    "_CallCounter",
    "_LoopParams",
    "_algorithm_3",
    "_compute_loop_params",
    "_cubic_grad",
    "_cubic_hess",
    "_iProx_Phi",
    "_iProx_Psi",
    "_make_g_problem",
    "_make_h_problem",
    "_make_phi_oracle",
    "_make_psi_oracle",
    "_restart_with_early_stop",
    "_solve_saddle_subproblem",
    # Backward compat
    "_HKernel",
]
