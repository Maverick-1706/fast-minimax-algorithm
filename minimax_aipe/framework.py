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
from minimax_aipe.oracles import eg_step
from minimax_aipe.problem import MinimaxProblem, SolverResult

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Numerical guard constants
# ═════════════════════════════════════════════════════════════════════════════

_ABS_TOL = 1e-12
_REG_MIN = 1e-6
_CUBIC_ZERO = 1e-15
_GAP_FLOOR = 1e-6


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


@dataclass
class _WarmStart:
    """Mutable warm-start state updated via ``jax.debug.callback``.

    Holds the last concrete output array from an inner solver so the
    *next* call can use it as an initial point rather than starting from
    zeros.  Safe to read/write from Python; the actual update is always
    issued through ``jax.debug.callback`` so it fires with concrete
    values even when the surrounding code is being traced.
    """
    value: Optional[Array] = None


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
    """Generic restart loop with between-epoch convergence checks.

    Parameters
    ----------
    run_epoch : callable
        ``z → (z_new, oracle_calls)``.  One epoch of the inner algorithm.
    z0 : Array
        Initial iterate.
    S : int
        Maximum number of epochs.
    residual_fn : callable or None
        ``z → float``.  Cheap residual (e.g. ‖F(z)‖ or ‖∇Φ(x)‖).
    residual_tol : float
        Stop when ``residual_fn(z) < residual_tol``.  Ignored when ≤ 0.
    step_tol : float
        Stop when ``‖z_new − z_old‖ < step_tol``.  Ignored when ≤ 0.

    Returns
    -------
    z_out : Array
    oracle_calls : int
    epochs_used : int
        Number of epochs actually executed (≤ S).
    """
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
) -> tuple[Array, int]:
    """JAX-compatible restart with early stopping via ``jax.lax.while_loop``.

    Unlike :func:`_restart_with_early_stop` (which uses Python-level
    ``float()`` comparisons), this function uses a traced while-loop so
    it can be called from inside JAX traces — e.g., from inside
    ``aipe``'s ``scan`` body.

    Parameters
    ----------
    epoch_fn : callable
        ``z → (z_new, calls)`` where *calls* is a Python int (discarded
        internally; only the iterate is used for convergence checking).
    z0 : Array
    S : int
        Maximum number of epochs.
    step_tol : float
        Stop when ``‖z_new − z_old‖² < step_tol²``.  0 disables early
        stopping (the loop runs all *S* epochs).

    Returns
    -------
    z_out : Array
    total_calls : int or jnp.int32
        ``actual_epochs * T_per_epoch``.  A Python int when called
        outside a trace; a JAX int32 scalar when called inside one.
    """
    dtype = z0.dtype
    # Negative threshold when disabled → condition always True
    tol_sq = jnp.asarray(
        step_tol ** 2 if step_tol > 0 else -1.0, dtype=dtype
    )
    S_jax = jnp.int32(S)

    def cond(carry):
        _z, prev_z, epoch = carry
        not_done = epoch < S_jax
        # Always run epoch 0; after that check convergence
        step_sq = jnp.sum((_z - prev_z) ** 2)
        step_big = step_sq > tol_sq
        return not_done & jnp.where(epoch > 0, step_big, jnp.bool_(True))

    def body(carry):
        z, _prev_z, epoch = carry
        z_new, _calls = epoch_fn(z)
        return (z_new, z, epoch + 1)

    z_final, _, epochs = jax.lax.while_loop(
        cond, body, (z0, z0, jnp.int32(0)),
    )
    return z_final, epochs

# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def solve(
    problem: MinimaxProblem,
    epsilon: float,
    *,
    gamma: float | None = None,
    M_saddle: str = "npe",
    m_lazy: int = 5,
    npe_T_factor: float = 1.0,
    verbose: bool = False,
) -> SolverResult:
    """Solve ``min_x max_y f(x, y)`` to approximately ``epsilon`` gap.

    Uses the full three-loop Minimax-AIPE reduction: an outer AIPE loop
    minimises the primal value function Φ whose proximal oracle (Alg 4)
    runs a middle AIPE loop on -Ψ whose proximal oracle (Alg 5) delegates
    to NPE/LEN-restart on the cubic-regularised saddle subproblem.  A final
    EG refinement step is applied for the gap certificate.

    When ``M_saddle="len"``, all sub-solvers use lazy Hessians (ALEN for
    M_min, LEN for M_saddle) and γ is set to ρ/√m per Theorem 5.6.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if M_saddle not in ("npe", "len"):
        raise ValueError("M_saddle must be either 'npe' or 'len'")

    if gamma is not None:
        gamma = float(gamma)
    elif M_saddle == "len":
        rho = float(problem.rho or 1.0)
        gamma = rho / max(m_lazy ** 0.5, 1.0)
    else:
        gamma = _default_gamma(problem, None)

    mu_x = epsilon / (2.0 * max(_diam(problem.D_x), _ABS_TOL) ** 3)
    mu_y = epsilon / (2.0 * max(_diam(problem.D_y), _ABS_TOL) ** 3)
    params = _compute_loop_params(
        problem, epsilon, gamma, npe_T_factor, m_lazy=m_lazy,
    )

    if verbose:
        logger.setLevel(logging.DEBUG)

    z0 = _initial_z(problem)
    z_hat, calls = _algorithm_3(
        problem, gamma, mu_x, mu_y, params.zeta_1,
        params=params, M_saddle=M_saddle, z0=z0, verbose=verbose,
    )

    eta = 1.0 / (2.0 * max(_ell(problem), _ABS_TOL))
    z_out, _cert = eg_step(problem, z_hat, eta)
    x_out, y_out = _split(problem, z_out)

    gap = _safe_gap(problem, x_out, y_out, epsilon)
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

    total = calls
    try:
        total = int(total)
    except (TypeError, Exception):
        pass

    return SolverResult(
        x=x_out,
        y=y_out,
        gap=gap,
        iterations=params.S_outer,
        oracle_calls=calls + 1,
        converged=gap <= epsilon,
        history=history,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 3 — Full triple-loop Minimax-AIPE reduction  (outer loop)
# ═════════════════════════════════════════════════════════════════════════════

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
) -> tuple[Array, int]:
    """Algorithm 3: Full three-loop Minimax-AIPE reduction.

    **Outer loop** — AIPE with restart and early stopping minimises
        Φ(x) = max_{y∈D_y} f(x, y)

    **Middle loop** — Algorithm 4: each proximal-oracle call on Φ runs
    AIPE on -Ψ(y; x̄) = -min_x g(x, y; x̄).

    **Inner loop** — Algorithm 5: each proximal-oracle call on -Ψ runs
    NPE/LEN-restart on the h-problem with CRN-based operator oracles.

    Parameters
    ----------
    mu_x, mu_y : float
        Strong-convexity regularisers from the theoretical reduction.
    zeta_1 : float
        Target accuracy for the outer AIPE loop.
    """
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)
    if z0 is None:
        z0 = _initial_z(problem)

    x0, _y0 = _split(problem, z0)
    counter = _CallCounter()

    # ── Build the Φ oracle ───────────────────────────────────────────
    phi_fn, grad_phi_fn, _hess_phi_fn = _make_phi_oracle(
        problem, gamma, params,
        M_saddle=M_saddle, m_lazy=params.m_lazy,
    )

    # ── Proximal oracle for Φ (delegates to Algorithm 4) ─────────────
    # warm_y holds the last y_hat from _iProx_Phi across outer restarts.
    warm_y = _WarmStart()

    def _prox_phi(x_bar: Array) -> tuple[Array, Array]:
        x_out, u_out = _iProx_Phi(
            problem, x_bar, gamma,
            zeta_2=params.zeta_2,
            params=params,
            M_saddle=M_saddle,
            counter=counter,
            y_init=warm_y.value,
            outer_warm_y=warm_y,
        )
        return x_out, u_out

    # ── Outer AIPE with restart + early stopping ─────────────────────
    def _run_outer_epoch(x_cur: Array) -> tuple[Array, int]:
        return aipe(
            _prox_phi, grad_phi_fn, x_cur,
            params.T_outer, gamma,
            project=problem.project_x, fn=phi_fn,
        )

    x_hat, _outer_calls, outer_epochs = _restart_with_early_stop(
        _run_outer_epoch, x0, params.S_outer,
        step_tol=params.zeta_1,
    )

    # ── Recover y ≈ argmax_y f(x_hat, y) ────────────────────────────
    y_hat = _maximize_y_auto(
        problem, x_hat,
        steps=max(20, params.T_middle * params.S_middle),
        M_saddle=M_saddle, gamma=gamma, m_lazy=params.m_lazy,
    )

    total_calls = counter.total

    grad_norm = float(jnp.linalg.norm(grad_phi_fn(x_hat)))
    phi_val = float(phi_fn(x_hat))
    logger.info(
        "Algorithm 3: φ=%.4e  |∇φ|=%.3e  inner_calls=%d  "
        "outer_epochs=%d/%d",
        phi_val, grad_norm, total_calls,
        outer_epochs, params.S_outer,
    )

    z_hat = jnp.concatenate([x_hat, y_hat])
    return z_hat, total_calls


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
        rho=(problem.rho or 0.0) + gamma,
        ell=(problem.ell or 0.0) + gamma * max(problem.D_x, 1.0),
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
        rho=(problem.rho or 0.0) + gamma,
        ell=(problem.ell or 0.0) + gamma * diameter,
        L=problem.L,
        project_x=problem.project_x, project_y=problem.project_y,
    )


# ═════════════════════════════════════════════════════════════════════════════
# JIT-stable h-subproblem kernel
# ═════════════════════════════════════════════════════════════════════════════

class _HKernel:
    """Parametric kernel for the h-subproblem that avoids JIT recompilation.

    **Problem:** Every call to ``_iProx_Psi`` previously constructed a fresh
    ``MinimaxProblem`` via ``_make_h_problem(problem, x_bar, y_bar, gamma)``,
    creating new Python closures that *captured* the current ``(x_bar, y_bar)``
    values.  JAX treats each new closure as a new callable and recompiles the
    entire inner ``npe()``/``len()`` ``jax.lax.scan`` + ``fori_loop`` graph.
    With hundreds of ``_iProx_Psi`` calls per solve, this causes hundreds of
    expensive JIT compilations.

    **Solution:** Build the kernel *once* per ``(problem, gamma)`` pair.  The
    kernel exposes parametric functions whose *Python identity is fixed*:
    ``operator_F_h(z, x_bar, y_bar)`` and ``hessian_h(x, y, x_bar, y_bar)``.
    JAX compiles these once and reuses the compiled binary for every subsequent
    call — only the array data (``x_bar``, ``y_bar``, ``z``) changes.
    """

    def __init__(self, problem: MinimaxProblem, gamma: float) -> None:
        self._problem = problem
        self._gamma = gamma
        self.dim_x = problem.dim_x
        self.dim_y = problem.dim_y
        self.D_x = problem.D_x
        self.D_y = problem.D_y
        diameter = max(problem.D_x, problem.D_y, 1.0)
        self.rho_h = (problem.rho or 0.0) + gamma
        self.ell_h = (problem.ell or 0.0) + gamma * diameter
        self.project_x = problem.project_x
        self.project_y = problem.project_y

    # ── parametric operator F_h(z; x_bar, y_bar) ────────────────────
    def operator_F_h(self, z: Array, x_bar: Array, y_bar: Array) -> Array:
        """F_h(z) = [∇_x h, -∇_y h] with x_bar, y_bar as explicit args."""
        x, y = z[: self.dim_x], z[self.dim_x :]
        gx, gy_neg = self._problem.grad_f(x, y)
        gx_h = gx + _cubic_grad(x - x_bar, self._gamma)
        gy_neg_h = gy_neg + _cubic_grad(y - y_bar, self._gamma)
        return jnp.concatenate([gx_h, gy_neg_h])

    # ── parametric Jacobian of F_h ───────────────────────────────────
    def jacobian_F_h(
        self, x: Array, y: Array, x_bar: Array, y_bar: Array
    ) -> Array:
        """∇F_h assembled from hessian_f plus cubic correction blocks."""
        (H_xx, H_xy), (H_yx, H_yy) = self._problem.hessian_f(x, y)
        H_xx_h = H_xx + _cubic_hess(x - x_bar, self._gamma)
        H_yy_h = H_yy - _cubic_hess(y - y_bar, self._gamma)
        top = jnp.concatenate([H_xx_h, H_xy], axis=1)
        bot = jnp.concatenate([-H_yx, -H_yy_h], axis=1)
        return jnp.concatenate([top, bot], axis=0)

    # ── CRN oracle for NPE (parametric) ─────────────────────────────
    def make_crn_oracle(
        self,
        x_bar: Array,
        y_bar: Array,
        npe_gamma: float,
        n_iters: int = 50,
        tol: float = 0.0,
    ) -> Callable[[Array], tuple[Array, Array]]:
        """Return a CRN NPE oracle bound to fixed ``(x_bar, y_bar)``.

        The returned callable's Python identity is a fresh closure, but the
        *underlying parametric methods* (``operator_F_h``, ``jacobian_F_h``)
        remain the same Python objects across all calls.  JAX reuses the same
        compiled kernel; only the captured ``x_bar``/``y_bar`` constants differ
        — those are folded in as constants at trace time (they are concrete
        arrays outside any JAX trace when ``_iProx_Psi`` is called from
        ``_restart_with_early_stop``, which is an eager Python loop).

        When ``tol > 0`` the secular-equation solver exits as soon as the
        regularisation parameter λ converges, saving unnecessary iterations on
        easy subproblems.  This mirrors the ``tol`` path in
        :func:`~minimax_aipe.oracles.lazy_crn_oracle`.
        """
        dim_x = self.dim_x
        _tiny = 1e-12

        def _project_z_h(z: Array) -> Array:
            xz, yz = z[:dim_x], z[dim_x:]
            return jnp.concatenate([
                self.project_x(xz), self.project_y(yz),
            ])

        if tol > 0:
            def oracle(z_bar: Array) -> tuple[Array, Array]:
                g = self.operator_F_h(z_bar, x_bar, y_bar)
                xb, yb = z_bar[:dim_x], z_bar[dim_x:]
                H = self.jacobian_F_h(xb, yb, x_bar, y_bar)
                d = z_bar.shape[0]
                dtype = z_bar.dtype
                eye = jnp.eye(d, dtype=dtype)
                tiny = jnp.asarray(_tiny, dtype=dtype)
                tol_jax = jnp.asarray(tol, dtype=dtype)

                def cond(state):
                    lam, _z, i, prev_lam = state
                    change = jnp.abs(lam - prev_lam)
                    return (i < n_iters) & (change > jnp.maximum(tol_jax * lam, tiny))

                def body(state):
                    lam, _z, i, _prev = state
                    delta = jnp.linalg.solve(H + (lam + tiny) * eye, -g)
                    z_new = _project_z_h(z_bar + delta)
                    d_eff = z_new - z_bar
                    return (
                        (npe_gamma / 2.0) * jnp.linalg.norm(d_eff),
                        z_new, i + 1, lam,
                    )

                lam, z, _i, _p = jax.lax.while_loop(
                    cond, body,
                    (jnp.zeros((), dtype=dtype), z_bar,
                     jnp.int32(0), jnp.asarray(-1.0, dtype=dtype)),
                )
                d_eff = z - z_bar
                u = -(g + H @ d_eff + lam * d_eff)
                return z, u
        else:
            def oracle(z_bar: Array) -> tuple[Array, Array]:
                g = self.operator_F_h(z_bar, x_bar, y_bar)
                xb, yb = z_bar[:dim_x], z_bar[dim_x:]
                H = self.jacobian_F_h(xb, yb, x_bar, y_bar)
                d = z_bar.shape[0]
                dtype = z_bar.dtype
                eye = jnp.eye(d, dtype=dtype)
                tiny = jnp.asarray(_tiny, dtype=dtype)

                def body(i, state):
                    lam, z = state
                    delta = jnp.linalg.solve(H + (lam + tiny) * eye, -g)
                    z_new = _project_z_h(z_bar + delta)
                    d_eff = z_new - z_bar
                    return (npe_gamma / 2.0) * jnp.linalg.norm(d_eff), z_new

                lam, z = jax.lax.fori_loop(
                    0, n_iters, body, (jnp.zeros((), dtype=dtype), z_bar)
                )
                d_eff = z - z_bar
                u = -(g + H @ d_eff + lam * d_eff)
                return z, u

        return oracle


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
    counter: Optional[_CallCounter] = None,
    y_init: Optional[Array] = None,
    outer_warm_y: Optional["_WarmStart"] = None,
) -> tuple[Array, Array]:
    """Algorithm 4: Inexact proximal oracle for ``Φ(x) = max_y f(x, y)``.

    Solves the equivalent regularised saddle subproblem
        min_x max_y  g(x, y; x̄)   where  g = f + (γ/3)·‖x−x̄‖³

    by running AIPE on -Ψ (the *middle loop*):
        -Ψ(y; x̄) = -min_x g(x, y; x̄)

    whose proximal oracle delegates to Algorithm 5 (the *inner loop*).

    The δ tolerance from Definition 4.1 is enforced by passing ``zeta_3``
    (derived from ``zeta_2``) to the innermost CRN solver.

    When *y_init* is provided the middle AIPE loop is seeded from that
    point instead of zeros, warm-starting from the previous outer iterate.
    When *outer_warm_y* is provided the final ``y_hat`` is written back
    to it so the caller can warm-start the next invocation.

    Returns ``(x_out, u_out)`` with ``u_out ∈ ∂Φ(x_out)``.
    """
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)

    g_problem = _make_g_problem(problem, x_bar, gamma)

    # Derive inner tolerance: one order tighter than zeta_2, but at
    # least as tight as the pre-computed zeta_3.
    inner_zeta_3 = min(params.zeta_3, zeta_2 * 0.1)

    # ── Build oracles for -Ψ(y) = -min_x g(x, y; x̄) ────────────────
    neg_psi_fn, grad_neg_psi_fn, _hess_neg_psi_fn = _make_psi_oracle(
        problem, x_bar, gamma, params,
        M_saddle=M_saddle, m_lazy=params.m_lazy,
    )

    # ── Build JIT-stable h-kernel once per middle-loop invocation ────
    h_kernel = _HKernel(problem, gamma)

    # ── Per-call warm-start cell for the inner NPE z0 ────────────────
    # Updated via jax.debug.callback after each _iProx_Psi so the next
    # _prox_psi call starts from the previous saddle solution.
    inner_warm = _WarmStart()
    # Caches the x_out from the last inner call — avoids re-running
    # _minimize_x_auto after the middle AIPE completes.
    x_hat_warm = _WarmStart()

    # ── Proximal oracle for -Ψ (delegates to Algorithm 5) ────────────
    def _prox_psi(y_bar: Array) -> tuple[Array, Array]:
        return _iProx_Psi(
            problem, x_bar, y_bar, gamma,
            zeta_3=inner_zeta_3,
            params=params,
            M_saddle=M_saddle,
            counter=counter,
            kernel=h_kernel,
            z_init=inner_warm.value,
            _z_hat_cell=inner_warm,
            _x_hat_cell=x_hat_warm,
        )

    # ── Middle AIPE with restart + early stopping ────────────────────
    if y_init is not None:
        y0 = problem.project_y(y_init)
    else:
        y0 = problem.project_y(jnp.zeros(problem.dim_y))

    def _run_middle_epoch(y_cur: Array) -> tuple[Array, int]:
        return aipe(
            _prox_psi, grad_neg_psi_fn, y_cur,
            params.T_middle, gamma,
            project=problem.project_y, fn=neg_psi_fn,
        )

    y_hat, _middle_calls = _restart_jax(
        _run_middle_epoch, y0, params.S_middle,
        step_tol=params.zeta_2,
    )

    x_hat: Array
    if x_hat_warm.value is not None:
        x_hat = problem.project_x(x_hat_warm.value)
    else:
        x_hat = _minimize_x_auto(
            g_problem, y_hat,
            steps=max(20, params.T_inner * params.S_inner),
            M_saddle=M_saddle, gamma=gamma, m_lazy=params.m_lazy,
        )

    gx, _ = g_problem.grad_f(x_hat, y_hat)
    u_out = -gx

    if outer_warm_y is not None:
        jax.debug.callback(
            lambda v: setattr(outer_warm_y, 'value', v), y_hat
        )

    return x_hat, u_out

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
    counter: Optional[_CallCounter] = None,
    kernel: Optional["_HKernel"] = None,
    z_init: Optional[Array] = None,
    _z_hat_cell: Optional["_WarmStart"] = None,
    _x_hat_cell: Optional["_WarmStart"] = None,
) -> tuple[Array, Array]:
    """Algorithm 5: Inexact proximal oracle for ``-Ψ(y; x̄)``.

    Solves the regularised saddle subproblem
        min_x max_y  h(x, y; x̄, ȳ)
    where  h = f + (γ/3)·‖x−x̄‖³ − (γ/3)·‖y−ȳ‖³

    via NPE/LEN-restart on the monotone operator F_h, followed by one EG
    refinement step.  The δ tolerance from Theorem 5.3 is enforced by
    passing ``zeta_3`` to the CRN secular-equation solver.

    When *kernel* is supplied (built once by ``_iProx_Phi``), the inner CRN
    oracle reuses the kernel's parametric methods instead of creating fresh
    closures, avoiding JIT recompilation on every call.

    When *z_init* is provided the NPE restart begins from that point
    (warm-starting from the previous call's solution).  When *_z_hat_cell*
    is provided the final ``z_hat`` is written back to it via
    ``jax.debug.callback`` so the caller can warm-start the next invocation.

    Returns ``(y_out, v_out)`` with ``v_out ∈ ∂(-Ψ)(y_out)``.
    """
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)

    if kernel is None:
        kernel = _HKernel(problem, gamma)

    sub_rho = max(kernel.rho_h, _REG_MIN)
    npe_gamma = 2.0 * sub_rho

    D = max(_diam(kernel.D_x), _diam(kernel.D_y), _ABS_TOL)
    complexity = (D ** (12.0 / 7.0)) * (
        (sub_rho / max(npe_gamma, _ABS_TOL)) ** (4.0 / 7.0)
    )
    inner_T = max(8, int(ceil(complexity)))
    inner_T = min(inner_T, params.T_inner)

    z0: Array
    if z_init is not None:
        z0 = jnp.concatenate([
            kernel.project_x(z_init[: kernel.dim_x]),
            kernel.project_y(z_init[kernel.dim_x :]),
        ])
    else:
        z0 = jnp.concatenate([
            problem.project_x(x_bar), problem.project_y(y_bar),
        ])
        z0 = jnp.concatenate([
            kernel.project_x(z0[: kernel.dim_x]),
            kernel.project_y(z0[kernel.dim_x :]),
        ])

    # ── Build a JIT-stable CRN oracle using the kernel ────────────────
    crn_oracle_fn = kernel.make_crn_oracle(x_bar, y_bar, npe_gamma, tol=zeta_3)

    def _F_h(z: Array) -> Array:
        return kernel.operator_F_h(z, x_bar, y_bar)

    proj = lambda z: jnp.concatenate([
        kernel.project_x(z[: kernel.dim_x]),
        kernel.project_y(z[kernel.dim_x :]),
    ])
    merit = lambda z: jnp.dot(_F_h(z), _F_h(z))

    def _run_inner(z: Array) -> tuple[Array, int]:
        return npe(
            crn_oracle_fn, _F_h, z,
            inner_T, npe_gamma, project=proj, fn=merit,
        )

    z_hat, epochs = _restart_jax(
        _run_inner, z0, params.S_inner,
        step_tol=max(zeta_3 * 0.01, 1e-14),
    )
    calls = epochs * inner_T

    # ── EG refinement ────────────────────────────────────────────────
    ell_h = max(kernel.ell_h, _ABS_TOL)
    diam_h = max(D, _ABS_TOL)
    rho_h = max(kernel.rho_h, _REG_MIN)
    eta = 1.0 / (2.0 * max(ell_h + 2.0 * rho_h * diam_h, _ABS_TOL))

    F_hat = _F_h(z_hat)
    z_half = proj(z_hat - eta * F_hat)
    F_half = _F_h(z_half)
    z_out = proj(z_hat - eta * F_half)

    _x_out, y_out = z_out[: kernel.dim_x], z_out[kernel.dim_x :]
    gy_neg_h = _F_h(z_out)[kernel.dim_x :]
    v_out = -gy_neg_h

    if _z_hat_cell is not None:
        jax.debug.callback(
            lambda v: setattr(_z_hat_cell, 'value', v), z_hat
        )

    if _x_hat_cell is not None:
        jax.debug.callback(
            lambda v: setattr(_x_hat_cell, 'value', v), z_out[: kernel.dim_x]
        )

    if counter is not None:
        jax.debug.callback(
            lambda c: setattr(counter, 'total', counter.total + int(c)),
            calls + 1
        )

    return y_out, v_out


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
    Callable[[Array], Array],
]:
    """Build approximate value, gradient, and Hessian oracles for Φ.

    ``Φ(x) = max_{y∈D_y} f(x, y)``.
    When ``M_saddle="len"``, uses ALEN (lazy Hessians) for the inner
    maximisation over y.
    """
    def _solve_y(x: Array) -> Array:
        return _maximize_y_auto(
            problem, x,
            steps=max(20, params.T_middle * params.S_middle),
            M_saddle=M_saddle, gamma=gamma, m_lazy=m_lazy,
        )

    def phi(x: Array):
        y = _solve_y(x)
        return problem.f(x, y)

    def grad_phi(x: Array) -> Array:
        y = _solve_y(x)
        gx, _gy_neg = problem.grad_f(x, y)
        return gx

    hess_phi = jax.jacfwd(grad_phi)
    return phi, grad_phi, hess_phi


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
    Callable[[Array], Array],
]:
    """Build approximate oracles for the convex function ``-Ψ(y; x̄)``.

    ``Ψ(y; x̄) = min_{x∈D_x} g(x, y; x̄)`` where ``g`` is the
    cubic-regularised surrogate.
    When ``M_saddle="len"``, uses ALEN (lazy Hessians) for the inner
    minimisation over x.
    """
    g_problem = _make_g_problem(problem, x_bar, gamma)

    def _solve_x(y: Array) -> Array:
        return _minimize_x_auto(
            g_problem, y,
            steps=max(20, params.T_inner * params.S_inner),
            M_saddle=M_saddle, gamma=gamma, m_lazy=m_lazy,
        )

    def neg_psi(y: Array):
        x = _solve_x(y)
        return -g_problem.f(x, y)

    def grad_neg_psi(y: Array) -> Array:
        x = _solve_x(y)
        _gx, gy_neg = g_problem.grad_f(x, y)
        return gy_neg

    hess_neg_psi = jax.jacfwd(grad_neg_psi)
    return neg_psi, grad_neg_psi, hess_neg_psi


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
) -> tuple[Array, int]:
    """Solve a saddle subproblem via NPE-restart or LEN-restart.

    Uses :func:`_restart_with_early_stop` so that the inner loop
    terminates as soon as ‖F(z)‖ drops below *tolerance* or the iterate
    stops moving, whichever comes first.

    Parameters
    ----------
    problem : MinimaxProblem
        The h-subproblem (ρ_h = ρ + γ).
    tolerance : float
        δ tolerance for the CRN secular-equation solver and residual-based
        early-stopping threshold.
    """
    sub_rho = max(float(problem.rho or 0.0), _REG_MIN)
    npe_gamma = 2.0 * sub_rho

    D = max(_diameter(problem), _ABS_TOL)
    complexity = (D ** (12.0 / 7.0)) * (
        (sub_rho / max(npe_gamma, _ABS_TOL)) ** (4.0 / 7.0)
    )
    inner_T = max(8, int(ceil(complexity)))
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

    z_hat, epochs = _restart_jax(
        _run_inner, z0, params.S_inner,
        step_tol=max(tolerance * 0.01, 1e-14),
    )
    # epochs is JAX int32 inside a trace, Python int outside.
    # Multiply by inner_T to get total oracle calls.
    return z_hat, epochs * inner_T

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
) -> Array:
    """Approximately minimise ``x ↦ f(x, y)``.

    Dispatches to plain gradient descent (NPE path) or ALEN-restart
    (LEN path) based on ``M_saddle``.
    """
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
) -> Array:
    """Approximately maximise ``y ↦ f(x, y)``.

    Dispatches to plain gradient ascent (NPE path) or ALEN-restart
    (LEN path) based on ``M_saddle``.
    """
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
) -> Array:
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

    return jax.lax.fori_loop(0, int(steps), body, y)


def _minimize_x(
    problem: MinimaxProblem,
    y: Array,
    *,
    steps: int,
    x_init: Optional[Array] = None,
) -> Array:
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

    return jax.lax.fori_loop(0, int(steps), body, x)


# ═════════════════════════════════════════════════════════════════════════════
# Parameter computation
# ═════════════════════════════════════════════════════════════════════════════

def _compute_loop_params(
    problem: MinimaxProblem,
    epsilon: float,
    gamma: float,
    npe_T_factor: float = 1.0,
    m_lazy: int = 5,
) -> _LoopParams:
    """Compute iteration counts and accuracy parameters for all three loops.

    * T values are derived from the paper's convergence theorems.
    * S values use the epsilon-based formula (not zeta-based) with a
      practical cap of 4 to limit triple-nested restart explosion.
    * Accuracy parameters zeta_{1,2,3} follow the hierarchy from
      Theorems 5.1–5.3 and are wired through to the CRN solvers at
      runtime.
    """
    D = max(_diameter(problem), _ABS_TOL)
    ell = max(_ell(problem), _ABS_TOL)
    rho = max(float(problem.rho or 0.0), gamma, _REG_MIN)

    # ── Accuracy scheduling ──────────────────────────────────────────
    mu_y = epsilon / (2.0 * max(_diam(problem.D_y), _ABS_TOL) ** 3)
    mu_x = epsilon / (2.0 * max(_diam(problem.D_x), _ABS_TOL) ** 3)
    zeta_1 = min(epsilon, mu_y * epsilon**2 / (147.0 * ell**3 * D**2 + _ABS_TOL))
    zeta_2 = min(1e-2, max(zeta_1 * 0.1, _ABS_TOL))
    zeta_3 = min(1e-2, max(zeta_2 * 0.1, _ABS_TOL))

    # ── Restart counts: epsilon-based with practical cap ──────────────
    _S_CAP = 4
    S = max(1, min(
        int(ceil(log2(max(D / max(epsilon, _ABS_TOL), 2.0)))),
        _S_CAP,
    ))

    # ── Outer loop: AIPE on Φ (Theorem 4.1) ───────────────────────────
    # Theorem 4.1: one AIPE epoch with an (0,γ)-proximal oracle halves
    # the distance with T = O((γ/μ)^{1/2}) iterations.
    # 2/7 is the overall triple-loop complexity, NOT the per-epoch count.
    T_outer = max(8, min(200, int(ceil(
        npe_T_factor * (gamma / max(mu_x, _ABS_TOL)) ** 0.5
    ))))

    # ── Middle loop: AIPE on -Ψ (Theorem 4.1) ───────────────────────
    T_middle = max(6, min(200, int(ceil(
        (gamma / max(mu_y, _ABS_TOL)) ** 0.5
    ))))

    # ── Inner loop: NPE/LEN on h-subproblem (Theorem E.1) ────────────
    mu_inner = gamma / 2.0
    rho_h = rho + gamma
    npe_gamma = 2.0 * rho_h
    T_inner = max(6, min(200, int(ceil(
        npe_T_factor * (npe_gamma / max(mu_inner, _ABS_TOL)) ** (2.0 / 3.0)
    ))))

    return _LoopParams(
        T_outer=T_outer,
        S_outer=S,
        T_middle=T_middle,
        S_middle=max(1, min(S, _S_CAP)),
        T_inner=T_inner,
        S_inner=max(1, min(S, _S_CAP)),
        zeta_1=zeta_1,
        zeta_2=zeta_2,
        zeta_3=zeta_3,
        m_lazy=max(1, m_lazy),
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
        return max(0.0, float(gap))
    except NotImplementedError:
        pass

    D = _diameter(problem)
    lr = 0.5 / max(_ell(problem), 1.0)
    num_steps = max(200, min(2000, int(200 * D / max(epsilon, _GAP_FLOOR))))
    gap = estimate_gap(
        problem, x, y,
        num_restarts=8, num_steps=num_steps, lr=lr,
    )
    return max(0.0, gap)


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
    "_CallCounter",
    "_HKernel",
    "_LoopParams",
    "_WarmStart",
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
]
