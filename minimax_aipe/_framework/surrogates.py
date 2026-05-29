"""Surrogate problems and reusable regularized saddle kernels."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe._compat import CRNResult
from minimax_aipe._precision import CUBIC_ZERO as _CUBIC_ZERO, TINY as _TINY
from minimax_aipe.oracles import _block_chol_solve, _stable_lam_update
from minimax_aipe.problem import MinimaxProblem


def _cubic_grad(delta: Array, gamma: float) -> Array:
    delta = jnp.asarray(delta)
    return gamma * jnp.linalg.norm(delta) * delta


def _cubic_hess(delta: Array, gamma: float) -> Array:
    delta = jnp.asarray(delta)
    norm = jnp.linalg.norm(delta)
    eye = jnp.eye(delta.shape[0], dtype=delta.dtype)
    safe_norm = jnp.maximum(norm, jnp.asarray(_CUBIC_ZERO, dtype=delta.dtype))
    hess = gamma * (norm * eye + jnp.outer(delta, delta) / safe_norm)
    return jnp.where(norm > _CUBIC_ZERO, hess, jnp.zeros_like(hess))


class RegularizedSubproblem:
    """Reusable kernel for the regularised h-subproblem."""

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
        self.rho_h = (problem.rho or 0.0) + 2 * gamma
        self.ell_h = (problem.ell or 0.0) + 2 * gamma * diameter
        self.project_x = problem.project_x
        self.project_y = problem.project_y

    def __repr__(self) -> str:
        return (
            f"RegularizedSubproblem(dim=({self.dim_x},{self.dim_y}), "
            f"gamma={self._gamma:.4e}, rho_h={self.rho_h:.4e})"
        )

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def base_problem(self) -> MinimaxProblem:
        return self._problem

    def operator_F_h(self, z: Array, x_bar: Array, y_bar: Array) -> Array:
        x, y = z[: self.dim_x], z[self.dim_x :]
        gx, gy_neg = self._problem.grad_f(x, y)
        gx_h = gx + _cubic_grad(x - x_bar, self._gamma)
        gy_neg_h = gy_neg + _cubic_grad(y - y_bar, self._gamma)
        return jnp.concatenate([gx_h, gy_neg_h])

    def jacobian_F_h(
        self, x: Array, y: Array, x_bar: Array, y_bar: Array
    ) -> Array:
        (H_xx, H_xy), (H_yx, H_yy) = self._problem.hessian_f(x, y)
        H_xx_h = H_xx + _cubic_hess(x - x_bar, self._gamma)
        H_yy_h = H_yy - _cubic_hess(y - y_bar, self._gamma)
        top = jnp.concatenate([H_xx_h, H_xy], axis=1)
        bot = jnp.concatenate([-H_yx, -H_yy_h], axis=1)
        return jnp.concatenate([top, bot], axis=0)

    def hessian_blocks_h(
        self, x: Array, y: Array, x_bar: Array, y_bar: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        (H_xx, H_xy), (H_yx, H_yy) = self._problem.hessian_f(x, y)
        H_xx_h = H_xx + _cubic_hess(x - x_bar, self._gamma)
        H_yy_h = H_yy - _cubic_hess(y - y_bar, self._gamma)
        return (H_xx_h, H_xy), (H_yx, H_yy_h)

    def project(self, z: Array) -> Array:
        return jnp.concatenate([
            self.project_x(z[: self.dim_x]),
            self.project_y(z[self.dim_x :]),
        ])

    def make_h_problem(self, x_bar: Array, y_bar: Array) -> MinimaxProblem:
        return _make_h_problem(self._problem, x_bar, y_bar, self._gamma)

    def make_crn_oracle(
        self,
        x_bar: Array,
        y_bar: Array,
        npe_gamma: float,
        n_iters: int = 15,
        tol: float = 0.0,
    ) -> Callable[[Array], tuple[Array, Array]]:
        dim_x = self.dim_x
        _tiny = _TINY

        def _project_z_h(z: Array) -> Array:
            xz, yz = z[:dim_x], z[dim_x:]
            return jnp.concatenate([self.project_x(xz), self.project_y(yz)])

        if tol > 0:
            def oracle(z_bar: Array) -> tuple[Array, Array]:
                g = self.operator_F_h(z_bar, x_bar, y_bar)
                xb, yb = z_bar[:dim_x], z_bar[dim_x:]
                H = self.jacobian_F_h(xb, yb, x_bar, y_bar)
                (H_xx, H_xy), (H_yx, H_yy) = self.hessian_blocks_h(xb, yb, x_bar, y_bar)
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
                    return (_stable_lam_update(lam, lam_candidate, i), z_new, i + 1, lam)

                lam, z, n_secular, _p = jax.lax.while_loop(
                    cond,
                    body,
                    (lam0, z_bar, jnp.int32(0), jnp.asarray(-1.0, dtype=dtype)),
                )
                d_eff = z - z_bar
                u = -(g + H @ d_eff + lam * d_eff)
                stats = jnp.stack([jnp.int32(1), n_secular, jnp.int32(1)])
                return CRNResult(z, u, stats)
        else:
            def oracle(z_bar: Array) -> tuple[Array, Array]:
                g = self.operator_F_h(z_bar, x_bar, y_bar)
                xb, yb = z_bar[:dim_x], z_bar[dim_x:]
                H = self.jacobian_F_h(xb, yb, x_bar, y_bar)
                (H_xx, H_xy), (H_yx, H_yy) = self.hessian_blocks_h(xb, yb, x_bar, y_bar)
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

                lam, z = jax.lax.fori_loop(0, n_iters, body, (lam0, z_bar))
                d_eff = z - z_bar
                u = -(g + H @ d_eff + lam * d_eff)
                stats = jnp.stack([jnp.int32(1), jnp.int32(n_iters), jnp.int32(1)])
                return CRNResult(z, u, stats)

        return oracle


_HKernel = RegularizedSubproblem


def _make_g_problem(problem: MinimaxProblem, x_bar: Array, gamma: float) -> MinimaxProblem:
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
        rho=(problem.rho or 0.0) + 2 * gamma,
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
        rho=(problem.rho or 0.0) + 2 * gamma,
        ell=(problem.ell or 0.0) + 2 * gamma * diameter,
        ell_x=(problem.ell_x or 0.0) + 2 * gamma * max(problem.D_x, 1.0),
        ell_y=(problem.ell_y or 0.0) + 2 * gamma * max(problem.D_y, 1.0),
        L=problem.L,
        project_x=problem.project_x, project_y=problem.project_y,
    )


def make_epsilon_regularized_problem(
    problem: MinimaxProblem,
    mu_x: float,
    mu_y: float,
) -> MinimaxProblem:
    """Return the epsilon-regularized minimax problem used by the reduction."""

    mu_x = float(mu_x)
    mu_y = float(mu_y)

    def f_reg(x: Array, y: Array):
        return (
            problem.f(x, y)
            + (mu_x / 3.0) * jnp.linalg.norm(x) ** 3
            - (mu_y / 3.0) * jnp.linalg.norm(y) ** 3
        )

    def grad_reg(x: Array, y: Array) -> tuple[Array, Array]:
        gx, gy_neg = problem.grad_f(x, y)
        return gx + _cubic_grad(x, mu_x), gy_neg + _cubic_grad(y, mu_y)

    def hess_reg(x: Array, y: Array):
        (H_xx, H_xy), (H_yx, H_yy) = problem.hessian_f(x, y)
        return (
            (H_xx + _cubic_hess(x, mu_x), H_xy),
            (H_yx, H_yy - _cubic_hess(y, mu_y)),
        )

    ell_x = (problem.ell_x or problem.ell or 0.0) + 2.0 * mu_x * max(problem.D_x, 1.0)
    ell_y = (problem.ell_y or problem.ell or 0.0) + 2.0 * mu_y * max(problem.D_y, 1.0)
    ell = max(problem.ell or 0.0, ell_x, ell_y)
    rho = (problem.rho or 0.0) + 2.0 * max(mu_x, mu_y)

    return MinimaxProblem(
        f=f_reg,
        grad_f=grad_reg,
        hessian_f=hess_reg,
        dim_x=problem.dim_x,
        dim_y=problem.dim_y,
        D_x=problem.D_x,
        D_y=problem.D_y,
        rho=rho,
        ell=ell,
        ell_x=ell_x,
        ell_y=ell_y,
        L=problem.L,
        project_x=problem.project_x,
        project_y=problem.project_y,
    )
