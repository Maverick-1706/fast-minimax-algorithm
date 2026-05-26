"""Approximate value/gradient oracles and simple x/y sub-solvers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import Array

from minimax_aipe.alen import maximize_y_alen, minimize_x_alen
from minimax_aipe.problem import MinimaxProblem
from minimax_aipe._framework.params import _LoopParams
from minimax_aipe._framework.surrogates import _make_g_problem
from minimax_aipe._precision import ABS_TOL as _ABS_TOL


def _minimize_x_auto(
    problem: MinimaxProblem,
    y: Array,
    *,
    steps: int,
    x_init: Array | None = None,
    M_saddle: str = "npe",
    gamma: float = 1.0,
    m_lazy: int = 5,
) -> tuple[Array, Array]:
    if M_saddle == "len":
        return minimize_x_alen(problem, y, steps=steps, gamma=gamma, m=m_lazy, x_init=x_init)
    return _minimize_x(problem, y, steps=steps, x_init=x_init)


def _maximize_y_auto(
    problem: MinimaxProblem,
    x: Array,
    *,
    steps: int,
    y_init: Array | None = None,
    M_saddle: str = "npe",
    gamma: float = 1.0,
    m_lazy: int = 5,
) -> tuple[Array, Array]:
    if M_saddle == "len":
        return maximize_y_alen(problem, x, steps=steps, gamma=gamma, m=m_lazy, y_init=y_init)
    return _maximize_y(problem, x, steps=steps, y_init=y_init)


def _make_phi_oracle(
    problem: MinimaxProblem,
    gamma: float,
    params: _LoopParams,
    M_saddle: str = "npe",
    m_lazy: int = 5,
):
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
        (_, H_xy), (_, H_yy) = problem.hessian_f(x, y)
        H_yy_pos = -(H_yy + H_yy.T) / 2.0
        min_curv = jnp.min(jnp.linalg.eigvalsh(H_yy_pos))
        damping = 1e-5 * jnp.eye(H_yy.shape[0], dtype=H_yy.dtype)

        def corrected(_):
            return H_xy @ jsp_linalg.solve(H_yy_pos + damping, -gy_neg)

        correction = jax.lax.cond(
            min_curv > 1e-8,
            corrected,
            lambda _: jnp.zeros_like(gx),
            operand=None,
        )
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
):
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
        (H_xx, _), (H_yx, _) = g_problem.hessian_f(x, y)
        H_xx_pos = (H_xx + H_xx.T) / 2.0
        min_curv = jnp.min(jnp.linalg.eigvalsh(H_xx_pos))
        damping = 1e-5 * jnp.eye(H_xx.shape[0], dtype=H_xx.dtype)

        def corrected(_):
            return H_yx @ jsp_linalg.solve(H_xx_pos + damping, gx)

        correction = jax.lax.cond(
            min_curv > 1e-8,
            corrected,
            lambda _: jnp.zeros_like(gy_neg),
            operand=None,
        )
        grad_call = jnp.array([jnp.int32(0), jnp.int32(0), jnp.int32(1)], dtype=calls.dtype)
        return gy_neg + correction, calls + grad_call

    return neg_psi, grad_neg_psi


def _maximize_y(
    problem: MinimaxProblem,
    x: Array,
    *,
    steps: int,
    y_init: Array | None = None,
) -> tuple[Array, Array]:
    dtype = x.dtype
    if y_init is not None:
        y = problem.project_y(y_init)
    else:
        y = problem.project_y(jnp.zeros(problem.dim_y, dtype=dtype))
    lr = 1.0 / max(float(problem.ell_y or problem.ell or 0.0), _ABS_TOL)
    beta = jnp.asarray(0.9, dtype=dtype)
    f_x = lambda yy: problem.f(x, yy)

    def body(_i, carry):
        y_cur, v_cur, best_y, best_val = carry
        y_ahead = y_cur + beta * v_cur
        _gx, gy_neg = problem.grad_f(x, y_ahead)
        y_new = problem.project_y(y_cur - lr * gy_neg + beta * v_cur)
        v_new = y_new - y_cur
        val = f_x(y_new)
        improve = val > best_val
        best_y = jnp.where(improve, y_new, best_y)
        best_val = jnp.where(improve, val, best_val)
        return y_new, v_new, best_y, best_val

    best_val0 = f_x(y)
    y_out, _v_out, best_y, _best_val = jax.lax.fori_loop(
        0, int(steps), body, (y, jnp.zeros_like(y), y, best_val0),
    )
    del y_out
    return best_y, jnp.stack([jnp.int32(0), jnp.int32(0), jnp.int32(steps)])


def _minimize_x(
    problem: MinimaxProblem,
    y: Array,
    *,
    steps: int,
    x_init: Array | None = None,
) -> tuple[Array, Array]:
    dtype = y.dtype
    if x_init is not None:
        x = problem.project_x(x_init)
    else:
        x = problem.project_x(jnp.zeros(problem.dim_x, dtype=dtype))
    lr = 1.0 / max(float(problem.ell_x or problem.ell or 0.0), _ABS_TOL)
    beta = jnp.asarray(0.9, dtype=dtype)
    f_y = lambda xx: problem.f(xx, y)

    def body(_i, carry):
        x_cur, v_cur, best_x, best_val = carry
        x_ahead = x_cur + beta * v_cur
        gx, _gy_neg = problem.grad_f(x_ahead, y)
        x_new = problem.project_x(x_cur - lr * gx + beta * v_cur)
        v_new = x_new - x_cur
        val = f_y(x_new)
        improve = val < best_val
        best_x = jnp.where(improve, x_new, best_x)
        best_val = jnp.where(improve, val, best_val)
        return x_new, v_new, best_x, best_val

    best_val0 = f_y(x)
    x_out, _v_out, best_x, _best_val = jax.lax.fori_loop(
        0, int(steps), body, (x, jnp.zeros_like(x), x, best_val0),
    )
    del x_out
    return best_x, jnp.stack([jnp.int32(0), jnp.int32(0), jnp.int32(steps)])

