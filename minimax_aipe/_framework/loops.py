"""Core nested-loop solver logic for the triple-loop reduction."""

from __future__ import annotations

import logging
from math import ceil
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe._compat import CallStats
from minimax_aipe._precision import ABS_TOL as _ABS_TOL, REG_MIN as _REG_MIN, TINY as _TINY
from minimax_aipe.aipe import aipe
from minimax_aipe.len import len_loop, make_lazy_crn_npe_oracle
from minimax_aipe.npe import make_crn_npe_oracle, npe, project_z
from minimax_aipe.oracles import _block_chol_solve, _stable_lam_update
from minimax_aipe.problem import MinimaxProblem, OracleStats
from minimax_aipe._framework.oracles import (
    _make_psi_oracle,
    _maximize_y_auto,
    _minimize_x_auto,
)
from minimax_aipe._framework.params import _LoopParams, _compute_loop_params, _diam, _ell, _initial_z, _split
from minimax_aipe._framework.pipeline import _get_pipeline
from minimax_aipe._framework.restarts import _restart_jax
from minimax_aipe._framework.surrogates import RegularizedSubproblem, _make_g_problem
from minimax_aipe._framework.types import _stats_array


logger = logging.getLogger(__name__)


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
    del neg_psi_fn

    def _prox_psi(y_bar: Array, warm_z: Optional[Array] = None) -> tuple[Array, Array, Array, Array]:
        return _iProx_Psi(
            problem, x_bar, y_bar, gamma,
            zeta_3=inner_zeta_3,
            params=params,
            M_saddle=M_saddle,
            kernel=kernel,
            z_init=warm_z,
        )

    y0 = problem.project_y(y_init) if y_init is not None else problem.project_y(jnp.zeros(problem.dim_y))

    def _run_middle_epoch(y_cur: Array, warm_z: Optional[Array] = None) -> tuple[Array, int, Array]:
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
    x_out = g_problem.project_x(x_hat)
    gx_out, _ = g_problem.grad_f(x_out, y_hat)
    u_out = -gx_out
    return x_out, u_out, y_hat, total_inner_calls


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
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)
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
        z0 = jnp.concatenate([kernel.project_x(x_bar), kernel.project_y(y_bar)])

    def _F_h(z: Array) -> Array:
        return kernel.operator_F_h(z, x_bar, y_bar)

    proj = lambda z: jnp.concatenate([kernel.project_x(z[: kernel.dim_x]), kernel.project_y(z[kernel.dim_x :])])
    merit = lambda z: jnp.dot(_F_h(z), _F_h(z))

    if M_saddle == "npe":
        crn_oracle_fn = kernel.make_crn_oracle(x_bar, y_bar, npe_gamma, tol=zeta_3)

        def _run_inner(z: Array) -> tuple[Array, int]:
            return npe(crn_oracle_fn, _F_h, z, inner_T, npe_gamma, project=proj, fn=merit)

    elif M_saddle == "len":
        def _run_inner(z: Array) -> tuple[Array, int]:
            def _crn_with_cached_hessian(z_bar: Array, H_snapshot: Array) -> tuple[Array, Array]:
                g = _F_h(z_bar)
                dtype_local = z_bar.dtype
                tiny_local = jnp.asarray(_TINY, dtype=dtype_local)
                tol_jax = jnp.asarray(zeta_3, dtype=dtype_local)
                lam0 = jnp.asarray(npe_gamma / 2.0, dtype=dtype_local)
                dim_x_local = kernel.dim_x
                J_xx = H_snapshot[:dim_x_local, :dim_x_local]
                J_xy = H_snapshot[:dim_x_local, dim_x_local:]
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
                    return (_stable_lam_update(lam, lam_candidate, i), z_new, i + 1, lam)

                lam, z_half, n_secular, _prev = jax.lax.while_loop(
                    cond, body,
                    (lam0, z_bar, jnp.int32(0), jnp.asarray(-1.0, dtype=dtype_local)),
                )
                d_eff = z_half - z_bar
                u = -(g + H_snapshot @ d_eff + lam * d_eff)
                stats = jnp.stack([jnp.int32(1), n_secular, jnp.int32(1)])
                return z_half, u, stats

            def _len_oracle(z_bar: Array, z_snapshot: Array) -> tuple[Array, Array, Array]:
                xs = z_snapshot[: kernel.dim_x]
                ys = z_snapshot[kernel.dim_x :]
                H = kernel.jacobian_F_h(xs, ys, x_bar, y_bar)
                return _crn_with_cached_hessian(z_bar, H)

            max_norm_val = 100.0 * max(D, 1.0)
            z_out, epoch_stats = len_loop(
                _len_oracle, _F_h, z, inner_T, npe_gamma,
                m=params.m_lazy, project=proj, fn=merit,
                eta_floor=float(_ABS_TOL), max_norm=float(max_norm_val),
            )
            return z_out, epoch_stats
    else:
        raise ValueError(f"Unknown M_saddle={M_saddle!r}; expected 'npe' or 'len'.")

    def _run_inner_warm(z: Array, _warm):
        z_new, inner_stats = _run_inner(z)
        return z_new, inner_stats, None, inner_stats

    z_hat, epochs, _, calls = _restart_jax(
        _run_inner_warm, z0, params.S_inner,
        step_tol=max(zeta_3 * 0.01, _ABS_TOL),
        stats_init=jnp.zeros(3, dtype=jnp.int32),
    )
    del epochs

    z_out = proj(z_hat)
    F_out = _F_h(z_out)
    _x_out, y_out = z_out[: kernel.dim_x], z_out[kernel.dim_x :]
    v_out = -F_out[kernel.dim_x :]
    return y_out, v_out, z_hat, calls


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
    del mu_x, mu_y, zeta_1, verbose
    if params is None:
        params = _compute_loop_params(problem, epsilon=0.1, gamma=gamma)
    if z0 is None:
        z0 = _initial_z(problem)

    x0, _y0 = _split(problem, z0)
    pipeline = _get_pipeline(problem, gamma, params, M_saddle)
    logger.debug("Pipeline kernel: %r", pipeline.kernel)

    if no_acceleration:
        def _non_accel_epoch(x: Array, w: Optional[Array] = None) -> tuple[Array, int, Optional[Array], Array]:
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
        "Algorithm 3: φ=%.4e  |∇φ|=%.3e  inner_crn=%d  inner_linear=%d  outer_epochs=%d/%d",
        phi_val, grad_norm, inner_crn, inner_linear, int(outer_epochs), params.S_outer,
    )

    z_hat = jnp.concatenate([x_hat, y_hat])
    return z_hat, CallStats(total_inner_calls), outer_epochs, final_y_calls


def _solve_saddle_subproblem(
    problem: MinimaxProblem,
    z0: Array,
    gamma: float,
    params: _LoopParams,
    M_saddle: str,
    tolerance: float = 0.0,
    kernel: Optional[RegularizedSubproblem] = None,
) -> tuple[Array, int]:
    del kernel
    sub_rho = max(float(problem.rho or 0.0), _REG_MIN)
    npe_gamma = 2.0 * sub_rho
    mu_inner = gamma / 2.0
    inner_T = max(1, min(200, int(ceil((npe_gamma / max(mu_inner, _ABS_TOL)) ** (2.0 / 3.0)))))
    inner_T = min(inner_T, params.T_inner)
    proj = lambda z: project_z(problem, z)
    merit = lambda z: jnp.dot(problem.operator_F(z), problem.operator_F(z))

    if M_saddle == "npe":
        oracle = make_crn_npe_oracle(problem, npe_gamma, tol=tolerance)

        def _run_inner(z: Array) -> tuple[Array, int]:
            return npe(oracle, problem.operator_F, z, inner_T, npe_gamma, project=proj, fn=merit)

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
    del epochs
    return z_hat, calls


def _build_oracle_stats(
    problem: MinimaxProblem,
    M_saddle: str,
    params: _LoopParams,
    stats_array,
    outer_epochs: int,
    final_y_calls: Array,
) -> OracleStats:
    stats = _stats_array(stats_array)
    inner_crn = int(stats[0])
    inner_linear = int(stats[1])
    inner_grad = int(stats[2])

    if M_saddle == "npe":
        inner_hessians = inner_crn
    else:
        inner_hessians = inner_crn // max(params.m_lazy, 1)

    inner_proj = inner_linear + inner_crn + 2
    actual_outer = max(1, int(outer_epochs))
    middle_grad = actual_outer * params.T_outer * params.S_middle * params.T_middle
    final_maximize_y_grad = (
        int(final_y_calls[2].item())
        if final_y_calls.shape[0] > 2
        else max(20, params.T_middle * params.S_middle)
    )
    total_hidden_grad = middle_grad + final_maximize_y_grad
    total_hidden_proj = total_hidden_grad
    final_eg_grad = 2
    final_eg_proj = 2
    final_y_crn = int(final_y_calls[0].item())
    total_oracle_calls = inner_crn + final_y_crn

    return OracleStats(
        grad_calls=inner_grad + total_hidden_grad + final_eg_grad,
        hessian_calls=inner_hessians,
        hvp_calls=0,
        crn_calls=inner_crn + final_y_crn,
        projection_calls=inner_proj + final_eg_proj + total_hidden_proj,
        linear_solves=inner_linear + int(final_y_calls[1].item()),
        oracle_calls=total_oracle_calls,
        call_type="crn",
        fn_evals=0,
    )
