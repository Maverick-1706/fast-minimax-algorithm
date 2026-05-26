"""Public solver entry points built on top of the internal framework modules."""

from __future__ import annotations

import logging
from typing import Optional

import jax.numpy as jnp
from jax import Array

from minimax_aipe.oracles import eg_step
from minimax_aipe.problem import OracleStats, SolverResult, MinimaxProblem
from minimax_aipe._precision import ABS_TOL as _ABS_TOL
from minimax_aipe._framework.loops import _algorithm_3, _build_oracle_stats
from minimax_aipe._framework.oracles import _maximize_y_auto
from minimax_aipe._framework.params import (
    _compute_loop_params,
    _diam,
    _ell,
    _normalize_initial_z,
    _resolve_gamma,
    _safe_gap,
    _split,
    _validate_solver_inputs,
)
from minimax_aipe._framework.pipeline import _get_pipeline
from minimax_aipe._framework.types import _stats_array


logger = logging.getLogger(__name__)


def _build_solver_setup(
    problem: MinimaxProblem,
    epsilon: float,
    *,
    gamma: float | None,
    M_saddle: str,
    m_lazy: int,
    npe_T_factor: float,
    z0: Optional[Array],
    no_restart: bool,
    fixed_inner_iters: Optional[int],
):
    _validate_solver_inputs(problem, epsilon, M_saddle)
    resolved_gamma = _resolve_gamma(problem, gamma, M_saddle, m_lazy)
    mu_x = epsilon / (2.0 * max(_diam(problem.D_x), _ABS_TOL) ** 3)
    mu_y = epsilon / (2.0 * max(_diam(problem.D_y), _ABS_TOL) ** 3)
    params = _compute_loop_params(
        problem, epsilon, resolved_gamma, npe_T_factor, m_lazy=m_lazy,
        no_restart=no_restart, fixed_inner_iters=fixed_inner_iters,
    )
    z0_start = _normalize_initial_z(problem, z0)
    return resolved_gamma, mu_x, mu_y, params, z0_start


def solve(
    problem: MinimaxProblem,
    epsilon: float,
    *,
    gamma: float | None = None,
    M_saddle: str = "npe",
    m_lazy: int = -1,
    npe_T_factor: float = 1.0,
    z0: Optional[Array] = None,
    verbose: bool = False,
    no_restart: bool = False,
    no_acceleration: bool = False,
    fixed_inner_iters: Optional[int] = None,
    _allow_recovery: bool = True,
) -> SolverResult:
    gamma, mu_x, mu_y, params, z0_start = _build_solver_setup(
        problem, epsilon,
        gamma=gamma,
        M_saddle=M_saddle,
        m_lazy=m_lazy,
        npe_T_factor=npe_T_factor,
        z0=z0,
        no_restart=no_restart,
        fixed_inner_iters=fixed_inner_iters,
    )

    if verbose:
        logger.setLevel(logging.DEBUG)

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

    actual_outer = int(jnp.maximum(1, outer_epochs).item())
    oracle_stats = _build_oracle_stats(
        problem, M_saddle, params, stats_array, actual_outer, final_y_calls,
    )
    result = SolverResult(
        x=x_out,
        y=y_out,
        gap=gap,
        iterations=actual_outer,
        oracle_calls=oracle_stats.oracle_calls,
        oracle_stats=oracle_stats,
        converged=gap <= epsilon,
        history=history,
    )

    if (
        not result.converged
        and not no_acceleration
        and M_saddle == "npe"
        and float(problem.rho or 0.0) <= 0.0
    ):
        fallback = solve(
            problem, epsilon, gamma=gamma, M_saddle=M_saddle, m_lazy=m_lazy,
            npe_T_factor=npe_T_factor, z0=z0_start, verbose=verbose,
            no_restart=no_restart, no_acceleration=True,
            fixed_inner_iters=fixed_inner_iters,
        )
        if float(fallback.gap) < float(result.gap):
            fallback_history = dict(fallback.history or {})
            fallback_history["fallback_from_accelerated"] = True
            fallback_history["accelerated_gap"] = float(result.gap)
            fallback_history["accelerated_iterations"] = result.iterations
            result = fallback._replace(history=fallback_history)

    return _maybe_recover_failed_result(
        result, problem, epsilon, gamma=gamma, M_saddle=M_saddle,
        m_lazy=m_lazy, npe_T_factor=npe_T_factor, z0=z0_start,
        verbose=verbose, no_restart=no_restart,
        no_acceleration=no_acceleration, fixed_inner_iters=fixed_inner_iters,
        allow_recovery=_allow_recovery,
    )


def solve_outer_trace(
    problem: MinimaxProblem,
    epsilon: float,
    *,
    gamma: float | None = None,
    M_saddle: str = "npe",
    m_lazy: int = -1,
    npe_T_factor: float = 1.0,
    z0: Optional[Array] = None,
    no_restart: bool = False,
    no_acceleration: bool = False,
    fixed_inner_iters: Optional[int] = None,
) -> SolverResult:
    gamma, mu_x, mu_y, params, z0_start = _build_solver_setup(
        problem, epsilon,
        gamma=gamma,
        M_saddle=M_saddle,
        m_lazy=m_lazy,
        npe_T_factor=npe_T_factor,
        z0=z0,
        no_restart=no_restart,
        fixed_inner_iters=fixed_inner_iters,
    )

    x_cur, _ = _split(problem, z0_start)
    pipeline = _get_pipeline(problem, gamma, params, M_saddle)

    if no_acceleration:
        def epoch_fn(x: Array, w: Optional[Array] = None):
            if w is not None:
                x_new, _u, y_new, inner_calls = pipeline.prox_phi(x, w)
            else:
                x_new, _u, y_new, inner_calls = pipeline.prox_phi(x)
            return x_new, 1, y_new, inner_calls
    else:
        epoch_fn = pipeline.run_outer_epoch

    warm_y = None
    total_inner_calls = jnp.zeros(3, dtype=jnp.int32)
    gap_endpoints: list[float] = []
    oracle_endpoints: list[float] = []
    d = problem.dim_x + problem.dim_y
    final_x = x_cur
    final_y = problem.project_y(jnp.zeros(problem.dim_y, dtype=x_cur.dtype))
    final_gap = float("inf")
    final_stats = OracleStats()
    epochs_used = 0

    for epoch in range(params.S_outer):
        x_new, _calls, warm_y_new, epoch_inner = epoch_fn(x_cur, warm_y)
        x_new.block_until_ready()
        total_inner_calls = total_inner_calls + _stats_array(epoch_inner)
        epochs_used = epoch + 1

        y_hat, final_y_calls = _maximize_y_auto(
            problem, x_new,
            steps=max(20, params.T_middle * params.S_middle),
            M_saddle=M_saddle, gamma=gamma, m_lazy=params.m_lazy,
        )
        z_refined, _cert = eg_step(
            problem, jnp.concatenate([x_new, y_hat]),
            1.0 / (2.0 * max(_ell(problem), _ABS_TOL)),
        )
        final_x, final_y = _split(problem, z_refined)
        final_gap = _safe_gap(problem, final_x, final_y, epsilon)
        final_stats = _build_oracle_stats(
            problem, M_saddle, params, total_inner_calls, epochs_used, final_y_calls,
        )
        gap_endpoints.append(float(final_gap))
        oracle_endpoints.append(float(final_stats.normalized_cost(d)))

        step = float(jnp.linalg.norm(x_new - x_cur))
        x_cur = x_new
        warm_y = warm_y_new
        if step <= params.zeta_1:
            break

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
        "gap_endpoints": gap_endpoints,
        "oracle_endpoints": oracle_endpoints,
    }

    result = SolverResult(
        x=final_x,
        y=final_y,
        gap=final_gap,
        iterations=epochs_used,
        oracle_calls=final_stats.oracle_calls,
        oracle_stats=final_stats,
        converged=final_gap <= epsilon,
        history=history,
    )

    if (
        not result.converged
        and not no_acceleration
        and M_saddle == "npe"
        and float(problem.rho or 0.0) <= 0.0
    ):
        fallback = solve_outer_trace(
            problem, epsilon, gamma=gamma, M_saddle=M_saddle, m_lazy=m_lazy,
            npe_T_factor=npe_T_factor, z0=z0_start, no_restart=no_restart,
            no_acceleration=True, fixed_inner_iters=fixed_inner_iters,
        )
        if float(fallback.gap) < float(result.gap):
            fallback_history = dict(fallback.history or {})
            fallback_history["fallback_from_accelerated"] = True
            fallback_history["accelerated_gap"] = float(result.gap)
            fallback_history["accelerated_iterations"] = result.iterations
            fallback_history["accelerated_gap_endpoints"] = gap_endpoints
            return fallback._replace(history=fallback_history)

    return result


def _maybe_recover_failed_result(
    result: SolverResult,
    problem: MinimaxProblem,
    epsilon: float,
    *,
    gamma: float,
    M_saddle: str,
    m_lazy: int,
    npe_T_factor: float,
    z0: Array,
    verbose: bool,
    no_restart: bool,
    no_acceleration: bool,
    fixed_inner_iters: Optional[int],
    allow_recovery: bool,
) -> SolverResult:
    if (
        not allow_recovery
        or result.converged
        or M_saddle != "npe"
        or float(problem.rho or 0.0) <= 0.0
    ):
        return result

    best = result
    tried: list[tuple[float, float, Optional[int], float]] = []
    candidates = [
        (gamma * 0.25, npe_T_factor, fixed_inner_iters),
        (gamma * 0.25, max(1.0, 2.0 * npe_T_factor), fixed_inner_iters),
        (gamma * 0.10, npe_T_factor, fixed_inner_iters),
    ]

    for cand_gamma, cand_tf, cand_inner in candidates:
        if cand_gamma <= 0:
            continue
        cand = solve(
            problem, epsilon, gamma=cand_gamma, M_saddle=M_saddle, m_lazy=m_lazy,
            npe_T_factor=cand_tf, z0=z0, verbose=verbose, no_restart=no_restart,
            no_acceleration=no_acceleration, fixed_inner_iters=cand_inner,
            _allow_recovery=False,
        )
        tried.append((cand_gamma, cand_tf, cand_inner, float(cand.gap)))
        if float(cand.gap) < float(best.gap):
            best = cand
            if best.converged:
                break

    if best is not result:
        best_history = dict(best.history or {})
        best_history["fallback_from_gamma"] = gamma
        best_history["fallback_from_t_factor"] = npe_T_factor
        best_history["fallback_trials"] = tried
        best = best._replace(history=best_history)
    return best

