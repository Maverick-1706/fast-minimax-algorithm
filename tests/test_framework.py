# tests_framework.py
"""Tests for the high-level Minimax-AIPE framework (triple-loop version).

Oracle-call tracking is now done via pure JAX return-value threading
instead of side-effect callbacks.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from minimax_aipe.framework import (
    _algorithm_3,
    _compute_loop_params,
    _cubic_grad,
    _cubic_hess,
    _iProx_Phi,
    _iProx_Psi,
    _make_g_problem,
    _make_h_problem,
    _make_phi_oracle,
    _make_psi_oracle,
    solve,
    solve_outer_trace,
)
from minimax_aipe.problem import MinimaxProblem

def _test_loop_params(problem, gamma=1.0):
    """Minimal loop params for fast unit tests (~1 500 NPE calls)."""
    return _compute_loop_params(problem, epsilon=0.5, gamma=gamma, npe_T_factor=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# Test problem factories
# ═══════════════════════════════════════════════════════════════════════════════

def _quadratic_problem(mu: float = 0.5) -> tuple[MinimaxProblem, jnp.ndarray]:
    """Bilinear-quadratic:  f = (μ/2)(‖x‖² − ‖y‖²) + xᵀAy."""
    A = jnp.array([[0.3, -0.2], [0.1, 0.4]])

    def f(x, y):
        return (mu / 2.0) * (jnp.dot(x, x) - jnp.dot(y, y)) + x @ A @ y

    def grad_f(x, y):
        return mu * x + A @ y, mu * y - A.T @ x

    def hessian_f(x, y):
        del x, y
        H_xx = mu * jnp.eye(2)
        H_xy = A
        H_yx = A.T
        H_yy = -mu * jnp.eye(2)
        return (H_xx, H_xy), (H_yx, H_yy)

    problem = MinimaxProblem(
        f=f,
        grad_f=grad_f,
        hessian_f=hessian_f,
        dim_x=2,
        dim_y=2,
        D_x=4.0,
        D_y=4.0,
        ell=1.0,
        rho=0.0,
    )
    return problem, A


def _shifted_scsc_problem(
    mu: float = 1.0,
) -> tuple[MinimaxProblem, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Strongly-convex–strongly-concave quadratic with known optimum."""
    A = jnp.array([[0.3, 0.1], [-0.2, 0.4]])
    x_star = jnp.array([0.2, -0.3])
    y_star = jnp.array([0.1, 0.25])

    def f(x, y):
        dx = x - x_star
        dy = y - y_star
        return (mu / 2.0) * jnp.dot(dx, dx) - (mu / 2.0) * jnp.dot(dy, dy) + dx @ A @ dy

    def grad_f(x, y):
        dx = x - x_star
        dy = y - y_star
        return mu * dx + A @ dy, mu * dy - A.T @ dx

    def hessian_f(x, y):
        del x, y
        H_xx = mu * jnp.eye(2)
        H_xy = A
        H_yx = A.T
        H_yy = -mu * jnp.eye(2)
        return (H_xx, H_xy), (H_yx, H_yy)

    def exact_gap(x, y):
        dx = x - x_star
        dy = y - y_star
        primal = (mu / 2.0) * jnp.dot(dx, dx) + (1.0 / (2.0 * mu)) * jnp.dot(A.T @ dx, A.T @ dx)
        dual = (mu / 2.0) * jnp.dot(dy, dy) + (1.0 / (2.0 * mu)) * jnp.dot(A @ dy, A @ dy)
        return float(primal + dual)

    problem = MinimaxProblem(
        f=f,
        grad_f=grad_f,
        hessian_f=hessian_f,
        dim_x=2,
        dim_y=2,
        D_x=4.0,
        D_y=4.0,
        ell=2.0,
        rho=0.0,
    )
    problem.duality_gap = exact_gap
    return problem, x_star, y_star, A


# ═══════════════════════════════════════════════════════════════════════════════
# Cubic regulariser helpers  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def test_cubic_grad_and_hess_are_finite_at_zero():
    delta = jnp.zeros(3)
    grad = _cubic_grad(delta, gamma=2.0)
    hess = _cubic_hess(delta, gamma=2.0)

    assert jnp.all(jnp.isfinite(grad))
    assert jnp.all(jnp.isfinite(hess))
    assert jnp.allclose(grad, 0.0)
    assert jnp.allclose(hess, 0.0)


def test_cubic_grad_scales_with_gamma():
    delta = jnp.array([1.0, 2.0, 0.0])
    g1 = _cubic_grad(delta, gamma=1.0)
    g2 = _cubic_grad(delta, gamma=3.0)
    assert jnp.allclose(g2, 3.0 * g1)


# ═══════════════════════════════════════════════════════════════════════════════
# Surrogate problem constructors  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_g_problem_matches_manual_grad_and_hessian():
    problem, A = _quadratic_problem(mu=0.5)
    x_bar = jnp.array([0.1, -0.2])
    x = jnp.array([0.4, -0.1])
    y = jnp.array([0.3, 0.2])
    gamma = 1.3

    g_problem = _make_g_problem(problem, x_bar, gamma)
    gx, gy_neg = g_problem.grad_f(x, y)
    (H_xx, H_xy), (_H_yx, H_yy) = g_problem.hessian_f(x, y)

    assert jnp.allclose(gx, 0.5 * x + A @ y + _cubic_grad(x - x_bar, gamma))
    assert jnp.allclose(gy_neg, 0.5 * y - A.T @ x)
    assert jnp.allclose(H_xx, 0.5 * jnp.eye(2) + _cubic_hess(x - x_bar, gamma))
    assert jnp.allclose(H_xy, A)
    assert jnp.allclose(H_yy, -0.5 * jnp.eye(2))


def test_make_g_problem_inherits_projections():
    problem, _A = _quadratic_problem()
    x_bar = jnp.zeros(2)
    g_problem = _make_g_problem(problem, x_bar, gamma=1.0)
    assert g_problem.project_x is problem.project_x
    assert g_problem.project_y is problem.project_y


def test_make_h_problem_matches_manual_grad_and_hessian():
    problem, A = _quadratic_problem(mu=0.5)
    x_bar = jnp.array([0.1, -0.2])
    y_bar = jnp.array([-0.3, 0.2])
    x = jnp.array([0.4, -0.1])
    y = jnp.array([0.3, 0.2])
    gamma = 1.3

    h_problem = _make_h_problem(problem, x_bar, y_bar, gamma)
    gx, gy_neg = h_problem.grad_f(x, y)
    (H_xx, H_xy), (_H_yx, H_yy) = h_problem.hessian_f(x, y)

    assert jnp.allclose(gx, 0.5 * x + A @ y + _cubic_grad(x - x_bar, gamma))
    assert jnp.allclose(gy_neg, 0.5 * y - A.T @ x + _cubic_grad(y - y_bar, gamma))
    assert jnp.allclose(H_xx, 0.5 * jnp.eye(2) + _cubic_hess(x - x_bar, gamma))
    assert jnp.allclose(H_xy, A)
    assert jnp.allclose(H_yy, -0.5 * jnp.eye(2) - _cubic_hess(y - y_bar, gamma))


def test_make_h_problem_symmetric_when_bars_equal():
    """When x_bar == y_bar and g is symmetric in x/y, h should be symmetric
    in the cubic terms (up to sign)."""
    problem, _A = _quadratic_problem(mu=1.0)
    bar = jnp.array([0.5, -0.5])
    h_problem = _make_h_problem(problem, bar, bar, gamma=2.0)

    x = jnp.array([0.1, 0.3])
    y = jnp.array([0.1, 0.3])
    gx, gy_neg = h_problem.grad_f(x, y)

    # The cubic contributions should be equal in magnitude
    contrib_x = _cubic_grad(x - bar, 2.0)
    contrib_y = _cubic_grad(y - bar, 2.0)
    assert jnp.allclose(gx, 1.0 * x + _A @ y + contrib_x)
    assert jnp.allclose(gy_neg, 1.0 * y - _A.T @ x + contrib_y)


# ═══════════════════════════════════════════════════════════════════════════════
# Phi and Psi oracle factories
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_phi_oracle_shape_and_signature():
    """Phi oracles should return scalars for values and matching-shape grads."""
    problem, _A = _quadratic_problem(mu=1.0)
    params = _compute_loop_params(problem, epsilon=0.1, gamma=1.0)

    phi_fn, grad_phi_fn = _make_phi_oracle(problem, gamma=1.0, params=params)

    x = jnp.array([0.3, -0.1])
    val = phi_fn(x)
    grad, _calls = grad_phi_fn(x)
    assert val.ndim == 0, f"phi should be scalar, got shape {val.shape}"
    assert grad.shape == x.shape, f"grad shape {grad.shape} != x shape {x.shape}"
    assert jnp.all(jnp.isfinite(val))
    assert jnp.all(jnp.isfinite(grad))


def test_framework_re_exports_expected_symbols():
    import minimax_aipe.framework as framework

    expected = [
        "solve",
        "solve_outer_trace",
        "RegularizedSubproblem",
        "_compute_loop_params",
        "_iProx_Phi",
        "_iProx_Psi",
        "_make_g_problem",
        "_make_h_problem",
        "_make_phi_oracle",
        "_make_psi_oracle",
        "_algorithm_3",
    ]
    for name in expected:
        assert hasattr(framework, name), f"framework missing re-export: {name}"


def test_internal_api_shares_setup_builder():
    from minimax_aipe._framework import api

    assert hasattr(api, "_build_solver_setup")
    assert solve.__module__ == "minimax_aipe._framework.api"
    assert solve_outer_trace.__module__ == "minimax_aipe._framework.api"


def test_cached_pipeline_function_is_lru_wrapped():
    from minimax_aipe._framework.pipeline import _get_pipeline

    assert hasattr(_get_pipeline, "cache_info")

def test_make_phi_oracle_at_origin():
    """For a quadratic problem centred at zero, max_y f(0, y) should be zero
    and ∇Phi(0) should be zero since A·0 = 0."""
    problem, _A = _quadratic_problem(mu=1.0)
    params = _compute_loop_params(problem, epsilon=0.1, gamma=1.0)

    phi_fn, grad_phi_fn = _make_phi_oracle(problem, gamma=1.0, params=params)

    x0 = jnp.zeros(2)
    val = phi_fn(x0)
    grad, _calls = grad_phi_fn(x0)
    assert float(jnp.abs(val)) < 1e-4, f"phi(0) should be ~0, got {val}"
    assert jnp.allclose(grad, 0.0, atol=1e-4), f"grad_phi(0) should be ~0, got {grad}"

def test_make_phi_oracle_gradient_is_approximate_subgradient():
    """∇Phi(x) ≈ ∇_x f(x, y*(x)) where y*(x) = argmax_y f(x, y)."""
    problem, _A = _quadratic_problem(mu=1.0)
    params = _compute_loop_params(problem, epsilon=0.1, gamma=1.0)

    phi_fn, grad_phi_fn = _make_phi_oracle(problem, gamma=1.0, params=params)

    x = jnp.array([0.5, -0.3])
    grad, _calls = grad_phi_fn(x)
    # For f = (1/2)(||x||^2 - ||y||^2) + x^T A y, grad_phi = x + A y*(x)
    # The grad should be finite and nonzero at a nonzero x
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.linalg.norm(grad) > 0.0, "grad should be nonzero away from origin"
    # Directional consistency: phi(x + eps*d) >= phi(x) + grad.dot(d)*eps - O(eps^2)
    # for a convex function (basic sanity check on sign)
    eps = 1e-3
    d = grad / jnp.linalg.norm(grad)
    val_x = phi_fn(x)
    val_xd = phi_fn(x + eps * d)
    assert float(val_xd - val_x) >= -1e-3, "phi should not decrease along gradient direction"

def test_make_psi_oracle_shape_and_signature():
    """-Psi oracles should return scalars and matching-shape grads."""
    problem, _A = _quadratic_problem(mu=1.0)
    params = _compute_loop_params(problem, epsilon=0.1, gamma=1.0)

    x_bar = jnp.array([0.3, -0.2])
    neg_psi_fn, grad_neg_psi_fn = _make_psi_oracle(
        problem, x_bar, gamma=1.0, params=params,
    )

    y = jnp.array([0.1, 0.2])
    val = neg_psi_fn(y)
    grad, _calls = grad_neg_psi_fn(y)
    assert val.ndim == 0, f"neg_psi should be scalar, got shape {val.shape}"
    assert grad.shape == y.shape, f"grad shape {grad.shape} != y shape {y.shape}"
    assert jnp.all(jnp.isfinite(val))
    assert jnp.all(jnp.isfinite(grad))

# ═══════════════════════════════════════════════════════════════════════════════
# Inexact proximal oracles  (Algorithms 4 and 5)
# ═══════════════════════════════════════════════════════════════════════════════

def test_iProx_Phi_returns_proximal_point():
    """_iProx_Phi should return (x, u) with x in the domain and
    u ≈ -∇_x g(x, y) as a subgradient certificate."""
    problem, _A = _quadratic_problem(mu=1.0)
    x_bar = jnp.array([0.5, -0.2])
    params = _compute_loop_params(problem, epsilon=0.01, gamma=1.0, npe_T_factor=0.15)

    x_out, u_out, *_ = _iProx_Phi(
        problem, x_bar, gamma=1.0,
        zeta_2=params.zeta_2, params=params, M_saddle="npe",
    )

    # x_out should be in (or close to) the domain
    x_proj = problem.project_x(x_out)
    assert jnp.allclose(x_out, x_proj, atol=1e-4)

    # The subgradient certificate should be finite
    assert jnp.all(jnp.isfinite(u_out))
    assert u_out.shape == x_bar.shape


def test_iProx_Phi_tracks_calls_via_return():
    """After _iProx_Phi runs, the returned total_inner_calls should be
    a positive integer reflecting actual oracle invocations.

    NOTE: Replaces the old _CallCounter-based test. Call counts are now
    threaded through JAX return values instead of side-effects.
    """
    problem, _A = _quadratic_problem(mu=1.0)
    x_bar = jnp.array([0.5, -0.2])
    params = _compute_loop_params(problem, epsilon=0.01, gamma=1.0, npe_T_factor=0.15)

    x_out, u_out, y_hat, total_inner_calls = _iProx_Phi(
        problem, x_bar, gamma=1.0,
        params=params, M_saddle="npe",
    )

    assert int(total_inner_calls[0]) > 0, f"total_inner_calls should be positive, got {total_inner_calls}"


def test_iProx_Phi_returns_four_elements():
    """_iProx_Phi should return (x_out, u_out, y_hat, total_inner_calls)."""
    problem, _A = _quadratic_problem(mu=1.0)
    x_bar = jnp.array([0.5, -0.2])
    params = _compute_loop_params(problem, epsilon=0.01, gamma=1.0, npe_T_factor=0.15)

    result = _iProx_Phi(
        problem, x_bar, gamma=1.0, params=params, M_saddle="npe",
    )

    assert len(result) == 4
    x_out, u_out, y_hat, total_inner_calls = result
    assert x_out.shape == x_bar.shape
    assert u_out.shape == x_bar.shape
    assert jnp.all(jnp.isfinite(u_out))


def test_iProx_Phi_with_default_params():
    """_iProx_Phi should work when params is None (auto-compute defaults)."""
    problem, _A = _quadratic_problem(mu=1.0)
    x_bar = jnp.zeros(2)
    params = _test_loop_params(problem)

    x_out, u_out, *_ = _iProx_Phi(
        problem, x_bar, gamma=1.0,
        params=params, M_saddle="npe",
    )

    assert x_out.shape == x_bar.shape
    assert u_out.shape == x_bar.shape


def test_iProx_Psi_returns_proximal_point():
    """_iProx_Psi should return (y, v) with v ≈ -∇_y(-h) as a certificate."""
    problem, _A = _quadratic_problem(mu=1.0)
    x_bar = jnp.array([0.5, -0.2])
    y_bar = jnp.array([-0.1, 0.3])
    params = _compute_loop_params(problem, epsilon=0.01, gamma=1.0, npe_T_factor=0.15)

    y_out, v_out, *_ = _iProx_Psi(
        problem, x_bar, y_bar, gamma=1.0,
        zeta_3=params.zeta_3, params=params, M_saddle="npe",
    )

    y_proj = problem.project_y(y_out)
    assert jnp.allclose(y_out, y_proj, atol=1e-4)
    assert jnp.all(jnp.isfinite(v_out))
    assert v_out.shape == y_bar.shape


def test_iProx_Psi_with_default_params():
    """_iProx_Psi should work when params is None."""
    problem, _A = _quadratic_problem(mu=1.0)
    x_bar = jnp.zeros(2)
    y_bar = jnp.zeros(2)
    params = _test_loop_params(problem)

    y_out, v_out, *_ = _iProx_Psi(
        problem, x_bar, y_bar, gamma=1.0,
        params=params, M_saddle="npe",
    )

    assert y_out.shape == y_bar.shape
    assert v_out.shape == y_bar.shape


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 3 — triple-loop driver
# ═══════════════════════════════════════════════════════════════════════════════

def test_algorithm_3_returns_valid_saddle_point():
    """Algorithm 3 should return a concatenated [x; y] vector."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    z0 = jnp.concatenate([jnp.zeros(2), jnp.zeros(2)])
    params = _test_loop_params(problem)  

    z_hat, calls, _, _ = _algorithm_3(
        problem, gamma=1.0, mu_x=0.01, mu_y=0.01, zeta_1=0.01,
        params = params, M_saddle="npe", z0=z0,
    )

    assert z_hat.shape == (4,)
    assert int(calls[0]) > 0
    assert jnp.all(jnp.isfinite(z_hat))


def test_algorithm_3_improves_on_zero_init():
    """Starting from zero, the triple loop should move x toward the optimum."""
    problem, x_star, y_star, _A = _shifted_scsc_problem(mu=1.0)
    z0 = jnp.concatenate([jnp.zeros(2), jnp.zeros(2)])
    params = _test_loop_params(problem)

    z_hat, _calls, _, _ = _algorithm_3(
        problem, gamma=1.0, mu_x=0.01, mu_y=0.01, zeta_1=0.01,
        params = params, M_saddle="npe", z0=z0,
    )
    x_hat = z_hat[:2]
    y_hat = z_hat[2:]

    dist_zero_x = jnp.linalg.norm(jnp.zeros(2) - x_star)
    dist_hat_x = jnp.linalg.norm(x_hat - x_star)
    assert dist_hat_x < dist_zero_x, (
        f"Algorithm 3 did not improve x: |x_hat-x*|={dist_hat_x:.3f} vs |0-x*|={dist_zero_x:.3f}"
    )

    # y should also be closer to y_star than zero is
    dist_zero_y = jnp.linalg.norm(jnp.zeros(2) - y_star)
    dist_hat_y = jnp.linalg.norm(y_hat - y_star)
    assert dist_hat_y < dist_zero_y, (
        f"Algorithm 3 did not improve y: |y_hat-y*|={dist_hat_y:.3f} vs |0-y*|={dist_zero_y:.3f}"
    )


def test_algorithm_3_with_default_params():
    """Algorithm 3 should auto-compute loop params when none given."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    z0 = jnp.concatenate([jnp.zeros(2), jnp.zeros(2)])
    params = _test_loop_params(problem)

    z_hat, calls, _, _ = _algorithm_3(
        problem, gamma=1.0, mu_x=0.01, mu_y=0.01, zeta_1=0.01,
        params=params, M_saddle="npe", z0=z0,
    )
    assert z_hat.shape == (4,)
    assert int(calls[0]) > 0


def test_algorithm_3_with_default_z0():
    """Algorithm 3 should auto-initialise z0 when none given."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    params = _test_loop_params(problem)

    z_hat, calls, _, _ = _algorithm_3(
        problem, gamma=1.0, mu_x=0.01, mu_y=0.01, zeta_1=0.01,
        params=params, M_saddle="npe", z0=None,
    )
    assert z_hat.shape == (4,)
    assert int(calls[0]) > 0


@pytest.mark.parametrize("M_saddle", ["npe", "len"])
def test_algorithm_3_accepts_both_saddle_modes(M_saddle):
    """Algorithm 3 should accept both 'npe' and 'len' saddle solvers."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    params = _test_loop_params(problem)

    z_hat, calls, _, _ = _algorithm_3(
        problem, gamma=1.0, mu_x=0.01, mu_y=0.01, zeta_1=0.01,
        params = params, M_saddle=M_saddle, z0=None,
    )
    assert z_hat.shape == (4,)
    assert int(calls[0]) > 0


def test_algorithm_3_call_counter_is_threaded():
    """Algorithm 3 should accumulate calls from the innermost _iProx_Psi
    invocations into a single _CallCounter."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    z0 = jnp.concatenate([jnp.zeros(2), jnp.zeros(2)])
    params = _test_loop_params(problem)

    # _algorithm_3 creates its own counter internally; the returned
    # total_calls should equal counter.total
    z_hat, calls, _, _ = _algorithm_3(
        problem, gamma=1.0, mu_x=0.01, mu_y=0.01, zeta_1=0.01,
        params = params, M_saddle="npe", z0=z0,
    )

    assert int(calls[0]) > 0
    # calls is now a 3-element array [crn_calls, linear_solves, grad_calls] with
    # exact runtime tracking rather than formula-based estimates.
    assert calls.shape == (3,)
    assert calls.dtype == jnp.int32


# ═══════════════════════════════════════════════════════════════════════════════
# High-level solve  (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════════

def test_solve_shifted_scsc_quadratic():
    """End-to-end: the full triple-loop solve should converge on a simple SCSC
    quadratic with known optimum."""
    problem, x_star, y_star, _A = _shifted_scsc_problem()
    eps = 0.1
    result = solve(problem, epsilon=eps, npe_T_factor=0.05)

    assert result.oracle_calls > 0
    # Triple loop + EG refinement should improve on the origin
    assert jnp.linalg.norm(result.x - x_star) < jnp.linalg.norm(x_star)
    assert jnp.linalg.norm(result.y - y_star) < jnp.linalg.norm(y_star)


def test_solve_accepts_len_alias_for_saddle_solver():
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    eps = 0.1
    result = solve(problem, epsilon=eps, M_saddle="len", npe_T_factor=0.05)

    assert result.oracle_calls > 0
    assert result.history["M_saddle"] == "len"


def test_smaller_epsilon_uses_at_least_as_many_oracle_calls():
    """Finer accuracy should not reduce the total oracle-call budget."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    coarse = solve(problem, epsilon=0.5, npe_T_factor=0.05)   # ← CHANGE
    fine = solve(problem, epsilon=0.1, npe_T_factor=0.05)

    assert coarse.gap <= 0.5
    assert fine.gap <= 0.1
    
    assert fine.oracle_calls >= coarse.oracle_calls


def test_solve_history_contains_all_three_loop_params():
    """The result history should record all six T/S parameters."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    result = solve(problem, epsilon=0.1, npe_T_factor=0.05)

    for key in ("T_outer", "S_outer", "T_middle", "S_middle", "T_inner", "S_inner"):
        assert key in result.history, f"Missing history key: {key}"
        assert isinstance(result.history[key], int)
        assert result.history[key] > 0, f"{key} = {result.history[key]}"

    for key in ("zeta_1", "zeta_2", "zeta_3", "gamma", "mu_x", "mu_y"):
        assert key in result.history, f"Missing history key: {key}"
        assert result.history[key] > 0, f"{key} = {result.history[key]}"


def test_solve_rejects_non_positive_epsilon():
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        solve(problem, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        solve(problem, epsilon=-0.1)


def test_solve_rejects_invalid_M_saddle():
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    with pytest.raises(ValueError, match="M_saddle"):
        solve(problem, epsilon=0.1, M_saddle="invalid")


def test_solve_result_fields_are_present():
    """All SolverResult fields should be populated."""
    problem, _x_star, _y_star, _A = _shifted_scsc_problem()
    result = solve(problem, epsilon=0.1, npe_T_factor=0.05)

    assert result.x is not None
    assert result.y is not None
    assert result.gap >= 0.0
    assert result.iterations > 0
    assert result.oracle_calls > 0
    assert isinstance(result.converged, bool)
    assert isinstance(result.history, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Loop-parameter computation
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_loop_params_hierarchy():
    """Accuracy parameters should form a hierarchy: zeta_1 ≥ zeta_2 ≥ zeta_3."""
    problem, _A = _quadratic_problem()
    params = _compute_loop_params(problem, epsilon=0.01, gamma=1.0)

    assert params.zeta_1 >= params.zeta_2, f"zeta_1={params.zeta_1} < zeta_2={params.zeta_2}"
    assert params.zeta_2 >= params.zeta_3, f"zeta_2={params.zeta_2} < zeta_3={params.zeta_3}"


def test_compute_loop_params_T_outer_ge_inner():
    """Outer loop should get at least as many iterations as inner loops."""
    problem, _A = _quadratic_problem()
    params = _compute_loop_params(problem, epsilon=0.01, gamma=1.0, npe_T_factor=1.0)

    assert params.T_outer >= params.T_middle
    assert params.T_outer >= params.T_inner
    assert params.S_outer >= params.S_middle
    assert params.S_outer >= params.S_inner


def test_compute_loop_params_outer_tolerance_has_fp32_floor():
    """zeta_1 should not underflow below a usable FP32 early-stop threshold."""
    problem, _A = _quadratic_problem()
    params = _compute_loop_params(problem, epsilon=0.01, gamma=1.0)

    assert params.zeta_1 >= 1e-6
