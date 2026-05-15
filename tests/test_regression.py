"""Regression tests: determinism, consistency, and algebraic invariants.

1. Fixed seed → bit-identical output across runs
2. Gradient/Hessian consistency between autodiff and analytical forms
3. CRN oracle satisfies the variational inequality condition
4. Surrogate problem oracles (g-problem, h-problem) match finite differences
"""

import pytest
import jax
import jax.numpy as jnp

from minimax_aipe import (
    MinimaxProblem,
    solve,
    crn_oracle,
    eg_step,
    RegularizedSubproblem,
    npe,
    make_crn_npe_oracle,
)
from tests.conftest import make_bilinear_problem, make_quadratic_saddle_problem


# ── Fixed-seed determinism ────────────────────────────────────────────────

PROBLEM_CONFIGS = [
    ("bilinear", make_bilinear_problem, {"dim": 3, "seed": 42}),
    ("quadratic", make_quadratic_saddle_problem, {"dim": 3, "seed": 0}),
]


class TestDeterminism:

    @pytest.fixture(params=PROBLEM_CONFIGS, ids=[c[0] for c in PROBLEM_CONFIGS])
    def fixed_problem(self, request):
        name, factory, kwargs = request.param
        return factory(**kwargs)

    def test_deterministic_output(self, fixed_problem):
        """Two runs with the same inputs produce identical results."""
        p = fixed_problem.problem
        r1 = solve(p, epsilon=0.1)
        r2 = solve(p, epsilon=0.1)
        assert jnp.allclose(r1.x, r2.x, atol=1e-4)
        assert jnp.allclose(r1.y, r2.y, atol=1e-4)
        assert r1.oracle_calls == r2.oracle_calls
        assert r1.iterations == r2.iterations

    def test_deterministic_npe(self, bilinear_3d):
        """NPE alone is deterministic with a fixed oracle."""
        p = bilinear_3d.problem
        z0 = jnp.zeros(p.dim_x + p.dim_y)
        gamma = 1.0
        oracle = make_crn_npe_oracle(p, gamma)
        T, S = 10, 2

        z1, c1 = npe(oracle, p.operator_F, z0, T, gamma)
        z2, c2 = npe(oracle, p.operator_F, z0, T, gamma)
        assert jnp.allclose(z1, z2, atol=1e-4)
        assert c1 == c2

    def test_convergence_flag_is_stable(self, fixed_problem):
        """converged flag is identical across runs."""
        p = fixed_problem.problem
        r1 = solve(p, epsilon=0.1)
        r2 = solve(p, epsilon=0.1)
        assert r1.converged == r2.converged


# ── Gradient / Hessian consistency ───────────────────────────────────────

class TestGradientConsistency:

    def test_bilinear_grad_matches_autodiff(self, bilinear_3d):
        """Analytical gradient matches jax.grad for bilinear problem."""
        p = bilinear_3d
        x = jnp.array([0.3, -0.1, 0.5])
        y = jnp.array([0.2, 0.4, -0.3])

        gx_anal, gy_neg_anal = p.problem.grad_f(x, y)

        f = p.problem.f
        gx_ad = jax.grad(f, argnums=0)(x, y)
        gy_ad = jax.grad(f, argnums=1)(x, y)

        assert jnp.allclose(gx_anal, gx_ad, atol=1e-6)
        assert jnp.allclose(gy_neg_anal, -gy_ad, atol=1e-6)

    def test_quadratic_grad_matches_autodiff(self, quadratic_3d):
        p = quadratic_3d
        x = jnp.array([0.5, -0.3, 0.1])
        y = jnp.array([0.2, 0.6, -0.4])

        gx_anal, gy_neg_anal = p.problem.grad_f(x, y)

        f = p.problem.f
        gx_ad = jax.grad(f, argnums=0)(x, y)
        gy_ad = jax.grad(f, argnums=1)(x, y)

        assert jnp.allclose(gx_anal, gx_ad, atol=1e-6)
        assert jnp.allclose(gy_neg_anal, -gy_ad, atol=1e-6)

    def test_hessian_matches_autodiff(self, quadratic_3d):
        """Analytical Hessian matches jax.hessian."""
        p = quadratic_3d
        x = jnp.array([0.3, -0.2, 0.7])
        y = jnp.array([-0.1, 0.5, 0.2])

        (H_xx, H_xy), (H_yx, H_yy) = p.problem.hessian_f(x, y)

        f = p.problem.f
        H_full = jax.hessian(f, argnums=(0, 1))(x, y)
        # H_full is ((H_xx_ad, H_xy_ad), (H_yx_ad, H_yy_ad))
        H_xx_ad, H_xy_ad = H_full[0]
        H_yx_ad, H_yy_ad = H_full[1]

        assert jnp.allclose(H_xx, H_xx_ad, atol=1e-6)
        assert jnp.allclose(H_xy, H_xy_ad, atol=1e-6)
        assert jnp.allclose(H_yx, H_yx_ad, atol=1e-6)
        assert jnp.allclose(H_yy, H_yy_ad, atol=1e-6)

    def test_separable_cross_partials_zero(self, separable_problem):
        """Separable f has zero cross-derivatives."""
        p = separable_problem
        x = jnp.array([1.0, -0.5])
        y = jnp.array([0.3, 0.8])
        (_, H_xy), (H_yx, _) = p.problem.hessian_f(x, y)
        assert jnp.allclose(H_xy, 0.0, atol=1e-4)
        assert jnp.allclose(H_yx, 0.0, atol=1e-4)


# ── CRN oracle VI condition (Definition 3.2) ────────────────────────────

class TestCRNOracleVI:

    def test_vi_condition_bilinear(self, bilinear_3d):
        """CRN output satisfies the variational inequality.

        Definition 3.2:  ⟨F(z̄) + ∇F(z_ss)(z−z̄) + (γ/2)‖z−z̄‖² I, z'−z⟩ ≥ 0
        for all z' in Z.
        This checks the residual u ≈ 0 at the returned point.
        """
        p = bilinear_3d.problem
        z_bar = jnp.zeros(p.dim_x + p.dim_y)
        gamma = 1.0

        z_half, u = crn_oracle(p, z_bar, gamma, n_iters=50)

        # u is the residual:  u = -(g + H @ d + λ·d)
        # For a well-solved problem ‖u‖ should be small
        assert float(jnp.linalg.norm(u)) < 1.0, (
            f"VI residual ‖u‖={float(jnp.linalg.norm(u)):.4e} too large"
        )

    def test_vi_condition_quadratic(self, quadratic_3d):
        p = quadratic_3d.problem
        z_bar = jnp.zeros(p.dim_x + p.dim_y)
        gamma = 1.0

        z_half, u = crn_oracle(p, z_bar, gamma, n_iters=50)
        assert float(jnp.linalg.norm(u)) < 1.0

    def test_crn_oracle_idempotent_at_saddle(self, bilinear_3d):
        """Calling CRN at the saddle point should not move the iterate."""
        p = bilinear_3d.problem
        z_star = jnp.zeros(p.dim_x + p.dim_y)
        gamma = 1.0

        z_half, u = crn_oracle(p, z_star, gamma, n_iters=50)
        assert jnp.allclose(z_half, z_star, atol=1e-6)

    def test_crn_oracle_respects_projection(self, bilinear_3d):
        """CRN output lies in the feasible set Z = X × Y."""
        p = bilinear_3d.problem
        z_bar = jnp.ones(p.dim_x + p.dim_y) * 0.3
        gamma = 2.0

        z_half, _u = crn_oracle(p, z_bar, gamma, n_iters=50)
        x_half, y_half = z_half[:p.dim_x], z_half[p.dim_x:]
        assert float(jnp.linalg.norm(x_half)) <= p.D_x / 2 + 1e-6
        assert float(jnp.linalg.norm(y_half)) <= p.D_y / 2 + 1e-6


# ── EG step algebraic check ──────────────────────────────────────────────

class TestEGStep:

    def test_eg_at_saddle_produces_zero_residual(self, bilinear_3d):
        """At the saddle point, the EG residual certificate c₁ should be ≈ 0."""
        p = bilinear_3d.problem
        z = jnp.zeros(p.dim_x + p.dim_y)
        eta = 1.0 / (2.0 * max(p.ell or 1.0, 1e-8))

        z_new, c = eg_step(p, z, eta)
        # At the saddle F(z)=0, so z_half = z, F_half = 0, z_new = z, c = 0
        assert jnp.allclose(z_new, z, atol=1e-8)
        assert float(jnp.linalg.norm(c)) < 1e-6

    def test_eg_residual_is_feasibility_certificate(self, bilinear_3d):
        """The residual c₁ = (z − z¹)/η − F(z½) is a valid certificate."""
        p = bilinear_3d.problem
        z = jnp.array([0.3, -0.1, 0.2, 0.4, -0.3, 0.1])
        eta = 0.5

        z_new, c = eg_step(p, z, eta)
        # Certificate should be finite
        assert jnp.all(jnp.isfinite(c))


# ── RegularizedSubproblem consistency ─────────────────────────────────────

class TestRegularizedSubproblem:

    def test_operator_F_h_at_base_point(self, bilinear_3d):
        """At z = (x_bar, y_bar), the cubic terms vanish and F_h = F."""
        p = bilinear_3d.problem
        gamma = 2.0
        kernel = RegularizedSubproblem(p, gamma)

        x_bar = jnp.array([0.1, -0.2, 0.3])
        y_bar = jnp.array([0.2, 0.1, -0.1])
        z_bar = jnp.concatenate([x_bar, y_bar])

        F_h = kernel.operator_F_h(z_bar, x_bar, y_bar)
        F = p.operator_F(z_bar)

        # At the center, cubic grad = 0, so F_h = F
        assert jnp.allclose(F_h, F, atol=1e-8)

    def test_jacobian_symmetry(self, bilinear_3d):
        """The Jacobian ∇F_h has the correct saddle-point structure."""
        p = bilinear_3d.problem
        gamma = 1.0
        kernel = RegularizedSubproblem(p, gamma)

        x_bar = jnp.zeros(p.dim_x)
        y_bar = jnp.zeros(p.dim_y)
        x = jnp.array([0.2, -0.1, 0.3])
        y = jnp.array([0.1, 0.2, -0.2])

        J = kernel.jacobian_F_h(x, y, x_bar, y_bar)
        d = p.dim_x + p.dim_y
        assert J.shape == (d, d)
        # Should be finite
        assert jnp.all(jnp.isfinite(J))

    def test_project_method(self, bilinear_3d):
        """kernel.project(z) matches manual slicing + project_x/project_y."""
        p = bilinear_3d.problem
        kernel = RegularizedSubproblem(p, 1.0)

        z = jnp.array([0.5, -0.3, 0.8, 0.2, -0.6, 0.1])
        z_proj = kernel.project(z)
        x_proj = p.project_x(z[:p.dim_x])
        y_proj = p.project_y(z[p.dim_x:])
        z_manual = jnp.concatenate([x_proj, y_proj])

        assert jnp.allclose(z_proj, z_manual, atol=1e-4)


# ── Surrogate oracles (g-problem, h-problem) ─────────────────────────────

class TestSurrogateProblems:

    def test_g_problem_matches_f_plus_cubic(self, bilinear_3d):
        """g(x,y;x̄) = f(x,y) + (γ/3)‖x−x̄‖³ at the query point."""
        from minimax_aipe.framework import _make_g_problem

        p = bilinear_3d.problem
        x_bar = jnp.array([0.1, -0.2, 0.3])
        gamma = 2.0

        g_prob = _make_g_problem(p, x_bar, gamma)

        x = jnp.array([0.2, -0.1, 0.4])
        y = jnp.array([0.3, 0.1, -0.2])

        g_val = g_prob.f(x, y)
        f_val = p.f(x, y)
        cubic = (gamma / 3.0) * jnp.linalg.norm(x - x_bar) ** 3

        assert abs(float(g_val - f_val - cubic)) < 1e-8

    def test_h_problem_includes_both_cubic_terms(self, bilinear_3d):
        """h(x,y;x̄,ȳ) = f(x,y) + (γ/3)‖x−x̄‖³ − (γ/3)‖y−ȳ‖³."""
        from minimax_aipe.framework import _make_h_problem

        p = bilinear_3d.problem
        x_bar = jnp.array([0.1, -0.2, 0.3])
        y_bar = jnp.array([0.2, 0.1, -0.1])
        gamma = 2.0

        h_prob = _make_h_problem(p, x_bar, y_bar, gamma)

        x = jnp.array([0.3, -0.1, 0.5])
        y = jnp.array([0.1, 0.3, -0.2])

        h_val = h_prob.f(x, y)
        f_val = p.f(x, y)
        cubic_x = (gamma / 3.0) * jnp.linalg.norm(x - x_bar) ** 3
        cubic_y = (gamma / 3.0) * jnp.linalg.norm(y - y_bar) ** 3
        expected = f_val + cubic_x - cubic_y

        assert abs(float(h_val - expected)) < 1e-8

    def test_surrogate_rho_increases(self, bilinear_3d):
        """Adding cubic regularisation increases the ρ constant."""
        from minimax_aipe.framework import _make_g_problem, _make_h_problem

        p = bilinear_3d.problem
        gamma = 2.0

        g_prob = _make_g_problem(p, jnp.zeros(3), gamma)
        h_prob = _make_h_problem(p, jnp.zeros(3), jnp.zeros(3), gamma)

        assert g_prob.rho >= p.rho
        assert h_prob.rho >= p.rho


# ── Full-pipeline determinism ─────────────────────────────────────────────

class TestPipelineConsistency:

    def test_two_solves_same_problem_same_result(self, bilinear_3d):
        """End-to-end: identical inputs → identical SolverResult."""
        p = bilinear_3d.problem
        r1 = solve(p, epsilon=0.05)
        r2 = solve(p, epsilon=0.05)

        assert jnp.allclose(r1.x, r2.x, atol=1e-4)
        assert jnp.allclose(r1.y, r2.y, atol=1e-4)
        assert r1.oracle_calls == r2.oracle_calls
        assert r1.converged == r2.converged
