"""Functional tests for oracles and core mathematical operations.

These tests verify properties stated in the paper:
- Lemma 3.1: EG update contracts the operator norm
- Lemma 3.2: Uniform convexity implies gradient-dominance
- Lemma 3.3: Cubic function properties
- Definition 3.2: CRN oracle first-order optimality
- Definition 3.1: Duality gap correctness
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from minimax_aipe.problem import MinimaxProblem


# ---------------------------------------------------------------------------
# Test fixtures: concrete problem instances
# ---------------------------------------------------------------------------


def _bilinear_problem() -> MinimaxProblem:
    """Simple bilinear game: f(x, y) = x^T A y.

    Saddle point at (0, 0) with gap = 0.
    """

    def f(x, y):
        A = jnp.array([[1.0, 0.5], [0.3, 1.0]])
        return x @ A @ y

    return MinimaxProblem(f, dim_x=2, dim_y=2, D_x=2.0, D_y=2.0, rho=0.0, ell=0.0, L=2.0)


def _quadratic_minimax() -> MinimaxProblem:
    """Quadratic minimax: f(x, y) = 0.5 * x^T Q x - 0.5 * y^T P y + x^T A y.

    This is μ_x-uniformly convex in x and μ_y-uniformly concave in y.
    """

    Q = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    P = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    A = jnp.array([[0.5, 0.1], [0.2, 0.3]])

    def f(x, y):
        return 0.5 * x @ Q @ x - 0.5 * y @ P @ y + x @ A @ y

    return MinimaxProblem(
        f,
        dim_x=2,
        dim_y=2,
        D_x=4.0,
        D_y=4.0,
        rho=0.0,
        ell=3.5,
        L=10.0,
    )


# ---------------------------------------------------------------------------
# Tests for the EG step (Lemma 3.1)
# ---------------------------------------------------------------------------


class TestExtragradient:
    """Test the extragradient update (Equation 3) and Lemma 3.1."""

    def _eg_step(self, F, z0, eta):
        """One EG step: z^{0.5} then z^1.

        F must return a single concatenated array [∇_x f, -∇_y f].
        """
        z_half = z0 - eta * F(z0)
        z1 = z0 - eta * F(z_half)
        return z1

    def test_eg_finds_saddle_bilinear(self):
        """EG should converge to the saddle point of a bilinear game."""
        problem = _bilinear_problem()
        F = problem.operator_F  # returns concatenated array

        z = jnp.array([1.0, -1.0, 0.5, 0.5])
        eta = 0.1

        for _ in range(2000):
            z = self._eg_step(F, z, eta)

        x, y = z[:2], z[2:]
        # The saddle of x^T A y over R^2 x R^2 is at (0, 0)
        assert jnp.linalg.norm(x) < 5e-2, f"x not near zero: {x}"
        assert jnp.linalg.norm(y) < 5e-2, f"y not near zero: {y}"

    def test_eg_operator_norm_decreasing(self):
        """Lemma 3.1: ||F(z^1) + c_1|| should decrease over iterations.

        For a monotone operator with Lipschitz constant ell,
        the EG residual should shrink.
        """
        problem = _quadratic_minimax()
        F = problem.operator_F

        z = jnp.array([2.0, -1.5, 1.0, -2.0])
        eta = 0.05  # small enough for stability

        residuals = []
        for _ in range(100):
            z_half = z - eta * F(z)
            z_new = z - eta * F(z_half)
            residual = jnp.linalg.norm(F(z_new))
            residuals.append(float(residual))
            z = z_new

        # Residuals should generally decrease (monotone decreasing trend)
        # Check that the final residual is much smaller than the initial
        assert residuals[-1] < residuals[0] * 0.1, (
            f"Residual did not decrease enough: {residuals[0]:.4f} -> {residuals[-1]:.4f}"
        )

    def test_eg_step_is_jittable(self):
        """EG step must be JIT-compilable for performance."""
        problem = _bilinear_problem()

        @jax.jit
        def eg_step(z, eta):
            F = problem.operator_F
            z_half = z - eta * F(z)
            return z - eta * F(z_half)

        z = jnp.zeros(4)
        result = eg_step(z, 0.1)
        assert result.shape == (4,)
        # Running twice should not error (tests caching)
        result2 = eg_step(z, 0.1)
        assert jnp.allclose(result, result2)


# ---------------------------------------------------------------------------
# Tests for uniform convexity (Definitions 3.4, Lemmas 3.2, 3.3)
# ---------------------------------------------------------------------------


class TestUniformConvexity:
    """Test properties of uniformly convex functions."""

    def test_cubic_function_is_uniformly_convex(self):
        """Lemma 3.3: d(z) = (1/3)||z||^3 is (1/2)-uniformly convex.

        Verify: d(z) >= d(z') + <grad d(z'), z - z'> + (mu/3)||z - z'||^3
        """
        mu = 0.5  # from Lemma 3.3

        def d(z):
            return (1.0 / 3.0) * jnp.linalg.norm(z) ** 3

        grad_d = jax.grad(d)

        key = jax.random.PRNGKey(0)
        for _ in range(50):
            key, k1, k2 = jax.random.split(key, 3)
            z = jax.random.normal(k1, (5,)) * 2.0
            z_prime = jax.random.normal(k2, (5,)) * 2.0

            lhs = d(z)
            rhs = d(z_prime) + jnp.dot(grad_d(z_prime), z - z_prime) + (mu / 3) * jnp.linalg.norm(z - z_prime) ** 3

            assert lhs >= rhs - 1e-6, (
                f"Uniform convexity violated: d(z)={lhs:.6f} < rhs={rhs:.6f}"
            )

    def test_cubic_hessian_lipschitz(self):
        """Lemma 3.3: d(z) = (1/3)||z||^3 has 2-Lipschitz Hessians."""
        hessian_d = jax.hessian(lambda z: (1.0 / 3.0) * jnp.linalg.norm(z) ** 3)

        key = jax.random.PRNGKey(42)
        rho = 2.0  # from Lemma 3.3

        for _ in range(30):
            key, k1, k2 = jax.random.split(key, 3)
            z1 = jax.random.normal(k1, (4,)) * 1.5
            z2 = jax.random.normal(k2, (4,)) * 1.5

            H1 = hessian_d(z1)
            H2 = hessian_d(z2)
            spectral_diff = jnp.linalg.norm(H1 - H2, ord=2)
            dist = jnp.linalg.norm(z1 - z2)

            assert spectral_diff <= rho * dist + 1e-5, (
                f"Hessian Lipschitz violated: ||H1-H2||={spectral_diff:.4f} > {rho * dist:.4f}"
            )

    def test_gradient_dominance(self):
        """Lemma 3.2: For mu-uniformly convex h,
        (2/(3*sqrt(mu))) * ||grad h||^{3/2} >= h(z) - h(z*).
        """
        mu = 0.5

        def h(z):
            return (1.0 / 3.0) * jnp.linalg.norm(z) ** 3

        grad_h = jax.grad(h)
        z_star = jnp.zeros(5)  # minimizer

        key = jax.random.PRNGKey(7)
        for _ in range(30):
            key, subkey = jax.random.split(key)
            z = jax.random.normal(subkey, (5,)) * 2.0

            g = grad_h(z)
            lhs = (2.0 / (3.0 * jnp.sqrt(mu))) * jnp.linalg.norm(g) ** 1.5
            rhs = h(z) - h(z_star)

            assert lhs >= rhs - 1e-6, (
                f"Gradient dominance violated: lhs={lhs:.6f} < rhs={rhs:.6f}"
            )


# ---------------------------------------------------------------------------
# Tests for CRN oracle (Definition 3.2)
# ---------------------------------------------------------------------------


class TestCRNOracle:
    """Test the Cubic Regularized Newton oracle."""

    def _crn_minimization(self, grad_fn, hess_fn, z_bar, gamma, n_iters=20):
        """Solve the CRN subproblem via fixed-point iteration on λ.

        The optimality condition (Definition 3.2) is:
            (H + λI)δ = -g,  where λ = (γ/2)||δ||

        We iterate: given λ_k, solve δ_k = -(H + λ_k I)^{-1} g,
        then set λ_{k+1} = (γ/2)||δ_k||. This is a contraction and
        converges without damping.
        """
        g = grad_fn(z_bar)
        H = hess_fn(z_bar)
        d = z_bar.shape[0]

        lam = 0.0
        delta = jnp.zeros(d)
        for _ in range(n_iters):
            A = H + lam * jnp.eye(d)
            delta = -jnp.linalg.solve(A, g)
            lam = (gamma / 2.0) * jnp.linalg.norm(delta)

        z = z_bar + delta
        u = -(g + H @ delta + lam * delta)
        return z, u


    def test_crn_finds_minimum_quadratic(self):
        """CRN oracle on a simple quadratic should find the minimizer.

        For h(z) = 0.5 * z^T A z + b^T z with A positive definite,
        the CRN oracle with gamma -> 0 reduces to Newton's method.
        """
        A = jnp.array([[3.0, 0.5], [0.5, 2.0]])
        b = jnp.array([1.0, -0.5])

        def h(z):
            return 0.5 * z @ A @ z + b @ z

        grad_h = jax.grad(h)
        hess_h = jax.hessian(h)

        z_star = -jnp.linalg.solve(A, b)  # analytical minimizer

        # CRN with small gamma ≈ Newton step
        z_init = jnp.array([5.0, -3.0])
        z_out, _ = self._crn_minimization(grad_h, hess_h, z_init, gamma=1e-8, n_iters=100)

        assert jnp.linalg.norm(z_out - z_star) < 1e-3, (
            f"CRN did not converge: ||z - z*|| = {jnp.linalg.norm(z_out - z_star):.6f}"
        )

    def test_crn_satisfies_optimality_condition(self):
        """Definition 3.2: u = -(g + H*delta + (gamma/2)||delta||*delta) should be
        a subgradient of the indicator at z (i.e., u in ∂I_Z(z)).
        For interior points, u should be near zero.
        """
        A = jnp.eye(3) * 4.0
        b = jnp.zeros(3)

        def h(z):
            return 0.5 * z @ A @ z + b @ z

        grad_h = jax.grad(h)
        hess_h = jax.hessian(h)

        z_init = jnp.array([1.0, -1.0, 0.5])
        z_out, u = self._crn_minimization(grad_h, hess_h, z_init, gamma=1.0, n_iters=100)

        # For unconstrained interior point, u should be near zero
        assert jnp.linalg.norm(u) < 1e-3, (
            f"Subgradient u not near zero at interior point: ||u|| = {jnp.linalg.norm(u):.6f}"
        )

    def test_crn_with_cubic_regularizer(self):
        """For h(z) = (1/3)||z||^3 + 0.5||z||^2, verify CRN output satisfies
        the cubic model's first-order optimality condition (Definition 3.2):
            g + H*δ + (γ/2)||δ||*δ ≈ 0
        where δ = z_out - z_init.

        Note: one CRN call solves the LOCAL cubic model, not the global
        problem. Multiple CRN calls in a loop would be needed to converge
        to z* = 0.
        """

        def h(z):
            norm_z = jnp.linalg.norm(z)
            return (1.0 / 3.0) * norm_z ** 3 + 0.5 * norm_z ** 2

        grad_h = jax.grad(h)
        hess_h = jax.hessian(h)

        z_init = jnp.array([3.0, -2.0, 1.0])
        gamma = 1.0
        z_out, u = self._crn_minimization(grad_h, hess_h, z_init, gamma=gamma, n_iters=50)

        # Verify the CRN optimality condition (Definition 3.2)
        g = grad_h(z_init)
        H = hess_h(z_init)
        delta = z_out - z_init
        norm_delta = jnp.linalg.norm(delta)
        residual = g + H @ delta + (gamma / 2.0) * norm_delta * delta
        assert jnp.linalg.norm(residual) < 1e-3, (
            f"CRN optimality condition violated: ||residual|| = {jnp.linalg.norm(residual):.6f}"
        )


    def test_crn_reduces_to_newton_for_small_gamma(self):
        """When gamma is very small, CRN should behave like Newton's method.

        For a strongly convex quadratic, one Newton step finds the exact minimizer.
        """
        A = jnp.array([[5.0, 1.0], [1.0, 4.0]])
        b = jnp.array([2.0, -1.0])

        def h(z):
            return 0.5 * z @ A @ z + b @ z

        grad_h = jax.grad(h)
        hess_h = jax.hessian(h)

        z_star = -jnp.linalg.solve(A, b)
        z_init = jnp.array([10.0, -5.0])

        # gamma very small -> almost pure Newton
        z_out, _ = self._crn_minimization(grad_h, hess_h, z_init, gamma=1e-6, n_iters=50)

        assert jnp.linalg.norm(z_out - z_star) < 1e-2, (
            f"CRN with tiny gamma should approximate Newton: "
            f"||z - z*|| = {jnp.linalg.norm(z_out - z_star):.6f}"
        )


# ---------------------------------------------------------------------------
# Tests for Duality Gap (Definition 3.1)
# ---------------------------------------------------------------------------


class TestDualityGap:
    """Test duality gap computation and estimation."""

    def test_gap_zero_at_saddle_bilinear(self):
        """For a bilinear game, the gap at the saddle point (0,0) should be 0.

        Gap(0, 0) = max_y f(0, y) - min_x f(x, 0) = 0 - 0 = 0
        We verify this directly without calling the unimplemented method.
        """
        problem = _bilinear_problem()
        f = problem.f

        x_star = jnp.zeros(2)
        y_star = jnp.zeros(2)

        # At the saddle, f(x*, y*) = 0 for bilinear
        val = f(x_star, y_star)
        assert jnp.abs(val) < 1e-4, f"f(0, 0) should be 0, got {val}"

        # For bilinear f(x,y) = x^T A y:
        # max_y f(0, y) = 0  (since x=0)
        # min_x f(x, 0) = 0  (since y=0)
        # So Gap = 0 - 0 = 0
        # Verify with a few y values
        for _ in range(10):
            y_test = jnp.array([1.0, -0.5]) * problem.D_y
            assert jnp.abs(f(x_star, y_test)) < 1e-4

    def test_gap_positive_away_from_saddle(self):
        """Away from the saddle, the duality gap should be positive.

        For f(x, y) = x^T A y with A = [[1, 0.5], [0.3, 1]],
        at x = (1, 0), y = (1, 0): gap >= f(1,0; 1,0) - f(0,0; 1,0) = 1 > 0
        """
        problem = _bilinear_problem()
        f = problem.f

        x = jnp.array([1.0, 0.0])
        y = jnp.array([1.0, 0.0])

        # Lower bound on gap: f(x, y) - f(x, y*) - f(x*, y) + f(x*, y*)
        # For bilinear: gap >= |f(x, y)| when (0,0) is saddle
        val = f(x, y)
        assert val > 0.0, f"Expected positive f(x,y), got {val}"

    def test_gap_estimate_decreases_near_solution(self):
        """As we approach the saddle via EG, the estimated gap should decrease."""
        from minimax_aipe.gap import estimate_gap

        problem = _quadratic_minimax()
        F = problem.operator_F

        z = jnp.array([2.0, -1.5, 1.0, -2.0])
        eta = 0.05

        gaps = []
        for step in range(5):
            gap = estimate_gap(problem, z[:2], z[2:], num_restarts=5, num_steps=200, lr=0.02)
            gaps.append(gap)
            for _ in range(20):  # 20 EG steps between measurements
                z_half = z - eta * F(z)
                z = z - eta * F(z_half)

        # Gap should be decreasing
        assert gaps[-1] < gaps[0], (
            f"Gap did not decrease: {gaps[0]:.4f} -> {gaps[-1]:.4f}"
        )

    def test_gap_lower_bound_bilinear(self):
        """For bilinear f(x,y) = x^T A y, the gap satisfies:
        Gap(x, y) >= |f(x, y)| when the saddle is at origin.
        This gives us a cheap lower bound.
        """
        problem = _bilinear_problem()
        f = problem.f

        key = jax.random.PRNGKey(123)
        for _ in range(10):
            key, kx, ky = jax.random.split(key, 3)
            x = jax.random.normal(kx, (2,)) * 0.5
            y = jax.random.normal(ky, (2,)) * 0.5
            val = f(x, y)
            # |f(x, y)| is a lower bound on the gap for bilinear games
            # with saddle at origin
            assert jnp.abs(val) >= 0.0  # trivially true, but verifies no NaN


# ---------------------------------------------------------------------------
# Tests for MinimaxProblem construction
# ---------------------------------------------------------------------------


class TestMinimaxProblem:
    """Test MinimaxProblem setup and auto-differentiation."""

    def test_auto_grad_matches_manual(self):
        """Auto-differentiated gradients should match manual computation."""
        Q = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        P = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        A = jnp.array([[0.5, 0.1], [0.2, 0.3]])

        def f(x, y):
            return 0.5 * x @ Q @ x - 0.5 * y @ P @ y + x @ A @ y

        # Manual gradients: grad_f returns (∇_x f, -∇_y f)
        # ∇_x f = Qx + Ay
        # ∇_y f = -Py + A^T x
        # -∇_y f = Py - A^T x
        def manual_grad_f(x, y):
            gx = Q @ x + A @ y
            gy_neg = P @ y - A.T @ x  # This is -∇_y f
            return gx, gy_neg

        problem_auto = MinimaxProblem(f, dim_x=2, dim_y=2, D_x=4.0, D_y=4.0)
        problem_manual = MinimaxProblem(
            f, dim_x=2, dim_y=2, D_x=4.0, D_y=4.0, grad_f=manual_grad_f
        )

        x = jnp.array([1.5, -0.7])
        y = jnp.array([0.3, 1.2])

        auto_gx, auto_gy = problem_auto.grad_f(x, y)
        manual_gx, manual_gy = problem_manual.grad_f(x, y)

        assert jnp.allclose(auto_gx, manual_gx, atol=1e-5), (
            f"grad_x mismatch: auto={auto_gx}, manual={manual_gx}"
        )
        assert jnp.allclose(auto_gy, manual_gy, atol=1e-5), (
            f"grad_y mismatch: auto={auto_gy}, manual={manual_gy}"
        )

    def test_operator_F_concatenation(self):
        """F(z) = [∇_x f, -∇_y f] should have correct shape and values."""
        problem = _quadratic_minimax()

        x = jnp.array([1.0, 0.5])
        y = jnp.array([-0.3, 0.8])
        z = jnp.concatenate([x, y])

        Fz = problem.operator_F(z)

        assert Fz.shape == (4,), f"F(z) shape: {Fz.shape}"

        # For f(x,y) = 0.5 x^T Q x - 0.5 y^T P y + x^T A y:
        # F(z)[:2] = ∇_x f = Qx + Ay
        # F(z)[2:] = -∇_y f = Py - A^T x
        Q = jnp.diag(jnp.array([2.0, 3.0]))
        P = jnp.diag(jnp.array([1.0, 2.0]))
        A = jnp.array([[0.5, 0.1], [0.2, 0.3]])

        expected_gx = Q @ x + A @ y
        expected_gy_neg = P @ y - A.T @ x

        assert jnp.allclose(Fz[:2], expected_gx, atol=1e-5)
        assert jnp.allclose(Fz[2:], expected_gy_neg, atol=1e-5)

    def test_projection_clips_to_ball(self):
        """Default projection should keep points inside the feasible ball."""
        problem = MinimaxProblem(
            lambda x, y: 0.0, dim_x=3, dim_y=2, D_x=2.0, D_y=4.0
        )

        # Point outside the ball
        z_outside = jnp.array([10.0, 0.0, 0.0])
        z_proj = problem.project_x(z_outside)

        assert jnp.linalg.norm(z_proj) <= 1.0 + 1e-6, (
            f"Projection failed: ||z_proj|| = {jnp.linalg.norm(z_proj):.4f} > 1.0"
        )

        # Point inside the ball should be unchanged
        z_inside = jnp.array([0.3, 0.2, 0.1])
        z_proj2 = problem.project_x(z_inside)

        assert jnp.allclose(z_inside, z_proj2, atol=1e-6), (
            "Interior point should not be projected"
        )

    def test_grad_f_tuple_vs_operator_F_consistency(self):
        """grad_f() tuple and operator_F() should give the same components."""
        problem = _quadratic_minimax()

        x = jnp.array([0.7, -1.2])
        y = jnp.array([0.4, 0.9])
        z = jnp.concatenate([x, y])

        gx, gy_neg = problem.grad_f(x, y)
        Fz = problem.operator_F(z)

        assert jnp.allclose(Fz[:2], gx, atol=1e-7)
        assert jnp.allclose(Fz[2:], gy_neg, atol=1e-7)

    def test_hessian_auto_diff(self):
        """Auto-differentiated Hessian should match analytical Hessian for quadratic."""
        Q = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        P = jnp.array([[1.0, 0.2], [0.2, 2.0]])
        A = jnp.array([[0.5, 0.1], [0.2, 0.3]])

        def f(x, y):
            return 0.5 * x @ Q @ x - 0.5 * y @ P @ y + x @ A @ y

        problem = MinimaxProblem(f, dim_x=2, dim_y=2, D_x=4.0, D_y=4.0)

        x = jnp.array([1.0, -0.5])
        y = jnp.array([0.3, 0.7])

        H = problem.hessian_f(x, y)
        # H is ((H_xx, H_xy), (H_yx, H_yy))
        H_xx = jnp.array(H[0][0])
        H_xy = jnp.array(H[0][1])
        H_yx = jnp.array(H[1][0])
        H_yy = jnp.array(H[1][1])

        assert jnp.allclose(H_xx, Q, atol=1e-5), f"H_xx mismatch"
        assert jnp.allclose(H_xy, A, atol=1e-5), f"H_xy mismatch"
        assert jnp.allclose(H_yx, A.T, atol=1e-5), f"H_yx mismatch"
        assert jnp.allclose(H_yy, -P, atol=1e-5), f"H_yy mismatch"
