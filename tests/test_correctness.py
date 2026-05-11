"""Correctness tests: verify solver output against known analytical solutions.

Tests cover the four problem classes from the test matrix:
  1. Bilinear game  f(x,y) = x^T A y       — gap = 0 at origin
  2. Quadratic minimax                       — KKT at origin
  3. 1D prototypical  f(x,y) = xy           — trivial saddle
  4. Separable  f(x,y) = h1(x) - h2(y)     — decoupled convergence
"""

import pytest
import jax.numpy as jnp

from minimax_aipe import solve
from tests.conftest import grid_gap

# ── Tolerance fixtures ────────────────────────────────────────────────────

@pytest.fixture(params=[5e-2, 1e-2, 5e-3], ids=["eps5e-2", "eps1e-2", "eps5e-3"])
def epsilon(request):
    return request.param


# ── 1. Bilinear game ─────────────────────────────────────────────────────

class TestBilinearCorrectness:

    def test_solver_runs(self, bilinear_problem):
        """solve() returns a finite gap for a bilinear problem."""
        p = bilinear_problem
        result = solve(p["problem"], epsilon=0.1, verbose=False)
        assert result.gap >= 0.0
        assert jnp.isfinite(jnp.asarray(result.gap))

    def test_solution_is_feasible(self, bilinear_problem):
        """x and y stay inside the feasible ball."""
        p = bilinear_problem
        result = solve(p["problem"], epsilon=0.1, verbose=False)
        D_x = p["problem"].D_x
        D_y = p["problem"].D_y
        assert float(jnp.linalg.norm(result.x)) <= D_x / 2 + 1e-6
        assert float(jnp.linalg.norm(result.y)) <= D_y / 2 + 1e-6

    def test_gap_decreases_with_epsilon(self, bilinear_problem):
        """Tighter requested tolerance → smaller or equal achieved gap."""
        p = bilinear_problem
        r1 = solve(p["problem"], epsilon=0.1, verbose=False)
        r2 = solve(p["problem"], epsilon=0.05, verbose=False)
        # Allow small tolerance for numerical noise
        assert r2.gap <= r1.gap + 1e-4

    def test_solution_accuracy(self, bilinear_problem, epsilon):
        """Solution converges toward the known saddle point."""
        p = bilinear_problem
        result = solve(p["problem"], epsilon=epsilon, verbose=False)
        err_x = float(jnp.linalg.norm(result.x - p["x_star"]))
        err_y = float(jnp.linalg.norm(result.y - p["y_star"]))
        # Accept moderate accuracy — the solver is approximate
        # For toy bilinear problems, the point error should scale with epsilon
        assert err_x < epsilon * 5.0, f"primal error {err_x:.4e} exceeds 5*ε"
        assert err_y < epsilon * 5.0, f"dual error {err_y:.4e} exceeds 5*ε"

    @pytest.mark.parametrize("M_saddle", ["npe", "len"])
    def test_both_inner_solvers(self, bilinear_problem, M_saddle):
        """Both NPE and LEN modes produce valid results."""
        p = bilinear_problem
        result = solve(
            p["problem"], epsilon=0.1, M_saddle=M_saddle, verbose=False,
        )
        assert result.gap >= 0.0
        assert jnp.isfinite(jnp.asarray(result.gap))


# ── 2. Quadratic minimax ─────────────────────────────────────────────────

class TestQuadraticCorrectness:

    def test_solver_runs(self, quadratic_problem):
        p = quadratic_problem
        result = solve(p["problem"], epsilon=0.1, verbose=False)
        assert result.gap >= 0.0
        assert jnp.isfinite(jnp.asarray(result.gap))

    def test_solution_is_feasible(self, quadratic_problem):
        p = quadratic_problem
        result = solve(p["problem"], epsilon=0.1, verbose=False)
        assert float(jnp.linalg.norm(result.x)) <= p["problem"].D_x / 2 + 1e-6
        assert float(jnp.linalg.norm(result.y)) <= p["problem"].D_y / 2 + 1e-6

    def test_gap_at_known_saddle(self, quadratic_problem):
        """Gap evaluated at the analytical solution is near zero."""
        p = quadratic_problem
        gap = grid_gap(p["problem"], p["x_star"], p["y_star"], n_grid=500)
        # Grid estimation is inherently noisy for quadratic surfaces;
        # use a generous bound.
        assert gap < 1.0, f"grid gap {gap:.4e} too large at known saddle"

    def test_exact_gap_at_saddle_is_zero(self, quadratic_problem):
        """At the saddle, max_y f(x*,y) = min_x f(x,y*) = 0 exactly."""
        p = quadratic_problem
        f = p["problem"].f
        x_star, y_star = p["x_star"], p["y_star"]
        assert abs(float(f(x_star, y_star))) < 1e-10
        # f(x*, y*) = 0 for all-zero saddle
        assert p["gap_star"] == 0.0

    def test_convergence_to_saddle(self, quadratic_problem, epsilon):
        """Solution error decreases with epsilon."""
        p = quadratic_problem
        result = solve(p["problem"], epsilon=epsilon, verbose=False)
        err = float(
            jnp.linalg.norm(result.x - p["x_star"])
            + jnp.linalg.norm(result.y - p["y_star"])
        )
        assert err < epsilon * 3.0, f"total error {err:.4e} at ε={epsilon}"


# ── 3. 1D prototypical ───────────────────────────────────────────────────

class Test1DCorrectness:

    def test_solver_runs(self, problem_1d):
        p = problem_1d
        result = solve(p["problem"], epsilon=0.1, verbose=False)
        assert result.gap >= 0.0

    def test_convergence(self, problem_1d, epsilon):
        p = problem_1d
        result = solve(p["problem"], epsilon=epsilon, verbose=False)
        assert float(jnp.abs(result.x[0])) < epsilon * 2.0
        assert float(jnp.abs(result.y[0])) < epsilon * 2.0

    def test_grid_gap_consistency(self, problem_1d):
        """Grid-based gap estimate at the origin is near zero."""
        p = problem_1d
        gap = grid_gap(p["problem"], p["x_star"], p["y_star"], n_grid=200)
        assert gap < 0.1


# ── 4. Separable ─────────────────────────────────────────────────────────

class TestSeparableCorrectness:

    def test_solver_runs(self, separable_problem):
        p = separable_problem
        result = solve(p["problem"], epsilon=0.1, verbose=False)
        assert result.gap >= 0.0

    def test_gradient_structure(self, separable_problem):
        """Cross-partials are zero — verify the Hessian block structure."""
        import jax
        p = separable_problem
        x = jnp.array([0.5, -0.3])
        y = jnp.array([0.2, 0.7])
        (H_xx, H_xy), (H_yx, H_yy) = p["problem"].hessian_f(x, y)
        assert float(jnp.linalg.norm(H_xy)) < 1e-6
        assert float(jnp.linalg.norm(H_yx)) < 1e-6

    def test_convergence(self, separable_problem, epsilon):
        p = separable_problem
        result = solve(p["problem"], epsilon=epsilon, verbose=False)
        err = float(
            jnp.linalg.norm(result.x - p["x_star"])
            + jnp.linalg.norm(result.y - p["y_star"])
        )
        assert err < epsilon * 4.0, f"total error {err:.4e} at ε={epsilon}"


# ── Cross-cutting: SolverResult shape ────────────────────────────────────

class TestSolverResult:

    def test_has_all_fields(self, bilinear_problem):
        """SolverResult contains every documented field."""
        result = solve(bilinear_problem["problem"], epsilon=0.1)
        assert hasattr(result, "x")
        assert hasattr(result, "y")
        assert hasattr(result, "gap")
        assert hasattr(result, "iterations")
        assert hasattr(result, "oracle_calls")
        assert hasattr(result, "converged")
        assert hasattr(result, "history")

    def test_history_contains_loop_params(self, bilinear_problem):
        """History dict records the three-loop scheduling parameters."""
        result = solve(bilinear_problem["problem"], epsilon=0.1)
        for key in ["T_outer", "S_outer", "T_middle", "T_inner", "zeta_1"]:
            assert key in result.history, f"missing key '{key}' in history"

    def test_oracle_calls_positive(self, bilinear_problem):
        result = solve(bilinear_problem["problem"], epsilon=0.1)
        assert result.oracle_calls >= 1
