"""Integration tests: deep verification of internal algorithmic mechanisms.

Covers five gaps that unit tests and black-box solver tests miss:

  1. Three-loop interaction on nontrivial problems (tolerance threading,
     parameter scheduling, history completeness)
  2. Oracle-call tracking via pure JAX return-value threading
  3. Gap estimator accuracy (estimate_gap vs known exact gaps)
  4. LEN-vs-NPE equivalence when m=1 (lazy Hessian degenerates to fresh)
  5. Nontrivial convergence (strongly convex-concave, offset saddle, 10D)
"""

import pytest
import jax
import jax.numpy as jnp

from minimax_aipe import (
    MinimaxProblem,
    solve,
    estimate_gap,
    npe,
    make_crn_npe_oracle,
)
from minimax_aipe.framework import _compute_loop_params
from tests._solve_cache import cached_solve

# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Three-Loop Interaction
# ═══════════════════════════════════════════════════════════════════════════


class TestToleranceHierarchy:
    """Verify that zeta_1 > zeta_2 > zeta_3 (inner tolerances get tighter)."""

    def test_zeta_decreasing(self, bilinear_3d):
        p = bilinear_3d.problem
        gamma = 1.0
        params = _compute_loop_params(p, epsilon=0.01, gamma=gamma)
        assert params.zeta_1 > 0, "zeta_1 must be positive"
        assert params.zeta_2 > 0, "zeta_2 must be positive"
        assert params.zeta_3 > 0, "zeta_3 must be positive"
        assert params.zeta_3 < params.zeta_2 < params.zeta_1, (
            f"Hierarchy violated: zeta_1={params.zeta_1:.2e}, "
            f"zeta_2={params.zeta_2:.2e}, zeta_3={params.zeta_3:.2e}"
        )

    def test_tighter_epsilon_tighter_zeta(self, bilinear_3d):
        p = bilinear_3d.problem
        gamma = 1.0
        params_wide = _compute_loop_params(p, epsilon=0.1, gamma=gamma)
        params_tight = _compute_loop_params(p, epsilon=0.01, gamma=gamma)
        assert params_tight.zeta_1 <= params_wide.zeta_1
        assert params_tight.zeta_2 <= params_wide.zeta_2
        assert params_tight.zeta_3 <= params_wide.zeta_3

    def test_hierarchy_holds_across_problem_types(self, quadratic_problem):
        """Hierarchy should hold regardless of problem structure."""
        p = quadratic_problem.problem
        gamma = 2.0
        params = _compute_loop_params(p, epsilon=0.005, gamma=gamma)
        assert params.zeta_3 < params.zeta_2 < params.zeta_1


class TestThreeLoopParameters:
    """Verify that loop iteration counts are reasonable and bounded."""

    def test_loop_params_structure(self, bilinear_3d):
        """_LoopParams contains all expected fields with positive values."""
        p = bilinear_3d.problem
        gamma = 1.0
        params = _compute_loop_params(p, epsilon=0.01, gamma=gamma)
        assert params.T_outer >= 1
        assert params.S_outer >= 1
        assert params.T_middle >= 1
        assert params.S_middle >= 1
        assert params.T_inner >= 1
        assert params.S_inner >= 1
        assert params.m_lazy >= 1

    def test_iteration_counts_bounded(self, bilinear_3d):
        """T values should be capped (max 200), S values capped (max 12)."""
        p = bilinear_3d.problem
        for eps in [0.1, 0.01, 0.001]:
            params = _compute_loop_params(p, epsilon=eps, gamma=1.0)
            assert params.T_outer <= 200, f"T_outer={params.T_outer} at ε={eps}"
            assert params.T_middle <= 200
            assert params.T_inner <= 200
            assert params.S_outer <= 12, f"S_outer={params.S_outer} at ε={eps}"
            assert params.S_middle <= 12
            assert params.S_inner <= 12
    def test_history_contains_all_loop_params(self, bilinear_3d):
        """solve() history dict records every scheduling parameter."""
        result = cached_solve(bilinear_3d, epsilon=0.05)
        required_keys = [
            "gamma", "mu_x", "mu_y",
            "zeta_1", "zeta_2", "zeta_3",
            "T_outer", "S_outer",
            "T_middle", "S_middle",
            "T_inner", "S_inner",
            "M_saddle",
        ]
        for key in required_keys:
            assert key in result.history, f"Missing '{key}' in solver history"


class TestThreeLoopOnHarderProblems:
    """Run the full pipeline on problems where inner-loop accuracy matters."""

    def test_offset_quadratic_solved(self, offset_quadratic):
        """Nonzero saddle point: solver should still converge."""
        p = offset_quadratic
        result = cached_solve(p, epsilon=0.05, verbose=False)
        assert result.gap >= -1e-6
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.y))

    def test_offset_quadratic_close_to_saddle(self, offset_quadratic):
        """Solution should be near the known nonzero saddle point."""
        p = offset_quadratic
        result = cached_solve(p, epsilon=0.05, verbose=False)
        err_x = float(jnp.linalg.norm(result.x - p.x_star))
        err_y = float(jnp.linalg.norm(result.y - p.y_star))
        assert err_x < 1.0, f"x error {err_x:.4e} too large"
        assert err_y < 1.0, f"y error {err_y:.4e} too large"

    def test_10d_quadratic_converges(self, large_quadratic_10d):
        """10D quadratic: solver must produce finite output."""
        p = large_quadratic_10d.problem
        result = cached_solve(large_quadratic_10d, epsilon=0.1, verbose=False)
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.y))
        assert result.x.shape == (5,)
        assert result.y.shape == (5,)

    @pytest.mark.slow
    def test_10d_quadratic_gap_small(self, large_quadratic_10d):
        """10D quadratic: achieved gap should be bounded."""
        p = large_quadratic_10d.problem
        result = cached_solve(large_quadratic_10d, epsilon=0.05, verbose=False)
        assert result.gap < 1.0, f"gap={result.gap:.4e} too large for 10D quad"

    def test_len_mode_on_harder_problem(self, offset_quadratic):
        """M_saddle='len' on a nontrivial problem should produce valid output."""
        p = offset_quadratic.problem
        result = cached_solve(offset_quadratic, epsilon=0.1, M_saddle="len", m_lazy=3, verbose=False)
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.y))
        assert result.gap >= -1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Oracle-call tracking (pure JAX return-value threading)
# ═══════════════════════════════════════════════════════════════════════════


class TestOracleCallTracking:
    """Verify that inner oracle calls are tracked via return values."""

    def test_solve_reports_positive_oracle_calls(self, bilinear_3d):
        """The solver should report a positive oracle_calls count."""
        p = bilinear_3d
        result = cached_solve(p, epsilon=0.5)
        assert result.oracle_calls > 0, (
            f"oracle_calls should be positive, got {result.oracle_calls}"
        )

    def test_oracle_calls_is_integer(self, bilinear_3d):
        """oracle_calls should be a plain Python int, not a JAX array."""
        p = bilinear_3d
        result = cached_solve(p, epsilon=0.5)
        assert isinstance(result.oracle_calls, int)


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Gap Estimator Accuracy
# ═══════════════════════════════════════════════════════════════════════════


class TestGapEstimatorAtKnownSolutions:
    """estimate_gap should return ≈ 0 at the saddle point."""

    def test_gap_at_saddle_bilinear(self, bilinear_3d):
        p = bilinear_3d
        gap = estimate_gap(
            p.problem, p.x_star, p.y_star,
            num_restarts=20, num_steps=500,
        )
        assert gap < 0.15, f"gap={gap:.4e} at known saddle (bilinear)"

    def test_gap_at_saddle_quadratic(self, quadratic_3d):
        p = quadratic_3d
        gap = estimate_gap(
            p.problem, p.x_star, p.y_star,
            num_restarts=20, num_steps=500,
        )
        assert gap < 0.2, f"gap={gap:.4e} at known saddle (quadratic)"

    def test_gap_at_saddle_1d(self, problem_1d):
        p = problem_1d
        gap = estimate_gap(
            p.problem, p.x_star, p.y_star,
            num_restarts=20, num_steps=500,
        )
        assert gap < 0.1, f"gap={gap:.4e} at known saddle (1D)"

    def test_gap_at_saddle_offset(self, offset_quadratic):
        p = offset_quadratic
        gap = estimate_gap(
            p.problem, p.x_star, p.y_star,
            num_restarts=20, num_steps=500,
        )
        assert gap < 0.15, f"gap={gap:.4e} at known saddle (offset)"


class TestGapEstimatorAtNonSaddles:
    """estimate_gap should return a positive gap away from the saddle."""

    def test_gap_positive_at_non_saddle_1d(self):
        """f(x,y)=xy on [-1,1]² at (0.5, 0.5): gap should be ≈ 1.0."""
        from tests.conftest import make_1d_bilinear
        p = make_1d_bilinear()
        x_far = jnp.array([0.5])
        y_far = jnp.array([0.5])
        gap = estimate_gap(
            p.problem, x_far, y_far,
            num_restarts=20, num_steps=500,
        )
        assert gap > 0.3, (
            f"Expected gap > 0.3 at non-saddle, got {gap:.4e}"
        )

    def test_gap_positive_for_bilinear(self):
        """Bilinear at a non-zero point should have positive gap."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42)
        x_far = jnp.ones(3) * 0.3
        y_far = jnp.ones(3) * 0.3
        gap = estimate_gap(
            p.problem, x_far, y_far,
            num_restarts=20, num_steps=500,
        )
        assert gap > 0.0, f"Expected positive gap at non-saddle, got {gap:.4e}"


class TestGapEstimatorConsistency:
    """estimate_gap should agree with other gap computation methods."""

    def test_agrees_with_grid_gap_at_saddle(self, bilinear_3d):
        """Both methods should give small gap at the saddle."""
        from tests.conftest import grid_gap
        p = bilinear_3d
        gap_est = estimate_gap(
            p.problem, p.x_star, p.y_star,
            num_restarts=20, num_steps=500,
        )
        gap_grid = grid_gap(
            p.problem, p.x_star, p.y_star, n_grid=200,
        )
        # Both should be small (close to zero)
        assert gap_est < 0.2
        assert gap_grid < 0.2
        # They should roughly agree
        assert abs(gap_est - gap_grid) < 0.3

    def test_agrees_with_grid_gap_off_saddle(self):
        """At a non-saddle 1D point, both methods should give similar values."""
        from tests.conftest import grid_gap, make_1d_bilinear
        p = make_1d_bilinear()
        x = jnp.array([0.5])
        y = jnp.array([0.5])
        gap_est = estimate_gap(
            p.problem, x, y,
            num_restarts=30, num_steps=1000,
        )
        gap_grid = grid_gap(p.problem, x, y, n_grid=200)
        assert abs(gap_est - gap_grid) < 0.5, (
            f"estimate_gap={gap_est:.4f} vs grid_gap={gap_grid:.4f}"
        )


class TestGapEstimatorMonotonicity:
    """Farther from the saddle → larger estimated gap."""

    def test_gap_increases_with_distance_1d(self):
        from tests.conftest import make_1d_bilinear
        p = make_1d_bilinear()
        gap_origin = estimate_gap(
            p.problem, jnp.array([0.0]), jnp.array([0.0]),
            num_restarts=20, num_steps=500,
        )
        gap_mid = estimate_gap(
            p.problem, jnp.array([0.3]), jnp.array([0.3]),
            num_restarts=20, num_steps=500,
        )
        gap_far = estimate_gap(
            p.problem, jnp.array([0.8]), jnp.array([0.8]),
            num_restarts=20, num_steps=500,
        )
        assert gap_mid >= gap_origin - 0.05, (
            f"gap_mid={gap_mid:.4f} < gap_origin={gap_origin:.4f}"
        )
        assert gap_far >= gap_mid - 0.05, (
            f"gap_far={gap_far:.4f} < gap_mid={gap_mid:.4f}"
        )

    def test_gap_at_solver_output_is_small(self, bilinear_3d):
        """The solver's own gap estimate should be small for a zero-gap problem."""
        result = cached_solve(bilinear_3d, epsilon=0.05, verbose=False)
        assert result.gap < 0.5, (
            f"Solver's own gap={result.gap:.4e} too large for zero-gap problem"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — LEN-vs-NPE Equivalence
# ═══════════════════════════════════════════════════════════════════════════


class TestLENvsNPEEquivalence:
    """With m=1, LEN must produce nearly identical output to NPE.

    When the Hessian is refreshed every iteration (m=1), LEN's lazy-CRN
    oracle evaluates ∇F at the current iterate — identical to NPE's fresh
    CRN oracle.  The only remaining differences are numerical guards:
    NPE uses ``tiny=1e-15`` as the distance floor; LEN uses
    ``eta_floor=1e-8``.  For well-conditioned problems where
    ‖z_t − z_{t+1/2}‖ ≫ 1e-8, both compute the same step size.

    For quadratics (constant Hessian), the equivalence holds for any m.
    """

    @staticmethod
    def _run_npe_and_len(problem, z0, T, gamma):
        """Helper: run both algorithms and return iterates."""
        from minimax_aipe.len import make_lazy_crn_npe_oracle, len_loop

        npe_oracle = make_crn_npe_oracle(problem, gamma)
        len_oracle = make_lazy_crn_npe_oracle(problem, gamma)
        F_fn = problem.operator_F

        z_npe, c_npe = npe(npe_oracle, F_fn, z0, T, gamma)
        z_len, c_len = len_loop(len_oracle, F_fn, z0, T, gamma, m=1)

        return z_npe, c_npe, z_len, c_len

    def test_m1_quadratic(self, quadratic_3d):
        """Quadratic: m=1 LEN should closely match NPE."""
        p = quadratic_3d.problem
        gamma = 2.0 * max(p.rho or 1.0, 1e-6)
        z0 = jnp.array([0.3, -0.1, 0.2, 0.4, -0.3, 0.1])
        T = 20

        z_npe, _, z_len, _ = self._run_npe_and_len(p, z0, T, gamma)

        diff = float(jnp.linalg.norm(z_npe - z_len))
        assert diff < 1e-4, (
            f"NPE and LEN(m=1) diverged on quadratic: "
            f"‖Δz‖={diff:.4e}"
        )

    def test_m1_bilinear(self, bilinear_3d):
        """Bilinear: m=1 LEN should closely match NPE."""
        p = bilinear_3d.problem
        gamma = 2.0 * max(p.rho or 1.0, 1e-6)
        z0 = jnp.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
        T = 15

        z_npe, _, z_len, _ = self._run_npe_and_len(p, z0, T, gamma)

        diff = float(jnp.linalg.norm(z_npe - z_len))
        assert diff < 1e-4, (
            f"NPE and LEN(m=1) diverged on bilinear: ‖Δz‖={diff:.4e}"
        )

    def test_m1_same_oracle_calls(self, quadratic_3d):
        """Both should report exactly T oracle calls."""
        p = quadratic_3d.problem
        gamma = 2.0
        z0 = jnp.zeros(p.dim_x + p.dim_y)
        T = 10

        _, c_npe, _, c_len = self._run_npe_and_len(p, z0, T, gamma)
        assert c_npe == T, f"NPE calls={c_npe}, expected {T}"
        assert c_len == T, f"LEN calls={c_len}, expected {T}"

    def test_m1_both_converge_to_saddle(self, quadratic_3d):
        """With enough iterations, both reach the neighbourhood of z*."""
        p = quadratic_3d.problem
        gamma = 2.0
        z0 = jnp.array([0.5, -0.3, 0.2, -0.4, 0.1, 0.3])
        T = 50

        z_npe, _, z_len, _ = self._run_npe_and_len(p, z0, T, gamma)

        # Both should be near the saddle (z*=0)
        assert float(jnp.linalg.norm(z_npe)) < 1.0
        assert float(jnp.linalg.norm(z_len)) < 1.0

    def test_quadratic_constant_hessian_any_m(self, quadratic_3d):
        """For quadratics, LEN with any m should match NPE.

        The Hessian of a quadratic is constant (doesn't depend on z),
        so lazy vs. fresh evaluation makes no difference.
        """
        from minimax_aipe.len import make_lazy_crn_npe_oracle, len_loop

        p = quadratic_3d.problem
        gamma = 2.0
        z0 = jnp.array([0.3, -0.2, 0.1, 0.4, -0.1, 0.2])
        T = 15

        npe_oracle = make_crn_npe_oracle(p, gamma)
        z_npe, _ = npe(npe_oracle, p.operator_F, z0, T, gamma)

        for m in [1, 3, 5]:
            len_oracle = make_lazy_crn_npe_oracle(p, gamma)
            z_len, _ = len_loop(len_oracle, p.operator_F, z0, T, gamma, m=m)
            diff = float(jnp.linalg.norm(z_npe - z_len))
            assert diff < 1e-3, (
                f"LEN(m={m}) diverged from NPE on constant-Hessian "
                f"quadratic: ‖Δz‖={diff:.4e}"
            )

    def test_bilinear_constant_hessian_any_m(self, bilinear_3d):
        """For bilinear problems, LEN with any m should also match NPE."""
        from minimax_aipe.len import make_lazy_crn_npe_oracle, len_loop

        p = bilinear_3d.problem
        gamma = 2.0
        z0 = jnp.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
        T = 15

        npe_oracle = make_crn_npe_oracle(p, gamma)
        z_npe, _ = npe(npe_oracle, p.operator_F, z0, T, gamma)

        for m in [1, 3, 5]:
            len_oracle = make_lazy_crn_npe_oracle(p, gamma)
            z_len, _ = len_loop(len_oracle, p.operator_F, z0, T, gamma, m=m)
            diff = float(jnp.linalg.norm(z_npe - z_len))
            assert diff < 1e-3, (
                f"LEN(m={m}) diverged from NPE on constant-Hessian "
                f"bilinear: ‖Δz‖={diff:.4e}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 — Nontrivial Convergence
# ═══════════════════════════════════════════════════════════════════════════


class TestStronglyConvexConcave:
    """Problems with nontrivial curvature that exercise the full pipeline."""

    def test_1d_strongly_convex_concave(self):
        """f(x,y) = ½(5x² - 3y²) + 2xy  has saddle at origin with gap=0.

        This problem has ℓ > 0 and a nontrivial curvature structure
        that exercises the full three-loop pipeline.
        """
        def f(x, y):
            return 0.5 * (5.0 * x[0] ** 2 - 3.0 * y[0] ** 2) + 2.0 * x[0] * y[0]

        problem = MinimaxProblem(
            f=f, dim_x=1, dim_y=1, D_x=4.0, D_y=4.0, rho=0.0, ell=5.0,
        )
        result = solve(problem, epsilon=0.01, verbose=False)
        assert result.gap < 0.1, f"gap={result.gap:.4e} too large for easy quadratic"
        assert jnp.abs(result.x[0]) < 0.5
        assert jnp.abs(result.y[0]) < 0.5

    def test_offset_quadratic_convergence(self, offset_quadratic):
        """Nonzero saddle: verify convergence in both coordinates."""
        p = offset_quadratic
        result = cached_solve(p, epsilon=0.05, verbose=False)
        err_x = float(jnp.linalg.norm(result.x - p.x_star))
        err_y = float(jnp.linalg.norm(result.y - p.y_star))
        assert err_x < 0.5, f"x error {err_x:.4e}"
        assert err_y < 0.5, f"y error {err_y:.4e}"

    def test_offset_gap_at_output(self, offset_quadratic):
        """Gap at solver output should be small for a zero-gap problem."""
        p = offset_quadratic
        result = cached_solve(p, epsilon=0.05, verbose=False)
        # Use estimate_gap for independent verification
        gap_check = estimate_gap(
            p.problem, result.x, result.y,
            num_restarts=20, num_steps=500,
        )
        assert gap_check < 0.3, (
            f"Independent gap check={gap_check:.4e} at solver output"
        )

    def test_10d_both_solvers(self, large_quadratic_10d):
        """10D problem should work with both NPE and LEN inner solvers."""
        p = large_quadratic_10d.problem
        for mode in ["npe", "len"]:
            result = cached_solve(
                large_quadratic_10d, epsilon=0.1, M_saddle=mode,
                m_lazy=3, verbose=False,
            )
            assert jnp.all(jnp.isfinite(result.x)), f"NaN in x ({mode})"
            assert jnp.all(jnp.isfinite(result.y)), f"NaN in y ({mode})"
            assert result.gap >= -1e-6, f"Negative gap ({mode}): {result.gap}"


class TestGapDecreasesWithTighterEpsilon:
    """Across a sequence of ε values, the achieved gap should decrease."""

    def test_bilinear_gap_sequence(self, bilinear_3d):
        p = bilinear_3d.problem
        epsilons = [0.1, 0.05, 0.02]
        gaps = []
        for eps in epsilons:
            result = cached_solve(bilinear_3d, epsilon=eps, verbose=False)
            gaps.append(result.gap)

        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-3, (
                f"Gap increased: {list(zip(epsilons, gaps))}"
            )

    def test_quadratic_gap_sequence(self, quadratic_3d):
        p = quadratic_3d.problem
        epsilons = [0.1, 0.05, 0.02]
        gaps = []
        for eps in epsilons:
            result = cached_solve(quadratic_3d, epsilon=eps, verbose=False)
            gaps.append(result.gap)

        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-3


class TestDirectNPEStronglyConvex:
    """Test NPE algorithm directly on strongly convex problems.

    Bypasses the triple-loop to verify that the core NPE loop
    converges when the problem has real curvature.
    """

    def test_npe_converges_on_strongly_convex(self):
        """NPE should drive ‖F(z)‖ to near zero on a well-conditioned problem."""
        def f(x, y):
            return 0.5 * (5.0 * x[0] ** 2 - 3.0 * y[0] ** 2) + 2.0 * x[0] * y[0]

        problem = MinimaxProblem(
            f=f, dim_x=1, dim_y=1, D_x=4.0, D_y=4.0, rho=0.0, ell=5.0,
        )
        gamma = 2.0
        oracle = make_crn_npe_oracle(problem, gamma)
        z0 = jnp.array([0.5, -0.3])
        T, S = 20, 3

        z = z0
        for _ in range(S):
            z, _ = npe(oracle, problem.operator_F, z, T, gamma)

        F_norm = float(jnp.linalg.norm(problem.operator_F(z)))
        assert F_norm < 0.5, f"‖F(z)‖={F_norm:.4e} after {S} NPE epochs"

    def test_npe_converges_on_10d(self, large_quadratic_10d):
        """NPE on 10D: ‖F(z)‖ should decrease over epochs."""
        p = large_quadratic_10d.problem
        gamma = 2.0
        oracle = make_crn_npe_oracle(p, gamma)
        z0 = jnp.ones(p.dim_x + p.dim_y) * 0.25
        T = 20

        F_norms = []
        z = z0
        for _ in range(3):
            F_norms.append(float(jnp.linalg.norm(p.operator_F(z))))
            z, _ = npe(oracle, p.operator_F, z, T, gamma)

        # The last ‖F(z)‖ should be smaller than the first
        assert F_norms[-1] < F_norms[0], (
            f"‖F(z)‖ did not decrease: {F_norms}"
        )
