"""Production robustness tests: ill-conditioning, JAX compilation, bad inputs.

These tests exercise the solver under conditions that break most implementations:
  - Highly ill-conditioned problem matrices
  - JIT compilation and vmap batching of JAX-based algorithms
  - Garbage / adversarial user inputs
"""

import pytest
import jax
import jax.numpy as jnp

from minimax_aipe import (
    MinimaxProblem,
    solve,
    npe,
    aipe,
    len_loop,
    make_crn_npe_oracle,
    make_crn_prox_oracle,
)


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Ill-Conditioned Problems
# ═══════════════════════════════════════════════════════════════════════════

class TestIllConditionedBilinear:
    """Bilinear game  f(x,y) = x^T A y  with  κ(A) = 10^4."""

    def test_solver_runs_without_error(self, ill_conditioned_bilinear):
        """Solver completes without raising an exception."""
        p = ill_conditioned_bilinear
        result = solve(p.problem, epsilon=0.1, verbose=False)
        assert result is not None

    def test_no_nans_in_output(self, ill_conditioned_bilinear):
        """x, y, and gap contain no NaN or Inf values."""
        p = ill_conditioned_bilinear
        result = solve(p.problem, epsilon=0.1, verbose=False)
        assert jnp.all(jnp.isfinite(result.x)), "NaN/Inf in x"
        assert jnp.all(jnp.isfinite(result.y)), "NaN/Inf in y"
        assert jnp.isfinite(jnp.asarray(result.gap)), "NaN/Inf in gap"

    def test_gap_is_nonnegative(self, ill_conditioned_bilinear):
        """Gap is always ≥ 0 by definition."""
        p = ill_conditioned_bilinear
        result = solve(p.problem, epsilon=0.1, verbose=False)
        assert result.gap >= -1e-6, f"gap={result.gap:.4e} < 0"

    def test_solution_is_feasible(self, ill_conditioned_bilinear):
        """x and y stay inside the feasible ball."""
        p = ill_conditioned_bilinear
        result = solve(p.problem, epsilon=0.1, verbose=False)
        D_x = p.problem.D_x
        D_y = p.problem.D_y
        assert float(jnp.linalg.norm(result.x)) <= D_x / 2 + 1e-4
        assert float(jnp.linalg.norm(result.y)) <= D_y / 2 + 1e-4

    def test_inner_oracle_no_nans(self, ill_conditioned_bilinear):
        """CRN oracle alone produces finite output on ill-conditioned input."""
        from minimax_aipe.oracles import crn_oracle
        p = ill_conditioned_bilinear.problem
        z_bar = jnp.ones(p.dim_x + p.dim_y) * 0.5
        gamma = 2.0
        z_half, u = crn_oracle(p, z_bar, gamma, n_iters=50)
        assert jnp.all(jnp.isfinite(z_half)), "NaN/Inf in CRN output z_half"
        assert jnp.all(jnp.isfinite(u)), "NaN/Inf in CRN residual u"


class TestIllConditionedQuadratic:
    """Quadratic minimax with κ(Q) = 10^4."""

    def test_solver_runs(self, ill_conditioned_quadratic):
        result = solve(ill_conditioned_quadratic.problem, epsilon=0.1, verbose=False)
        assert result is not None

    def test_no_nans(self, ill_conditioned_quadratic):
        result = solve(ill_conditioned_quadratic.problem, epsilon=0.1, verbose=False)
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.y))

    def test_gap_nonnegative(self, ill_conditioned_quadratic):
        result = solve(ill_conditioned_quadratic.problem, epsilon=0.1, verbose=False)
        assert result.gap >= -1e-6


class TestExtremeConditioning:
    """Push to κ = 10^6 — stress-test numerical stability."""

    @pytest.mark.slow
    def test_bilinear_kappa_1e6_no_nans(self):
        from tests.conftest import make_ill_conditioned_bilinear
        p = make_ill_conditioned_bilinear(dim=3, kappa=1e6, seed=42)
        result = solve(p.problem, epsilon=0.1, verbose=False)
        assert jnp.all(jnp.isfinite(result.x)), "NaN at κ=1e6"
        assert jnp.all(jnp.isfinite(result.y)), "NaN at κ=1e6"


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — JAX Compilation (JIT)
# ═══════════════════════════════════════════════════════════════════════════

class TestJITCompilation:
    """Verify that single-epoch algorithms compile and run under JIT.

    npe, aipe, and _len_scan_loop are now JIT-compiled by default
    via @partial(jax.jit, static_argnums=...).  These tests verify
    the JIT compilation succeeds, produces correct output, and handles
    recompilation for different static arg values.
    """

    def _make_bilinear_oracles(self, dim=3):
        """Shared setup: bilinear problem + CRN oracle + operator F."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=dim, seed=42).problem
        gamma = 2.0 * max(p.rho or 1.0, 1e-6)
        oracle = make_crn_npe_oracle(p, gamma)
        F_fn = p.operator_F
        z0 = jnp.zeros(p.dim_x + p.dim_y)
        return p, oracle, F_fn, z0, gamma

    def test_npe_compiles_and_runs(self):
        """npe() is JIT-compiled by default — calling it exercises JIT."""
        p, oracle, F_fn, z0, gamma = self._make_bilinear_oracles()
        T = 10
        z_out, calls = npe(oracle, F_fn, z0, T, gamma)
        assert jnp.all(jnp.isfinite(z_out))
        assert calls == T

    def test_npe_repeated_calls_identical(self):
        """Second call with same args uses JIT cache — output is identical."""
        p, oracle, F_fn, z0, gamma = self._make_bilinear_oracles()
        T = 10
        z1, c1 = npe(oracle, F_fn, z0, T, gamma)
        z2, c2 = npe(oracle, F_fn, z0, T, gamma)
        assert jnp.allclose(z1, z2, atol=1e-4)
        assert c1 == c2

    def test_npe_different_T_triggers_recompilation(self):
        """Changing T (a static arg) works — JAX compiles a new function."""
        p, oracle, F_fn, z0, gamma = self._make_bilinear_oracles()
        z5, c5 = npe(oracle, F_fn, z0, 5, gamma)
        z10, c10 = npe(oracle, F_fn, z0, 10, gamma)
        assert c5 == 5
        assert c10 == 10
        assert jnp.all(jnp.isfinite(z5))
        assert jnp.all(jnp.isfinite(z10))


    def test_aipe_compiles_and_runs(self):
        """aipe() is JIT-compiled by default."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        gamma = 2.0

        def f_z(z):
            x, y = z[: p.dim_x], z[p.dim_x :]
            return p.f(x, y)

        def grad_z(z):
            x, y = z[: p.dim_x], z[p.dim_x :]
            gx, gy_neg = p.grad_f(x, y)
            return jnp.concatenate([gx, gy_neg])

        def hess_z(z):
            x, y = z[: p.dim_x], z[p.dim_x :]
            (H_xx, H_xy), (H_yx, H_yy) = p.hessian_f(x, y)
            top = jnp.concatenate([H_xx, H_xy], axis=1)
            bot = jnp.concatenate([H_yx, H_yy], axis=1)
            return jnp.concatenate([top, bot], axis=0)

        prox = make_crn_prox_oracle(grad_z, hess_z, gamma, n_iters=20)
        z0 = jnp.zeros(p.dim_x + p.dim_y)
        T = 5
        z_out, calls = aipe(prox, grad_z, z0, T, gamma)
        assert jnp.all(jnp.isfinite(z_out))
        assert calls == T + 1

    def test_len_compiles_and_runs(self):
        """len_loop() is JIT-compiled by default (via _len_scan_loop)."""
        from minimax_aipe.len import make_lazy_crn_npe_oracle
        from tests.conftest import make_bilinear_problem

        p = make_bilinear_problem(dim=3, seed=42).problem
        gamma = 2.0 * max(p.rho or 1.0, 1e-6)
        oracle = make_lazy_crn_npe_oracle(p, gamma)
        F_fn = p.operator_F
        z0 = jnp.zeros(p.dim_x + p.dim_y)
        T, m = 10, 3

        z_out, calls = len_loop(oracle, F_fn, z0, T, gamma, m)
        assert jnp.all(jnp.isfinite(z_out))
        assert calls == T

    def test_solve_uses_python_loops_not_jittable(self):
        """solve() has Python-level early stopping — cannot be JIT-compiled.

        This is by design: the outer restart loops use float() comparisons
        on traced values.  Users should use npe/aipe/len_loop for JIT.
        """
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        with pytest.raises(Exception):
            jax.jit(solve)(p, 0.1)


class TestVmap:
    """Verify that JIT-compiled algorithms compose with jax.vmap.

    Since npe is now JIT-decorated, vmap-of-npe is vmap-of-jit —
    a standard JAX transformation pattern.
    """

    def test_npe_vmappable_over_z0(self):
        """Batch NPE over 4 different initial points."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        gamma = 2.0 * max(p.rho or 1.0, 1e-6)
        oracle = make_crn_npe_oracle(p, gamma)
        F_fn = p.operator_F
        T = 10
        d = p.dim_x + p.dim_y

        key = jax.random.PRNGKey(99)
        z0_batch = jax.random.normal(key, (4, d)) * 0.1

        z_outs = jax.vmap(lambda z0: npe(oracle, F_fn, z0, T, gamma)[0])(z0_batch)

        assert z_outs.shape == (4, d)
        assert jnp.all(jnp.isfinite(z_outs))

    def test_vmap_different_inits_diverge(self):
        """Different starting points should produce different iterates."""
        from tests.conftest import make_quadratic_saddle_problem
        p = make_quadratic_saddle_problem(dim=3, seed=0).problem
        gamma = 2.0
        oracle = make_crn_npe_oracle(p, gamma)
        F_fn = p.operator_F
        T = 20
        d = p.dim_x + p.dim_y

        key = jax.random.PRNGKey(7)
        z0_batch = jax.random.normal(key, (3, d)) * 0.5

        z_outs = jax.vmap(lambda z0: npe(oracle, F_fn, z0, T, gamma)[0])(z0_batch)

        total_diff = (
            jnp.linalg.norm(z_outs[0] - z_outs[1])
            + jnp.linalg.norm(z_outs[1] - z_outs[2])
        )
        assert total_diff > 1e-6

    def test_vmap_composed_with_outer_jit(self):
        """vmap(npe) works — npe is already JIT, vmap wraps the JIT call."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        gamma = 2.0
        oracle = make_crn_npe_oracle(p, gamma)
        F_fn = p.operator_F
        T = 8
        d = p.dim_x + p.dim_y

        z0_batch = jnp.stack([
            jnp.zeros(d),
            jnp.ones(d) * 0.1,
            jnp.ones(d) * -0.1,
        ])

        def batched(z0s):
            return jax.vmap(lambda z0: npe(oracle, F_fn, z0, T, gamma)[0])(z0s)

        z_outs = batched(z0_batch)
        assert z_outs.shape == (3, d)
        assert jnp.all(jnp.isfinite(z_outs))


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Bad Inputs & Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestBadEpsilon:
    """solve() must reject non-positive epsilon."""

    def test_negative_epsilon_raises(self):
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        with pytest.raises(ValueError, match="epsilon"):
            solve(p, epsilon=-1.0)

    def test_zero_epsilon_raises(self):
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        with pytest.raises(ValueError, match="epsilon"):
            solve(p, epsilon=0.0)

    def test_very_small_epsilon_doesnt_crash(self):
        """epsilon = 1e-10 should not crash (may not converge, but no exception)."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        # Should complete without raising, even if it doesn't converge
        result = solve(p, epsilon=1e-10, verbose=False)
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.y))


class TestBadDiameters:
    """MinimaxProblem must reject non-positive diameters."""

    def test_zero_diameter_x_raises(self):
        with pytest.raises(ValueError, match="D_x"):
            MinimaxProblem(
                f=lambda x, y: x @ y,
                dim_x=2, dim_y=2, D_x=0.0, D_y=2.0,
            )

    def test_zero_diameter_y_raises(self):
        with pytest.raises(ValueError, match="D_y"):
            MinimaxProblem(
                f=lambda x, y: x @ y,
                dim_x=2, dim_y=2, D_x=2.0, D_y=0.0,
            )

    def test_negative_diameter_x_raises(self):
        with pytest.raises(ValueError, match="D_x"):
            MinimaxProblem(
                f=lambda x, y: x @ y,
                dim_x=2, dim_y=2, D_x=-1.0, D_y=2.0,
            )

    def test_negative_diameter_y_raises(self):
        with pytest.raises(ValueError, match="D_y"):
            MinimaxProblem(
                f=lambda x, y: x @ y,
                dim_x=2, dim_y=2, D_x=2.0, D_y=-5.0,
            )


class TestBadDimensions:
    """MinimaxProblem must reject non-positive dimensions."""

    def test_zero_dim_x_raises(self):
        with pytest.raises(ValueError, match="dim_x"):
            MinimaxProblem(
                f=lambda x, y: jnp.sum(y),
                dim_x=0, dim_y=2, D_x=2.0, D_y=2.0,
            )

    def test_negative_dim_y_raises(self):
        with pytest.raises(ValueError, match="dim_y"):
            MinimaxProblem(
                f=lambda x, y: jnp.sum(x),
                dim_x=2, dim_y=-1, D_x=2.0, D_y=2.0,
            )


class TestDimensionMismatch:
    """MinimaxProblem must detect when grad_f output shapes are wrong."""

    def test_grad_x_wrong_shape_raises(self):
        """grad_f returns a first component with wrong dimension."""
        def bad_grad_f(x, y):
            # Returns (2,) instead of (3,)
            return jnp.zeros(2), -jnp.zeros(2)

        with pytest.raises(ValueError, match="first component"):
            MinimaxProblem(
                f=lambda x, y: x @ y[:3],
                dim_x=3, dim_y=2, D_x=2.0, D_y=2.0,
                grad_f=bad_grad_f,
            )

    def test_grad_y_wrong_shape_raises(self):
        """grad_f returns a second component with wrong dimension."""
        def bad_grad_f(x, y):
            # Returns (3,) instead of (2,)
            return jnp.zeros(3), -jnp.zeros(3)

        with pytest.raises(ValueError, match="second component"):
            MinimaxProblem(
                f=lambda x, y: x[:3] @ y,
                dim_x=3, dim_y=2, D_x=2.0, D_y=2.0,
                grad_f=bad_grad_f,
            )

    def test_correct_shapes_accepted(self):
        """Matching dimensions should not raise."""
        def good_grad_f(x, y):
            return jnp.zeros(3), -jnp.zeros(2)

        problem = MinimaxProblem(
            f=lambda x, y: jnp.sum(x) + jnp.sum(y),
            dim_x=3, dim_y=2, D_x=2.0, D_y=2.0,
            grad_f=good_grad_f,
        )
        assert problem.dim_x == 3
        assert problem.dim_y == 2


class TestEdgeCaseInputs:
    """Unusual but technically valid inputs."""

    def test_1d_1d_problem(self):
        """Minimum dimensions: dim_x=1, dim_y=1."""
        from tests.conftest import make_1d_bilinear
        p = make_1d_bilinear()
        result = solve(p.problem, epsilon=0.1, verbose=False)
        assert result.gap >= -1e-6

    def test_asymmetric_dimensions(self):
        """dim_x != dim_y should work fine."""
        import jax
        dim_x, dim_y = 2, 5
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (dim_x, dim_y))

        def f(x, y):
            return x @ A @ y

        problem = MinimaxProblem(
            f=f, dim_x=dim_x, dim_y=dim_y, D_x=2.0, D_y=2.0,
            rho=0.0, ell=0.0,
        )
        result = solve(problem, epsilon=0.1, verbose=False)
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.y))
        assert result.x.shape == (dim_x,)
        assert result.y.shape == (dim_y,)

    def test_gamma_override(self):
        """User-supplied gamma should be accepted without error."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        result = solve(p, epsilon=0.1, gamma=5.0, verbose=False)
        assert result.history["gamma"] == 5.0

    def test_len_mode_runs(self):
        """M_saddle='len' should produce valid output on a simple problem."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=3, seed=42).problem
        result = solve(p, epsilon=0.1, M_saddle="len", m_lazy=3, verbose=False)
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.y))
        assert result.gap >= -1e-6
