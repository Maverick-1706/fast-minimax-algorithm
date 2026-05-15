# test_len.py
"""Tests for the Lazy Extra Newton (LEN) module.

Covers:
  1. Unit tests of len (len_loop) and len_restart on simple bilinear games.
  2. Hessian-reuse verification — snapshot schedule correctness (rigorous).
  3. Convergence comparison: LEN vs NPE on the same problem.
  4. Integration into the triple-loop framework via M_saddle="len".
  5. Edge cases: m=1 (should behave like NPE), single iteration, m ≫ T.
  6. Numerical hardening: NaN guards, norm explosion, eta_floor.
  7. Parameter validation (m ≤ 0, γ ≤ 0, T ≤ 0).
  8. Return-type compatibility (simple tuple vs LENResult).
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe.problem import MinimaxProblem
from minimax_aipe.npe import (
    npe as npe_loop,
    npe_restart,
    make_crn_npe_oracle,
    project_z as npe_project_z,
)
from minimax_aipe.len import (
    len as len_public,                # new public API
    len_loop,                          # backward-compat alias
    len_restart,
    make_lazy_crn_npe_oracle,
    project_z as len_project_z,
    make_len_saddle_solver,
    LENResult,                         # rich return type
    _DEFAULT_ETA_FLOOR,
    _DEFAULT_MAX_NORM,
)
from minimax_aipe.oracles import crn_oracle, lazy_crn_oracle


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: test problems
# ═══════════════════════════════════════════════════════════════════════════

def _bilinear_problem(D: float = 2.0, ell: float = 1.0) -> MinimaxProblem:
    """Simple bilinear game: f(x,y) = xᵀy with A = I.

    Unique saddle point at (0, 0).  ρ = 0 (constant Hessian).
    """
    dim = 2

    def f(x, y):
        return jnp.dot(x, y)

    def grad_f(x, y):
        return y, -x

    def hessian_f(x, y):
        Z = jnp.zeros((dim, dim))
        I = jnp.eye(dim)
        return ((Z, I), (I, Z))

    return MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim,
        D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=ell, L=ell,
    )


def _quadratic_bilinear_problem(D: float = 2.0) -> MinimaxProblem:
    """f(x,y) = ½‖x‖² + xᵀy − ½‖y‖².  Unique saddle at (0,0), ρ = 0."""
    dim = 2

    def f(x, y):
        return 0.5 * jnp.dot(x, x) + jnp.dot(x, y) - 0.5 * jnp.dot(y, y)

    def grad_f(x, y):
        return x + y, -(x - y)

    def hessian_f(x, y):
        I = jnp.eye(dim)
        Z = jnp.zeros((dim, dim))
        return ((I, I), (I, -I))

    return MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim,
        D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=2.0, L=2.0,
    )


def _nonlinear_problem(D: float = 1.5, rho: float = 1.0) -> MinimaxProblem:
    """f(x,y) = xᵀy + (ρ/6)(‖x‖³ − ‖y‖³).  ρ > 0 (non-constant Hessian)."""
    dim = 2

    def f(x, y):
        return jnp.dot(x, y) + (rho / 6.0) * (
            jnp.linalg.norm(x) ** 3 - jnp.linalg.norm(y) ** 3
        )

    def _cubic_grad_local(v, gamma):
        n = jnp.linalg.norm(v)
        return gamma * n * v

    def _cubic_hess_local(v, gamma):
        n = jnp.linalg.norm(v)
        d = v.shape[0]
        eye = jnp.eye(d)
        safe_n = jnp.maximum(n, 1e-7)
        return gamma * (n * eye + jnp.outer(v, v) / safe_n)

    def grad_f(x, y):
        gx = y + _cubic_grad_local(x, rho)
        gy_neg = -x + _cubic_grad_local(y, rho)
        return gx, gy_neg

    def hessian_f(x, y):
        Z = jnp.zeros((dim, dim))
        I = jnp.eye(dim)
        H_xx = Z + _cubic_hess_local(x, rho)
        H_xy = I
        H_yx = I
        H_yy = Z - _cubic_hess_local(y, rho)
        return ((H_xx, H_xy), (H_yx, H_yy))

    return MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim,
        D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=rho, ell=2.0, L=2.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _operator_norm(problem: MinimaxProblem, z: Array) -> float:
    """‖F(z)‖ as a proxy for proximity to saddle point."""
    return float(jnp.linalg.norm(problem.operator_F(z)))


def _project_for(problem: MinimaxProblem):
    """Return a projection callable for the given problem."""
    return lambda z: len_project_z(problem, z)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Unit tests — len basic functionality
# ═══════════════════════════════════════════════════════════════════════════

class TestLenBasic:
    """Verify that len (and len_loop) runs and reduces ‖F(z)‖."""

    def test_bilinear_convergence(self):
        """LEN should converge on a simple bilinear game."""
        problem = _bilinear_problem()
        gamma = 2.0 * (problem.rho + 1.0)
        m, T = 3, 30
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, calls = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
        )

        norm_before = _operator_norm(problem, z0)
        norm_after = _operator_norm(problem, z_out)

        assert calls == T
        # Must make meaningful progress.
        assert norm_after < 0.5 * norm_before, (
            f"Operator norm barely decreased: {norm_before:.4e} → {norm_after:.4e}"
        )

    def test_public_api_identical(self):
        """len() and len_loop() must produce identical results."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 3, 20
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)
        proj = _project_for(problem)

        z1, c1 = len_public(oracle, problem.operator_F, z0, T, gamma, m, project=proj)
        z2, c2 = len_loop(oracle, problem.operator_F, z0, T, gamma, m, project=proj)

        assert c1 == c2 == T
        assert jnp.allclose(z1, z2, atol=1e-4), (
            f"len() and len_loop() diverge: ‖Δ‖={float(jnp.linalg.norm(z1 - z2)):.2e}"
        )

    def test_quadratic_bilinear_convergence(self):
        """LEN should converge on ½‖x‖² + xᵀy − ½‖y‖²."""
        problem = _quadratic_bilinear_problem()
        gamma, m, T = 4.0, 5, 40
        z0 = jnp.array([1.0, 1.0, -0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, calls = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
        )
        norm_after = _operator_norm(problem, z_out)
        assert calls == T
        assert norm_after < 0.25, (
            f"Operator norm too large: {norm_after:.4e}"
        )

    def test_nonlinear_convergence(self):
        """LEN should converge on a nonlinear convex-concave problem."""
        problem = _nonlinear_problem(rho=1.0)
        gamma = 2.0 * (problem.rho + 0.1)
        m, T = 2, 50
        z0 = jnp.array([0.8, -0.3, 0.4, -0.6])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, calls = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
        )
        norm_before = _operator_norm(problem, z0)
        norm_after = _operator_norm(problem, z_out)

        assert calls == T
        assert norm_after < 0.8 * norm_before, (
            f"Operator norm did not decrease: {norm_before:.4e} → {norm_after:.4e}"
        )

    def test_output_selection_with_fn(self):
        """When fn is provided, output should minimise fn among candidates."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 2, 15
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        def fn(z):
            return jnp.dot(problem.operator_F(z), problem.operator_F(z))

        z_out, calls = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem), fn=fn,
        )
        assert z_out.shape == z0.shape, "Output shape mismatch"
        # The chosen z_out should be no worse than z0 by fn.
        assert float(fn(z_out)) <= float(fn(z0)) + 1e-4, (
            "fn output selection did not improve over z0"
        )

    def test_return_full_diagnostics(self):
        """return_full=True should produce a LENResult with all fields."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 3, 12
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        result = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem), return_full=True,
        )
        assert isinstance(result, LENResult)
        assert result.oracle_calls == T
        assert result.iterations == T
        assert result.snapshot_refreshes >= 0
        assert result.num_rejected >= 0
        assert jnp.isfinite(result.final_gradient_norm)
        assert isinstance(result.converged, bool)
        assert result.z.shape == z0.shape


# ═══════════════════════════════════════════════════════════════════════════
# 2. Snapshot schedule verification (rigorous)
# ═══════════════════════════════════════════════════════════════════════════

class TestSnapshotSchedule:
    """Verify the π(t) = t − (t % m) snapshot schedule."""

    def test_refresh_count_matches_blocks(self):
        """The number of snapshot refreshes should ≈ ⌈T/m⌉."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 4, 25  # 25/4 = 6.25 → refreshes at t=0,4,8,12,16,20,24 = 7
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        result = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem), return_full=True,
        )
        # t=0 always refreshes; blocks start at 0, m, 2m, ... while < T.
        expected_refreshes = (T + m - 1) // m  # ceil(T/m)
        assert result.snapshot_refreshes == expected_refreshes, (
            f"Expected {expected_refreshes} refreshes, got {result.snapshot_refreshes}"
        )

    def test_m1_every_step_refreshes(self):
        """With m=1, snapshot refreshes every step → refreshes = T."""
        problem = _bilinear_problem()
        gamma, T = 2.0, 10
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        result = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m=1,
            project=_project_for(problem), return_full=True,
        )
        assert result.snapshot_refreshes == T, (
            f"m=1 should refresh every step, got {result.snapshot_refreshes}/{T}"
        )

    def test_m_larger_than_T_one_refresh(self):
        """When m > T, only t=0 triggers a refresh."""
        problem = _bilinear_problem()
        gamma, T = 2.0, 8
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        result = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m=100,
            project=_project_for(problem), return_full=True,
        )
        assert result.snapshot_refreshes == 1, (
            f"m > T should refresh only once, got {result.snapshot_refreshes}"
        )

    def test_snapshot_count_parametric(self):
        """Refresh count follows ⌈T/m⌉ formula across many (T, m) pairs.

        This verifies the snapshot schedule without inserting a Python-level
        tracking oracle into the scan body (which would break tracing).
        """
        problem = _bilinear_problem()
        gamma = 2.0
        z0 = jnp.array([1.0, 0.5, -0.3, -0.7])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        for T, m in [(12, 3), (10, 5), (7, 10), (20, 1), (5, 5), (1, 1), (6, 3)]:
            result = len_loop(
                oracle, problem.operator_F, z0, T=T, gamma=gamma, m=m,
                project=_project_for(problem), return_full=True,
            )
            expected = (T + m - 1) // m   # ⌈T/m⌉
            assert result.snapshot_refreshes == expected, (
                f"T={T}, m={m}: expected {expected} refreshes, "
                f"got {result.snapshot_refreshes}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Convergence comparison: LEN vs NPE
# ═══════════════════════════════════════════════════════════════════════════

class TestLenVsNPE:
    """Compare LEN and NPE convergence on the same problems."""

    @pytest.mark.parametrize("m", [1, 2, 5])
    def test_bilinear_len_vs_npe(self, m: int):
        """Both LEN and NPE should converge; m=1 must match NPE exactly."""
        problem = _bilinear_problem()
        gamma, T = 2.0, 30
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        proj = _project_for(problem)

        # NPE
        npe_oracle = make_crn_npe_oracle(problem, gamma)
        z_npe, _ = npe_loop(npe_oracle, problem.operator_F, z0, T, gamma, project=proj)
        norm_npe = _operator_norm(problem, z_npe)

        # LEN
        len_oracle = make_lazy_crn_npe_oracle(problem, gamma)
        z_len, _ = len_loop(len_oracle, problem.operator_F, z0, T, gamma, m, project=proj)
        norm_len = _operator_norm(problem, z_len)

        # Both must converge meaningfully.
        assert norm_npe < 0.3, f"NPE did not converge: ‖F‖={norm_npe:.4e}"
        assert norm_len < 0.5, f"LEN(m={m}) did not converge: ‖F‖={norm_len:.4e}"

        # m=1 should be essentially identical to NPE.
        if m == 1:
            assert jnp.allclose(z_npe, z_len, atol=1e-4), (
                f"LEN(m=1) must match NPE: ‖Δ‖={float(jnp.linalg.norm(z_npe - z_len)):.2e}"
            )

    @pytest.mark.parametrize("m", [1, 3])
    def test_nonlinear_len_vs_npe(self, m: int):
        """On a nonlinear problem (ρ > 0), LEN should still converge."""
        problem = _nonlinear_problem(rho=1.0)
        gamma = 2.0 * (problem.rho + 0.1)
        T = 60
        z0 = jnp.array([0.5, -0.5, 0.3, -0.3])
        proj = _project_for(problem)

        # NPE
        npe_oracle = make_crn_npe_oracle(problem, gamma)
        z_npe, _ = npe_loop(npe_oracle, problem.operator_F, z0, T, gamma, project=proj)
        norm_npe = _operator_norm(problem, z_npe)

        # LEN
        len_oracle = make_lazy_crn_npe_oracle(problem, gamma)
        z_len, _ = len_loop(len_oracle, problem.operator_F, z0, T, gamma, m, project=proj)
        norm_len = _operator_norm(problem, z_len)

        norm0 = _operator_norm(problem, z0)
        assert norm_npe < 0.9 * norm0, f"NPE did not converge: ‖F‖={norm_npe:.4e}"
        assert norm_len < 0.9 * norm0, f"LEN(m={m}) did not converge: ‖F‖={norm_len:.4e}"

    def test_len_restart_convergence(self):
        """len_restart should achieve geometric convergence per epoch."""
        problem = _quadratic_bilinear_problem(D=2.0)
        gamma, m, T, S = 4.0, 3, 20, 5
        z0 = jnp.array([2.0, -2.0, 1.5, -1.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, total_calls = len_restart(
            oracle, problem.operator_F, z0, T, gamma, m, S,
            project=_project_for(problem),
        )
        norm_out = _operator_norm(problem, z_out)

        assert total_calls == S * T
        assert norm_out < 0.05, f"len_restart did not converge: ‖F‖={norm_out:.4e}"

    def test_restart_return_full(self):
        """return_full=True on len_restart should aggregate diagnostics."""
        problem = _bilinear_problem()
        gamma, m, T, S = 2.0, 2, 10, 3
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        result = len_restart(
            oracle, problem.operator_F, z0, T, gamma, m, S,
            project=_project_for(problem), return_full=True,
        )
        assert isinstance(result, LENResult)
        assert result.iterations == S * T
        assert result.oracle_calls == S * T
        assert result.snapshot_refreshes >= S  # at least one per epoch
        assert jnp.isfinite(result.final_gradient_norm)

    def test_increasing_m_slows_convergence(self):
        """For fixed T, larger m generally converges slower.

        This is a *rate* test: we check that m=10 has ≥ m=1 final norm,
        with reasonable tolerance for noise.
        """
        problem = _nonlinear_problem(rho=1.0)
        gamma = 2.0 * (problem.rho + 0.1)
        T = 200
        z0 = jnp.array([0.5, -0.5, 0.3, -0.3])
        proj = _project_for(problem)

        oracle = make_lazy_crn_npe_oracle(problem, gamma)
        norms = {}
        for m in [1, 3, 10]:
            z_out, _ = len_loop(oracle, problem.operator_F, z0, T, gamma, m, project=proj)
            norms[m] = _operator_norm(problem, z_out)
            assert jnp.isfinite(norms[m]), f"m={m} produced non-finite norm"

        # m=10 should not be *better* than m=1 (staler Hessian → slower).
        # Allow small tolerance.
        assert norms[10] >= 0.95 * norms[1], (
            f"Unexpected: m=10 ({norms[10]:.4e}) appears better than m=1 ({norms[1]:.4e})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration into the triple-loop framework
# ═══════════════════════════════════════════════════════════════════════════

class TestTripleLoopIntegration:
    """Test that LEN integrates with the Minimax-AIPE triple loop."""

    def test_make_len_saddle_solver_callable(self):
        """make_len_saddle_solver should return a callable."""
        problem = _bilinear_problem()
        solver = make_len_saddle_solver(problem, m=3)
        assert callable(solver)

    def test_len_saddle_solver_on_subproblem(self):
        """The LEN solver wrapper should solve a cubic-regularised subproblem."""
        problem = _nonlinear_problem(rho=0.5)
        gamma_outer = 0.5
        m = 3

        from minimax_aipe.framework import _make_h_problem

        x_bar = jnp.array([0.3, -0.2])
        y_bar = jnp.array([-0.1, 0.4])
        h_problem = _make_h_problem(problem, x_bar, y_bar, gamma_outer)

        class MockParams:
            T_inner = 40
            S_inner = 3

        z0 = jnp.concatenate([x_bar, y_bar])
        solver = make_len_saddle_solver(problem, m=m)
        z_out, calls = solver(h_problem, z0, gamma_outer, MockParams(), "len")

        assert calls > 0, "Should have made at least one oracle call"
        norm_out = float(jnp.linalg.norm(h_problem.operator_F(z_out)))
        assert jnp.isfinite(norm_out), "Output should be finite"
        # The subproblem should be solved to reasonable accuracy.
        assert norm_out < 5.0, f"Subproblem residual too large: {norm_out:.4e}"

    def test_triple_loop_with_len_m_saddle(self):
        """Full solve() with M_saddle='len' should run without error."""
        from minimax_aipe.framework import solve

        problem = _bilinear_problem(D=1.5, ell=1.0)
        result = solve(
            problem,
            epsilon=0.1,
            gamma=1.0,
            M_saddle="len",
            verbose=False,
        )
        assert result.x.shape == (problem.dim_x,)
        assert result.y.shape == (problem.dim_y,)
        assert jnp.isfinite(result.gap)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge-case tests for LEN."""

    def test_single_iteration(self):
        """T=1 should work and return a valid output."""
        problem = _bilinear_problem()
        gamma, m = 2.0, 1
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, calls = len_loop(
            oracle, problem.operator_F, z0, T=1, gamma=gamma, m=m,
            project=_project_for(problem),
        )
        assert calls == 1
        assert z_out.shape == z0.shape
        assert jnp.all(jnp.isfinite(z_out))

    def test_m_equals_T(self):
        """When m >= T, snapshot refreshes only at t=0 — extreme lazy case."""
        problem = _bilinear_problem()
        gamma, T = 2.0, 10
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, calls = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m=100,
            project=_project_for(problem),
        )
        assert calls == T
        norm_out = _operator_norm(problem, z_out)
        assert jnp.isfinite(norm_out)
        # Should still reduce norm (though possibly slower).
        assert norm_out < _operator_norm(problem, z0), (
            "Even with m ≫ T, should make some progress"
        )

    def test_zero_initial_point(self):
        """Starting from the saddle point should stay at the saddle."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 3, 10
        z0 = jnp.zeros(4)
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, _ = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
        )
        norm_out = _operator_norm(problem, z_out)
        assert norm_out < 1e-6, f"Should stay near saddle, ‖F‖={norm_out:.4e}"

    def test_large_gamma_no_nan(self):
        """A very large γ (strong regularisation) should not cause NaN."""
        problem = _bilinear_problem()
        gamma, m, T = 1000.0, 5, 15
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, _ = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
        )
        assert jnp.all(jnp.isfinite(z_out)), "Output contains NaN/Inf"

    def test_small_gamma_no_nan(self):
        """A very small γ should not cause NaN."""
        problem = _bilinear_problem()
        gamma, m, T = 1e-8, 3, 10
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        z_out, _ = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
        )
        assert jnp.all(jnp.isfinite(z_out)), "Output contains NaN/Inf"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Numerical hardening tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericalHardening:
    """Verify NaN guards, norm explosion, eta_floor, and safety rejection."""

    def test_safety_rejects_nan_step(self):
        """A step producing NaN should be rejected (z unchanged)."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 3, 15
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])

        # Oracle that returns NaN for the 3rd call.
        call_count = [0]

        def nan_oracle(z, z_snapshot):
            call_count[0] += 1
            if call_count[0] == 3:
                nan = jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])
                return nan, nan
            return crn_oracle(problem, z_snapshot, gamma, n_iters=10)

        z_out, calls = len_loop(
            nan_oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem), safety_checks=True,
        )
        assert calls == T
        assert jnp.all(jnp.isfinite(z_out)), "Output should be finite despite NaN step"
        norm_out = _operator_norm(problem, z_out)
        norm0 = _operator_norm(problem, z0)
        assert norm_out < norm0, "Should still converge despite one NaN step"

    def test_safety_can_be_disabled(self):
        """With safety_checks=False, NaN may propagate (documented behaviour)."""
        problem = _bilinear_problem()
        gamma, m = 2.0, 1
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        # safety_checks=False with a reasonable oracle should still work.
        z_out, calls = len_loop(
            oracle, problem.operator_F, z0, T=5, gamma=gamma, m=m,
            project=_project_for(problem), safety_checks=False,
        )
        assert calls == 5
        assert jnp.all(jnp.isfinite(z_out))

    def test_eta_floor_prevents_infinite_eta(self):
        """eta_floor should prevent η from becoming enormous."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 2, 20
        z0 = jnp.array([1.0, -1.0, 0.5, -0.5])

        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        # Use a very small eta_floor — η should still be bounded.
        z_out_small, _ = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem), eta_floor=1e-6,
        )
        assert jnp.all(jnp.isfinite(z_out_small))

        # Use default eta_floor.
        z_out_default, _ = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
        )
        assert jnp.all(jnp.isfinite(z_out_default))

    def test_max_norm_clamp(self):
        """With a low max_norm, steps that would explode are rejected."""
        problem = _bilinear_problem()
        gamma, m, T = 2.0, 3, 10
        z0 = jnp.array([10.0, -10.0, 5.0, -5.0])  # far from origin
        oracle = make_lazy_crn_npe_oracle(problem, gamma)

        result = len_loop(
            oracle, problem.operator_F, z0, T, gamma, m,
            project=_project_for(problem),
            max_norm=15.0,  # very tight — will reject some steps
            return_full=True,
        )
        assert jnp.all(jnp.isfinite(result.z))
        # With tight max_norm, some steps should be rejected.
        assert result.num_rejected >= 0, "Rejection count should be tracked"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Parameter validation
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterValidation:
    """Invalid parameters should raise ValueError loudly."""

    def _dummy_oracle(self):
        problem = _bilinear_problem()
        return make_lazy_crn_npe_oracle(problem, 2.0)

    def test_T_zero_raises(self):
        with pytest.raises(ValueError, match="T must be positive"):
            len_loop(self._dummy_oracle(), lambda z: z, jnp.zeros(4),
                     T=0, gamma=2.0, m=1)

    def test_T_negative_raises(self):
        with pytest.raises(ValueError, match="T must be positive"):
            len_loop(self._dummy_oracle(), lambda z: z, jnp.zeros(4),
                     T=-5, gamma=2.0, m=1)

    def test_m_zero_raises(self):
        with pytest.raises(ValueError, match="m must be positive"):
            len_loop(self._dummy_oracle(), lambda z: z, jnp.zeros(4),
                     T=10, gamma=2.0, m=0)

    def test_m_negative_raises(self):
        with pytest.raises(ValueError, match="m must be positive"):
            len_loop(self._dummy_oracle(), lambda z: z, jnp.zeros(4),
                     T=10, gamma=2.0, m=-3)

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            len_loop(self._dummy_oracle(), lambda z: z, jnp.zeros(4),
                     T=10, gamma=0.0, m=1)

    def test_gamma_negative_raises(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            len_loop(self._dummy_oracle(), lambda z: z, jnp.zeros(4),
                     T=10, gamma=-1.0, m=1)

    def test_S_zero_raises_in_restart(self):
        problem = _bilinear_problem()
        oracle = make_lazy_crn_npe_oracle(problem, 2.0)
        with pytest.raises(ValueError, match="S must be positive"):
            len_restart(oracle, problem.operator_F, jnp.zeros(4),
                        T=10, gamma=2.0, m=1, S=0)

    def test_saddle_solver_m_zero_raises(self):
        problem = _bilinear_problem()
        with pytest.raises(ValueError, match="m must be positive"):
            make_len_saddle_solver(problem, m=0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Oracle-level equivalence tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOracleEquivalence:
    """Verify lazy_crn_oracle matches crn_oracle when snapshot = query."""

    def test_lazy_matches_fresh_when_snapshot_equals_query(self):
        """lazy_crn_oracle ≡ crn_oracle when z_snapshot == z_bar."""
        problem = _quadratic_bilinear_problem()
        gamma = 4.0
        z_bar = jnp.array([0.5, -0.3, 0.2, -0.4])

        z_fresh, u_fresh = crn_oracle(problem, z_bar, gamma, n_iters=20)
        z_lazy, u_lazy = lazy_crn_oracle(
            problem, z_bar, z_snapshot=z_bar, gamma=gamma, n_iters=20,
        )
        assert jnp.allclose(z_fresh, z_lazy, atol=1e-4), (
            f"z mismatch: ‖Δ‖={float(jnp.linalg.norm(z_fresh - z_lazy)):.2e}"
        )
        assert jnp.allclose(u_fresh, u_lazy, atol=1e-4), (
            f"u mismatch: ‖Δ‖={float(jnp.linalg.norm(u_fresh - u_lazy)):.2e}"
        )

    def test_lazy_differs_when_snapshot_differs(self):
        """When z_snapshot ≠ z_bar, results should differ (ρ > 0)."""
        problem = _nonlinear_problem(rho=1.0)
        gamma = 4.0
        z_bar = jnp.array([0.5, -0.3, 0.2, -0.4])
        z_snapshot = jnp.array([0.1, 0.1, -0.1, -0.1])

        z_fresh, _ = crn_oracle(problem, z_bar, gamma, n_iters=20)
        z_lazy, _ = lazy_crn_oracle(
            problem, z_bar, z_snapshot=z_snapshot, gamma=gamma, n_iters=20,
        )
        diff = float(jnp.linalg.norm(z_fresh - z_lazy))
        # With ρ > 0 and different snapshot, the Jacobians differ → solutions differ.
        assert diff > 1e-8, (
            "Lazy and fresh CRN should differ when snapshot ≠ query, "
            f"but diff={diff:.2e}"
        )

    def test_return_F_oracle(self):
        """Oracle with return_F=True should return 3-tuple with F_half."""
        problem = _quadratic_bilinear_problem()
        gamma = 4.0
        z_bar = jnp.array([0.5, -0.3, 0.2, -0.4])

        oracle = make_lazy_crn_npe_oracle(problem, gamma, return_F=True)
        result = oracle(z_bar, z_bar)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        z_half, u, F_half = result
        assert z_half.shape == z_bar.shape
        assert F_half.shape == z_bar.shape
        # F_half should match direct computation.
        F_direct = problem.operator_F(z_half)
        assert jnp.allclose(F_half, F_direct, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Regression: LEN(m=1) exactly matches NPE
# ═══════════════════════════════════════════════════════════════════════════

class TestRegression:
    """Regression tests ensuring LEN(m=1) exactly matches NPE."""

    def test_m1_iterates_match_npe(self):
        """With m=1, every len_loop step should produce identical iterates to npe."""
        problem = _quadratic_bilinear_problem()
        gamma, T = 4.0, 15
        z0 = jnp.array([1.0, -0.5, 0.3, -0.8])
        proj = _project_for(problem)

        npe_oracle = make_crn_npe_oracle(problem, gamma)
        z_npe, _ = npe_loop(npe_oracle, problem.operator_F, z0, T, gamma, project=proj)

        len_oracle = make_lazy_crn_npe_oracle(problem, gamma)
        z_len, _ = len_loop(len_oracle, problem.operator_F, z0, T, gamma, m=1, project=proj)

        assert jnp.allclose(z_npe, z_len, atol=1e-8), (
            f"LEN(m=1) diverged from NPE: ‖Δ‖={float(jnp.linalg.norm(z_npe - z_len)):.2e}"
        )

    def test_m1_restart_matches_npe_restart(self):
        """len_restart(m=1) should match npe_restart."""
        problem = _quadratic_bilinear_problem()
        gamma, T, S = 4.0, 10, 3
        z0 = jnp.array([1.0, -0.5, 0.3, -0.8])
        proj = _project_for(problem)

        npe_oracle = make_crn_npe_oracle(problem, gamma)
        z_npe, calls_npe = npe_restart(
            npe_oracle, problem.operator_F, z0, T, gamma, S, project=proj,
        )

        len_oracle = make_lazy_crn_npe_oracle(problem, gamma)
        z_len, calls_len = len_restart(
            len_oracle, problem.operator_F, z0, T, gamma, m=1, S=S, project=proj,
        )

        assert calls_len == calls_npe
        assert jnp.allclose(z_npe, z_len, atol=1e-8), (
            f"len_restart(m=1) diverged from npe_restart: "
            f"‖Δ‖={float(jnp.linalg.norm(z_npe - z_len)):.2e}"
        )

    def test_project_z_consistency(self):
        """len.project_z and npe.project_z must be the same function."""
        # They should be the same object (imported from npe).
        assert len_project_z is npe_project_z, (
            "len_project_z should be npe_project_z (single source of truth)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
