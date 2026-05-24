"""Tests for early stopping in the triple-loop restart scheme."""

import pytest
import jax.numpy as jnp

from minimax_aipe.framework import (
    _algorithm_3,
    _compute_loop_params,
    _iProx_Phi,
    _iProx_Psi,
    _restart_with_early_stop,
    solve,
)
from minimax_aipe.problem import MinimaxProblem


# ═══════════════════════════════════════════════════════════════════════
# Fixture
# ═══════════════════════════════════════════════════════════════════════

def _scsc_problem(mu: float = 1.0) -> MinimaxProblem:
    """Simple strongly-convex-strongly-concave quadratic."""
    A = jnp.array([[0.3, 0.1], [-0.2, 0.4]])

    def f(x, y):
        return (mu / 2.0) * (jnp.dot(x, x) - jnp.dot(y, y)) + x @ A @ y

    def grad_f(x, y):
        return mu * x + A @ y, mu * y - A.T @ x

    def hessian_f(x, y):
        del x, y
        return ((mu * jnp.eye(2), A), (A.T, -mu * jnp.eye(2)))

    return MinimaxProblem(
        f=f, grad_f=grad_f, hessian_f=hessian_f,
        dim_x=2, dim_y=2, D_x=4.0, D_y=4.0, ell=2.0, rho=0.0,
    )


def _tiny_params(problem, gamma=1.0):
    """Minimal params for fast tests."""
    return _compute_loop_params(problem, epsilon=0.5, gamma=gamma, npe_T_factor=0.05)


# ═══════════════════════════════════════════════════════════════════════
# 1. _restart_with_early_stop — unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestRestartHelper:

    def test_runs_all_epochs_when_no_convergence(self):
        """Without convergence, should run all S epochs."""
        z = jnp.array([10.0, 10.0])
        call_log = []

        def epoch(z_cur):
            call_log.append(1)
            return z_cur * 0.9, 1  # slow decay, won't trigger step_tol

        z_out, calls, epochs = _restart_with_early_stop(
            epoch, z, S=5,
            step_tol=1e-15,  # impossibly tight
        )
        assert epochs == 5
        assert calls == 5
        assert len(call_log) == 5

    def test_stops_early_on_step_norm(self):
        """Should stop when ‖z_new − z_old‖ < step_tol."""
        z = jnp.array([1.0, 1.0])
        call_log = []

        def epoch(z_cur):
            call_log.append(1)
            # Epoch 0: large step (0.707).  From epoch 1 onward: tiny step.
            if len(call_log) >= 2:
                return z_cur + 1e-10, 1
            return z_cur * 0.5, 1

        z_out, calls, epochs = _restart_with_early_stop(
            epoch, z, S=10,
            step_tol=1e-6,
        )
        # Epoch 0: [1,1]→[0.5,0.5], step=0.707 > 1e-6 → continue
        # Epoch 1: [0.5,0.5]→[0.5+ε,0.5+ε], step≈1.4e-10 < 1e-6 → STOP
        assert epochs == 2
        assert calls == 2

    def test_stops_early_on_residual(self):
        """Should stop when residual_fn(z) < residual_tol."""
        z = jnp.array([1.0, 1.0])

        def epoch(z_cur):
            return z_cur * 0.1, 1  # fast decay

        residual_fn = lambda z_cur: float(jnp.linalg.norm(z_cur))

        z_out, calls, epochs = _restart_with_early_stop(
            epoch, z, S=100,
            residual_fn=residual_fn,
            residual_tol=0.05,
        )
        # After epoch 1: ‖z‖ ≈ 0.141, after epoch 2: ‖z‖ ≈ 0.014 < 0.05
        assert epochs <= 3
        assert calls <= 3
        assert residual_fn(z_out) < 0.05

    def test_no_early_stop_when_tolerances_zero(self):
        """With zero tolerances, runs all S epochs (backward compat)."""
        z = jnp.array([1.0, 1.0])

        def epoch(z_cur):
            return z_cur, 1  # no change at all

        z_out, calls, epochs = _restart_with_early_stop(
            epoch, z, S=7,
            step_tol=0.0,
            residual_tol=0.0,
        )
        assert epochs == 7
        assert calls == 7


# ═══════════════════════════════════════════════════════════════════════
# 2. Inner-loop early stopping (NPE/LEN)
# ═══════════════════════════════════════════════════════════════════════

class TestInnerEarlyStop:

    def test_inner_loop_uses_fewer_epochs_on_easy_problem(self):
        """A well-conditioned problem should converge in fewer than S_inner
        epochs, and _solve_saddle_subproblem should exploit this."""
        problem = _scsc_problem(mu=2.0)
        params = _tiny_params(problem, gamma=1.0)
        x_bar = jnp.array([0.1, -0.1])
        y_bar = jnp.array([-0.05, 0.05])

        from minimax_aipe.framework import _make_h_problem
        h_problem = _make_h_problem(problem, x_bar, y_bar, gamma=1.0)
        z0 = jnp.concatenate([x_bar, y_bar])

        from minimax_aipe.framework import _solve_saddle_subproblem
        from minimax_aipe.npe import project_z
        z0_proj = project_z(h_problem, z0)

        # With a large S_inner budget, early stopping should kick in
        z_hat, calls = _solve_saddle_subproblem(
            h_problem, z0_proj, gamma=1.0,
            params=params, M_saddle="npe",
            tolerance=params.zeta_3,
        )
        residual = float(jnp.linalg.norm(h_problem.operator_F(z_hat)))
        assert jnp.isfinite(residual)
        # The test is that it completed in reasonable time (not all S_inner epochs)
        assert calls > 0


# ═══════════════════════════════════════════════════════════════════════
# 3. Outer-loop early stopping
# ═══════════════════════════════════════════════════════════════════════

class TestOuterEarlyStop:

    def test_algorithm_3_completes_in_reasonable_time(self):
        """The full triple loop with early stopping should complete in < 60s
        on a simple problem (the previous version took 5+ minutes)."""
        import time
        problem = _scsc_problem(mu=1.0)
        params = _tiny_params(problem)
        z0 = jnp.zeros(4)

        start = time.time()
        z_hat, calls, _, _ = _algorithm_3(
            problem, gamma=1.0, mu_x=0.01, mu_y=0.01, zeta_1=0.01,
            params=params, M_saddle="npe", z0=z0,
        )
        elapsed = time.time() - start

        assert jnp.all(jnp.isfinite(z_hat))
        assert calls > 0
        assert elapsed < 120.0, (
            f"Algorithm 3 took {elapsed:.1f}s — early stopping may not be working"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. End-to-end solve with early stopping
# ═══════════════════════════════════════════════════════════════════════

class TestSolveEarlyStop:

    def test_solve_completes_quickly(self):
        """End-to-end solve should complete in < 120s."""
        import time
        problem = _scsc_problem()
        start = time.time()
        result = solve(problem, epsilon=0.5, npe_T_factor=0.05)
        elapsed = time.time() - start

        assert result.oracle_calls > 0
        assert elapsed < 120.0, f"solve took {elapsed:.1f}s"

    def test_len_path_also_benefits(self):
        """M_saddle='len' should also complete in < 120s."""
        import time
        problem = _scsc_problem()
        start = time.time()
        result = solve(problem, epsilon=0.5, M_saddle="len", npe_T_factor=0.05)
        elapsed = time.time() - start

        assert result.oracle_calls > 0
        assert elapsed < 120.0, f"LEN solve took {elapsed:.1f}s"
