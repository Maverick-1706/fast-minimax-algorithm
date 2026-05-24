"""Regression test for the Jacobian-vs-Hessian block sign bug.

Verifies that _block_chol_solve receives raw Hessian blocks (not
Jacobian-convention blocks with negated bottom row) in all three
framework call sites: make_crn_oracle (tol>0 and tol==0) and the
LEN cached-hessian path.

See: Bug 1 (CRITICAL) — Jacobian Blocks Passed to _block_chol_solve.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import pytest

from minimax_aipe.oracles import _block_chol_solve
from minimax_aipe.framework import RegularizedSubproblem
from minimax_aipe.problem import MinimaxProblem


def _make_simple_quadratic():
    """f(x,y) = x²/2 − y²/2 + xy  (dim_x=1, dim_y=1).

    Raw Hessian blocks:
        H_xx = 1,  H_xy = 1
        H_yx = 1,  H_yy = -1
    """
    def f(x, y):
        return 0.5 * x[0] ** 2 - 0.5 * y[0] ** 2 + x[0] * y[0]

    def grad_f(x, y):
        gx = jnp.array([x[0] + y[0]])
        gy_neg = jnp.array([-(-y[0] + x[0])])
        return gx, gy_neg

    def hessian_f(x, y):
        H_xx = jnp.array([[1.0]])
        H_xy = jnp.array([[1.0]])
        H_yx = jnp.array([[1.0]])
        H_yy = jnp.array([[-1.0]])
        return (H_xx, H_xy), (H_yx, H_yy)

    return MinimaxProblem(
        f=f, grad_f=grad_f, hessian_f=hessian_f,
        dim_x=1, dim_y=1, D_x=10.0, D_y=10.0,
        rho=0.0, ell=2.0,
    )


class TestBlockCholSolveSignConvention:
    """Verify _block_chol_solve produces correct results with raw blocks."""

    def test_raw_blocks_give_correct_delta(self):
        """The user's numerical example: g=[1,2], λ=2, expect δ=[-0.1, -0.7]."""
        g = jnp.array([1.0, 2.0])
        H_xx = jnp.array([[1.0]])
        H_xy = jnp.array([[1.0]])
        H_yx = jnp.array([[1.0]])
        H_yy = jnp.array([[-1.0]])
        lam = 2.0
        eye_x = jnp.eye(1)
        eye_y = jnp.eye(1)
        tiny = jnp.float32(1e-30)

        delta = _block_chol_solve(g, H_xx, H_xy, H_yx, H_yy, lam, eye_x, eye_y, tiny)
        expected = jnp.array([-0.1, -0.7])
        assert jnp.allclose(delta, expected, atol=1e-5)

    def test_jacobian_blocks_give_wrong_delta(self):
        """Negated blocks (Jacobian convention) should NOT match the expected result."""
        g = jnp.array([1.0, 2.0])
        H_xx = jnp.array([[1.0]])
        H_xy = jnp.array([[1.0]])
        J_yx = jnp.array([[-1.0]])
        J_yy = jnp.array([[1.0]])
        lam = 2.0
        eye_x = jnp.eye(1)
        eye_y = jnp.eye(1)
        tiny = jnp.float32(1e-30)

        delta = _block_chol_solve(g, H_xx, H_xy, J_yx, J_yy, lam, eye_x, eye_y, tiny)
        expected = jnp.array([-0.1, -0.7])
        assert not jnp.allclose(delta, expected, atol=1e-5)


class TestFrameworkCRNOracleBlocks:
    """Verify make_crn_oracle uses raw blocks (not Jacobian blocks)."""

    @pytest.fixture
    def kernel_and_problem(self):
        problem = _make_simple_quadratic()
        gamma = 0.0
        kernel = RegularizedSubproblem(problem, gamma)
        return kernel, problem

    def test_hessian_blocks_h_returns_raw(self, kernel_and_problem):
        """hessian_blocks_h should return unsigned Hessian blocks."""
        kernel, _ = kernel_and_problem
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        (H_xx, H_xy), (H_yx, H_yy) = kernel.hessian_blocks_h(x, y, x, y)
        assert jnp.allclose(H_xx, jnp.array([[1.0]]), atol=1e-6)
        assert jnp.allclose(H_xy, jnp.array([[1.0]]), atol=1e-6)
        assert jnp.allclose(H_yx, jnp.array([[1.0]]), atol=1e-6)
        assert jnp.allclose(H_yy, jnp.array([[-1.0]]), atol=1e-6)

    def test_jacobian_F_h_has_negated_bottom(self, kernel_and_problem):
        """jacobian_F_h should have [-H_yx, -H_yy] in the bottom row."""
        kernel, _ = kernel_and_problem
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        J = kernel.jacobian_F_h(x, y, x, y)
        assert jnp.allclose(J[1, 0], -1.0, atol=1e-6)
        assert jnp.allclose(J[1, 1], 1.0, atol=1e-6)

    def test_make_crn_oracle_no_tol(self, kernel_and_problem):
        """CRN oracle (tol=0) must match the standalone oracle direction."""
        kernel, problem = kernel_and_problem
        z_bar = jnp.array([0.5, 0.3])
        npe_gamma = 2.0

        oracle = kernel.make_crn_oracle(
            x_bar=jnp.array([0.5]),
            y_bar=jnp.array([0.3]),
            npe_gamma=npe_gamma,
            n_iters=30,
            tol=0.0,
        )
        z_out, u = oracle(z_bar)
        delta = z_out - z_bar

        g = kernel.operator_F_h(z_bar, jnp.array([0.5]), jnp.array([0.3]))
        assert jnp.dot(delta, g) < 0, (
            "CRN step should descend along the operator direction"
        )

    def test_make_crn_oracle_with_tol(self, kernel_and_problem):
        """CRN oracle (tol>0) must also produce a descent direction."""
        kernel, problem = kernel_and_problem
        z_bar = jnp.array([0.5, 0.3])
        npe_gamma = 2.0

        oracle = kernel.make_crn_oracle(
            x_bar=jnp.array([0.5]),
            y_bar=jnp.array([0.3]),
            npe_gamma=npe_gamma,
            n_iters=30,
            tol=1e-6,
        )
        z_out, u = oracle(z_bar)
        delta = z_out - z_bar

        g = kernel.operator_F_h(z_bar, jnp.array([0.5]), jnp.array([0.3]))
        assert jnp.dot(delta, g) < 0, (
            "CRN step (tol path) should descend along the operator direction"
        )

    def test_framework_oracle_matches_standalone(self):
        """Framework CRN oracle must produce the same result as the standalone."""
        from minimax_aipe.oracles import crn_oracle

        problem = _make_simple_quadratic()
        gamma = 0.0
        kernel = RegularizedSubproblem(problem, gamma)

        z_bar = jnp.array([0.5, 0.3])
        npe_gamma = 2.0

        z_standalone, u_standalone = crn_oracle(
            problem, z_bar, npe_gamma, n_iters=30
        )

        oracle_fn = kernel.make_crn_oracle(
            x_bar=jnp.array([0.5]),
            y_bar=jnp.array([0.3]),
            npe_gamma=npe_gamma,
            n_iters=30,
            tol=0.0,
        )
        z_framework, u_framework = oracle_fn(z_bar)

        assert jnp.allclose(z_framework, z_standalone, atol=1e-4), "Framework oracle diverges from standalone (sign bug?)"
