"""Tests for minimax_aipe.oracles — verifies EG, CRN, and lazy CRN against the paper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from minimax_aipe.problem import MinimaxProblem
from minimax_aipe.oracles import (
    eg_step,
    crn_oracle,
    crn_oracle_minimization,
    lazy_crn_oracle,
)


# ── fixtures ──────────────────────────────────────────────────────────────

def _bilinear() -> MinimaxProblem:
    def f(x, y):
        return x @ jnp.array([[1.0, 0.5], [0.3, 1.0]]) @ y
    return MinimaxProblem(f, dim_x=2, dim_y=2, D_x=10.0, D_y=10.0, rho=0.0, ell=0.0, L=2.0)


def _quadratic() -> MinimaxProblem:
    Q = jnp.diag(jnp.array([2.0, 3.0]))
    P = jnp.diag(jnp.array([1.0, 2.0]))
    A = jnp.array([[0.5, 0.1], [0.2, 0.3]])

    def f(x, y):
        return 0.5 * x @ Q @ x - 0.5 * y @ P @ y + x @ A @ y

    return MinimaxProblem(f, dim_x=2, dim_y=2, D_x=10.0, D_y=10.0, rho=0.0, ell=3.5, L=10.0)


# ── EG tests ──────────────────────────────────────────────────────────────

class TestEGStep:
    def test_residual_is_zero_at_saddle(self):
        """At the saddle z*=0 of a bilinear game, c should be near zero."""
        prob = _bilinear()
        z = jnp.zeros(4)
        z_new, c = eg_step(prob, z, eta=0.1)
        assert jnp.linalg.norm(c) < 1e-6, f"||c|| = {jnp.linalg.norm(c):.6f}"

    def test_contracts_toward_saddle(self):
        """Repeated EG steps should converge to the saddle."""
        prob = _bilinear()
        F = prob.operator_F
        z = jnp.array([1.0, -1.0, 0.5, 0.5])

        for _ in range(2000):
            z, _ = eg_step(prob, z, eta=0.1)

        assert jnp.linalg.norm(z) < 5e-2

    def test_matches_manual_eg(self):
        """eg_step output should match hand-rolled EG on the same problem."""
        prob = _quadratic()
        z = jnp.array([2.0, -1.0, 0.5, -0.5])
        eta = 0.05

        # Manual EG
        Fz = prob.operator_F(z)
        z_half = z - eta * Fz
        F_half = prob.operator_F(z_half)
        z_manual = z - eta * F_half

        # Our oracle
        z_oracle, _ = eg_step(prob, z, eta)

        assert jnp.allclose(z_manual, z_oracle, atol=1e-6)

    def test_is_jittable(self):
        prob = _bilinear()

        @jax.jit
        def step(z):
            return eg_step(prob, z, 0.1)

        z = jnp.array([1.0, 0.5, -0.3, 0.8])
        z_new, c = step(z)
        assert z_new.shape == (4,) and c.shape == (4,)


# ── CRN oracle (minimax) tests ────────────────────────────────────────────

class TestCRNOracle:
    def test_optimality_unconstrained(self):
        """u should be near zero for an interior point (no active constraints)."""
        prob = _quadratic()
        z_bar = jnp.array([0.1, -0.05, 0.03, -0.02])  # well inside ball
        z, u = crn_oracle(prob, z_bar, gamma=1.0)
        assert jnp.linalg.norm(u) < 1e-4, f"||u|| = {jnp.linalg.norm(u):.6f}"

    def test_residual_satisfies_definition(self):
        """Verify: g + Hδ + (γ/2)||δ||δ ≈ 0 for unconstrained interior."""
        prob = _quadratic()
        z_bar = jnp.array([0.5, -0.3, 0.2, -0.1])
        gamma = 2.0
        z, u = crn_oracle(prob, z_bar, gamma)

        delta = z - z_bar
        g = prob.operator_F(z_bar)
        H = jnp.concatenate([
            jnp.concatenate([jnp.array(Hb) for Hb in Ha], axis=1)
            for Ha in prob.hessian_f(z_bar[:2], z_bar[2:])
        ], axis=0)
        # ... actually let me simplify this
        from minimax_aipe.oracles import _build_jacobian
        H = _build_jacobian(prob, z_bar[:2], z_bar[2:])
        lam = (gamma / 2.0) * jnp.linalg.norm(delta)
        residual = g + H @ delta + lam * delta
        assert jnp.linalg.norm(residual) < 1e-4

    def test_bilinear_no_crash(self):
        """CRN should not crash on bilinear (skew-symmetric Jacobian)."""
        prob = _bilinear()
        z_bar = jnp.array([1.0, -0.5, 0.3, 0.7])
        z, u = crn_oracle(prob, z_bar, gamma=1.0)
        assert z.shape == (4,) and u.shape == (4,)
        assert jnp.all(jnp.isfinite(z)) and jnp.all(jnp.isfinite(u))

    def test_is_jittable(self):
        prob = _quadratic()

        @jax.jit
        def call_crn(z):
            return crn_oracle(prob, z, gamma=1.0)

        z = jnp.array([0.5, -0.3, 0.2, -0.1])
        z_out, u = call_crn(z)
        assert z_out.shape == (4,)


# ── CRN oracle (minimization) tests ──────────────────────────────────────

class TestCRNMinimization:
    def test_finds_quadratic_minimum(self):
        """CRN with tiny gamma ≈ Newton: should find exact minimizer."""
        A = jnp.array([[3.0, 0.5], [0.5, 2.0]])
        b = jnp.array([1.0, -0.5])
        z_star = -jnp.linalg.solve(A, b)

        grad_fn = lambda z: A @ z + b
        hess_fn = lambda z: A

        z_out, _ = crn_oracle_minimization(grad_fn, hess_fn, jnp.array([5.0, -3.0]), gamma=1e-8)
        assert jnp.linalg.norm(z_out - z_star) < 1e-3

    def test_optimality_condition(self):
        """g + Hδ + (γ/2)||δ||δ should be near zero."""
        A = jnp.eye(3) * 4.0
        grad_fn = lambda z: A @ z
        hess_fn = lambda z: A

        z_bar = jnp.array([1.0, -1.0, 0.5])
        gamma = 1.0
        z, u = crn_oracle_minimization(grad_fn, hess_fn, z_bar, gamma)

        delta = z - z_bar
        g = grad_fn(z_bar)
        H = hess_fn(z_bar)
        lam = (gamma / 2.0) * jnp.linalg.norm(delta)
        residual = g + H @ delta + lam * delta
        assert jnp.linalg.norm(residual) < 1e-4


# ── Lazy CRN tests ───────────────────────────────────────────────────────

class TestLazyCRN:
    def test_matches_crn_when_snapshot_equals_zbar(self):
        """When z_snapshot == z_bar, lazy CRN should give identical results."""
        prob = _quadratic()
        z_bar = jnp.array([0.3, -0.2, 0.1, -0.05])
        gamma = 1.0

        z_crn, u_crn = crn_oracle(prob, z_bar, gamma)
        z_lazy, u_lazy = lazy_crn_oracle(prob, z_bar, z_bar, gamma)

        assert jnp.allclose(z_crn, z_lazy, atol=1e-6)
        assert jnp.allclose(u_crn, u_lazy, atol=1e-6)

    def test_close_when_snapshot_near_zbar(self):
        """When z_snapshot ≈ z_bar, results should be close."""
        prob = _quadratic()
        z_bar = jnp.array([0.3, -0.2, 0.1, -0.05])
        z_snap = z_bar + 1e-4 * jnp.ones(4)
        gamma = 1.0

        z_crn, _ = crn_oracle(prob, z_bar, gamma)
        z_lazy, _ = lazy_crn_oracle(prob, z_bar, z_snap, gamma)

        assert jnp.linalg.norm(z_crn - z_lazy) < 1e-2
