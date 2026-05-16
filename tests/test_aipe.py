"""Tests for AIPE and AIPE-restart (Algorithms 1 & 2)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import pytest

from minimax_aipe.aipe import aipe, aipe_restart, make_crn_prox_oracle
from minimax_aipe.oracles import crn_oracle_minimization


# ── helpers ────────────────────────────────────────────────────────────────

def _quadratic_2d():
    """h(z) = ½ zᵀQz + bᵀz  with Q ≻ 0."""
    Q = jnp.array([[3.0, 0.5], [0.5, 2.0]], dtype=jnp.float32)
    b = jnp.array([1.0, -2.0], dtype=jnp.float32)
    f = lambda z: 0.5 * z @ Q @ z + b @ z
    grad_f = lambda z: Q @ z + b
    hess_f = lambda z: Q
    z_star = -jnp.linalg.solve(Q, b)
    return f, grad_f, hess_f, z_star


def _project_ball(z: jnp.ndarray) -> jnp.ndarray:
    """Project onto the unit Euclidean ball."""
    norm = jnp.linalg.norm(z)
    return jnp.where(norm > 1.0, z / norm, z)


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 1 — basic convergence
# ═══════════════════════════════════════════════════════════════════════════

class TestAIPE:
    def test_quadratic_2d(self):
        f, grad_f, hess_f, z_star = _quadratic_2d()
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.5, n_iters=20)
        z_out, _ = aipe(prox, grad_f, jnp.array([10.0, -10.0]), T=30, gamma=0.5, fn=f)
        assert jnp.linalg.norm(z_out - z_star) < 0.5

    def test_quadratic_1d(self):
        Q = jnp.array([[2.0]])
        grad_f = lambda z: Q @ z
        hess_f = lambda z: Q
        f = lambda z: z[0] ** 2
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.1)
        z_out, _ = aipe(prox, grad_f, jnp.array([5.0]), T=20, gamma=0.1, fn=f)
        assert jnp.abs(z_out[0]) < 0.5

    def test_output_selection_with_fn(self):
        """Passing *fn* should pick a point at least as good as the default."""
        f, grad_f, hess_f, _ = _quadratic_2d()
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.5)
        z0 = jnp.array([5.0, 5.0])
        z_with, _ = aipe(prox, grad_f, z0, T=15, gamma=0.5, fn=f)
        z_without, _ = aipe(prox, grad_f, z0, T=15, gamma=0.5, fn=None)
        assert f(z_with) <= f(z_without) + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Constrained optimisation
# ═══════════════════════════════════════════════════════════════════════════

class TestConstrained:
    def test_box_projection(self):
        """min (z−3)²  s.t. z ∈ [−1, 1].   Optimum z* = 1."""
        grad_f = lambda z: 2.0 * (z - 3.0)
        hess_f = lambda z: jnp.eye(1) * 2.0
        f = lambda z: (z[0] - 3.0) ** 2
        project = lambda z: jnp.clip(z, -1.0, 1.0)
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.5, project=project)
        z_out, _ = aipe(
            prox, grad_f, jnp.array([0.0]), T=20, gamma=0.5,
            project=project, fn=f,
        )
        assert jnp.abs(z_out[0] - 1.0) < 0.2

    def test_ball_projection(self):
        """min ‖z − c‖²  s.t. ‖z‖ ≤ 1.   Optimum z* = c/‖c‖."""
        c = jnp.array([3.0, 4.0])
        grad_f = lambda z: 2.0 * (z - c)
        hess_f = lambda z: jnp.eye(2) * 2.0
        f = lambda z: jnp.sum((z - c) ** 2)
        prox = make_crn_prox_oracle(
            grad_f, hess_f, gamma=0.5, project=_project_ball,
        )
        z_out, _ = aipe(
            prox, grad_f, jnp.array([0.5, 0.5]), T=20, gamma=0.5,
            project=_project_ball, fn=f,
        )
        z_star = c / jnp.linalg.norm(c)
        assert jnp.linalg.norm(z_out - z_star) < 0.3

    def test_iterates_feasible_box(self):
        """All scan outputs should remain inside the feasible set."""
        grad_f = lambda z: 2.0 * (z - 5.0)
        hess_f = lambda z: jnp.eye(1) * 2.0
        project = lambda z: jnp.clip(z, -1.0, 1.0)
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.5, project=project)

        # Run with fn=None so final_state.z is returned (last iterate)
        z_out, _ = aipe(
            prox, grad_f, jnp.array([0.0]), T=10, gamma=0.5, project=project,
        )
        assert jnp.all(z_out >= -1.0 - 1e-6)
        assert jnp.all(z_out <= 1.0 + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 2 — restarts
# ═══════════════════════════════════════════════════════════════════════════

class TestAIPERestart:
    def test_convergence(self):
        Q = jnp.array([[2.0, 0.0], [0.0, 5.0]])
        grad_f = lambda z: Q @ z
        hess_f = lambda z: Q
        f = lambda z: 0.5 * z @ Q @ z
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=1.0)
        z_out, _ = aipe_restart(
            prox, grad_f, jnp.array([10.0, 10.0]),
            T=10, gamma=1.0, S=5, fn=f,
        )
        assert jnp.linalg.norm(z_out) < 1.0

    def test_more_restarts_help(self):
        Q = jnp.array([[3.0, 0.0], [0.0, 3.0]])
        grad_f = lambda z: Q @ z
        hess_f = lambda z: Q
        f = lambda z: 0.5 * z @ Q @ z
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.5)
        z0 = jnp.array([10.0, 10.0])
        z_few, _ = aipe_restart(prox, grad_f, z0, T=5, gamma=0.5, S=3, fn=f)
        z_many, _ = aipe_restart(prox, grad_f, z0, T=5, gamma=0.5, S=10, fn=f)
        assert jnp.linalg.norm(z_many) < jnp.linalg.norm(z_few)


# ═══════════════════════════════════════════════════════════════════════════
# Oracle-call accounting
# ═══════════════════════════════════════════════════════════════════════════

class TestOracleCalls:
    def test_aipe_count(self):
        Q = jnp.array([[2.0]])
        prox = make_crn_prox_oracle(lambda z: Q @ z, lambda z: Q, gamma=0.1)
        _, calls = aipe(prox, lambda z: Q @ z, jnp.array([1.0]), T=5, gamma=0.1)
        assert calls == 5  # Algorithm 1 makes T proximal oracle calls.

    def test_restart_count(self):
        Q = jnp.array([[2.0]])
        prox = make_crn_prox_oracle(lambda z: Q @ z, lambda z: Q, gamma=0.1)
        _, calls = aipe_restart(
            prox, lambda z: Q @ z, jnp.array([1.0]), T=5, gamma=0.1, S=3,
        )
        assert calls == 15  # 3 × T


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_start_at_solution(self):
        f, grad_f, hess_f, z_star = _quadratic_2d()
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.5)
        z_out, _ = aipe(prox, grad_f, z_star, T=10, gamma=0.5, fn=f)
        assert jnp.linalg.norm(z_out - z_star) < 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive CRN (tol > 0)
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveCRN:
    def test_tol_converges(self):
        """CRN with tol > 0 should still produce an accurate solution."""
        Q = jnp.array([[3.0, 0.5], [0.5, 2.0]])
        b = jnp.array([1.0, -2.0])
        grad_f = lambda z: Q @ z + b
        hess_f = lambda z: Q
        z_bar = jnp.array([0.0, 0.0])

        z, u = crn_oracle_minimization(
            grad_f, hess_f, z_bar, gamma=1.0, n_iters=50, tol=1e-6,
        )
        delta = z - z_bar
        g = grad_f(z_bar)
        H = hess_f(z_bar)
        lam = 0.5 * jnp.linalg.norm(delta)
        residual = jnp.linalg.norm(g + H @ delta + lam * delta)
        assert residual < 0.1

    def test_adaptive_aipe_converges(self):
        """AIPE with an adaptive proximal oracle should still converge."""
        Q = jnp.array([[2.0]])
        grad_f = lambda z: Q @ z
        hess_f = lambda z: Q
        f = lambda z: z[0] ** 2
        prox = make_crn_prox_oracle(
            grad_f, hess_f, gamma=0.1, n_iters=50, tol=1e-4,
        )
        z_out, _ = aipe(prox, grad_f, jnp.array([5.0]), T=10, gamma=0.1, fn=f)
        assert jnp.abs(z_out[0]) < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# JIT compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestJIT:
    def test_aipe_jit(self):
        Q = jnp.array([[2.0]])
        grad_f = lambda z: Q @ z
        hess_f = lambda z: Q
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.1)

        # T must be static under jit
        @functools.partial(jax.jit, static_argnums=(0,1,3))
        def run(prox_, grad_, z0, T_, gamma_):
            return aipe(prox_, grad_, z0, T=T_, gamma=gamma_)

        z_out, calls = run(prox, grad_f, jnp.array([1.0]), 5, 0.1)
        assert z_out.shape == (1,)
        assert calls == 5

    def test_aipe_jit_twice(self):
        """Second JIT call should hit the cache (no re-trace)."""
        Q = jnp.array([[2.0]])
        grad_f = lambda z: Q @ z
        hess_f = lambda z: Q
        prox = make_crn_prox_oracle(grad_f, hess_f, gamma=0.1)

        @functools.partial(jax.jit, static_argnums=(0,1,3))
        def run(prox_, grad_, z0, T_, gamma_):
            return aipe(prox_, grad_, z0, T=T_, gamma=gamma_)

        z1, _ = run(prox, grad_f, jnp.array([3.0]), 5, 0.1)
        z2, _ = run(prox, grad_f, jnp.array([3.0]), 5, 0.1)
        assert jnp.allclose(z1, z2)
