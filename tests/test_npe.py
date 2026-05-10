"""Tests for NPE (Algorithm 6) and NPE-restart (Algorithm 7)."""

import jax
import jax.numpy as jnp
import pytest

from minimax_aipe.problem import MinimaxProblem
from minimax_aipe.npe import (
    NPEState,
    make_crn_npe_oracle,
    npe,
    npe_restart,
    project_z,
)
from minimax_aipe.gap import estimate_gap


# ── reusable test problems ─────────────────────────────────────────────────

def _bilinear(dim: int = 2, D: float = 2.0) -> MinimaxProblem:
    """f(x, y) = xᵀy on ball(D/2) × ball(D/2).  Saddle at origin."""
    return MinimaxProblem(
        f=lambda x, y: jnp.dot(x, y),
        dim_x=dim,
        dim_y=dim,
        D_x=D,
        D_y=D,
    )


def _scsc_quadratic(dim: int = 2, mu: float = 0.5) -> MinimaxProblem:
    """Strongly-convex / strongly-concave quadratic.

    f(x, y) = (μ/2)(‖x‖² − ‖y‖²) + xᵀy.   Saddle at origin.
    """
    return MinimaxProblem(
        f=lambda x, y: (mu / 2.0) * (jnp.dot(x, x) - jnp.dot(y, y))
        + jnp.dot(x, y),
        dim_x=dim,
        dim_y=dim,
        D_x=2.0,
        D_y=2.0,
        ell=mu + 1.0,
        rho=0.0,
    )


def _npe_args(problem, gamma: float):
    """Extract (oracle, F_fn, project) from a MinimaxProblem."""
    oracle = make_crn_npe_oracle(problem, gamma)
    F_fn = problem.operator_F
    proj = lambda z: project_z(problem, z)
    return oracle, F_fn, proj


# ── helper utilities ───────────────────────────────────────────────────────

class TestProjectZ:

    def test_identity_at_origin(self):
        problem = _bilinear(dim=2)
        z = jnp.zeros(4)
        assert jnp.allclose(project_z(problem, z), z)

    def test_clamps_outside_ball(self):
        problem = _bilinear(dim=2, D=2.0)  # ball radius = 1
        z = jnp.array([5.0, 5.0, -3.0, 0.0])
        z_proj = project_z(problem, z)
        x, y = z_proj[:2], z_proj[2:]
        assert jnp.linalg.norm(x) <= 1.0 + 1e-6
        assert jnp.linalg.norm(y) <= 1.0 + 1e-6

    def test_no_change_inside_ball(self):
        problem = _bilinear(dim=2, D=4.0)  # ball radius = 2
        z = jnp.array([0.5, -0.5, 1.0, -1.0])
        assert jnp.allclose(project_z(problem, z), z)


class TestMakeCRNNPEOracle:

    def test_returns_callable(self):
        problem = _bilinear()
        oracle = make_crn_npe_oracle(problem, gamma=1.0)
        assert callable(oracle)

    def test_output_shapes(self):
        problem = _bilinear(dim=2)
        oracle = make_crn_npe_oracle(problem, gamma=1.0)
        z0 = jnp.array([0.5, -0.5, 0.3, 0.7])
        z_half, u = oracle(z0)
        assert z_half.shape == z0.shape
        assert u.shape == z0.shape

    def test_returns_jax_arrays(self):
        problem = _bilinear()
        oracle = make_crn_npe_oracle(problem, gamma=1.0)
        z_half, u = oracle(jnp.zeros(4))
        assert isinstance(z_half, jax.Array)
        assert isinstance(u, jax.Array)


# ── Algorithm 6: NPE ──────────────────────────────────────────────────────

class TestNPE:

    def test_oracle_count_matches_T(self):
        """Each iteration costs exactly one CRN call."""
        problem = _bilinear()
        oracle, F_fn, proj = _npe_args(problem, gamma=1.0)
        z0 = jnp.array([0.5, -0.5, 0.3, 0.7])
        for T in (1, 5, 10):
            _, calls = npe(oracle, F_fn, z0, T, gamma=1.0, project=proj)
            assert calls == T

    def test_output_shape(self):
        """z_out has the same shape as z0."""
        problem = _bilinear(dim=3)
        oracle, F_fn, proj = _npe_args(problem, gamma=1.0)
        z0 = jnp.ones(6)
        z_out, _ = npe(oracle, F_fn, z0, T=5, gamma=1.0, project=proj)
        assert z_out.shape == z0.shape

    def test_bilinear_gap_decreases(self):
        """Duality gap shrinks on a bilinear game after NPE."""
        problem = _bilinear(dim=2, D=2.0)
        oracle, F_fn, proj = _npe_args(problem, gamma=1.0)
        z0 = jnp.array([0.8, -0.6, 0.4, 0.9])
        x0, y0 = z0[:2], z0[2:]

        z_out, _ = npe(oracle, F_fn, z0, T=30, gamma=1.0, project=proj)
        x_out, y_out = z_out[:2], z_out[2:]

        gap_before = estimate_gap(
            problem, x0, y0, num_restarts=10, num_steps=300
        )
        gap_after = estimate_gap(
            problem, x_out, y_out, num_restarts=10, num_steps=300
        )

        assert gap_after < gap_before

    def test_stationary_at_saddle(self):
        """Starting at the saddle (0, 0) should stay near (0, 0)."""
        problem = _bilinear(dim=2)
        oracle, F_fn, proj = _npe_args(problem, gamma=1.0)
        z_out, _ = npe(
            oracle, F_fn, jnp.zeros(4), T=10, gamma=1.0, project=proj,
        )
        assert jnp.allclose(z_out, 0.0, atol=1e-6)

    def test_returns_jax_array(self):
        """Output must be a JAX array, not a Python scalar."""
        problem = _bilinear()
        oracle, F_fn, proj = _npe_args(problem, gamma=1.0)
        z_out, _ = npe(
            oracle, F_fn, jnp.zeros(4), T=3, gamma=1.0, project=proj,
        )
        assert isinstance(z_out, jax.Array)

    def test_no_projection(self):
        """NPE works without a projection (unconstrained)."""
        problem = _bilinear()
        oracle = make_crn_npe_oracle(problem, gamma=1.0)
        F_fn = problem.operator_F
        z0 = jnp.array([0.5, -0.5, 0.3, 0.7])
        z_out, calls = npe(oracle, F_fn, z0, T=5, gamma=1.0)
        assert z_out.shape == z0.shape
        assert calls == 5

    def test_fn_output_selection(self):
        """When fn is provided, output minimises fn over all candidates."""
        problem = _bilinear(dim=2)
        oracle, F_fn, proj = _npe_args(problem, gamma=1.0)
        z0 = jnp.array([0.8, -0.6, 0.4, 0.9])

        # fn returns squared norm — best iterate should be closest to origin
        fn = lambda z: jnp.dot(z, z)

        z_no_fn, _ = npe(
            oracle, F_fn, z0, T=20, gamma=1.0, project=proj,
        )
        z_fn, _ = npe(
            oracle, F_fn, z0, T=20, gamma=1.0, project=proj, fn=fn,
        )

        # both should be valid arrays of the right shape
        assert z_no_fn.shape == z0.shape
        assert z_fn.shape == z0.shape
        # fn-selected iterate should have fn value ≤ the default iterate's
        assert fn(z_fn) <= fn(z_no_fn) + 1e-6

    def test_scan_state_is_named_tuple(self):
        """NPEState is a NamedTuple (inspectable pytree)."""
        assert issubclass(NPEState, tuple)
        assert NPEState._fields == ("z", "weighted_sum", "eta_sum")


# ── Algorithm 7: NPE-restart ──────────────────────────────────────────────

class TestNPERestart:

    def test_oracle_count(self):
        """Total oracle calls = T × S."""
        problem = _bilinear()
        oracle, F_fn, proj = _npe_args(problem, gamma=1.0)
        z0 = jnp.array([0.5, -0.5, 0.3, 0.7])
        _, calls = npe_restart(
            oracle, F_fn, z0, T=5, gamma=1.0, S=4, project=proj,
        )
        assert calls == 20

    def test_scsc_convergence(self):
        """On a strongly-convex problem, restarts drive z toward z*."""
        mu = 0.5
        problem = _scsc_quadratic(dim=2, mu=mu)
        oracle, F_fn, proj = _npe_args(problem, gamma=2.0)
        z0 = jnp.array([1.0, -1.0, 0.5, 0.5])
        d0 = float(jnp.linalg.norm(z0))

        z_out, _ = npe_restart(
            oracle, F_fn, z0, T=20, gamma=2.0, S=10, project=proj,
        )
        d_out = float(jnp.linalg.norm(z_out))

        assert d_out < 0.5 * d0

    def test_more_epochs_reduce_error(self):
        """More restart epochs → smaller distance to saddle."""
        problem = _scsc_quadratic(dim=2, mu=0.5)
        oracle, F_fn, proj = _npe_args(problem, gamma=2.0)
        z0 = jnp.array([1.0, -1.0, 0.5, 0.5])

        z_few, _ = npe_restart(
            oracle, F_fn, z0, T=15, gamma=2.0, S=3, project=proj,
        )
        z_many, _ = npe_restart(
            oracle, F_fn, z0, T=15, gamma=2.0, S=10, project=proj,
        )

        assert jnp.linalg.norm(z_many) < jnp.linalg.norm(z_few)

    def test_output_shape(self):
        problem = _scsc_quadratic(dim=3)
        oracle, F_fn, proj = _npe_args(problem, gamma=2.0)
        z0 = jnp.ones(6)
        z_out, _ = npe_restart(
            oracle, F_fn, z0, T=5, gamma=2.0, S=3, project=proj,
        )
        assert z_out.shape == z0.shape

    def test_fn_output_selection(self):
        """fn-based selection propagates through restart epochs."""
        problem = _scsc_quadratic(dim=2, mu=0.5)
        oracle, F_fn, proj = _npe_args(problem, gamma=2.0)
        z0 = jnp.array([1.0, -1.0, 0.5, 0.5])
        fn = lambda z: jnp.dot(z, z)

        z_out, _ = npe_restart(
            oracle, F_fn, z0, T=10, gamma=2.0, S=5, project=proj, fn=fn,
        )
        assert z_out.shape == z0.shape
        # should be closer to origin than z0
        assert fn(z_out) < fn(z0)
