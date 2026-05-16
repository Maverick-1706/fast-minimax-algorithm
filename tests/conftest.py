"""Pytest configuration.

JAX-Metal on Apple Silicon has known issues with default_memory_space.
Force CPU backend for reliable testing. Set JAX_PLATFORMS=metal explicitly
if you want to test Metal.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "")
"""Shared fixtures for the comprehensive test suite.

Problem constructors return a dict with:
    problem      — MinimaxProblem
    x_star, y_star — known optimal solution
    gap_star     — known optimal gap (0.0 for most toy problems)
"""

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import Array
from minimax_aipe.problem import BenchmarkProblem, build_benchmark_meta

# ── Shared constants ─────────────────────────────────────────────────────

TOLERANCE_LEVELS = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

MAX_DIM = 6   # keep problems small for test speed


# ── Gap estimation helper ────────────────────────────────────────────────

def grid_gap(problem, x: Array, y: Array, n_grid: int = 64) -> float:
    """Estimate duality gap via brute-force grid search.

    For small problems only.  Returns max_y f(x,y) - min_x f(x,y).
    Uses uniform random samples on the feasible ball for better coverage
    than normal-distributed points.
    """
    import jax
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    D_y = float(problem.D_y)
    D_x = float(problem.D_x)

    # Uniform samples inside the ball of radius D/2
    y_raw = jax.random.normal(k1, (n_grid, problem.dim_y))
    y_norms = jnp.linalg.norm(y_raw, axis=1, keepdims=True)
    y_r = jax.random.uniform(k2, (n_grid, 1)) ** (1.0 / problem.dim_y)
    y_candidates = y_raw / (y_norms + 1e-12) * y_r * (D_y / 2)
    y_candidates = jax.vmap(problem.project_y)(y_candidates)
    # Include the origin and a few axis-aligned points
    origin_y = jnp.zeros((1, problem.dim_y))
    y_candidates = jnp.concatenate([y_candidates, origin_y], axis=0)
    max_f = float(jnp.max(jax.vmap(lambda yy: problem.f(x, yy))(y_candidates)))

    k3, k4 = jax.random.split(k2)
    x_raw = jax.random.normal(k3, (n_grid, problem.dim_x))
    x_norms = jnp.linalg.norm(x_raw, axis=1, keepdims=True)
    x_r = jax.random.uniform(k4, (n_grid, 1)) ** (1.0 / problem.dim_x)
    x_candidates = x_raw / (x_norms + 1e-12) * x_r * (D_x / 2)
    x_candidates = jax.vmap(problem.project_x)(x_candidates)
    origin_x = jnp.zeros((1, problem.dim_x))
    x_candidates = jnp.concatenate([x_candidates, origin_x], axis=0)
    min_f = float(jnp.min(jax.vmap(lambda xx: problem.f(xx, y))(x_candidates)))

    return max(0.0, max_f - min_f)

# ── Problem constructors (imported from shared module) ────────────────────

from benchmarks.problem_constructors import (  # noqa: E402, F401
    make_bilinear_problem,
    make_quadratic_saddle_problem,
    make_1d_bilinear,
    make_separable_problem,
    make_ill_conditioned_bilinear,
    make_ill_conditioned_quadratic,
    make_offset_quadratic,
    make_10d_quadratic,
)


# ── Convenience fixture wrappers ─────────────────────────────────────────

import pytest


@pytest.fixture(params=[
    ("bilinear_3d", 3, 42),
    ("bilinear_5d", 5, 123),
])
def bilinear_problem(request):
    _, dim, seed = request.param
    return make_bilinear_problem(dim=dim, seed=seed)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def offset_quadratic():
    return make_offset_quadratic()


@pytest.fixture
def large_quadratic_10d():
    return make_10d_quadratic(seed=0)


@pytest.fixture
def ill_conditioned_bilinear():
    return make_ill_conditioned_bilinear(dim=4, condition_number=1e4, seed=42)


@pytest.fixture
def ill_conditioned_quadratic():
    return make_ill_conditioned_quadratic(dim=4, condition_number=1e4, seed=0)


@pytest.fixture(params=[
    ("quad_3d", 3, 0),
    ("quad_5d", 5, 7),
])
def quadratic_problem(request):
    _, dim, seed = request.param
    return make_quadratic_saddle_problem(dim=dim, seed=seed)


@pytest.fixture
def problem_1d():
    return make_1d_bilinear()


@pytest.fixture
def separable_problem():
    return make_separable_problem()


@pytest.fixture
def bilinear_3d():
    return make_bilinear_problem(dim=3, seed=42)


@pytest.fixture
def quadratic_3d():
    return make_quadratic_saddle_problem(dim=3, seed=0)
