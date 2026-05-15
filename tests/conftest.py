"""Pytest configuration.

JAX-Metal on Apple Silicon has known issues with default_memory_space.
Force CPU backend for reliable testing. Set JAX_PLATFORMS=metal explicitly
if you want to test Metal.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

"""Shared fixtures for the comprehensive test suite.

Problem constructors return a BenchmarkProblem instance with fields:
    problem      — MinimaxProblem
    x_star, y_star — known optimal solution
    gap_star     — known optimal gap (0.0 for most toy problems)
    meta         — BenchmarkMeta instance containing:
"""

# Import the package first so _precision.py sets jax_enable_x64=False before
# any JAX array is created by this file or downstream test modules.
import minimax_aipe  # noqa: F401 — side-effect: FP32 config
from minimax_aipe._precision import TEST_ATOL as ATOL, PROJ_EPS as _PROJ_EPS
from minimax_aipe.problem import BenchmarkMeta, BenchmarkProblem, build_benchmark_meta

import jax
import jax.numpy as jnp
from jax import Array

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
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    D_y = float(problem.D_y)
    D_x = float(problem.D_x)

    # Uniform samples inside the ball of radius D/2
    y_raw = jax.random.normal(k1, (n_grid, problem.dim_y))
    y_norms = jnp.linalg.norm(y_raw, axis=1, keepdims=True)
    y_r = jax.random.uniform(k2, (n_grid, 1)) ** (1.0 / problem.dim_y)
    y_candidates = y_raw / (y_norms + _PROJ_EPS) * y_r * (D_y / 2)
    y_candidates = jax.vmap(problem.project_y)(y_candidates)
    # Include the origin and a few axis-aligned points
    origin_y = jnp.zeros((1, problem.dim_y))
    y_candidates = jnp.concatenate([y_candidates, origin_y], axis=0)
    max_f = float(jnp.max(jax.vmap(lambda yy: problem.f(x, yy))(y_candidates)))

    k3, k4 = jax.random.split(k2)
    x_raw = jax.random.normal(k3, (n_grid, problem.dim_x))
    x_norms = jnp.linalg.norm(x_raw, axis=1, keepdims=True)
    x_r = jax.random.uniform(k4, (n_grid, 1)) ** (1.0 / problem.dim_x)
    x_candidates = x_raw / (x_norms + _PROJ_EPS) * x_r * (D_x / 2)
    x_candidates = jax.vmap(problem.project_x)(x_candidates)
    origin_x = jnp.zeros((1, problem.dim_x))
    x_candidates = jnp.concatenate([x_candidates, origin_x], axis=0)
    min_f = float(jnp.min(jax.vmap(lambda xx: problem.f(xx, y))(x_candidates)))

    return max(0.0, max_f - min_f)


# ── Problem constructors ─────────────────────────────────────────────────

def make_bilinear_problem(dim: int = 3, seed: int = 42) -> BenchmarkProblem:
    """Bilinear game  f(x,y) = x^T A y.

    Solution: x* = y* = 0, gap = 0.
    """
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (dim, dim))
    D = 2.0

    def f(x, y):
        return x @ A @ y

    def grad_f(x, y):
        return A @ y, -(A.T @ x)  # (∇_x f, -∇_y f)

    def hessian_f(x, y):
        zeros_xx = jnp.zeros((dim, dim))
        zeros_yy = jnp.zeros((dim, dim))
        return ((zeros_xx, A), (A.T, zeros_yy))

    from minimax_aipe import MinimaxProblem
    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=0.0,
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=0.0, mu_y=0.0),
        name="bilinear",
        dim=dim,
        z0=None,
    )



def make_quadratic_saddle_problem(dim: int = 3, seed: int = 0) -> BenchmarkProblem:
    """Quadratic minimax  f(x,y) = ½ x^T Q x + x^T B y - ½ y^T R y.

    With Q, R ≻ 0.  KKT at the origin for all valid Q, R, B.
    """
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)

    L_q = jax.random.normal(k1, (dim, dim))
    Q = L_q @ L_q.T + jnp.eye(dim)
    L_r = jax.random.normal(k2, (dim, dim))
    R = L_r @ L_r.T + jnp.eye(dim)
    B = jax.random.normal(k3, (dim, dim)) * 0.5

    D = 4.0

    def f(x, y):
        return 0.5 * x @ Q @ x + x @ B @ y - 0.5 * y @ R @ y

    def grad_f(x, y):
        return Q @ x + B @ y, -(B.T @ x - R @ y)

    def hessian_f(x, y):
        return ((Q, B), (B.T, -R))

    from minimax_aipe import MinimaxProblem
    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=float(jnp.linalg.norm(jnp.block([[Q, B], [B.T, R]]), ord=2)),
    )
    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = float(jnp.min(jnp.linalg.eigvalsh(R)))

    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y),
        name="quadratic",
        dim=dim,
        z0=None,
    )


def make_1d_bilinear() -> BenchmarkProblem:
    """Simplest case: f(x,y) = xy on [-1, 1]^2.

    Solution: x* = y* = 0, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    def f(x, y):
        return x[0] * y[0]

    problem = MinimaxProblem(
        f=f, dim_x=1, dim_y=1, D_x=2.0, D_y=2.0,
        rho=0.0, ell=0.0,
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.array([0.0]),
        y_star=jnp.array([0.0]),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=0.0, mu_y=0.0),
        name="1d_bilinear",
        dim=1,
        z0=None,
    )



def make_separable_problem(dim: int = 2) -> BenchmarkProblem:
    """Separable convex-concave: f(x,y) = h1(x) - h2(y).

    h1(x) = ½·(3x₁² + x₂²),  h2(y) = ½·(y₁² + 2y₂²).

    Solution: x* = y* = 0, gap = 0.
    """
    if dim != 2:
        import warnings
        warnings.warn(f"make_separable_problem is fixed at dim=2; ignoring dim={dim}", RuntimeWarning)
    from minimax_aipe import MinimaxProblem

    def f(x, y):
        h1 = 0.5 * (3.0 * x[0]**2 + x[1]**2)
        h2 = 0.5 * (y[0]**2 + 2.0 * y[1]**2)
        return h1 - h2

    def grad_f(x, y):
        return jnp.array([3.0 * x[0], x[1]]), -jnp.array([y[0], 2.0 * y[1]])

    def hessian_f(x, y):
        H_x = jnp.diag(jnp.array([3.0, 1.0]))
        H_y = jnp.diag(jnp.array([1.0, 2.0]))
        zeros = jnp.zeros((2, 2))
        return ((H_x, zeros), (zeros, -H_y))

    problem = MinimaxProblem(
        f=f, dim_x=2, dim_y=2, D_x=4.0, D_y=4.0,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=3.0,
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(2),
        y_star=jnp.zeros(2),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=1.0, mu_y=1.0),
        name="separable",
        dim=2,
        z0=None,
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

def make_ill_conditioned_bilinear(dim: int = 4, condition_number: float = 1e4, seed: int = 42) -> BenchmarkProblem:
    """Bilinear game  f(x,y) = x^T A y  where  κ(A) = condition_number.

    Constructs A = U @ diag(σ) @ V^T with σ log-spaced from 1 to condition_number.
    Saddle at origin, gap = 0.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    U, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    sigmas = jnp.logspace(0.0, jnp.log10(condition_number), dim)
    A = U @ jnp.diag(sigmas) @ V.T

    ell = float(jnp.max(sigmas))  # spectral norm = σ_max
    D = 2.0

    def f(x, y):
        return x @ A @ y

    def grad_f(x, y):
        return A @ y, -(A.T @ x)

    def hessian_f(x, y):
        zeros = jnp.zeros((dim, dim))
        return ((zeros, A), (A.T, zeros))

    from minimax_aipe import MinimaxProblem
    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=ell,
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=0.0, mu_y=0.0),
        name="ill_conditioned_bilinear",
        dim=dim,
        z0=None,
    )



def make_ill_conditioned_quadratic(dim: int = 4, condition_number: float = 1e4, seed: int = 0) -> BenchmarkProblem:
    """Quadratic minimax with ill-conditioned Hessian block Q.

    f(x,y) = ½ x^T Q x + x^T B y - ½ y^T R y
    Q has eigenvalues log-spaced from 1 to condition_number.
    R = I, B = small random.
    Saddle at origin, gap = 0.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Ill-conditioned Q via eigendecomposition
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    eigvals = jnp.logspace(0.0, jnp.log10(condition_number), dim)
    Q = U_q @ jnp.diag(eigvals) @ U_q.T

    R = jnp.eye(dim)
    B = jax.random.normal(k2, (dim, dim)) * 0.1

    KKT = jnp.block([[Q, B], [B.T, -R]])
    ell = float(jnp.linalg.norm(KKT, ord=2))
    D = 4.0

    def f(x, y):
        return 0.5 * x @ Q @ x + x @ B @ y - 0.5 * y @ R @ y

    def grad_f(x, y):
        return Q @ x + B @ y, -(B.T @ x - R @ y)

    def hessian_f(x, y):
        return ((Q, B), (B.T, -R))

    from minimax_aipe import MinimaxProblem
    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=ell,
    )
    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = 1.0
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y),
        name="ill_conditioned_quadratic",
        dim=dim,
        z0=None,
    )

# ── Nontrivial problem constructors ──────────────────────────────────────

def make_offset_quadratic(dim: int = 1) -> BenchmarkProblem:
    """1D quadratic with nonzero saddle point.

    f(x,y) = ½(5x² - 3y²) + 2xy + x − y

    Saddle at x* = -1/19, y* = -7/19 (inside ball of radius 2).
    Gap = 0 at the saddle.
    """
    if dim != 1:
        import warnings
        warnings.warn(f"make_offset_quadratic is fixed at dim=1; ignoring dim={dim}", RuntimeWarning)
    from minimax_aipe import MinimaxProblem

    x_star = jnp.array([-1.0 / 19.0])
    y_star = jnp.array([-7.0 / 19.0])

    def f(x, y):
        return (
            0.5 * (5.0 * x[0] ** 2 - 3.0 * y[0] ** 2)
            + 2.0 * x[0] * y[0]
            + x[0]
            - y[0]
        )

    def grad_f(x, y):
        gx = jnp.array([5.0 * x[0] + 2.0 * y[0] + 1.0])
        gy_neg = jnp.array([3.0 * y[0] - 2.0 * x[0] + 1.0])
        return gx, gy_neg

    def hessian_f(x, y):
        H_xx = jnp.array([[5.0]])
        H_xy = jnp.array([[2.0]])
        H_yx = jnp.array([[2.0]])
        H_yy = jnp.array([[-3.0]])
        return ((H_xx, H_xy), (H_yx, H_yy))

    problem = MinimaxProblem(
        f=f,
        dim_x=1,
        dim_y=1,
        D_x=4.0,
        D_y=4.0,
        grad_f=grad_f,
        hessian_f=hessian_f,
        rho=0.0,
        ell=float(4.0 + jnp.sqrt(jnp.array(5.0))),
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=x_star,
        y_star=y_star,
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=5.0, mu_y=3.0),
        name="offset_quadratic",
        dim=1,
        z0=None,
    )


def make_10d_quadratic(seed: int = 0) -> BenchmarkProblem:
    """10D (5+5) strongly convex quadratic with coupling.

    f(x,y) = ½ x^T Q x + x^T B y − ½ y^T R y

    Q ∈ R^{5×5}, R ∈ R^{5×5} positive definite with controlled eigenvalue
    spread.  B provides coupling between x and y.  Saddle at origin.
    """
    from minimax_aipe import MinimaxProblem

    dim = 5
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Q with eigenvalues in [1, 10]
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(jnp.linspace(1.0, 10.0, dim)) @ U_q.T

    # R with eigenvalues in [1, 5]
    U_r, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    R = U_r @ jnp.diag(jnp.linspace(1.0, 5.0, dim)) @ U_r.T

    # Coupling matrix (not too large relative to Q, R)
    B = jax.random.normal(k3, (dim, dim)) * 0.3

    KKT = jnp.block([[Q, B], [B.T, -R]])
    ell = float(jnp.linalg.norm(KKT, ord=2))

    def f(x, y):
        return 0.5 * x @ Q @ x + x @ B @ y - 0.5 * y @ R @ y

    def grad_f(x, y):
        return Q @ x + B @ y, -(B.T @ x - R @ y)

    def hessian_f(x, y):
        return ((Q, B), (B.T, -R))

    problem = MinimaxProblem(
        f=f,
        dim_x=dim,
        dim_y=dim,
        D_x=4.0,
        D_y=4.0,
        grad_f=grad_f,
        hessian_f=hessian_f,
        rho=0.0,
        ell=ell,
    )
    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = float(jnp.min(jnp.linalg.eigvalsh(R)))

    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y),
        name="10d_quadratic",
        dim=dim,
        z0=None,
    )


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
