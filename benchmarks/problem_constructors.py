"""Shared minimax problem constructors.

Used by both tests (tests/conftest.py) and benchmarks (benchmarks/problems.py).
Every constructor returns a :class:`minimax_aipe.problem.BenchmarkProblem`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from minimax_aipe import MinimaxProblem
from minimax_aipe.problem import BenchmarkProblem, build_benchmark_meta


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
        return A @ y, -(A.T @ x)

    def hessian_f(x, y):
        zeros_xx = jnp.zeros((dim, dim))
        zeros_yy = jnp.zeros((dim, dim))
        return ((zeros_xx, A), (A.T, zeros_yy))

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
    )


def make_1d_bilinear() -> BenchmarkProblem:
    """Simplest case: f(x,y) = xy on [-1, 1]^2.

    Solution: x* = y* = 0, gap = 0.
    """
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
    )


def make_separable_problem() -> BenchmarkProblem:
    """Separable convex-concave: f(x,y) = h1(x) - h2(y).

    h1(x) = ½·(3x₁² + x₂²),  h2(y) = ½·(y₁² + 2y₂²).

    Solution: x* = y* = 0, gap = 0.
    """
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
    )


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

    ell = float(jnp.max(sigmas))
    D = 2.0

    def f(x, y):
        return x @ A @ y

    def grad_f(x, y):
        return A @ y, -(A.T @ x)

    def hessian_f(x, y):
        zeros = jnp.zeros((dim, dim))
        return ((zeros, A), (A.T, zeros))

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
        meta=build_benchmark_meta(problem, mu_x=1.0, mu_y=1.0),
        name="ill_conditioned_quadratic",
        dim=dim,
    )


def make_offset_quadratic() -> BenchmarkProblem:
    """1D quadratic with nonzero saddle point.

    f(x,y) = ½(5x² - 3y²) + 2xy + x − y

    Saddle at x* = -1/19, y* = -7/19 (inside ball of radius 2).
    Gap = 0 at the saddle.
    """
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
    )


def make_10d_quadratic(seed: int = 0) -> BenchmarkProblem:
    """10D (5+5) strongly convex quadratic with coupling.

    f(x,y) = ½ x^T Q x + x^T B y − ½ y^T R y

    Q ∈ R^{5×5}, R ∈ R^{5×5} positive definite with controlled eigenvalue
    spread.  B provides coupling between x and y.  Saddle at origin.
    """
    dim = 5
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(jnp.linspace(1.0, 10.0, dim)) @ U_q.T

    U_r, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    R = U_r @ jnp.diag(jnp.linspace(1.0, 5.0, dim)) @ U_r.T

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
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=1.0, mu_y=1.0),
        name="10d_quadratic",
        dim=dim,
    )
