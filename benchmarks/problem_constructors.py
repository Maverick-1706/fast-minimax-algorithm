"""Shared minimax problem constructors.

Used by both tests (tests/conftest.py) and benchmarks (benchmarks/problems.py).
Every constructor returns a :class:`minimax_aipe.problem.BenchmarkProblem`.

All constructors accept a ``kappa`` parameter controlling the condition number
of the problem's eigenvalue spectrum for reproducible scaling-law studies.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from minimax_aipe import MinimaxProblem
from minimax_aipe.problem import BenchmarkProblem, build_benchmark_meta


def _log_spaced_eigenvalues(dim: int, kappa: float) -> jnp.ndarray:
    """Return *dim* eigenvalues log-spaced from 1 to *kappa*."""
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1.0, got {kappa}")
    if kappa == 1.0:
        return jnp.ones(dim)
    return jnp.logspace(0.0, jnp.log10(kappa), dim)


def make_bilinear_problem(dim: int = 3, kappa: float = 1.0, seed: int = 42) -> BenchmarkProblem:
    """Bilinear game  f(x,y) = x^T A y.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of A.  Singular values are log-spaced in
        ``[1, kappa]``.  ``kappa=1`` gives the identity (unit singular
        values).
    seed : int
        Deterministic seed.

    Solution: x* = y* = 0, gap = 0.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    U, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    sigmas = _log_spaced_eigenvalues(dim, kappa)
    A = U @ jnp.diag(sigmas) @ V.T

    ell = float(jnp.max(sigmas))
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
        rho=0.0, ell=ell,
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


def make_quadratic_saddle_problem(
    dim: int = 3,
    kappa: float = 1.0,
    coupling_strength: float = 0.5,
    seed: int = 0,
) -> BenchmarkProblem:
    """Quadratic minimax  f(x,y) = ½ x^T Q x + x^T B y - ½ y^T R y.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of Q.  Eigenvalues are log-spaced in ``[1, kappa]``.
        R = I (well-conditioned dual side).
    coupling_strength : float
        Scale of the random coupling matrix B.
    seed : int
        Deterministic seed.

    With Q, R ≻ 0.  KKT at the origin for all valid Q, R, B.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    eigvals = _log_spaced_eigenvalues(dim, kappa)
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(eigvals) @ U_q.T

    R = jnp.eye(dim)
    B = jax.random.normal(k2, (dim, dim)) * coupling_strength

    D = 4.0

    def f(x, y):
        return 0.5 * x @ Q @ x + x @ B @ y - 0.5 * y @ R @ y

    def grad_f(x, y):
        return Q @ x + B @ y, -(B.T @ x - R @ y)

    def hessian_f(x, y):
        return ((Q, B), (B.T, -R))

    KKT = jnp.block([[Q, B], [B.T, -R]])
    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=float(jnp.linalg.norm(KKT, ord=2)),
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


def make_ill_conditioned_bilinear(dim: int = 4, kappa: float = 1e4, seed: int = 42, **kwargs) -> BenchmarkProblem:
    """Bilinear game  f(x,y) = x^T A y  where  κ(A) = *kappa*.

    Constructs A = U @ diag(σ) @ V^T with σ log-spaced from 1 to *kappa*.
    Saddle at origin, gap = 0.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of A.
    seed : int
        Deterministic seed.
    **kwargs
        Deprecated: ``condition_number`` is accepted as an alias for ``kappa``.
    """
    if "condition_number" in kwargs:
        warnings.warn(
            "condition_number is deprecated, use kappa instead",
            DeprecationWarning,
            stacklevel=2,
        )
        kappa = kwargs.pop("condition_number")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    U, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    sigmas = _log_spaced_eigenvalues(dim, kappa)
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


def make_ill_conditioned_quadratic(dim: int = 4, kappa: float = 1e4, seed: int = 0, **kwargs) -> BenchmarkProblem:
    """Quadratic minimax with ill-conditioned Hessian block Q.

    f(x,y) = ½ x^T Q x + x^T B y - ½ y^T R y
    Q has eigenvalues log-spaced from 1 to *kappa*.
    R = I, B = small random.
    Saddle at origin, gap = 0.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of Q.
    seed : int
        Deterministic seed.
    **kwargs
        Deprecated: ``condition_number`` is accepted as an alias for ``kappa``.
    """
    if "condition_number" in kwargs:
        warnings.warn(
            "condition_number is deprecated, use kappa instead",
            DeprecationWarning,
            stacklevel=2,
        )
        kappa = kwargs.pop("condition_number")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    eigvals = _log_spaced_eigenvalues(dim, kappa)
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


def make_10d_quadratic(dim: int = 5, kappa: float = 10.0, seed: int = 0) -> BenchmarkProblem:
    """Strongly convex quadratic with coupling — scalable beyond 10D.

    f(x,y) = ½ x^T Q x + x^T B y − ½ y^T R y

    Q eigenvalues log-spaced in [1, kappa], R = I.

    Parameters
    ----------
    dim : int
        Dimension of x (and y).  Default 5 (= 10D total).
    kappa : float
        Condition number of Q.  Default 10.0.
    seed : int
        Deterministic seed.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    eigvals = _log_spaced_eigenvalues(dim, kappa)
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(eigvals) @ U_q.T

    R = jnp.eye(dim)
    B = jax.random.normal(k2, (dim, dim)) * 0.3

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
