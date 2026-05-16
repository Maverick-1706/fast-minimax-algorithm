"""Problem zoo for benchmarks.

Catalogue of minimax problems at varying dimensions and conditionings.
Reuses constructors from tests/conftest.py and adds benchmark-specific variants.
"""

from __future__ import annotations

import warnings
from typing import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe._precision import PROJ_EPS as _PROJ_EPS
from minimax_aipe.problem import BenchmarkMeta, BenchmarkProblem, build_benchmark_meta

from benchmarks.families import (
    make_bilinear_polytope,
    make_logsumexp_saddle,
    make_sparse_bilinear,
    make_random_cubic_quadratic,
    make_adversarial_training_toy,
    make_scalable_diagonal,
)

from benchmarks.problem_constructors import (
    make_bilinear_problem,
    make_ill_conditioned_bilinear,
    make_ill_conditioned_quadratic,
    make_offset_quadratic,
    make_quadratic_saddle_problem,
    make_separable_problem,
)


# ── New problem constructors ─────────────────────────────────────────────


def make_nonzero_rho_quadratic(dim: int = 5, rho: float = 1.0, seed: int = 0) -> BenchmarkProblem:
    """Quadratic saddle + cubic perturbation so ρ > 0.

    f(x,y) = ½ xᵀQ x + xᵀ B y − ½ yᵀ R y + (ρ/3)(‖x‖³ − ‖y‖³)

    The cubic terms make the Hessian Lipschitz constant ρ > 0, exercising
    the full cubic-regularisation path in the solver.
    """
    from minimax_aipe import MinimaxProblem

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)

    L_q = jax.random.normal(k1, (dim, dim))
    Q = L_q @ L_q.T + jnp.eye(dim)
    L_r = jax.random.normal(k2, (dim, dim))
    R = L_r @ L_r.T + jnp.eye(dim)
    B = jax.random.normal(k3, (dim, dim)) * 0.3

    KKT = jnp.block([[Q, B], [B.T, -R]])
    ell = float(jnp.linalg.norm(KKT, ord=2))
    D = 4.0

    def f(x, y):
        quad = 0.5 * x @ Q @ x + x @ B @ y - 0.5 * y @ R @ y
        cubic_x = (rho / 3.0) * jnp.linalg.norm(x) ** 3
        cubic_y = (rho / 3.0) * jnp.linalg.norm(y) ** 3
        return quad + cubic_x - cubic_y

    def grad_f(x, y):
        gx_quad = Q @ x + B @ y
        gy_neg_quad = -(B.T @ x - R @ y)
        # ∇_x (ρ/3 ‖x‖³) = ρ ‖x‖ x
        norm_x = jnp.linalg.norm(x)
        norm_y = jnp.linalg.norm(y)
        gx = gx_quad + rho * norm_x * x
        gy_neg = gy_neg_quad + rho * norm_y * y
        return gx, gy_neg

    def hessian_f(x, y):
        norm_x = jnp.linalg.norm(x)
        norm_y = jnp.linalg.norm(y)
        H_cubic_x = rho * (norm_x * jnp.eye(dim) + jnp.outer(x, x) / jnp.maximum(norm_x, _PROJ_EPS))
        H_cubic_y = rho * (norm_y * jnp.eye(dim) + jnp.outer(y, y) / jnp.maximum(norm_y, _PROJ_EPS))
        H_xx = Q + H_cubic_x
        H_yy = -R - H_cubic_y
        # Note: We return a tuple of tuples ((H_xx, H_xy), (H_yx, H_yy)).
        # This matches the structure returned by jax.hessian(f, argnums=(0, 1))
        # and is the exact format expected by MinimaxProblem and make_jacobian.
        return ((H_xx, B), (B.T, H_yy))

    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=rho, ell=ell + rho * D,
    )
    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = float(jnp.min(jnp.linalg.eigvalsh(R)))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y),
        name="nonzero_rho",
        dim=dim,
        z0=None,
    )


def make_rosenbrock_bilinear(dim: int = 5, seed: int = 0) -> BenchmarkProblem:
    """Rosenbrock-style objective in x + bilinear coupling with y.

    f(x,y) = Σ_i [100(x_{i+1} − x_i²)² + (1 − x_i)²] + xᵀ A y

    Note: The Rosenbrock function is non-convex globally, violating the 
    strict convex-concave assumption (Assumption 3.1) of the Minimax-AIPE paper. 
    We include it in the benchmark suite intentionally as a challenging stress-test
    to evaluate the algorithm's robustness and empirical performance when 
    theoretical guarantees do not hold.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (dim, dim)) * 0.5

    D = 4.0

    def f(x, y):
        x_shifted = x + 1.0
        rosen = jnp.sum(
            100.0 * (x_shifted[1:] - x_shifted[:-1] ** 2) ** 2
            + (1.0 - x_shifted[:-1]) ** 2
        )
        bilin = x @ A @ y
        return rosen + bilin

    problem = MinimaxProblem(
        f=f,
        dim_x=dim, dim_y=dim,
        D_x=D, D_y=D,
        rho=200.0,
        ell=1e4,
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=None,
        meta=build_benchmark_meta(problem, mu_x=None, mu_y=None, has_analytical_solution=False),
        name="rosenbrock_bilin",
        dim=dim,
        z0=None,
    )


def make_diagonal_saddle(dim: int = 10, seed: int = 0) -> BenchmarkProblem:
    """Diagonal quadratic saddle — cleanest scaling test bed.

    f(x,y) = Σ_i (λ_i/2) x_i² + Σ_i σ_i x_i y_i − Σ_i (μ_i/2) y_i²

    where λ_i, μ_i > 0 are log-spaced eigenvalues and σ_i are coupling
    coefficients.  All matrices are diagonal, so:
      - Hessian is diagonal → O(n) linear solves
      - Operator F(z) is element-wise → O(n) per step
      - Scaling is purely algorithmic (oracle calls × per-call cost)

    Saddle at origin, gap = 0.  Condition number controlled by eigenvalue
    spread.
    """
    from minimax_aipe import MinimaxProblem

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)

    lam = jnp.exp(jax.random.uniform(k1, (dim,), minval=0.0, maxval=3.0))
    mu = jnp.exp(jax.random.uniform(k2, (dim,), minval=0.0, maxval=3.0))
    sigma = jax.random.uniform(k3, (dim,), minval=-1.0, maxval=1.0)

    ell = float(jnp.max(jnp.concatenate([lam + jnp.abs(sigma),
                                           mu + jnp.abs(sigma)])))
    D = 4.0

    def f(x, y):
        return (
            0.5 * jnp.dot(lam, x ** 2)
            + jnp.dot(sigma, x * y)
            - 0.5 * jnp.dot(mu, y ** 2)
        )

    def grad_f(x, y):
        gx = lam * x + sigma * y
        gy_neg = mu * y - sigma * x
        return gx, gy_neg

    def hessian_f(x, y):
        H_xx = jnp.diag(lam)
        H_xy = jnp.diag(sigma)
        H_yx = jnp.diag(sigma)
        H_yy = jnp.diag(-mu)
        return ((H_xx, H_xy), (H_yx, H_yy))

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
        meta=build_benchmark_meta(problem, mu_x=float(jnp.min(lam)), mu_y=float(jnp.min(mu))),
        name="diagonal_saddle",
        dim=dim,
        z0=None,
    )


# ── Problem registry ─────────────────────────────────────────────────────


def _make_fixed_dim(constructor: Callable, dim_fixed: int):
    """Wrap a no-dim constructor so it accepts a dim argument (ignored)."""
    def wrapper(dim=None, **kwargs):
        return constructor(**kwargs)
    return wrapper


def _seed_wrapper(constructor, accepts_seed: bool = True):
    """Wrap a constructor so it forwards a seed kwarg only if accepted."""
    if not accepts_seed:
        def wrapper(dim=None, seed=None, **kwargs):
            if seed is not None:
                warnings.warn(f"Constructor {constructor.__name__} does not accept a seed argument; ignoring seed={seed}.", RuntimeWarning)
            return constructor(dim=dim, **kwargs) if dim is not None else constructor(**kwargs)
        return wrapper
    return constructor


_PROBLEM_REGISTRY: list[tuple[str, Callable, list[int]]] = [
    ("bilinear",         make_bilinear_problem,                                      [2, 5, 10, 20, 50]),
    ("quadratic",        make_quadratic_saddle_problem,                              [2, 5, 10, 20, 50]),
    ("ill_bilinear",     lambda dim=4, seed=42, **kw: make_ill_conditioned_bilinear(dim=dim, seed=seed, **kw),  [4, 10, 20]),
    ("ill_quadratic",    lambda dim=4, seed=0, **kw: make_ill_conditioned_quadratic(dim=dim, seed=seed, **kw),  [4, 10, 20]),
    ("separable",        _seed_wrapper(_make_fixed_dim(make_separable_problem, 2), accepts_seed=False),  [2]),
    ("offset_quadratic", _seed_wrapper(_make_fixed_dim(make_offset_quadratic, 1), accepts_seed=False),  [1]),
    ("nonzero_rho",      make_nonzero_rho_quadratic,                                 [5, 10, 20]),
    ("rosenbrock_bilin", make_rosenbrock_bilinear,                                   [5, 10, 20]),
    ("diagonal_saddle",  make_diagonal_saddle,                                        [5, 10, 20, 50, 100]),
    ("logsumexp_saddle", make_logsumexp_saddle,                                      [5, 10, 20, 50]),
    ("sparse_bilinear",  make_sparse_bilinear,                                       [10, 50, 100, 200]),
    ("random_cubic",     make_random_cubic_quadratic,                                [5, 10, 20]),
    ("adversarial_training", make_adversarial_training_toy,                         [5, 10, 20]),
    ("bilinear_polytope", make_bilinear_polytope,                                   [5, 10, 20]),
    ("scalable_diagonal", make_scalable_diagonal,                                    [100, 500, 1000, 2000]),

]


# ── Public API ───────────────────────────────────────────────────────────

def generate_benchmark_z0(problem) -> jnp.ndarray:
    """Deterministic nonzero start so origin-saddle problems do real work."""
    x_dir = jnp.linspace(1.0, 2.0, problem.dim_x)
    y_dir = -jnp.linspace(2.0, 1.0, problem.dim_y)
    x_dir = x_dir / jnp.maximum(jnp.linalg.norm(x_dir), _PROJ_EPS)
    y_dir = y_dir / jnp.maximum(jnp.linalg.norm(y_dir), _PROJ_EPS)
    x0 = problem.project_x(0.25 * float(problem.D_x) * x_dir)
    y0 = problem.project_y(0.25 * float(problem.D_y) * y_dir)
    return jnp.concatenate([x0, y0])


def get_problem(name: str, dim: int, *, seed: int | None = None, **kwargs) -> BenchmarkProblem:
    """Return a single problem by name and dimension.

    Parameters
    ----------
    name : str
        Problem name (see _PROBLEM_REGISTRY).
    dim : int
        Problem dimension.
    seed : int or None
        Deterministic seed for the problem constructor.  When ``None``,
        each constructor uses its own default seed.
    **kwargs
        Extra arguments forwarded to the constructor (e.g. rho).

    Returns
    -------
    BenchmarkProblem
        The requested problem.
    """
    for reg_name, constructor, _ in _PROBLEM_REGISTRY:
        if reg_name == name:
            kw = dict(kwargs)
            if seed is not None:
                kw["seed"] = seed
            prob = constructor(dim=dim, **kw)
            return replace(
                prob,
                z0=generate_benchmark_z0(prob.problem),
                name=reg_name,
                dim=dim,
            )
    available = [r[0] for r in _PROBLEM_REGISTRY]
    raise ValueError(f"Unknown problem {name!r}. Available: {available}")


def get_all_problems(
    dims: list[int] | None = None,
    names: list[str] | None = None,
    *,
    seed: int | None = None,
) -> list[BenchmarkProblem]:
    """Return every (problem, dim) combination as a list of BenchmarkProblems.

    Parameters
    ----------
    dims : list[int] or None
        If provided, override default dims for all problems.
        Problems with fixed dims (separable, offset_quadratic) ignore this.
    names : list[str] or None
        If provided, only include these problem names.
    seed : int or None
        Deterministic seed for all problem constructors.  When ``None``,
        each constructor uses its own default seed, UNLESS BENCHMARK_SEED is set.

    Returns
    -------
    list[BenchmarkProblem]
        A list of benchmark problems ready for evaluation.
    """
    from benchmarks import GLOBAL_SEED
    if seed is None and GLOBAL_SEED is not None:
        seed = GLOBAL_SEED

    results = []
    idx = 0
    for reg_name, constructor, default_dims in _PROBLEM_REGISTRY:
        if names is not None and reg_name not in names:
            continue
        target_dims = dims if dims is not None else default_dims
        for dim in target_dims:
            if dim not in default_dims and len(default_dims) == 1:
                continue
            try:
                kw = {}
                if seed is not None:
                    kw["seed"] = seed + idx
                
                # --- Updated Logic ---
                prob = constructor(dim=dim, **kw)
                
                results.append(
                    replace(
                        prob,
                        z0=generate_benchmark_z0(prob.problem),
                        name=reg_name,
                        dim=dim,
                    )
                )
            except Exception as e:
                warnings.warn(f"  [skip] {reg_name} dim={dim}: {e}", RuntimeWarning)
            idx += 1
    return results

def list_problems() -> list[tuple[str, list[int]]]:
    """Return (name, default_dims) for all registered problems."""
    return [(name, dims) for name, _, dims in _PROBLEM_REGISTRY]
