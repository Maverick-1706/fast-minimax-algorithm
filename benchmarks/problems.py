"""Problem zoo for benchmarks.

Catalogue of minimax problems at varying dimensions and conditionings.
Reuses constructors from tests/conftest.py and adds benchmark-specific variants.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

# ── Import existing constructors from tests/conftest.py ──────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import (  # noqa: E402
    make_bilinear_problem,
    make_ill_conditioned_bilinear,
    make_ill_conditioned_quadratic,
    make_offset_quadratic,
    make_quadratic_saddle_problem,
    make_separable_problem,
)


# ── New problem constructors ─────────────────────────────────────────────


def make_nonzero_rho_quadratic(dim: int = 5, rho: float = 1.0, seed: int = 0) -> dict:
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
        H_cubic_x = rho * (norm_x * jnp.eye(dim) + jnp.outer(x, x) / jnp.maximum(norm_x, 1e-12))
        H_cubic_y = rho * (norm_y * jnp.eye(dim) + jnp.outer(y, y) / jnp.maximum(norm_y, 1e-12))
        H_xx = Q + H_cubic_x
        H_yy = -R - H_cubic_y
        return ((H_xx, B), (B.T, H_yy))

    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=rho, ell=ell + rho * D,
    )
    return dict(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
    )


def make_rosenbrock_bilinear(dim: int = 5, seed: int = 0) -> dict:
    """Rosenbrock-style objective in x + bilinear coupling with y.

    f(x,y) = Σ_i [100(x_{i+1} − x_i²)² + (1 − x_i)²] + xᵀ A y

    The Rosenbrock part is convex (but ill-conditioned) in x; the bilinear
    term couples x and y.  ρ > 0 due to the non-quadratic x terms.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (dim, dim)) * 0.5

    D = 4.0

    def f(x, y):
        # Rosenbrock in x (shifted so minimum is near origin inside ball)
        x_shifted = x + 1.0  # minimum of Rosenbrock is at (1,1,...,1)
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
        rho=200.0,   # Hessian Lipschitz from Rosenbrock curvature
        ell=1e4,      # gradient Lipschitz (Rosenbrock is stiff)
    )
    return dict(
        problem=problem,
        x_star=jnp.zeros(dim),  # approximate — true saddle is hard to compute
        y_star=jnp.zeros(dim),
        gap_star=None,  # unknown analytically
    )


# ── Problem registry ─────────────────────────────────────────────────────


def _make_fixed_dim(constructor: Callable, dim: int):
    """Wrap a no-dim constructor so it accepts a dim argument (ignored)."""
    def wrapper(_dim=None, **kwargs):
        return constructor(**kwargs)
    return wrapper


_PROBLEM_REGISTRY: list[tuple[str, Callable, list[int]]] = [
    ("bilinear",         make_bilinear_problem,          [2, 5, 10, 20, 50]),
    ("quadratic",        make_quadratic_saddle_problem,  [2, 5, 10, 20, 50]),
    ("ill_bilinear",     lambda dim=4, **kw: make_ill_conditioned_bilinear(dim=dim, **kw),  [4, 10, 20]),
    ("ill_quadratic",    lambda dim=4, **kw: make_ill_conditioned_quadratic(dim=dim, **kw), [4, 10, 20]),
    ("separable",        _make_fixed_dim(make_separable_problem, 2),  [2]),
    ("offset_quadratic", _make_fixed_dim(make_offset_quadratic, 1),  [1]),
    ("nonzero_rho",      make_nonzero_rho_quadratic,     [5, 10, 20]),
    ("rosenbrock_bilin", make_rosenbrock_bilinear,       [5, 10, 20]),
]


# ── Public API ───────────────────────────────────────────────────────────


def get_problem(name: str, dim: int, **kwargs) -> dict:
    """Return a single problem by name and dimension.

    Parameters
    ----------
    name : str
        Problem name (see _PROBLEM_REGISTRY).
    dim : int
        Problem dimension.
    **kwargs
        Extra arguments forwarded to the constructor (e.g. seed, rho).

    Returns
    -------
    dict
        {problem, x_star, y_star, gap_star}
    """
    for reg_name, constructor, _ in _PROBLEM_REGISTRY:
        if reg_name == name:
            return constructor(dim=dim, **kwargs)
    available = [r[0] for r in _PROBLEM_REGISTRY]
    raise ValueError(f"Unknown problem {name!r}. Available: {available}")


def get_all_problems(
    dims: list[int] | None = None,
    names: list[str] | None = None,
) -> list[dict]:
    """Return every (problem, dim) combination as a list of dicts.

    Parameters
    ----------
    dims : list[int] or None
        If provided, override default dims for all problems.
        Problems with fixed dims (separable, offset_quadratic) ignore this.
    names : list[str] or None
        If provided, only include these problem names.

    Returns
    -------
    list[dict]
        Each dict has keys: name, dim, problem, x_star, y_star, gap_star.
    """
    results = []
    for reg_name, constructor, default_dims in _PROBLEM_REGISTRY:
        if names is not None and reg_name not in names:
            continue
        target_dims = dims if dims is not None else default_dims
        for dim in target_dims:
            # Skip dims that don't match fixed-dim problems
            if dim not in default_dims and len(default_dims) == 1:
                continue
            try:
                prob_dict = constructor(dim=dim)
                prob_dict["name"] = reg_name
                prob_dict["dim"] = dim
                results.append(prob_dict)
            except Exception as e:
                print(f"  [skip] {reg_name} dim={dim}: {e}")
    return results


def list_problems() -> list[tuple[str, list[int]]]:
    """Return (name, default_dims) for all registered problems."""
    return [(name, dims) for name, _, dims in _PROBLEM_REGISTRY]
