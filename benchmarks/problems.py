"""Problem zoo for benchmarks.

Catalogue of minimax problems at varying dimensions and conditionings.
All problem constructors accept a ``kappa`` parameter for controlled
condition-number sweeps, enabling reproducible scaling-law studies.

Reuses constructors from:
  - ``benchmarks/families.py`` — parametric families (κ, ρ, sparsity control)
  - ``benchmarks/problem_constructors.py`` — legacy constructors (now κ-aware)
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
    make_diagonal_saddle,
    make_logsumexp_saddle,
    make_sparse_bilinear,
    make_random_cubic_quadratic,
    make_adversarial_training_toy,
    make_scalable_diagonal,
    make_nonzero_rho_quadratic,
    make_bilinear_saddle,
    make_quadratic_saddle,
    make_box_constrained_quadratic,
    make_rosenbrock_bilinear,
)

from benchmarks.problem_constructors import (
    make_bilinear_problem,
    make_ill_conditioned_bilinear,
    make_ill_conditioned_quadratic,
    make_offset_quadratic,
    make_quadratic_saddle_problem,
    make_separable_problem,
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
    ("bilinear",            make_bilinear_problem,                                      [2, 5, 10, 20, 50]),
    ("quadratic",           make_quadratic_saddle_problem,                              [2, 5, 10, 20, 50]),
    ("ill_bilinear",        make_ill_conditioned_bilinear,                              [4, 10, 20]),
    ("ill_quadratic",       make_ill_conditioned_quadratic,                             [4, 10, 20]),
    ("separable",           _seed_wrapper(_make_fixed_dim(make_separable_problem, 2), accepts_seed=False),  [2]),
    ("offset_quadratic",    _seed_wrapper(_make_fixed_dim(make_offset_quadratic, 1), accepts_seed=False),  [1]),
    ("nonzero_rho",         make_nonzero_rho_quadratic,                                 [5, 10, 20]),
    ("rosenbrock_bilin",    make_rosenbrock_bilinear,                                   [5, 10, 20]),
    ("diagonal_saddle",     make_diagonal_saddle,                                        [5, 10, 20, 50, 100]),
    ("logsumexp_saddle",    make_logsumexp_saddle,                                      [5, 10, 20, 50]),
    ("sparse_bilinear",     make_sparse_bilinear,                                       [10, 50, 100, 200]),
    ("random_cubic",        make_random_cubic_quadratic,                                [5, 10, 20]),
    ("adversarial_training", make_adversarial_training_toy,                             [5, 10, 20]),
    ("bilinear_polytope",   make_bilinear_polytope,                                     [5, 10, 20]),
    ("scalable_diagonal",   make_scalable_diagonal,                                     [100, 500, 1000, 2000]),
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
