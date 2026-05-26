"""Parametric problem families for reproducible scaling studies.

Every constructor accepts explicit ``kappa``, ``rho``, ``sparsity``, and
``seed`` parameters so that condition number, Hessian Lipschitz constant,
coupling structure, and reproducibility are all under caller control.

Shared helpers
--------------
- ``_log_spaced_eigenvalues(dim, kappa)`` — deterministic eigenvalue spectrum
- ``_banded_mask(dim, bandwidth)`` — banded sparsity pattern
- ``project_box(lo, hi)`` — box constraint projection

Sweep generators
----------------
- ``sweep_kappa(constructor, dim, kappas, ...)`` — κ sweep
- ``sweep_rho(constructor, dim, rho_values, ...)`` — ρ sweep
- ``sweep_sparsity(constructor, dim, sparsity_values, ...)`` — sparsity sweep
"""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe._precision import PROJ_EPS as _PROJ_EPS
from minimax_aipe.problem import BenchmarkProblem, build_benchmark_meta


# ── Shared helpers ──────────────────────────────────────────────────────


def _log_spaced_eigenvalues(dim: int, kappa: float) -> Array:
    """Return *dim* eigenvalues log-spaced from 1 to *kappa*.

    When ``kappa == 1.0`` all eigenvalues are 1 (identity spectrum).
    """
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1.0, got {kappa}")
    if kappa == 1.0:
        return jnp.ones(dim)
    return jnp.logspace(0.0, jnp.log10(kappa), dim)


def _banded_mask(dim: int, bandwidth: int) -> Array:
    """Boolean mask that is True within *bandwidth* of the diagonal.

    ``bandwidth=0`` → diagonal only, ``bandwidth=1`` → tridiagonal, etc.
    """
    idx = jnp.arange(dim)
    return jnp.abs(idx[:, None] - idx[None, :]) <= bandwidth


def _block_banded_mask(dim: int, n_blocks: int, bandwidth: int) -> Array:
    """Boolean mask: block-diagonal structure with banded sub-blocks.

    Divides the dim x dim matrix into ``n_blocks`` diagonal blocks,
    each of which is banded with the given ``bandwidth`` (using
    :func:`_banded_mask` within each block).
    """
    # FIX: Explicit guard rails against division-by-zero, negative sizes, and over-segmentation
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be strictly greater than 0, got {n_blocks}")
    if dim <= 0:
        raise ValueError(f"dim must be strictly greater than 0, got {dim}")
    if n_blocks > dim:
        raise ValueError(f"n_blocks ({n_blocks}) cannot be greater than dim ({dim})")
    if bandwidth < 0:
        raise ValueError(f"bandwidth must be non-negative, got {bandwidth}")

    bs = dim // n_blocks
    remainder = dim - bs * n_blocks
    mask = jnp.zeros((dim, dim), dtype=bool)
    start = 0
    for i in range(n_blocks):
        bsz = bs + (remainder if i == n_blocks - 1 else 0)
        end = start + bsz
        local = _banded_mask(bsz, bandwidth)
        idx = jnp.arange(start, end)
        mask = mask.at[jnp.ix_(idx, idx)].set(local)
        start = end
    return mask

def project_box(lo: float, hi: float) -> Callable[[Array], Array]:
    """Project element-wise onto ``[lo, hi]``."""

    def project(z: Array) -> Array:
        return jnp.clip(z, lo, hi)

    return project

def _make_polytope_projector(
    G: Array, h: Array, n_steps: int = 200, tol: float = 1e-8,
) -> Callable[[Array], Array]:
    """Exact-ish projection onto the polytope ``{z : Gz ≤ h}``.

    Uses accelerated projected gradient descent (FISTA) on the dual
    problem: ``α* = argmin_{α ≥ 0} ½ α^T G G^T α − α^T (Gz₀ − h)``,
    then ``z* = z₀ − G^T α*``.

    Converges in O(√κ · log(1/ε)) iterations instead of O(κ · log(1/ε))
    for plain PGD, where κ = L/μ is the dual condition number.
    """
    GGT = G @ G.T
    L_dual = float(jnp.linalg.norm(GGT, ord=2))
    step = 1.0 / (L_dual + 1e-8)

    def project(z0: Array) -> Array:
        alpha = jnp.zeros(G.shape[0])
        rhs = G @ z0 - h
        y = alpha
        t = 1.0

        def cond(state):
            i, alpha, y, t, change = state
            return (i < n_steps) & (change > tol)

        def body(state):
            i, alpha, y, t, _ = state
            grad = GGT @ y - rhs
            alpha_new = jnp.maximum(y - step * grad, 0.0)
            t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            y = alpha_new + ((t - 1.0) / t_new) * (alpha_new - alpha)
            
            # FIX: Better to evaluate exact primal constraint violation for the stopping condition 
            # to guarantee the polytope projector doesn't return infeasible points.
            primal_viol = jnp.linalg.norm(jnp.maximum(rhs - GGT @ alpha_new, 0.0))
            return (i + 1, alpha_new, y, t_new, primal_viol)

        init_state = (jnp.int32(0), alpha, y, t, jnp.array(1.0 + tol))
        _, alpha, _, _, _ = jax.lax.while_loop(cond, body, init_state)
        return z0 - G.T @ alpha

    return project


# ── Bilinear saddle ────────────────────────────────────────────────────


def make_bilinear_saddle(
    dim: int,
    kappa: float = 1.0,
    bandwidth: int | None = None,
    seed: int = 0,
) -> BenchmarkProblem:
    """Bilinear game f(x,y) = xᵀ A y with controlled κ(A).

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of A.  Singular values are log-spaced in [1, kappa].
    bandwidth : int or None
        If set, A is projected onto a banded structure.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    U, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    sigmas = _log_spaced_eigenvalues(dim, kappa)
    A = U @ jnp.diag(sigmas) @ V.T

    if bandwidth is not None:
        mask = _banded_mask(dim, bandwidth)
        A = jnp.where(mask, A, 0.0)

    # FIX: Compute the operator norm AFTER sparsification so that 
    # step sizes and scaling profiles match the true masked operator.
    ell = float(jnp.linalg.norm(A, ord=2))
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
        meta=build_benchmark_meta(problem, mu_x=0.0, mu_y=0.0, seed=seed),
        name="bilinear_saddle",
        dim=dim,
        z0=None,
    )

# ── Quadratic saddle ───────────────────────────────────────────────────


def make_quadratic_saddle(
    dim: int,
    kappa: float = 1.0,
    coupling_strength: float = 0.3,
    seed: int = 0,
) -> BenchmarkProblem:
    """Full quadratic f(x,y) = ½ xᵀ Q x + xᵀ B y − ½ yᵀ R y.

    Q has eigenvalues log-spaced in [1, kappa].
    R = I (well-conditioned dual side).
    B = random scaled by *coupling_strength*.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of Q.
    coupling_strength : float
        Scale of the random coupling matrix B.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    eigvals = _log_spaced_eigenvalues(dim, kappa)
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(eigvals) @ U_q.T

    R = jnp.eye(dim)
    B = jax.random.normal(k2, (dim, dim)) * coupling_strength

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
    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = float(jnp.min(jnp.linalg.eigvalsh(R)))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y, seed=seed),
        name="quadratic_saddle",
        dim=dim,
        z0=None,
    )


# ── Nonzero-rho quadratic ──────────────────────────────────────────────


def make_nonzero_rho_quadratic(
    dim: int,
    kappa: float = 1.0,
    rho: float = 1.0,
    seed: int = 0,
) -> BenchmarkProblem:
    """Quadratic saddle + cubic perturbation so ρ > 0.

    .. math::

        f(x,y) = \\tfrac{1}{2} x^\\top Q x + x^\\top B y - \\tfrac{1}{2} y^\\top R y
                + \\frac{\\rho}{3}(\\|x\\|^3 - \\|y\\|^3)

    Q, R have eigenvalues controlled by *kappa*.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number controlling eigenvalue spread of Q and R.
    rho : float
        Cubic coefficient for the ‖x‖³ perturbation.
        The actual Hessian Lipschitz constant is ``2 * rho``, computed
        and passed to :class:`MinimaxProblem` automatically.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    eigvals_q = _log_spaced_eigenvalues(dim, kappa)
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(eigvals_q) @ U_q.T

    eigvals_r = _log_spaced_eigenvalues(dim, kappa)
    U_r, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    R = U_r @ jnp.diag(eigvals_r) @ U_r.T

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
        norm_x = jnp.linalg.norm(x)
        norm_y = jnp.linalg.norm(y)
        gx = gx_quad + rho * norm_x * x
        gy_neg = gy_neg_quad + rho * norm_y * y
        return gx, gy_neg

    def hessian_f(x, y):
        norm_x = jnp.linalg.norm(x)
        norm_y = jnp.linalg.norm(y)
        H_cubic_x = rho * (
            norm_x * jnp.eye(dim)
            + jnp.outer(x, x) / jnp.maximum(norm_x, _PROJ_EPS)
        )
        H_cubic_y = rho * (
            norm_y * jnp.eye(dim)
            + jnp.outer(y, y) / jnp.maximum(norm_y, _PROJ_EPS)
        )
        H_xx = Q + H_cubic_x
        H_yy = -R - H_cubic_y
        return ((H_xx, B), (B.T, H_yy))

    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=2 * rho, ell=ell + 2 * rho * D,
    )
    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = float(jnp.min(jnp.linalg.eigvalsh(R)))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y, seed=seed),
        name="nonzero_rho",
        dim=dim,
        z0=None,
    )


# ── Box-constrained quadratic ──────────────────────────────────────────


def make_box_constrained_quadratic(
    dim: int,
    kappa: float = 1.0,
    seed: int = 0,
) -> BenchmarkProblem:
    """Quadratic saddle on box [-1, 1]^n instead of Euclidean ball.

    Same objective as :func:`make_quadratic_saddle` but with box
    projection constraints, exercising non-smooth feasible-set geometry.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of Q.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    eigvals = _log_spaced_eigenvalues(dim, kappa)
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(eigvals) @ U_q.T

    R = jnp.eye(dim)
    B = jax.random.normal(k2, (dim, dim)) * 0.3

    KKT = jnp.block([[Q, B], [B.T, -R]])
    ell = float(jnp.linalg.norm(KKT, ord=2))
    D = 2.0 * math.sqrt(dim)

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
        project_x=project_box(-1.0, 1.0),
        project_y=project_box(-1.0, 1.0),
    )
    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = float(jnp.min(jnp.linalg.eigvalsh(R)))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y, seed=seed),
        name="box_quadratic",
        dim=dim,
        z0=None,
    )

# ── LogSumExp saddle ────────────────────────────────────────────────────

def make_logsumexp_saddle(
    dim: int,
    kappa: float = 1.0,
    seed: int = 0,
) -> BenchmarkProblem:
    """LogSumExp saddle — near-nonsmooth gradient stress test.

    .. math::

        f(x,y) = \\mathrm{logsumexp}(x) + x^\\top A y - \\mathrm{logsumexp}(y)

    The softmax gradient ``softmax(x)`` saturates when one component
    dominates, producing near-nonsmooth behaviour that probes the
    boundary of the gradient-Lipschitz assumption (Assumption 3.4).

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Scale of the coupling matrix A.  Singular values of A are
        log-spaced in ``[1, kappa]``.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Approximate saddle near origin, gap_star = None.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    U, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    sigmas = _log_spaced_eigenvalues(dim, kappa)
    A = U @ jnp.diag(sigmas) @ V.T

    D = 4.0

    def _lse(z):
        m = jnp.max(z)
        return m + jnp.log(jnp.sum(jnp.exp(z - m)))

    def f(x, y):
        return _lse(x) + x @ A @ y - _lse(y)

    def grad_f(x, y):
        ex = jnp.exp(x - jnp.max(x))
        px = ex / jnp.sum(ex)
        ey = jnp.exp(y - jnp.max(y))
        py = ey / jnp.sum(ey)
        return px + A @ y, py - A.T @ x

    def hessian_f(x, y):
        ex = jnp.exp(x - jnp.max(x))
        px = ex / jnp.sum(ex)
        ey = jnp.exp(y - jnp.max(y))
        py = ey / jnp.sum(ey)
        H_xx = jnp.diag(px) - jnp.outer(px, px)
        H_yy = -jnp.diag(py) + jnp.outer(py, py)
        return ((H_xx, A), (A.T, H_yy))

    ell = float(jnp.linalg.norm(A, ord=2)) + 1.0
    rho = 1.0

    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=rho, ell=ell,
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=None,
        y_star=None,
        gap_star=None,
        meta=build_benchmark_meta(problem, mu_x=0.0, mu_y=0.0, seed=seed),
        name="logsumexp_saddle",
        dim=dim,
        z0=None,
    )
# ── Sparse bilinear ────────────────────────────────────────────────────


def make_sparse_bilinear(
    dim: int,
    kappa: float = 1.0,
    n_blocks: int = 2,
    bandwidth: int = 3,
    seed: int = 0,
) -> BenchmarkProblem:
    """Block-banded bilinear game — tests structure exploitation.

    .. math::

        f(x,y) = x^\\top A y

    where ``A`` is block-diagonal with banded sub-blocks.  Dense bilinear
    solvers waste work on the zero pattern; structure-aware solvers can
    be much faster.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number of A (before sparsification).  Singular values
        are log-spaced in ``[1, kappa]``.
    n_blocks : int
        Number of diagonal blocks in the block-banded structure.
    bandwidth : int
        Bandwidth within each diagonal block.
        ``bandwidth=0`` keeps only the block-diagonal entries of A.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    U, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    sigmas = _log_spaced_eigenvalues(dim, kappa)
    A_dense = U @ jnp.diag(sigmas) @ V.T

    # Enforce block-banded sparsity pattern
    mask = _block_banded_mask(dim, n_blocks, bandwidth)
    A = jnp.where(mask, A_dense, 0.0)

    ell = float(jnp.linalg.norm(A, ord=2))
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
        meta=build_benchmark_meta(problem, mu_x=0.0, mu_y=0.0, seed=seed),
        name="sparse_bilinear",
        dim=dim,
        z0=None,
    )

# ── Random cubic quadratic ──────────────────────────────────────────────


def make_random_cubic_quadratic(
    dim: int,
    kappa: float = 1.0,
    rho: float = 1.0,
    seed: int = 0,
) -> BenchmarkProblem:
    """Random per-coordinate cubic perturbation — anisotropic ρ test bed.

    .. math::

        f(x,y) = \\tfrac{1}{2} x^\\top Q x + x^\\top B y
                - \\tfrac{1}{2} y^\\top R y
                + \\frac{\\rho}{3} \\sum_i c_i (|x_i|^3 - |y_i|^3)

    Per-coordinate coefficients ``c_i ~ U[0.5, 1.5]`` create heterogeneous
    curvature, directly testing Theorem 5.5's ρ-dependence beyond the
    isotropic ``‖x‖³`` perturbation in :func:`make_nonzero_rho_quadratic`.

    Varying ``rho`` over ``{0, 0.1, 1, 10, 100}`` sweeps the cubic regime.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number controlling eigenvalue spread of Q and R.
    rho : float
        Cubic coefficient scaling the per-coordinate |x_i|^3 perturbation.
        The actual Hessian Lipschitz constant is ``2 * rho * max(c_i)``,
        computed and passed to :class:`MinimaxProblem` automatically.
        ``rho=0`` recovers a standard quadratic saddle.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    eigvals_q = _log_spaced_eigenvalues(dim, kappa)
    U_q, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    Q = U_q @ jnp.diag(eigvals_q) @ U_q.T

    eigvals_r = _log_spaced_eigenvalues(dim, kappa)
    U_r, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    R = U_r @ jnp.diag(eigvals_r) @ U_r.T

    B = jax.random.normal(k3, (dim, dim)) * 0.3
    c = jax.random.uniform(k4, (dim,), minval=0.5, maxval=1.5)

    KKT = jnp.block([[Q, B], [B.T, -R]])
    ell_quad = float(jnp.linalg.norm(KKT, ord=2))
    D = 4.0
    ell = ell_quad + 2 * rho * D * float(jnp.max(c))

    def f(x, y):
        quad = 0.5 * x @ Q @ x + x @ B @ y - 0.5 * y @ R @ y
        # FIX: Use absolute values to ensure global convex-concave structure
        cubic = (rho / 3.0) * jnp.dot(c, jnp.abs(x) ** 3 - jnp.abs(y) ** 3)
        return quad + cubic

    def grad_f(x, y):
        # FIX: Gradient of |z|^3 / 3 is z * |z|
        gx = Q @ x + B @ y + rho * c * x * jnp.abs(x)
        gy_neg = -(B.T @ x) + R @ y + rho * c * y * jnp.abs(y)
        return gx, gy_neg

    def hessian_f(x, y):
        H_xx = Q + rho * jnp.diag(2.0 * c * jnp.abs(x))
        H_yy = -R - rho * jnp.diag(2.0 * c * jnp.abs(y))
        return ((H_xx, B), (B.T, H_yy))

    rho_hess = 2 * rho * float(jnp.max(c)) if rho > 0 else 0.0
    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=rho_hess, ell=ell,
    )

    mu_x = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y = float(jnp.min(jnp.linalg.eigvalsh(R)))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y, seed=seed),
        name="random_cubic",
        dim=dim,
        z0=None,
    )

# ── Adversarial training toy ────────────────────────────────────────────


def make_adversarial_training_toy(
    dim: int,
    n_samples: int = 20,
    lambda_reg: float = 1.0,
    seed: int = 0,
) -> BenchmarkProblem:
    """Data-dependent quadratic saddle — adversarial training flavour.

    .. math::

        f(\\theta, \\delta)
            = \\tfrac{1}{2} \\theta^\\top Q \\theta
            + \\theta^\\top B \\delta
            - \\tfrac{1}{2} \\delta^\\top R \\delta

    where ``Q = X^T X / n + μ_x I``, ``B = X^T X / n``, and
    ``R = λ I``, with ``X ∈ R^{n × d}`` a synthetic dataset.

    The coupling matrix ``B`` is the data Gram matrix, giving this
    problem the same algebraic structure as a linearised adversarial
    training objective.  The solver must handle data-dependent
    conditioning rather than hand-picked eigenvalue spectra.

    Parameters
    ----------
    dim : int
        Dimension of θ (model) and δ (perturbation).
    n_samples : int
        Number of synthetic training samples.
    lambda_reg : float
        Dual regularisation strength (strong concavity in δ).
    seed : int
        Deterministic seed for data generation.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k_data, _ = jax.random.split(key)

    # ── Synthetic data ────────────────────────────────────────────────
    X = jax.random.normal(k_data, (n_samples, dim))
    Gram = X.T @ X / float(n_samples)

    mu_x = 0.1
    Q = Gram + mu_x * jnp.eye(dim)
    B = Gram
    R = lambda_reg * jnp.eye(dim)

    KKT = jnp.block([[Q, B], [B.T, -R]])
    ell = float(jnp.linalg.norm(KKT, ord=2))
    D = 4.0

    # ── Problem definition ────────────────────────────────────────────

    def f(x, y):
        return 0.5 * x @ Q @ x + x @ B @ y - 0.5 * y @ R @ y

    def grad_f(x, y):
        gx = Q @ x + B @ y
        gy_neg = -(B.T @ x - R @ y)
        return gx, gy_neg

    def hessian_f(x, y):
        return ((Q, B), (B.T, -R))

    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=0.0, ell=ell,
    )
    mu_x_val = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    mu_y_val = float(jnp.min(jnp.linalg.eigvalsh(R)))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x_val, mu_y=mu_y_val, seed=seed),
        name="adversarial_training",
        dim=dim,
        z0=None,
    )

# ── Bilinear polytope ──────────────────────────────────────────────────


def make_bilinear_polytope(
    dim: int,
    n_constraints: int = 10,
    seed: int = 0,
) -> BenchmarkProblem:
    """Bilinear game on random polytopes — projection cost stress test.

    .. math::

        \\min_{x \\in P_x} \\max_{y \\in P_y}  x^\\top A y

    where ``P_x = {x : G_x x ≤ h_x}`` and ``P_y = {y : G_y y ≤ h_y}``
    are random polytopes with ``n_constraints`` facets each.

    Polytope projection requires solving a QP (via dual projected gradient
    descent), making each projection step far more expensive than the
    cheap Euclidean-ball or box projections used by other benchmarks.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    n_constraints : int
        Number of halfspace constraints per player.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin (if feasible), gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    n_constraints = max(n_constraints, 2 * dim)

    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Coupling matrix A with controlled spectrum
    U, _ = jnp.linalg.qr(jax.random.normal(k1, (dim, dim)))
    V, _ = jnp.linalg.qr(jax.random.normal(k2, (dim, dim)))
    sigmas = jnp.ones(dim)  # κ = 1 for simplicity
    A = U @ jnp.diag(sigmas) @ V.T

    # Random polytope constraints: Gz ≤ h, with origin feasible (h > 0)
    G_x = jax.random.normal(k3, (n_constraints, dim))
    G_y = jax.random.normal(k4, (n_constraints, dim))
    h_x = jnp.abs(jax.random.uniform(k5, (n_constraints,))) + 0.5
    k6, _ = jax.random.split(k5)
    h_y = jnp.abs(jax.random.uniform(k6, (n_constraints,))) + 0.5

    project_x = _make_polytope_projector(G_x, h_x)
    project_y = _make_polytope_projector(G_y, h_y)

    ell = float(jnp.linalg.norm(A, ord=2))
    D = 4.0

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
        project_x=project_x,
        project_y=project_y,
    )
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=0.0, mu_y=0.0, seed=seed),
        name="bilinear_polytope",
        dim=dim,
        z0=None,
    )

# ── Scalable diagonal ──────────────────────────────────────────────────


def make_scalable_diagonal(
    dim: int,
    alpha: float = 2.0,
    sparsity: float = 0.05,
    n_blocks: int = 1,
    seed: int = 0,
) -> BenchmarkProblem:
    """Large-scale diagonal saddle with power-law spectrum and sparse coupling.

    .. math::

        f(x,y) = \\sum_i \\tfrac{\\lambda_i}{2} x_i^2
                + \\sum_i \\sigma_i x_i y_i
                - \\sum_i \\tfrac{\\mu_i}{2} y_i^2

    Designed for scaling analysis at ``dim ∈ {100, 500, 1000, 2000}``.
    Differs from :func:`make_diagonal_saddle` in three ways:

    1. **Power-law eigenvalue spectrum**:  λ_i = i^{−α}, modelling the
       heavy-tailed curvature found in real ML problems (vs. the
       artificial log-spacing used elsewhere).
    2. **Sparse random coupling**:  only ``sparsity`` fraction of
       coordinates are coupled, with each active coupling drawn from
       a Bernoulli-then-uniform scheme.  This creates realistic
       sparsity in the operator ``F(z)``.
    3. **Optional multi-block conditioning**:  ``n_blocks`` groups of
       coordinates share a common eigenvalue scale, modelling
       heterogeneous difficulty across parameter subspaces.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    alpha : float
        Power-law exponent for eigenvalue decay.  ``α=0`` gives uniform
        eigenvalues; ``α=2`` gives aggressive decay (many near-zero
        eigenvalues, a few large ones).
    sparsity : float
        Fraction of coordinates that are zeroes (``0 ≤ sparsity ≤ 1``).
        ``sparsity=0.0`` recovers dense coupling, ``sparsity=1.0`` is fully decoupled.
    n_blocks : int
        Number of conditioning blocks.  Each block gets a separate
        eigenvalue rescaling factor ``β_b`` drawn uniformly from
        ``[0.5, 2.0]``, creating heterogeneous curvature across
        coordinate groups.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k_eig, k_coupling, k_block, k1, k2 = jax.random.split(key, 5)

    # ── 1. Power-law eigenvalue spectrum ──────────────────────────────
    idx = jnp.arange(1, dim + 1, dtype=jnp.float32)
    lam = idx ** (-alpha) + 1e-3          # regularise away from zero
    mu  = idx ** (-alpha) + 1e-3
    lam = lam / lam[0]                    # normalise so max = 1
    mu  = mu / mu[0]

    # ── 2. Block rescaling for heterogeneous curvature ───────────────
    if n_blocks > 1:
        # Save original log-range (= log condition number) before scaling.
        log_range_lam = float(jnp.log(jnp.max(lam)) - jnp.log(jnp.min(lam)))
        log_range_mu  = float(jnp.log(jnp.max(mu))  - jnp.log(jnp.min(mu)))

        bs = dim // n_blocks
        block_ids = jnp.arange(dim) // bs
        block_ids = jnp.minimum(block_ids, n_blocks - 1)
        block_scale = jax.random.uniform(k_block, (n_blocks,), minval=0.5, maxval=2.0)
        lam = lam * block_scale[block_ids]
        mu  = mu  * block_scale[block_ids]

        # Re-normalise in log-space so max/min ratio (condition number)
        # equals the original power-law dynamic range.
        def _restore_condition_number(eigenvalues, log_range):
            log_e = jnp.log(eigenvalues)
            log_e = log_e - jnp.min(log_e)
            log_e = log_e / jnp.maximum(jnp.max(log_e), 1e-30) * log_range
            return jnp.exp(log_e)

        lam = _restore_condition_number(lam, log_range_lam)
        mu  = _restore_condition_number(mu,  log_range_mu)

    # ── 3. Sparse random coupling ────────────────────────────────────
    p_active = jnp.clip(1.0 - sparsity, 0.0, 1.0)
    mask = jax.random.bernoulli(k_coupling, p=p_active, shape=(dim,))
    raw_sigma = jax.random.uniform(k1, (dim,), minval=-1.0, maxval=1.0)
    sigma = jnp.where(mask, raw_sigma, 0.0)
    nnz = int(jnp.sum(mask))

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
    mu_x = float(jnp.min(lam))
    mu_y = float(jnp.min(mu))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y, sparsity=sparsity, seed=seed),
        name="scalable_diagonal",
        dim=dim,
        z0=None,
    )


# ── Diagonal saddle (κ-parameterized) ─────────────────────────────────


def make_diagonal_saddle(
    dim: int,
    kappa: float = 1e4,
    rho: float = 0.0,
    sparsity: float = 1.0,
    n_blocks: int = 1,
    seed: int = 0,
) -> BenchmarkProblem:
    """Diagonal quadratic saddle with explicit κ, ρ, and sparsity control.

    .. math::

        f(x,y) = \\sum_i \\tfrac{\\lambda_i}{2} x_i^2
                + \\sum_i \\sigma_i x_i y_i
                - \\sum_i \\tfrac{\\mu_i}{2} y_i^2
                + \\frac{\\rho}{3}(\\|x\\|^3 - \\|y\\|^3)

    The primary parametric diagonal family for scaling-law studies.
    All matrices are diagonal → O(n) per oracle call.

    Parameters
    ----------
    dim : int
        Dimension of x and y.
    kappa : float
        Condition number.  Eigenvalues ``λ_i, μ_i`` are log-spaced in
        ``[1, kappa]``.  ``kappa=1`` gives uniform spectrum.
    rho : float
        Hessian Lipschitz constant.  ``rho=0`` (default) gives a pure
        quadratic with ``ρ = 0``.  ``rho > 0`` adds an isotropic cubic
        perturbation ``(ρ/3)(‖x‖³ − ‖y‖³)``.
    sparsity : float
        Fraction of zeros in the coupling matrix (``0 ≤ sparsity ≤ 1``).
        ``sparsity=0.0`` couples all coordinates (fully dense coupling).
        ``sparsity=1.0`` couples zero coordinates (fully decoupled).
    n_blocks : int
        Number of conditioning blocks for heterogeneous curvature.
        Each block gets an independent eigenvalue rescaling factor
        drawn uniformly from ``[0.5, 2.0]``.
    seed : int
        Deterministic seed.

    Returns
    -------
    BenchmarkProblem
        Saddle at origin, gap = 0.
    """
    from minimax_aipe import MinimaxProblem

    key = jax.random.PRNGKey(seed)
    k_eig_x, k_eig_y, k_coupling, k_block = jax.random.split(key, 4)

    # ── 1. Log-spaced eigenvalue spectrum controlled by κ ─────────────
    lam = _log_spaced_eigenvalues(dim, kappa)
    mu = _log_spaced_eigenvalues(dim, kappa)

    # ── 2. Block rescaling for heterogeneous curvature ───────────────
    if n_blocks > 1:
        log_range = float(jnp.log(kappa)) if kappa > 1.0 else 0.0

        bs = dim // n_blocks
        block_ids = jnp.arange(dim) // bs
        block_ids = jnp.minimum(block_ids, n_blocks - 1)
        block_scale = jax.random.uniform(k_block, (n_blocks,), minval=0.5, maxval=2.0)
        lam = lam * block_scale[block_ids]
        mu = mu * block_scale[block_ids]

        # Re-normalise in log-space so max/min ratio equals exactly κ.
        if log_range > 0:
            def _restore_kappa(eigenvalues, target_log_range):
                log_e = jnp.log(eigenvalues)
                log_e = log_e - jnp.min(log_e)
                log_e = log_e / jnp.max(log_e) * target_log_range
                return jnp.exp(log_e)

            lam = _restore_kappa(lam, log_range)
            mu  = _restore_kappa(mu,  log_range)

    # ── 3. Coupling coefficients with optional sparsity ──────────────
    p_active = jnp.clip(1.0 - sparsity, 0.0, 1.0)
    k_mask, k_val = jax.random.split(k_coupling)
    mask = jax.random.bernoulli(k_mask, p=p_active, shape=(dim,))
    raw_sigma = jax.random.uniform(k_val, (dim,), minval=-1.0, maxval=1.0)
    sigma = jnp.where(mask, raw_sigma, 0.0)

    # ── Constants ─────────────────────────────────────────────────────
    ell_quad = float(jnp.max(jnp.concatenate([lam + jnp.abs(sigma),
                                               mu + jnp.abs(sigma)])))
    D = 4.0
    ell = ell_quad + 2 * rho * D if rho > 0 else ell_quad

    # ── Problem definition ────────────────────────────────────────────

    def f(x, y):
        quad = (
            0.5 * jnp.dot(lam, x ** 2)
            + jnp.dot(sigma, x * y)
            - 0.5 * jnp.dot(mu, y ** 2)
        )
        if rho == 0.0:
            return quad
        cubic_x = (rho / 3.0) * jnp.linalg.norm(x) ** 3
        cubic_y = (rho / 3.0) * jnp.linalg.norm(y) ** 3
        return quad + cubic_x - cubic_y

    def grad_f(x, y):
        gx = lam * x + sigma * y
        gy_neg = mu * y - sigma * x
        if rho > 0.0:
            norm_x = jnp.linalg.norm(x)
            norm_y = jnp.linalg.norm(y)
            gx = gx + rho * norm_x * x
            gy_neg = gy_neg + rho * norm_y * y
        return gx, gy_neg

    def hessian_f(x, y):
        H_xx = jnp.diag(lam)
        H_xy = jnp.diag(sigma)
        H_yx = jnp.diag(sigma)
        H_yy = jnp.diag(-mu)
        if rho > 0.0:
            norm_x = jnp.linalg.norm(x)
            norm_y = jnp.linalg.norm(y)
            H_xx = H_xx + rho * (
                norm_x * jnp.eye(dim)
                + jnp.outer(x, x) / jnp.maximum(norm_x, _PROJ_EPS)
            )
            H_yy = H_yy - rho * (
                norm_y * jnp.eye(dim)
                + jnp.outer(y, y) / jnp.maximum(norm_y, _PROJ_EPS)
            )
        return ((H_xx, H_xy), (H_yx, H_yy))

    problem = MinimaxProblem(
        f=f, dim_x=dim, dim_y=dim, D_x=D, D_y=D,
        grad_f=grad_f, hessian_f=hessian_f,
        rho=2 * rho, ell=ell,
    )

    mu_x = float(jnp.min(lam))
    mu_y = float(jnp.min(mu))
    return BenchmarkProblem(
        problem=problem,
        x_star=jnp.zeros(dim),
        y_star=jnp.zeros(dim),
        gap_star=0.0,
        meta=build_benchmark_meta(problem, mu_x=mu_x, mu_y=mu_y, sparsity=sparsity, seed=seed),
        name="diagonal_saddle",
        dim=dim,
        z0=None,
    )


# ── Sweep generators ─────────────────────────────────────────────────


def sweep_kappa(
    constructor: Callable[..., BenchmarkProblem],
    dim: int,
    kappas: list[float],
    seed: int = 0,
    **fixed_kwargs: Any,
) -> list[BenchmarkProblem]:
    """Generate problems sweeping κ at fixed dimension and other params.

    Parameters
    ----------
    constructor : callable
        A problem constructor (e.g. :func:`make_diagonal_saddle`).
    dim : int
        Problem dimension.
    kappas : list[float]
        κ values to sweep.
    seed : int
        Base seed (incremented per instance for diversity).
    **fixed_kwargs
        Additional keyword arguments forwarded to *constructor*
        (e.g. ``rho=0.0``, ``sparsity=0.1``).

    Returns
    -------
    list[BenchmarkProblem]
    """
    return [constructor(dim=dim, kappa=k, seed=seed + i, **fixed_kwargs)
            for i, k in enumerate(kappas)]


def sweep_rho(
    constructor: Callable[..., BenchmarkProblem],
    dim: int,
    rho_values: list[float],
    seed: int = 0,
    **fixed_kwargs: Any,
) -> list[BenchmarkProblem]:
    """Generate problems sweeping ρ at fixed dimension and other params.

    Parameters
    ----------
    constructor : callable
        A problem constructor (e.g. :func:`make_diagonal_saddle`).
    dim : int
        Problem dimension.
    rho_values : list[float]
        ρ values to sweep.
    seed : int
        Base seed.
    **fixed_kwargs
        Additional keyword arguments forwarded to *constructor*.

    Returns
    -------
    list[BenchmarkProblem]
    """
    return [constructor(dim=dim, rho=r, seed=seed + i, **fixed_kwargs)
            for i, r in enumerate(rho_values)]


def sweep_sparsity(
    constructor: Callable[..., BenchmarkProblem],
    dim: int,
    sparsity_values: list[float],
    seed: int = 0,
    **fixed_kwargs: Any,
) -> list[BenchmarkProblem]:
    """Generate problems sweeping sparsity at fixed dimension and other params.

    Parameters
    ----------
    constructor : callable
        A problem constructor (e.g. :func:`make_diagonal_saddle`).
    dim : int
        Problem dimension.
    sparsity_values : list[float]
        Sparsity values to sweep (0 = dense, 1 = fully sparse).
    seed : int
        Base seed.
    **fixed_kwargs
        Additional keyword arguments forwarded to *constructor*.

    Returns
    -------
    list[BenchmarkProblem]
    """
    return [constructor(dim=dim, sparsity=s, seed=seed + i, **fixed_kwargs)
            for i, s in enumerate(sparsity_values)]
