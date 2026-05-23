"""Core data types for minimax optimization problems.

References
----------
Problem (1) from the paper:
    min_{x in X} max_{y in Y} f(x, y)
where X, Y are convex and compact sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import jax.numpy as jnp
from jax import Array

from minimax_aipe._precision import PROJ_EPS as _PROJ_EPS


@dataclass
class OracleStats:
    """Standardized oracle call accounting for complexity validation.

    Tracks all second-order oracle operations.  The paper's primary
    metric is second-order oracle complexity — the number of Hessian
    evaluations and CRN subproblem solves.

    Attributes
    ----------
    grad_calls : int
        First-order (gradient / operator_F) evaluations.
    hessian_calls : int
        Full Hessian evaluations (∇²f).
    hvp_calls : int
        Hessian-vector product evaluations.
    crn_calls : int
        Cubic-regularised Newton subproblem solves.
    projection_calls : int
        Feasible-set projections.
    linear_solves : int
        Linear system solves (inside CRN secular equation).
    fn_evals : int
        Output-selection ``vmap(fn)`` evaluations.  Each evaluation may
        involve an inner first-order solve (e.g. gradient ascent for
        ``Φ`` or descent for ``−Ψ``), so these can dominate runtime.
    """

    grad_calls: int = 0
    hessian_calls: int = 0
    hvp_calls: int = 0
    crn_calls: int = 0
    projection_calls: int = 0
    linear_solves: int = 0
    oracle_calls: int = 0  # Unified metric: CRN for 2nd order, grad for 1st order
    call_type: str = "crn" # Type of the primary metric (crn, gradient, etc)
    fn_evals: int = 0      # Output-selection vmap(fn) evaluations

    def normalized_cost(self, dim: int) -> float:
        """FLOP-equivalent work units.

        Costs: grad=d, hessian=d², hvp=d², crn=d³, projection=d,
        fn_eval=d (same as grad).

        ``linear_solves`` are **not** separately costed because they are
        a sub-operation inside each CRN call; counting both
        ``crn_calls * d³`` and ``linear_solves * d³`` would double-count
        the dominant d³ term.
        """
        return (
            self.grad_calls * dim
            + self.hessian_calls * dim * dim
            + self.hvp_calls * dim * dim
            + self.crn_calls * dim * dim * dim
            + self.projection_calls * dim
            + self.fn_evals * dim
        )

    def __add__(self, other: OracleStats) -> OracleStats:
        if not isinstance(other, OracleStats):
            return NotImplemented
        return OracleStats(
            grad_calls=self.grad_calls + other.grad_calls,
            hessian_calls=self.hessian_calls + other.hessian_calls,
            hvp_calls=self.hvp_calls + other.hvp_calls,
            crn_calls=self.crn_calls + other.crn_calls,
            projection_calls=self.projection_calls + other.projection_calls,
            linear_solves=self.linear_solves + other.linear_solves,
            oracle_calls=self.oracle_calls + other.oracle_calls,
            call_type=self.call_type,
            fn_evals=self.fn_evals + other.fn_evals,
        )

    def __radd__(self, other) -> OracleStats:
        if other == 0:
            return self
        return self.__add__(other)

    def to_dict(self) -> dict[str, int | str]:
        return {
            "grad_calls": self.grad_calls,
            "hessian_calls": self.hessian_calls,
            "hvp_calls": self.hvp_calls,
            "crn_calls": self.crn_calls,
            "projection_calls": self.projection_calls,
            "linear_solves": self.linear_solves,
            "oracle_calls": self.oracle_calls,
            "call_type": self.call_type,
            "fn_evals": self.fn_evals,
        }

    def __repr__(self) -> str:
        return (
            f"OracleStats(grad={self.grad_calls}, hessian={self.hessian_calls}, "
            f"hvp={self.hvp_calls}, crn={self.crn_calls}, "
            f"proj={self.projection_calls}, linear={self.linear_solves}, "
            f"oracle_calls={self.oracle_calls}, fn_evals={self.fn_evals})"
        )


class MinimaxProblem:
    """A convex-concave minimax problem min_x max_y f(x, y).

    Parameters
    ----------
    f : callable
        Scalar function f(x, y) -> float. Must be JAX-traceable.
    grad_f : callable, optional
        Gradient function (x, y) -> (grad_x f, -grad_y f).
        If None, computed via ``jax.grad``.
    hessian_f : callable, optional
        Hessian function (x, y) -> Hessian matrix of f w.r.t. (x, y).
        If None, computed via ``jax.hessian``.
    dim_x : int
        Dimension of x.
    dim_y : int
        Dimension of y.
    D_x : float
        Diameter of the feasible set X.
    D_y : float
        Diameter of feasible set Y.
    rho : float, optional
        Hessian Lipschitz constant (Assumption 3.5).
    ell : float, optional
        Gradient Lipschitz constant (Assumption 3.4).
    ell_x : float, optional
        Gradient Lipschitz constant for x. Defaults to ``ell``.
    ell_y : float, optional
        Gradient Lipschitz constant for y. Defaults to ``ell``.
    L : float, optional
        Function Lipschitz constant (Assumption 3.3).
    """

    def __init__(
        self,
        f: Callable[[Array, Array], float],
        dim_x: int,
        dim_y: int,
        D_x: float,
        D_y: float,
        *,
        grad_f: Optional[Callable[[Array, Array], tuple[Array, Array]]] = None,
        hessian_f: Optional[Callable[[Array, Array], Array]] = None,
        rho: Optional[float] = None,
        ell: Optional[float] = None,
        ell_x: Optional[float] = None,
        ell_y: Optional[float] = None,
        L: Optional[float] = None,
        project_x: Optional[Callable[[Array], Array]] = None,
        project_y: Optional[Callable[[Array], Array]] = None,
    ):
        # ── Input validation ──────────────────────────────────────────────
        if dim_x <= 0 or dim_y <= 0:
            raise ValueError(
                f"Dimensions must be positive, got dim_x={dim_x}, dim_y={dim_y}"
            )
        if D_x <= 0 or D_y <= 0:
            raise ValueError(
                f"Diameters must be positive, got D_x={D_x}, D_y={D_y}"
            )

        self.f = f
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.D_x = D_x
        self.D_y = D_y
        self.rho = rho
        self.ell = ell
        self.ell_x = ell_x if ell_x is not None else ell
        self.ell_y = ell_y if ell_y is not None else ell
        self.L = L

        # Projections onto X and Y (default: Euclidean ball of radius D/2)
        self.project_x = project_x or self._default_project(D_x)
        self.project_y = project_y or self._default_project(D_y)

        # Auto-differentiation if not provided
        if grad_f is None:
            import jax

            _grad_x = jax.grad(f, argnums=0)
            _grad_y = jax.grad(f, argnums=1)

            def _grad_f(x, y):
                return _grad_x(x, y), -_grad_y(x, y)

            self._grad_f_tuple = _grad_f
        else:
            self._grad_f_tuple = grad_f

        if hessian_f is None:
            import jax

            self.hessian_f = jax.hessian(f, argnums=(0, 1))
        else:
            self.hessian_f = hessian_f

        # ── Shape and Convex-Concavity Validation ────────────────────────
        _test_x = jnp.zeros(dim_x)
        _test_y = jnp.zeros(dim_y)

        if grad_f is not None:
            _gx, _gy = grad_f(_test_x, _test_y)
            if _gx.shape != (dim_x,):
                raise ValueError(
                    f"grad_f first component (grad_x f) has shape {_gx.shape}, "
                    f"expected ({dim_x},)"
                )
            if _gy.shape != (dim_y,):
                raise ValueError(
                    f"grad_f second component (-grad_y f) has shape {_gy.shape}, "
                    f"expected ({dim_y},)"
                )

        try:
            H = self.hessian_f(_test_x, _test_y)
            H_xx = H[0][0]
            H_yy = H[1][1]
            
            eig_xx = jnp.linalg.eigvalsh((H_xx + H_xx.T) / 2.0)
            min_eig_xx = float(jnp.min(eig_xx))
            if min_eig_xx < -1e-5:
                import warnings
                warnings.warn(
                    f"Problem may not be convex in x. Min eigenvalue of H_xx at origin is {min_eig_xx:.2e}."
                )
                
            eig_yy = jnp.linalg.eigvalsh((H_yy + H_yy.T) / 2.0)
            max_eig_yy = float(jnp.max(eig_yy))
            if max_eig_yy > 1e-5:
                import warnings
                warnings.warn(
                    f"Problem may not be concave in y. Max eigenvalue of H_yy at origin is {max_eig_yy:.2e}."
                )
        except Exception as e:
            if "Tracer" in str(type(e)) or "tracer" in str(e).lower():
                pass # Ignore JAX tracing concretization errors
            else:
                import warnings
                warnings.warn(f"Could not validate convex-concavity at origin: {e}")

    def grad_f(self, x: Array, y: Array) -> tuple[Array, Array]:
        """Return (∇_x f, -∇_y f) as a tuple."""
        return self._grad_f_tuple(x, y)

    def operator_F(self, z: Array) -> Array:
        """Return F(z) = [∇_x f, -∇_y f] as a single concatenated array.

        This is the monotone operator from Equation (2).
        """
        x, y = z[: self.dim_x], z[self.dim_x :]
        gx, gy_neg = self._grad_f_tuple(x, y)
        return jnp.concatenate([gx, gy_neg])

    @staticmethod
    def _default_project(D: float) -> Callable[[Array], Array]:
        def project(z: Array) -> Array:
            norm = jnp.linalg.norm(z)
            return jnp.where(norm <= D / 2, z, z * (D / 2) / (norm + _PROJ_EPS))

        return project

    def duality_gap(self, x: Array, y: Array) -> float:
        """Compute the duality gap Gap(x, y) = max_y' f(x, y') - min_x' f(x', y).

        This is Definition 3.1 from the paper. For practical computation
        on small problems, this uses a grid or solver; subclasses should
        override for exact computation.
        """
        raise NotImplementedError(
            "Exact duality gap requires solving inner min/max. "
            "Override this method or use gap.estimate_gap()."
        )

@dataclass(frozen=True)
class BenchmarkMeta:
    """Metadata describing a benchmark problem."""

    rho: float | None
    """Hessian Lipschitz constant."""
    ell: float | None
    """Gradient Lipschitz constant."""
    kappa: float | None
    """Condition number estimate."""
    D_x: float
    """Diameter of primal feasible set."""
    D_y: float
    """Diameter of dual feasible set."""
    mu_x: float | None
    """Strong convexity constant in x."""
    mu_y: float | None
    """Strong concavity constant in y."""
    has_analytical_solution: bool
    """Whether a closed-form saddle point is known."""
    sparsity: float | None = None
    """Fraction of coupled coordinates (0 = dense, 1 = fully sparse)."""
    seed: int | None = None
    """Deterministic seed used when constructing the problem."""

    @property
    def strongly_convex_concave(self) -> bool:
        return (
            self.mu_x is not None
            and self.mu_y is not None
            and self.mu_x > 0
            and self.mu_y > 0
        )

@dataclass(frozen=True)
class BenchmarkProblem:
    """Typed benchmark problem container."""

    problem: MinimaxProblem
    x_star: Array | None
    y_star: Array | None
    gap_star: float | None
    meta: BenchmarkMeta
    z0: Array | None = None
    name: str | None = None
    dim: int | None = None

    def __getitem__(self, key):
        """Allow dict-style access for backward compatibility."""
        return getattr(self, key)

def build_benchmark_meta(
    problem,
    *,
    mu_x: float | None,
    mu_y: float | None,
    has_analytical_solution: bool = True,
    sparsity: float | None = None,
    seed: int | None = None,
) -> BenchmarkMeta:
    """Construct standardized benchmark metadata."""

    positive_mus = [
        m for m in (mu_x, mu_y)
        if m is not None and m > 0
    ]

    if positive_mus and problem.ell is not None:
        kappa = float(problem.ell) / min(positive_mus)
    else:
        kappa = None

    return BenchmarkMeta(
        rho=problem.rho,
        ell=problem.ell,
        kappa=kappa,
        D_x=problem.D_x,
        D_y=problem.D_y,
        mu_x=mu_x,
        mu_y=mu_y,
        has_analytical_solution=has_analytical_solution,
        sparsity=sparsity,
        seed=seed,
    )

class SolverResult(NamedTuple):
    """Result returned by a minimax solver.

    Attributes
    ----------
    x : Array
        Approximate primal solution x̂.
    y : Array
        Approximate dual solution ŷ.
    gap : float
        Estimated duality gap Gap(x̂, ŷ).
    iterations : int
        Total number of outer-loop iterations.
    oracle_calls : int
        Total number of second-order (CRN) oracle calls.
    oracle_stats : OracleStats
        Detailed per-type oracle call counts.
    converged : bool
        Whether the solver converged to the requested tolerance.
    history : dict
        Optional logging data (gap values, iterates, etc.).
    """

    x: Array
    y: Array
    gap: float
    iterations: int
    oracle_calls: int
    converged: bool
    oracle_stats: Optional[OracleStats] = None
    history: Optional[dict] = None

__all__ = [
    "MinimaxProblem",
    "OracleStats",
    "SolverResult",
    "BenchmarkMeta",
    "BenchmarkProblem",
    "build_benchmark_meta",
]
