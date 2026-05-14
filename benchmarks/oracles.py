"""Oracle call counting hook.

The paper's primary metric is second-order oracle complexity — the number of
Hessian evaluations and cubic-regularised Newton (CRN) subproblem solves.

This module provides a lightweight counter that can be threaded through any
solver.  The Minimax-AIPE solver already exposes ``result.oracle_calls`` for
CRN calls; this module adds gradient-only counting for the first-order
baselines (EG, GDA) so that all solvers report a common cost metric.

Usage::

    counter = OracleCounter()
    # ... run solver, calling counter.tick_hessian(), counter.tick_grad(), etc. ...
    print(counter)

For the main solver, ``result.oracle_calls`` is the CRN count.  For baselines,
we count gradient evaluations (each EG step = 2 F-evaluations = 2 gradient
calls per player = 4 total; each GDA step = 1 per player = 2 total).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OracleCounter:
    """Counts oracle invocations by type.

    Attributes
    ----------
    grad_calls : int
        First-order (gradient) evaluations.
    hessian_calls : int
        Second-order (Hessian) evaluations.
    crn_calls : int
        Cubic-regularised Newton subproblem solves.
    linear_solves : int
        Linear system solves (inside CRN).
    """

    grad_calls: int = 0
    hessian_calls: int = 0
    crn_calls: int = 0
    linear_solves: int = 0

    def tick_grad(self, n: int = 1) -> None:
        self.grad_calls += n

    def tick_hessian(self, n: int = 1) -> None:
        self.hessian_calls += n

    def tick_crn(self, n: int = 1) -> None:
        self.crn_calls += n

    def tick_linear_solve(self, n: int = 1) -> None:
        self.linear_solves += n

    @property
    def total(self) -> int:
        """Total oracle calls (all types)."""
        return self.grad_calls + self.hessian_calls + self.crn_calls

    def to_dict(self) -> dict[str, int]:
        return {
            "grad_calls": self.grad_calls,
            "hessian_calls": self.hessian_calls,
            "crn_calls": self.crn_calls,
            "linear_solves": self.linear_solves,
            "total": self.total,
        }

    def __repr__(self) -> str:
        return (
            f"OracleCounter(grad={self.grad_calls}, hessian={self.hessian_calls}, "
            f"crn={self.crn_calls}, linear={self.linear_solves}, total={self.total})"
        )


def count_eg_oracles(n_iters: int) -> OracleCounter:
    """Oracle cost of *n_iters* extragradient steps.

    Each EG step evaluates F(z) and F(z_half).  Each F evaluation requires
    one gradient call per player (2 total).  So per iteration: 2 F-evals
    = 4 gradient calls.
    """
    return OracleCounter(grad_calls=4 * n_iters)


def count_gda_oracles(n_iters: int) -> OracleCounter:
    """Oracle cost of *n_iters* GDA steps.

    Each GDA step evaluates ∇_x f and ∇_y f once = 2 gradient calls.
    """
    return OracleCounter(grad_calls=2 * n_iters)


def count_solver_oracles(result) -> OracleCounter:
    """Extract oracle counts from a SolverResult.

    The main solver reports ``oracle_calls`` which counts CRN subproblem
    solves.  We map that directly to ``crn_calls``.
    """
    return OracleCounter(crn_calls=getattr(result, "oracle_calls", 0))
