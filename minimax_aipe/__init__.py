"""Minimax-AIPE: Accelerated second-order methods for convex-concave minimax problems.

Solves min_x max_y f(x, y) to ε-accuracy with Õ(ε^{-4/7}) second-order
oracle complexity via a triple-loop reduction (Chen, Liu, Luo & Zhang, 2025).

Modules
-------
problem       Core types: MinimaxProblem, SolverResult.
framework     Full triple-loop solver (Algorithms 3–5) + RegularizedSubproblem.
npe           Newton Proximal Extragradient (Algorithms 6–7).
len           Lazy Extra Newton (Algorithms 8–9).
aipe          Accelerated Inexact Proximal Extragradient (Algorithms 1–2).
alen          Accelerated Lazy Extra Newton (lazy-AIPE variant).
oracles       Cubic-regularised Newton oracles + extragradient step.
operators     Monotone operator F(z) = [∇_x f, −∇_y f] construction.
gap           Duality gap estimation.
"""

from minimax_aipe.problem import MinimaxProblem, SolverResult

from minimax_aipe.framework import (
    RegularizedSubproblem,
    solve,
)

from minimax_aipe.npe import (
    make_crn_npe_oracle,
    npe,
    npe_restart,
)

from minimax_aipe.len import (
    make_lazy_crn_npe_oracle,
    len_loop,
    len_restart,
)

from minimax_aipe.aipe import (
    make_crn_prox_oracle,
    aipe,
    aipe_restart,
)

from minimax_aipe.alen import (
    make_lazy_crn_prox_oracle,
    aipe_restart_lazy,
    minimize_x_alen,
    maximize_y_alen,
)

from minimax_aipe.oracles import (
    crn_oracle,
    crn_oracle_minimization,
    eg_step,
    lazy_crn_oracle,
)

from minimax_aipe.operators import make_jacobian, make_operator

from minimax_aipe.gap import estimate_gap

__version__ = "0.1.0"

__all__ = [
    # Core types
    "MinimaxProblem",
    "SolverResult",
    # Full solver
    "solve",
    "RegularizedSubproblem",
    # Individual algorithms
    "aipe",
    "aipe_restart",
    "npe",
    "npe_restart",
    "len_loop",
    "len_restart",
    "aipe_restart_lazy",
    # Oracle factories
    "make_crn_prox_oracle",
    "make_crn_npe_oracle",
    "make_lazy_crn_prox_oracle",
    "make_lazy_crn_npe_oracle",
    # Oracles
    "crn_oracle",
    "crn_oracle_minimization",
    "eg_step",
    "lazy_crn_oracle",
    # Sub-solvers
    "minimize_x_alen",
    "maximize_y_alen",
    # Utilities
    "make_operator",
    "make_jacobian",
    "estimate_gap",
]
