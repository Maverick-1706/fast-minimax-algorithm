"""Public compatibility facade for the Minimax-AIPE triple-loop framework.

This module intentionally stays thin. The implementation lives under
``minimax_aipe._framework`` and is re-exported here to preserve the existing
``minimax_aipe.framework`` import surface.
"""

from minimax_aipe._framework.api import solve, solve_outer_trace
from minimax_aipe._framework.loops import (
    _algorithm_3,
    _build_oracle_stats,
    _iProx_Phi,
    _iProx_Psi,
    _solve_saddle_subproblem,
)
from minimax_aipe._framework.oracles import (
    _make_phi_oracle,
    _make_psi_oracle,
    _maximize_y,
    _maximize_y_auto,
    _minimize_x,
    _minimize_x_auto,
)
from minimax_aipe._framework.params import (
    _LoopParams,
    _compute_loop_params,
    _default_gamma,
    _diam,
    _diameter,
    _ell,
    _initial_z,
    _safe_gap,
    _split,
)
from minimax_aipe._framework.restarts import _restart_jax, _restart_with_early_stop
from minimax_aipe._framework.surrogates import (
    RegularizedSubproblem,
    _HKernel,
    _cubic_grad,
    _cubic_hess,
    _make_g_problem,
    _make_h_problem,
)
from minimax_aipe._framework.types import _CallCounter

__all__ = [
    "solve",
    "solve_outer_trace",
    "RegularizedSubproblem",
    "_CallCounter",
    "_LoopParams",
    "_algorithm_3",
    "_compute_loop_params",
    "_cubic_grad",
    "_cubic_hess",
    "_iProx_Phi",
    "_iProx_Psi",
    "_make_g_problem",
    "_make_h_problem",
    "_make_phi_oracle",
    "_make_psi_oracle",
    "_restart_with_early_stop",
    "_solve_saddle_subproblem",
    "_HKernel",
]
