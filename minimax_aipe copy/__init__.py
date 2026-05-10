"""Minimax-AIPE: Accelerated second-order methods for convex-concave minimax problems."""

from minimax_aipe.problem import MinimaxProblem, SolverResult
from minimax_aipe.framework import solve

__version__ = "0.1.0"
__all__ = ["MinimaxProblem", "SolverResult", "solve"]
