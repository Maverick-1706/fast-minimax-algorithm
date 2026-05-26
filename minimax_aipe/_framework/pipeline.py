"""Stable-identity JIT pipeline objects used by the outer solver."""

from __future__ import annotations

import functools
from typing import Optional

from jax import Array

from minimax_aipe.aipe import aipe
from minimax_aipe.problem import MinimaxProblem
from minimax_aipe._framework.oracles import _make_phi_oracle
from minimax_aipe._framework.params import _LoopParams
from minimax_aipe._framework.surrogates import RegularizedSubproblem


class _CachedPipeline:
    """Provides stable Python identities for closures passed to ``@jax.jit``."""

    def __init__(self, problem: MinimaxProblem, gamma: float, params: _LoopParams, M_saddle: str):
        self.problem = problem
        self.gamma = gamma
        self.params = params
        self.M_saddle = M_saddle
        self.kernel = RegularizedSubproblem(problem, gamma)
        self.phi_fn, self.grad_phi_fn = _make_phi_oracle(
            problem, gamma, params, M_saddle=M_saddle, m_lazy=params.m_lazy,
        )

    def prox_phi(self, x_bar: Array, y_init: Optional[Array] = None) -> tuple[Array, Array, Array, Array]:
        from minimax_aipe._framework.loops import _iProx_Phi

        return _iProx_Phi(
            self.problem, x_bar, self.gamma,
            zeta_2=self.params.zeta_2,
            params=self.params,
            M_saddle=self.M_saddle,
            y_init=y_init,
            kernel=self.kernel,
        )

    def run_outer_epoch(self, x_cur: Array, warm_y: Optional[Array] = None) -> tuple[Array, int, Array, Array]:
        result = aipe(
            self.prox_phi, self.grad_phi_fn, x_cur,
            self.params.T_outer, self.gamma,
            project=self.problem.project_x,
            warm_init=warm_y,
        )
        return result[0], result[1], result[2], result[3]


@functools.lru_cache(maxsize=1)
def _get_pipeline(problem: MinimaxProblem, gamma: float, params: _LoopParams, M_saddle: str):
    return _CachedPipeline(problem, gamma, params, M_saddle)

