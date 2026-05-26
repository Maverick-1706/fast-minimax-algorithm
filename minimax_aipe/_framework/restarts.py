"""Restart helpers used by the nested AIPE/NPE/LEN loops."""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from minimax_aipe._framework.types import _stats_array


def _restart_with_early_stop(
    run_epoch: Callable[[Array], tuple[Array, int]],
    z0: Array,
    S: int,
    *,
    residual_fn: Optional[Callable[[Array], float]] = None,
    residual_tol: float = 0.0,
    step_tol: float = 0.0,
) -> tuple[Array, int, int]:
    z = z0
    total_calls = 0
    epochs_used = 0

    for s in range(S):
        z_new, calls = run_epoch(z)
        total_calls += calls
        epochs_used = s + 1
        if step_tol > 0:
            step = float(jnp.linalg.norm(z_new - z))
            if step < step_tol:
                return z_new, total_calls, epochs_used
        if residual_fn is not None and residual_tol > 0:
            res = float(residual_fn(z_new))
            if res < residual_tol:
                return z_new, total_calls, epochs_used
        z = z_new

    return z, total_calls, epochs_used


def _restart_jax(
    epoch_fn: Callable,
    z0: Array,
    S: int,
    *,
    step_tol: float = 0.0,
    warm: Optional[Array] = None,
    stats_init: Optional[Array] = None,
) -> tuple[Array, int, Optional[Array], Array]:
    dtype = z0.dtype
    tol_sq = jnp.asarray(step_tol ** 2 if step_tol > 0 else -1.0, dtype=dtype)
    S_jax = jnp.int32(S)
    tol_sq_cast = tol_sq.astype(dtype)
    int_zero = stats_init if stats_init is not None else jnp.int32(0)

    if warm is not None:
        def cond(carry):
            _z, prev_z, _w, epoch, _tic = carry
            not_done = epoch < S_jax
            diff = (_z - prev_z).astype(dtype)
            step_sq = jnp.dot(diff, diff)
            step_big = step_sq > tol_sq_cast
            return not_done & jnp.where(epoch > 0, step_big, jnp.bool_(True))

        def body(carry):
            z, _prev_z, w, epoch, total_inner_calls = carry
            result = epoch_fn(z, w)
            z_new, _calls, w_new = result[0], result[1], result[2]
            epoch_inner = _stats_array(result[3] if len(result) > 3 else int_zero)
            return (z_new, z, w_new, epoch + 1, total_inner_calls + epoch_inner)

        z_final, _, warm_out, epochs, total_inner_calls = jax.lax.while_loop(
            cond, body, (z0, z0, warm, jnp.int32(0), int_zero),
        )
        return z_final, epochs, warm_out, total_inner_calls

    def cond(carry):
        _z, prev_z, epoch, _tic = carry
        not_done = epoch < S_jax
        diff = (_z - prev_z).astype(dtype)
        step_sq = jnp.dot(diff, diff)
        step_big = step_sq > tol_sq_cast
        return not_done & jnp.where(epoch > 0, step_big, jnp.bool_(True))

    def body(carry):
        z, _prev_z, epoch, total_inner_calls = carry
        result = epoch_fn(z, None)
        z_new = result[0]
        epoch_inner = _stats_array(result[3] if len(result) > 3 else int_zero)
        return (z_new, z, epoch + 1, total_inner_calls + epoch_inner)

    z_final, _, epochs, total_inner_calls = jax.lax.while_loop(
        cond, body, (z0, z0, jnp.int32(0), int_zero),
    )
    return z_final, epochs, None, total_inner_calls

