"""Monotone operator F(z) = [∇_x f, -∇_y f] and related utilities.

Reference: Equation (2) from the paper.
"""

from __future__ import annotations

from typing import Callable, Protocol

import jax.numpy as jnp
from jax import Array


class MonotoneOperator(Protocol):
    """Protocol for a monotone operator F: Z -> R^d.

    For the minimax problem min_x max_y f(x, y), the operator is:
        F(z) = [∇_x f(x, y), -∇_y f(x, y)]
    where z = (x, y).
    """

    def __call__(self, z: Array) -> Array: ...

    def jacobian(self, z: Array) -> Array: ...


def make_operator(
    grad_x: Callable[[Array, Array], Array],
    grad_y: Callable[[Array, Array], Array],
    dim_x: int,
    dim_y: int,
) -> Callable[[Array], Array]:
    """Construct F(z) from separate gradient functions.

    Parameters
    ----------
    grad_x : callable
        (x, y) -> ∇_x f(x, y)
    grad_y : callable
        (x, y) -> ∇_y f(x, y)
    dim_x, dim_y : int
        Dimensions.

    Returns
    -------
    F : callable
        z -> [∇_x f(x, y), -∇_y f(x, y)]
    """

    def F(z: Array) -> Array:
        x, y = z[:dim_x], z[dim_x:]
        return jnp.concatenate([grad_x(x, y), -grad_y(x, y)])

    return F


def make_jacobian(
    hessian_f: Callable,
    dim_x: int,
    dim_y: int,
) -> Callable[[Array], Array]:
    """Construct ∇F(z) from the Hessian of f.

    For F(z) = [∇_x f, -∇_y f], the Jacobian is:
        [[ ∇²_xx f,  ∇²_xy f],
         [-∇²_yx f, -∇²_yy f]]

    Parameters
    ----------
    hessian_f : callable
        (x, y) -> ((H_xx, H_xy), (H_yx, H_yy))
    """
    import jax

    def jacobian_F(z: Array) -> Array:
        x, y = z[:dim_x], z[dim_x:]
        H = hessian_f(x, y)
        # H is ((H_xx, H_xy), (H_yx, H_yy))
        H_xx, H_xy = H[0][0], H[0][1]
        H_yx, H_yy = H[1][0], H[1][1]
        top = jnp.concatenate([H_xx, H_xy], axis=1)
        bot = jnp.concatenate([-H_yx, -H_yy], axis=1)
        return jnp.concatenate([top, bot], axis=0)

    return jacobian_F