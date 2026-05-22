import jax
import jax.numpy as jnp
from minimax_aipe.problem import MinimaxProblem
from minimax_aipe.oracles import _block_chol_solve

H_xx = jnp.zeros((2, 2))
H_yy = jnp.zeros((2, 2))
H_xy = jnp.eye(2)
H_yx = jnp.eye(2)
g = jnp.zeros(4)
lam = jnp.array(0.5)
tiny = jnp.array(1e-7)

delta = _block_chol_solve(g, H_xx, H_xy, H_yx, H_yy, lam, jnp.eye(2), jnp.eye(2), tiny)
print("delta:", delta)

