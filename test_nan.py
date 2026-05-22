import jax
import jax.numpy as jnp
from minimax_aipe.problem import MinimaxProblem
from minimax_aipe.npe import make_crn_npe_oracle

def _bilinear(dim=2, D=2.0):
    return MinimaxProblem(
        f=lambda x, y: jnp.dot(x, y),
        dim_x=dim, dim_y=dim, D_x=D, D_y=D,
    )

problem = _bilinear()
oracle = make_crn_npe_oracle(problem, 1.0)
z_half, u = oracle(jnp.zeros(4))
print("z_half:", z_half)
print("u:", u)
