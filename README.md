# README.md

```markdown
# Minimax-AIPE

Accelerated second-order methods for convex-concave minimax problems.

Solves **min_x max_y f(x, y)** to ε-accuracy with Õ(ε^{-4/7}) second-order
oracle complexity via a triple-loop reduction that composes Newton Proximal
Extragradient (NPE), Lazy Extra Newton (LEN), and Accelerated Inexact Proximal
Extragradient (AIPE).

Based on: Chen, Liu, Luo & Zhang (2025), *"Solving Convex-Concave Problems with
Õ(ε^{-4/7}) Second-Order Oracle Complexity."*

---

## Installation

```bash
pip install -e .
```

For Apple Silicon Metal acceleration:

```bash
pip install -e ".[metal]"
```

Requires Python ≥ 3.10, JAX ≥ 0.4.30.

---

## Quick Start

### Solve a problem in three lines

```python
import jax.numpy as jnp
from minimax_aipe import MinimaxProblem, solve

def f(x, y):
    return jnp.dot(x, y)  # bilinear game

problem = MinimaxProblem(
    f=f,
    dim_x=2, dim_y=2,
    D_x=2.0, D_y=2.0,       # feasible sets are Euclidean balls of diameter 2
    rho=0.0,                  # bilinear → Hessian Lipschitz = 0
)

result = solve(problem, epsilon=0.01)
print(f"Gap: {result.gap:.6f}  |  Converged: {result.converged}")
print(f"x = {result.x},  y = {result.y}")
```

### Use LEN instead of NPE for the inner solver

```python
result = solve(problem, epsilon=0.01, M_saddle="len", m_lazy=5)
```

When `M_saddle="len"`, all sub-solvers use lazy Hessians (ALEN for scalar
minimisation, LEN for saddle subproblems) and γ is set to ρ/√m per
Theorem 5.6. This trades per-iteration accuracy for fewer total Hessian
computations.

---

## Constructing Problems

`MinimaxProblem` wraps a convex-concave function `f(x, y)` with metadata about
the feasible sets and problem constants.

### Minimal — auto-differentiated gradients and Hessians

```python
def f(x, y):
    return 0.5 * jnp.dot(x, Q @ x) - 0.5 * jnp.dot(y, R @ y) + jnp.dot(x, A @ y)

problem = MinimaxProblem(
    f=f,
    dim_x=n, dim_y=m,
    D_x=2.0, D_y=2.0,
    rho=1.0,  # Hessian Lipschitz constant (Assumption 3.5)
    ell=1.0,  # Gradient Lipschitz constant (Assumption 3.4)
)
```

When `grad_f` and `hessian_f` are not provided, `MinimaxProblem` computes them
via `jax.grad` and `jax.hessian`. For large problems, providing explicit
implementations avoids tracing overhead.

### With explicit gradient and Hessian

```python
def grad_f(x, y):
    gx = Q @ x + A @ y         # ∇_x f
    gy_neg = R @ y - A.T @ x   # -∇_y f  (note the sign)
    return gx, gy_neg

def hessian_f(x, y):
    H_xx = Q
    H_xy = A
    H_yx = A.T
    H_yy = -R
    return ((H_xx, H_xy), (H_yx, H_yy))

problem = MinimaxProblem(
    f=f,
    dim_x=n, dim_y=m,
    D_x=2.0, D_y=2.0,
    grad_f=grad_f,
    hessian_f=hessian_f,
    rho=1.0,
    ell=1.0,
)
```

**Gradient convention:** `grad_f` returns `(∇_x f, -∇_y f)` — both components
point in the *descent* direction for their respective players. This matches the
monotone operator F(z) = [∇_x f, -∇_y f] from Equation (2) of the paper.

**Hessian convention:** `hessian_f` returns `((H_xx, H_xy), (H_yx, H_yy))`
where the blocks are the raw second derivatives of f (no sign flips).

### With custom projections

By default, `project_x` and `project_y` project onto a Euclidean ball of
radius D/2. Override them for polytopes, boxes, or other constraint sets:

```python
def project_box(lo, hi):
    """Project onto [lo, hi] element-wise."""
    def project(z):
        return jnp.clip(z, lo, hi)
    return project

problem = MinimaxProblem(
    f=f, dim_x=n, dim_y=m,
    D_x=2.0, D_y=2.0,
    project_x=project_box(-1.0, 1.0),
    project_y=project_box(-1.0, 1.0),
)
```

---

## Solver Parameters

```python
result = solve(
    problem,
    epsilon,            # target duality gap (required)
    gamma=None,         # regularisation parameter (auto-computed if None)
    M_saddle="npe",     # inner solver: "npe" or "len"
    m_lazy=5,           # Hessian reuse interval (only used when M_saddle="len")
    npe_T_factor=1.0,   # multiplier on inner-loop iteration count
    verbose=False,      # enable DEBUG logging
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `epsilon` | — | Target duality gap. The solver terminates when Gap ≤ ε. |
| `gamma` | `None` | Cubic regularisation strength. When `None`: uses `rho` for NPE, `rho/√m` for LEN. |
| `M_saddle` | `"npe"` | `"npe"` for fresh Hessians every step; `"len"` for lazy Hessians. |
| `m_lazy` | `5` | Hessian reuse interval for LEN. Larger → fewer Hessian computations, more iterations. |
| `npe_T_factor` | `1.0` | Scales the computed iteration count for inner NPE/LEN loops. Increase if the solver stalls. |

### Return value

`solve` returns a `SolverResult` (named tuple):

```python
result.x            # primal solution x̂
result.y            # dual solution ŷ
result.gap          # estimated duality gap
result.iterations   # outer-loop epochs executed
result.oracle_calls # total second-order oracle calls
result.converged    # True if gap ≤ epsilon
result.history      # dict with solver internals (zeta values, loop counts, etc.)
```

---

## Using Individual Algorithms

You don't have to go through the full triple loop. Each algorithm is usable
standalone.

### NPE — Newton Proximal Extragradient (Algorithm 6/7)

Solves a monotone variational inequality F(z) = 0 using cubic-regularised
Newton steps.

```python
from minimax_aipe import (
    MinimaxProblem, make_crn_npe_oracle, npe_restart, project_z,
)

problem = MinimaxProblem(f=f, dim_x=2, dim_y=2, D_x=2.0, D_y=2.0, rho=1.0)
z0 = jnp.zeros(4)
gamma = 2.0 * problem.rho
T, S = 50, 5

oracle = make_crn_npe_oracle(problem, gamma)

z_hat, calls = npe_restart(
    oracle, problem.operator_F, z0,
    T=T, gamma=gamma, S=S,
    project=lambda z: project_z(problem, z),
    fn=lambda z: jnp.dot(problem.operator_F(z), problem.operator_F(z)),
)
```

### LEN — Lazy Extra Newton (Algorithm 8/9)

Same as NPE but reuses the Hessian for `m` consecutive iterations.

```python
from minimax_aipe import (
    MinimaxProblem, make_lazy_crn_npe_oracle, len_restart, project_z,
)

problem = MinimaxProblem(f=f, dim_x=2, dim_y=2, D_x=2.0, D_y=2.0, rho=1.0)
z0 = jnp.zeros(4)
gamma = 2.0 * problem.rho
T, S, m = 50, 5, 5

oracle = make_lazy_crn_npe_oracle(problem, gamma)

z_hat, calls = len_restart(
    oracle, problem.operator_F, z0,
    T=T, gamma=gamma, m=m, S=S,
    project=lambda z: project_z(problem, z),
    fn=lambda z: jnp.dot(problem.operator_F(z), problem.operator_F(z)),
)
```

### AIPE — Accelerated Inexact Proximal Extragradient (Algorithm 1/2)

Solves min_z h(z) for a convex function using inexact proximal oracles.

```python
from minimax_aipe import make_crn_prox_oracle, aipe_restart

grad_fn = lambda z: ...
hess_fn = lambda z: ...

prox = make_crn_prox_oracle(grad_fn, hess_fn, gamma=1.0)

z_out, calls = aipe_restart(
    prox, grad_fn, z0,
    T=50, gamma=1.0, S=5,
)
```

### Duality Gap Estimation

```python
from minimax_aipe import estimate_gap

gap = estimate_gap(problem, x, y, num_restarts=10, num_steps=500)
```

Uses repeated gradient ascent/descent from random initial points to estimate
max_y f(x, y) - min_x f(x, y). Both the per-restart step loop and the restart
loop are compiled into single XLA computations via `jax.lax.fori_loop`.

---

## Architecture

The solver implements the triple-loop reduction from the paper:

```
Algorithm 3 (outer) — AIPE minimises Φ(x) = max_y f(x,y)
│                       via aipe_restart + inexact proximal oracle for Φ
│
└── Algorithm 4 (middle) — Inexact proximal oracle for Φ
│       Solves min_x max_y g(x,y;x̄) by running AIPE on -Ψ:
│           g = f + (γ/3)·‖x − x̄‖³          (cubic regularisation in x)
│           -Ψ(y;x̄) = -min_x g(x,y;x̄)
│
└── Algorithm 5 (inner) — Inexact proximal oracle for -Ψ
        Solves the regularised saddle subproblem via NPE/LEN-restart:
            h = f + (γ/3)·‖x − x̄‖³ − (γ/3)·‖y − ȳ‖³
            (cubic regularisation in both x and y)
```

Each layer delegates to the one below it through proximal oracle abstractions.
The `RegularizedSubproblem` (formerly `_HKernel`) provides JIT-stable parametric
operator methods so that JAX compiles the inner solver graph exactly once per
`(problem, gamma)` pair.

When `M_saddle="npe"`: NPE-restart for saddle subproblems, plain gradient
descent/ascent for scalar sub-solves.

When `M_saddle="len"`: LEN-restart for saddle subproblems, ALEN-restart for
scalar sub-solves. γ is set to ρ/√m per Theorem 5.6.

---

## Module Reference

| Module | Contents |
|--------|----------|
| `minimax_aipe.problem` | `MinimaxProblem`, `SolverResult` |
| `minimax_aipe.framework` | `solve`, `RegularizedSubproblem`, triple-loop internals |
| `minimax_aipe.npe` | `npe`, `npe_restart`, `make_crn_npe_oracle`, `project_z` |
| `minimax_aipe.len` | `len_loop`, `len_restart`, `make_lazy_crn_npe_oracle` |
| `minimax_aipe.aipe` | `aipe`, `aipe_restart`, `make_crn_prox_oracle` |
| `minimax_aipe.alen` | `aipe_restart_lazy`, `make_lazy_crn_prox_oracle`, `minimize_x_alen`, `maximize_y_alen` |
| `minimax_aipe.oracles` | `crn_oracle`, `crn_oracle_minimization`, `lazy_crn_oracle`, `eg_step` |
| `minimax_aipe.operators` | `make_operator`, `make_jacobian` |
| `minimax_aipe.gap` | `estimate_gap` |

---

## Key Constants (framework.py)

These are numerical guards and scheduling parameters inside the triple loop.
They rarely need tuning but are documented here for reference:

| Constant | Value | Purpose |
|----------|-------|---------|
| `_ABS_TOL` | `1e-12` | Floor for denominators to prevent division by zero |
| `_REG_MIN` | `1e-6` | Minimum regularisation to keep ρ_h positive |
| `_CUBIC_ZERO` | `1e-15` | Threshold below which cubic Hessian is treated as zero |
| `_GAP_FLOOR` | `1e-6` | Minimum epsilon for gap estimation step-size computation |
| `_S_CAP` | `4` | Practical cap on restart counts to limit nested-loop explosion |

---

## Citation

```bibtex
@article{chen2025minimax,
  title={Solving Convex-Concave Problems with $\tilde{O}(\epsilon^{-4/7})$
         Second-Order Oracle Complexity},
  author={Chen, Jianhao and Liu, Xiang and Luo, Haishan and Zhang, Tong},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Tests

Tests are located in the `tests/` directory and can be run with:

```bash
pytest
```

For parallel execution:

```bash
pytest -n auto
```

---

## License

MIT
```

A few things to note:

The **gradient convention** callout is critical — it's the single most common source of bugs when users construct `MinimaxProblem` with explicit `grad_f`. The sign on `∇_y f` catches everyone at least once.

The **Architecture** section mirrors the exact algorithm numbering from the paper, so a reader cross-referencing the paper can orient themselves immediately.

The **Module Reference** table at the bottom is the "look up which file to import from" cheat sheet — this is the deeper namespace in action.

Want to move on to the example scripts, or is there anything in the README you'd want adjusted first?