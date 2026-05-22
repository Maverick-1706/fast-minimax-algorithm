"""Tests for parametric problem families.

Validates that:
  - κ, ρ, sparsity parameters are correctly wired through
  - seed produces deterministic/reproducible output
  - sweep helpers generate the expected number of problems
  - backward-compat shim for condition_number works
"""

import warnings

import jax.numpy as jnp
import pytest

from benchmarks.families import (
    make_diagonal_saddle,
    make_nonzero_rho_quadratic,
    make_bilinear_saddle,
    make_scalable_diagonal,
    sweep_kappa,
    sweep_rho,
    sweep_sparsity,
)


# ── make_diagonal_saddle ─────────────────────────────────────────────────


class TestDiagonalSaddle:
    """Test the new parameterized make_diagonal_saddle."""

    def test_default_creates_valid_problem(self):
        p = make_diagonal_saddle(10)
        assert p.problem.dim_x == 10
        assert p.problem.dim_y == 10
        assert p.gap_star == 0.0

    def test_kappa_controls_condition(self):
        for kappa in [1, 10, 100, 1000]:
            p = make_diagonal_saddle(20, kappa=kappa)
            assert p.meta.kappa is not None
            assert p.meta.kappa > 0

    def test_kappa_1_gives_identity_spectrum(self):
        p = make_diagonal_saddle(10, kappa=1.0)
        assert p.meta.kappa is not None
        # With κ=1, all eigenvalues are 1, so ell ≈ 2 (λ + |σ|)
        # and mu_x = mu_y = 1
        assert p.meta.mu_x is not None
        assert p.meta.mu_x == pytest.approx(1.0, abs=0.01)

    def test_rho_zero_gives_zero_hessian_lipschitz(self):
        p = make_diagonal_saddle(10, rho=0.0)
        assert p.problem.rho == 0.0

    def test_rho_positive_adds_cubic(self):
        p = make_diagonal_saddle(10, rho=5.0)
        assert p.problem.rho == 10.0
        assert p.problem.ell is not None
        # ell should be larger than the quadratic-only case
        p0 = make_diagonal_saddle(10, rho=0.0)
        assert p.problem.ell > p0.problem.ell

    def test_sparsity_zero_gives_dense_coupling(self):
        p = make_diagonal_saddle(50, sparsity=0.0, seed=42)
        assert p.meta.sparsity == 0.0

    def test_sparsity_controls_coupling(self):
        p = make_diagonal_saddle(100, sparsity=0.9, seed=42)
        assert p.meta.sparsity == 0.9

    def test_seed_is_deterministic(self):
        p1 = make_diagonal_saddle(10, kappa=1e4, seed=42)
        p2 = make_diagonal_saddle(10, kappa=1e4, seed=42)
        x1 = p1.problem.f(jnp.ones(10), jnp.ones(10))
        x2 = p2.problem.f(jnp.ones(10), jnp.ones(10))
        assert float(x1) == float(x2)

    def test_different_seeds_give_different_problems(self):
        p1 = make_diagonal_saddle(10, seed=0)
        p2 = make_diagonal_saddle(10, seed=1)
        x1 = p1.problem.f(jnp.ones(10), jnp.ones(10))
        x2 = p2.problem.f(jnp.ones(10), jnp.ones(10))
        assert float(x1) != float(x2)

    def test_grad_is_finite(self):
        p = make_diagonal_saddle(10, kappa=1e4, rho=1.0, seed=0)
        x = jnp.ones(10) * 0.5
        y = jnp.ones(10) * 0.3
        gx, gy = p.problem.grad_f(x, y)
        assert jnp.all(jnp.isfinite(gx))
        assert jnp.all(jnp.isfinite(gy))

    def test_hessian_is_finite(self):
        p = make_diagonal_saddle(10, kappa=1e4, rho=1.0, seed=0)
        x = jnp.ones(10) * 0.5
        y = jnp.ones(10) * 0.3
        H = p.problem.hessian_f(x, y)
        for block_row in H:
            for block in block_row:
                assert jnp.all(jnp.isfinite(block))

    def test_n_blocks_heterogeneous(self):
        p = make_diagonal_saddle(20, kappa=100, n_blocks=4, seed=0)
        assert p.problem.dim_x == 20


# ── Sweep generators ─────────────────────────────────────────────────────


class TestSweepGenerators:

    def test_sweep_kappa_length(self):
        kappas = [1, 10, 100, 1000]
        problems = sweep_kappa(make_diagonal_saddle, 10, kappas)
        assert len(problems) == 4

    def test_sweep_kappa_forwarded(self):
        kappas = [1, 100]
        problems = sweep_kappa(make_diagonal_saddle, 10, kappas, rho=2.0)
        for p in problems:
            assert p.problem.rho == 4.0

    def test_sweep_rho_length(self):
        rhos = [0.0, 0.1, 1.0, 10.0]
        problems = sweep_rho(make_diagonal_saddle, 10, rhos)
        assert len(problems) == 4

    def test_sweep_sparsity_length(self):
        sparsities = [0.0, 0.5, 0.9]
        problems = sweep_sparsity(make_diagonal_saddle, 50, sparsities)
        assert len(problems) == 3


# ── Backward compatibility ───────────────────────────────────────────────


class TestBackwardCompat:

    def test_condition_number_deprecated_bilinear(self):
        from benchmarks.problem_constructors import make_ill_conditioned_bilinear
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = make_ill_conditioned_bilinear(dim=4, condition_number=1e3, seed=0)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "condition_number" in str(w[0].message)
        assert p.problem.dim_x == 4

    def test_condition_number_deprecated_quadratic(self):
        from benchmarks.problem_constructors import make_ill_conditioned_quadratic
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = make_ill_conditioned_quadratic(dim=4, condition_number=1e3, seed=0)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        assert p.problem.dim_x == 4

    def test_kappa_works_bilinear(self):
        from benchmarks.problem_constructors import make_ill_conditioned_bilinear
        p = make_ill_conditioned_bilinear(dim=4, kappa=1e4, seed=42)
        assert p.problem.dim_x == 4

    def test_kappa_works_quadratic(self):
        from benchmarks.problem_constructors import make_ill_conditioned_quadratic
        p = make_ill_conditioned_quadratic(dim=4, kappa=1e4, seed=0)
        assert p.problem.dim_x == 4

    def test_scale_condition_number_deprecated_kwarg(self):
        from benchmarks.scaling import scale_condition_number
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This should work but emit DeprecationWarning
            rows = scale_condition_number(
                "ill_bilinear", condition_numbers=[1e2], dim=4,
                epsilon=0.5, n_repeats=1, seed=0,
            )
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
        assert len(rows) > 0
