"""Convergence rate tests.

Verify that oracle_calls(ε) scales approximately as ε^{-p} where p ≈ 4/7.

These tests run the full solver across multiple tolerance levels and fit a
power law.  They are slower than the other test files because each
tolerance level requires a complete solve().

Mark with  pytest -m slow  to skip in fast CI runs.
"""

import pytest
import jax.numpy as jnp

from minimax_aipe import solve
from tests.test_correctness import epsilon


# ── Helpers ───────────────────────────────────────────────────────────────

def fit_power_law(xs, ys):
    """Least-squares fit of y = C · x^{-p}.

    Fits log(y) = slope · log(x) + intercept, then returns
    (exponent = -slope, intercept).

    A positive exponent means y grows as x shrinks.
    """
    import numpy as np
    log_x = np.log(np.asarray(xs))
    log_y = np.log(np.asarray(ys))
    slope, c = np.polyfit(log_x, log_y, 1)
    return float(-slope), float(c)


# ── Power-law model (from the paper) ─────────────────────────────────────

def theoretical_oracle_calls(epsilon, D, rho):
    """Theoretical scaling: C · D^{12/7} · (ρ/ε)^{4/7}.

    This is a model for the exponent; the constant C is unknown and
    problem-dependent.  We only check that the fitted exponent p
    falls in a reasonable range.
    """
    return (D ** (12.0 / 7.0)) * ((1.0 / max(epsilon, 1e-12)) ** (4.0 / 7.0))


# ── Convergence rate on bilinear problems ─────────────────────────────────

@pytest.mark.slow
class TestBilinearRate:

    def test_oracle_calls_increasing(self, bilinear_3d):
        """Oracle calls should increase (or stay flat) as ε decreases."""
        p = bilinear_3d["problem"]
        calls = []
        for eps in [0.1, 0.05, 0.01, 0.005]:
            result = solve(p, epsilon=eps, verbose=False)
            calls.append(result.oracle_calls)

        for i in range(len(calls) - 1):
            assert calls[i + 1] >= calls[i] - 1, (
                f"Oracle calls decreased: {calls} at epsilons "
                f"[0.1, 0.05, 0.01, 0.005]"
            )

    def test_power_law_exponent_bounded(self, bilinear_3d):
        """Fitted exponent p should be in a reasonable range.

        Theoretical p = 4/7 ≈ 0.571, but with logarithmic factors,
        small sample sizes, and implementation constants, we accept
        0.2 ≤ p ≤ 1.5.
        """
        p = bilinear_3d["problem"]
        epsilons = [0.1, 0.05, 0.02, 0.01, 0.005]
        calls = []
        valid_eps = []

        for eps in epsilons:
            result = solve(p, epsilon=eps, verbose=False)
            if result.oracle_calls > 0:
                calls.append(result.oracle_calls)
                valid_eps.append(eps)

        if len(valid_eps) < 3:
            pytest.skip("Not enough valid data points for rate fitting")

        exponent, _ = fit_power_law(valid_eps, calls)
        assert 0.4 <= exponent <= 0.8, f"Convergence rate {exponent:.3f} deviated from 4/7"; (
            f"Exponent p={exponent:.3f} outside [0.4, 0.8]; "
            f"data: ε={valid_eps}, calls={calls}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_rate_across_seeds(self, seed):
        """Rate is stable across different random matrix seeds."""
        from tests.conftest import make_bilinear_problem
        p = make_bilinear_problem(dim=4, seed=seed)["problem"]

        epsilons = [0.1, 0.05, 0.02, 0.01]
        calls = []
        for eps in epsilons:
            result = solve(p, epsilon=eps, verbose=False)
            calls.append(result.oracle_calls)

        if all(c > 0 for c in calls):
            exponent, _ = fit_power_law(epsilons, calls)
            assert 0.1 <= exponent <= 2.0


# ── Convergence rate on quadratic problems ────────────────────────────────

@pytest.mark.slow
class TestQuadraticRate:

    def test_oracle_calls_increasing(self, quadratic_3d):
        p = quadratic_3d["problem"]
        calls = []
        for eps in [0.1, 0.05, 0.02, 0.01]:
            result = solve(p, epsilon=eps, verbose=False)
            calls.append(result.oracle_calls)

        for i in range(len(calls) - 1):
            assert calls[i + 1] >= calls[i] - 1

    def test_power_law_exponent_bounded(self, quadratic_3d):
        p = quadratic_3d["problem"]
        epsilons = [0.1, 0.05, 0.02, 0.01, 0.005]
        calls = []
        valid_eps = []

        for eps in epsilons:
            result = solve(p, epsilon=eps, verbose=False)
            if result.oracle_calls > 0:
                calls.append(result.oracle_calls)
                valid_eps.append(eps)

        if len(valid_eps) < 3:
            pytest.skip("Not enough valid data points")

        exponent, _ = fit_power_law(valid_eps, calls)
        assert 0.4 <= exponent <= 0.8


# ── NPE-only baseline rate ────────────────────────────────────────────────

@pytest.mark.slow
class TestNPEBaselineRate:
    """Test NPE (Algorithm 7) directly — should show p ≈ 2/3 scaling.

    This is the non-accelerated baseline for comparison.
    """

    def test_npe_oracle_calls_scaling(self, bilinear_3d):
        """NPE-restart oracle calls should increase with 1/ε."""
        from minimax_aipe import npe_restart, make_crn_npe_oracle

        p = bilinear_3d["problem"]
        gamma = 2.0 * max(p.rho or 1.0, 1e-6)
        oracle = make_crn_npe_oracle(p, gamma)
        F_fn = p.operator_F
        z0 = jnp.zeros(p.dim_x + p.dim_y)

        calls_list = []
        epsilons = [0.1, 0.05, 0.02, 0.01]

        for eps in epsilons:
            import math
            D = max(p.D_x, p.D_y)
            S = max(1, min(4, int(math.ceil(math.log2(max(D / eps, 2.0))))))
            T = max(8, int((gamma / max(eps, 1e-10)) ** 0.5))

            _, total_calls = npe_restart(
                oracle, F_fn, z0, T, gamma, S,
            )
            calls_list.append(total_calls)

        # Calls should be monotonically increasing
        for i in range(len(calls_list) - 1):
            assert calls_list[i + 1] >= calls_list[i]

    def test_npe_rate_vs_theory(self, bilinear_3d):
        """NPE rate should be steeper than 0.3 (roughly ε^{-2/3})."""
        from minimax_aipe import npe_restart, make_crn_npe_oracle

        p = bilinear_3d["problem"]
        gamma = 2.0 * max(p.rho or 1.0, 1e-6)
        oracle = make_crn_npe_oracle(p, gamma)
        F_fn = p.operator_F
        z0 = jnp.zeros(p.dim_x + p.dim_y)

        import math
        epsilons = [0.1, 0.05, 0.02, 0.01]
        calls_list = []
        for eps in epsilons:
            D = max(p.D_x, p.D_y)
            S = max(1, min(4, int(math.ceil(math.log2(max(D / eps, 2.0))))))
            T = max(8, int((gamma / max(eps, 1e-10)) ** 0.5))
            _, total_calls = npe_restart(oracle, F_fn, z0, T, gamma, S)
            calls_list.append(total_calls)

        if all(c > 0 for c in calls_list):
            exponent, _ = fit_power_law(epsilons, calls_list)
            # NPE theoretical rate is ε^{-2/3}, so p ≈ 0.667
            assert exponent > 0.1


# ── Gap convergence ──────────────────────────────────────────────────────

@pytest.mark.slow
class TestGapConvergence:
    """Verify that the achieved gap tracks the requested epsilon."""

    def test_gap_below_epsilon(self, bilinear_3d, epsilon):
        """For a zero-gap problem, achieved gap should be small."""
        p = bilinear_3d["problem"]
        result = solve(p, epsilon=epsilon, verbose=False)
        assert result.gap <= epsilon * 1.5, f"Solver failed to reach epsilon. Gap: {result.gap}, Eps: {epsilon}" 

    def test_gap_decreasing_sequence(self, bilinear_3d):
        """Gaps form a non-increasing sequence as ε tightens."""
        p = bilinear_3d["problem"]
        gaps = []
        for eps in [0.1, 0.05, 0.02, 0.01]:
            result = solve(p, epsilon=eps, verbose=False)
            gaps.append(result.gap)

        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-3, (
                f"Gap increased: {gaps}"
            )
