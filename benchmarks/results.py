"""First-class benchmark result type.

Replaces loose dicts throughout the benchmark suite.  Every benchmark
function returns list[BenchmarkResult]; formatting and export consume
this type directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from minimax_aipe import OracleStats


@dataclass
class BenchmarkResult:
    """First-class benchmark result for a single solver on a single problem.

    Replaces loose dicts throughout the benchmark suite.  Every benchmark
    function returns list[BenchmarkResult]; formatting and export consume
    this type directly.
    """

    # ── Identity ──────────────────────────────────────────────────
    solver: str
    problem: str
    dim: int
    epsilon: float

    # ── Timing ────────────────────────────────────────────────────
    wall_time_mean: float
    wall_time_std: float
    ci: tuple[float, float]

    # ── Oracle accounting ─────────────────────────────────────────
    oracle_stats: OracleStats

    # ── Convergence ───────────────────────────────────────────────
    converged: bool
    gap_achieved: bool
    final_gap: float
    iterations: int

    # ── Extended (optional) ───────────────────────────────────────
    m_lazy: Optional[int] = None
    npe_T_factor: Optional[float] = None
    condition_number: Optional[float] = None
    rho: Optional[float] = None
    sparsity: Optional[float] = None
    peak_bytes: Optional[int] = None
    jax_bytes: Optional[int] = None
    wall_time_min: Optional[float] = None
    wall_time_max: Optional[float] = None
    n_outliers: Optional[int] = None
    final_residual: Optional[float] = None
    normalized_cost: Optional[float] = None
    # ── Per-iteration traces (Experiment 5) ───────────────────────────
    gap_trace: Optional[list[float]] = None
    """Duality gap after each outer iteration.  Length = outer_iters."""
    oracle_trace: Optional[list[int]] = None
    """Cumulative oracle calls after each outer iteration."""
    outer_iterations: Optional[int] = None
    """Number of outer iterations completed."""

    def to_dict(self) -> dict:
        """Return a flat JSON-serializable dict.

        Oracle statistics are flattened into the top-level dict
        with an 'oracle_' prefix.
        """
        d = asdict(self)
        
        # Flatten CI tuple
        ci = d.pop("ci")
        d["ci_lo"] = ci[0]
        d["ci_hi"] = ci[1]

        stats = d.pop("oracle_stats")
        for k, v in stats.items():
            # Standardize names: use 'oracle_calls' as the primary metric
            # but also keep the specific counts (grad, hessian, etc)
            d[f"oracle_{k}"] = v
        
        # Add a unified 'calls' alias for the primary metric
        primary_calls = stats.get("oracle_calls", 0)
        d["oracle_calls"] = primary_calls
        
        # Add solver-specific aliases (e.g., 'npe_calls', 'eg_calls')
        # for easier comparison in wide-format downstream analysis.
        solver_name = self.solver.replace("aipe_", "")
        d[f"{solver_name}_calls"] = primary_calls
        
        # Explicitly label the call type for the unified metric
        d["oracle_call_type"] = stats.get("call_type", "crn")

        # Traces — keep as lists (JSON-serializable)
        d["gap_trace"] = self.gap_trace
        d["oracle_trace"] = self.oracle_trace
        d["outer_iterations"] = self.outer_iterations
        
        return d

    def to_row(self) -> dict:
        """Alias for :meth:`to_dict` — semantic clarity in comprehensions."""
        return self.to_dict()

    def __getitem__(self, key):
        """Allow dict-style access for backward compatibility."""
        return getattr(self, key)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(solver={self.solver}, problem={self.problem}, "
            f"dim={self.dim}, gap={self.final_gap:.4f})"
        )
