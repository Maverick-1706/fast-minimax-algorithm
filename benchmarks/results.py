"""First-class benchmark result type.

Replaces loose dicts throughout the benchmark suite.  Every benchmark
function returns list[BenchmarkResult]; formatting and export consume
this type directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
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
    fixed_inner_iters: Optional[int] = None
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
    gap_source: str = "unknown"
    best_gap: Optional[float] = None
    best_gap_epoch: Optional[int] = None
    best_oracle_cost: Optional[float] = None
    # ── Independent Convergence Endpoints (Experiment 5) ──────────────
    gap_endpoints: Optional[list[float]] = None
    """Duality gap at different epsilon targets.  Length = len(epsilons)."""
    oracle_endpoints: Optional[list[float]] = None
    """Independent cold-start normalized cost for each target epsilon solve."""
    outer_iterations: Optional[int] = None
    """Number of outer iterations completed."""

    # ── Dynamic Metadata ──────────────────────────────────────────
    extra_metadata: dict = field(default_factory=dict)
    """Flexible dictionary for arbitrary hyperparameter or ablation tracking."""

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
        
        stats_obj = d.pop("oracle_stats", {})
        if stats_obj is None:
            stats = {}
        elif isinstance(stats_obj, dict):
            stats = stats_obj
        elif hasattr(stats_obj, "to_dict"):
            stats = stats_obj.to_dict()
        else:
            stats = asdict(stats_obj) if is_dataclass(stats_obj) else dict(stats_obj)
            
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
        d["oracle_type"] = stats.get("call_type", "crn")
        
        # Export normalized_cost (gradient-equivalent FLOP units) for
        # cross-solver comparison in downstream analysis.
        if self.dim:
            d["normalized_cost"] = float(
                self.oracle_stats.normalized_cost(self.dim * 2)
            ) if self.oracle_stats else 0.0
        
        # Endpoints — keep as lists (JSON-serializable)
        d["gap_endpoints"] = self.gap_endpoints
        d["oracle_endpoints"] = self.oracle_endpoints
        d["outer_iterations"] = self.outer_iterations
        
        # Unpack any arbitrary ablation parameters safely into the root dict
        extra = d.pop("extra_metadata", {})
        for k, v in extra.items():
            # FIX: Prevent silent overwrites of core fields or dynamically 
            # generated columns (like 'oracle_calls' or 'ci_lo') by 
            # namespacing colliding keys.
            if k in d:
                d[f"extra_{k}"] = v
            else:
                d[k] = v
                
        return d
    def to_row(self) -> dict:
        """Alias for :meth:`to_dict` — semantic clarity in comprehensions."""
        return self.to_dict()

    def __getitem__(self, key):
        """Allow dict-style access for backward compatibility."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(solver={self.solver}, problem={self.problem}, "
            f"dim={self.dim}, gap={self.final_gap:.4f})"
        )
