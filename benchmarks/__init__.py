"""Benchmark suite for Minimax-AIPE solver.

Force CPU backend for reliable benchmarking (same as test suite).
Set JAX_PLATFORMS=metal explicitly if you want to benchmark Metal.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
