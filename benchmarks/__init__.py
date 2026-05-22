"""Benchmark suite for Minimax-AIPE solver.

Common configuration at package import time:
  - Forces CPU backend (set JAX_PLATFORMS=metal explicitly for GPU)
  - Enforces float32 precision (configurable via config.ENABLE_X64)
  - Sets global random seed from config or BENCHMARK_SEED env var

IMPORTANT: JAX_PLATFORMS must be set before JAX is imported.  This module
is imported before any JAX code runs, so the env var takes effect here.
"""

from __future__ import annotations

import os
import random

# os.environ.setdefault("JAX_PLATFORMS", "cpu") # Removed to allow GPU acceleration

import jax
import numpy as np

from benchmarks import config

jax.config.update("jax_enable_x64", config.ENABLE_X64)

GLOBAL_SEED: int | None = config.BENCHMARK_SEED
if GLOBAL_SEED is not None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
