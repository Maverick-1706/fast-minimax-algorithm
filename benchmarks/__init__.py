"""Benchmark suite for Minimax-AIPE solver.

Common configuration at package import time:
  - Forces CPU backend (set JAX_PLATFORMS=metal explicitly for GPU)
  - Enforces float64 precision for reproducibility
  - Sets global random seed if BENCHMARK_SEED env var is set

IMPORTANT: JAX_PLATFORMS must be set before JAX is imported.  This module
is imported before any JAX code runs, so the env var takes effect here.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

_SEED = os.environ.get("BENCHMARK_SEED")
GLOBAL_SEED = int(_SEED) if _SEED is not None else None
if _SEED is not None:
    random.seed(int(_SEED))
    np.random.seed(int(_SEED))
