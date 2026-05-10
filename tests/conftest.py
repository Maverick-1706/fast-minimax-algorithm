"""Pytest configuration.

JAX-Metal on Apple Silicon has known issues with default_memory_space.
Force CPU backend for reliable testing. Set JAX_PLATFORMS=metal explicitly
if you want to test Metal.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
