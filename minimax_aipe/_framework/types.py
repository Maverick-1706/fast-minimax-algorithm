"""Shared internal types for the framework implementation."""

from __future__ import annotations

from dataclasses import dataclass

from minimax_aipe._compat import CallStats


def _stats_array(value):
    return value.stats if isinstance(value, CallStats) else value


@dataclass
class _CallCounter:
    """Simple mutable call counter for bookkeeping across nested loops."""

    total: int = 0

