"""Small return-value adapters for public API compatibility."""

from __future__ import annotations

import dis
import sys
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def _requested_unpack_count(default: int) -> int:
    """Best-effort count for the active ``UNPACK_*`` bytecode instruction."""
    try:
        frame = sys._getframe(2)
        for instr in dis.get_instructions(frame.f_code):
            if instr.offset == frame.f_lasti:
                if instr.opname == "UNPACK_SEQUENCE":
                    return int(instr.arg)
                if instr.opname == "UNPACK_EX":
                    before = instr.arg & 0xFF
                    after = instr.arg >> 8
                    return int(before + after)
                break
    except Exception:
        pass
    return default


@jax.tree_util.register_pytree_node_class
class CRNResult:
    """CRN result that supports both historical 2- and newer 3-unpacks."""

    __slots__ = ("z", "u", "stats")

    def __init__(self, z: Array, u: Array, stats: Array):
        self.z = z
        self.u = u
        self.stats = stats

    def __iter__(self) -> Iterator[Array]:
        values = (self.z, self.u, self.stats)
        yield from values[:_requested_unpack_count(3)]

    def __getitem__(self, idx):
        return (self.z, self.u, self.stats)[idx]

    def __len__(self) -> int:
        return 3

    def tree_flatten(self):
        return (self.z, self.u, self.stats), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class CallStats:
    """Vector stats that compare like the primary CRN call count."""

    __slots__ = ("stats",)

    def __init__(self, stats: Array):
        self.stats = stats

    @property
    def primary(self):
        arr = jnp.asarray(self.stats)
        return arr[0] if arr.ndim else arr

    @property
    def shape(self):
        return self.stats.shape

    @property
    def dtype(self):
        return self.stats.dtype

    def __getitem__(self, idx):
        return self.stats[idx]

    def __iter__(self):
        return iter(self.stats)

    def __int__(self) -> int:
        return int(self.primary)

    def __index__(self) -> int:
        return int(self)

    def __array__(self, dtype=None):
        return np.asarray(self.primary, dtype=dtype)

    def __eq__(self, other):
        other_value = other.primary if isinstance(other, CallStats) else other
        return self.primary == other_value

    def __lt__(self, other):
        other_value = other.primary if isinstance(other, CallStats) else other
        return self.primary < other_value

    def __le__(self, other):
        other_value = other.primary if isinstance(other, CallStats) else other
        return self.primary <= other_value

    def __gt__(self, other):
        other_value = other.primary if isinstance(other, CallStats) else other
        return self.primary > other_value

    def __ge__(self, other):
        other_value = other.primary if isinstance(other, CallStats) else other
        return self.primary >= other_value

    def __add__(self, other):
        other_value = other.stats if isinstance(other, CallStats) else other
        return CallStats(self.stats + other_value)

    def __radd__(self, other):
        other_value = other.stats if isinstance(other, CallStats) else other
        return CallStats(other_value + self.stats)

    def tree_flatten(self):
        return (self.stats,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])
