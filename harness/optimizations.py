"""7-bit configuration encoding.

A configuration is a string of 7 characters '0'/'1'; character i-1 is optimization i (1-based, spec order).
Internally a bit set is an int with bit (i-1) set when optimization i is on.
"""
from __future__ import annotations

from dataclasses import dataclass

N_OPTS = 7


@dataclass(frozen=True)
class Optimization:
    number: int          # 1-based, as in the spec table
    key: str             # short machine name
    label: str           # human-readable
    numeric: bool        # changes model numerics (needs a quality run)
    min_gpus: int = 1


OPTIMIZATIONS: tuple[Optimization, ...] = (
    Optimization(1, "quant", "Quantized weights", numeric=True),
    Optimization(2, "kvq", "Reduced-precision KV cache", numeric=True),
    Optimization(3, "spec", "Speculative decoding", numeric=True),
    Optimization(4, "prefix", "Prefix reuse", numeric=False),
    Optimization(5, "batch", "Tuned batching", numeric=False),
    Optimization(6, "chunk", "Prefill scheduling", numeric=False),
    Optimization(7, "par", "Parallelism", numeric=False, min_gpus=2),
)
BY_KEY = {o.key: o for o in OPTIMIZATIONS}

NUMERIC_MASK = sum(1 << (o.number - 1) for o in OPTIMIZATIONS if o.numeric)
ALL_MASK = (1 << N_OPTS) - 1


def bit(number: int) -> int:
    return 1 << (number - 1)


def is_on(mask: int, number: int) -> bool:
    return bool(mask & bit(number))


def to_bits(mask: int) -> str:
    """int mask -> '1010000' (optimization 1 first)."""
    if not 0 <= mask <= ALL_MASK:
        raise ValueError(f"mask {mask} out of range")
    return "".join("1" if mask & bit(i) else "0" for i in range(1, N_OPTS + 1))


def from_bits(bits: str) -> int:
    if len(bits) != N_OPTS or set(bits) - {"0", "1"}:
        raise ValueError(f"bad config bits {bits!r}")
    return sum(bit(i + 1) for i, c in enumerate(bits) if c == "1")


def on_numbers(mask: int) -> list[int]:
    return [o.number for o in OPTIMIZATIONS if mask & bit(o.number)]


def names(mask: int) -> str:
    """'quant+prefix', or 'default' for all off."""
    keys = [OPTIMIZATIONS[n - 1].key for n in on_numbers(mask)]
    return "+".join(keys) if keys else "default"


def from_names(spec: str) -> int:
    if spec in ("", "default"):
        return 0
    return sum(bit(BY_KEY[k].number) for k in spec.split("+"))


def active_mask(num_gpus: int) -> int:
    """Optimizations that can be exercised with this many GPUs (bit 7 needs >= 2)."""
    return sum(bit(o.number) for o in OPTIMIZATIONS if num_gpus >= o.min_gpus)


def active_numbers(num_gpus: int) -> list[int]:
    return on_numbers(active_mask(num_gpus))


def numeric_setting(mask: int) -> int:
    """The part of a configuration that changes numerics (bits 1-3)."""
    return mask & NUMERIC_MASK


def numeric_label(mask: int) -> str:
    """'101' style label of the numeric bits 1-3."""
    return to_bits(numeric_setting(mask))[:3]
