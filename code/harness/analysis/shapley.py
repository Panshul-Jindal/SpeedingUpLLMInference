"""Shapley shares of a final configuration's log gain, v(S) = log(T_S / T_0)."""
from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations

from harness import optimizations as opt


def shapley(members: list[int], v: Callable[[int], float]) -> dict[int, float]:
    """Exact Shapley values for players `members` (optimization numbers); v takes a bit mask."""
    n = len(members)
    phi: dict[int, float] = {}
    for i in members:
        others = [m for m in members if m != i]
        total = 0.0
        for k in range(len(others) + 1):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for combo in combinations(others, k):
                s = sum(opt.bit(m) for m in combo)
                total += w * (v(s | opt.bit(i)) - v(s))
        phi[i] = total
    return phi


def shapley_for_config(final_mask: int, measured: Callable[[int], float | None],
                       model: Callable[[int], float] | None) -> tuple[dict[int, float], bool, str]:
    """Return (phi, exact, status). `measured(mask)` gives log T or None; `model(mask)` predicts log T.

    Missing subsets are filled from the model (shares labelled approximate). If T_0 or a needed subset
    is missing and no model is available, status is 'insufficient data'.
    """
    members = opt.on_numbers(final_mask)
    if not members:
        return {}, True, "ok (Default: no gain to attribute)"
    exact = True
    cache: dict[int, float] = {}

    def logt(mask: int) -> float:
        nonlocal exact
        if mask not in cache:
            m = measured(mask)
            if m is None or math.isnan(m):
                exact = False
                m = model(mask) if model else math.nan
            cache[mask] = m
        return cache[mask]

    base = logt(0)
    phi = shapley(members, lambda s: logt(s) - base)
    if any(math.isnan(x) for x in phi.values()):
        return {}, False, "insufficient data"
    return phi, exact, "ok"
