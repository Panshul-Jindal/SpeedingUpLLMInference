"""Run plan: full 2^n factorial at the centre, even-parity 2^(n-1) half-fractions off-centre."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from harness import optimizations as opt
from harness.regimes import WorkloadPoint


@dataclass(frozen=True)
class Cell:
    point_id: str
    mask: int

    @property
    def bits(self) -> str:
        return opt.to_bits(self.mask)


@dataclass(frozen=True)
class RunSpec:
    point_id: str
    mask: int
    rep: int

    @property
    def bits(self) -> str:
        return opt.to_bits(self.mask)


def _subsets(numbers: list[int]) -> list[int]:
    masks = []
    for k in range(len(numbers) + 1):
        for combo in combinations(numbers, k):
            masks.append(sum(opt.bit(n) for n in combo))
    return masks


def full_factorial(active: list[int]) -> list[int]:
    return sorted(_subsets(active), key=lambda m: (bin(m).count("1"), m))


def half_fraction(active: list[int]) -> list[int]:
    """Configurations with an even number of active optimizations on (defining relation I = +ABC...)."""
    return [m for m in full_factorial(active) if bin(m).count("1") % 2 == 0]


def smoke_configs(active: list[int]) -> list[int]:
    """Default, each single optimization, all-on."""
    all_on = sum(opt.bit(n) for n in active)
    return [0] + [opt.bit(n) for n in active] + [all_on]


def configs_for_point(point: WorkloadPoint, active: list[int], design: str) -> list[int]:
    if design == "smoke":
        return smoke_configs(active)
    if design == "full":
        return full_factorial(active) if point.is_center else half_fraction(active)
    raise ValueError(f"unknown design {design!r}")


def build_cells(points: list[WorkloadPoint], active: list[int], design: str) -> list[Cell]:
    return [Cell(p.point_id, m) for p in points for m in configs_for_point(p, active, design)]


def build_runs(cells: list[Cell], reps: int) -> list[RunSpec]:
    return [RunSpec(c.point_id, c.mask, r) for c in cells for r in range(1, reps + 1)]


def model_matrix(masks: list[int], active: list[int], interactions: bool = True) -> tuple[np.ndarray, list[str]]:
    """±1-coded columns: intercept, main effects, (two-way interactions)."""
    cols: list[np.ndarray] = [np.ones(len(masks))]
    names = ["intercept"]
    x = {n: np.array([1.0 if opt.is_on(m, n) else -1.0 for m in masks]) for n in active}
    for n in active:
        cols.append(x[n])
        names.append(opt.OPTIMIZATIONS[n - 1].key)
    if interactions:
        for a, b in combinations(active, 2):
            cols.append(x[a] * x[b])
            names.append(f"{opt.OPTIMIZATIONS[a - 1].key}:{opt.OPTIMIZATIONS[b - 1].key}")
    return np.column_stack(cols), names


def describe(cells: list[Cell], active: list[int]) -> str:
    by_point: dict[str, int] = {}
    for c in cells:
        by_point[c.point_id] = by_point.get(c.point_id, 0) + 1
    lines = [f"n = {len(active)} active optimizations: "
             + ", ".join(f"{n}:{opt.OPTIMIZATIONS[n - 1].key}" for n in active)]
    for pid, k in by_point.items():
        lines.append(f"  {pid:<18} {k:>4} configurations")
    lines.append(f"  {'TOTAL':<18} {len(cells):>4} (configuration, point) cells")
    return "\n".join(lines)
