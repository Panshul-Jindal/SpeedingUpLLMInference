"""Workload points: the Mid-Mid-Mid centre plus one-regime-at-a-time Low/High points."""
from __future__ import annotations

from dataclasses import dataclass

from harness.config import Regimes

CENTER = "center"
REGIME_NAMES = ("context", "concurrency", "reuse")


@dataclass(frozen=True)
class WorkloadPoint:
    point_id: str
    context: int        # prompt tokens
    concurrency: int    # in-flight requests
    reuse: float        # share of requests with a pooled prefix
    varied: str | None  # regime that is off-centre, None for the centre
    level: str          # low | mid | high

    @property
    def is_center(self) -> bool:
        return self.point_id == CENTER


def all_points(regimes: Regimes) -> list[WorkloadPoint]:
    mid = dict(context=int(regimes.context.mid), concurrency=int(regimes.concurrency.mid),
               reuse=float(regimes.reuse.mid))
    pts = [WorkloadPoint(CENTER, varied=None, level="mid", **mid)]
    for name in REGIME_NAMES:
        levels = getattr(regimes, name)
        for level in ("low", "high"):
            vals = dict(mid)
            raw = getattr(levels, level)
            vals[name] = float(raw) if name == "reuse" else int(raw)
            pts.append(WorkloadPoint(f"{name}_{level}", varied=name, level=level, **vals))
    return pts


def select_points(regimes: Regimes, wanted: str | list[str]) -> list[WorkloadPoint]:
    pts = all_points(regimes)
    if wanted == "all":
        return pts
    by_id = {p.point_id: p for p in pts}
    unknown = [w for w in wanted if w not in by_id]
    if unknown:
        raise ValueError(f"unknown point ids {unknown}; known: {list(by_id)}")
    return [by_id[w] for w in wanted]
