"""Main effects and two-way interactions of log-throughput by OLS with ±1 effect coding, per point.

Fits what the data supports: main effects of optimizations that vary, then two-way terms that are not
aliased with terms already in the model and leave at least one residual degree of freedom.
Everything else is reported as "insufficient data".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from harness import optimizations as opt

INSUFFICIENT = "insufficient data"


@dataclass
class Fit:
    env: str
    point_id: str
    active: list[int]
    terms: list[tuple[int, ...]] = field(default_factory=list)    # () intercept, (i,), (i, j)
    params: dict[tuple[int, ...], float] = field(default_factory=dict)
    bse: dict[tuple[int, ...], float] = field(default_factory=dict)
    n_obs: int = 0
    n_configs: int = 0
    r2: float = math.nan
    residual_sd: float = math.nan
    df_resid: float = 0.0
    noise_sd: float = math.nan
    status: str = INSUFFICIENT

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def predict_log(self, mask: int) -> float:
        if not self.ok:
            return math.nan
        val = 0.0
        for t, b in self.params.items():
            x = 1.0
            for n in t:
                x *= 1.0 if opt.is_on(mask, n) else -1.0
            val += b * x
        return val


def _col(masks: list[int], term: tuple[int, ...]) -> np.ndarray:
    x = np.ones(len(masks))
    for n in term:
        x = x * np.array([1.0 if opt.is_on(m, n) else -1.0 for m in masks])
    return x


def fit_point(df: pd.DataFrame, env: str, point_id: str, active: list[int]) -> Fit:
    """df: rep-level rows with config_bits and output_tok_s (status ok) for one (env, point)."""
    fit = Fit(env, point_id, active)
    d = df[(df["output_tok_s"] > 0)].copy()
    if d.empty:
        return fit
    masks = [opt.from_bits(b) for b in d["config_bits"]]
    y = np.log(d["output_tok_s"].to_numpy(dtype=float))
    fit.n_obs = len(y)
    fit.n_configs = len(set(masks))

    # Pure-error noise: pooled within-configuration sd of log T (needs replicates).
    resid_ss, dof = 0.0, 0
    for _, g in d.assign(_y=y).groupby("config_bits"):
        if len(g) > 1:
            resid_ss += float(((g["_y"] - g["_y"].mean()) ** 2).sum())
            dof += len(g) - 1
    fit.noise_sd = math.sqrt(resid_ss / dof) if dof else math.nan

    terms: list[tuple[int, ...]] = [()]
    cols = [_col(masks, ())]
    rank = 1

    def try_add(t: tuple[int, ...]) -> None:
        nonlocal rank
        c = _col(masks, t)
        cand = np.column_stack(cols + [c])
        r = np.linalg.matrix_rank(cand)
        if r > rank and fit.n_configs - r >= 0 and len(y) - r >= 1:
            terms.append(t)
            cols.append(c)
            rank = r

    for n in active:
        try_add((n,))
    for a, b in combinations(active, 2):
        if (a,) in terms and (b,) in terms:
            try_add((a, b))
    if len(terms) < 2:
        return fit
    X = np.column_stack(cols)
    res = sm.OLS(y, X).fit()
    fit.terms = terms
    fit.params = {t: float(v) for t, v in zip(terms, res.params)}
    fit.bse = {t: float(v) for t, v in zip(terms, res.bse)}
    fit.df_resid = float(res.df_resid)
    fit.r2 = float(res.rsquared) if res.centered_tss > 0 else math.nan
    fit.residual_sd = float(math.sqrt(res.mse_resid)) if res.df_resid > 0 else math.nan
    fit.status = "ok"
    return fit


def main_effects_table(fits: list[Fit]) -> pd.DataFrame:
    rows = []
    for f in fits:
        for n in f.active:
            o = opt.OPTIMIZATIONS[n - 1]
            row = {"env": f.env, "point_id": f.point_id, "optimization": n, "name": o.key}
            b = f.params.get((n,)) if f.ok else None
            if b is None:
                row.update(multiplier=math.nan, se=math.nan, beta=math.nan, beta_se=math.nan, status=INSUFFICIENT)
            else:
                se = f.bse.get((n,), math.nan)
                mult = math.exp(2 * b)            # -1 -> +1 on the ±1 scale
                row.update(multiplier=mult, se=mult * 2 * se if not math.isnan(se) else math.nan,
                           beta=b, beta_se=se, status="ok" if not math.isnan(se) else "ok (no SE: df=0)")
            rows.append(row)
    return pd.DataFrame(rows)


def model_fit_table(fits: list[Fit]) -> pd.DataFrame:
    return pd.DataFrame([{
        "env": f.env, "point_id": f.point_id, "n_obs": f.n_obs, "n_configs": f.n_configs,
        "n_params": len(f.terms), "n_interaction_terms": sum(1 for t in f.terms if len(t) == 2),
        "r2": f.r2, "residual_sd_log": f.residual_sd, "df_resid": f.df_resid, "noise_sd_log": f.noise_sd,
        "status": f.status} for f in fits])


def interactions_table(fits: list[Fit], cell_thr: dict[tuple[str, str, int], float]) -> pd.DataFrame:
    """I_AB = (T_AB * T_0) / (T_A * T_B): measured where all four configurations ran, else from the model."""
    rows = []
    for f in fits:
        for a, b in combinations(f.active, 2):
            A, B = opt.bit(a), opt.bit(b)
            masks = {"0": 0, "A": A, "B": B, "AB": A | B}
            t = {k: cell_thr.get((f.env, f.point_id, m)) for k, m in masks.items()}
            row = {"env": f.env, "point_id": f.point_id, "a": a, "b": b,
                   "pair": f"{opt.OPTIMIZATIONS[a - 1].key}:{opt.OPTIMIZATIONS[b - 1].key}"}
            if all(v is not None and v > 0 for v in t.values()):
                row.update(ratio=t["AB"] * t["0"] / (t["A"] * t["B"]), label="measured", se_log=math.nan)
            elif f.ok and (a, b) in f.params:
                pred = {k: f.predict_log(m) for k, m in masks.items()}
                se = f.bse.get((a, b), math.nan)
                row.update(ratio=math.exp(pred["AB"] + pred["0"] - pred["A"] - pred["B"]), label="model",
                           se_log=4 * se if not math.isnan(se) else math.nan)
            else:
                row.update(ratio=math.nan, label=INSUFFICIENT, se_log=math.nan)
            rows.append(row)
    return pd.DataFrame(rows)
