"""Gates and Value: throughput speedup over Default at the same point, gated by p99 latency and accuracy."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from harness import optimizations as opt
from harness.config import Gates

CELL_KEYS = ["env", "point_id", "config_bits"]


def quality_table(quality_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """(env, numeric_label) -> {task: points} for successful quality runs."""
    out = {}
    for r in quality_rows:
        if r.get("status") == "ok" and r.get("scores"):
            out[(r["env"], r["numeric_bits"])] = r["scores"]
    return out


def accuracy_gate(env: str, bits: str, qtab: dict[tuple[str, str], dict[str, float]], gates: Gates
                  ) -> tuple[str, dict[str, float]]:
    """('pass' | 'fail' | 'unknown', per-task drop). Configurations with Default numerics always pass."""
    label = opt.numeric_label(opt.from_bits(bits))
    if label == "000":
        return "pass", {}
    ref, cur = qtab.get((env, "000")), qtab.get((env, label))
    if not ref or not cur:
        return "unknown", {}
    drops = {t: ref[t] - cur[t] for t in ref if t in cur}
    if not drops:
        return "unknown", {}
    return ("fail" if any(d > gates.max_accuracy_drop for d in drops.values()) else "pass"), drops


def _gmean(x: pd.Series) -> float:
    x = x.dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else math.nan


def _latency_gates(tpot: float | None, ttft: float | None, gates: Gates) -> list[str]:
    failed = []
    if tpot is None or (isinstance(tpot, float) and math.isnan(tpot)) or tpot > gates.p99_tpot_ms:
        failed.append("p99_tpot")
    if ttft is None or (isinstance(ttft, float) and math.isnan(ttft)) or ttft > gates.p99_ttft_ms:
        failed.append("p99_ttft")
    return failed


def default_throughput(runs: pd.DataFrame) -> dict[tuple[str, str], float]:
    ok = runs[(runs["status"] == "ok") & (runs["config_bits"] == "0000000")]
    return {k: _gmean(g["output_tok_s"]) for k, g in ok.groupby(["env", "point_id"])}


def per_run_values(runs: pd.DataFrame, qtab: dict[tuple[str, str], dict[str, float]], gates: Gates
                   ) -> pd.DataFrame:
    base = default_throughput(runs)
    rows = []
    for _, r in runs.iterrows():
        d = base.get((r["env"], r["point_id"]), math.nan)
        status = r.get("status")
        out = {"env": r["env"], "point_id": r["point_id"], "config_bits": r["config_bits"],
               "config_names": r.get("config_names"), "rep": r["rep"], "status": status,
               "output_tok_s": r.get("output_tok_s") if status == "ok" else math.nan,
               "default_tok_s": d, "p99_tpot_ms": r.get("tpot_p99_ms"), "p99_ttft_ms": r.get("ttft_p99_ms")}
        acc, drops = accuracy_gate(r["env"], r["config_bits"], qtab, gates)
        if status != "ok":
            out.update(speedup=math.nan, tpot_gate="n/a", ttft_gate="n/a", accuracy_gate=acc,
                       failed_gates=status, value=1.0)
        else:
            sp = out["output_tok_s"] / d if d and not math.isnan(d) else math.nan
            failed = _latency_gates(r.get("tpot_p99_ms"), r.get("ttft_p99_ms"), gates)
            if acc == "fail":
                failed.append("accuracy")
            out.update(speedup=sp, tpot_gate="fail" if "p99_tpot" in failed else "pass",
                       ttft_gate="fail" if "p99_ttft" in failed else "pass", accuracy_gate=acc,
                       failed_gates=";".join(failed),
                       value=(1.0 if failed or math.isnan(sp) else sp))
        out["max_accuracy_drop"] = max(drops.values()) if drops else math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def cell_values(runs: pd.DataFrame, qtab: dict[tuple[str, str], dict[str, float]], gates: Gates) -> pd.DataFrame:
    """One row per (env, point, config): mean over reps, gates on the mean p99s, Value."""
    base = default_throughput(runs)
    rows = []
    for (env, pid, bits), g in runs.groupby(CELL_KEYS):
        ok = g[g["status"] == "ok"]
        d = base.get((env, pid), math.nan)
        acc, drops = accuracy_gate(env, bits, qtab, gates)
        row: dict[str, Any] = {"env": env, "point_id": pid, "config_bits": bits,
                               "config_names": opt.names(opt.from_bits(bits)), "n_reps_ok": len(ok),
                               "n_reps": len(g), "default_tok_s": d, "accuracy_gate": acc,
                               "max_accuracy_drop": max(drops.values()) if drops else math.nan}
        if ok.empty:
            row.update(status=";".join(sorted(set(g["status"]))), output_tok_s=math.nan, speedup=math.nan,
                       p99_tpot_ms=math.nan, p99_ttft_ms=math.nan, failed_gates="infeasible"
                       if (g["status"] == "infeasible").any() else ";".join(sorted(set(g["status"]))),
                       value=1.0, error=str(g["error"].dropna().iloc[0])[:300] if "error" in g and
                       g["error"].notna().any() else None)
        else:
            thr = _gmean(ok["output_tok_s"])
            tpot = float(ok["tpot_p99_ms"].mean())
            ttft = float(ok["ttft_p99_ms"].mean())
            sp = thr / d if d and not math.isnan(d) else math.nan
            failed = _latency_gates(tpot, ttft, gates)
            if acc == "fail":
                failed.append("accuracy")
            row.update(status="ok", output_tok_s=thr, speedup=sp, p99_tpot_ms=tpot, p99_ttft_ms=ttft,
                       failed_gates=";".join(failed), value=1.0 if failed or math.isnan(sp) else sp, error=None)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Flag points where Default itself fails a gate (do not silently pass them).
    dflt = df[df["config_bits"] == "0000000"].set_index(["env", "point_id"])["failed_gates"]
    df["default_fails_gate"] = [bool(dflt.get((e, p), "missing")) for e, p in zip(df["env"], df["point_id"])]
    df["default_failed_gates"] = [dflt.get((e, p), "default missing") for e, p in zip(df["env"], df["point_id"])]
    return df


def final_configs(cells: pd.DataFrame) -> pd.DataFrame:
    """Highest Value per (env, point); ties go to fewer optimizations (then Default)."""
    rows = []
    for (env, pid), g in cells.groupby(["env", "point_id"]):
        g = g.assign(_n=g["config_bits"].str.count("1"))
        best = g.sort_values(["value", "_n"], ascending=[False, True]).iloc[0]
        rows.append({"env": env, "point_id": pid, "final_config": best["config_bits"],
                     "final_names": best["config_names"], "value": best["value"],
                     "output_tok_s": best["output_tok_s"], "default_tok_s": best["default_tok_s"],
                     "default_fails_gate": best["default_fails_gate"],
                     "default_failed_gates": best["default_failed_gates"]})
    return pd.DataFrame(rows)
