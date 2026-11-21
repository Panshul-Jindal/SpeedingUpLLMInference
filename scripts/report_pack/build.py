"""Build the report pack: combined clean dataset, analysis CSVs, figures and generated docs.

Inputs (from the 2 x H100 NVL runs on ccs-srv004):
  results/fast/          fast plan (384 runs, 128 cells, 7 points) + quality (trimmed)
  results/investigate/   focused re-runs with deep telemetry (126 runs + 18 profiler passes) + paired quality
Output: evaluation/report_pack/ (recreated on every run; then run evaluation/make_report_figures.py
for figures 10-12 and the report appendix).

Every number in the generated docs comes from the CSVs written here; nothing is typed by hand.
Run:  .venv/bin/python scripts/report_pack/build.py   (needs matplotlib: uv pip install matplotlib;
      it is not in requirements.txt because the cluster runs never need it)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import optimizations as opt                                  # noqa: E402
from harness.analysis.investigate import measured_interactions            # noqa: E402
from harness.analysis.report import analyze                               # noqa: E402
from harness.analysis.value import accuracy_gate, cell_values, final_configs, quality_table  # noqa: E402
from harness.config import load_env, load_experiment, load_profile        # noqa: E402
from harness.results import ResultStore                                   # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "evaluation" / "report_pack"
DATA, FIG = OUT / "data", OUT / "figures"
ENV = "h100"
# GPU 0 was shared with another process from this time to the end of the fast run (see contamination evidence).
CUT = pd.Timestamp("2026-11-07T11:23:00Z")
OPTS = ["quant", "kvq", "spec", "prefix", "batch"]
POINT_ORDER = ["center", "context_low", "context_high", "concurrency_low", "concurrency_high", "reuse_low",
               "reuse_high"]
POINT_DESC = {
    "center": "2048-token prompts, 32 in flight, 50% prefix reuse",
    "context_low": "256-token prompts (else centre)", "context_high": "8192-token prompts (else centre)",
    "concurrency_low": "2 requests in flight (else centre)", "concurrency_high": "128 in flight (else centre)",
    "reuse_low": "0% prefix reuse (else centre)", "reuse_high": "90% prefix reuse (else centre)",
}
FAST_PROFILE = load_profile("fast")
EXP = load_experiment(FAST_PROFILE)
GATES = EXP.gates
PEAKS = load_env(ENV).peaks


def gmean(s: pd.Series) -> float:
    s = s.dropna()
    s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else math.nan


def names(bits: str) -> str:
    return opt.names(opt.from_bits(bits))


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(DATA / name, index=False, float_format="%.6g")
    return df


# ====================================================================== 1. load + contamination

def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    fast = ResultStore(RES / "fast" / "runs.jsonl").frame()
    fast = fast[fast["kind"] == "run"].copy()
    inv = ResultStore(RES / "investigate" / "investigate.jsonl").frame()
    for d in (fast, inv):
        d["t"] = pd.to_datetime(d["written_at"])
        d["gpu"] = d["gpu_ids"].apply(lambda g: int(g[0]))
    fast["contaminated"] = (fast.gpu == 0) & (fast.t >= CUT)
    inv["exclusive"] = inv["exclusive"].fillna(False).astype(bool)
    return fast, inv


def clean_runs(fast: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """Combined dataset at the 7 study points:
    - fast-plan cells with no contaminated rep (source "fast", 3 reps);
    - contaminated cells replaced by their investigation re-run (source "re-run", 2-5 reps);
    - cells the half-fraction never ran but the investigation did, e.g. single optimizations off-centre
      (source "investigation", 2-5 reps). Exclusive (other-GPU-idle) runs are not mixed in.
    """
    key = lambda d: d[["point_id", "config_bits"]].apply(tuple, axis=1)        # noqa: E731
    bad_cells = set(map(tuple, fast.loc[fast.contaminated, ["point_id", "config_bits"]].values))
    fast_cells = set(map(tuple, fast[["point_id", "config_bits"]].values))
    keep = fast[~key(fast).isin(bad_cells)].assign(source="fast")
    rer = inv[(inv.kind == "investigate") & ~inv.exclusive & ~inv.point_id.str.contains("@")].copy()
    k = key(rer)
    rer = rer[k.isin(bad_cells) | ~k.isin(fast_cells)].copy()
    rer["source"] = np.where(key(rer).isin(bad_cells), "re-run", "investigation")
    rer = rer.assign(kind="run", profile="fast", replaced_from=rer.run_key)
    rer["run_key"] = [f"fast__{ENV}__{p}__{b}__inv{r}" for p, b, r in zip(rer.point_id, rer.config_bits, rer.rep)]
    clean = pd.concat([keep, rer], ignore_index=True)
    missing = sorted(bad_cells - set(map(tuple, rer[["point_id", "config_bits"]].values)))
    return clean, missing


# ====================================================================== 2. standard analysis on clean data

def standard_report(clean: pd.DataFrame) -> Path:
    """Write runs/quality for a 'fast_clean' results dir and run the harness's own analysis on it."""
    d = RES / "fast_clean"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    cols = [c for c in clean.columns if c not in ("t", "gpu", "contaminated")]
    with open(d / "runs.jsonl", "w") as f:
        for r in clean[cols].to_dict("records"):
            f.write(json.dumps({k: v for k, v in r.items() if not (isinstance(v, float) and math.isnan(v))},
                               default=str) + "\n")
    shutil.copy(RES / "fast" / "quality.jsonl", d / "quality.jsonl")
    return analyze(d, "fast")


# ====================================================================== 3. cells + composition

def cell_table(clean: pd.DataFrame, qtab: dict) -> pd.DataFrame:
    cv = cell_values(clean, qtab, GATES)
    agg = clean.groupby(["point_id", "config_bits"]).agg(
        source=("source", "first"), tok_s_cv=("output_tok_s", lambda s: s.std() / s.mean() if len(s) > 1 else np.nan),
        spec_accept_rate=("spec_accept_rate", "mean"), tokens_per_step=("tokens_per_step", "mean"),
        prefix_hit_rate=("prefix_hit_rate", "mean"), running_mean=("running_mean", "mean"),
        kv_usage_max=("kv_usage_max", "max"), preemptions=("preemptions", "max"),
        max_num_seqs=("max_num_seqs_effective", "first"), sm_clock_mhz=("gpu_sm_clock_mean_mhz", "mean"),
        power_w=("gpu_power_mean_w", "mean"), tpot_p50_ms=("tpot_p50_ms", "mean"), ttft_p50_ms=("ttft_p50_ms", "mean"),
    ).reset_index()
    t = cv.merge(agg, on=["point_id", "config_bits"], how="left")
    t["latency_gates_pass"] = ~t.failed_gates.fillna("").str.contains("p99")
    t["n_opts"] = t.config_bits.str[:5].str.count("1")
    t["point_order"] = t.point_id.map({p: i for i, p in enumerate(POINT_ORDER)})
    return t.sort_values(["point_order", "n_opts", "config_bits"]).drop(columns=["point_order"])


def x_matrix(bits: list[str]) -> np.ndarray:
    return np.array([[1.0] + [1.0 if b[i] == "1" else -1.0 for i in range(5)] for b in bits])


def composition(cells: pd.DataFrame) -> pd.DataFrame:
    """Observed speedup vs. the no-interaction (main-effects-only, multiplicative) prediction, per cell.

    At the centre every single optimization was measured, so the product of measured single speedups and the
    additive prediction 1 + sum(s_i - 1) are reported too.
    """
    rows = []
    for pid, g in cells[cells.status == "ok"].groupby("point_id"):
        g = g.dropna(subset=["output_tok_s"])
        X, y = x_matrix(list(g.config_bits)), np.log(g.output_tok_s.values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        d0 = float(np.exp(beta @ x_matrix(["0000000"])[0]))
        dflt = g[g.config_bits == "0000000"].output_tok_s
        d_obs = float(dflt.iloc[0]) if len(dflt) else math.nan
        singles = {b: float(g[g.config_bits == b].output_tok_s.iloc[0]) / d_obs
                   for b in ("1000000", "0100000", "0010000", "0001000", "0000100") if (g.config_bits == b).any()}
        for r in g.itertuples():
            model = float(np.exp(x_matrix([r.config_bits])[0] @ beta)) / d0
            obs = r.output_tok_s / d_obs
            on = [b for b in ("1000000", "0100000", "0010000", "0001000", "0000100")
                  if r.config_bits[b.index("1")] == "1"]
            have = bool(on) and all(b in singles for b in on)
            prod = math.prod(singles[b] for b in on) if have else math.nan
            add = 1 + sum(singles[b] - 1 for b in on) if have else math.nan
            # No-interaction baseline: the product of the measured single-optimization speedups where all were
            # measured at this point; otherwise the main-effects-only (multiplicative) model fitted to the point.
            indep = prod if have else model
            rows.append({"point_id": pid, "config_bits": r.config_bits, "config": names(r.config_bits),
                         "n_opts": len(on), "source": r.source, "speedup_observed": obs,
                         "speedup_independent": indep,
                         "independent_basis": "product of measured singles" if have else "main-effects-only model",
                         "ratio_observed_to_independent": obs / indep, "speedup_main_effects_model": model,
                         "speedup_product_of_singles": prod, "speedup_additive_singles": add,
                         "mechanism": mechanism(r)})
    df = pd.DataFrame(rows)
    df["verdict"] = np.select([df.ratio_observed_to_independent > 1.05, df.ratio_observed_to_independent < 0.95],
                              ["super-multiplicative (synergy)", "sub-multiplicative (overlap)"],
                              "multiplicative (independent)")
    return df


def mechanism(r: object) -> str:
    """One factual line per optimization that is on, from this cell's own telemetry."""
    b = r.config_bits
    parts = []
    if b[0] == "1":
        parts.append("quant: FP8 weights halve the weight bytes read per step")
    if b[1] == "1":
        kv = getattr(r, "kv_usage_max", math.nan)
        parts.append(f"kvq: FP8 KV halves KV bytes; KV cache {kv:.0%} full at most" if not math.isnan(kv)
                     else "kvq: FP8 KV halves KV bytes")
    if b[2] == "1":
        a, tps = getattr(r, "spec_accept_rate", math.nan), getattr(r, "tokens_per_step", math.nan)
        parts.append(f"spec: EAGLE-3 accepts {a:.0%} of drafts -> {tps:.2f} tokens per sequence per step"
                     if not math.isnan(a) else "spec: EAGLE-3 speculative decoding")
    if b[3] == "1":
        h = getattr(r, "prefix_hit_rate", math.nan)
        parts.append(f"prefix: {h:.0%} of prompt tokens served from cache (prefill skipped)"
                     if not math.isnan(h) else "prefix: prefix cache on")
    if b[4] == "1":
        m, run = getattr(r, "max_num_seqs", math.nan), getattr(r, "running_mean", math.nan)
        parts.append(f"batch: limit {m:.0f} vs {run:.0f} running -> " + ("limit never binds" if run < m - 1 else
                                                                       "limit binds")
                     if not (math.isnan(m) or math.isnan(run)) else "batch: tuned max_num_seqs")
    return "; ".join(parts) or "Default: all off"


# ====================================================================== 4. focused tables

def inv_cells() -> pd.DataFrame:
    c = pd.read_csv(RES / "investigate" / "report" / "cells.csv", dtype={"config_bits": str})
    c["exclusive"] = c["exclusive"].astype(bool)
    return c


def reuse_table(ic: pd.DataFrame) -> pd.DataFrame:
    pts = {"reuse_low": 0.0, "center@reuse=0.1": 0.10, "center@reuse=0.25": 0.25, "center": 0.5}
    c = ic[~ic.exclusive & ic.point_id.isin(pts)].copy()
    c["reuse"] = c.point_id.map(pts)
    hi = pd.read_csv(RES / "fast_clean" / "report" / "value_per_config.csv", dtype={"config_bits": str})
    rows = []
    for reuse, g in c.groupby("reuse"):
        t = g.set_index("config").tok_s
        rows.append({"reuse": reuse, "default": t["default"], "prefix": t["prefix"], "spec": t["spec"],
                     "spec+prefix": t["spec+prefix"], "source": "investigation (2 reps)"})
    # 90% reuse: measured singles + pair from the high_load_dealias investigation
    g = ic[~ic.exclusive & (ic.point_id == "reuse_high")].set_index("config").tok_s
    rows.append({"reuse": 0.9, "default": g["default"], "prefix": g["prefix"], "spec": g["spec"],
                 "spec+prefix": g["spec+prefix"], "source": "investigation (2 reps)"})
    df = pd.DataFrame(rows).sort_values("reuse")
    df["prefix_gain"] = df.prefix / df.default
    df["prefix_gain_with_spec"] = df["spec+prefix"] / df.spec
    df["spec_gain"] = df.spec / df.default
    df["spec_prefix_interaction"] = df["spec+prefix"] * df.default / (df.spec * df.prefix)
    del hi
    return df


def context_table(ic: pd.DataFrame) -> pd.DataFrame:
    pts = {"center": 2048, "center@context=4096": 4096, "center@context=6144": 6144, "context_high": 8192}
    c = ic[~ic.exclusive & ic.point_id.isin(pts) & ic.config.isin(["default", "spec"])].copy()
    c["context"] = c.point_id.map(pts)
    rows = []
    for ctx, g in c.groupby("context"):
        d, s = g[g.config == "default"].iloc[0], g[g.config == "spec"].iloc[0]
        pos = s.deep_spec_accept_by_pos
        pos = json.loads(pos.replace("'", '"')) if isinstance(pos, str) else {}
        rows.append({"context_tokens": ctx, "default_tok_s": d.tok_s, "spec_tok_s": s.tok_s,
                     "spec_speedup": s.tok_s / d.tok_s, "accept_rate": s.spec_accept_rate,
                     "tokens_per_step": s.tokens_per_step, "accept_pos1": pos.get("0"), "accept_pos2": pos.get("1"),
                     "accept_pos3": pos.get("2"), "default_tpot_p50_ms": d.tpot_p50_ms, "spec_tpot_p50_ms": s.tpot_p50_ms,
                     "default_ttft_p50_ms": d.ttft_p50_ms, "spec_ttft_p50_ms": s.ttft_p50_ms})
    return pd.DataFrame(rows)


def contamination_tables(fast: pd.DataFrame, ic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = {(r.point_id, r.config_bits): r.tok_s for r in ic[~ic.exclusive].itertuples() if "@" not in r.point_id}
    m = fast.copy()
    m["rerun_tok_s"] = [ref.get((p, b)) for p, b in zip(m.point_id, m.config_bits)]
    m = m[m.rerun_tok_s.notna()].copy()
    m["ratio_fast_to_rerun"] = m.output_tok_s / m.rerun_tok_s
    runs = m[["written_at", "gpu", "point_id", "config_bits", "rep", "output_tok_s", "rerun_tok_s",
              "ratio_fast_to_rerun", "contaminated", "gpu_mem_used_max_mb", "gpu_power_mean_w",
              "gpu_sm_clock_mean_mhz"]].sort_values("written_at")
    fast = fast.copy()
    fast["phase"] = np.where(fast.contaminated, "GPU0 from 11:24 UTC", np.where(fast.gpu == 0, "GPU0 before", "GPU1"))
    ph = fast.groupby("phase").agg(runs=("rep", "size"), device_mem_used_mb=("gpu_mem_used_max_mb", "median"),
                                   power_w=("gpu_power_mean_w", "median"), sm_clock_mhz=("gpu_sm_clock_mean_mhz", "median"),
                                   first=("written_at", "min"), last=("written_at", "max")).reset_index()
    rr = m.groupby("phase" if "phase" in m else m.contaminated.map({True: "GPU0 from 11:24 UTC", False: "clean"}))
    ph["median_ratio_to_rerun"] = ph.phase.map({
        "GPU0 from 11:24 UTC": m[m.contaminated].ratio_fast_to_rerun.median(),
        "GPU0 before": m[(m.gpu == 0) & ~m.contaminated].ratio_fast_to_rerun.median(),
        "GPU1": m[m.gpu == 1].ratio_fast_to_rerun.median()})
    del rr
    return runs, ph


def quality_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in ResultStore(RES / "fast" / "quality.jsonl").latest().values():
        if r.get("status") == "ok":
            rows.append({"numeric_bits": r["numeric_bits"], "names": names(r["numeric_bits"] + "0000"),
                         **{f"{k}_pts": v for k, v in r["scores"].items()}, "agreement_with_default": r.get("agreement"),
                         "samples": "trimmed: GSM8K 500, HumanEval 164, MMLU 4 subjects"})
    trimmed = pd.DataFrame(rows).sort_values("numeric_bits")
    paired = pd.read_csv(RES / "investigate" / "report" / "quality_paired.csv", dtype={"setting": str})
    return trimmed, paired


def final_policies(cells: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Best configuration per point under three accuracy policies (latency gates always apply)."""
    sig_fail = set(paired[paired.verdict == "significant drop beyond gate"].setting)   # e.g. {"110"}
    tested = set(paired.setting) | {"000"}

    def paired_ok(bits: str) -> tuple[bool, str]:
        nb = bits[:3]
        if nb in sig_fail or any(all(nb[i] == "1" for i in range(3) if s[i] == "1") for s in sig_fail):
            return False, "fails (significant drop)" if nb in sig_fail else "fails (contains a failing pair; inferred)"
        return True, "passes" if nb in tested else "untested (assumed pass)"

    rows = []
    ok = cells[(cells.status == "ok") & cells.latency_gates_pass]
    for pid in POINT_ORDER:
        g = ok[ok.point_id == pid]
        dflt = cells[(cells.point_id == pid) & (cells.config_bits == "0000000")].output_tok_s.iloc[0]
        for policy, mask in (("A: original trimmed evals, 1-pt gate", g.accuracy_gate != "fail"),
                             ("B: paired test (full GSM8K, 8 MMLU), 1-pt gate", g.config_bits.map(lambda b: paired_ok(b)[0])),
                             ("C: latency gates only", pd.Series(True, index=g.index))):
            c = g[mask].sort_values("speedup", ascending=False)
            if c.empty:
                continue
            b = c.iloc[0]
            rows.append({"point_id": pid, "policy": policy, "best_config": b.config_bits, "best_names": b.config_names,
                         "speedup": b.speedup, "tok_s": b.output_tok_s, "default_tok_s": dflt, "source": b.source,
                         "accuracy_status_B": paired_ok(b.config_bits)[1]})
    return pd.DataFrame(rows)


def binding_table(ic: pd.DataFrame) -> pd.DataFrame:
    prof = pd.read_csv(RES / "investigate" / "report" / "profiles.csv", dtype={"config_bits": str})
    prof["exclusive"] = prof.exclusive.astype(bool)
    b = ic[ic.exclusive & ic.point_id.isin(["center", "concurrency_low", "concurrency_high"])]
    t = b.merge(prof[prof.exclusive][["point_id", "config_bits", "gpu_busy_frac", "top_kernel"]],
                on=["point_id", "config_bits"], how="left")
    keep = ["point_id", "config", "tok_s", "running_mean", "deep_steps_per_s", "deep_step_ms", "deep_sm_util_mean",
            "deep_mem_util_mean", "deep_sm_clock_mean_mhz", "deep_power_mean_w", "gpu_busy_frac", "top_kernel"]
    t = t[[k for k in keep if k in t]].copy()
    t["power_cap_w"] = 400.0
    t["throttle_reason"] = "SW power cap (NVML reason 0x4), every sample"
    return t


def all_profiles() -> pd.DataFrame:
    p = pd.read_csv(RES / "investigate" / "report" / "profiles.csv", dtype={"config_bits": str})
    return p


# ====================================================================== main

def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in (DATA, FIG):
        d.mkdir(parents=True)
    fast, inv = load_runs()
    clean, missing = clean_runs(fast, inv)
    rep_dir = standard_report(clean)
    qtab = quality_table(ResultStore(RES / "fast" / "quality.jsonl").rows())
    cells = cell_table(clean, qtab)
    ic = inv_cells()

    # raw run tables
    save(fast.drop(columns=["t"]).assign(contaminated=fast.contaminated), "runs_fast_plan_all.csv")
    save(clean.drop(columns=["t", "contaminated"], errors="ignore"), "runs_combined_clean.csv")
    save(inv[inv.kind == "investigate"].drop(columns=["t"]), "runs_investigation.csv")
    save(cells.assign(missing=False), "cells_combined_clean.csv")
    save(pd.DataFrame(missing, columns=["point_id", "config_bits"]).assign(
        config=lambda d: d.config_bits.map(names), reason="contaminated in fast plan, not re-run"),
        "cells_missing_after_cleaning.csv")
    for f in ("main_effects.csv", "interactions.csv", "model_fit.csv", "shapley.csv", "final_configs.csv",
              "no_benefit.csv", "config_x_regime.csv"):
        shutil.copy(rep_dir / f, DATA / f"clean_{f}")
    comp = save(composition(cells), "composition_per_config.csv")
    inter = measured_interactions(cells.assign(exclusive=False, tok_s=cells.output_tok_s)
                                  [["point_id", "config_bits", "exclusive", "tok_s"]].dropna())
    inv_inter = pd.read_csv(RES / "investigate" / "report" / "interactions_measured.csv")
    inter = pd.concat([inter.assign(source="fast plan (clean cells)"), inv_inter.assign(source="investigation")])
    save(inter, "interactions_measured_2x2.csv")
    save(reuse_table(ic), "prefix_vs_reuse.csv")
    save(context_table(ic), "spec_vs_context.csv")
    cruns, cphase = contamination_tables(fast, ic)
    save(cruns, "contamination_matched_runs.csv")
    save(cphase, "contamination_by_phase.csv")
    trimmed, paired = quality_tables()
    save(trimmed, "quality_trimmed_fast_plan.csv")
    save(paired, "quality_paired_vs_default.csv")
    save(final_policies(cells, paired), "final_configs_by_policy.csv")
    save(binding_table(ic), "binding_check.csv")
    save(all_profiles(), "profiler_passes.csv")
    save(ic, "investigation_cells.csv")
    outlier = inv[(inv.kind == "investigate") & (inv.point_id == "concurrency_high") & (inv.config_bits == "0110000")]
    save(outlier[["exclusive", "rep", "gpu", "output_tok_s", "tpot_p99_ms", "spec_accept_rate"]], "outlier_rerun.csv")
    reported = pd.read_csv(RES / "fast" / "report" / "final_configs.csv").set_index("point_id").value.to_dict()
    save(pd.DataFrame([{"point_id": k, "best_value_as_first_reported": v} for k, v in reported.items()]),
         "fast_plan_as_first_reported.csv")
    from figures import build_all
    figs = build_all(DATA, FIG, reported)
    import docs
    docs.build(DATA, OUT)
    docs.data_dictionary(DATA, OUT)
    for f in (Path(__file__).resolve().parent / "static").glob("*.md"):   # hand-written docs (no numbers)
        shutil.copy(f, OUT / f.name)
    print(f"wrote {len(list(DATA.glob('*.csv')))} CSVs, {len(figs)} figures and "
          f"{len(list(OUT.glob('*.md'))) + len(list(OUT.glob('*.csv')))} docs to {OUT}")


if __name__ == "__main__":
    main()
