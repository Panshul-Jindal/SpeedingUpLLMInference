"""Report for profile `investigate`: one section per investigation, built from the deep-telemetry rows,
plus a paired accuracy test from lm-eval's per-question samples. Runs on partial data; never crashes."""
from __future__ import annotations

import json
import logging
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness import optimizations as opt
from harness.config import REPO_ROOT, load_env, load_experiment, load_profile
from harness.results import ResultStore

log = logging.getLogger(__name__)

CELL = ["point_id", "config_bits", "exclusive"]


def _geo(s: pd.Series) -> float:
    s = s.dropna()
    s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else math.nan


def _first(s: pd.Series) -> Any:
    s = s.dropna()
    return s.iloc[0] if len(s) else None


def cell_table(meas: pd.DataFrame, peaks: Any, fast: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (point, config, exclusive): throughput, latency and the deep-telemetry means."""
    if meas.empty:
        return pd.DataFrame()
    ok = meas[meas.status == "ok"].copy()
    num = [c for c in ok.columns if c.startswith("deep_") and pd.api.types.is_numeric_dtype(ok[c])]
    agg: dict[str, Any] = {"output_tok_s": [_geo, lambda s: s.std() / s.mean() if len(s) > 1 else math.nan, "count"]}
    for c in ["tpot_p50_ms", "tpot_p99_ms", "ttft_p50_ms", "ttft_p99_ms", "spec_accept_rate", "tokens_per_step",
              "prefix_hit_rate", "running_mean", "kv_usage_max", "preemptions", "gpu_util_mean", *num]:
        if c in ok:
            agg[c] = "mean"
    t = ok.groupby(CELL).agg(agg)
    t.columns = ["tok_s", "tok_s_cv", "reps_ok", *[c for c in agg if c != "output_tok_s"]]
    t = t.reset_index()
    if "deep_spec_accept_by_pos" in ok:
        t = t.merge(ok.groupby(CELL)["deep_spec_accept_by_pos"].agg(_first).reset_index(), on=CELL, how="left")
    t.insert(2, "config", t.config_bits.map(lambda b: opt.names(opt.from_bits(b))))
    if peaks is not None:
        if "deep_read_gbps" in t and peaks.mem_bandwidth_gbps:
            t["deep_read_frac_peak"] = t.deep_read_gbps / peaks.mem_bandwidth_gbps
        if "deep_tflops" in t and peaks.bf16_tflops:
            fp8 = t.config_bits.str[0] == "1"
            t["deep_tflops_frac_peak"] = t.deep_tflops / np.where(fp8, peaks.fp8_tflops or np.nan, peaks.bf16_tflops)
    if fast is not None and not fast.empty:
        f = fast[fast.status == "ok"].groupby(["point_id", "config_bits"]).output_tok_s.agg(_geo).rename("fast_tok_s")
        t = t.merge(f.reset_index(), on=["point_id", "config_bits"], how="left")
        t["vs_fast"] = t.tok_s / t.fast_tok_s
    return t


def profile_table(prof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in prof.itertuples():
        p = r.profile if isinstance(getattr(r, "profile", None), dict) else {}
        tk, tc = p.get("top_kernels") or [], p.get("top_cpu_ops") or []
        rows.append({"point_id": r.point_id, "config_bits": r.config_bits, "exclusive": r.exclusive,
                     "config": opt.names(opt.from_bits(r.config_bits)), "status": r.status,
                     "gpu_busy_frac": p.get("gpu_busy_frac"), "span_ms": p.get("span_ms"),
                     "cpu_step_ms_mean": p.get("cpu_step_ms_mean"),
                     "top_kernel": f"{tk[0]['name']} ({tk[0]['total_ms']} ms)" if tk else None,
                     "top_cpu_op": f"{tc[0]['name']} ({tc[0]['total_ms']} ms)" if tc else None,
                     "error": getattr(r, "error", None)})
    return pd.DataFrame(rows)


def measured_interactions(cells: pd.DataFrame) -> pd.DataFrame:
    """I_AB = T_AB * T_0 / (T_A * T_B) wherever all four cells were run (shared runs only)."""
    c = cells[~cells.exclusive.astype(bool)]
    thr = {(r.point_id, opt.from_bits(r.config_bits)): r.tok_s for r in c.itertuples()}
    rows = []
    for pid in sorted(c.point_id.unique()):
        masks = {m for (p, m) in thr if p == pid}
        for base in sorted(masks, key=lambda m: bin(m).count("1")):
            for a in opt.on_numbers(opt.ALL_MASK):
                for b in opt.on_numbers(opt.ALL_MASK):
                    if a >= b or base & (opt.bit(a) | opt.bit(b)):
                        continue
                    need = [base, base | opt.bit(a), base | opt.bit(b), base | opt.bit(a) | opt.bit(b)]
                    if all((pid, m) in thr for m in need):
                        t0, ta, tb, tab = (thr[(pid, m)] for m in need)
                        rows.append({"point_id": pid, "with": opt.names(base),
                                     "pair": f"{opt.OPTIMIZATIONS[a - 1].key}:{opt.OPTIMIZATIONS[b - 1].key}",
                                     f"gain_a": ta / t0, "gain_b": tb / t0, "gain_ab": tab / t0,
                                     "interaction": tab * t0 / (ta * tb)})
    return pd.DataFrame(rows)


def outlier_table(meas: pd.DataFrame, results_dir: Path, bin_s: float = 10.0) -> pd.DataFrame:
    """Per rep: throughput, latency and the coefficient of variation of throughput across 10 s bins."""
    rows = []
    for r in meas.itertuples():
        cv = None
        f = results_dir / "runs" / r.run_key / "requests.parquet"
        if f.exists():
            q = pd.read_parquet(f)
            q = q[q.in_window & q.ok]
            if len(q):
                b = ((q.end_t - q.end_t.min()) // bin_s).astype(int)
                per = q.groupby(b).n_out.sum() / bin_s
                cv = float(per.std() / per.mean()) if len(per) > 2 else None
        rows.append({"point_id": r.point_id, "config": opt.names(opt.from_bits(r.config_bits)),
                     "exclusive": r.exclusive, "rep": r.rep, "status": r.status, "tok_s": r.output_tok_s,
                     "tpot_p99_ms": getattr(r, "tpot_p99_ms", None), "within_run_cv_10s": cv,
                     "engine_core_cpu_pct": getattr(r, "deep_cpu_engine_core_pct", None),
                     "host_cpu_pct": getattr(r, "deep_cpu_host_pct", None),
                     "sm_clock_mhz": getattr(r, "deep_sm_clock_mean_mhz", None),
                     "throttle": getattr(r, "deep_throttle_reasons", None)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- paired quality

def _samples(qdir: Path, label: str, task: str) -> list[Path]:
    return sorted((qdir / label).rglob(f"samples_{task}_*.jsonl"))


def per_question(qdir: Path, label: str, task: str, metric: str) -> dict[str, float]:
    """{doc key: score} from lm-eval --log_samples output; metric is e.g. 'exact_match,strict-match'."""
    name, _, filt = metric.partition(",")
    out: dict[str, float] = {}
    for f in _samples(qdir, label, task):
        # Iterate the file, not str.splitlines(): model text may hold \x0c, \x85 or \u2028, which splitlines()
        # also treats as line breaks, cutting a JSON record in two.
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if filt and d.get("filter") not in (filt, None):
                    continue
                if name in d:
                    out[f"{task}:{d.get('doc_id')}"] = float(d[name])
    return out


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


def paired_quality(qdir: Path, quality_cfg: Any, labels: list[str], gate: float) -> pd.DataFrame:
    tasks = [(t, [t], quality_cfg.metrics.get(t)) for t in quality_cfg.tasks if t != "mmlu"]
    if "mmlu" in quality_cfg.tasks:
        tasks.append(("mmlu", list(quality_cfg.mmlu_subtasks), quality_cfg.metrics.get("mmlu", "acc,none")))
    rows = []
    for task, subs, metric in tasks:
        base: dict[str, float] = {}
        for sub in subs:
            base.update(per_question(qdir, "000", sub, metric))
        for lab in labels:
            if lab == "000":
                continue
            other: dict[str, float] = {}
            for sub in subs:
                other.update(per_question(qdir, lab, sub, metric))
            keys = sorted(set(base) & set(other))
            if not keys:
                rows.append({"setting": lab, "task": task, "n": 0, "verdict": "no per-question samples"})
                continue
            x = np.array([base[k] for k in keys])
            y = np.array([other[k] for k in keys])
            d = 100.0 * (y - x)
            diff, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else math.nan
            b = int(((x >= 0.5) & (y < 0.5)).sum())     # right before, wrong now
            c = int(((x < 0.5) & (y >= 0.5)).sum())
            p = mcnemar_p(b, c)
            lo, hi = diff - 1.96 * se, diff + 1.96 * se
            if p < 0.05 and -diff > gate:
                verdict = "significant drop beyond gate"
            elif -lo <= gate:
                verdict = "within gate (CI excludes a larger drop)"
            elif p >= 0.05:
                verdict = "not significant; CI too wide to confirm the gate"
            else:
                verdict = "significant change within gate"
            rows.append({"setting": lab, "names": opt.names(opt.from_bits(lab + "0000")), "task": task,
                         "n": len(keys), "base_acc": 100.0 * x.mean(), "acc": 100.0 * y.mean(), "diff_pts": diff,
                         "ci95_lo": lo, "ci95_hi": hi, "lost": b, "gained": c, "mcnemar_p": p, "verdict": verdict})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def _md(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".3g") -> str:
    if df is None or df.empty:
        return "_no data yet_\n"
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"


CORE = ["point_id", "config", "exclusive", "reps_ok", "tok_s", "tok_s_cv", "fast_tok_s", "vs_fast", "tpot_p50_ms",
        "tpot_p99_ms", "ttft_p50_ms"]
DEEP = ["deep_steps_per_s", "deep_step_ms", "deep_tokens_per_step", "deep_cpu_engine_core_pct",
        "deep_cpu_api_server_pct", "deep_cpu_host_pct", "deep_sm_util_mean", "deep_mem_util_mean",
        "deep_sm_clock_mean_mhz", "deep_read_frac_peak", "deep_tflops_frac_peak"]
FOCUS = {
    "spec_prefix_no_reuse": ["prefix_hit_rate", "deep_cached_prompt_frac", "spec_accept_rate", "deep_prefill_s",
                             "deep_decode_s"],
    "spec_long_context": ["spec_accept_rate", "deep_spec_accept_by_pos", "tokens_per_step", "deep_decode_s"],
    "high_load_dealias": ["running_mean", "kv_usage_max", "preemptions", "prefix_hit_rate", "spec_accept_rate"],
    "binding_check": ["deep_read_gbps", "deep_tflops", "running_mean", "kv_usage_max"],
}


def analyze_investigation(results_dir: Path, profile_name: str) -> Path:
    out = results_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    prof = load_profile(profile_name)
    exp = load_experiment(prof)
    errors: list[str] = []
    rows = list(ResultStore(results_dir / "investigate.jsonl").latest().values())
    df = pd.DataFrame(rows)
    if df.empty:
        (out / "investigation.md").write_text(f"# Investigation report\n\nNo results in {results_dir}.\n")
        return out
    for c in ("exclusive", "investigations", "status"):
        if c not in df:
            df[c] = None
    df["exclusive"] = df.exclusive.fillna(False).astype(bool)
    meas, profs = df[df.kind == "investigate"], df[df.kind == "profile"]
    env = load_env(str(df.env.iloc[0])) if "env" in df else None
    fast_path = REPO_ROOT / "results" / "fast" / "runs.jsonl"
    fast = ResultStore(fast_path).frame() if fast_path.exists() else None

    def safe(name: str, fn: Any, *a: Any) -> pd.DataFrame:
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
            return pd.DataFrame()

    cells = safe("cells", cell_table, meas, env.peaks if env else None, fast)
    ptab = safe("profiles", profile_table, profs)
    inter = safe("interactions", measured_interactions, cells) if not cells.empty else pd.DataFrame()
    labels = exp.quality.settings or []
    paired = safe("paired_quality", paired_quality, results_dir / "quality", exp.quality, labels,
                  exp.gates.max_accuracy_drop)
    for name, t in (("cells", cells), ("profiles", ptab), ("interactions_measured", inter),
                    ("quality_paired", paired)):
        t.to_csv(out / f"{name}.csv", index=False)

    md = [f"# Investigation report (`{profile_name}`)\n",
          f"{len(meas)} measured runs ({(meas.status == 'ok').sum()} ok), {len(profs)} profiler passes "
          f"({(profs.status == 'ok').sum() if len(profs) else 0} ok). `vs_fast` = throughput / same cell in "
          f"the fast plan. CPU % is of one core (host: of all cores).\n"]
    for inv in prof.investigations:
        md.append(f"## {inv.id}\n\n{inv.why}\n")
        mine = meas[meas.investigations.apply(lambda v, i=inv.id: isinstance(v, list) and i in v)]
        keys = set(map(tuple, mine[CELL].astype(str).values)) if not mine.empty else set()
        sub = cells[cells[CELL].astype(str).apply(tuple, axis=1).isin(keys)] if not cells.empty and keys else pd.DataFrame()
        if inv.id.startswith("kvq_spec_outlier"):
            md.append(_md(safe("outlier", outlier_table, mine, results_dir)))
        else:
            md.append(_md(sub, CORE + FOCUS.get(inv.id, []) + DEEP))
        if inv.id == "high_load_dealias":
            md.append("Measured interaction ratios (1 = independent):\n\n" + _md(inter))
        pmine = ptab[ptab.apply(lambda r: (r.point_id, r.config_bits) in
                                {(a, b) for a, b, _ in keys}, axis=1)] if not ptab.empty and keys else pd.DataFrame()
        if not pmine.empty:
            md.append("Profiler passes (GPU-busy fraction over the recorded steps):\n\n" + _md(pmine.drop(
                columns=["config_bits"])))
    md.append("## Paired accuracy vs Default\n\nSame questions for both settings; `lost` = right with Default, "
              f"wrong with the setting. Gate: drop ≤ {exp.gates.max_accuracy_drop} pt.\n\n" + _md(paired))
    if errors:
        md.append("## Analysis errors\n\n```\n" + "\n".join(errors) + "\n```\n")
    (out / "investigation.md").write_text("\n".join(md))
    return out
