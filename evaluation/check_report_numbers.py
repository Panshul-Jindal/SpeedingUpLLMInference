"""Re-derive the numbers quoted in docs/Project_Report.pdf from the report-pack CSVs.

Usage (no GPU needed; pandas + numpy only):
    python evaluation/check_report_numbers.py [--data evaluation/report_pack/data]

Each check recomputes a value from run- or cell-level data and compares it with the value written in the
report (rounded as printed). Exit code 0 = every check passed. Check ids (e.g. S5.1-centre) name the report
section they back.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ["quant", "kvq", "spec", "prefix", "batch", "chunk", "par"]
POINTS = ["center", "context_low", "context_high", "concurrency_low", "concurrency_high", "reuse_low", "reuse_high"]


def bits(x) -> str:
    return str(int(x)).zfill(7)


class Checker:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, object, object, bool]] = []

    def eq(self, cid: str, what: str, report, observed, tol: float = 0.0) -> None:
        if isinstance(report, (int, float)) and isinstance(observed, (int, float, np.floating, np.integer)):
            ok = abs(float(observed) - float(report)) <= tol + 1e-12
        else:
            ok = report == observed
        self.rows.append((cid, what, report, observed, bool(ok)))

    def summary(self) -> int:
        w = max(len(r[0]) for r in self.rows)
        for cid, what, rep, obs, ok in self.rows:
            o = f"{obs:.4g}" if isinstance(obs, (float, np.floating)) else obs
            print(f"{'PASS' if ok else 'FAIL'}  {cid:<{w}}  {what}: report {rep}, data {o}")
        n_ok = sum(r[4] for r in self.rows)
        print(f"\n{n_ok}/{len(self.rows)} checks passed")
        return 0 if n_ok == len(self.rows) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="evaluation/report_pack/data")
    d = Path(ap.parse_args().data)
    rd = lambda f: pd.read_csv(d / f)  # noqa: E731
    c = Checker()

    fast, inv = rd("runs_fast_plan_all.csv"), rd("runs_investigation.csv")
    runs = pd.concat([fast, inv], ignore_index=True)
    clean_runs, cells = rd("runs_combined_clean.csv"), rd("cells_combined_clean.csv")
    cells["bits"] = cells.config_bits.map(bits)

    # ---- Section 4: what was run ------------------------------------------------------------------------
    c.eq("S4-runs", "phase-1 runs", 384, len(fast))
    c.eq("S4-runs", "phase-2 runs", 126, len(inv))
    c.eq("S4-runs", "all runs status ok", 510, int((runs.status == "ok").sum()))
    c.eq("S4-runs", "phase-1 cells", 128, fast.groupby(["point_id", "config_bits"]).ngroups)
    c.eq("S4-runs", "phase-2 cells", 60, inv.groupby(["point_id", "config_bits", "exclusive"]).ngroups)
    c.eq("S4-time", "phase-1 window (UTC)", "03:23-14:52", f"{fast.written_at.min()[11:16]}-{fast.written_at.max()[11:16]}")
    c.eq("S4-time", "phase-2 window (UTC)", "19:08-23:31", f"{inv.written_at.min()[11:16]}-{inv.written_at.max()[11:16]}")
    c.eq("S4-trace", "points with more than one trace hash", 0, int((runs.groupby("point_id").trace_hash.nunique() > 1).sum()))
    c.eq("S4-trace", "request errors in 510 runs", 0, int(runs.n_errors.sum()))
    c.eq("S4-trace", "wrong output lengths in 510 runs", 0, int(runs.n_wrong_output_len.sum()))
    default = runs[runs.config_bits == 0]
    c.eq("S2-base", "Default runs with prefix caching and chunked prefill off (engine-reported)", len(default),
         int((default.engine_info.str.contains("'chunked_prefill_enabled': False")
              & default.engine_info.str.contains("'prefix_caching_enabled': False")).sum()))
    c.eq("S2-base", "Default max_num_seqs reported by the engine", 256, int(default.max_num_seqs_effective.max()))
    c.eq("S8-git", "runs recording harness_git = nogit", 510, int((runs.harness_git == "nogit").sum()))
    prof = rd("profiler_passes.csv")
    c.eq("S4-prof", "profiler passes", 18, int((prof.status == "ok").sum()))

    # ---- Section 5.1: baseline vs best (policy A) -----------------------------------------------------
    table = {  # point: (default tok/s, tpot, ttft, best names, best tok/s, tpot, ttft, speedup)
        "center": (1774, 18.0, 1508, "spec+prefix", 3231, 16.6, 326, 1.82),
        "context_low": (4027, 7.9, 258, "spec+prefix", 7777, 8.6, 67, 1.93),
        "context_high": (495, 59.2, 1038, "prefix+batch", 743, 45.6, 1539, 1.50),
        "concurrency_low": (303, 6.3, 151, "spec+prefix", 753, 4.2, 122, 2.48),
        "concurrency_high": (2162, 57.7, 1586, "spec+prefix", 3602, 60.4, 891, 1.67),
        "reuse_low": (1776, 17.8, 1500, "spec+prefix", 2474, 22.6, 406, 1.39),
        "reuse_high": (1775, 17.7, 1785, "spec+prefix+batch", 4657, 12.1, 199, 2.62),
    }
    for p, (dt, dtp, dtt, bn, bt, btp, btt, sp) in table.items():
        g = cells[cells.point_id == p]
        df = g[g.bits == "0000000"].iloc[0]
        b = g.loc[g.value.idxmax()]
        c.eq(f"S5.1-{p}", "Default tok/s", dt, round(df.output_tok_s))
        c.eq(f"S5.1-{p}", "Default p99 TPOT ms", dtp, round(df.p99_tpot_ms, 1))
        c.eq(f"S5.1-{p}", "Default p99 TTFT ms", dtt, round(df.p99_ttft_ms))
        c.eq(f"S5.1-{p}", "best configuration (max Value)", bn, b.config_names)
        c.eq(f"S5.1-{p}", "best tok/s", bt, round(b.output_tok_s))
        c.eq(f"S5.1-{p}", "best p99 TPOT ms", btp, round(b.p99_tpot_ms, 1))
        c.eq(f"S5.1-{p}", "best p99 TTFT ms", btt, round(b.p99_ttft_ms))
        c.eq(f"S5.1-{p}", "best speedup", sp, round(b.value, 2))
    c.eq("S5.1-gates", "cells failing a latency gate", 0, int((~cells.latency_gates_pass).sum()))
    c.eq("S5.1-gates", "cells failing the accuracy gate (policy A)", 86,
         int(cells.failed_gates.fillna("").str.contains("accuracy").sum()))
    c.eq("S5.1-cv", "max rep CV of Default and best cells (%)", 1.53, round(100 * max(
        max(cells[(cells.point_id == p) & (cells.bits == "0000000")].tok_s_cv.max(),
            cells.loc[cells[cells.point_id == p].value.idxmax()].tok_s_cv) for p in POINTS), 2))
    pol = rd("final_configs_by_policy.csv")
    polB = {"center": 2.40, "context_low": 2.54, "context_high": 1.82, "concurrency_low": 3.30,
            "concurrency_high": 2.12, "reuse_low": 1.39, "reuse_high": 3.30}
    for p, v in polB.items():
        c.eq(f"S5.1-B-{p}", "policy B upper bound", v,
             round(float(pol[(pol.point_id == p) & pol.policy.str.startswith("B")].speedup.iloc[0]), 2))
    for p, v in {"center": 2.48, "context_high": 2.05}.items():
        c.eq(f"S5.1-C-{p}", "policy C (latency only)", v,
             round(float(pol[(pol.point_id == p) & pol.policy.str.startswith("C")].speedup.iloc[0]), 2))

    # ---- Section 5.2: contamination --------------------------------------------------------------------
    c.eq("S5.2", "contaminated runs", 71, int(fast.contaminated.sum()))
    c.eq("S5.2", "contaminated cells", 24, fast[fast.contaminated].groupby(["point_id", "config_bits"]).ngroups)
    g = fast.groupby(["point_id", "config_bits"])
    full = g.contaminated.all()
    cvs = [100 * np.std(g.get_group(k).output_tok_s, ddof=1) / np.mean(g.get_group(k).output_tok_s)
           for k in full[full].index]
    c.eq("S5.2", "fully contaminated cells", 23, len(cvs))
    c.eq("S5.2", "their within-cell CV min %", 0.2, round(min(cvs), 1))
    c.eq("S5.2", "their within-cell CV median %", 2.1, round(float(np.median(cvs)), 1))
    c.eq("S5.2", "their within-cell CV max %", 3.8, round(max(cvs), 1))
    c.eq("S5.2", "median rep CV of the cleaned cells %", 0.2, round(100 * float(cells.tok_s_cv.median()), 1))
    m = rd("contamination_matched_runs.csv")
    c.eq("S5.2", "contaminated/clean ratio min", 0.56, round(m[m.contaminated].ratio_fast_to_rerun.min(), 2))
    c.eq("S5.2", "contaminated/clean ratio max", 0.69, round(m[m.contaminated].ratio_fast_to_rerun.max(), 2))
    c.eq("S5.2", "uncontaminated matched runs", 40, int((~m.contaminated).sum()))
    c.eq("S5.2", "their max deviation from re-run %", 2.2, round(100 * (m[~m.contaminated].ratio_fast_to_rerun - 1).abs().max(), 1))
    ph = rd("contamination_by_phase.csv").set_index("phase")
    for phase, (n, ratio, mem, clk, pw) in {"GPU0 before": (124, 0.992, 87498, 1172, 396),
                                            "GPU0 from 11:24 UTC": (71, 0.596, 91917, 1476, 387),
                                            "GPU1": (189, 1.006, 87985, 1171, 395)}.items():
        r = ph.loc[phase]
        c.eq("S5.2-tab", f"{phase} runs", n, int(r.runs))
        c.eq("S5.2-tab", f"{phase} median ratio", ratio, round(r.median_ratio_to_rerun, 3))
        c.eq("S5.2-tab", f"{phase} device MB", mem, round(r.device_mem_used_mb))
        c.eq("S5.2-tab", f"{phase} SM MHz", clk, round(r.sm_clock_mhz))
        c.eq("S5.2-tab", f"{phase} power W", pw, round(r.power_w))
    first = rd("fast_plan_as_first_reported.csv").set_index("point_id").best_value_as_first_reported
    c.eq("S5.2", "first-reported best at 90% reuse", 4.50, round(first["reuse_high"], 2))
    c.eq("S5.2", "first-reported best at high concurrency", 2.64, round(first["concurrency_high"], 2))
    c.eq("S5.2", "cells replaced / excluded / added", "8/16/16",
         f"{(cells.source == 're-run').sum()}/{len(rd('cells_missing_after_cleaning.csv'))}/{(cells.source == 'investigation').sum()}")

    # ---- Section 5.3: main effects, fit, Shapley, no-benefit -----------------------------------------
    me = rd("clean_main_effects.csv").pivot(index="name", columns="point_id", values="multiplier")
    effects = {"quant": [1.27, 1.27, 1.23, 1.34, 1.27, 1.22, 1.25], "kvq": [1.04, 1.00, 1.13, 1.00, 1.07, 1.04, 1.05],
               "spec": [1.49, 1.82, 1.00, 2.31, 1.21, 1.36, 1.57], "prefix": [1.29, 1.07, 1.39, 1.04, 1.35, 1.00, 1.70],
               "batch": [1.00, 1.00, 1.02, 1.00, 1.01, 1.00, 1.00]}
    for name, vals in effects.items():
        for p, v in zip(POINTS, vals):
            c.eq(f"S5.3-eff-{p}", f"{name} main effect", v, round(me.loc[name, p], 2))
    fit = rd("clean_model_fit.csv")
    c.eq("S5.3-fit", "min R^2 over points", 0.9976, round(fit.r2.min(), 4))
    sh = rd("clean_shapley.csv")
    for p, name, share in [("center", "spec", 59), ("center", "prefix", 41), ("reuse_low", "spec", 96),
                           ("concurrency_high", "prefix", 61), ("concurrency_low", "spec", 95),
                           ("context_high", "prefix", 89)]:
        c.eq(f"S5.3-shap-{p}", f"Shapley share of {name} %", share,
             round(100 * float(sh[(sh.point_id == p) & (sh.name == name)].share.iloc[0])))
    c.eq("S5.3-shap", "points with exact Shapley shares", 3, sh[sh.exact].point_id.nunique())
    nb = rd("clean_no_benefit.csv")
    c.eq("S5.3-nb", "no-benefit cases (gain < 2%), each with a reason", 11, int(nb.reason.notna().sum()))
    c.eq("S5.3-nb", "batch max multiplier", 1.02, round(me.loc["batch"].max(), 2))

    # ---- Section 5.4: composition ----------------------------------------------------------------------
    co = rd("composition_per_config.csv")
    x = co[(co.n_opts >= 2) & (co.independent_basis == "product of measured singles")]
    r = x.ratio_observed_to_independent
    c.eq("S5.4", "combinations with all singles measured", 50, len(x))
    c.eq("S5.4", "more than 5% below product", 0, int((r < 0.95).sum()))
    c.eq("S5.4", "more than 5% above product", 18, int((r > 1.05).sum()))
    c.eq("S5.4", "all of those contain spec", True, bool(x[r > 1.05].config.str.contains("spec").all()))
    c.eq("S5.4", "max observed/product", 1.31, round(r.max(), 2))
    c.eq("S5.4", "mean observed/product with spec", 1.111, round(r[x.config.str.contains("spec")].mean(), 3))
    c.eq("S5.4", "mean observed/product without spec", 1.013, round(r[~x.config.str.contains("spec")].mean(), 3))
    ce = co[(co.point_id == "center") & (co.n_opts >= 2)]
    em = (ce.speedup_observed / ce.speedup_product_of_singles - 1).abs() * 100
    ea = (ce.speedup_observed / ce.speedup_additive_singles - 1).abs() * 100
    c.eq("S5.4", "centre multiplicative median error %", 4.8, round(em.median(), 1))
    c.eq("S5.4", "centre multiplicative max error %", 21.6, round(em.max(), 1))
    c.eq("S5.4", "centre additive median error %", 9.1, round(ea.median(), 1))
    c.eq("S5.4", "centre additive max error %", 37.4, round(ea.max(), 1))
    it = rd("interactions_measured_2x2.csv")
    qs = it[(it.point_id == "center") & (it["with"] == "default") & (it.pair == "quant:spec")].iloc[0]
    c.eq("S5.4", "quant alone at centre", 1.20, round(qs.gain_a, 2))
    c.eq("S5.4", "quant on top of spec", 1.34, round(qs.gain_ab / qs.gain_b, 2))
    c.eq("S5.4", "quant x spec interaction", 1.115, round(qs.interaction, 3))
    ic = rd("investigation_cells.csv")
    cen = ic[(ic.point_id == "center") & (~ic.exclusive)].set_index("config")
    c.eq("S5.4", "tokens per engine step, Default", 284, round(cen.loc["default", "deep_tokens_per_step"]))
    c.eq("S5.4", "tokens per engine step, spec", 928, round(cen.loc["spec", "deep_tokens_per_step"]))
    pr = rd("prefix_vs_reuse.csv")
    for col, vals in {"prefix_gain": [1.01, 1.02, 1.11, 1.20, 1.52], "prefix_gain_with_spec": [1.02, 1.03, 1.12, 1.35, 1.85],
                      "spec_prefix_interaction": [1.011, 1.015, 1.014, 1.122, 1.217]}.items():
        nd = 3 if col == "spec_prefix_interaction" else 2
        for reuse, v in zip(pr.reuse, vals):
            c.eq(f"S5.4-reuse{int(reuse * 100)}", col, v, round(float(pr[pr.reuse == reuse][col].iloc[0]), nd))
    hi = ic[ic.point_id == "reuse_high"].set_index("config").deep_decode_s
    c.eq("S5.4", "decode-time cut by prefix at 90% reuse, no spec %", 27, round(100 * (1 - hi["prefix"] / hi["default"])))
    c.eq("S5.4", "decode-time cut by prefix at 90% reuse, with spec %", 45, round(100 * (1 - hi["spec+prefix"] / hi["spec"])))

    # ---- Section 5.5: EAGLE-3 vs context ---------------------------------------------------------------
    sc = rd("spec_vs_context.csv").set_index("context_tokens")
    for ctx, (spd, acc, tps, a2) in {2048: (1.35, 0.78, 3.35, 0.76), 4096: (1.14, 0.37, 2.12, 0.16),
                                     6144: (1.00, 0.19, 1.56, 0.01), 8192: (1.09, 0.10, 1.31, 0.00)}.items():
        r = sc.loc[ctx]
        c.eq(f"S5.5-{ctx}", "spec speedup", spd, round(r.spec_speedup, 2))
        c.eq(f"S5.5-{ctx}", "acceptance", acc, round(r.accept_rate, 2), tol=0.005)
        c.eq(f"S5.5-{ctx}", "tokens per step", tps, round(r.tokens_per_step, 2))
        c.eq(f"S5.5-{ctx}", "2nd-draft acceptance", a2, round(r.accept_pos2, 2))
    c.eq("S5.5", "median TTFT Default -> spec at 2048 (ms)", "1046->183",
         f"{round(sc.loc[2048].default_ttft_p50_ms)}->{round(sc.loc[2048].spec_ttft_p50_ms)}")

    # ---- Section 5.6: paired accuracy ---------------------------------------------------------------
    q = rd("quality_paired_vs_default.csv")
    q["setting"] = q.setting.map(lambda s: str(int(s)).zfill(3))
    for (s, task), (base, acc, dp, lo, hi_, p) in {
        ("100", "gsm8k"): (69.6, 69.0, -0.61, -2.6, 1.4, 0.61), ("010", "gsm8k"): (69.6, 69.4, -0.23, -2.3, 1.8, 0.89),
        ("110", "gsm8k"): (69.6, 67.0, -2.58, -4.9, -0.3, 0.03), ("001", "gsm8k"): (69.6, 69.2, -0.38, -1.4, 0.6, 0.56),
        ("100", "humaneval"): (63.4, 61.6, -1.83, -5.8, 2.1, 0.55), ("010", "humaneval"): (63.4, 62.2, -1.22, -5.7, 3.3, 0.79),
        ("110", "humaneval"): (63.4, 62.2, -1.22, -6.0, 3.6, 0.80), ("001", "humaneval"): (63.4, 63.4, 0.00, 0.0, 0.0, 1.00),
        ("100", "mmlu"): (65.7, 65.9, 0.18, -0.9, 1.2, 0.83), ("010", "mmlu"): (65.7, 65.3, -0.47, -1.5, 0.6, 0.45),
        ("110", "mmlu"): (65.7, 65.3, -0.41, -1.6, 0.7, 0.55), ("001", "mmlu"): (65.7, 65.6, -0.12, -0.4, 0.2, 0.73),
    }.items():
        r = q[(q.setting == s) & (q.task == task)].iloc[0]
        tag = f"S5.6-{r.names}-{task}"
        c.eq(tag, "Default %", base, round(r.base_acc, 1))
        c.eq(tag, "setting %", acc, round(r.acc, 1))
        c.eq(tag, "delta pts", dp, round(r.diff_pts, 2))
        c.eq(tag, "CI low", lo, round(r.ci95_lo, 1))
        c.eq(tag, "CI high", hi_, round(r.ci95_hi, 1))
        c.eq(tag, "McNemar p", p, round(r.mcnemar_p, 2))
    c.eq("S5.6", "GSM8K questions changed by spec", 47,
         int(q[(q.setting == "001") & (q.task == "gsm8k")].eval("lost + gained").iloc[0]))

    # ---- Section 5.7: binding constraint and slot independence ---------------------------------------
    c.eq("S5.7", "GPU-busy fraction min %", 96, round(100 * prof.gpu_busy_frac.min()))
    c.eq("S5.7", "GPU-busy fraction max %", 99, round(100 * prof.gpu_busy_frac.max()))
    c.eq("S5.7", "investigation runs with SW power-cap throttle (NVML reason 0x4)", 126,
         int((inv.deep_throttle_reasons.astype(int) & 0x4 != 0).sum()))
    c.eq("S5.7", "mean power min W", 394, math.floor(inv.deep_power_mean_w.min()))
    c.eq("S5.7", "mean power max W", 400, round(inv.deep_power_mean_w.max()))
    dcl = cells[cells.bits == "0000000"].set_index("point_id").sm_clock_mhz
    c.eq("S5.7", "Default SM clock at 2 in flight (MHz)", 1493, round(dcl["concurrency_low"]))
    c.eq("S5.7", "Default SM clock at 128 in flight (MHz)", 885, round(dcl["concurrency_high"]))
    o = rd("outlier_rerun.csv")
    busy, idle = o[~o.exclusive].output_tok_s, o[o.exclusive].output_tok_s
    c.eq("S5.7", "neighbour busy: range tok/s", "2690-2740", f"{round(busy.min())}-{round(busy.max())}")
    c.eq("S5.7", "neighbour idle: range tok/s", "2666-2713", f"{round(idle.min())}-{round(idle.max())}")
    c.eq("S5.7", "both CVs below 1%", True, bool(max(busy.std() / busy.mean(), idle.std() / idle.mean()) < 0.01))

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
