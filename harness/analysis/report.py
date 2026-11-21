"""Build every report file in results/<profile>/report/. Runs on partial data and never crashes."""
from __future__ import annotations

import logging
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness import optimizations as opt
from harness.analysis.binding import binding_constraint
from harness.analysis.effects import INSUFFICIENT, Fit, fit_point, interactions_table, main_effects_table, \
    model_fit_table
from harness.analysis.shapley import shapley_for_config
from harness.analysis.value import cell_values, final_configs, per_run_values, quality_table
from harness.config import ExperimentCfg, active_optimizations, load_env, load_experiment, load_profile, \
    slot_envs
from harness.regimes import CENTER, all_points
from harness.results import ResultStore

log = logging.getLogger(__name__)

REPORT_FILES = ("config_x_regime.csv", "value_per_config.csv", "main_effects.csv", "interactions.csv",
                "model_fit.csv", "shapley.csv", "final_configs.csv", "no_benefit.csv", "summary.md")


def _point_order(exp: ExperimentCfg) -> list[str]:
    return [p.point_id for p in all_points(exp.regimes)]


def _safe(name: str, errors: list[str], fn: Any, *args: Any, empty_cols: list[str] | None = None) -> pd.DataFrame:
    try:
        out = fn(*args)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)
    except Exception as e:  # noqa: BLE001 - analysis never crashes; it reports what failed
        log.error("analysis step %s failed: %s", name, e)
        errors.append(f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        return pd.DataFrame(columns=empty_cols or ["status"])


def analyze(results_dir: Path, profile_name: str) -> Path:
    out = results_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    profile = load_profile(profile_name)
    exp = load_experiment(profile)
    gates, acfg = exp.gates, exp.analysis
    runs = ResultStore(results_dir / "runs.jsonl").frame()
    if not runs.empty and "kind" in runs:
        runs = runs[runs["kind"] == "run"]
    qrows = ResultStore(results_dir / "quality.jsonl").rows()
    qtab = quality_table(qrows)

    if runs.empty:
        for f in REPORT_FILES[:-1]:
            pd.DataFrame(columns=["status"]).to_csv(out / f, index=False)
        (out / "summary.md").write_text(f"# Report: {profile_name}\n\nNo run results found in {results_dir}.\n")
        return out
    for col in ("output_tok_s", "tpot_p99_ms", "ttft_p99_ms", "error"):
        if col not in runs:
            runs[col] = np.nan
    envs = sorted(runs["env"].unique())
    # the env as one cell sees it (gpus_per_cell); binding estimates and the design use this
    env_cfgs = {e: slot_envs(load_env(e), exp.run)[0] for e in envs}
    order = _point_order(exp)

    # Value per run and per cell
    per_run = _safe("value_per_config", errors, per_run_values, runs, qtab, gates)
    per_run.to_csv(out / "value_per_config.csv", index=False)
    cells = _safe("cell_values", errors, cell_values, runs, qtab, gates)
    finals = _safe("final_configs", errors, final_configs, cells) if not cells.empty else pd.DataFrame()

    # config x regime
    try:
        piv = cells.pivot_table(index=["env", "config_bits", "config_names"], columns="point_id", values="value",
                                aggfunc="first")
        piv = piv[[p for p in order if p in piv.columns]].reset_index()
        if not finals.empty:
            frow = []
            for env in envs:
                f = finals[finals["env"] == env].set_index("point_id")
                r = {"env": env, "config_bits": "FINAL", "config_names": "final configuration per point"}
                r.update({p: f"{f.loc[p, 'final_config']} ({f.loc[p, 'value']:.3f})" for p in f.index})
                frow.append(r)
            piv = pd.concat([piv, pd.DataFrame(frow)], ignore_index=True)
        piv.to_csv(out / "config_x_regime.csv", index=False)
    except Exception as e:  # noqa: BLE001
        errors.append(f"config_x_regime: {e}")
        pd.DataFrame(columns=["status"]).to_csv(out / "config_x_regime.csv", index=False)

    # Effects per (env, point)
    fits: list[Fit] = []
    ok_runs = runs[(runs["status"] == "ok") & runs["output_tok_s"].notna()]
    for env in envs:
        act = active_optimizations(env_cfgs[env], exp.run, profile)
        for pid in [p for p in order if p in set(runs.loc[runs["env"] == env, "point_id"])]:
            try:
                fits.append(fit_point(ok_runs[(ok_runs["env"] == env) & (ok_runs["point_id"] == pid)], env, pid, act))
            except Exception as e:  # noqa: BLE001
                errors.append(f"fit {env}/{pid}: {e}")
                fits.append(Fit(env, pid, act))
    fit_by = {(f.env, f.point_id): f for f in fits}
    main = _safe("main_effects", errors, main_effects_table, fits)
    main.to_csv(out / "main_effects.csv", index=False)
    _safe("model_fit", errors, model_fit_table, fits).to_csv(out / "model_fit.csv", index=False)
    cell_thr = {(r.env, r.point_id, opt.from_bits(r.config_bits)): r.output_tok_s
                for r in cells.itertuples() if r.status == "ok" and not math.isnan(r.output_tok_s)}
    inter = _safe("interactions", errors, interactions_table, fits, cell_thr)
    inter.to_csv(out / "interactions.csv", index=False)

    # Shapley + binding for each final configuration
    shap_rows, final_rows = [], []
    tel_cols = ["output_tok_s", "req_s", "running_mean", "running_max", "waiting_mean", "kv_usage_max",
                "preemptions", "prefix_hit_rate", "max_num_seqs_effective", "spec_accept_rate", "tokens_per_step"]
    for f in finals.itertuples() if not finals.empty else []:
        env_cfg = env_cfgs[f.env]
        fmask = opt.from_bits(f.final_config)
        fit = fit_by.get((f.env, f.point_id))
        try:
            def measured(m: int, env: str = f.env, pid: str = f.point_id) -> float | None:
                t = cell_thr.get((env, pid, m))
                return math.log(t) if t else None
            model = fit.predict_log if fit is not None and fit.ok else None
            phi, exact, status = shapley_for_config(fmask, measured, model)
            total = sum(phi.values())
            for n, v in phi.items():
                shap_rows.append({"env": f.env, "point_id": f.point_id, "final_config": f.final_config,
                                  "optimization": n, "name": opt.OPTIMIZATIONS[n - 1].key, "phi_log": v,
                                  "share": v / total if total else math.nan, "total_log_gain": total,
                                  "exact": exact, "label": "exact" if exact else "approximate", "status": status})
            if not phi:
                shap_rows.append({"env": f.env, "point_id": f.point_id, "final_config": f.final_config,
                                  "status": status})
        except Exception as e:  # noqa: BLE001
            errors.append(f"shapley {f.env}/{f.point_id}: {e}")
        try:
            rr = ok_runs[(ok_runs["env"] == f.env) & (ok_runs["point_id"] == f.point_id)
                         & (ok_runs["config_bits"] == f.final_config)]
            tel = {c: (rr[c].mean() if c in rr and rr[c].notna().any() else None) for c in tel_cols}
            layout = rr["layout"].iloc[0] if "layout" in rr and len(rr) else None
            ctx = float(rr["context"].iloc[0]) if len(rr) else math.nan
            b = binding_constraint(tel, env_cfg, fmask, acfg, layout, ctx, exp.workload.output_tokens,
                                   tel.get("max_num_seqs_effective"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"binding {f.env}/{f.point_id}: {e}")
            b = {"binding_constraint": INSUFFICIENT}
        final_rows.append({**f._asdict(), **b})
    shap_df = pd.DataFrame(shap_rows) if shap_rows else pd.DataFrame(columns=["status"])
    shap_df.to_csv(out / "shapley.csv", index=False)
    final_df = pd.DataFrame(final_rows).drop(columns=["Index"], errors="ignore") if final_rows \
        else pd.DataFrame(columns=["status"])
    final_df.to_csv(out / "final_configs.csv", index=False)

    nb = _safe("no_benefit", errors, no_benefit, main, ok_runs, acfg.no_benefit_gain)
    nb.to_csv(out / "no_benefit.csv", index=False)

    (out / "summary.md").write_text(_summary(profile_name, runs, cells, final_df, main, inter, fits, nb, errors,
                                             qrows, exp))
    log.info("[analyze] wrote %s", out)
    return out


def _telemetry_mean(rows: pd.DataFrame, col: str) -> float | None:
    if col not in rows or not rows[col].notna().any():
        return None
    return float(rows[col].mean())


def no_benefit(main: pd.DataFrame, ok_runs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Optimizations whose main-effect gain is under the threshold, with a first-pass telemetry reason."""
    rows = []
    for r in main.itertuples():
        if r.status == INSUFFICIENT or math.isnan(r.multiplier) or r.multiplier - 1.0 >= threshold:
            continue
        sel = ok_runs[(ok_runs["env"] == r.env) & (ok_runs["point_id"] == r.point_id)]
        on = sel[sel["config_bits"].str[r.optimization - 1] == "1"]
        dflt = sel[sel["config_bits"] == "0000000"]
        reason = _reason(r.optimization, on, dflt, sel)
        rows.append({"env": r.env, "point_id": r.point_id, "optimization": r.optimization, "name": r.name,
                     "multiplier": r.multiplier, "gain_pct": 100 * (r.multiplier - 1), "reason": reason})
    return pd.DataFrame(rows, columns=["env", "point_id", "optimization", "name", "multiplier", "gain_pct",
                                       "reason"])


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.0f}%" if pct else f"{x:.2f}"


def _reason(n: int, on: pd.DataFrame, dflt: pd.DataFrame, sel: pd.DataFrame) -> str:
    reuse = float(sel["reuse"].iloc[0]) if len(sel) else math.nan
    conc = float(sel["concurrency"].iloc[0]) if len(sel) else math.nan
    if n == 4:
        return f"prefix-cache hit rate {_fmt(_telemetry_mean(on, 'prefix_hit_rate'), True)} at {reuse:.0%} reuse"
    if n == 3:
        return (f"draft acceptance {_fmt(_telemetry_mean(on, 'spec_accept_rate'), True)} at concurrency "
                f"{conc:.0f}; extra verify work competes with batching")
    if n == 5:
        return (f"batch limit not binding: mean running {_fmt(_telemetry_mean(dflt, 'running_mean'))} vs default "
                f"limit {_fmt(_telemetry_mean(dflt, 'max_num_seqs_effective'))}, waiting "
                f"{_fmt(_telemetry_mean(dflt, 'waiting_mean'))}")
    if n == 6:
        return (f"prefill not stalling decode: default waiting {_fmt(_telemetry_mean(dflt, 'waiting_mean'))}, "
                f"p99 TTFT {_fmt(_telemetry_mean(dflt, 'ttft_p99_ms'))} ms")
    if n == 2:
        return (f"KV capacity not binding: default KV usage {_fmt(_telemetry_mean(dflt, 'kv_usage_max'), True)}, "
                f"preemptions {_fmt(_telemetry_mean(dflt, 'preemptions'))}")
    if n == 1:
        return (f"decode not weight-bandwidth bound at concurrency {conc:.0f} "
                f"(GPU util {_fmt(_telemetry_mean(dflt, 'gpu_util_mean'))}%)")
    if n == 7:
        return "layout gain offset by communication / replica overhead"
    return "no telemetry rule"


def _summary(profile: str, runs: pd.DataFrame, cells: pd.DataFrame, finals: pd.DataFrame, main: pd.DataFrame,
             inter: pd.DataFrame, fits: list[Fit], nb: pd.DataFrame, errors: list[str],
             qrows: list[dict[str, Any]], exp: ExperimentCfg) -> str:
    L = [f"# Report: profile `{profile}`", ""]
    st = runs["status"].value_counts().to_dict()
    L.append(f"Runs: {len(runs)} rows — " + ", ".join(f"{k}: {v}" for k, v in st.items()))
    if "used_placeholder" in runs and runs["used_placeholder"].fillna(False).astype(bool).any():
        L.append("**Some runs used placeholder (untuned) values for optimizations 5/6/7.**")
    if "dry_run" in runs and runs["dry_run"].fillna(False).astype(bool).any():
        L.append("**Synthetic data (FakeEngine) — not measurements.**")
    L.append(f"Gates: p99 TPOT ≤ {exp.gates.p99_tpot_ms} ms, p99 TTFT ≤ {exp.gates.p99_ttft_ms} ms, "
             f"accuracy drop ≤ {exp.gates.max_accuracy_drop} pt.")
    L.append(f"Quality runs: {len(qrows)} ({sum(r.get('status') == 'ok' for r in qrows)} ok)")
    L.append("")
    if not finals.empty and "final_config" in finals:
        L += ["## Final configuration per point", "",
              "| env | point | final | Value | binding constraint | Default fails gate |", "|---|---|---|---|---|---|"]
        for r in finals.itertuples():
            L.append(f"| {r.env} | {r.point_id} | {r.final_config} ({r.final_names}) | {r.value:.3f} | "
                     f"{getattr(r, 'binding_constraint', 'n/a')} | "
                     f"{r.default_failed_gates if r.default_fails_gate else 'no'} |")
        L.append("")
    if not main.empty:
        ok = main[main["status"] != INSUFFICIENT]
        if not ok.empty:
            L += ["## Main effects (throughput multiplier)", ""]
            piv = ok.pivot_table(index="name", columns=["env", "point_id"], values="multiplier")
            try:
                L.append(piv.round(3).to_markdown())
            except ImportError:   # tabulate not installed
                L += ["```", piv.round(3).to_string(), "```"]
            L.append("")
    fit_ok = sum(f.ok for f in fits)
    L.append(f"Model fits: {fit_ok}/{len(fits)} points fitted; the rest are marked '{INSUFFICIENT}'.")
    if not inter.empty and "label" in inter:
        L.append("Interaction ratios: " + ", ".join(f"{k}: {v}" for k, v in inter["label"].value_counts().items()))
    infeas = cells[cells["status"] != "ok"] if not cells.empty else cells
    if len(infeas):
        L += ["", f"## Infeasible / failed cells ({len(infeas)}; full list in value_per_config.csv)", ""]
        for r in list(infeas.itertuples())[:25]:
            L.append(f"- {r.env} {r.point_id} {r.config_bits} ({r.config_names}): {r.status} — "
                     f"{(r.error or '')[:160]}")
    if len(nb):
        L += ["", f"## No-benefit list ({len(nb)} entries; gain < {100 * exp.analysis.no_benefit_gain:.0f}%)",
              "See no_benefit.csv."]
    if errors:
        L += ["", "## Analysis steps that could not run", ""] + [f"- {e.splitlines()[0]}" for e in errors]
    return "\n".join(L) + "\n"
