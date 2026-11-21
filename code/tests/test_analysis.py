import json
import math

import numpy as np
import pandas as pd
import pytest

from harness import optimizations as opt
from harness.analysis.binding import binding_constraint
from harness.analysis.effects import fit_point, interactions_table
from harness.analysis.report import REPORT_FILES, analyze
from harness.analysis.shapley import shapley, shapley_for_config
from harness.config import AnalysisCfg, load_env
from harness.design import full_factorial, half_fraction, smoke_configs

ACT6 = [1, 2, 3, 4, 5, 6]
MAIN = {1: 0.10, 2: 0.03, 3: -0.05, 4: 0.20, 5: 0.0, 6: 0.04}       # ±1-coded betas
INTER = {(1, 2): -0.04, (4, 6): 0.03}                                 # two planted interactions


def _logt(mask):
    x = {n: (1.0 if opt.is_on(mask, n) else -1.0) for n in ACT6}
    v = 7.0 + sum(b * x[n] for n, b in MAIN.items())
    v += sum(b * x[a] * x[c] for (a, c), b in INTER.items())
    return v


def _frame(masks, reps=3, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in masks:
        for r in range(reps):
            rows.append({"config_bits": opt.to_bits(m), "rep": r + 1,
                         "output_tok_s": math.exp(_logt(m) + rng.normal(0, sd))})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("design", ["full", "half"])
def test_fit_recovers_planted_effects(design):
    masks = full_factorial(ACT6) if design == "full" else half_fraction(ACT6)
    f = fit_point(_frame(masks), "test", "center", ACT6)
    assert f.ok
    for n, b in MAIN.items():
        assert f.params[(n,)] == pytest.approx(b, abs=0.006)
    for pair, b in INTER.items():
        assert f.params[pair] == pytest.approx(b, abs=0.006)
    others = [t for t in f.terms if len(t) == 2 and t not in INTER]
    assert all(abs(f.params[t]) < 0.006 for t in others)
    assert f.r2 > 0.99
    assert f.noise_sd == pytest.approx(0.01, rel=0.3)


def test_interaction_ratio_measured_and_model():
    masks = half_fraction(ACT6)
    f = fit_point(_frame(masks), "test", "p", ACT6)
    # half fraction: singles are not run, so ratios come from the model
    cell = {("test", "p", m): math.exp(_logt(m)) for m in masks}
    tab = interactions_table([f], cell).set_index("pair")
    assert tab.loc["quant:kvq", "label"] == "model"
    assert math.log(tab.loc["quant:kvq", "ratio"]) == pytest.approx(4 * INTER[(1, 2)], abs=0.03)
    # full factorial: measured
    full = {("test", "p", m): math.exp(_logt(m)) for m in full_factorial(ACT6)}
    tab2 = interactions_table([f], full).set_index("pair")
    assert tab2.loc["prefix:chunk", "label"] == "measured"
    assert math.log(tab2.loc["prefix:chunk", "ratio"]) == pytest.approx(4 * INTER[(4, 6)], abs=1e-9)


def test_shapley_sums_to_total_gain():
    F = opt.from_bits("1101010")
    phi, exact, status = shapley_for_config(F, lambda m: _logt(m), None)
    assert exact and status == "ok"
    assert sum(phi.values()) == pytest.approx(_logt(F) - _logt(0), abs=1e-12)
    # symmetry/efficiency on a tiny additive game
    phi2 = shapley([1, 2], lambda s: {0: 0, 1: 1.0, 2: 2.0, 3: 3.0}[s])
    assert phi2 == {1: pytest.approx(1.0), 2: pytest.approx(2.0)}


def test_shapley_approximate_fill():
    F = opt.from_bits("1100000")
    measured = lambda m: _logt(m) if m != opt.bit(2) else None   # one subset missing
    phi, exact, status = shapley_for_config(F, measured, _logt)
    assert not exact and status == "ok"
    assert sum(phi.values()) == pytest.approx(_logt(F) - _logt(0))
    phi, exact, status = shapley_for_config(F, measured, None)
    assert status == "insufficient data"


def test_binding_rules():
    env = load_env("test")
    cfg = AnalysisCfg()
    b = binding_constraint({"kv_usage_max": 0.95, "output_tok_s": 100, "running_mean": 8}, env, 0, cfg,
                           None, 1024, 64, 256)
    assert b["binding_constraint"] == "kv_capacity"
    # peaks are null in the env config -> bandwidth/compute reported unavailable, not a crash
    b = binding_constraint({"kv_usage_max": 0.3, "output_tok_s": 100, "running_mean": 8, "running_max": 8,
                            "waiting_mean": 3}, env, opt.bit(5), cfg, None, 1024, 64, 8)
    assert b["bw_frac"] == "unavailable" and b["binding_constraint"] == "scheduler_batch_limit"
    env2 = env.model_copy(update={"peaks": env.peaks.model_copy(update={"mem_bandwidth_gbps": 100.0})})
    b = binding_constraint({"kv_usage_max": 0.3, "output_tok_s": 300, "running_mean": 8}, env2, 0, cfg,
                           None, 1024, 64, 256)
    assert b["binding_constraint"] == "memory_bandwidth"


def _write_runs(path, masks, profile, env="test", points=("center",), reps=1, extra=None):
    with open(path, "w") as f:
        for pid in points:
            for m in masks:
                for r in range(1, reps + 1):
                    row = {"run_key": f"{profile}__{env}__{pid}__{opt.to_bits(m)}__r{r}", "kind": "run",
                           "profile": profile, "env": env, "config_bits": opt.to_bits(m),
                           "config_names": opt.names(m), "point_id": pid, "rep": r, "status": "ok",
                           "output_tok_s": math.exp(_logt(m)), "tpot_p99_ms": 40.0, "ttft_p99_ms": 800.0,
                           "context": 1024, "concurrency": 8, "reuse": 0.5, "running_mean": 6.0,
                           "running_max": 8, "waiting_mean": 0.0, "kv_usage_max": 0.4, "preemptions": 0.0,
                           "max_num_seqs_effective": 256, "layout": "tp1"}
                    row.update((extra or {}).get(m, {}))
                    f.write(json.dumps(row) + "\n")


def test_analyze_partial_smoke_data(tmp_path):
    """8 smoke configurations, 1 rep, one infeasible: fits what it can, never crashes."""
    masks = smoke_configs(ACT6)
    _write_runs(tmp_path / "runs.jsonl", masks, "smoke", points=("center", "context_high"),
                extra={masks[-1]: {"status": "infeasible", "output_tok_s": None, "error": "OOM"}})
    out = analyze(tmp_path, "smoke")
    for f in REPORT_FILES:
        assert (out / f).exists(), f
    main = pd.read_csv(out / "main_effects.csv")
    assert len(main) == 12
    final = pd.read_csv(out / "final_configs.csv")
    assert set(final["point_id"]) == {"center", "context_high"}
    assert "binding_constraint" in final


def test_analyze_empty_and_default_missing(tmp_path):
    out = analyze(tmp_path, "smoke")                 # no data at all
    assert (out / "summary.md").exists()
    _write_runs(tmp_path / "runs.jsonl", [opt.bit(1), opt.bit(4)], "smoke")   # no Default
    out = analyze(tmp_path, "smoke")
    for f in REPORT_FILES:
        assert (out / f).exists()


def test_bandwidth_estimate_accounts_for_speculative_tokens():
    env = load_env("test")
    base = {"output_tok_s": 400.0, "running_mean": 8.0}
    a = binding_constraint(base, env, 0, AnalysisCfg(), None, 1024, 64, 256)
    b = binding_constraint({**base, "tokens_per_step": 2.0}, env, 0, AnalysisCfg(), None, 1024, 64, 256)
    assert b["bw_gbps"] == pytest.approx(a["bw_gbps"] / 2)
