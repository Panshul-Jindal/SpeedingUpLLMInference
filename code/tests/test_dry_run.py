"""Dry run end to end (FakeEngine): run + quality + sweep + analyze, crash survival, 4-GPU layouts, resume."""
import pandas as pd

from harness.analysis.report import REPORT_FILES, analyze
from harness.config import load_settings
from harness.orchestrator import cmd_run, cmd_startup_check, cmd_sweep, make_engine, plan_masks
from harness.quality import run_quality
from harness.results import ResultStore


def test_dry_end_to_end(tmp_path):
    rows_all = []
    for env in ("test", "main"):
        s = load_settings("dry", env, results_root=tmp_path)
        eng = make_engine(s)
        rows_all += cmd_run(s, eng)
        run_quality(s, plan_masks(s), eng)
        if env == "main":
            layouts = {l.layout for l in eng.launch_log}
            assert "tp4" in layouts and "tp2x2" in layouts      # 4-GPU paths exercised
            assert any(len(l.servers) == 2 for l in eng.launch_log)
    statuses = {r["status"] for r in rows_all}
    assert {"ok", "infeasible", "failed"} <= statuses           # planted infeasible + injected crash
    crash = [r for r in rows_all if r["status"] == "failed"]
    assert len(crash) == 1 and "injected crash" in crash[0]["error"]
    assert len([r for r in rows_all if r["env"] == "main"]) == 512 * 3
    assert len([r for r in rows_all if r["env"] == "test"]) == 256 * 3

    rdir = s.results_dir
    out = analyze(rdir, "dry")
    for f in REPORT_FILES:
        assert (out / f).exists(), f
    vals = pd.read_csv(out / "value_per_config.csv")
    assert vals["failed_gates"].fillna("").str.contains("p99_ttft").any()       # planted gate failure
    assert vals["failed_gates"].fillna("").str.contains("accuracy").any()       # planted quality failure
    final = pd.read_csv(out / "final_configs.csv")
    assert len(final) == 14

    # resume: re-running does nothing; --force reruns
    s = load_settings("dry", "main", results_root=tmp_path)
    assert cmd_run(s, make_engine(s)) == []


def test_dry_sweep_and_startup(tmp_path):
    s = load_settings("dry", "main", results_root=tmp_path)
    eng = make_engine(s)
    chosen = cmd_sweep(s, eng)
    assert any(k.endswith("max_num_seqs") for k in chosen)
    assert s.tuned_path.parent == s.results_dir                # fake sweeps never touch configs/tuned.yaml
    rows = cmd_startup_check(s, eng, all_configs=True)
    assert len(rows) == 128


def _small(s):
    """Shrink a real profile's workload so a FakeEngine dry run is quick."""
    exp = s.experiment.model_copy(deep=True)
    exp.workload.trace_requests, exp.workload.output_tokens = 64, 16
    exp.regimes.context.low, exp.regimes.context.mid, exp.regimes.context.high = 32, 64, 128
    exp.run.warmup_s = exp.run.window_s = 0
    return s.model_copy(update={"experiment": exp})


def test_fast_profile_runs_cells_concurrently_one_gpu_each(tmp_path):
    s = _small(load_settings("fast", "main", dry_run=True, results_root=tmp_path))
    rows = cmd_run(s, make_engine(s))
    assert len(rows) == 128 * 3
    assert {r["status"] for r in rows} == {"ok"}
    assert {r["layout"] for r in rows} == {"tp1"}
    assert {tuple(r["gpu_ids"]) for r in rows} == {(0,), (1,), (2,), (3,)}    # all 4 slots used
    assert {r["config_bits"][5:] for r in rows} == {"00"}                     # chunk and par stay off
    assert len(ResultStore(s.results_dir / "runs.jsonl").latest()) == 128 * 3  # concurrent appends intact
    assert cmd_run(s, make_engine(s)) == []                                    # resume

    q = run_quality(s, plan_masks(s), make_engine(s))
    assert sorted(r["numeric_bits"] for r in q) == sorted(f"{i:03b}" for i in range(8))

    out = analyze(s.results_dir, "fast")
    main = pd.read_csv(out / "main_effects.csv")
    assert set(main["name"].dropna()) == {"quant", "kvq", "spec", "prefix", "batch"}
