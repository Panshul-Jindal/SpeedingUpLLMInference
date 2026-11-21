"""Investigation profile: job list, deep-telemetry summaries, profiler-trace parsing, paired quality, dry run."""
import gzip
import json
import math

import pandas as pd

from harness.analysis.investigate import analyze_investigation, mcnemar_p, paired_quality
from harness.config import load_settings
from harness.deep_telemetry import parse_prom_raw, summarize_deep, summarize_trace
from harness.orchestrator import cmd_investigate, estimate_investigation, investigate_jobs, make_engine

from test_dry_run import _small


def _scrape(steps, toks, read_bytes, drafts, pos):
    return f"""# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_count{{model_name="t",engine="0"}} {steps}
vllm:iteration_tokens_total_sum{{model_name="t",engine="0"}} {toks}
vllm:iteration_tokens_total_bucket{{model_name="t",engine="0",le="+Inf"}} {steps}
# TYPE vllm:estimated_read_bytes_per_gpu counter
vllm:estimated_read_bytes_per_gpu_total{{model_name="t",engine="0"}} {read_bytes}
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total{{model_name="t",engine="0"}} {drafts}
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name="t",engine="0",position="0"}} {pos[0]}
vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name="t",engine="0",position="1"}} {pos[1]}
# TYPE vllm:unrelated_gauge gauge
vllm:unrelated_gauge 5
"""


def test_deep_summary_from_raw_series():
    prom = pd.DataFrame(parse_prom_raw(_scrape(100, 3200, 0, 50, (0, 0)), 0.0, 0)
                        + parse_prom_raw(_scrape(1100, 35200, 2e12, 550, (450, 300)), 10.0, 0))
    assert "vllm:unrelated_gauge" not in set(prom.name)
    nv = pd.DataFrame([{"t": 0, "gpu": 0, "sm_util": 90, "mem_util": 40, "sm_clock_mhz": 1700, "mem_clock_mhz": 2600,
                        "power_w": 300, "temp_c": 60, "throttle": 4},
                       {"t": 1, "gpu": 0, "sm_util": 100, "mem_util": 60, "sm_clock_mhz": 1500, "mem_clock_mhz": 2600,
                        "power_w": 350, "temp_c": 65, "throttle": 1}])
    procs = pd.DataFrame([{"t": t, "server": 0, "pid": 1, "role": "engine_core", "cpu_pct": 99.0, "threads": 9,
                           "rss_mb": 1.0} for t in (0, 1, 2)]
                         + [{"t": t, "server": -1, "pid": -1, "role": "host", "cpu_pct": 30.0, "threads": 0,
                             "rss_mb": 1.0} for t in (0, 1, 2)])
    s = summarize_deep(nv, prom, procs)
    assert s["deep_steps_per_s"] == 100.0 and s["deep_step_ms"] == 10.0 and s["deep_tokens_per_step"] == 32.0
    assert s["deep_read_gbps"] == 200.0
    assert s["deep_spec_accept_by_pos"] == {"0": 0.9, "1": 0.6}
    assert s["deep_cpu_engine_core_pct"] == 99.0 and s["deep_cpu_host_pct"] == 30.0
    assert s["deep_throttle_reasons"] == 5 and s["deep_sm_clock_min_mhz"] == 1500


def test_process_cpu_is_measured_not_zero():
    """Regression: fresh psutil.Process objects per sample always read 0.0% CPU."""
    import subprocess
    import sys
    import time

    from harness.deep_telemetry import DeepCollector

    busy = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        dc = DeepCollector([], [], [busy.pid], proc_interval_s=0.2)
        dc.start()
        time.sleep(1.6)
        dc.stop()
    finally:
        busy.kill()
        busy.wait()
    procs = pd.DataFrame(dc.proc_rows)
    mine = procs[procs.pid == busy.pid]
    assert mine.cpu_pct.notna().sum() >= 3 and mine.cpu_pct.max() > 50.0
    assert mine.cpu_pct.isna().sum() == 0            # primed in start(), so every sample is a real value


def test_disabled_flop_counters_are_missing_not_zero():
    off = pd.DataFrame(parse_prom_raw(_scrape(100, 3200, 0, 50, (0, 0)), 0.0, 0)
                       + parse_prom_raw(_scrape(1100, 35200, 0, 550, (450, 300)), 10.0, 0))
    s = summarize_deep(pd.DataFrame(), off, pd.DataFrame())
    assert s["deep_read_gbps"] is None and s["deep_tflops"] is None and s["deep_steps_per_s"] == 100.0


def test_mfu_flag_reaches_the_server_command():
    from harness.engine.vllm_flags import build_launch
    s = load_settings("investigate", "h100")
    assert "--enable-mfu-metrics" in build_launch(0, s.env, s.experiment, point_id="center").flags
    base = load_settings("fast", "h100")
    assert "--enable-mfu-metrics" not in build_launch(0, base.env, base.experiment, point_id="center").flags


def test_profiler_trace_gpu_busy_fraction(tmp_path):
    ev = [{"ph": "X", "cat": "kernel", "name": "gemm", "ts": 0, "dur": 30},
          {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 20, "dur": 20},      # overlaps the first
          {"ph": "X", "cat": "kernel", "name": "attn", "ts": 70, "dur": 30},
          {"ph": "X", "cat": "cpu_op", "name": "execute_model", "ts": 0, "dur": 100}]
    f = tmp_path / "rank0.pt.trace.json.gz"
    with gzip.open(f, "wt") as fh:
        json.dump({"traceEvents": ev}, fh)
    s = summarize_trace(f)
    assert s["span_ms"] == 0.1 and math.isclose(s["gpu_busy_frac"], 0.7)     # 40 + 30 busy of 100 us
    assert s["top_kernels"][0]["name"] == "gemm" and s["n_steps"] == 1


def test_paired_quality_mcnemar(tmp_path):
    q = load_settings("investigate", "h100").experiment.quality
    q = q.model_copy(update={"tasks": ["gsm8k"]})

    def write(label, scores):
        d = tmp_path / label / "lm_eval_gsm8k" / "target"
        d.mkdir(parents=True)
        with open(d / "samples_gsm8k_2026.jsonl", "w") as fh:
            for i, v in enumerate(scores):
                for filt in ("strict-match", "flexible-extract"):
                    fh.write(json.dumps({"doc_id": i, "filter": filt,
                                         "exact_match": v if filt == "strict-match" else 1.0}) + "\n")
    write("000", [1] * 700 + [0] * 300)
    write("100", [1] * 640 + [0] * 360)                  # loses 60 questions, gains none: -6 pts
    write("010", [1] * 695 + [0] * 300 + [1] * 5)         # 5 lost, 5 gained: no change
    t = paired_quality(tmp_path, q, ["000", "100", "010"], gate=1.0).set_index("setting")
    assert t.loc["100", "lost"] == 60 and t.loc["100", "diff_pts"] == -6.0
    assert t.loc["100", "verdict"] == "significant drop beyond gate"
    assert t.loc["010", "diff_pts"] == 0.0 and t.loc["010", "mcnemar_p"] == 1.0
    assert mcnemar_p(0, 0) == 1.0


def test_samples_with_unicode_line_separators_are_read(tmp_path):
    """lm-eval writes model text unescaped; \x0c / \u2028 inside a record must not split it."""
    from harness.analysis.investigate import per_question
    d = tmp_path / "000" / "lm_eval_gsm8k" / "target"
    d.mkdir(parents=True)
    recs = [{"doc_id": 0, "filter": "strict-match", "exact_match": 1.0, "resps": [["a\x0cb\u2028c\x85d"]]},
            {"doc_id": 1, "filter": "strict-match", "exact_match": 0.0, "resps": [["plain"]]}]
    (d / "samples_gsm8k_2026.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                                                encoding="utf-8")
    assert per_question(tmp_path, "000", "gsm8k", "exact_match,strict-match") == {"gsm8k:0": 1.0, "gsm8k:1": 0.0}


def test_investigation_jobs_and_estimate():
    s = load_settings("investigate", "h100")
    jobs = investigate_jobs(s)
    keys = [j.key for j in jobs]
    assert len(keys) == len(set(keys))                                   # shared cells run once
    assert sum(j.tag == "investigate" for j in jobs) == 126 and sum(j.tag == "profile" for j in jobs) == 18
    derived = [j for j in jobs if "@" in j.point.point_id]
    assert derived and {j.tuned_point for j in derived} == {"center"}
    assert {j.point.point_id for j in jobs if j.point.reuse == 0.0 and j.point.context == 2048} == {"reuse_low"}
    est = estimate_investigation(s)
    assert 3.0 < est["total_h"] < 8.0                                     # fits one 8 h job


def test_investigate_dry_run_end_to_end(tmp_path):
    s = _small(load_settings("investigate", "h100", dry_run=True, results_root=tmp_path))
    rows = cmd_investigate(s, make_engine(s))
    assert len(rows) == len(investigate_jobs(s)) and {r["status"] for r in rows} == {"ok"}
    assert sum(r["kind"] == "profile" for r in rows) >= 18
    alone = [r for r in rows if r.get("exclusive")]
    assert alone and {tuple(r["gpu_ids"]) for r in alone} == {(0,)}     # other slot idle
    assert {tuple(r["gpu_ids"]) for r in rows if not r.get("exclusive")} == {(0,), (1,)}
    assert cmd_investigate(s, make_engine(s)) == []                      # resume
    out = analyze_investigation(s.results_dir, "investigate")
    md = (out / "investigation.md").read_text()
    for inv in s.profile.investigations:
        assert f"## {inv.id}" in md
    assert "Analysis errors" not in md
    assert (out / "interactions_measured.csv").stat().st_size > 10
