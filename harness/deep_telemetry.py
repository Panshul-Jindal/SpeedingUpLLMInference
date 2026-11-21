"""Deep telemetry for investigation runs (profile `investigate`): time series instead of window summaries.

Collected during the measurement window only, in background threads:
- NVML every `nvml_interval_s`: SM/memory-controller utilisation, SM/memory clocks, power, temperature,
  clock-throttle reasons;
- every vLLM server's /metrics every `prom_interval_s`, raw (all label sets and histogram buckets of the
  wanted families);
- CPU% / threads / RSS of each server's process tree (API server, EngineCore, workers) and of the host.

Series are written as Parquet next to the run; `summary()` returns the scalar columns added to the row.
Also: parsing of vLLM torch-profiler traces (profiler passes) into GPU-busy fraction and top kernels/ops.
"""
from __future__ import annotations

import gzip
import json
import logging
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
from prometheus_client.parser import text_string_to_metric_families

from harness.telemetry import Nvml

log = logging.getLogger(__name__)

# Metric families kept in the raw series (prefix match on the family name).
PROM_FAMILIES = (
    "vllm:iteration_tokens_total", "vllm:estimated_read_bytes_per_gpu", "vllm:estimated_write_bytes_per_gpu",
    "vllm:estimated_flops_per_gpu", "vllm:request_prefill_time_seconds", "vllm:request_decode_time_seconds",
    "vllm:inter_token_latency_seconds", "vllm:time_to_first_token_seconds", "vllm:request_queue_time_seconds",
    "vllm:spec_decode_", "vllm:prefix_cache_", "vllm:prompt_tokens", "vllm:generation_tokens",
    "vllm:num_requests_", "vllm:kv_cache_usage_perc", "vllm:num_preemptions",
)


def _role(p: psutil.Process) -> str:
    try:
        text = " ".join(p.cmdline()) + " " + p.name()
    except psutil.Error:
        return "gone"
    if "EngineCore" in text:
        return "engine_core"
    if "vllm serve" in text or "vllm.entrypoints" in text or text.split(" ")[0].endswith("vllm"):
        return "api_server"
    if "Worker" in text:
        return "worker"
    return "other"


def parse_prom_raw(text: str, t: float, server: int) -> list[dict[str, Any]]:
    rows = []
    for fam in text_string_to_metric_families(text):
        if not any(fam.name.startswith(f) or fam.name + "_total" == f for f in PROM_FAMILIES):
            continue
        for s in fam.samples:
            if s.name.endswith("_created"):
                continue
            labels = {k: v for k, v in s.labels.items() if k not in ("model_name", "engine")}
            rows.append({"t": t, "server": server, "name": s.name,
                         "labels": json.dumps(labels, sort_keys=True), "value": float(s.value)})
    return rows


class DeepCollector:
    def __init__(self, gpu_ids: list[int], metrics_urls: list[str], root_pids: list[int],
                 nvml_interval_s: float = 0.25, prom_interval_s: float = 1.0, proc_interval_s: float = 1.0):
        self.gpu_ids, self.urls, self.root_pids = gpu_ids, metrics_urls, root_pids
        self.nvml_interval_s, self.prom_interval_s, self.proc_interval_s = \
            nvml_interval_s, prom_interval_s, proc_interval_s
        self.nvml = Nvml()
        self.nv_rows: list[dict[str, Any]] = []
        self.prom_rows: list[dict[str, Any]] = []
        self.proc_rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # psutil's cpu_percent() measures since the previous call ON THE SAME Process object; a fresh object
        # always reports 0.0. Keep one object per pid for the whole window.
        self._proc_cache: dict[int, psutil.Process] = {}
        self.t0 = self.t1 = 0.0

    # -------------------------------------------------------------- samplers
    def _loop(self, fn: Any, interval: float) -> None:
        while not self._stop.is_set():
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - telemetry never fails a run
                log.debug("deep telemetry sample failed: %s", e)
            self._stop.wait(interval)
        try:
            fn()   # final sample at the window end (counters need both ends)
        except Exception:  # noqa: BLE001
            pass

    def _sample_nvml(self) -> None:
        if not self.nvml.ok:
            return
        nv = self.nvml.nv
        t = time.monotonic()
        for g in self.gpu_ids:
            h = nv.nvmlDeviceGetHandleByIndex(g)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            self.nv_rows.append({
                "t": t, "gpu": g, "sm_util": float(util.gpu), "mem_util": float(util.memory),
                "sm_clock_mhz": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                "mem_clock_mhz": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "temp_c": float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                "throttle": int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
            })

    def _sample_prom(self) -> None:
        t = time.monotonic()
        for i, url in enumerate(self.urls):
            with urllib.request.urlopen(url, timeout=10) as r:
                self.prom_rows += parse_prom_raw(r.read().decode(), t, i)

    def _procs(self) -> list[tuple[int, psutil.Process, bool]]:
        """(server index, process, first_seen) for each root and its descendants, reusing Process objects."""
        out = []
        for i, pid in enumerate(self.root_pids):
            try:
                root = self._proc_cache.get(pid) or psutil.Process(pid)
                for p in [root, *root.children(recursive=True)]:
                    first = p.pid not in self._proc_cache
                    out.append((i, self._proc_cache.setdefault(p.pid, p), first))
            except psutil.Error:
                continue
        return out

    def _sample_procs(self) -> None:
        t = time.monotonic()
        for i, p, first in self._procs():
            try:
                with p.oneshot():
                    # a process seen for the first time has no baseline yet: prime it, record no CPU value
                    cpu = float("nan") if first and p.cpu_percent(None) == 0.0 else p.cpu_percent(None)
                    self.proc_rows.append({"t": t, "server": i, "pid": p.pid, "role": _role(p), "cpu_pct": cpu,
                                           "threads": p.num_threads(), "rss_mb": p.memory_info().rss / 2**20})
            except psutil.Error:
                continue
        self.proc_rows.append({"t": t, "server": -1, "pid": -1, "role": "host",
                               "cpu_pct": psutil.cpu_percent(None), "threads": 0,
                               "rss_mb": psutil.virtual_memory().used / 2**20})

    def start(self) -> None:
        for _, p, _ in self._procs():   # prime cpu_percent so the first sample is meaningful
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        psutil.cpu_percent(None)
        self.t0 = time.monotonic()
        for fn, iv in ((self._sample_nvml, self.nvml_interval_s), (self._sample_prom, self.prom_interval_s),
                       (self._sample_procs, self.proc_interval_s)):
            th = threading.Thread(target=self._loop, args=(fn, iv), daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=30)
        self.t1 = time.monotonic()

    # -------------------------------------------------------------- outputs
    def write(self, run_dir: Path) -> None:
        for name, rows in (("nvml", self.nv_rows), ("prom", self.prom_rows), ("procs", self.proc_rows)):
            if rows:
                pd.DataFrame(rows).assign(t=lambda d: d.t - self.t0).to_parquet(
                    run_dir / f"telemetry_{name}.parquet", index=False)

    def summary(self) -> dict[str, Any]:
        return summarize_deep(pd.DataFrame(self.nv_rows), pd.DataFrame(self.prom_rows),
                              pd.DataFrame(self.proc_rows))


def _delta(prom: pd.DataFrame, name: str, labels: str | None = None) -> tuple[float | None, float]:
    """Window delta of a counter (summed over servers / label sets) and the time it spans."""
    d = prom[prom.name == name]
    if labels is not None:
        d = d[d.labels == labels]
    if d.empty or d.t.nunique() < 2:
        return None, 0.0
    by_t = d.groupby("t").value.sum().sort_index()
    return float(by_t.iloc[-1] - by_t.iloc[0]), float(by_t.index[-1] - by_t.index[0])


def summarize_deep(nv: pd.DataFrame, prom: pd.DataFrame, procs: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not nv.empty:
        out.update(deep_sm_util_mean=nv.sm_util.mean(), deep_mem_util_mean=nv.mem_util.mean(),
                   deep_sm_clock_mean_mhz=nv.sm_clock_mhz.mean(), deep_sm_clock_min_mhz=nv.sm_clock_mhz.min(),
                   deep_power_mean_w=nv.power_w.mean(), deep_temp_max_c=nv.temp_c.max(),
                   deep_throttle_reasons=_bit_or(nv.throttle))
    if not prom.empty:
        steps, dt = _delta(prom, "vllm:iteration_tokens_total_count")
        toks, _ = _delta(prom, "vllm:iteration_tokens_total_sum")
        if steps and dt > 0:
            out["deep_steps_per_s"] = steps / dt
            out["deep_step_ms"] = 1000.0 * dt / steps
            out["deep_tokens_per_step"] = toks / steps if toks is not None else None
        for key, name in (("deep_read_gbps", "vllm:estimated_read_bytes_per_gpu_total"),
                          ("deep_write_gbps", "vllm:estimated_write_bytes_per_gpu_total")):
            v, dt2 = _delta(prom, name)
            out[key] = v / dt2 / 1e9 if v and dt2 > 0 else None     # 0.0 = counter not enabled
        v, dt2 = _delta(prom, "vllm:estimated_flops_per_gpu_total")
        out["deep_tflops"] = v / dt2 / 1e12 if v and dt2 > 0 else None
        for key, fam in (("deep_prefill_s", "vllm:request_prefill_time_seconds"),
                         ("deep_decode_s", "vllm:request_decode_time_seconds"),
                         ("deep_itl_s", "vllm:inter_token_latency_seconds"),
                         ("deep_queue_s", "vllm:request_queue_time_seconds")):
            s, _ = _delta(prom, f"{fam}_sum")
            c, _ = _delta(prom, f"{fam}_count")
            out[key] = s / c if s is not None and c else None
        cached, _ = _delta(prom, "vllm:prompt_tokens_cached_total")
        prompt, _ = _delta(prom, "vllm:prompt_tokens_total")
        out["deep_cached_prompt_frac"] = cached / prompt if cached is not None and prompt else None
        drafts, _ = _delta(prom, "vllm:spec_decode_num_drafts_total")
        pos = prom[prom.name == "vllm:spec_decode_num_accepted_tokens_per_pos_total"]
        if drafts and not pos.empty:
            rates = {}
            for lab in sorted(pos.labels.unique()):
                acc, _ = _delta(prom, "vllm:spec_decode_num_accepted_tokens_per_pos_total", lab)
                rates[str(json.loads(lab).get("position", lab))] = round(acc / drafts, 4) if acc is not None else None
            out["deep_spec_accept_by_pos"] = rates
    if not procs.empty:
        p = procs[procs.cpu_pct.notna() | (procs.role == "host")]    # drop unprimed first sightings
        per_t = p[p.role != "host"].groupby(["t", "role"]).cpu_pct.sum().unstack(fill_value=0.0)
        for role in ("engine_core", "api_server", "worker"):
            if role in per_t:
                out[f"deep_cpu_{role}_pct"] = per_t[role].mean()      # % of one core
                out[f"deep_cpu_{role}_max_pct"] = per_t[role].max()
        host = p[p.role == "host"]
        if not host.empty:
            out["deep_cpu_host_pct"] = host.cpu_pct.mean()           # % of all cores
    return out


def _bit_or(s: pd.Series) -> int:
    v = 0
    for x in s:
        v |= int(x)
    return v


# ---------------------------------------------------------------- torch profiler traces

def find_traces(trace_dir: Path) -> list[Path]:
    return sorted(p for p in trace_dir.rglob("*.json*") if p.is_file())


def wait_for_traces(trace_dir: Path, timeout_s: float, settle_s: float = 10.0) -> list[Path]:
    """vLLM writes the traces after /stop_profile; wait until some exist and their sizes stop changing."""
    deadline = time.monotonic() + timeout_s
    last: dict[Path, int] = {}
    stable_since = None
    while time.monotonic() < deadline:
        files = find_traces(trace_dir)
        sizes = {f: f.stat().st_size for f in files}
        if files and sizes == last:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= settle_s:
                return files
        else:
            stable_since = None
        last = sizes
        time.sleep(2.0)
    return find_traces(trace_dir)


def _load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def _union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def summarize_trace(path: Path, top: int = 12) -> dict[str, Any]:
    """GPU-busy fraction over the profiled span, plus the top kernels and CPU ops by total time."""
    events = [e for e in _load_trace(path).get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    out: dict[str, Any] = {"trace": path.name, "n_kernels": len(kernels)}
    if not kernels:
        return out
    k0 = min(e["ts"] for e in kernels)
    k1 = max(e["ts"] + e["dur"] for e in kernels)
    busy = _union_us([(e["ts"], e["ts"] + e["dur"]) for e in kernels])
    out.update(span_ms=(k1 - k0) / 1e3, gpu_busy_ms=busy / 1e3, gpu_busy_frac=busy / (k1 - k0) if k1 > k0 else None)

    def top_by(evts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        agg: dict[str, list[float]] = {}
        for e in evts:
            agg.setdefault(e.get("name", "?"), []).append(e["dur"])
        rows = sorted(((n, sum(d), len(d)) for n, d in agg.items()), key=lambda r: -r[1])[:top]
        return [{"name": n[:120], "total_ms": round(t / 1e3, 3), "calls": c} for n, t, c in rows]

    out["top_kernels"] = top_by(kernels)
    out["top_cpu_ops"] = top_by(cpu)
    steps = [e for e in cpu if "execute_model" in e.get("name", "") or e.get("name", "").startswith("step")]
    if steps:
        out["n_steps"] = len(steps)
        out["cpu_step_ms_mean"] = sum(e["dur"] for e in steps) / len(steps) / 1e3
    return out


def summarize_profile_dir(trace_dir: Path) -> dict[str, Any]:
    """Summary of the worker trace with the most kernels (the frontend trace has none)."""
    best: dict[str, Any] = {}
    for f in find_traces(trace_dir):
        try:
            s = summarize_trace(f)
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse profiler trace %s: %s", f, e)
            continue
        if s.get("n_kernels", 0) > best.get("n_kernels", -1):
            best = s
    return best
