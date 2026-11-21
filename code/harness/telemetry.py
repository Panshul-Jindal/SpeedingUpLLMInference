"""Telemetry during the measurement window: NVML GPU sampling + vLLM Prometheus scraping."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

log = logging.getLogger(__name__)

# Wanted vLLM metrics -> candidate sample names, newest first. Names change between vLLM versions.
# kind: gauge (mean/max over samples, summed across replicas), counter (window delta),
# histogram (window mean via _sum/_count deltas).
METRIC_TABLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "kv_usage": ("gauge", ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")),
    "running": ("gauge", ("vllm:num_requests_running",)),
    "waiting": ("gauge", ("vllm:num_requests_waiting",)),
    "preemptions": ("counter", ("vllm:num_preemptions_total", "vllm:num_preemptions")),
    "prefix_hits": ("counter", ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total",
                                "vllm:prefix_cache_hits")),
    "prefix_queries": ("counter", ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total",
                                   "vllm:prefix_cache_queries")),
    "spec_draft_tokens": ("counter", ("vllm:spec_decode_num_draft_tokens_total",
                                      "vllm:spec_decode_num_draft_tokens")),
    "spec_accepted_tokens": ("counter", ("vllm:spec_decode_num_accepted_tokens_total",
                                         "vllm:spec_decode_num_accepted_tokens")),
    "spec_drafts": ("counter", ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts")),
    "queue_time": ("histogram", ("vllm:request_queue_time_seconds",)),
    "prompt_tokens": ("counter", ("vllm:prompt_tokens_total", "vllm:prompt_tokens")),
    "generation_tokens": ("counter", ("vllm:generation_tokens_total", "vllm:generation_tokens")),
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every sample over its label sets: {sample_name: value}. Histogram buckets are dropped."""
    out: dict[str, float] = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name.endswith("_bucket") or s.name.endswith("_created"):
                continue
            out[s.name] = out.get(s.name, 0.0) + float(s.value)
    return out


def resolve_names(available: set[str]) -> tuple[dict[str, str], list[str]]:
    """Map each wanted metric to the first available candidate. Returns (mapping, missing)."""
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for key, (kind, cands) in METRIC_TABLE.items():
        found = None
        for c in cands:
            probe = f"{c}_sum" if kind == "histogram" else c
            if probe in available:
                found = c
                break
        if found:
            mapping[key] = found
        else:
            missing.append(key)
    return mapping, missing


# Gauges that are fractions per server: averaged (not summed) across replicas.
PER_SERVER_FRACTIONS = {"kv_usage"}


def summarize_scrapes(scrapes: list[dict[str, float]], n_servers: int = 1
                      ) -> tuple[dict[str, float | None], list[str]]:
    """Summaries over a list of (replica-summed) scrapes taken during the window."""
    if not scrapes:
        return {}, list(METRIC_TABLE)
    available = set().union(*[s.keys() for s in scrapes])
    mapping, missing = resolve_names(available)
    base = scrapes[0]
    last = scrapes[-1]
    out: dict[str, float | None] = {}
    for key, name in mapping.items():
        kind = METRIC_TABLE[key][0]
        if kind == "gauge":
            div = n_servers if key in PER_SERVER_FRACTIONS else 1
            vals = [s[name] / div for s in scrapes if name in s]
            out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
            out[f"{key}_max"] = max(vals) if vals else None
        elif kind == "counter":
            out[key] = last.get(name, 0.0) - base.get(name, 0.0)
        else:
            ds = last.get(f"{name}_sum", 0.0) - base.get(f"{name}_sum", 0.0)
            dc = last.get(f"{name}_count", 0.0) - base.get(f"{name}_count", 0.0)
            out[f"{key}_mean_s"] = ds / dc if dc > 0 else None
    q, h = out.get("prefix_queries"), out.get("prefix_hits")
    out["prefix_hit_rate"] = (h / q) if q and h is not None and q > 0 else None
    d, a = out.get("spec_draft_tokens"), out.get("spec_accepted_tokens")
    out["spec_accept_rate"] = (a / d) if d and a is not None and d > 0 else None
    n = out.get("spec_drafts")
    # output tokens per sequence per engine step (1 without speculative decoding)
    out["tokens_per_step"] = 1.0 + a / n if n and a is not None and n > 0 else None
    return out, missing


class PrometheusScraper:
    """Scrapes every server's /metrics periodically while running; sums across replicas per tick."""

    def __init__(self, metrics_urls: list[str], interval_s: float):
        self.urls = metrics_urls
        self.interval = interval_s
        self.scrapes: list[dict[str, float]] = []
        self.last_text: dict[str, str] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def scrape_once(self, session: aiohttp.ClientSession) -> dict[str, float]:
        total: dict[str, float] = {}
        for url in self.urls:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    text = await r.text()
            except Exception as e:  # noqa: BLE001 - telemetry must never fail a run
                log.warning("metrics scrape failed for %s: %s", url, e)
                continue
            self.last_text[url] = text
            for k, v in parse_prometheus(text).items():
                total[k] = total.get(k, 0.0) + v
        return total

    async def _loop(self) -> None:
        async with aiohttp.ClientSession() as session:
            while not self._stop.is_set():
                self.scrapes.append(await self.scrape_once(session))
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass
            self.scrapes.append(await self.scrape_once(session))

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    def summary(self) -> tuple[dict[str, float | None], list[str]]:
        return summarize_scrapes(self.scrapes, n_servers=len(self.urls))


# ---------------------------------------------------------------- NVML

class Nvml:
    """Thin wrapper; all calls degrade to empty results when NVML is unavailable."""

    def __init__(self) -> None:
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.ok = True
        except Exception as e:  # noqa: BLE001
            log.debug("NVML unavailable: %s", e)

    def count(self) -> int:
        return self.nv.nvmlDeviceGetCount() if self.ok else 0

    def handle(self, i: int) -> Any:
        return self.nv.nvmlDeviceGetHandleByIndex(i)

    def driver(self) -> str | None:
        if not self.ok:
            return None
        v = self.nv.nvmlSystemGetDriverVersion()
        return v.decode() if isinstance(v, bytes) else v

    def device_info(self, i: int) -> dict[str, Any]:
        h = self.handle(i)
        name = self.nv.nvmlDeviceGetName(h)
        mem = self.nv.nvmlDeviceGetMemoryInfo(h)
        cc = self.nv.nvmlDeviceGetCudaComputeCapability(h)
        return {"index": i, "name": name.decode() if isinstance(name, bytes) else name,
                "compute_capability": f"{cc[0]}.{cc[1]}", "mem_total_mb": mem.total / 2**20,
                "mem_used_mb": mem.used / 2**20, "mem_free_mb": mem.free / 2**20}

    def mem_used_mb(self, i: int) -> float:
        return self.nv.nvmlDeviceGetMemoryInfo(self.handle(i)).used / 2**20 if self.ok else 0.0

    def processes(self, i: int) -> list[tuple[int, float]]:
        """(pid, used MB) of compute processes on GPU i."""
        if not self.ok:
            return []
        procs = self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle(i))
        return [(p.pid, (p.usedGpuMemory or 0) / 2**20) for p in procs]

    def sample(self, i: int) -> dict[str, float]:
        h = self.handle(i)
        util = self.nv.nvmlDeviceGetUtilizationRates(h)
        return {"gpu_util": float(util.gpu),
                "mem_used_mb": self.nv.nvmlDeviceGetMemoryInfo(h).used / 2**20,
                "power_w": self.nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "sm_clock_mhz": float(self.nv.nvmlDeviceGetClockInfo(h, self.nv.NVML_CLOCK_SM))}


@dataclass
class NvmlSampler:
    """Background thread sampling the given GPUs."""
    gpu_ids: list[int]
    interval_s: float
    nvml: Nvml = field(default_factory=Nvml)
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for g in self.gpu_ids:
                    self.samples.append({"gpu": g, **self.nvml.sample(g)})
            except Exception as e:  # noqa: BLE001
                log.warning("NVML sample failed: %s", e)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if not self.nvml.ok:
            log.warning("NVML not available; GPU telemetry will be empty")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {k: None for k in ("gpu_util_mean", "gpu_mem_used_max_mb", "gpu_power_mean_w",
                                      "gpu_sm_clock_mean_mhz")}
        def mean(k: str) -> float:
            return sum(s[k] for s in self.samples) / len(self.samples)
        return {"gpu_util_mean": mean("gpu_util"),
                "gpu_mem_used_max_mb": max(s["mem_used_mb"] for s in self.samples),
                "gpu_power_mean_w": mean("power_w"),
                "gpu_sm_clock_mean_mhz": mean("sm_clock_mhz")}
