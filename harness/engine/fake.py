"""FakeEngine: synthetic engine + measurements with planted effects, for dry runs without a GPU.

Planted (log-throughput) structure, so the analysis has something known to recover:
  main effects depend on the workload point (e.g. prefix reuse scales with the reuse level, batching helps
  only at high concurrency, speculative decoding hurts at high concurrency);
  two interactions: quant x kvq (negative) and prefix x chunk (positive).
It also exercises infeasible configurations, latency-gate failures, an injected crash and replica layouts.
"""
from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness import optimizations as opt
from harness.config import EnvCfg, FakeCfg
from harness.engine.base import Endpoint, EngineLaunch, EngineStartError
from harness.regimes import WorkloadPoint

PLANTED_INTERACTIONS = {(1, 2): -0.05, (4, 6): 0.06}
FAKE_DEFAULT_MAX_NUM_SEQS = 256


def planted_main_effects(point: WorkloadPoint) -> dict[int, float]:
    """Log-throughput gain of switching each optimization on, at this point."""
    conc, ctx_hi = point.concurrency, point.varied == "context" and point.level == "high"
    return {
        1: 0.20 if conc <= 32 else 0.08,
        2: 0.03 + (0.10 if ctx_hi or conc > 32 else 0.0),
        3: 0.25 if conc <= 2 else (0.10 if conc <= 32 else -0.10),
        4: 0.40 * point.reuse,
        5: 0.12 if conc > 32 else 0.0,
        6: 0.04 + (0.05 if ctx_hi else 0.0),
        7: 0.20,
    }


def planted_log_throughput(mask: int, point: WorkloadPoint) -> float:
    base = math.log(1000.0) + 0.3 * math.log(max(point.concurrency, 1) / 32 + 1) - 0.1 * math.log(
        max(point.context, 1) / 64)
    eff = planted_main_effects(point)
    val = base + sum(eff[n] for n in opt.on_numbers(mask))
    for (a, b), v in PLANTED_INTERACTIONS.items():
        if opt.is_on(mask, a) and opt.is_on(mask, b):
            val += v
    return val


class FakeEngine:
    def __init__(self, env: EnvCfg, fake: FakeCfg):
        self.env = env
        self.fake = fake
        self.current: EngineLaunch | None = None
        self.launch_log: list[EngineLaunch] = []
        self.starts = 0

    def _validate_layout(self, launch: EngineLaunch) -> None:
        """The replica layout must partition the env's GPUs with distinct ports (4-GPU path check)."""
        used: list[int] = []
        ports = set()
        for s in launch.servers:
            cvd = [int(x) for x in s.env["CUDA_VISIBLE_DEVICES"].split(",")]
            if tuple(cvd) != s.gpu_ids:
                raise AssertionError(f"CUDA_VISIBLE_DEVICES {cvd} != gpu_ids {s.gpu_ids}")
            used += cvd
            ports.add(s.port)
            if f"--port" not in s.argv or str(s.port) not in s.argv:
                raise AssertionError("server argv lacks its port")
        if sorted(used) != sorted(self.env.gpu_ids) or len(ports) != len(launch.servers):
            raise AssertionError(f"layout {launch.layout} does not partition GPUs {self.env.gpu_ids}: {used}")

    def start(self, launch: EngineLaunch, log_dir: Path) -> list[Endpoint]:
        self._validate_layout(launch)
        on = set(opt.on_numbers(launch.mask))
        for combo in self.fake.infeasible_combos:
            if set(combo) <= on:
                raise EngineStartError("infeasible", "unsupported",
                                       f"FakeEngine: combination {sorted(combo)} is not supported (planted)")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "server_0.log").write_text("FakeEngine\n$ " + " ".join(launch.servers[0].argv) + "\n")
        self.current = launch
        self.launch_log.append(launch)
        self.starts += 1
        return [Endpoint(s.base_url, s.gpu_ids) for s in launch.servers]

    def stop(self) -> None:
        self.current = None

    def health(self) -> bool:
        return self.current is not None

    def info(self) -> dict[str, Any]:
        return {"startup_s": 0.0, "max_num_seqs": FAKE_DEFAULT_MAX_NUM_SEQS, "kv_cache_tokens": 200000,
                "chunked_prefill_enabled": bool(self.current and opt.is_on(self.current.mask, 6))}

    # ---------------------------------------------------------------- measurement
    def measure(self, point: WorkloadPoint, rep: int, output_tokens: int, max_num_seqs: int,
                ) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
        assert self.current is not None, "engine not started"
        mask = self.current.mask
        crash = self.fake.crash
        if crash and crash.get("env") == self.env.name and crash.get("point") == point.point_id \
                and crash.get("config") == opt.to_bits(mask) and int(crash.get("rep", 1)) == rep:
            raise RuntimeError("FakeEngine: injected crash during measurement (planted)")
        seed = zlib.crc32(f"{self.fake.seed}|{self.env.name}|{point.point_id}|{mask}|{rep}".encode())
        rng = np.random.default_rng(seed)
        log_t = planted_log_throughput(mask, point) + rng.normal(0, self.fake.noise_sd)
        thr = math.exp(log_t)
        conc = point.concurrency
        running = float(min(conc, max_num_seqs))
        tpot = 0.7 * 1000.0 * running / thr
        ttft = 50.0 + 0.5 * point.context * (1 + conc / 32)
        if opt.is_on(mask, 5) and point.varied == "concurrency" and point.level == "high":
            ttft *= 40.0   # planted latency-gate failure: aggressive batching at high concurrency
        kv = min(1.0, 0.35 + 0.4 * point.context / 128 * conc / 32) * (0.5 if opt.is_on(mask, 2) else 1.0)
        metrics = {
            "window_s": 1.0, "n_requests_window": 20, "n_requests_total": 24, "n_errors": 0,
            "n_wrong_output_len": 0, "output_tok_s": thr, "req_s": thr / output_tokens,
            "ttft_p50_ms": ttft * 0.6, "ttft_p99_ms": ttft, "ttft_mean_ms": ttft * 0.65,
            "tpot_p50_ms": tpot * 0.8, "tpot_p99_ms": tpot, "tpot_mean_ms": tpot * 0.82,
            "e2e_p50_ms": ttft * 0.6 + tpot * output_tokens, "e2e_p99_ms": ttft + tpot * output_tokens * 1.2,
            "e2e_mean_ms": ttft * 0.65 + tpot * output_tokens, "trace_wrapped": False, "concurrency": conc,
        }
        prefix_on = opt.is_on(mask, 4)
        telemetry = {
            "gpu_util_mean": 85.0, "gpu_mem_used_max_mb": 20000.0, "gpu_power_mean_w": 250.0,
            "gpu_sm_clock_mean_mhz": 2200.0,
            "kv_usage_mean": kv * 0.8, "kv_usage_max": kv,
            "running_mean": running, "running_max": running,
            "waiting_mean": max(0.0, conc - running), "waiting_max": max(0.0, conc - running),
            "preemptions": 3.0 if kv >= 0.99 else 0.0,
            "prefix_hits": 1000.0 * point.reuse * 0.75 if prefix_on else 0.0,
            "prefix_queries": 1000.0 if prefix_on else 0.0,
            "prefix_hit_rate": point.reuse * 0.75 if prefix_on else None,
            "spec_draft_tokens": 400.0 if opt.is_on(mask, 3) else None,
            "spec_accepted_tokens": 240.0 if opt.is_on(mask, 3) else None,
            "spec_accept_rate": 0.6 if opt.is_on(mask, 3) else None,
            "queue_time_mean_s": 0.01,
            "prompt_tokens": 20.0 * point.context, "generation_tokens": thr,
            "telemetry_missing": [],
        }
        n = 20
        req = pd.DataFrame({
            "idx": range(n), "trace_idx": range(n), "endpoint": [i % len(self.current.servers) for i in range(n)],
            "prompt_tokens": point.context, "prefix_id": -1, "send_t": np.arange(n, dtype=float),
            "first_t": np.arange(n) + ttft / 1e3, "end_t": np.arange(n) + ttft / 1e3 + tpot * output_tokens / 1e3,
            "n_out": output_tokens, "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": ttft + tpot * output_tokens,
            "in_window": True, "ok": True, "error": None})
        return metrics, telemetry, req


def fake_quality(numeric_label: str, fake: FakeCfg, tasks: list[str]) -> dict[str, float]:
    """Accuracy (points) per task for a numeric setting such as '101'."""
    base = {"gsm8k": 78.0, "mmlu": 66.0, "humaneval": 60.0}
    drop = fake.quality_drop.get(numeric_label, 0.0)
    jitter = {lab: 0.1 * (i % 3) for i, lab in enumerate(["000", "100", "010", "001", "110", "101", "011", "111"])}
    return {t: base.get(t, 50.0) - drop - jitter.get(numeric_label, 0.0) for t in tasks}
