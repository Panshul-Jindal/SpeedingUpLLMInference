"""Closed-loop async load generator against OpenAI-compatible /v1/completions (streaming).

Exactly C requests are in flight. Timeline: [start, t0) warm-up (discarded), [t0, t1) measurement window.
At t1 in-flight requests are cancelled. Latency percentiles use requests that start and finish inside the
window; output tokens/s counts every token whose arrival time falls inside the window.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

from harness.engine.base import Endpoint
from harness.workload import Trace

log = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    idx: int
    trace_idx: int
    endpoint: int
    prompt_tokens: int
    prefix_id: int
    send_t: float
    first_t: float | None = None
    end_t: float | None = None
    n_out: int = 0
    events: list[tuple[float, int]] = field(default_factory=list)   # (arrival time, new tokens)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.end_t is not None


@dataclass
class LoadResult:
    metrics: dict[str, Any]
    requests: pd.DataFrame
    t0: float
    t1: float


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(records: list[RequestRecord], t0: float, t1: float, expected_out: int | None = None
              ) -> tuple[dict[str, Any], pd.DataFrame]:
    window = t1 - t0
    rows = []
    ttft, tpot, e2e = [], [], []
    tokens_in_window = 0
    completed_in_window = 0
    short_outputs = 0
    for r in records:
        tokens_in_window += sum(k for t, k in r.events if t0 <= t < t1)
        in_win = r.ok and r.send_t >= t0 and r.end_t is not None and r.end_t <= t1
        row = {"idx": r.idx, "trace_idx": r.trace_idx, "endpoint": r.endpoint, "prompt_tokens": r.prompt_tokens,
               "prefix_id": r.prefix_id, "send_t": r.send_t, "first_t": r.first_t, "end_t": r.end_t,
               "n_out": r.n_out, "ttft_ms": None, "tpot_ms": None, "e2e_ms": None,
               "in_window": in_win, "ok": r.ok, "error": r.error}
        if r.ok and r.first_t is not None and r.end_t is not None:
            row["ttft_ms"] = (r.first_t - r.send_t) * 1e3
            row["e2e_ms"] = (r.end_t - r.send_t) * 1e3
            last_t = r.events[-1][0] if r.events else r.end_t
            if r.n_out > 1:
                row["tpot_ms"] = (last_t - r.first_t) * 1e3 / (r.n_out - 1)
        if in_win:
            completed_in_window += 1
            ttft.append(row["ttft_ms"])
            e2e.append(row["e2e_ms"])
            if row["tpot_ms"] is not None:
                tpot.append(row["tpot_ms"])
            if expected_out is not None and r.n_out != expected_out:
                short_outputs += 1
        rows.append(row)
    metrics: dict[str, Any] = {
        "window_s": window,
        "n_requests_window": completed_in_window,
        "n_requests_total": len(records),
        "n_errors": sum(1 for r in records if r.error and r.error != "cancelled"),
        "n_wrong_output_len": short_outputs,
        "output_tok_s": tokens_in_window / window if window > 0 else None,
        "req_s": completed_in_window / window if window > 0 else None,
    }
    for name, vals in (("ttft", ttft), ("tpot", tpot), ("e2e", e2e)):
        metrics[f"{name}_p50_ms"] = _pct(vals, 50)
        metrics[f"{name}_p99_ms"] = _pct(vals, 99)
        metrics[f"{name}_mean_ms"] = float(np.mean(vals)) if vals else None
    return metrics, pd.DataFrame(rows)


async def _one_request(session: aiohttp.ClientSession, ep: Endpoint, model: str, prompt: list[int],
                       output_tokens: int, rec: RequestRecord, timeout_s: float) -> None:
    body = {"model": model, "prompt": prompt, "max_tokens": output_tokens, "min_tokens": output_tokens,
            "ignore_eos": True, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    try:
        async with session.post(ep.completions_url, json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                rec.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return
            buf = b""
            async for chunk in resp.content.iter_any():
                now = time.perf_counter()
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    msg = json.loads(data)
                    usage = msg.get("usage") or {}
                    choices = msg.get("choices") or []
                    if "completion_tokens" in usage and usage["completion_tokens"] is not None:
                        new = int(usage["completion_tokens"]) - rec.n_out
                    else:
                        new = 1 if choices and choices[0].get("text") else 0
                    if new > 0:
                        if rec.first_t is None:
                            rec.first_t = now
                        rec.events.append((now, new))
                        rec.n_out += new
            rec.end_t = time.perf_counter()
    except asyncio.CancelledError:
        rec.error = "cancelled"
        raise
    except Exception as e:  # noqa: BLE001 - a failed request is data, not a crash
        rec.error = f"{type(e).__name__}: {e}"[:300]


async def run_closed_loop(endpoints: list[Endpoint], model: str, trace: Trace, concurrency: int,
                          warmup_s: float, window_s: float, request_timeout_s: float,
                          on_window_start: Callable[[], Awaitable[None]] | None = None,
                          on_window_end: Callable[[], Awaitable[None]] | None = None,
                          ) -> LoadResult:
    records: list[RequestRecord] = []
    counter = 0
    n_trace = len(trace)
    wrapped = False

    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.perf_counter()
        t0 = start + warmup_s
        t1 = t0 + window_s

        async def worker() -> None:
            nonlocal counter, wrapped
            while time.perf_counter() < t1:
                i = counter
                counter += 1
                ti = i % n_trace
                if i >= n_trace:
                    wrapped = True
                ep_i = i % len(endpoints)
                rec = RequestRecord(i, ti, ep_i, trace.prompt_len(ti), int(trace.prefix_ids[ti]),
                                    time.perf_counter())
                records.append(rec)
                await _one_request(session, endpoints[ep_i], model, trace.prompt(ti), trace.output_tokens,
                                   rec, request_timeout_s)
                if rec.error and not rec.error.startswith("cancelled"):
                    await asyncio.sleep(0.05)   # do not spin on a broken server

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.sleep(max(0.0, t0 - time.perf_counter()))
        if on_window_start:
            await on_window_start()
        await asyncio.sleep(max(0.0, t1 - time.perf_counter()))
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if on_window_end:
            await on_window_end()

    if wrapped:
        log.warning("trace of %d requests wrapped around (%d sent); increase workload.trace_requests",
                    n_trace, counter)
    metrics, df = summarize(records, t0, t1, expected_out=trace.output_tokens)
    metrics["trace_wrapped"] = wrapped
    metrics["concurrency"] = concurrency
    errs = df.loc[df["error"].notna() & (df["error"] != "cancelled"), "error"]
    if len(errs):
        metrics["first_error"] = str(errs.iloc[0])
    return LoadResult(metrics, df, t0, t1)


async def single_request(endpoint: Endpoint, model: str, prompt: list[int] | str, max_tokens: int,
                         timeout_s: float = 300.0) -> dict[str, Any]:
    """Non-streaming greedy request (start-up check, agreement prompts)."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint.completions_url, json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
            return json.loads(text)


async def post_all(endpoints: list[Endpoint], path: str, timeout_s: float = 30.0) -> bool:
    """POST `path` on every server; True if all answered 200."""
    ok = True
    async with aiohttp.ClientSession() as session:
        for ep in endpoints:
            try:
                async with session.post(f"{ep.base_url}{path}", timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
                    ok &= r.status == 200
            except Exception:  # noqa: BLE001
                ok = False
    return ok


async def reset_prefix_cache(endpoints: list[Endpoint]) -> bool:
    """POST /reset_prefix_cache on every server (needs VLLM_SERVER_DEV_MODE=1)."""
    return await post_all(endpoints, "/reset_prefix_cache")
