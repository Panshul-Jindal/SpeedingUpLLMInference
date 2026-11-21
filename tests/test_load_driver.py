"""Load driver against a local mock OpenAI streaming server."""
import asyncio
import json

import pytest
from aiohttp import web

from harness.engine.base import Endpoint
from harness.load_driver import RequestRecord, run_closed_loop, summarize
from harness.config import WorkloadCfg
from harness.regimes import WorkloadPoint
from harness.workload import FakeTokenizer, generate_trace

TTFT_S = 0.05
GAP_S = 0.01


def _app(state):
    async def completions(request):
        body = await request.json()
        state["inflight"] += 1
        state["max_inflight"] = max(state["max_inflight"], state["inflight"])
        state["prompt_lens"].append(len(body["prompt"]))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            await asyncio.sleep(TTFT_S)
            n = body["max_tokens"]
            for k in range(1, n + 1):
                if k > 1:
                    await asyncio.sleep(GAP_S)
                msg = {"choices": [{"text": "x"}], "usage": {"completion_tokens": k}}
                await resp.write(f"data: {json.dumps(msg)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
        finally:
            state["inflight"] -= 1
        return resp

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    return app


async def _serve(state):
    runner = web.AppRunner(_app(state))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


async def test_closed_loop_against_mock():
    state = {"inflight": 0, "max_inflight": 0, "prompt_lens": []}
    runner, port = await _serve(state)
    try:
        wl = WorkloadCfg(output_tokens=10, trace_requests=500)
        tr = generate_trace(WorkloadPoint("center", 32, 4, 0.5, None, "mid"), wl, 1, FakeTokenizer())
        ep = Endpoint(f"http://127.0.0.1:{port}", (0,))
        res = await run_closed_loop([ep], "m", tr, concurrency=4, warmup_s=0.3, window_s=1.5,
                                    request_timeout_s=10)
    finally:
        await runner.cleanup()
    m = res.metrics
    assert state["max_inflight"] == 4                     # exactly C in flight
    assert set(state["prompt_lens"]) == {32}
    assert m["n_requests_window"] > 0 and m["n_errors"] == 0
    assert m["ttft_p50_ms"] == pytest.approx(TTFT_S * 1e3, rel=0.5, abs=15)
    assert m["tpot_p50_ms"] == pytest.approx(GAP_S * 1e3, rel=0.6, abs=5)
    # window filtering: every counted request started and finished inside [t0, t1]
    w = res.requests[res.requests["in_window"]]
    assert (w["send_t"] >= res.t0).all() and (w["end_t"] <= res.t1).all()
    assert (w["n_out"] == 10).all()
    # 4 streams, ~ (10 tokens per 0.14 s) each => ~ 280 tok/s
    assert 120 < m["output_tok_s"] < 400


def test_summarize_percentiles_and_window():
    recs = []
    # 100 requests in window with TTFT 1..100 ms, 5 tokens each 10 ms apart
    for i in range(100):
        s = 10.0 + i * 0.01
        f = s + (i + 1) / 1000
        r = RequestRecord(i, i, 0, 8, -1, s, first_t=f, end_t=f + 0.04, n_out=5,
                          events=[(f + k * 0.01, 1) for k in range(5)])
        recs.append(r)
    # one request starting before the window, one ending after it: excluded from latency stats
    recs.append(RequestRecord(100, 0, 0, 8, -1, 9.0, first_t=9.5, end_t=10.5, n_out=5,
                              events=[(9.5 + k * 0.2, 1) for k in range(5)]))
    recs.append(RequestRecord(101, 0, 0, 8, -1, 19.9, first_t=19.95, end_t=20.5, n_out=5,
                              events=[(19.95 + k * 0.1, 1) for k in range(5)]))
    m, df = summarize(recs, 10.0, 20.0, expected_out=5)
    assert m["n_requests_window"] == 100
    assert m["ttft_p50_ms"] == pytest.approx(50.5, abs=0.01)
    assert m["ttft_p99_ms"] == pytest.approx(99.01, abs=0.01)
    assert m["tpot_p50_ms"] == pytest.approx(10.0, abs=1e-6)
    # tokens inside the window: 500 + 3 (boundary req 1: 10.1, 10.3...) -> count exactly
    in_win_extra = sum(1 for t in [9.5 + k * 0.2 for k in range(5)] if t >= 10) + \
        sum(1 for t in [19.95 + k * 0.1 for k in range(5)] if t < 20)
    assert m["output_tok_s"] == pytest.approx((500 + in_win_extra) / 10.0)
    assert m["req_s"] == pytest.approx(10.0)
