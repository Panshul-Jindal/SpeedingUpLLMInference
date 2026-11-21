"""Process control of the vLLM launcher, using a tiny stand-in HTTP server instead of vLLM."""
import sys
import time

import psutil
import pytest

from harness.engine.base import EngineLaunch, EngineStartError, ServerSpec
from harness.engine.vllm_server import VllmServerGroup, classify_failure, parse_engine_info

SERVER = r"""
import http.server, socketserver, sys, subprocess, time
port = int(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])  # like EngineCore
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
print("max_num_seqs=256 enable_chunked_prefill=False", flush=True)
print("GPU KV cache size: 34,000 tokens", flush=True)
socketserver.TCPServer(("127.0.0.1", port), H).serve_forever()
"""


def _launch(port, script=SERVER):
    spec = ServerSpec((sys.executable, "-c", script, str(port)), {"CUDA_VISIBLE_DEVICES": ""}, port, (0,))
    return EngineLaunch(0, "tp1", (spec,), ("m",), "k", "m")


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_start_health_info_and_group_teardown(tmp_path):
    g = VllmServerGroup(startup_timeout_s=30, gpu_release_timeout_s=5)
    eps = g.start(_launch(_free_port()), tmp_path)
    assert g.health() and len(eps) == 1
    info = g.info()
    assert info["max_num_seqs"] == 256 and info["kv_cache_tokens"] == 34000
    assert info["chunked_prefill_enabled"] is False
    pids = set(g._pids)
    assert len(pids) >= 2                                   # server + child
    g.stop()
    time.sleep(0.5)
    assert not any(psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE for p in pids)
    assert (tmp_path / "server_0.log").read_text().startswith("$ ")


def test_startup_failure_is_infeasible(tmp_path):
    g = VllmServerGroup(startup_timeout_s=30, gpu_release_timeout_s=5)
    bad = "import sys; print('torch.OutOfMemoryError: CUDA out of memory'); sys.exit(1)"
    with pytest.raises(EngineStartError) as e:
        g.start(_launch(_free_port(), bad), tmp_path)
    assert e.value.status == "infeasible" and e.value.kind == "oom"


def test_startup_timeout(tmp_path):
    g = VllmServerGroup(startup_timeout_s=3, gpu_release_timeout_s=5)
    with pytest.raises(EngineStartError) as e:
        g.start(_launch(_free_port(), "import time; time.sleep(60)"), tmp_path)
    assert e.value.status == "timeout"


def test_classify():
    assert classify_failure("ValueError: ngram is not supported with pipeline parallel") == "unsupported"
    assert classify_failure("No available memory for the cache blocks") == "oom"
    assert classify_failure("something else") == "startup_error"
    assert parse_engine_info("Chunked prefill is enabled with max_num_batched_tokens=2048.") == {
        "chunked_prefill_enabled": True, "max_num_batched_tokens": 2048}
