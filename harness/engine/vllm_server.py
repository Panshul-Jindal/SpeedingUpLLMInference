"""Launch `vllm serve` processes (one per replica), wait for health, capture logs, tear down cleanly."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import IO, Any

import psutil

from harness.config import REPO_ROOT
from harness.engine.base import Endpoint, EngineLaunch, EngineStartError, ServerSpec
from harness.telemetry import Nvml

log = logging.getLogger(__name__)

# Log patterns -> failure kind. Start-up failures are "infeasible" results, not crashes (spec 4.6).
_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("port_in_use", r"Address already in use|EADDRINUSE"),
    ("oom", r"CUDA out of memory|OutOfMemoryError|No available memory for the cache blocks|"
            r"larger than the (?:maximum number of tokens|available KV cache)|"
            r"is larger than the maximum number of tokens that can be stored in KV cache|"
            r"Insufficient memory|not enough KV cache"),
    ("unsupported", r"NotImplementedError|not supported|is not supported|Unsupported|"
                    r"does not support|unsupported"),
    ("download", r"RepositoryNotFoundError|GatedRepoError|401 Client Error|404 Client Error|"
                 r"Cannot access gated repo|LocalEntryNotFoundError"),
    ("bad_argument", r"error: unrecognized arguments|error: argument|invalid choice"),
)

# Values the engine reports at start-up; patterns tried in order.
_INFO_PATTERNS: dict[str, tuple[str, ...]] = {
    "max_num_seqs": (r"max_num_seqs[=:]\s*(\d+)",),
    "max_num_batched_tokens": (r"max_num_batched_tokens[=:]\s*(\d+)",),
    "chunked_prefill_enabled": (r"enable_chunked_prefill[=:]\s*(True|False)",
                                r"(Chunked prefill is enabled)"),
    "kv_cache_tokens": (r"GPU KV cache size:\s*([\d,]+)\s*tokens",),
    "max_concurrency": (r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x",),
    "prefix_caching_enabled": (r"enable_prefix_caching[=:]\s*(True|False)",),
}


def toolchain_env() -> dict[str, str]:
    """CUDA_HOME/PATH for FlashInfer's JIT (it compiles some kernels, e.g. FP8-KV XQA decode, with nvcc).

    Uses the CUDA 12.9 toolkit that scripts/setup_env.sh installs in <repo>/.cuda-toolkit when present (it
    matches the cu129 wheels, unlike e.g. a cluster's cuda-13 module), else nvcc from PATH.
    Not part of the engine flags: it does not change what is measured.
    """
    local = REPO_ROOT / ".cuda-toolkit"
    if (local / "bin" / "nvcc").exists():
        return {"CUDA_HOME": str(local), "PATH": f"{local / 'bin'}:{os.environ.get('PATH', '')}"}
    return {}


# Start-up failures that say nothing about the configuration: recorded as `failed` (retryable), not infeasible.
TRANSIENT_KINDS = ("port_in_use",)


def port_free(port: int) -> bool:
    """True unless another socket is listening on the port. SO_REUSEADDR, as uvicorn uses: connections of
    our own previous server lingering in TIME_WAIT must not count as busy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sk.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def classify_failure(log_text: str) -> str:
    for kind, pat in _FAILURE_PATTERNS:
        if re.search(pat, log_text):
            return kind
    return "startup_error"


def error_excerpt(log_text: str, max_lines: int = 12) -> str:
    """The most informative lines of a failed start-up log."""
    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    errs = [ln for ln in lines if re.search(r"Error|error|Exception|exception|Traceback|FAILED", ln)]
    pick = errs[-max_lines:] if errs else lines[-max_lines:]
    return "\n".join(pick)[-3000:]


def parse_engine_info(log_text: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for key, pats in _INFO_PATTERNS.items():
        for pat in pats:
            m = re.search(pat, log_text)
            if m:
                raw = m.group(1)
                if raw in ("True", "False"):
                    info[key] = raw == "True"
                elif raw.startswith("Chunked"):
                    info[key] = True
                else:
                    num = raw.replace(",", "")
                    info[key] = float(num) if "." in num else int(num)
                break
    return info


def fetch_server_config(base_url: str, save_to: Path | None = None) -> dict[str, Any]:
    """Effective engine settings from GET /server_info (needs VLLM_SERVER_DEV_MODE=1).

    vLLM 0.31 does not log the resolved max_num_seqs, so this is the engine-reported value we record.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/server_info?config_format=json", timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        log.warning("could not read %s/server_info: %s", base_url, e)
        return {}
    if save_to is not None:
        save_to.write_text(json.dumps(data, indent=1, default=str))
    cfg = data.get("vllm_config") or {}
    if not isinstance(cfg, dict):
        return {}
    sched, cache = cfg.get("scheduler_config") or {}, cfg.get("cache_config") or {}
    out = {"max_num_seqs": sched.get("max_num_seqs"),
           "max_num_batched_tokens": sched.get("max_num_batched_tokens"),
           "chunked_prefill_enabled": sched.get("enable_chunked_prefill"),
           "prefix_caching_enabled": cache.get("enable_prefix_caching"),
           "kv_cache_dtype": cache.get("cache_dtype"),
           "num_gpu_blocks": cache.get("num_gpu_blocks")}
    return {k: v for k, v in out.items() if v is not None}


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


class VllmServerGroup:
    """Engine implementation for real vLLM: k independent `vllm serve` processes."""

    def __init__(self, startup_timeout_s: float, gpu_release_timeout_s: float, nvml: Nvml | None = None):
        self.startup_timeout_s = startup_timeout_s
        self.gpu_release_timeout_s = gpu_release_timeout_s
        self.nvml = nvml or Nvml()
        self._procs: list[tuple[ServerSpec, subprocess.Popen[bytes], Path, IO[bytes]]] = []
        self._pids: set[int] = set()
        self._info: dict[str, Any] = {}
        self.current: EngineLaunch | None = None

    # -------------------------------------------------------------- lifecycle
    def start(self, launch: EngineLaunch, log_dir: Path) -> list[Endpoint]:
        if self._procs:
            self.stop()
        log_dir.mkdir(parents=True, exist_ok=True)
        # A shared node may have another server on our port; its /health would look like ours.
        busy = [s.port for s in launch.servers if not port_free(s.port)]
        if busy:
            raise EngineStartError("failed", "port_in_use", f"ports {busy} are in use by another process")
        t0 = time.monotonic()
        for i, spec in enumerate(launch.servers):
            log_path = log_dir / f"server_{i}.log"
            fh = open(log_path, "ab")
            fh.write(("$ " + " ".join(spec.argv) + f"\n# env: {spec.env}\n").encode())
            fh.flush()
            env = {**os.environ, **toolchain_env(), **spec.env}
            proc = subprocess.Popen(spec.argv, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)  # own process group
            self._procs.append((spec, proc, log_path, fh))
            log.info("started vllm serve pid=%d port=%d gpus=%s", proc.pid, spec.port, spec.gpu_ids)
        self.current = launch
        try:
            self._wait_healthy(t0)
        except EngineStartError:
            self.stop()
            raise
        self._info = {"startup_s": round(time.monotonic() - t0, 1),
                      **parse_engine_info(self._log_text(0)),
                      **fetch_server_config(launch.servers[0].base_url, log_dir / "server_info.json")}
        return [Endpoint(s.base_url, s.gpu_ids) for s, *_ in self._procs]

    def _log_text(self, i: int) -> str:
        try:
            return self._procs[i][2].read_text(errors="replace")
        except (IndexError, OSError):
            return ""

    def _wait_healthy(self, t0: float) -> None:
        pending = set(range(len(self._procs)))
        while pending:
            for i in list(pending):
                spec, proc, _, _ = self._procs[i]
                self._collect_pids(proc.pid)
                if proc.poll() is not None:
                    text = self._log_text(i)
                    kind = classify_failure(text)
                    raise EngineStartError("failed" if kind in TRANSIENT_KINDS else "infeasible", kind,
                                           f"server {i} exited with code {proc.returncode} during start-up:\n"
                                           + error_excerpt(text))
                if _http_ok(f"{spec.base_url}/health"):
                    pending.discard(i)
            if not pending:
                return
            if time.monotonic() - t0 > self.startup_timeout_s:
                raise EngineStartError("timeout", "startup_timeout",
                                       f"servers {sorted(pending)} not healthy after {self.startup_timeout_s:.0f}s")
            time.sleep(2.0)

    def _collect_pids(self, pid: int) -> None:
        try:
            p = psutil.Process(pid)
            self._pids.add(pid)
            self._pids.update(c.pid for c in p.children(recursive=True))
        except psutil.Error:
            pass

    def root_pids(self) -> list[int]:
        """PID of each running `vllm serve` (its process tree holds the API server and EngineCore)."""
        return [proc.pid for _, proc, _, _ in self._procs]

    def health(self) -> bool:
        return bool(self._procs) and all(
            proc.poll() is None and _http_ok(f"{spec.base_url}/health") for spec, proc, _, _ in self._procs)

    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def stop(self) -> None:
        if not self._procs:
            return
        for _, proc, _, _ in self._procs:
            self._collect_pids(proc.pid)
        gpu_ids = sorted({g for spec, *_ in self._procs for g in spec.gpu_ids})
        for _, proc, _, _ in self._procs:
            _killpg(proc, signal.SIGTERM)
        deadline = time.monotonic() + 30
        for _, proc, _, _ in self._procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for _, proc, _, fh in self._procs:
            _killpg(proc, signal.SIGKILL)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.error("pid %d did not exit after SIGKILL", proc.pid)
            fh.close()
        for pid in list(self._pids):  # stragglers that left the process group
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        self._procs.clear()
        self.current = None
        self._wait_gpu_released(gpu_ids)
        self._pids.clear()

    def _wait_gpu_released(self, gpu_ids: list[int]) -> None:
        """Confirm through NVML that none of our processes still hold GPU memory."""
        if not self.nvml.ok:
            time.sleep(5)
            return
        deadline = time.monotonic() + self.gpu_release_timeout_s
        while True:
            holders = [(g, pid, mb) for g in gpu_ids for pid, mb in self.nvml.processes(g) if pid in self._pids]
            if not holders:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"GPU memory not released after {self.gpu_release_timeout_s:.0f}s: {holders}")
            time.sleep(1.0)


def _killpg(proc: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass
