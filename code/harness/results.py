"""Append-only results store (JSONL), resume by run key, per-run artifact directories, version capture."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from harness.config import REPO_ROOT

STATUSES = ("ok", "infeasible", "failed", "timeout")
KEY_FIELDS = ("profile", "env", "config_bits", "point_id", "rep")


def run_key(profile: str, env: str, config_bits: str, point_id: str, rep: int | str) -> str:
    return f"{profile}__{env}__{point_id}__{config_bits}__r{rep}"


class ResultStore:
    """One JSONL file per kind (runs, startup, sweep, quality); the latest row per key wins."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue   # a torn last line after a hard kill
        return out

    def latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for r in self.rows():
            latest[r["run_key"]] = r
        return latest

    def done_keys(self, retry_failed: bool = False) -> set[str]:
        skip = {"failed", "timeout"} if retry_failed else set()
        return {k for k, r in self.latest().items() if r.get("status") not in skip}

    def append(self, row: dict[str, Any]) -> None:
        row = {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
        line = json.dumps(row, default=_json_default) + "\n"
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)   # concurrent slot workers append to the same file
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.latest().values()))


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def run_dir(results_dir: Path, key: str) -> Path:
    d = results_dir / "runs" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_artifacts(d: Path, *, flags: Any = None, trace_hash: str | None = None,
                        metrics_snapshot: dict[str, str] | None = None,
                        requests: pd.DataFrame | None = None) -> None:
    if flags is not None:
        (d / "flags.json").write_text(json.dumps(flags, indent=2, default=_json_default))
    if trace_hash is not None:
        (d / "trace_hash.txt").write_text(trace_hash + "\n")
    if metrics_snapshot:
        (d / "metrics_snapshot.txt").write_text(
            "\n".join(f"# {url}\n{text}" for url, text in metrics_snapshot.items()))
    if requests is not None and len(requests):
        requests.to_parquet(d / "requests.parquet", index=False)


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@cache
def versions() -> dict[str, Any]:
    """Versions recorded in every result row (no heavy imports)."""
    torch_v = _pkg_version("torch")
    cuda = torch_v.split("+cu")[-1] if torch_v and "+cu" in torch_v else None
    if cuda and cuda.isdigit():
        cuda = f"{cuda[:-1]}.{cuda[-1]}"
    driver = None
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
    except Exception:  # noqa: BLE001
        pass
    return {"vllm_version": _pkg_version("vllm"), "torch_version": torch_v, "cuda_version": cuda,
            "driver_version": driver, "lm_eval_version": _pkg_version("lm-eval"),
            "harness_git": git_hash()}


def git_hash() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                           text=True, timeout=10).stdout.strip() or "nogit"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "nogit"
