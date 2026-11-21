"""Engine protocol shared by the real vLLM launcher and the FakeEngine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

StartFailure = Literal["infeasible", "timeout", "failed"]


@dataclass(frozen=True)
class ServerSpec:
    """One `vllm serve` process."""
    argv: tuple[str, ...]
    env: dict[str, str]          # extra environment (CUDA_VISIBLE_DEVICES, ...)
    port: int
    gpu_ids: tuple[int, ...]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class EngineLaunch:
    """Everything needed to start one engine configuration (possibly several replicas)."""
    mask: int
    layout: str                          # e.g. tp1, tp4, tp2x2
    servers: tuple[ServerSpec, ...]
    flags: tuple[str, ...]               # engine flags shared by all replicas (no host/port)
    flags_key: str                       # hash of flags + layout: equal key => no restart needed
    served_model_name: str
    tuned: dict[str, dict[str, Any]] = field(default_factory=dict)   # param -> {value, source}

    @property
    def uses_placeholders(self) -> bool:
        return any(v["source"] == "placeholder" for v in self.tuned.values())


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    gpu_ids: tuple[int, ...]

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    @property
    def metrics_url(self) -> str:
        return f"{self.base_url}/metrics"


class EngineStartError(Exception):
    def __init__(self, status: StartFailure, kind: str, message: str):
        super().__init__(f"[{status}/{kind}] {message}")
        self.status: StartFailure = status
        self.kind = kind
        self.message = message


class Engine(Protocol):
    def start(self, launch: EngineLaunch, log_dir: Path) -> list[Endpoint]: ...
    def stop(self) -> None: ...
    def health(self) -> bool: ...
    def info(self) -> dict[str, Any]:
        """Values the engine reported at start-up (e.g. effective max_num_seqs)."""
        ...
