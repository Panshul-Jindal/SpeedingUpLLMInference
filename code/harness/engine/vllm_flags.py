"""The ONLY place where a configuration is turned into vLLM CLI flags.

Verified against `vllm serve --help=all` for vLLM 0.31.0 (docs/vllm_serve_help_0.31.0.txt).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness import optimizations as opt
from harness.config import EnvCfg, ExperimentCfg
from harness.engine.base import EngineLaunch, ServerSpec

# Every flag this module can emit; checked against `vllm serve --help` by doctor / start-up.
REQUIRED_FLAGS: tuple[str, ...] = (
    "--host", "--port", "--served-model-name", "--dtype", "--max-model-len", "--gpu-memory-utilization",
    "--seed", "--quantization", "--kv-cache-dtype", "--speculative-config",
    "--enable-prefix-caching", "--no-enable-prefix-caching", "--max-num-seqs",
    "--enable-chunked-prefill", "--no-enable-chunked-prefill", "--max-num-batched-tokens",
    "--tensor-parallel-size", "--pipeline-parallel-size", "--profiler-config", "--enable-mfu-metrics",
)

_LAYOUT_RE = re.compile(r"^tp(\d+)(?:pp(\d+))?(?:x(\d+))?$")


@dataclass(frozen=True)
class Layout:
    tp: int
    pp: int
    replicas: int

    @property
    def gpus_per_replica(self) -> int:
        return self.tp * self.pp

    @property
    def name(self) -> str:
        s = f"tp{self.tp}"
        if self.pp > 1:
            s += f"pp{self.pp}"
        if self.replicas > 1:
            s += f"x{self.replicas}"
        return s


def parse_layout(text: str) -> Layout:
    m = _LAYOUT_RE.match(text.strip())
    if not m:
        raise ValueError(f"bad layout {text!r}; expected tp<T>[pp<P>][x<R>]")
    return Layout(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))


def default_layout(num_gpus: int) -> Layout:
    """Default (optimization 7 off): tensor parallel across all GPUs."""
    return Layout(num_gpus, 1, 1)


def vllm_executable() -> str:
    cand = Path(sys.executable).parent / "vllm"
    if cand.exists():
        return str(cand)
    return shutil.which("vllm") or "vllm"


def _tuned_value(param: str, point_id: str | None, env: EnvCfg, exp: ExperimentCfg,
                 tuned: dict[str, Any], overrides: dict[str, Any]) -> tuple[Any, str]:
    if param in overrides:
        return overrides[param], "override"
    if point_id is not None:
        val = tuned.get(env.name, {}).get(point_id, {}).get(param)
        if val is not None:
            return val, "tuned"
    return getattr(exp.placeholders, param), "placeholder"


def speculative_json(env: EnvCfg) -> str:
    spec = env.speculative.model_dump()
    if spec["method"] == "eagle3":
        if not env.models.eagle3_draft:
            raise ValueError("speculative.method=eagle3 but models.eagle3_draft is not set")
        spec.setdefault("model", env.models.eagle3_draft)
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def profiler_json(trace_dir: str, active_iterations: int) -> str:
    """vLLM's built-in torch profiler: records `active_iterations` engine steps after POST /start_profile.
    Kernels + CPU ops, no stacks or shapes (keeps traces small). Diagnostic passes only, never measured runs."""
    return json.dumps({"profiler": "torch", "torch_profiler_dir": trace_dir, "torch_profiler_with_stack": False,
                       "torch_profiler_record_shapes": False, "torch_profiler_use_gzip": True,
                       "active_iterations": active_iterations}, sort_keys=True, separators=(",", ":"))


def engine_flags(mask: int, env: EnvCfg, exp: ExperimentCfg, *, point_id: str | None,
                 tuned: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None,
                 profiler: tuple[str, int] | None = None,
                 ) -> tuple[str, list[str], Layout, dict[str, dict[str, Any]]]:
    """Return (model, flags, layout, tuned-value provenance) for a configuration.

    Flags exclude --host/--port and the per-replica parallel sizes, which `build_launch` adds.
    """
    tuned = tuned or {}
    overrides = overrides or {}
    provenance: dict[str, dict[str, Any]] = {}
    model = env.models.target
    flags = ["--served-model-name", env.served_model_name, "--dtype", env.dtype,
             "--max-model-len", str(env.max_model_len),
             "--gpu-memory-utilization", str(env.gpu_memory_utilization),
             "--seed", str(exp.seed)]

    # 1. Quantized weights
    if opt.is_on(mask, 1):
        if env.quantization_on == "fp8":
            flags += ["--quantization", "fp8"]
        else:
            if not env.models.awq_int4:
                raise ValueError("quantization_on=awq_int4 but models.awq_int4 is not set")
            model = env.models.awq_int4  # vLLM detects the AWQ format from the checkpoint

    # 2. KV-cache precision
    flags += ["--kv-cache-dtype", "fp8" if opt.is_on(mask, 2) else "auto"]

    # 3. Speculative decoding (off = no flag)
    if opt.is_on(mask, 3):
        flags += ["--speculative-config", speculative_json(env)]

    # 4. Prefix caching: always explicit
    flags.append("--enable-prefix-caching" if opt.is_on(mask, 4) else "--no-enable-prefix-caching")

    # 5. Tuned batching (off = engine default max_num_seqs, recorded from the start-up log)
    if opt.is_on(mask, 5):
        v, src = _tuned_value("max_num_seqs", point_id, env, exp, tuned, overrides)
        flags += ["--max-num-seqs", str(int(v))]
        provenance["max_num_seqs"] = {"value": int(v), "source": src}

    # 6. Prefill scheduling
    if opt.is_on(mask, 6):
        v, src = _tuned_value("max_num_batched_tokens", point_id, env, exp, tuned, overrides)
        flags += ["--enable-chunked-prefill", "--max-num-batched-tokens", str(int(v))]
        provenance["max_num_batched_tokens"] = {"value": int(v), "source": src}
    else:
        off = exp.chunked_prefill_off
        if off.disable_flag:
            flags.append("--no-enable-chunked-prefill")
        if off.budget_tokens is not None:
            flags += ["--max-num-batched-tokens", str(off.budget_tokens)]

    # 7. Parallelism
    if opt.is_on(mask, 7):
        if env.num_gpus < 2:
            raise ValueError("optimization 7 (parallelism) needs >= 2 GPUs")
        v, src = _tuned_value("parallel_layout", point_id, env, exp, tuned, overrides)
        layout = parse_layout(str(v))
        provenance["parallel_layout"] = {"value": layout.name, "source": src}
    else:
        layout = default_layout(env.num_gpus)
    if layout.gpus_per_replica * layout.replicas != env.num_gpus:
        raise ValueError(f"layout {layout.name} uses {layout.gpus_per_replica * layout.replicas} GPUs, "
                         f"env {env.name} has {env.num_gpus}")
    flags += ["--tensor-parallel-size", str(layout.tp)]
    if layout.pp > 1:
        flags += ["--pipeline-parallel-size", str(layout.pp)]

    flags += list(env.extra_args) + list(exp.engine_extra_flags)
    if profiler:
        flags += ["--profiler-config", profiler_json(*profiler)]
    return model, flags, layout, provenance


def build_launch(mask: int, env: EnvCfg, exp: ExperimentCfg, *, point_id: str | None,
                 tuned: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None,
                 profiler: tuple[str, int] | None = None) -> EngineLaunch:
    """`profiler` = (absolute trace dir, engine steps to record) adds vLLM's torch profiler."""
    model, flags, layout, provenance = engine_flags(mask, env, exp, point_id=point_id,
                                                    tuned=tuned, overrides=overrides, profiler=profiler)
    exe = vllm_executable()
    servers = []
    per = layout.gpus_per_replica
    for r in range(layout.replicas):
        gpus = tuple(env.gpu_ids[r * per:(r + 1) * per])
        port = env.base_port + r
        argv = (exe, "serve", model, *flags, "--host", "127.0.0.1", "--port", str(port))
        servers.append(ServerSpec(
            argv=argv,
            env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
                 # indices mean the same to CUDA as to NVML (and to Slurm's allocation)
                 "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                 # enables POST /reset_prefix_cache and GET /server_info
                 "VLLM_SERVER_DEV_MODE": "1",
                 # vLLM 0.31 rejects > 4 stop strings; HumanEval uses 5 (quality runs only)
                 "VLLM_MAX_STOP_STRINGS": "16"},
            port=port, gpu_ids=gpus))
    key_src = json.dumps({"model": model, "flags": flags, "layout": layout.name})
    return EngineLaunch(
        mask=mask, layout=layout.name, servers=tuple(servers), flags=(model, *flags),
        flags_key=hashlib.sha256(key_src.encode()).hexdigest()[:16],
        served_model_name=env.served_model_name, tuned=provenance)


def missing_flags(help_text: str, required: tuple[str, ...] = REQUIRED_FLAGS) -> list[str]:
    """Flags used here that the installed `vllm serve --help=all` does not mention."""
    return [f for f in required if not re.search(re.escape(f) + r"(?=[\s,=\[]|$)", help_text)]


def offline_engine_args(mask: int, env: EnvCfg, exp: ExperimentCfg) -> tuple[str, dict[str, Any]]:
    """(model, kwargs) for lm-eval's offline `vllm` backend, mirroring the numeric flags above.

    Only used for loglikelihood tasks, which speculative decoding does not affect, so bit 3 is ignored.
    """
    model = env.models.target
    kw: dict[str, Any] = {"dtype": env.dtype, "max_model_len": env.max_model_len,
                          "gpu_memory_utilization": env.gpu_memory_utilization, "seed": exp.seed,
                          "kv_cache_dtype": "fp8" if opt.is_on(mask, 2) else "auto",
                          "enable_prefix_caching": False, "tensor_parallel_size": 1}
    if opt.is_on(mask, 1):
        if env.quantization_on == "fp8":
            kw["quantization"] = "fp8"
        else:
            model = env.models.awq_int4 or model
    return model, kw
