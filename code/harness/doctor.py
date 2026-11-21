"""Environment checks: GPU, compute capability, CUDA, driver, versions, FP8, free VRAM, flags, models."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from harness.config import REPO_ROOT, EnvCfg
from harness.engine.vllm_flags import missing_flags, vllm_executable
from harness.telemetry import Nvml

PINNED = {"vllm": "0.31.0+cu129", "torch": "2.13.0+cu129"}


@dataclass
class Check:
    name: str
    status: str      # PASS | WARN | FAIL | INFO | SKIP
    detail: str


def vllm_help_text(timeout: float = 300.0) -> str:
    proc = subprocess.run([vllm_executable(), "serve", "--help=all"], capture_output=True, text=True,
                          timeout=timeout)
    return proc.stdout + proc.stderr


def check_flags() -> Check:
    try:
        text = vllm_help_text()
    except Exception as e:  # noqa: BLE001
        return Check("vllm flags", "FAIL", f"could not run `vllm serve --help=all`: {e}")
    if "--tensor-parallel-size" not in text:
        return Check("vllm flags", "FAIL", "`vllm serve --help=all` failed:\n" + text[-1500:])
    miss = missing_flags(text)
    if miss:
        return Check("vllm flags", "FAIL", f"flags used in harness/engine/vllm_flags.py missing from this vLLM: {miss}")
    eagle = "eagle3" in text
    return Check("vllm flags", "PASS", "all flags in vllm_flags.py present"
                 + ("" if eagle else " (WARN: 'eagle3' not mentioned in help text)"))


def _owner(pid: int) -> str:
    try:
        import psutil
        p = psutil.Process(pid)
        return f"{p.username()}:{p.name()}"
    except Exception:  # noqa: BLE001
        return "?"


def gpu_checks(env: EnvCfg) -> list[Check]:
    out: list[Check] = []
    nv = Nvml()
    if not nv.ok:
        return [Check("nvml", "FAIL", "NVML not available (no NVIDIA driver?)")]
    out.append(Check("driver", "INFO", f"NVIDIA driver {nv.driver()}"))
    n = nv.count()
    out.append(Check("gpu count", "PASS" if n >= env.num_gpus else "FAIL",
                     f"{n} GPU(s) visible, env '{env.name}' expects {env.num_gpus}"))
    fp8_ok = True
    for g in env.gpu_ids:
        if g >= n:
            out.append(Check(f"gpu {g}", "FAIL", "not present"))
            continue
        d = nv.device_info(g)
        major, minor = (int(x) for x in d["compute_capability"].split("."))
        fp8 = (major, minor) >= (8, 9)
        fp8_ok &= fp8
        out.append(Check(f"gpu {g}", "INFO",
                         f"{d['name']}, compute capability {d['compute_capability']}, "
                         f"{d['mem_total_mb'] / 1024:.1f} GiB, FP8 {'supported' if fp8 else 'NOT supported'}"))
        free_gb = d["mem_free_mb"] / 1024
        procs = nv.processes(g)
        own = ", ".join(f"pid {pid} ({_owner(pid)}) {mb / 1024:.1f} GiB" for pid, mb in procs) or "none"
        out.append(Check(f"gpu {g} free VRAM", "PASS" if free_gb >= env.min_free_vram_gb else "FAIL",
                         f"{free_gb:.1f} GiB free (need {env.min_free_vram_gb}); processes: {own}"))
    out.append(Check("fp8 support", "PASS" if fp8_ok else "FAIL", "compute capability >= 8.9 on all env GPUs"
                     if fp8_ok else "FP8 W8A8 / FP8 KV need compute capability >= 8.9"))
    if env.num_gpus > 1 and shutil.which("nvidia-smi"):
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
        nvlink = "NV" in "".join(l.split("\t", 1)[-1] for l in topo.splitlines()[1:env.num_gpus + 1])
        out.append(Check("interconnect", "INFO", ("NVLink present" if nvlink else "no NVLink (PCIe)")
                         + "\n" + topo.strip()))
    return out


def gpu_mapping_check(env: EnvCfg) -> Check:
    """Each env GPU index must name the same physical GPU for CUDA (servers, CUDA_DEVICE_ORDER=PCI_BUS_ID)
    and for NVML (telemetry, teardown, free-VRAM check). Matters under Slurm, where the allocated GPUs
    come from CUDA_VISIBLE_DEVICES."""
    try:
        import pynvml as nv
        nv.nvmlInit()
        nvml_uuid = {g: str(nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(g))).removeprefix("GPU-")
                     for g in env.gpu_ids}
    except Exception as e:  # noqa: BLE001
        return Check("gpu mapping", "WARN", f"could not read NVML UUIDs: {e}")
    code = ("import torch, json; print(json.dumps([str(torch.cuda.get_device_properties(i).uuid) "
            "for i in range(torch.cuda.device_count())]))")
    bad = []
    for g in env.gpu_ids:
        child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=child_env,
                             timeout=120)
        try:
            import json
            cuda = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return Check("gpu mapping", "WARN", f"torch could not list GPU {g}: {out.stderr[-500:]}")
        if not cuda or cuda[0] != nvml_uuid[g]:
            bad.append(f"index {g}: CUDA sees {cuda[:1]}, NVML sees {nvml_uuid[g]}")
    if bad:
        return Check("gpu mapping", "FAIL", "CUDA and NVML disagree on GPU indices: " + "; ".join(bad))
    return Check("gpu mapping", "PASS", f"GPU indices {env.gpu_ids} match between CUDA and NVML"
                 + (f" (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
                    if os.environ.get("CUDA_VISIBLE_DEVICES") else ""))


def torch_checks() -> list[Check]:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return [Check("torch", "FAIL", f"import failed: {e}")]
    out = [Check("torch", "INFO", f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
                 f"arch list {torch.cuda.get_arch_list()}")]
    if not torch.cuda.is_available():
        out.append(Check("torch cuda", "FAIL", "torch.cuda.is_available() is False"))
        return out
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i)
        archs = torch.cuda.get_arch_list()
        native = f"sm_{cc[0]}{cc[1]}" in archs
        # sm_89 runs sm_86 cubins (same major); sm_120 needs sm_120 kernels.
        compat = native or any(a.startswith(f"sm_{cc[0]}") for a in archs)
        out.append(Check(f"torch gpu {i}", "PASS" if compat else "FAIL",
                         f"{torch.cuda.get_device_name(i)} cc {cc[0]}.{cc[1]}: "
                         + ("native kernels" if native else "binary-compatible kernels" if compat
                            else "no compatible kernels in this torch build")))
    return out


def version_checks() -> list[Check]:
    out = [Check("python", "PASS" if sys.version_info[:2] == (3, 11) else "WARN", platform.python_version())]
    for pkg, want in PINNED.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out.append(Check(pkg, "FAIL", "not installed (run scripts/setup_env.sh)"))
            continue
        out.append(Check(pkg, "PASS" if have == want else "FAIL", f"{have} (pinned {want})"))
    for pkg in ("lm-eval", "nvidia-ml-py", "transformers"):
        try:
            out.append(Check(pkg, "INFO", metadata.version(pkg)))
        except metadata.PackageNotFoundError:
            out.append(Check(pkg, "FAIL", "not installed"))
    return out


def model_checks(env: EnvCfg) -> list[Check]:
    out = [Check("HF_TOKEN", "INFO", "set" if os.environ.get("HF_TOKEN") else
                 "not set (fine for the ungated models configured; needed for gated repos)")]
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    except Exception as e:  # noqa: BLE001
        return out + [Check("huggingface_hub", "FAIL", str(e))]
    roles: list[tuple[str, str | None, bool]] = [
        ("target", env.models.target, True),
        ("tokenizer", env.models.tokenizer, False),
        ("awq_int4", env.models.awq_int4, env.quantization_on == "awq_int4"),
        ("eagle3_draft", env.models.eagle3_draft, env.speculative.method == "eagle3"),
        ("fallback", env.models.fallback, False),
    ]
    for role, repo, required in roles:
        if not repo:
            continue
        try:
            info = api.model_info(repo)
            gated = getattr(info, "gated", False)
            out.append(Check(f"model {role}", "PASS", f"{repo} exists (gated={gated})"))
        except Exception as e:  # noqa: BLE001
            from huggingface_hub import try_to_load_from_cache
            local = Path(repo).exists()
            cached = isinstance(try_to_load_from_cache(repo, "config.json"), str)   # offline compute nodes
            status = "PASS" if local or cached else ("FAIL" if required else "WARN")
            where = "local path" if local else "in HF cache (hub unreachable)" if cached else type(e).__name__
            out.append(Check(f"model {role}", status, f"{repo}: {where}"))
    try:
        cfg = api.hf_hub_download(env.models.target, "config.json")
        import json
        c = json.loads(Path(cfg).read_text())
        mi = env.model_info
        got = (c.get("num_hidden_layers"), c.get("num_key_value_heads"),
               c.get("head_dim") or c.get("hidden_size", 0) // max(c.get("num_attention_heads", 1), 1))
        want = (mi.num_layers, mi.num_kv_heads, mi.head_dim)
        out.append(Check("model_info", "PASS" if got == want else "WARN",
                         f"config.json layers/kv_heads/head_dim = {got}; env model_info = {want}"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("model_info", "WARN", f"could not verify against config.json: {e}"))
    return out


def nvcc_check() -> Check:
    from harness.engine.vllm_server import toolchain_env
    env = toolchain_env()
    nvcc = shutil.which("nvcc", path=env.get("PATH", os.environ.get("PATH")))
    if not nvcc:
        return Check("nvcc", "FAIL", "no nvcc: FlashInfer JIT (FP8 KV, sampler) will fail; run scripts/setup_env.sh")
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    return Check("nvcc", "PASS", f"{nvcc} ({out[-2] if len(out) > 1 else '?'})"
                 + (f"; CUDA_HOME={env['CUDA_HOME']}" if env else ""))


def disk_check(path: Path) -> Check:
    free = shutil.disk_usage(path).free / 2**30
    return Check("disk", "PASS" if free > 60 else "WARN", f"{free:.0f} GiB free at {path}")


def run_doctor(env: EnvCfg, dry_run: bool = False) -> tuple[bool, list[Check]]:
    checks = version_checks()
    if dry_run:
        checks.append(Check("gpu", "SKIP", "--dry-run: GPU, flag and model checks skipped"))
    else:
        checks += gpu_checks(env)
        checks.append(gpu_mapping_check(env))
        checks += torch_checks()
        checks.append(check_flags())
        checks.append(nvcc_check())
        checks += model_checks(env)
    checks.append(disk_check(REPO_ROOT))
    ok = not any(c.status == "FAIL" for c in checks)
    return ok, checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        first, *rest = c.detail.splitlines() or [""]
        lines.append(f"[{c.status:4}] {c.name:<22} {first}")
        lines += [" " * 30 + r for r in rest]
    return "\n".join(lines)


def preflight_flags() -> None:
    """Start-up version check: every flag in vllm_flags.py must exist in the installed vLLM."""
    c = check_flags()
    if c.status == "FAIL":
        raise SystemExit(f"vLLM flag check failed: {c.detail}")


def free_vram_ok(env: EnvCfg) -> tuple[bool, str]:
    checks = [c for c in gpu_checks(env) if c.name.endswith("free VRAM")]
    bad = [c for c in checks if c.status == "FAIL"]
    return not bad, "\n".join(c.detail for c in bad)


def summary_dict(checks: list[Check]) -> dict[str, Any]:
    return {c.name: {"status": c.status, "detail": c.detail} for c in checks}
