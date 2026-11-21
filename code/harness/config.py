"""Load and validate YAML configuration: experiment + env + profile (+ tuned values)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegimeLevels(_Strict):
    low: float
    mid: float
    high: float


class Regimes(_Strict):
    context: RegimeLevels
    concurrency: RegimeLevels
    reuse: RegimeLevels


class WorkloadCfg(_Strict):
    source: Literal["synthetic"] = "synthetic"
    output_tokens: int
    prefix_pool_size: int = 4
    prefix_fraction: float = 0.75
    trace_requests: int


class Gates(_Strict):
    p99_tpot_ms: float
    p99_ttft_ms: float
    max_accuracy_drop: float


class RunCfg(_Strict):
    reps: int
    warmup_s: float
    window_s: float
    startup_timeout_s: float = 1200.0
    request_timeout_s: float = 900.0
    gpu_release_timeout_s: float = 120.0
    telemetry_interval_s: float = 1.0
    reset_prefix_cache_between_reps: bool = True
    # GPUs per measured cell. null = all env GPUs (spec Default: TP across all GPUs, one cell at a time).
    # A smaller value splits the machine into num_gpus / gpus_per_cell slots that run cells concurrently.
    gpus_per_cell: int | None = None


class ChunkedPrefillOff(_Strict):
    disable_flag: bool = True
    budget_tokens: int | None = 16384


class Placeholders(_Strict):
    max_num_seqs: int
    max_num_batched_tokens: int
    parallel_layout: str


class Sweeps(_Strict):
    max_num_seqs: list[int]
    max_num_batched_tokens: list[int]
    parallel_layout: list[str]
    reps: int = 1


class QualityCfg(_Strict):
    tasks: list[str]
    mmlu_subtasks: list[str]
    metrics: dict[str, str]
    limit: int | None = None
    task_limits: dict[str, int] = Field(default_factory=dict)   # per-task override of `limit` (mmlu = per subtask)
    num_concurrent: int = 16
    agreement_prompts: str
    agreement_max_tokens: int = 128
    # Numeric settings to evaluate (e.g. ["000", "100"]); null = every one the plan needs.
    settings: list[str] | None = None
    log_samples: bool = False      # per-question outputs (lm-eval --log_samples), for paired comparisons
    mmlu_backend: Literal["auto", "server", "offline"] = "auto"


class AnalysisCfg(_Strict):
    kv_usage_threshold: float = 0.90
    bandwidth_fraction_threshold: float = 0.70
    compute_fraction_threshold: float = 0.60
    no_benefit_gain: float = 0.02


class ExperimentCfg(_Strict):
    seed: int
    engine_extra_flags: list[str] = Field(default_factory=list)   # appended to every vLLM server command
    regimes: Regimes
    workload: WorkloadCfg
    gates: Gates
    run: RunCfg
    chunked_prefill_off: ChunkedPrefillOff
    placeholders: Placeholders
    sweeps: Sweeps
    quality: QualityCfg
    analysis: AnalysisCfg


class Models(_Strict):
    target: str
    tokenizer: str | None = None
    awq_int4: str | None = None
    eagle3_draft: str | None = None
    fallback: str | None = None


class ModelInfo(_Strict):
    params_b: float
    num_layers: int
    num_kv_heads: int
    head_dim: int


class SpeculativeCfg(BaseModel):
    """Extra keys are passed straight into vLLM's --speculative-config JSON."""
    model_config = ConfigDict(extra="allow")
    method: Literal["ngram", "eagle3"]
    num_speculative_tokens: int


class Peaks(_Strict):
    mem_bandwidth_gbps: float | None = None
    bf16_tflops: float | None = None
    fp8_tflops: float | None = None


def visible_gpu_ids(num_gpus: int) -> list[int]:
    """GPU indices for `gpu_ids: auto`: the ones the scheduler allocated (CUDA_VISIBLE_DEVICES), else 0..n-1."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return list(range(num_gpus))
    parts = [x.strip() for x in cvd.split(",") if x.strip()]
    if not all(x.isdigit() for x in parts):
        raise ValueError(f"gpu_ids: auto needs integer CUDA_VISIBLE_DEVICES, got {cvd!r}; set gpu_ids explicitly")
    if len(parts) < num_gpus:
        raise ValueError(f"env needs {num_gpus} GPUs but CUDA_VISIBLE_DEVICES={cvd!r} has {len(parts)}")
    return [int(x) for x in parts[:num_gpus]]


class EnvCfg(_Strict):
    name: str
    num_gpus: int
    gpu_ids: list[int]
    models: Models
    served_model_name: str = "target"
    model_info: ModelInfo
    quantization_on: Literal["fp8", "awq_int4"] = "fp8"
    speculative: SpeculativeCfg
    dtype: str = "bfloat16"
    max_model_len: int
    gpu_memory_utilization: float = 0.90
    base_port: int = 8100
    min_free_vram_gb: float = 0.0
    extra_args: list[str] = Field(default_factory=list)
    peaks: Peaks = Field(default_factory=Peaks)

    @model_validator(mode="before")
    @classmethod
    def _runtime(cls, data: Any) -> Any:
        """Values only known at run time: Slurm-allocated GPUs (`gpu_ids: auto`) and a per-job port base."""
        if isinstance(data, dict):
            data = dict(data)
            if data.get("gpu_ids") == "auto":
                data["gpu_ids"] = visible_gpu_ids(int(data["num_gpus"]))
            if os.environ.get("HARNESS_BASE_PORT"):
                data["base_port"] = int(os.environ["HARNESS_BASE_PORT"])
        return data

    @model_validator(mode="after")
    def _gpus(self) -> "EnvCfg":
        if len(self.gpu_ids) != self.num_gpus:
            raise ValueError(f"gpu_ids {self.gpu_ids} does not match num_gpus={self.num_gpus}")
        return self

    @property
    def tokenizer(self) -> str:
        return self.models.tokenizer or self.models.target


class FakeCfg(_Strict):
    seed: int = 7
    noise_sd: float = 0.02
    infeasible_combos: list[list[int]] = Field(default_factory=list)
    crash: dict[str, Any] | None = None
    quality_drop: dict[str, float] = Field(default_factory=dict)


class Investigation(_Strict):
    """A focused re-run of selected cells with deep telemetry (profile `investigate`)."""
    id: str
    why: str
    points: list[str] = Field(default_factory=list)          # existing point ids
    base: str | None = None                                   # with `vary`: derive points from this one
    vary: dict[Literal["context", "concurrency", "reuse"], list[float]] = Field(default_factory=dict)
    configs: list[str]                                        # "default", "spec+prefix", or 7-bit strings
    reps: int | None = None                                   # default: run.reps
    profile: list[str] | Literal["all"] = Field(default_factory=list)   # configs that get a profiler pass
    exclusive: bool = False                                   # run with the other GPU slots idle

    @model_validator(mode="after")
    def _where(self) -> "Investigation":
        if bool(self.points) == bool(self.base):
            raise ValueError(f"investigation {self.id}: give either `points` or `base` (+ `vary`)")
        return self


class ProfilerPass(_Strict):
    warmup_s: float = 30.0
    window_s: float = 15.0           # load kept up while profiling; throughput in this pass is not a result
    active_iterations: int = 64      # engine steps recorded by vLLM's torch profiler
    flush_timeout_s: float = 300.0


class ProfileCfg(_Strict):
    name: str
    engine: Literal["vllm", "fake"]
    design: Literal["full", "smoke"]
    points: Literal["all"] | list[str] = "all"
    envs: list[str] | None = None
    # Optimization keys to vary (e.g. [quant, kvq, spec, prefix, batch]); null = all the hardware allows.
    optimizations: list[str] | None = None
    tuned_file: str = "tuned.yaml"   # under configs/; profiles with a different cell size keep their own
    investigations: list[Investigation] = Field(default_factory=list)
    profiler: ProfilerPass = Field(default_factory=ProfilerPass)
    overrides: dict[str, Any] = Field(default_factory=dict)
    fake: FakeCfg = Field(default_factory=FakeCfg)


class Settings(BaseModel):
    """Everything one command needs: merged experiment config, env, profile and paths."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    experiment: ExperimentCfg
    env: EnvCfg
    profile: ProfileCfg
    dry_run: bool
    results_dir: Path
    tuned_path: Path

    @property
    def use_fake(self) -> bool:
        return self.dry_run or self.profile.engine == "fake"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_profile(name: str, config_dir: Path = CONFIG_DIR) -> ProfileCfg:
    path = config_dir / "profiles" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown profile {name!r} (no {path})")
    return ProfileCfg.model_validate(_read_yaml(path))


def load_env(name: str, config_dir: Path = CONFIG_DIR) -> EnvCfg:
    path = config_dir / "env" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown env {name!r} (no {path})")
    return EnvCfg.model_validate(_read_yaml(path))


def load_experiment(profile: ProfileCfg, config_dir: Path = CONFIG_DIR) -> ExperimentCfg:
    raw = deep_merge(_read_yaml(config_dir / "experiment.yaml"), profile.overrides)
    return ExperimentCfg.model_validate(raw)


def results_dir_for(profile: ProfileCfg, dry_run: bool, root: Path = REPO_ROOT) -> Path:
    # A real profile forced onto FakeEngine with --dry-run must never mix with real results.
    suffix = "_dryrun" if dry_run and profile.engine != "fake" else ""
    return root / "results" / f"{profile.name}{suffix}"


def load_settings(profile: str, env: str, dry_run: bool = False, config_dir: Path = CONFIG_DIR,
                  results_root: Path = REPO_ROOT) -> Settings:
    prof = load_profile(profile, config_dir)
    rdir = results_dir_for(prof, dry_run, results_root)
    fake = dry_run or prof.engine == "fake"
    return Settings(
        experiment=load_experiment(prof, config_dir),
        env=load_env(env, config_dir),
        profile=prof,
        dry_run=dry_run,
        results_dir=rdir,
        # Fake sweeps must not overwrite the real tuned values.
        tuned_path=(rdir / "tuned.yaml") if fake else (config_dir / prof.tuned_file),
    )


def cell_gpus(env: EnvCfg, run: RunCfg) -> int:
    n = run.gpus_per_cell or env.num_gpus
    if n < 1 or env.num_gpus % n:
        raise ValueError(f"gpus_per_cell={n} does not divide the {env.num_gpus} GPUs of env {env.name}")
    return n


def active_optimizations(env: EnvCfg, run: RunCfg, profile: ProfileCfg) -> list[int]:
    """Optimization numbers varied in the design: what one cell's GPUs allow, limited to the profile's list."""
    from harness import optimizations as opt
    act = opt.active_numbers(cell_gpus(env, run))
    if profile.optimizations is None:
        return act
    unknown = set(profile.optimizations) - set(opt.BY_KEY)
    if unknown:
        raise ValueError(f"profile {profile.name}: unknown optimizations {sorted(unknown)}; known: {sorted(opt.BY_KEY)}")
    want = {opt.BY_KEY[k].number for k in profile.optimizations}
    return [n for n in act if n in want]


def slot_envs(env: EnvCfg, run: RunCfg) -> list[EnvCfg]:
    """One env per concurrent slot: a disjoint slice of the GPUs and its own port range."""
    n = cell_gpus(env, run)
    return [env.model_copy(update={"num_gpus": n, "gpu_ids": env.gpu_ids[i * n:(i + 1) * n],
                                   "base_port": env.base_port + 10 * i})
            for i in range(env.num_gpus // n)]


def envs_for(profile: str, env: str | None, config_dir: Path = CONFIG_DIR) -> list[str]:
    """Envs a command should cover: --env if given, else the profile's list."""
    if env:
        return [env]
    prof = load_profile(profile, config_dir)
    if prof.envs:
        return list(prof.envs)
    raise SystemExit(f"--env is required for profile {profile!r}")


def load_tuned(path: Path) -> dict[str, Any]:
    return _read_yaml(path) if path.exists() else {}


def save_tuned(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Written by `python -m harness sweep`. Keyed by env -> point -> parameter.\n")
        yaml.safe_dump(data, f, sort_keys=True)
