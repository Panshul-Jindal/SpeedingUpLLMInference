"""First-order binding-constraint label for a configuration, from its telemetry (thresholds in config)."""
from __future__ import annotations

import math
from typing import Any

from harness import optimizations as opt
from harness.config import AnalysisCfg, EnvCfg
from harness.engine.vllm_flags import default_layout, parse_layout

UNAVAILABLE = "unavailable"


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def kv_bytes_per_token(env: EnvCfg, mask: int) -> float:
    mi = env.model_info
    elem = 1 if opt.is_on(mask, 2) else 2
    return mi.num_layers * mi.num_kv_heads * mi.head_dim * 2 * elem


def weight_bytes(env: EnvCfg, mask: int) -> float:
    if opt.is_on(mask, 1):
        per = 1.0 if env.quantization_on == "fp8" else 0.5
    else:
        per = 2.0
    return env.model_info.params_b * 1e9 * per


def layout_for(env: EnvCfg, mask: int, layout_name: str | None) -> tuple[int, int, int]:
    if opt.is_on(mask, 7) and layout_name:
        lay = parse_layout(layout_name)
    else:
        lay = default_layout(env.num_gpus)
    return lay.tp, lay.pp, lay.replicas


def estimates(tel: dict[str, Any], env: EnvCfg, mask: int, layout_name: str | None,
              context: float, output_tokens: float) -> dict[str, Any]:
    """Achieved bandwidth and FLOP/s per GPU (first-order) and fractions of peak (if peaks are known)."""
    tp, pp, replicas = layout_for(env, mask, layout_name)
    out: dict[str, Any] = {"bw_gbps": None, "bw_frac": None, "flops_tflops": None, "flops_frac": None}
    thr = _num(tel.get("output_tok_s"))
    running = _num(tel.get("running_mean"))
    if thr and running and running > 0:
        # Spec: steps/s ~ output tok/s / mean running batch. With speculative decoding a step emits
        # (1 + accepted/drafts) tokens per sequence, so divide by that too (1 otherwise).
        tps = _num(tel.get("tokens_per_step")) or 1.0
        steps_per_s = thr / (running * tps)           # per replica (running is summed over replicas)
        run_per_replica = running / replicas
        avg_ctx = context + output_tokens / 2
        bytes_step_gpu = (weight_bytes(env, mask) + run_per_replica * avg_ctx * kv_bytes_per_token(env, mask)) \
            / (tp * pp)
        out["bw_gbps"] = steps_per_s * bytes_step_gpu / 1e9
        peak = env.peaks.mem_bandwidth_gbps
        out["bw_frac"] = out["bw_gbps"] / peak if peak else UNAVAILABLE
    if thr:
        req_s = _num(tel.get("req_s")) or 0.0
        hit = _num(tel.get("prefix_hit_rate")) or 0.0
        processed = thr + req_s * context * (1 - hit)
        flops_gpu = 2 * env.model_info.params_b * 1e9 * processed / env.num_gpus
        out["flops_tflops"] = flops_gpu / 1e12
        peak = env.peaks.fp8_tflops if opt.is_on(mask, 1) and env.quantization_on == "fp8" else env.peaks.bf16_tflops
        out["flops_frac"] = out["flops_tflops"] / peak if peak else UNAVAILABLE
    return out


def binding_constraint(tel: dict[str, Any], env: EnvCfg, mask: int, cfg: AnalysisCfg, layout_name: str | None,
                       context: float, output_tokens: float, max_num_seqs: float | None) -> dict[str, Any]:
    est = estimates(tel, env, mask, layout_name, context, output_tokens)
    kv = _num(tel.get("kv_usage_max"))
    pre = _num(tel.get("preemptions")) or 0.0
    running_max = _num(tel.get("running_max"))
    waiting = _num(tel.get("waiting_mean")) or 0.0
    notes = []
    label = None
    if kv is not None and kv > cfg.kv_usage_threshold or pre > 0:
        label = "kv_capacity"
    if label is None:
        if est["bw_frac"] == UNAVAILABLE:
            notes.append("bandwidth check unavailable (peaks.mem_bandwidth_gbps is null)")
        elif est["bw_frac"] is not None and est["bw_frac"] > cfg.bandwidth_fraction_threshold:
            label = "memory_bandwidth"
    if label is None:
        if est["flops_frac"] == UNAVAILABLE:
            notes.append("compute check unavailable (peak TFLOP/s is null)")
        elif est["flops_frac"] is not None and est["flops_frac"] > cfg.compute_fraction_threshold:
            label = "compute"
    if label is None and max_num_seqs and running_max is not None:
        replicas = layout_for(env, mask, layout_name)[2]
        if running_max >= max_num_seqs * replicas and waiting > 0 and (kv is None or kv < cfg.kv_usage_threshold):
            label = "scheduler_batch_limit"
    if label is None:
        label = "per_step_overhead"
    if kv is None:
        notes.append("kv usage metric missing")
    return {"binding_constraint": label, "kv_usage_max": kv, "preemptions": pre, "running_max": running_max,
            "waiting_mean": waiting, "max_num_seqs": max_num_seqs, **est, "notes": "; ".join(notes)}
