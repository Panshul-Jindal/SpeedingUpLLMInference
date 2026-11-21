"""Quality checks: lm-eval against the running server, once per numeric setting (bits 1-3), plus agreement.

Generative tasks (gsm8k, humaneval) go through the server with `local-completions`, so speculative decoding
is exercised. MMLU (loglikelihood) uses the server too; if that fails it falls back to lm-eval's offline
`vllm` backend with the same numeric engine arguments.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harness import optimizations as opt
from harness.config import REPO_ROOT, Settings
from harness.engine.base import Endpoint, EngineStartError
from harness.engine.vllm_flags import build_launch, offline_engine_args
from harness.load_driver import single_request
from harness.results import ResultStore, versions

log = logging.getLogger(__name__)


def quality_key(settings: Settings, numeric_mask: int) -> str:
    return f"{settings.profile.name}__{settings.env.name}__q{opt.numeric_label(numeric_mask)}"


def numeric_settings(masks: list[int]) -> list[int]:
    """Distinct numeric settings among the planned configurations; Default (000) first."""
    return sorted({opt.numeric_setting(m) for m in masks} | {0}, key=lambda m: (bin(m).count("1"), m))


def quality_masks(settings: Settings, masks: list[int]) -> list[int]:
    """Restrict to quality.settings (numeric labels such as "100") when the profile lists them."""
    want = settings.experiment.quality.settings
    return masks if want is None else [m for m in masks if opt.numeric_label(m) in set(want)]


def load_agreement_prompts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def _lm_eval_tasks(settings: Settings, which: str) -> list[str]:
    q = settings.experiment.quality
    if which == "generative":
        return [t for t in q.tasks if t != "mmlu"]
    return list(q.mmlu_subtasks) if "mmlu" in q.tasks else []


def _newest_results_json(out_dir: Path) -> dict[str, Any]:
    files = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"lm-eval wrote no results under {out_dir}")
    return json.loads(files[-1].read_text())


def run_lm_eval(model_type: str, model_args: str, tasks: list[str], out_dir: Path, limit: int | None,
                seed: int, log_path: Path, extra_env: dict[str, str] | None = None,
                log_samples: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "lm_eval", "run", "--model", model_type, "--model_args", model_args,
           "--tasks", ",".join(tasks), "--output_path", str(out_dir), "--seed", str(seed),
           "--confirm_run_unsafe_code", "--batch_size", "auto" if model_type == "vllm" else "1"]
    if limit:
        cmd += ["--limit", str(limit)]
    if log_samples:
        cmd.append("--log_samples")     # samples_<task>_*.jsonl: per-question outputs and scores
    from harness.engine.vllm_server import toolchain_env
    env = {**os.environ, **toolchain_env(), "HF_ALLOW_CODE_EVAL": "1", **(extra_env or {})}
    with open(log_path, "ab") as fh:
        fh.write(("$ " + " ".join(cmd) + "\n").encode())
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-15:]
        raise RuntimeError(f"lm-eval exited {proc.returncode}: " + "\n".join(tail)[-2000:])
    return _newest_results_json(out_dir)


def extract_scores(results: dict[str, Any], settings: Settings, which: str) -> dict[str, float]:
    """Accuracy in points (0-100) per study task."""
    q = settings.experiment.quality
    res = results.get("results", {})
    scores: dict[str, float] = {}
    if which == "generative":
        for t in _lm_eval_tasks(settings, "generative"):
            metric = q.metrics.get(t)
            if t in res and metric in res[t]:
                scores[t] = 100.0 * float(res[t][metric])
    else:
        metric = q.metrics.get("mmlu", "acc,none")
        accs = [float(res[s][metric]) for s in q.mmlu_subtasks if s in res and metric in res[s]]
        if accs:
            scores["mmlu"] = 100.0 * sum(accs) / len(accs)
    return scores


async def _agreement_outputs(endpoint: Endpoint, model: str, prompts: list[dict[str, Any]], tokenizer: Any,
                             max_tokens: int, concurrency: int = 16) -> dict[str, str]:
    sem = asyncio.Semaphore(concurrency)

    async def one(p: dict[str, Any]) -> tuple[str, str]:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                            add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):   # transformers 5 returns a BatchEncoding
            ids = ids["input_ids"]
        async with sem:
            resp = await single_request(endpoint, model, list(ids), max_tokens)
        return p["id"], resp["choices"][0]["text"]

    pairs = await asyncio.gather(*[one(p) for p in prompts])
    return dict(pairs)


def agreement_score(outputs: dict[str, str], reference: dict[str, str]) -> float | None:
    common = [k for k in reference if k in outputs]
    if not common:
        return None
    return sum(outputs[k] == reference[k] for k in common) / len(common)


def run_quality(settings: Settings, masks: list[int], engine: Any, force: bool = False) -> list[dict[str, Any]]:
    """Run quality for every numeric setting needed by `masks`. Returns the rows written.

    With several GPU slots (run.gpus_per_cell) the settings run concurrently, one per slot; agreement with
    Default is then filled in afterwards for settings that finished before Default did.
    """
    from harness.orchestrator import run_on_slots, slot_settings
    store = ResultStore(settings.results_dir / "quality.jsonl")
    done = set() if force else store.done_keys(retry_failed=True)
    todo = []
    for nmask in quality_masks(settings, numeric_settings(masks)):
        if quality_key(settings, nmask) in done:
            log.info("[quality] %s already done, skipping", quality_key(settings, nmask))
        else:
            todo.append(nmask)
    slots = slot_settings(settings)
    if len(slots) == 1:
        rows = [r for nmask in todo for r in _quality_task(slots[0], engine, nmask)]
    else:
        rows = run_on_slots(settings, todo, _quality_task)
    filled = {r["run_key"]: r for r in _fill_agreement(settings, store)}
    return [filled.get(r["run_key"], r) for r in rows]


def _quality_task(settings: Settings, engine: Any, nmask: int) -> list[dict[str, Any]]:
    from harness.orchestrator import QUALITY_RESERVE_S, fits_before_deadline
    if not settings.use_fake and not fits_before_deadline(QUALITY_RESERVE_S, f"quality {opt.numeric_label(nmask)}"):
        return []
    store = ResultStore(settings.results_dir / "quality.jsonl")
    q = settings.experiment.quality
    key = quality_key(settings, nmask)
    label = opt.numeric_label(nmask)
    t0 = time.monotonic()
    row: dict[str, Any] = {"run_key": key, "profile": settings.profile.name, "env": settings.env.name,
                           "numeric_bits": label, "config_bits": opt.to_bits(nmask),
                           "gpu_ids": list(settings.env.gpu_ids), **versions()}
    try:
        if settings.use_fake:
            from harness.engine.fake import fake_quality
            row.update(status="ok", scores=fake_quality(label, settings.profile.fake, list(q.tasks)),
                       agreement=1.0 if nmask == 0 else 0.9, mmlu_backend="fake")
        else:
            row.update(_real_quality(settings, nmask, engine, settings.results_dir / "quality" / label))
    except EngineStartError as e:
        row.update(status=e.status, error=e.message, error_kind=e.kind)
    except Exception as e:  # noqa: BLE001 - failures are results
        log.exception("[quality] %s failed", key)
        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:3000])
    row["duration_s"] = round(time.monotonic() - t0, 1)
    store.append(row)
    log.info("[quality] %s status=%s scores=%s agreement=%s (%.0fs)", label, row["status"],
             row.get("scores"), row.get("agreement"), row["duration_s"])
    return [row]


def _fill_agreement(settings: Settings, store: ResultStore) -> list[dict[str, Any]]:
    """Agreement for rows written before Default's reference outputs existed (concurrent slots)."""
    qdir = settings.results_dir / "quality"
    ref_path = qdir / "000" / "agreement_outputs.json"
    if not ref_path.exists():
        return []
    ref = json.loads(ref_path.read_text())
    updated = []
    for r in store.latest().values():
        out = qdir / str(r.get("numeric_bits")) / "agreement_outputs.json"
        if r.get("env") != settings.env.name or r.get("agreement") is not None or not out.exists():
            continue
        r = {k: v for k, v in r.items() if k != "written_at"}
        r["agreement"] = agreement_score(json.loads(out.read_text()), ref)
        store.append(r)
        updated.append(r)
        log.info("[quality] %s agreement filled in: %s", r["numeric_bits"], r["agreement"])
    return updated


def _real_quality(settings: Settings, nmask: int, engine: Any, out: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    env, exp, q = settings.env, settings.experiment, settings.experiment.quality
    out.mkdir(parents=True, exist_ok=True)
    launch = build_launch(nmask, env, exp, point_id=None)
    tok = AutoTokenizer.from_pretrained(env.tokenizer)
    result: dict[str, Any] = {"status": "ok", "flags": list(launch.flags), "errors": {}}
    scores: dict[str, float] = {}
    endpoints = engine.start(launch, out / "server")
    try:
        ep = endpoints[0]
        base = f"model={env.served_model_name},base_url={ep.completions_url},tokenizer={env.tokenizer}," \
               f"num_concurrent={q.num_concurrent},max_retries=3,tokenized_requests=False," \
               f"max_length={env.max_model_len},tokenizer_backend=huggingface"
        # Agreement with Default outputs on the fixed prompt set.
        prompts = load_agreement_prompts(REPO_ROOT / q.agreement_prompts, q.limit)
        outputs = asyncio.run(_agreement_outputs(ep, env.served_model_name, prompts, tok, q.agreement_max_tokens))
        (out / "agreement_outputs.json").write_text(json.dumps(outputs, indent=1))
        ref_path = out.parent / "000" / "agreement_outputs.json"
        result["agreement"] = agreement_score(outputs, json.loads(ref_path.read_text())) \
            if ref_path.exists() else None

        for task in _lm_eval_tasks(settings, "generative"):   # one call per task: failures stay isolated
            try:
                res = run_lm_eval("local-completions", base, [task], out / f"lm_eval_{task}",
                                  q.task_limits.get(task, q.limit), exp.seed,
                                  out / "lm_eval.log", log_samples=q.log_samples)
                scores.update(extract_scores(res, settings, "generative"))
            except Exception as e:  # noqa: BLE001
                result["errors"][task] = str(e)[:2000]
        mmlu = _lm_eval_tasks(settings, "mmlu")
        mmlu_done = False
        if mmlu and q.mmlu_backend in ("auto", "server"):
            try:
                res = run_lm_eval("local-completions", base, mmlu, out / "lm_eval_mmlu",
                                  q.task_limits.get("mmlu", q.limit), exp.seed,
                                  out / "lm_eval.log", log_samples=q.log_samples)
                scores.update(extract_scores(res, settings, "mmlu"))
                result["mmlu_backend"] = "server"
                mmlu_done = "mmlu" in scores
            except Exception as e:  # noqa: BLE001
                result["errors"]["mmlu_server"] = str(e)[:2000]
    finally:
        engine.stop()
    if mmlu and not mmlu_done and q.mmlu_backend in ("auto", "offline"):
        model, kw = offline_engine_args(nmask, env, exp)
        margs = ",".join([f"pretrained={model}"] + [f"{k}={v}" for k, v in kw.items()])
        try:
            res = run_lm_eval("vllm", margs, mmlu, out / "lm_eval_mmlu_offline",
                              q.task_limits.get("mmlu", q.limit), exp.seed, out / "lm_eval.log",
                              extra_env={"CUDA_VISIBLE_DEVICES": str(env.gpu_ids[0]),
                                         "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, log_samples=q.log_samples)
            scores.update(extract_scores(res, settings, "mmlu"))
            result["mmlu_backend"] = "offline-vllm"
        except Exception as e:  # noqa: BLE001
            result["errors"]["mmlu_offline"] = str(e)[:2000]
    result["scores"] = scores
    missing = [t for t in q.tasks if t not in scores]
    if missing:
        # the accuracy gate needs every task; partial scores are kept but the run is not "ok"
        result["status"] = "failed"
        result["missing_tasks"] = missing
    return result
