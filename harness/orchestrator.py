"""Walks a list of jobs (runs, sweep runs, start-up checks), grouping by engine flags to minimise restarts.

Failures are results: start-up failures become `infeasible`/`timeout` rows, measurement errors become
`failed` rows, and the walk continues. Only an environment fault (GPU memory not released) stops it.
"""
from __future__ import annotations

import asyncio
import logging
import math
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

from harness import optimizations as opt
from harness.config import Settings, active_optimizations, load_tuned, save_tuned, slot_envs
from harness.design import Cell, build_cells, build_runs, configs_for_point, full_factorial
from harness.engine.base import Endpoint, EngineLaunch, EngineStartError
from harness.engine.vllm_flags import build_launch
from harness.load_driver import post_all, reset_prefix_cache, run_closed_loop, single_request
from harness.regimes import CENTER, WorkloadPoint, all_points, select_points
from harness.results import ResultStore, run_dir, run_key, versions, write_run_artifacts
from harness.telemetry import NvmlSampler, PrometheusScraper
from harness.workload import FakeTokenizer, HFTokenizer, Trace, ensure_trace

log = logging.getLogger(__name__)


class EnvironmentFault(RuntimeError):
    """The machine is in a state where continuing would corrupt results (e.g. GPU memory not released)."""


# ---------------------------------------------------------------- deadline (time-limited batch jobs)

DEADLINE_ENV = "HARNESS_DEADLINE"     # unix time; set by scripts/slurm/harness.sbatch from the job's end time
STARTUP_RESERVE_S = 300.0             # engine start + teardown, worst case seen
QUALITY_RESERVE_S = 2700.0            # one numeric setting of the trimmed quality run


def time_left_s() -> float:
    d = os.environ.get(DEADLINE_ENV)
    return float(d) - time.time() if d else math.inf


def fits_before_deadline(needed_s: float, what: str) -> bool:
    """False if `what` would not finish before the deadline; it is then left for the next job."""
    left = time_left_s()
    if left >= needed_s:
        return True
    log.warning("[deadline] skipping %s: needs ~%.0f min, %.0f min left; the next job resumes it",
                what, needed_s / 60, max(left, 0) / 60)
    return False


def group_reserve_s(settings: Settings, n_runs: int, mode: str) -> float:
    r = settings.experiment.run
    per_run = 60.0 if mode == "startup" else r.warmup_s + r.window_s + 30.0
    return STARTUP_RESERVE_S + n_runs * per_run + (settings.profile.profiler.flush_timeout_s
                                                   if settings.profile.investigations else 0.0)


@dataclass
class Job:
    point: WorkloadPoint
    mask: int
    rep: int
    key: str
    overrides: dict[str, Any] = field(default_factory=dict)
    tag: str = "run"                 # run | sweep | startup | investigate | profile
    launch: EngineLaunch | None = None
    launch_error: str | None = None
    deep: bool = False               # deep telemetry (investigations)
    profile_dir: str | None = None   # profiler pass: absolute dir for vLLM's torch-profiler traces
    tuned_point: str | None = None   # point whose tuned values apply (derived investigation points)
    exclusive: bool = False          # run with the other GPU slots idle
    inv_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------- factories

def make_engine(settings: Settings) -> Any:
    if settings.use_fake:
        from harness.engine.fake import FakeEngine
        return FakeEngine(settings.env, settings.profile.fake)
    from harness.engine.vllm_server import VllmServerGroup
    r = settings.experiment.run
    return VllmServerGroup(r.startup_timeout_s, r.gpu_release_timeout_s)


def tokenizer_factory(settings: Settings) -> Callable[[], Any]:
    if settings.use_fake:
        return FakeTokenizer
    return lambda: HFTokenizer(settings.env.tokenizer)


def plan_points(settings: Settings) -> list[WorkloadPoint]:
    return select_points(settings.experiment.regimes, settings.profile.points)


def active(settings: Settings) -> list[int]:
    return active_optimizations(settings.env, settings.experiment.run, settings.profile)


def slot_settings(settings: Settings) -> list[Settings]:
    """Settings per concurrent slot (run.gpus_per_cell); one slot = the whole env, as in the spec."""
    return [settings.model_copy(update={"env": e}) for e in slot_envs(settings.env, settings.experiment.run)]


def plan_cells(settings: Settings) -> list[Cell]:
    return build_cells(plan_points(settings), active(settings), settings.profile.design)


def plan_masks(settings: Settings) -> list[int]:
    return sorted({c.mask for c in plan_cells(settings)})


def get_traces(settings: Settings, points: list[WorkloadPoint], force: bool = False) -> dict[str, Trace]:
    exp = settings.experiment
    fac = tokenizer_factory(settings)
    cache: dict[str, Any] = {}

    def cached() -> Any:
        if "tok" not in cache:
            cache["tok"] = fac()
        return cache["tok"]

    out = {}
    for p in points:
        tr = ensure_trace(settings.results_dir, settings.env.name, p, exp.workload, exp.seed, cached, force=force)
        out[p.point_id] = tr
        log.info("[traces] %s: %d requests x %d prompt tokens, reuse %.2f, sha256 %s", p.point_id, len(tr),
                 p.context, tr.reuse_fraction, tr.content_hash[:12])
    return out


# ---------------------------------------------------------------- measurement

def _measure_real(settings: Settings, endpoints: list[Endpoint], launch: EngineLaunch, point: WorkloadPoint,
                  trace: Trace, deep_dir: Path | None = None, engine: Any = None,
                  ) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, str]]:
    """One measured repetition. With `deep_dir`, deep telemetry series are written there as well."""
    r = settings.experiment.run
    gpu_ids = sorted({g for ep in endpoints for g in ep.gpu_ids})
    deep = None
    if deep_dir is not None:
        from harness.deep_telemetry import DeepCollector
        roots = engine.root_pids() if engine is not None and hasattr(engine, "root_pids") else []
        deep = DeepCollector(gpu_ids, [ep.metrics_url for ep in endpoints], roots)

    async def go() -> tuple[Any, Any, Any, bool]:
        reset_ok = await reset_prefix_cache(endpoints) if r.reset_prefix_cache_between_reps else False
        scraper = PrometheusScraper([ep.metrics_url for ep in endpoints], r.telemetry_interval_s)
        sampler = NvmlSampler(gpu_ids, r.telemetry_interval_s)

        async def ws() -> None:
            sampler.start()
            scraper.start()
            if deep:
                deep.start()

        async def we() -> None:
            await scraper.stop()
            sampler.stop()
            if deep:
                await asyncio.to_thread(deep.stop)

        res = await run_closed_loop(endpoints, launch.served_model_name, trace, point.concurrency,
                                    r.warmup_s, r.window_s, r.request_timeout_s, ws, we)
        return res, scraper, sampler, reset_ok

    res, scraper, sampler, reset_ok = asyncio.run(go())
    tel, missing = scraper.summary()
    if missing:
        log.warning("vLLM metrics not found (logged, not fatal): %s", ", ".join(missing))
    tel.update(sampler.summary())
    if deep and deep_dir is not None:
        deep.write(deep_dir)
        tel.update(deep.summary())
    tel["telemetry_missing"] = missing
    tel["prefix_cache_reset"] = reset_ok
    return res.metrics, tel, res.requests, scraper.last_text


def _effective_max_num_seqs(launch: EngineLaunch, info: dict[str, Any]) -> int:
    if "max_num_seqs" in launch.tuned:
        return int(launch.tuned["max_num_seqs"]["value"])
    return int(info.get("max_num_seqs") or 256)


# ---------------------------------------------------------------- row helpers

def _base_row(settings: Settings, job: Job, trace: Trace | None) -> dict[str, Any]:
    p = job.point
    row: dict[str, Any] = {
        "run_key": job.key, "kind": job.tag, "profile": settings.profile.name, "env": settings.env.name,
        "config_bits": opt.to_bits(job.mask), "config_names": opt.names(job.mask), "point_id": p.point_id,
        "rep": job.rep, "context": p.context, "concurrency": p.concurrency, "reuse": p.reuse,
        "n_active": len(active(settings)), "overrides": job.overrides or None,
        "trace_hash": trace.content_hash if trace else None, "seed": settings.experiment.seed,
        "dry_run": settings.use_fake, **versions(),
    }
    if job.inv_ids:
        row.update(investigations=list(job.inv_ids), tuned_point=job.tuned_point or p.point_id,
                   exclusive=job.exclusive)
    if job.launch:
        row.update(layout=job.launch.layout, gpu_ids=sorted({g for sv in job.launch.servers for g in sv.gpu_ids}), flags=list(job.launch.flags), flags_key=job.launch.flags_key,
                   tuned_values=job.launch.tuned, used_placeholder=job.launch.uses_placeholders)
    return row


def _progress(job: Job, row: dict[str, Any]) -> None:
    thr = row.get("output_tok_s")
    thr_s = f"{thr:9.1f} tok/s" if isinstance(thr, (int, float)) and not math.isnan(thr) else "        - tok/s"
    log.info("[%s] %-14s %s %-28s rep=%d %-10s %s", job.tag, job.point.point_id, opt.to_bits(job.mask),
             opt.names(job.mask)[:28], job.rep, row.get("status"), thr_s)


# ---------------------------------------------------------------- the walk

def _attach_launches(settings: Settings, jobs: list[Job]) -> None:
    tuned = load_tuned(settings.tuned_path)
    for j in jobs:
        try:
            profiler = (j.profile_dir, settings.profile.profiler.active_iterations) if j.profile_dir else None
            j.launch = build_launch(j.mask, settings.env, settings.experiment,
                                    point_id=j.tuned_point or j.point.point_id, tuned=tuned,
                                    overrides=j.overrides, profiler=profiler)
        except ValueError as e:
            j.launch_error = str(e)


def execute(settings: Settings, jobs: list[Job], store: ResultStore, engine: Any, traces: dict[str, Trace],
            mode: str = "measure") -> list[dict[str, Any]]:
    """Run jobs grouped by (point, engine flags). mode: measure | startup.

    With several slots (run.gpus_per_cell), groups are spread over one worker process per slot; `engine`
    is then unused (each worker owns an engine on its GPUs).
    """
    slots = slot_settings(settings)
    _attach_launches(slots[0], jobs)   # flags_key does not depend on the slot's GPUs or ports

    written: list[dict[str, Any]] = []
    point_order = {p: i for i, p in enumerate(dict.fromkeys(j.point.point_id for j in jobs))}

    def gkey(j: Job) -> tuple[int, str]:
        return point_order[j.point.point_id], (j.launch.flags_key if j.launch else f"~{j.key}")

    jobs = sorted(jobs, key=lambda j: (*gkey(j), j.rep))
    groups: list[list[Job]] = []
    for _, grp in groupby(jobs, key=gkey):
        group = list(grp)
        trace = traces.get(group[0].point.point_id)
        if group[0].launch is None:
            for j in group:
                row = {**_base_row(settings, j, trace), "status": "infeasible", "error_kind": "bad_config",
                       "error": j.launch_error}
                store.append(row); written.append(row); _progress(j, row)
            continue
        groups.append(group)
    if len(slots) == 1:
        for group in groups:
            if not fits_before_deadline(group_reserve_s(settings, len(group), mode), _group_label(group)):
                continue
            written += _run_group(slots[0], group, store, engine, traces.get(group[0].point.point_id), mode)
    else:
        log.info("[%s] %d engine groups over %d slots of %d GPU(s)", mode, len(groups), len(slots),
                 slots[0].env.num_gpus)
        written += run_on_slots(settings, [(g, store.path, mode) for g in groups], _group_task)
    return written


def _group_task(settings: Settings, engine: Any, payload: tuple[list[Job], Path, str]) -> list[dict[str, Any]]:
    """One engine group inside a slot worker: launches rebuilt for this slot's GPUs and ports."""
    group, store_path, mode = payload
    if not fits_before_deadline(group_reserve_s(settings, len(group), mode), _group_label(group)):
        return []
    _attach_launches(settings, group)
    if group[0].launch is None:
        raise ValueError(f"slot cannot build launch: {group[0].launch_error}")
    trace = get_traces(settings, [group[0].point])[group[0].point.point_id]
    return _run_group(settings, group, ResultStore(store_path), engine, trace, mode)


def _group_label(group: list[Job]) -> str:
    j = group[0]
    return f"{j.tag} {j.point.point_id} {opt.to_bits(j.mask)} ({len(group)} runs)"


# ---------------------------------------------------------------- slot workers

TaskFn = Callable[[Settings, Any, Any], list[dict[str, Any]]]


def run_on_slots(settings: Settings, payloads: list[Any], fn: TaskFn) -> list[dict[str, Any]]:
    """Run fn(slot_settings, engine, payload) for every payload, one worker process per GPU slot.

    Workers write their own result rows (the store is append-safe across processes) and send them back.
    An EnvironmentFault in any worker stops dispatch; the others finish their current task, then it is raised.
    """
    slots = slot_settings(settings)
    ctx = mp.get_context("spawn")      # no CUDA/NVML/asyncio state inherited from the parent
    tasks: Any = ctx.Queue()
    results: Any = ctx.Queue()
    stop = ctx.Event()
    for p in payloads:
        tasks.put(p)
    for _ in slots:
        tasks.put(None)
    log_path = next((h.baseFilename for h in logging.getLogger().handlers
                     if isinstance(h, logging.FileHandler)), None)
    procs = [ctx.Process(target=_slot_worker, name=f"slot{i}",
                         args=(s, i, fn, tasks, results, stop, log_path, logging.getLogger().level))
             for i, s in enumerate(slots)]
    for pr in procs:
        pr.start()
    written: list[dict[str, Any]] = []
    fault: str | None = None
    pending = len(payloads)
    try:
        while pending and (any(pr.is_alive() for pr in procs) or not results.empty()):
            try:
                kind, data = results.get(timeout=5)
            except queue.Empty:
                continue
            pending -= 1
            if kind == "rows":
                written += data
            elif kind == "fault":
                fault = fault or data
                stop.set()
            else:   # task error: its rows were not written, so a re-run redoes it
                log.error("slot task failed (re-run the command to retry it): %s", data)
    except KeyboardInterrupt:
        stop.set()
        raise
    finally:
        tasks.cancel_join_thread()     # undispatched payloads after a stop must not block exit
        for pr in procs:
            pr.join()
    dead = [f"{pr.name} exited with code {pr.exitcode}" for pr in procs if pr.exitcode]
    if dead and not fault:
        fault = "; ".join(dead)
    if fault:
        raise EnvironmentFault(fault)
    return written


def _slot_worker(settings: Settings, slot: int, fn: TaskFn, tasks: Any, results: Any, stop: Any,
                 log_path: str | None, level: int) -> None:
    _worker_logging(slot, settings.env.gpu_ids, log_path, level)
    engine = make_engine(settings)
    try:
        while not stop.is_set():
            payload = tasks.get()
            if payload is None:
                break
            try:
                results.put(("rows", fn(settings, engine, payload)))
            except EnvironmentFault as e:
                results.put(("fault", f"slot {slot} (GPUs {settings.env.gpu_ids}): {e}"))
                break
            except Exception as e:  # noqa: BLE001 - one task's crash must not take the slot down
                log.exception("slot %d task crashed", slot)
                results.put(("error", f"slot {slot}: {type(e).__name__}: {e}"))
    finally:
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            log.exception("slot %d: engine stop failed", slot)


def _worker_logging(slot: int, gpu_ids: list[int], log_path: str | None, level: int) -> None:
    from harness.cli import _JsonFormatter
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    con = logging.StreamHandler()
    con.setFormatter(logging.Formatter(f"%(asctime)s %(levelname).1s [gpu{','.join(map(str, gpu_ids))}] "
                                       "%(message)s", "%H:%M:%S"))
    root.addHandler(con)
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)
    for noisy in ("urllib3", "httpx", "filelock", "huggingface_hub", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _run_group(settings: Settings, group: list[Job], store: ResultStore, engine: Any, trace: Trace | None,
               mode: str) -> list[dict[str, Any]]:
    launch = group[0].launch
    assert launch is not None
    written = []
    log_dir = settings.results_dir / "engine_logs" / f"{group[0].point.point_id}__{launch.flags_key}__{int(time.time())}"
    t_start = time.monotonic()
    try:
        endpoints = engine.start(launch, log_dir)
    except EngineStartError as e:
        for j in group:
            row = {**_base_row(settings, j, trace), "status": e.status, "error_kind": e.kind, "error": e.message,
                   "startup_s": round(time.monotonic() - t_start, 1)}
            _link_logs(settings, j, log_dir)
            store.append(row); written.append(row); _progress(j, row)
        _stop(engine)
        return written
    info = engine.info()
    try:
        for j in group:
            t0 = time.monotonic()
            row = {**_base_row(settings, j, trace), "engine_info": info, "startup_s": info.get("startup_s")}
            rdir = run_dir(settings.results_dir, j.key)
            try:
                if mode == "startup":
                    row.update(_startup_probe(settings, endpoints, launch, trace))
                    requests = None
                    snapshot = None
                elif j.tag == "profile":
                    row.update(_profile_pass(settings, endpoints, launch, j, trace))
                    requests = snapshot = None
                elif settings.use_fake:
                    m, tel, requests = engine.measure(j.point, j.rep, settings.experiment.workload.output_tokens,
                                                      _effective_max_num_seqs(launch, info))
                    snapshot = None
                    row.update(status="ok", **m, **tel)
                else:
                    m, tel, requests, snapshot = _measure_real(settings, endpoints, launch, j.point, trace,
                                                               deep_dir=rdir if j.deep else None, engine=engine)
                    status = "ok" if m.get("n_requests_window", 0) > 0 and m.get("output_tok_s") else "failed"
                    row.update(status=status, **m, **tel)
                    if status == "failed":
                        row["error"] = m.get("first_error") or "no requests completed inside the window"
                row["max_num_seqs_effective"] = _effective_max_num_seqs(launch, info)
            except Exception as e:  # noqa: BLE001 - a crash in one run is a result
                log.exception("run %s failed", j.key)
                row.update(status="failed", error_kind="exception", error=f"{type(e).__name__}: {e}"[:3000])
                requests, snapshot = None, None
            row["duration_s"] = round(time.monotonic() - t0, 1)
            write_run_artifacts(rdir, flags={"argv": [list(s.argv) for s in launch.servers],
                                             "env": [s.env for s in launch.servers], "tuned": launch.tuned},
                                trace_hash=trace.content_hash if trace else None, metrics_snapshot=snapshot,
                                requests=requests)
            _link_logs(settings, j, log_dir)
            store.append(row); written.append(row); _progress(j, row)
            if row["status"] != "ok" and not engine.health():
                log.warning("engine unhealthy after %s; restarting for the remaining runs", j.key)
                _stop(engine)
                try:
                    endpoints = engine.start(launch, log_dir.with_name(log_dir.name + "_restart"))
                    info = engine.info()
                except EngineStartError as e:
                    rest = group[group.index(j) + 1:]
                    for k in rest:
                        r2 = {**_base_row(settings, k, trace), "status": e.status, "error_kind": e.kind,
                              "error": "restart after failure: " + e.message}
                        store.append(r2); written.append(r2); _progress(k, r2)
                    break
    finally:
        _stop(engine)
    return written


def _stop(engine: Any) -> None:
    try:
        engine.stop()
    except RuntimeError as e:
        raise EnvironmentFault(f"{e}. Stop here: later runs would fail or be contaminated. "
                               f"Free the GPU, then re-run the same command to resume.") from e


def _link_logs(settings: Settings, job: Job, log_dir: Path) -> None:
    rdir = run_dir(settings.results_dir, job.key)
    if not log_dir.exists():
        return
    for f in log_dir.glob("server_*.log"):
        link = rdir / f.name
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.relpath(f, rdir), link)
        except OSError:
            pass


def _startup_probe(settings: Settings, endpoints: list[Endpoint], launch: EngineLaunch,
                   trace: Trace | None) -> dict[str, Any]:
    """One short request per server."""
    if settings.use_fake:
        return {"status": "ok", "probe_text": "fake"}
    prompt = trace.prompt(0)[:64] if trace else "Hello"
    texts = []
    for ep in endpoints:
        resp = asyncio.run(single_request(ep, launch.served_model_name, prompt, 8, timeout_s=300))
        texts.append(resp["choices"][0]["text"])
    return {"status": "ok", "probe_text": texts[0][:200]}


def _profile_pass(settings: Settings, endpoints: list[Endpoint], launch: EngineLaunch, job: Job,
                  trace: Trace | None) -> dict[str, Any]:
    """Diagnostic pass: under the same closed-loop load, vLLM's torch profiler records a few engine steps.
    Its throughput is distorted by profiling and is not a result; the trace summary is."""
    pp = settings.profile.profiler
    tdir = Path(job.profile_dir or "")
    if settings.use_fake:
        return {"status": "ok", "profile": {"gpu_busy_frac": 0.8, "n_kernels": 1, "fake": True}}
    assert trace is not None
    tdir.mkdir(parents=True, exist_ok=True)
    r = settings.experiment.run
    started: dict[str, bool] = {}

    async def ws() -> None:
        started["start"] = await post_all(endpoints, "/start_profile", timeout_s=120)

    async def we() -> None:
        started["stop"] = await post_all(endpoints, "/stop_profile", timeout_s=pp.flush_timeout_s)

    res = asyncio.run(run_closed_loop(endpoints, launch.served_model_name, trace, job.point.concurrency,
                                      pp.warmup_s, pp.window_s, r.request_timeout_s, ws, we))
    from harness.deep_telemetry import summarize_profile_dir, wait_for_traces
    files = wait_for_traces(tdir, pp.flush_timeout_s)
    summary = summarize_profile_dir(tdir)
    ok = bool(started.get("start") and summary.get("n_kernels"))
    return {"status": "ok" if ok else "failed", "profile": summary, "profile_files": [str(f) for f in files],
            "profile_output_tok_s": res.metrics.get("output_tok_s"),
            **({} if ok else {"error": f"profiler: start={started.get('start')} stop={started.get('stop')}, "
                                       f"{len(files)} trace file(s), kernels={summary.get('n_kernels')}"})}


# ---------------------------------------------------------------- investigations (profile `investigate`)

def parse_config(text: str) -> int:
    """'default', 'spec+prefix' or a 7-bit string."""
    t = text.strip()
    return opt.from_bits(t) if len(t) == opt.N_OPTS and set(t) <= {"0", "1"} else opt.from_names(t)


def investigation_points(settings: Settings, inv: Any) -> list[tuple[WorkloadPoint, str]]:
    """(point, point whose tuned values apply). Varied points equal to an existing one reuse its id."""
    known = all_points(settings.experiment.regimes)
    by_id = {p.point_id: p for p in known}
    if inv.points:
        return [(by_id[i], i) for i in inv.points]
    base = by_id[inv.base]
    out = []
    for regime, values in inv.vary.items():
        for v in values:
            vals = {"context": base.context, "concurrency": base.concurrency, "reuse": base.reuse}
            vals[regime] = float(v) if regime == "reuse" else int(v)
            same = [p for p in known if (p.context, p.concurrency, p.reuse) == tuple(vals.values())]
            if same:
                out.append((same[0], same[0].point_id))
            else:
                pid = f"{base.point_id}@{regime}={v:g}"
                out.append((WorkloadPoint(pid, varied=regime, level="custom", **vals), base.point_id))
    return out


def investigate_jobs(settings: Settings) -> list[Job]:
    """Union of all investigations' cells (a cell named by several runs once), then reps + profiler passes."""
    cells: dict[tuple[str, int, bool], dict[str, Any]] = {}
    for inv in settings.profile.investigations:
        prof = {parse_config(c) for c in (inv.configs if inv.profile == "all" else inv.profile)}
        for point, tuned_point in investigation_points(settings, inv):
            for c in inv.configs:
                m = parse_config(c)
                c_ = cells.setdefault((point.point_id, m, inv.exclusive),
                                      {"point": point, "tuned": tuned_point, "reps": 0, "profile": False, "inv": []})
                c_["reps"] = max(c_["reps"], inv.reps or settings.experiment.run.reps)
                c_["profile"] |= m in prof
                c_["inv"].append(inv.id)
    pre = f"{settings.profile.name}__{settings.env.name}"
    jobs: list[Job] = []
    for (pid, m, excl), c in cells.items():
        stem = f"{pre}__{pid}__{opt.to_bits(m)}" + ("__alone" if excl else "")
        common = dict(tuned_point=c["tuned"], exclusive=excl, inv_ids=tuple(dict.fromkeys(c["inv"])))
        for rep in range(1, c["reps"] + 1):
            jobs.append(Job(c["point"], m, rep, f"{stem}__r{rep}", tag="investigate", deep=True, **common))
        if c["profile"]:
            pdir = (settings.results_dir / "profiles" / Path(stem).name).resolve()
            jobs.append(Job(c["point"], m, 0, f"{stem}__profile", tag="profile", profile_dir=str(pdir), **common))
    return jobs


def cmd_investigate(settings: Settings, engine: Any, force: bool = False, retry_failed: bool = False
                    ) -> list[dict[str, Any]]:
    """Shared cells first (all GPU slots), then exclusive cells on one slot with the others idle."""
    store = ResultStore(settings.results_dir / "investigate.jsonl")
    jobs = _filter_done(investigate_jobs(settings), store, force, retry_failed)
    points = list({j.point.point_id: j.point for j in jobs}.values())
    log.info("[investigate] %d runs to do (%d shared, %d exclusive) at %d points", len(jobs),
             sum(not j.exclusive for j in jobs), sum(j.exclusive for j in jobs), len(points))
    traces = get_traces(settings, points)      # in the parent: workers only load them
    rows = execute(settings, [j for j in jobs if not j.exclusive], store, engine, traces)
    alone = [j for j in jobs if j.exclusive]
    if alone:
        one = settings.model_copy(update={"env": slot_envs(settings.env, settings.experiment.run)[0]})
        log.info("[investigate] exclusive phase: %d runs on GPU %s, other slots idle", len(alone), one.env.gpu_ids)
        rows += execute(one, alone, store, make_engine(one), traces)
    return rows


# Per-unit costs measured on 2 x H100 NVL (fast plan, 2026-11-07): engine start median 55 s / p90 88 s.
EST_START_S = 70.0
EST_REP_OVERHEAD_S = 8.0          # prefix-cache reset, telemetry flush, parquet writes
EST_PROFILE_EXTRA_S = 90.0        # trace flush + parse after /stop_profile
EST_QUALITY_SETTING_S = 730.0     # trimmed setting took ~375 s; full GSM8K (2.6x) + 8 MMLU subjects (1.8x)
EST_TRACE_TOK_PER_S = 650_000.0   # trace generation, tokenizer-bound (smoke: 4.6M tokens in 7 s)


def estimate_investigation(settings: Settings, include_quality: bool = True) -> dict[str, float]:
    """Wall-clock estimate in hours, by phase; shared work is spread over the GPU slots."""
    from harness.quality import numeric_settings, quality_masks
    r, pp = settings.experiment.run, settings.profile.profiler
    n_slots = len(slot_settings(settings))
    jobs = investigate_jobs(settings)

    def cost(js: list[Job]) -> float:
        starts = len({(j.point.point_id, j.mask, j.tag) for j in js})
        reps = sum(j.tag == "investigate" for j in js)
        profs = sum(j.tag == "profile" for j in js)
        return (starts * EST_START_S + reps * (r.warmup_s + r.window_s + EST_REP_OVERHEAD_S)
                + profs * (pp.warmup_s + pp.window_s + EST_PROFILE_EXTRA_S))

    shared = cost([j for j in jobs if not j.exclusive]) / n_slots
    alone = cost([j for j in jobs if j.exclusive])
    from harness.workload import trace_path
    missing = {j.point.point_id: j.point for j in jobs
               if not trace_path(settings.results_dir, settings.env.name, j.point.point_id).exists()}
    traces = sum(settings.experiment.workload.trace_requests * p.context for p in missing.values()) \
        / EST_TRACE_TOK_PER_S
    out = {"traces_h": traces / 3600, "shared_h": shared / 3600, "exclusive_h": alone / 3600}
    if include_quality:
        nq = len(quality_masks(settings, numeric_settings(plan_masks(settings))))
        out["quality_h"] = math.ceil(nq / n_slots) * EST_QUALITY_SETTING_S / 3600
    out["total_h"] = sum(out.values())
    out["gpu_h"] = (out["shared_h"] + out["exclusive_h"] + out.get("quality_h", 0.0)) * settings.env.num_gpus
    return out


# ---------------------------------------------------------------- commands

def _filter_done(jobs: list[Job], store: ResultStore, force: bool, retry_failed: bool) -> list[Job]:
    if force:
        return jobs
    done = store.done_keys(retry_failed=retry_failed)
    todo = [j for j in jobs if j.key not in done]
    if len(todo) < len(jobs):
        log.info("resume: skipping %d completed run keys (use --force to rerun)", len(jobs) - len(todo))
    return todo


def run_jobs(settings: Settings, only_default_center: bool = False) -> list[Job]:
    points = plan_points(settings)
    cells = build_cells(points, active(settings), settings.profile.design)
    if only_default_center:
        cells = [c for c in cells if c.mask == 0 and c.point_id == CENTER]
    by_id = {p.point_id: p for p in points}
    runs = build_runs(cells, settings.experiment.run.reps)
    return [Job(by_id[r.point_id], r.mask, r.rep,
                run_key(settings.profile.name, settings.env.name, r.bits, r.point_id, r.rep)) for r in runs]


def startup_jobs(settings: Settings, all_configs: bool = False) -> list[Job]:
    center = select_points(settings.experiment.regimes, [CENTER])[0]
    act = active(settings)
    masks = full_factorial(act) if all_configs else configs_for_point(center, act, settings.profile.design)
    return [Job(center, m, 1, f"{settings.profile.name}__{settings.env.name}__startup__{opt.to_bits(m)}",
                tag="startup") for m in masks]


SWEPT = (("max_num_seqs", 5), ("max_num_batched_tokens", 6), ("parallel_layout", 7))


def sweep_jobs(settings: Settings) -> list[Job]:
    sw = settings.experiment.sweeps
    act = active(settings)
    jobs: list[Job] = []
    for p in plan_points(settings):
        for param, number in SWEPT:
            if number not in act:
                continue
            for v in getattr(sw, param):
                for rep in range(1, sw.reps + 1):
                    key = f"{settings.profile.name}__{settings.env.name}__sweep__{p.point_id}__{param}={v}__r{rep}"
                    jobs.append(Job(p, opt.bit(number), rep, key, overrides={param: v}, tag="sweep"))
    return jobs


def progress(settings: Settings) -> dict[str, dict[str, int]]:
    """Per step: total keys, finished (ok/infeasible), failed (failed/timeout), not yet attempted."""
    from harness.quality import numeric_settings, quality_key, quality_masks
    qkeys = [quality_key(settings, m) for m in quality_masks(settings, numeric_settings(plan_masks(settings)))]
    if settings.profile.investigations:
        steps = {"investigate": ("investigate.jsonl", [j.key for j in investigate_jobs(settings)]),
                 "quality": ("quality.jsonl", qkeys)}
    else:
        steps = {"sweep": ("sweep.jsonl", [j.key for j in sweep_jobs(settings)]),
                 "startup-check": ("startup.jsonl", [j.key for j in startup_jobs(settings)]),
                 "quality": ("quality.jsonl", qkeys),
                 "run": ("runs.jsonl", [j.key for j in run_jobs(settings)])}
    out = {}
    for step, (fname, keys) in steps.items():
        latest = ResultStore(settings.results_dir / fname).latest()
        st = [latest[k].get("status") if k in latest else None for k in keys]
        out[step] = {"total": len(keys), "finished": sum(x in ("ok", "infeasible") for x in st),
                     "failed": sum(x in ("failed", "timeout") for x in st), "todo": st.count(None)}
    return out


def cmd_run(settings: Settings, engine: Any, force: bool = False, retry_failed: bool = False,
            only_default_center: bool = False) -> list[dict[str, Any]]:
    points = plan_points(settings)
    store = ResultStore(settings.results_dir / "runs.jsonl")
    jobs = _filter_done(run_jobs(settings, only_default_center), store, force, retry_failed)
    log.info("[run] %d runs to do (%s, env=%s, n=%d)", len(jobs), settings.profile.name, settings.env.name,
             len(active(settings)))
    traces = get_traces(settings, [p for p in points if any(j.point.point_id == p.point_id for j in jobs)])
    return execute(settings, jobs, store, engine, traces, mode="measure")


def cmd_startup_check(settings: Settings, engine: Any, all_configs: bool = False, force: bool = False,
                      retry_failed: bool = False) -> list[dict[str, Any]]:
    store = ResultStore(settings.results_dir / "startup.jsonl")
    jobs = _filter_done(startup_jobs(settings, all_configs), store, force, retry_failed)
    traces = get_traces(settings, select_points(settings.experiment.regimes, [CENTER]))
    return execute(settings, jobs, store, engine, traces, mode="startup")


def cmd_sweep(settings: Settings, engine: Any, force: bool = False, retry_failed: bool = False
              ) -> dict[str, Any]:
    store = ResultStore(settings.results_dir / "sweep.jsonl")
    todo = _filter_done(sweep_jobs(settings), store, force, retry_failed)
    traces = get_traces(settings, plan_points(settings))
    execute(settings, todo, store, engine, traces, mode="measure")
    return write_tuned_from_sweep(settings, store)


def write_tuned_from_sweep(settings: Settings, store: ResultStore) -> dict[str, Any]:
    """Best mean throughput that passes the latency gates, per (point, parameter) -> tuned.yaml."""
    g = settings.experiment.gates
    rows = [r for r in store.latest().values() if r.get("env") == settings.env.name and r.get("status") == "ok"
            and r.get("overrides")]
    tuned = load_tuned(settings.tuned_path)
    env_t = tuned.setdefault(settings.env.name, {})
    chosen: dict[str, Any] = {}
    for p in plan_points(settings):
        for param, _ in SWEPT:
            cands: dict[Any, list[dict[str, Any]]] = {}
            for r in rows:
                if r["point_id"] == p.point_id and param in r["overrides"]:
                    cands.setdefault(r["overrides"][param], []).append(r)
            if not cands:
                continue
            scored = []
            for v, rs in cands.items():
                passes = all((r.get("tpot_p99_ms") or math.inf) <= g.p99_tpot_ms
                             and (r.get("ttft_p99_ms") or math.inf) <= g.p99_ttft_ms for r in rs)
                thr = sum(r.get("output_tok_s") or 0 for r in rs) / len(rs)
                scored.append((passes, thr, v))
            ok = [s for s in scored if s[0]]
            if not ok:
                log.warning("[sweep] %s %s: no candidate passes the latency gates; keeping placeholder",
                            p.point_id, param)
                continue
            best = max(ok, key=lambda s: s[1])
            env_t.setdefault(p.point_id, {})[param] = best[2]
            chosen[f"{p.point_id}.{param}"] = best[2]
            log.info("[sweep] %s %s -> %s (%.1f tok/s)", p.point_id, param, best[2], best[1])
    save_tuned(settings.tuned_path, tuned)
    return chosen
