"""Entry point: `python -m harness <command>`."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from harness import optimizations as opt
from harness.config import REPO_ROOT, Settings, envs_for, load_env, load_settings, results_dir_for, load_profile

log = logging.getLogger("harness")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
             "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


def setup_logging(log_dir: Path, command: str, verbose: bool = False) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{command}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    con = logging.StreamHandler(sys.stderr)
    con.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%H:%M:%S"))
    fh = logging.FileHandler(path)
    fh.setFormatter(_JsonFormatter())
    root.addHandler(con)
    root.addHandler(fh)
    for noisy in ("urllib3", "httpx", "filelock", "huggingface_hub", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


# ---------------------------------------------------------------- commands

def _settings(args: argparse.Namespace, env: str) -> Settings:
    return load_settings(args.profile, env, dry_run=args.dry_run)


def _engine(settings: Settings) -> Any:
    from harness.orchestrator import make_engine
    if not settings.use_fake:
        from harness.doctor import free_vram_ok, preflight_flags
        preflight_flags()
        ok, detail = free_vram_ok(settings.env)
        if not ok:
            raise SystemExit(f"Not enough free GPU memory to start vLLM:\n{detail}\n"
                             "Free the GPU (or lower min_free_vram_gb in the env config) and re-run.")
    return make_engine(settings)


def cmd_doctor(args: argparse.Namespace) -> int:
    from harness.doctor import format_checks, run_doctor
    ok, checks = run_doctor(load_env(args.env), dry_run=args.dry_run)
    print(format_checks(checks))
    print("\ndoctor:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _plan_investigate(s: Settings, verbose: bool) -> None:
    from harness.orchestrator import estimate_investigation, investigate_jobs, slot_settings
    jobs = investigate_jobs(s)
    print(f"profile={s.profile.name} env={s.env.name}: {len(s.profile.investigations)} investigations, "
          f"{len(slot_settings(s))} GPU slot(s)")
    for inv in s.profile.investigations:
        mine = [j for j in jobs if inv.id in j.inv_ids]
        cells = {(j.point.point_id, j.mask) for j in mine}
        print(f"  {inv.id:<24} {len(cells):>3} cells, {sum(j.tag == 'investigate' for j in mine):>3} measured runs, "
              f"{sum(j.tag == 'profile' for j in mine):>2} profiler passes" + ("  [exclusive]" if inv.exclusive else ""))
        if verbose:
            for pid, m in sorted(cells):
                print(f"      {pid:<22} {opt.to_bits(m)} {opt.names(m)}")
    n_meas = sum(j.tag == "investigate" for j in jobs)
    print(f"  TOTAL (shared cells run once): {len({(j.point.point_id, j.mask, j.exclusive) for j in jobs})} cells, "
          f"{n_meas} measured runs, {sum(j.tag == 'profile' for j in jobs)} profiler passes")
    est = estimate_investigation(s)
    print("  estimated wall clock: " + ", ".join(f"{k[:-2]} {v:.2f} h" for k, v in est.items() if k != "gpu_h")
          + f"  ({est['gpu_h']:.1f} GPU-hours)")


def cmd_plan(args: argparse.Namespace) -> int:
    from harness.design import build_runs, describe
    from harness.orchestrator import active, plan_cells
    for env in envs_for(args.profile, args.env):
        s = _settings(args, env)
        if s.profile.investigations:
            _plan_investigate(s, args.verbose)
            continue
        cells = plan_cells(s)
        runs = build_runs(cells, s.experiment.run.reps)
        print(f"profile={s.profile.name} env={env} design={s.profile.design}")
        print(describe(cells, active(s)))
        print(f"  reps per cell: {s.experiment.run.reps} -> {len(runs)} measured runs")
        if args.verbose:
            for c in cells:
                print(f"    {c.point_id:<18} {c.bits} {opt.names(c.mask)}")
        print()
    return 0


def cmd_traces(args: argparse.Namespace) -> int:
    from harness.orchestrator import get_traces, plan_points
    for env in envs_for(args.profile, args.env):
        s = _settings(args, env)
        get_traces(s, plan_points(s), force=args.force)
    return 0


def cmd_startup_check(args: argparse.Namespace) -> int:
    from harness.orchestrator import cmd_startup_check as run
    for env in envs_for(args.profile, args.env):
        s = _settings(args, env)
        rows = run(s, _engine(s), all_configs=args.all, force=args.force, retry_failed=args.retry_failed)
        _status_summary("startup-check", rows)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from harness.orchestrator import cmd_sweep as run
    for env in envs_for(args.profile, args.env):
        s = _settings(args, env)
        chosen = run(s, _engine(s), force=args.force, retry_failed=args.retry_failed)
        log.info("[sweep] tuned values written to %s: %s", s.tuned_path, chosen)
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    from harness.orchestrator import plan_masks
    from harness.quality import run_quality
    for env in envs_for(args.profile, args.env):
        s = _settings(args, env)
        rows = run_quality(s, plan_masks(s), _engine(s), force=args.force)
        _status_summary("quality", rows)
    return 0


def cmd_investigate(args: argparse.Namespace) -> int:
    from harness.orchestrator import cmd_investigate as run
    for env in envs_for(args.profile, args.env):
        s = _settings(args, env)
        if not s.profile.investigations:
            raise SystemExit(f"profile {args.profile!r} has no `investigations:`")
        rows = run(s, _engine(s), force=args.force, retry_failed=args.retry_failed)
        _status_summary("investigate", rows)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from harness.orchestrator import cmd_run as run
    for env in envs_for(args.profile, args.env):
        s = _settings(args, env)
        rows = run(s, _engine(s), force=args.force, retry_failed=args.retry_failed,
                   only_default_center=args.baseline)
        _status_summary("run", rows)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    prof = load_profile(args.profile)
    rdir = results_dir_for(prof, args.dry_run)
    if prof.investigations:
        from harness.analysis.investigate import analyze_investigation
        out = analyze_investigation(rdir, args.profile)
        print(f"investigation report: {out / 'investigation.md'}")
        return 0
    from harness.analysis.report import analyze
    out = analyze(rdir, args.profile)
    print((out / "summary.md").read_text())
    return 0


def _run_steps(args: argparse.Namespace, steps: list[tuple[str, Callable[[argparse.Namespace], int]]],
               timings_path: Path, what: str) -> int:
    """Run steps in order with timings; stop on a failed doctor or an environment fault."""
    from harness.orchestrator import EnvironmentFault
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timings: list[dict[str, Any]] = []
    rc_all = 0
    for name, fn in steps:
        t0 = time.monotonic()
        log.info("===== %s step: %s =====", what, name)
        stop = False
        try:
            rc = fn(args)
            err = None
        except EnvironmentFault as e:
            log.error("ENVIRONMENT FAULT: %s", e)
            rc, err, stop = 3, f"EnvironmentFault: {e}", True
        except SystemExit as e:
            rc, err = (e.code if isinstance(e.code, int) else 1), str(e)
        except Exception as e:  # noqa: BLE001
            log.exception("%s step %s crashed", what, name)
            rc, err = 1, f"{type(e).__name__}: {e}"
        dt = time.monotonic() - t0
        timings.append({"step": name, "seconds": round(dt, 1), "rc": rc, "error": err})
        log.info("===== %s finished in %.1fs (rc=%s) =====", name, dt, rc)
        timings_path.write_text(json.dumps(timings, indent=2))
        if rc != 0:
            rc_all = rc
            if name == "doctor":
                log.error("doctor failed; stopping the %s sequence", what)
                stop = True
        if stop:
            break
    for t in timings:
        print(f"  {t['step']:<15} {t['seconds']:>8.1f}s  rc={t['rc']}" + (f"  {t['error']}" if t["error"] else ""))
    return rc_all


def cmd_smoke(args: argparse.Namespace) -> int:
    """doctor -> traces -> startup-check -> sweep -> quality -> run -> analyze, with timings."""
    args.profile = "smoke"
    steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
        ("doctor", cmd_doctor), ("traces", cmd_traces), ("startup-check", cmd_startup_check),
        ("sweep", cmd_sweep), ("quality", cmd_quality), ("run", cmd_run), ("analyze", cmd_analyze)]
    rdir = results_dir_for(load_profile("smoke"), args.dry_run)
    return _run_steps(args, steps, rdir / "smoke_timings.json", "smoke")


INCOMPLETE = 10   # pipeline exit code: work left for the next job


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Whole study in RUNBOOK order, resumable; meant to be run by a chain of time-limited batch jobs.

    doctor -> traces -> baseline -> sweep -> startup-check -> quality -> run -> analyze. Completed keys are
    skipped and failed ones retried once per invocation. With HARNESS_DEADLINE set, no engine group or
    quality setting is started that would not finish in time. Exit 0 when nothing is left, 10 otherwise.
    """
    import os
    args.force, args.retry_failed, args.all = False, True, False

    def baseline(a: argparse.Namespace) -> int:
        a.baseline = True
        try:
            return cmd_run(a)
        finally:
            a.baseline = False

    steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
        ("doctor", cmd_doctor), ("traces", cmd_traces), ("baseline", baseline), ("sweep", cmd_sweep),
        ("startup-check", cmd_startup_check), ("quality", cmd_quality), ("run", cmd_run), ("analyze", cmd_analyze)]
    if load_profile(args.profile).investigations:
        steps = [("doctor", cmd_doctor), ("investigate", cmd_investigate), ("quality", cmd_quality),
                 ("analyze", cmd_analyze)]
    if args.dry_run:
        steps = steps[1:]
    rdir = results_dir_for(load_profile(args.profile), args.dry_run)
    tag = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    rc = _run_steps(args, steps, rdir / "pipeline" / f"timings_{tag}.json", "pipeline")
    left = _print_status(args)
    if rc == 0 and left:
        log.warning("INCOMPLETE: %d keys not attempted yet; re-run (or let the next chained job run) to resume",
                    left)
        return INCOMPLETE
    if rc == 0:
        log.info("pipeline complete: every key attempted; report in %s", rdir / "report")
    return rc


def _print_status(args: argparse.Namespace) -> int:
    from harness.orchestrator import progress
    left = 0
    for env in envs_for(args.profile, args.env):
        prog = progress(_settings(args, env))
        print(f"profile={args.profile} env={env}")
        print(f"  {'step':<15} {'total':>6} {'finished':>9} {'failed':>7} {'todo':>6}")
        for step, c in prog.items():
            print(f"  {step:<15} {c['total']:>6} {c['finished']:>9} {c['failed']:>7} {c['todo']:>6}")
            left += c["todo"]
    return left


def cmd_status(args: argparse.Namespace) -> int:
    _print_status(args)
    return 0


def _status_summary(what: str, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    log.info("[%s] done: %d rows %s", what, len(rows), counts)


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m harness", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], int], profile: bool = True, env: bool = True,
            env_required: bool = False, resumable: bool = False) -> argparse.ArgumentParser:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
        if profile:
            sp.add_argument("--profile", required=True, help="dry | smoke | full")
        if env:
            sp.add_argument("--env", required=env_required, default=None, help="test | main")
        sp.add_argument("--dry-run", action="store_true", help="use FakeEngine; no GPU needed")
        sp.add_argument("-v", "--verbose", action="store_true")
        if resumable:
            sp.add_argument("--force", action="store_true", help="rerun completed run keys")
            sp.add_argument("--retry-failed", action="store_true", help="rerun keys whose status is failed/timeout")
        return sp

    add("doctor", cmd_doctor, profile=False, env_required=True)
    add("plan", cmd_plan)
    tr = add("traces", cmd_traces)
    tr.add_argument("--force", action="store_true", help="regenerate traces")
    sc = add("startup-check", cmd_startup_check, resumable=True)
    sc.add_argument("--all", action="store_true", help="all 2^n configurations, not just the profile's")
    add("sweep", cmd_sweep, resumable=True)
    q = add("quality", cmd_quality)
    q.add_argument("--force", action="store_true")
    r = add("run", cmd_run, resumable=True)
    r.add_argument("--baseline", action="store_true", help="only Default at the central point (noise level)")
    add("analyze", cmd_analyze, env=False)
    sm = add("smoke", cmd_smoke, profile=False, env_required=True, resumable=True)
    sm.add_argument("--all", action="store_true", help="start-up check over all configurations")
    sm.set_defaults(baseline=False)
    add("investigate", cmd_investigate, resumable=True)
    pl = add("pipeline", cmd_pipeline)
    pl.set_defaults(baseline=False)
    add("status", cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = getattr(args, "profile", None) or ("smoke" if args.command == "smoke" else None)
    if profile:
        log_dir = results_dir_for(load_profile(profile), args.dry_run) / "logs"
    else:
        log_dir = REPO_ROOT / "logs"
    path = setup_logging(log_dir, args.command, args.verbose)
    log.debug("log file %s", path)
    from harness.orchestrator import EnvironmentFault
    try:
        return int(args.func(args) or 0)
    except EnvironmentFault as e:
        log.error("ENVIRONMENT FAULT: %s", e)
        return 3
