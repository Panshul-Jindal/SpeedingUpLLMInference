"""Batch-cluster support: Slurm GPU ids, per-job ports, port clashes, job deadlines, progress counts."""
import socket
import time

import pytest

from harness.config import load_env, load_settings
from harness.engine.base import EngineStartError
from harness.engine.vllm_flags import build_launch
from harness.engine.vllm_server import VllmServerGroup
from harness.orchestrator import cmd_run, make_engine, progress, slot_settings

from test_dry_run import _small


def test_auto_gpu_ids_follow_slurm_allocation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    env = load_env("h100")
    assert env.gpu_ids == [4, 5]
    s = load_settings("fast", "h100")
    assert [x.env.gpu_ids for x in slot_settings(s)] == [[4], [5]]
    launch = build_launch(0, slot_settings(s)[1].env, s.experiment, point_id="center")
    assert launch.servers[0].env["CUDA_VISIBLE_DEVICES"] == "5"
    assert launch.servers[0].env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_auto_gpu_ids_without_slurm_and_bad_values(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert load_env("h100").gpu_ids == [0, 1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(ValueError, match="needs 2 GPUs"):
        load_env("h100")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")
    with pytest.raises(ValueError, match="integer"):
        load_env("h100")


def test_job_port_base_override(monkeypatch):
    monkeypatch.setenv("HARNESS_BASE_PORT", "23450")
    s = load_settings("fast", "h100")
    assert [x.env.base_port for x in slot_settings(s)] == [23450, 23460]


def test_busy_port_is_a_retryable_failure_not_infeasible(monkeypatch):
    with socket.socket() as sk:
        sk.bind(("0.0.0.0", 0))
        sk.listen()
        monkeypatch.setenv("HARNESS_BASE_PORT", str(sk.getsockname()[1]))
        s = load_settings("smoke", "test")
        launch = build_launch(0, s.env, s.experiment, point_id="center")
        with pytest.raises(EngineStartError) as e:
            VllmServerGroup(10, 10).start(launch, s.results_dir / "never")
    assert e.value.status == "failed" and e.value.kind == "port_in_use"


def test_deadline_skips_work_that_cannot_finish_then_resumes(tmp_path, monkeypatch):
    s = _small(load_settings("fast", "h100", dry_run=True, results_root=tmp_path))
    monkeypatch.setenv("HARNESS_DEADLINE", str(time.time() + 60))     # < one engine start-up reserve
    assert cmd_run(s, make_engine(s)) == []
    assert progress(s)["run"]["todo"] == 128 * 3

    monkeypatch.delenv("HARNESS_DEADLINE")
    rows = cmd_run(s, make_engine(s))
    assert len(rows) == 128 * 3 and {tuple(r["gpu_ids"]) for r in rows} == {(0,), (1,)}
    prog = progress(s)
    assert prog["run"] == {"total": 384, "finished": 384, "failed": 0, "todo": 0}
    assert prog["quality"]["todo"] == 8


def test_port_with_lingering_connections_is_free():
    """A restarted server's port has TIME_WAIT connections; only a live listener makes it busy."""
    from harness.engine.vllm_server import port_free
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    cli = socket.create_connection(("127.0.0.1", port))
    conn, _ = srv.accept()
    conn.close()            # server side closes first -> its end of the connection goes to TIME_WAIT
    cli.close()
    assert not port_free(port)
    srv.close()
    assert port_free(port)
