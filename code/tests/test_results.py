from harness.results import ResultStore, run_key


def test_resume_skips_completed(tmp_path):
    s = ResultStore(tmp_path / "runs.jsonl")
    k1 = run_key("smoke", "test", "0000000", "center", 1)
    k2 = run_key("smoke", "test", "1000000", "center", 1)
    k3 = run_key("smoke", "test", "0100000", "center", 1)
    s.append({"run_key": k1, "status": "ok"})
    s.append({"run_key": k2, "status": "failed"})
    s.append({"run_key": k3, "status": "infeasible"})
    assert s.done_keys() == {k1, k2, k3}
    assert s.done_keys(retry_failed=True) == {k1, k3}
    # a rerun appends; the latest row wins
    s.append({"run_key": k2, "status": "ok"})
    assert s.latest()[k2]["status"] == "ok"
    assert s.done_keys(retry_failed=True) == {k1, k2, k3}


def test_torn_line_tolerated(tmp_path):
    p = tmp_path / "runs.jsonl"
    s = ResultStore(p)
    s.append({"run_key": "a", "status": "ok"})
    with open(p, "a") as f:
        f.write('{"run_key": "b", "sta')          # killed mid-write
    assert set(s.latest()) == {"a"}
