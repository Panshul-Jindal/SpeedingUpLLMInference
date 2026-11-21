from harness.telemetry import parse_prometheus, resolve_names, summarize_scrapes

TEXT_NEW = """# HELP vllm:kv_cache_usage_perc KV usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 7
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 1
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{engine="0",model_name="m"} %(q)s
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} %(h)s
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_bucket{le="1.0",model_name="m"} 3
vllm:request_queue_time_seconds_count{model_name="m"} %(c)s
vllm:request_queue_time_seconds_sum{model_name="m"} %(s)s
"""


def test_parse_and_resolve():
    d = parse_prometheus(TEXT_NEW % {"q": 100, "h": 40, "c": 2, "s": 1.0})
    assert d["vllm:kv_cache_usage_perc"] == 0.5
    mapping, missing = resolve_names(set(d))
    assert mapping["kv_usage"] == "vllm:kv_cache_usage_perc"
    assert mapping["prefix_hits"] == "vllm:prefix_cache_hits_total"
    assert "preemptions" in missing and "spec_draft_tokens" in missing


def test_old_name_fallback():
    d = parse_prometheus("# TYPE vllm:gpu_cache_usage_perc gauge\nvllm:gpu_cache_usage_perc 0.3\n")
    mapping, _ = resolve_names(set(d))
    assert mapping["kv_usage"] == "vllm:gpu_cache_usage_perc"


def test_window_deltas_and_replica_average():
    a = parse_prometheus(TEXT_NEW % {"q": 100, "h": 40, "c": 2, "s": 1.0})
    b = parse_prometheus(TEXT_NEW % {"q": 300, "h": 140, "c": 6, "s": 3.0})
    # two replicas summed per scrape
    two = [{k: 2 * v for k, v in a.items()}, {k: 2 * v for k, v in b.items()}]
    out, missing = summarize_scrapes(two, n_servers=2)
    assert out["kv_usage_max"] == 0.5              # averaged across replicas, not summed
    assert out["running_mean"] == 14               # summed across replicas
    assert out["prefix_queries"] == 400 and out["prefix_hits"] == 200
    assert out["prefix_hit_rate"] == 0.5
    assert out["queue_time_mean_s"] == 0.5
    assert "preemptions" in missing
