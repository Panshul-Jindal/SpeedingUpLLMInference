import json
from pathlib import Path

import pytest

from harness import optimizations as opt
from harness.config import REPO_ROOT
from harness.engine.vllm_flags import REQUIRED_FLAGS, build_launch, missing_flags, parse_layout


def _argv(launch, i=0):
    return list(launch.servers[i].argv)


def _val(argv, flag):
    return argv[argv.index(flag) + 1]


def test_default_explicit_off_flags(env_main, exp_full):
    l = build_launch(0, env_main, exp_full, point_id="center")
    a = _argv(l)
    assert "--no-enable-prefix-caching" in a and "--enable-prefix-caching" not in a
    assert "--no-enable-chunked-prefill" in a
    assert _val(a, "--max-num-batched-tokens") == "16384"
    assert _val(a, "--tensor-parallel-size") == "4"
    assert _val(a, "--kv-cache-dtype") == "auto"
    assert "--quantization" not in a and "--speculative-config" not in a and "--max-num-seqs" not in a
    assert len(l.servers) == 1 and l.servers[0].env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_default_single_gpu(env_test, exp_smoke):
    a = _argv(build_launch(0, env_test, exp_smoke, point_id="center"))
    assert _val(a, "--tensor-parallel-size") == "1"
    assert _val(a, "--max-model-len") == "4096"


def test_all_on_flags(env_test, exp_smoke):
    mask = opt.from_bits("1111110")
    l = build_launch(mask, env_test, exp_smoke, point_id="center")
    a = _argv(l)
    assert _val(a, "--quantization") == "fp8"
    assert _val(a, "--kv-cache-dtype") == "fp8"
    spec = json.loads(_val(a, "--speculative-config"))
    assert spec["method"] == "ngram" and spec["num_speculative_tokens"] == 4
    assert "--enable-prefix-caching" in a
    assert _val(a, "--max-num-seqs") == "32"            # smoke placeholder
    assert "--enable-chunked-prefill" in a and _val(a, "--max-num-batched-tokens") == "1024"
    assert l.uses_placeholders


def test_eagle3_spec_config(env_main, exp_full):
    a = _argv(build_launch(opt.bit(3), env_main, exp_full, point_id="center"))
    spec = json.loads(_val(a, "--speculative-config"))
    assert spec["method"] == "eagle3" and spec["model"] == env_main.models.eagle3_draft


def test_tuned_values_override_placeholders(env_test, exp_smoke):
    tuned = {"test": {"center": {"max_num_seqs": 64}}}
    l = build_launch(opt.bit(5), env_test, exp_smoke, point_id="center", tuned=tuned)
    assert _val(_argv(l), "--max-num-seqs") == "64"
    assert l.tuned["max_num_seqs"]["source"] == "tuned" and not l.uses_placeholders


@pytest.mark.parametrize("layout,cvds,ports", [
    ("tp2x2", ["0,1", "2,3"], [8100, 8101]),
    ("tp1x4", ["0", "1", "2", "3"], [8100, 8101, 8102, 8103]),
    ("tp4", ["0,1,2,3"], [8100]),
    ("tp2pp2", ["0,1,2,3"], [8100]),
])
def test_replica_layouts(env_main, exp_full, layout, cvds, ports):
    l = build_launch(opt.bit(7), env_main, exp_full, point_id="center", overrides={"parallel_layout": layout})
    assert [s.env["CUDA_VISIBLE_DEVICES"] for s in l.servers] == cvds
    assert [s.port for s in l.servers] == ports
    for s in l.servers:
        a = list(s.argv)
        assert _val(a, "--port") == str(s.port)
    lay = parse_layout(layout)
    a = _argv(l)
    assert _val(a, "--tensor-parallel-size") == str(lay.tp)
    if lay.pp > 1:
        assert _val(a, "--pipeline-parallel-size") == str(lay.pp)


def test_parallelism_needs_gpus(env_test, exp_smoke):
    with pytest.raises(ValueError):
        build_launch(opt.bit(7), env_test, exp_smoke, point_id="center")


def test_flags_key_ignores_rep_but_tracks_flags(env_test, exp_smoke):
    a = build_launch(0, env_test, exp_smoke, point_id="center")
    b = build_launch(0, env_test, exp_smoke, point_id="context_high")
    c = build_launch(opt.bit(4), env_test, exp_smoke, point_id="center")
    assert a.flags_key == b.flags_key != c.flags_key


def test_required_flags_in_pinned_help():
    help_files = sorted((REPO_ROOT / "docs").glob("vllm_serve_help_*.txt"))
    if not help_files:
        pytest.skip("no saved vllm help text")
    text = help_files[-1].read_text()
    assert missing_flags(text) == []
    assert missing_flags("--port --host") != []
    assert "--seed" in REQUIRED_FLAGS
