import numpy as np
import pytest

from harness.config import WorkloadCfg
from harness.regimes import WorkloadPoint
from harness.workload import FakeTokenizer, generate_trace, load_trace, save_trace

WL = WorkloadCfg(output_tokens=16, prefix_pool_size=4, prefix_fraction=0.75, trace_requests=200)


def _pt(ctx=256, reuse=0.5):
    return WorkloadPoint("center", ctx, 8, reuse, None, "mid")


def test_exact_lengths_fake():
    tr = generate_trace(_pt(300), WL, 1, FakeTokenizer())
    assert all(tr.prompt_len(i) == 300 for i in range(len(tr)))


@pytest.mark.parametrize("reuse", [0.0, 0.5, 0.9])
def test_reuse_fraction(reuse):
    tr = generate_trace(_pt(64, reuse), WL, 3, FakeTokenizer())
    assert tr.reuse_fraction == pytest.approx(reuse, abs=1 / len(tr))
    plen = int(round(0.75 * 64))
    for i in range(len(tr)):
        pid = tr.prefix_ids[i]
        if pid >= 0:
            # every pooled-prefix request shares the first plen tokens with the others in its pool slot
            j = int(np.flatnonzero(tr.prefix_ids == pid)[0])
            assert tr.prompt(i)[:plen] == tr.prompt(j)[:plen]
    assert len(set(tr.prefix_ids[tr.prefix_ids >= 0].tolist())) <= 4


def test_same_seed_same_hash_and_roundtrip(tmp_path):
    a = generate_trace(_pt(), WL, 42, FakeTokenizer())
    b = generate_trace(_pt(), WL, 42, FakeTokenizer())
    c = generate_trace(_pt(), WL, 43, FakeTokenizer())
    assert a.content_hash == b.content_hash != c.content_hash
    p = tmp_path / "t.npz"
    h = save_trace(a, p)
    loaded = load_trace(p)
    assert loaded.content_hash == h == a.content_hash
    assert loaded.prompt(5) == a.prompt(5)


@pytest.mark.tokenizer
def test_exact_lengths_real_tokenizer():
    """Prompt lengths are exact with a real HF tokenizer (small ungated fallback model)."""
    try:
        from harness.workload import HFTokenizer
        tok = HFTokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as e:  # offline CI
        pytest.skip(f"tokenizer unavailable: {e}")
    tr = generate_trace(_pt(512, 0.5), WorkloadCfg(output_tokens=16, trace_requests=20), 7, tok)
    for i in range(len(tr)):
        ids = tr.prompt(i)
        assert len(ids) == 512
        assert max(ids) < len(tok.tok)          # valid ids for this tokenizer
    # decoded text re-tokenizes to (nearly) the same ids; requests are sent as ids so length is exact
    assert tok.decode(tr.prompt(0))
