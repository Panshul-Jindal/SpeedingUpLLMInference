"""Request traces: token-exact synthetic prompts, a pooled shared prefix, saved with seed + content hash.

Prompts are stored and sent as token IDs, so the server sees exactly `context` prompt tokens.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from harness.config import WorkloadCfg
from harness.regimes import WorkloadPoint

TRACE_FORMAT_VERSION = 1

# Seeded synthetic text source: random sequences over a fixed word list.
_WORDS = (
    "the of and to in a is that for it as was with be by on not he this are or his from at which but have an "
    "they you were her she there been one all we their has would when if so no will can more other its into "
    "time about some only new could these two may first then do any like my now over such our man me even "
    "most made after also did many before must through back years where much your way well down should "
    "because each just those people how too little state good very make world still own see men work long "
    "get here between both life being under never day same another know while last might us great old year "
    "off come since against go came right used take three river garden window market silver engine theory "
    "harbor signal forest valley planet copper lantern meadow canvas ladder pocket thunder whisper orbit "
    "bridge castle desert falcon glacier harvest island jungle kettle lemon marble needle ocean pepper quartz "
    "rocket saddle tunnel umbrella velvet walnut yellow zebra anchor bamboo candle dragon feather galaxy "
    "hammer insect jacket kitten ladle magnet nectar olive parrot quiver ribbon sphinx tiger violin wagon"
).split()


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class HFTokenizer:
    """The model's own tokenizer (no special tokens added)."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return str(self.tok.decode(ids))


class FakeTokenizer:
    """Deterministic word-level tokenizer for dry runs (no download)."""
    name = "fake-word-tokenizer"

    def encode(self, text: str) -> list[int]:
        return [zlib.crc32(w.encode()) % 32000 + 100 for w in text.split()]


@dataclass
class Trace:
    point_id: str
    seed: int
    tokens: np.ndarray        # flat int32
    offsets: np.ndarray       # len n+1
    prefix_ids: np.ndarray    # -1 = no pooled prefix
    output_tokens: int
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def prompt(self, i: int) -> list[int]:
        return self.tokens[self.offsets[i]:self.offsets[i + 1]].tolist()

    def prompt_len(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    @property
    def content_hash(self) -> str:
        return trace_hash(self.tokens, self.offsets, self.prefix_ids, self.output_tokens)

    @property
    def reuse_fraction(self) -> float:
        return float((self.prefix_ids >= 0).mean()) if len(self) else 0.0


def trace_hash(tokens: np.ndarray, offsets: np.ndarray, prefix_ids: np.ndarray, output_tokens: int) -> str:
    h = hashlib.sha256()
    for arr in (tokens.astype(np.int32), offsets.astype(np.int64), prefix_ids.astype(np.int32)):
        h.update(arr.tobytes())
    h.update(str(output_tokens).encode())
    return h.hexdigest()


def point_seed(seed: int, point_id: str) -> int:
    return (seed * 1_000_003 + zlib.crc32(point_id.encode())) % (2**32)


class _TextSource:
    """Seeded synthetic token source. Extension point for ShareGPT / long-document sources."""

    def __init__(self, tokenizer: Tokenizer, rng: np.random.Generator):
        self.tok = tokenizer
        self.rng = rng

    def tokens(self, n: int) -> list[int]:
        if n <= 0:
            return []
        out: list[int] = []
        while len(out) < n:
            words = self.rng.choice(_WORDS, size=max(16, int((n - len(out)) * 1.1) + 8))
            out.extend(self.tok.encode(" " + " ".join(words)))
        return out[:n]


def generate_trace(point: WorkloadPoint, wl: WorkloadCfg, seed: int, tokenizer: Tokenizer,
                   n_requests: int | None = None) -> Trace:
    n = n_requests or wl.trace_requests
    s = point_seed(seed, point.point_id)
    rng = np.random.default_rng(s)
    src = _TextSource(tokenizer, rng)
    length = int(point.context)
    plen = int(round(wl.prefix_fraction * length))
    pool = [src.tokens(plen) for _ in range(wl.prefix_pool_size)]

    n_reuse = int(round(point.reuse * n))
    reuse_idx = set(rng.choice(n, size=n_reuse, replace=False).tolist()) if n_reuse else set()
    prompts: list[list[int]] = []
    prefix_ids = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        if i in reuse_idx:
            pid = int(rng.integers(wl.prefix_pool_size))
            prefix_ids[i] = pid
            prompts.append(pool[pid] + src.tokens(length - plen))
        else:
            prompts.append(src.tokens(length))
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in prompts])
    tokens = np.fromiter((t for p in prompts for t in p), dtype=np.int32, count=int(offsets[-1]))
    meta = {"format": TRACE_FORMAT_VERSION, "point_id": point.point_id, "seed": seed, "point_seed": s,
            "context": length, "concurrency": point.concurrency, "reuse": point.reuse,
            "prefix_pool_size": wl.prefix_pool_size, "prefix_fraction": wl.prefix_fraction,
            "prefix_tokens": plen, "output_tokens": wl.output_tokens, "source": wl.source,
            "tokenizer": getattr(tokenizer, "name", type(tokenizer).__name__), "n_requests": n}
    return Trace(point.point_id, seed, tokens, offsets, prefix_ids, wl.output_tokens, meta)


def save_trace(trace: Trace, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**trace.meta, "sha256": trace.content_hash}
    np.savez(path, tokens=trace.tokens, offsets=trace.offsets, prefix_ids=trace.prefix_ids,
             meta=np.array(json.dumps(meta)))
    return meta["sha256"]


def load_trace(path: Path) -> Trace:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        tr = Trace(meta["point_id"], meta["seed"], z["tokens"], z["offsets"], z["prefix_ids"],
                   meta["output_tokens"], meta)
    if tr.content_hash != meta["sha256"]:
        raise ValueError(f"trace {path} content hash mismatch (corrupted or edited)")
    return tr


def trace_path(results_dir: Path, env_name: str, point_id: str) -> Path:
    return results_dir / "traces" / f"{env_name}__{point_id}.npz"


def ensure_trace(results_dir: Path, env_name: str, point: WorkloadPoint, wl: WorkloadCfg, seed: int,
                 tokenizer_factory: Any, force: bool = False) -> Trace:
    """Load the saved trace, or generate and save it. Regenerates if the settings changed."""
    path = trace_path(results_dir, env_name, point.point_id)
    if path.exists() and not force:
        tr = load_trace(path)
        m = tr.meta
        if (m["context"], m["reuse"], m["output_tokens"], m["seed"], m["n_requests"],
                m["prefix_pool_size"], m["prefix_fraction"]) == (
                int(point.context), point.reuse, wl.output_tokens, seed, wl.trace_requests,
                wl.prefix_pool_size, wl.prefix_fraction):
            return tr
    tr = generate_trace(point, wl, seed, tokenizer_factory())
    save_trace(tr, path)
    return tr
