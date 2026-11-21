#!/usr/bin/env python3
"""Draw the hand-sketched explainer diagrams for the readable report.

Each figure is an SVG built from slightly wobbly strokes (fixed random seed, so the output is identical on every
run) and the Kalam handwriting font (docs/report_readable/fonts, SIL OFL). Numbers in the sketches are the measured
values quoted in the report; proportions in the "why" sketch are illustrative and labelled as such.

    python docs/report_readable/sketches/make_sketches.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent

INK = "#263238"
MUTED = "#5f6b73"
TEAL, TEAL_L = "#17756f", "#dcefec"
AMBER, AMBER_L = "#c7801a", "#fcefd6"
RED, RED_L = "#c0452f", "#f9e0da"
GREY_L = "#eef1f3"
BLUE, BLUE_L = "#2f5f8f", "#e2ecf6"


class Sketch:
    def __init__(self, w: int, h: int, seed: int = 7):
        self.w, self.h = w, h
        self.r = random.Random(seed)
        self.parts: list[str] = []

    # ---- primitives --------------------------------------------------------------------------------------------
    def j(self, v: float, a: float = 1.1) -> float:
        return v + self.r.uniform(-a, a)

    def line(self, x1, y1, x2, y2, color=INK, w=1.7, passes=2, dash=None):
        for p in range(passes):
            bow = self.r.uniform(-1.6, 1.6)
            mx, my = (x1 + x2) / 2 + bow, (y1 + y2) / 2 - bow
            d = f"M{self.j(x1):.1f},{self.j(y1):.1f} Q{mx:.1f},{my:.1f} {self.j(x2):.1f},{self.j(y2):.1f}"
            da = f' stroke-dasharray="{dash}"' if dash else ""
            self.parts.append(
                f'<path d="{d}" stroke="{color}" stroke-width="{w if p == 0 else w * 0.55:.2f}" fill="none" '
                f'stroke-linecap="round" opacity="{1 if p == 0 else 0.5}"{da}/>')

    def rect(self, x, y, w, h, stroke=INK, fill=None, sw=1.7, dash=None):
        if fill:
            pts = [(x + 2, y + 2), (x + w - 2, y + 1), (x + w - 1, y + h - 2), (x + 1, y + h - 1)]
            d = "M" + " L".join(f"{self.j(px, 1.5):.1f},{self.j(py, 1.5):.1f}" for px, py in pts) + " Z"
            self.parts.append(f'<path d="{d}" fill="{fill}" stroke="none"/>')
        for (a, b, c, e) in ((x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)):
            self.line(a, b, c, e, stroke, sw, dash=dash)

    def ellipse(self, cx, cy, rx, ry, stroke=INK, fill=None, sw=1.7):
        for p in range(2):
            pts = []
            n = 28
            start = self.r.uniform(0, 2 * math.pi)
            for i in range(n + 3):
                t = start + 2 * math.pi * i / n
                k = 1 + self.r.uniform(-0.015, 0.015)
                pts.append((cx + rx * k * math.cos(t), cy + ry * k * math.sin(t)))
            d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f} " + " ".join(f"L{px:.1f},{py:.1f}" for px, py in pts[1:])
            f = fill if (fill and p == 0) else "none"
            self.parts.append(f'<path d="{d}" fill="{f}" stroke="{stroke}" stroke-width="{sw if p == 0 else sw * 0.5:.2f}" '
                              f'opacity="{1 if p == 0 else 0.5}" stroke-linejoin="round"/>')

    def arrow(self, x1, y1, x2, y2, color=INK, w=1.8, curve=0.0, label=None, lx=0, ly=-6, dash=None):
        if curve:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = -(y2 - y1), (x2 - x1)
            L = math.hypot(nx, ny) or 1
            cx, cy = mx + curve * nx / L, my + curve * ny / L
            for p in range(2):
                d = f"M{self.j(x1):.1f},{self.j(y1):.1f} Q{self.j(cx, 2):.1f},{self.j(cy, 2):.1f} {x2:.1f},{y2:.1f}"
                da = f' stroke-dasharray="{dash}"' if dash else ""
                self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{w if p == 0 else w * 0.55:.2f}" '
                                  f'fill="none" stroke-linecap="round" opacity="{1 if p == 0 else 0.5}"{da}/>')
            ang = math.atan2(y2 - cy, x2 - cx)
        else:
            self.line(x1, y1, x2, y2, color, w, dash=dash)
            ang = math.atan2(y2 - y1, x2 - x1)
        for s in (-1, 1):
            a = ang + math.pi + s * 0.45
            self.line(x2, y2, x2 + 11 * math.cos(a), y2 + 11 * math.sin(a), color, w, passes=1)
        if label:
            self.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, 12.5, color=MUTED, anchor="middle")

    def text(self, x, y, s, size=15.0, weight=400, color=INK, anchor="start", italic=False):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Kalam" font-size="{size}" font-weight="{weight}" '
                          f'fill="{color}" text-anchor="{anchor}"{st}>{s}</text>')

    def lines(self, x, y, rows, size=14.0, gap=1.25, **kw):
        for i, r in enumerate(rows):
            self.text(x, y + i * size * gap, r, size, **kw)

    def save(self, name: str):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
               f'height="{self.h}">' + "".join(self.parts) + "</svg>")
        (OUT / name).write_text(svg)
        print("wrote", name)


# ---- 1. how a request is served -----------------------------------------------------------------------------------
def request_lifecycle():
    s = Sketch(760, 332, seed=11)
    s.rect(10, 92, 118, 92, INK, GREY_L)
    s.lines(22, 120, ["Your prompt", "e.g. 2,048", "tokens"], 15, weight=700)
    s.arrow(132, 138, 172, 138)

    s.rect(178, 52, 196, 166, AMBER, AMBER_L, 2)
    s.text(276, 80, "1. PREFILL", 19, 700, AMBER, "middle")
    s.lines(192, 108, ["reads the whole prompt", "in one big pass", "lots of maths at once", "(compute-bound)"], 14.5)

    s.arrow(380, 138, 420, 138)
    s.rect(426, 52, 210, 166, TEAL, TEAL_L, 2)
    s.text(531, 80, "2. DECODE", 19, 700, TEAL, "middle")
    s.lines(440, 108, ["writes ONE token per step", "and every step reads all", "8 billion weights again", "(memory-bound)"], 14.5)
    s.arrow(606, 50, 456, 50, TEAL, 1.6, curve=26)
    s.text(531, 16, "repeat for each of the 256 output tokens", 13.5, color=TEAL, anchor="middle")

    s.arrow(642, 138, 676, 138)
    s.rect(682, 102, 70, 72, INK, GREY_L)
    s.text(717, 144, "Answer", 15, 700, anchor="middle")

    s.rect(196, 236, 160, 52, BLUE, BLUE_L)
    s.lines(206, 256, ["KV cache = the model's", "notes on your prompt"], 13.5, color=BLUE)
    s.arrow(276, 222, 276, 234, BLUE, 1.5)
    s.arrow(358, 262, 470, 222, BLUE, 1.5, curve=-14)
    s.text(452, 268, "decode re-reads these notes every step", 13, color=BLUE)

    s.line(12, 304, 426, 304, MUTED, 1.2, dash="4 4")
    s.text(12, 324, "TTFT = time to first token (p99 gate: 5 s)", 13, color=MUTED)
    s.line(430, 304, 636, 304, MUTED, 1.2, dash="1 4")
    s.text(430, 324, "TPOT = time per token after that (p99 gate: 100 ms)", 13, color=MUTED)
    s.save("request_lifecycle.svg")


# ---- 2. the five switches -----------------------------------------------------------------------------------------
def five_switches():
    s = Sketch(782, 300, seed=23)
    cards = [
        ("FP8 weights", "quant", AMBER, AMBER_L,
         ["keep the model's", "numbers in 8 bits", "instead of 16"], ["half the bytes to", "read, faster maths"]),
        ("FP8 KV cache", "kvq", BLUE, BLUE_L,
         ["keep the notes", "(KV cache) in 8 bits", "instead of 16"], ["less memory traffic", "on long prompts"]),
        ("EAGLE-3", "spec", TEAL, TEAL_L,
         ["a small helper guesses", "the next 3 tokens; the", "model checks all 3", "in one pass"], ["fewer slow", "decode steps"]),
        ("Prefix caching", "prefix", "#6a4c93", "#ece6f4",
         ["prompt starts like an", "earlier one? reuse", "those notes"], ["skips repeated", "prefill work"]),
        ("Tuned batching", "batch", RED, RED_L,
         ["change how many", "requests run together", "(max_num_seqs)"], ["only matters if the", "limit is ever hit"]),
    ]
    w, gap, x0 = 148, 9, 6
    for i, (name, tag, c, cl, how, gain) in enumerate(cards):
        x = x0 + i * (w + gap)
        s.rect(x, 18, w, 238, c, cl, 2)
        s.ellipse(x + 22, 40, 13, 13, c, "#ffffff", 1.6)
        s.text(x + 22, 46, str(i + 1), 15, 700, c, "middle")
        s.text(x + 40, 46, name, 14, 700, c)
        s.text(x + 12, 72, f"switch: {tag}", 12.5, color=MUTED)
        s.lines(x + 12, 98, how, 12.6, gap=1.2)
        s.line(x + 12, 160, x + w - 12, 160, c, 1.1, passes=1, dash="3 4")
        s.text(x + 12, 183, "what it saves:", 12.5, 700, c)
        s.lines(x + 12, 205, gain, 12.8)
    s.text(391, 286, "Default = all five OFF (and vLLM's own prefix caching and chunked prefill switched off too)",
           13.5, color=MUTED, anchor="middle")
    s.save("five_switches.svg")


# ---- 3. the seven workloads ---------------------------------------------------------------------------------------
def workload_map():
    s = Sketch(780, 400, seed=5)
    cx, cy = 390, 190
    sats = [  # (x, y, title, setting, best, config, color)
        (10, 34, "LOW LOAD", "2 requests at once", "2.48×", "spec + prefix", TEAL),
        (566, 34, "HIGH LOAD", "128 requests at once", "1.67×", "spec + prefix", TEAL),
        (10, 156, "SHORT PROMPTS", "256 tokens", "1.93×", "spec + prefix", AMBER),
        (566, 156, "LONG PROMPTS", "8,192 tokens", "1.50×", "prefix + batch", AMBER),
        (10, 278, "NO SHARING", "0% share a prefix", "1.39×", "spec + prefix", "#6a4c93"),
        (566, 278, "HEAVY SHARING", "90% share a prefix", "2.62×", "spec+prefix+batch", "#6a4c93"),
    ]
    for (x, y, title, setting, best, cfg, c) in sats:
        ex = x + 204 if x < cx else x
        s.line(cx + (-120 if x < cx else 120), cy + (y + 36 - cy) * 0.35, ex, y + 36, c, 1.4, passes=1, dash="5 4")
        s.rect(x, y, 204, 74, c, "#ffffff", 1.8)
        s.text(x + 10, y + 22, title, 15, 700, c)
        s.text(x + 10, y + 42, setting, 13.5)
        s.text(x + 10, y + 63, f"best {best}  ({cfg})", 12.6, 700)
    s.ellipse(cx, cy, 128, 74, INK, GREY_L, 2)
    s.text(cx, cy - 36, "CENTRE", 17, 700, anchor="middle")
    s.lines(cx, cy - 13, ["2,048-token prompts", "32 requests at once", "50% share a prefix"], 13.5, anchor="middle")
    s.text(cx, cy + 50, "best 1.82×  (spec + prefix)", 13.5, 700, anchor="middle")
    s.text(cx, 22, "turn ONE dial at a time; keep the other two at the centre", 13.5, color=MUTED, anchor="middle")
    s.text(cx, 388, "best = fastest setup passing every gate, as a multiple of Default's throughput", 13,
           color=MUTED, anchor="middle")
    s.save("workload_map.svg")


# ---- 4. the intruder on GPU 0 -------------------------------------------------------------------------------------
def intruder():
    s = Sketch(760, 250, seed=31)
    # GPU 1 (clean)
    s.rect(30, 30, 300, 150, TEAL, TEAL_L, 2)
    s.text(46, 56, "GPU 1", 17, 700, TEAL)
    s.rect(52, 74, 256, 62, INK, "#ffffff")
    s.text(180, 111, "our vLLM server", 15, anchor="middle")
    s.text(180, 168, "speed vs. clean re-run: 1.006×", 14.5, 700, TEAL, "middle")
    # GPU 0 (shared)
    s.rect(410, 30, 320, 150, RED, RED_L, 2)
    s.text(426, 56, "GPU 0  (from 11:24 UTC, 7 Nov)", 17, 700, RED)
    s.rect(432, 74, 170, 62, INK, "#ffffff")
    s.text(517, 111, "our vLLM server", 15, anchor="middle")
    s.rect(614, 74, 100, 62, RED, "#ffffff", 1.6, dash="5 4")
    s.lines(664, 97, ["someone", "else's job", "+4.3 GB"], 12.5, color=RED, anchor="middle")
    s.text(570, 168, "speed vs. clean re-run: 0.596×", 14.5, 700, RED, "middle")
    s.lines(30, 210, ["Same node, same time. 71 of our 384 phase-1 runs (18%) landed on GPU 0 while it was shared.",
                      "The repeats inside each run still agreed with each other (spread 0.2-3.8%), so it looked like normal noise."],
            13.5, color=MUTED)
    s.save("intruder.svg")


# ---- 5. one measurement, end to end (architecture) ----------------------------------------------------------------
def measurement():
    s = Sketch(770, 360, seed=41)
    s.rect(8, 8, 548, 196, RED, None, 1.3, dash="7 5")
    s.text(18, 28, "inside one Slurm job on a shared GPU node (GPUs not isolated between users)", 13, 700, RED)
    top = [
        ("1  pick a combination", ["7 on/off switches,", "e.g. 0011000", "= spec + prefix"], INK, GREY_L),
        ("2  start vLLM", ["exactly those flags;", "one server per GPU;", "restart only on change"], TEAL, TEAL_L),
        ("3  replay requests", ["the same recorded,", "hashed trace for every", "combination"], AMBER, AMBER_L),
    ]
    for i, (t, rows, c, cl) in enumerate(top):
        x = 20 + i * 178
        s.rect(x, 44, 158, 92, c, cl, 1.9)
        s.text(x + 10, 66, t, 14.5, 700, c)
        s.lines(x + 10, 88, rows, 12.8)
        if i < 2:
            s.arrow(x + 160, 90, x + 176, 90)
    s.rect(20, 150, 524, 44, BLUE, BLUE_L, 1.6)
    s.text(32, 177, "4  measure: 30 s warm-up + 120 s;  tokens/s, TTFT, TPOT;  2-3 repeats", 13.8,
           700, BLUE)
    s.arrow(456, 136, 456, 148, INK, 1.4)
    # side: telemetry + quality
    s.rect(578, 44, 182, 70, INK, "#ffffff", 1.6)
    s.lines(590, 66, ["watch the hardware", "GPU clocks, power, memory;", "vLLM's own metrics"], 12.8)
    s.rect(578, 126, 182, 70, INK, "#ffffff", 1.6)
    s.lines(590, 148, ["check the answers", "GSM8K, HumanEval, MMLU", "on the same server"], 12.8)
    s.arrow(578, 80, 538, 80, MUTED, 1.4)
    s.arrow(578, 160, 548, 170, MUTED, 1.4)
    # bottom row
    bottom = [
        ("5  record everything", ["runs.jsonl: versions, exact", "flags, results, telemetry"], INK, GREY_L),
        ("6  analyse", ["speedups, gates, effects,", "interactions, best setup"], TEAL, TEAL_L),
        ("7  check (no GPU)", ["289 number checks,", "69 unit tests, dry run"], "#3f7d46", "#e3f0e4"),
    ]
    for i, (t, rows, c, cl) in enumerate(bottom):
        x = 20 + i * 256
        s.rect(x, 246, 228, 74, c, cl, 1.9)
        s.text(x + 10, 270, t, 14.5, 700, c)
        s.lines(x + 10, 292, rows, 12.8)
        if i < 2:
            s.arrow(x + 230, 283, x + 254, 283)
    s.arrow(280, 196, 134, 244, INK, 1.6, curve=10)
    s.text(385, 345, "steps 1-5 ran 510 times on 2 × H100 NVL; steps 6-7 run on any laptop", 13, color=MUTED,
           anchor="middle")
    s.save("measurement.svg")


# ---- 6. why prefix caching helps more with speculation (interpretation) -------------------------------------------
def why_compound():
    s = Sketch(760, 250, seed=53)
    s.text(10, 22, "time to serve a batch of work (a sketch, not to scale)", 13.5, color=MUTED)
    rows = [
        ("plain", [(0.36, AMBER_L, AMBER, "prompt work"), (0.64, TEAL_L, TEAL, "decode work")]),
        ("+ EAGLE-3", [(0.36, AMBER_L, AMBER, "prompt work"), (0.26, TEAL_L, TEAL, "decode")]),
        ("+ EAGLE-3 + prefix", [(0.10, AMBER_L, AMBER, ""), (0.26, TEAL_L, TEAL, "decode")]),
    ]
    full = 470
    for i, (label, segs) in enumerate(rows):
        y = 44 + i * 58
        s.text(10, y + 27, label, 14.5, 700)
        x = 168
        for frac, fill, c, t in segs:
            w = full * frac
            s.rect(x, y, w, 40, c, fill, 1.6)
            if t:
                s.text(x + w / 2, y + 26, t, 13.5, anchor="middle")
            x += w
    s.lines(664, 96, ["prompt work is", "a bigger share", "of what's left,", "so removing it", "helps more"], 12.8,
            color=MUTED)
    s.text(10, 236, "Measured at 90% reuse: prefix caching gives 1.52× alone but 1.85× on top of EAGLE-3.", 13.5, 700)
    s.save("why_compound.svg")


if __name__ == "__main__":
    request_lifecycle()
    five_switches()
    workload_map()
    intruder()
    measurement()
    why_compound()
