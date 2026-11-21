"""Figures for the report pack (PNG, 16:9-friendly, light theme). Reads only report_pack/data/*.csv.

Palette: the dataviz skill's validated categorical order (light mode), recessive grid, text in ink tokens.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CRITICAL = "#d03b3b"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POINT_LABEL = {"center": "centre", "context_low": "short\nprompts", "context_high": "long\nprompts",
               "concurrency_low": "low\nconcurrency", "concurrency_high": "high\nconcurrency",
               "reuse_low": "no prefix\nreuse", "reuse_high": "90% prefix\nreuse"}
POINT_ORDER = list(POINT_LABEL)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
})


def _finish(fig: plt.Figure, path: Path, note: str | None = None) -> None:
    if note:   # below everything (bbox_inches="tight" keeps it), never over the axis labels
        fig.text(0.01, -0.02, note, color=MUTED, fontsize=9.5, ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_contamination(d: Path, out: Path) -> None:
    m = pd.read_csv(d / "contamination_matched_runs.csv", parse_dates=["written_at"])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for gpu, color, label in ((1, ORANGE, "GPU 1"), (0, BLUE, "GPU 0")):
        g = m[m.gpu == gpu]
        ax.scatter(g.written_at, g.ratio_fast_to_rerun, s=46, color=color, edgecolor=SURFACE, linewidth=1.2,
                   label=label, zorder=3)
    cut = pd.Timestamp("2026-11-07T11:23:30Z")
    ax.axvline(cut, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(cut, 1.08, "  11:24 UTC: another process\n  appears on GPU 0", color=INK2, fontsize=11, va="top")
    ax.axhline(1.0, color=AXIS, linewidth=1.2)
    ax.set_ylim(0.45, 1.12)
    ax.set_ylabel("fast-plan throughput / clean re-run")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax.set_xlabel("time of run, 7 Nov 2026 (UTC)")
    ax.set_title("From 11:24, every run on GPU 0 lost ~40% of its throughput")
    ax.legend(loc="lower left")
    _finish(fig, out / "01_contamination_timeline.png",
            "Each dot is one fast-plan run whose cell was re-run cleanly later (same config, point and trace). "
            "GPU 1 stayed at 1.00x; GPU 0 dropped to 0.56-0.69x.")


def fig_corrected(d: Path, out: Path, reported: dict[str, float]) -> None:
    f = pd.read_csv(d / "final_configs_by_policy.csv", dtype={"best_config": str})
    f = f[f.policy.str.startswith("A")].set_index("point_id")
    pts = POINT_ORDER
    x = np.arange(len(pts))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    rep = [reported[p] for p in pts]
    cor = [f.loc[p, "speedup"] for p in pts]
    b1 = ax.bar(x - w / 2 - 0.01, rep, w, color=AXIS, label="as first reported (contaminated)", zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, cor, w, color=BLUE, label="corrected", zorder=3)
    for bars, vals, col in ((b1, rep, INK2), (b2, cor, INK)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}x", ha="center", color=col, fontsize=10.5)
    ax.axhline(1.0, color=INK2, linewidth=1.0)
    ax.set_xticks(x, [POINT_LABEL[p] for p in pts])
    ax.set_ylabel("best speedup over Default")
    ax.set_ylim(0, max(rep) * 1.15)
    ax.set_title("Correcting the contaminated cells cuts the headline gains at 3 of 7 workloads")
    ax.legend(loc="upper left")
    _finish(fig, out / "02_best_speedup_corrected.png",
            "Best configuration per workload under the original accuracy policy (trimmed evals, 1-point gate); "
            "latency gates p99 TPOT <= 100 ms, p99 TTFT <= 5 s.")


def fig_eagle(d: Path, out: Path) -> None:
    c = pd.read_csv(d / "spec_vs_context.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.28})
    for col, color, lab in (("accept_pos1", BLUE, "1st draft token"), ("accept_pos2", ORANGE, "2nd draft token"),
                            ("accept_pos3", AQUA, "3rd draft token")):
        a1.plot(c.context_tokens, c[col], color=color, linewidth=2, marker="o", markersize=7, label=lab)
    a1.legend(loc="upper right")
    a1.set_xlim(1800, 8450)
    a1.set_ylim(-0.03, 1.0)
    a1.set_xticks(c.context_tokens)
    a1.set_xlabel("prompt length (tokens)")
    a1.set_ylabel("acceptance rate")
    a1.set_title("EAGLE-3 acceptance by draft position")
    a2.plot(c.context_tokens, c.spec_speedup, color=BLUE, linewidth=2, marker="o", markersize=7)
    for xx, yy in zip(c.context_tokens, c.spec_speedup):
        a2.text(xx, yy + 0.02, f"{yy:.2f}x", ha="center", color=INK, fontsize=10.5)
    a2.axhline(1.0, color=INK2, linewidth=1.0)
    a2.set_xticks(c.context_tokens)
    a2.set_ylim(0.9, 1.45)
    a2.set_xlabel("prompt length (tokens)")
    a2.set_ylabel("throughput, spec / Default")
    a2.set_title("Speculative decoding speedup")
    _finish(fig, out / "03_eagle3_long_context.png",
            "32 requests in flight, 50% prefix reuse, 256 output tokens; 2 reps per cell (rep-to-rep variation <= 1.6%).")


def fig_reuse(d: Path, out: Path) -> None:
    r = pd.read_csv(d / "prefix_vs_reuse.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pct = r.reuse * 100
    ax.plot(pct, r.prefix_gain, color=BLUE, linewidth=2, marker="o", markersize=7, label="prefix caching alone")
    ax.plot(pct, r.prefix_gain_with_spec, color=ORANGE, linewidth=2, marker="o", markersize=7,
            label="prefix caching on top of spec")
    for col, color in (("prefix_gain", BLUE), ("prefix_gain_with_spec", ORANGE)):
        ax.text(pct.iloc[-1] + 2, r[col].iloc[-1], f"{r[col].iloc[-1]:.2f}x", color=INK, va="center", fontsize=11)
    ax.axhline(1.0, color=INK2, linewidth=1.0)
    ax.set_xlim(-3, 100)
    ax.set_xlabel("share of requests that reuse a cached prefix (%)")
    ax.set_ylabel("throughput gain from prefix caching")
    ax.set_title("Prefix caching is worth more when speculative decoding is on")
    ax.legend(loc="upper left")
    _finish(fig, out / "04_prefix_gain_vs_reuse.png",
            "Spec x prefix interaction: 1.01 (0%), 1.02 (10%), 1.01 (25%), 1.12 (50%), 1.22 (90%). "
            "2048-token prompts, 32 in flight.")


def fig_composition(d: Path, out: Path) -> None:
    c = pd.read_csv(d / "composition_per_config.csv", dtype={"config_bits": str})
    c = c[(c.n_opts >= 2) & (c.independent_basis == "product of measured singles")]
    spec = c.config_bits.str[2] == "1"
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    lim = (0.95, max(c.speedup_observed.max(), c.speedup_independent.max()) * 1.08)
    ax.plot(lim, lim, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(lim[1] * 0.97, lim[1] * 0.93, "no interaction", color=INK2, ha="right", fontsize=10.5, rotation=36)
    ax.scatter(c[~spec].speedup_independent, c[~spec].speedup_observed, s=60, color=BLUE, edgecolor=SURFACE,
               linewidth=1.5, label="combinations without spec", zorder=3)
    ax.scatter(c[spec].speedup_independent, c[spec].speedup_observed, s=60, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.5, label="combinations with spec", zorder=3)
    t = c.assign(r=c.ratio_observed_to_independent).sort_values("r", ascending=False).iloc[0]
    ax.annotate(f"{t.config}\n@ {POINT_LABEL[t.point_id].replace(chr(10), ' ')}: +{(t.r - 1) * 100:.0f}% over prediction",
                (t.speedup_independent, t.speedup_observed), xytext=(-150, -10), textcoords="offset points",
                ha="right", va="center", fontsize=10, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 1})
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted: product of the single-optimization speedups")
    ax.set_ylabel("observed speedup")
    ax.set_title("Combinations compound at least multiplicatively;\nspec combinations compound more")
    ax.legend(loc="lower right")
    _finish(fig, out / "05_composition_observed_vs_predicted.png",
            f"{len(c)} multi-optimization cells at points where every single optimization was measured "
            "(centre, high concurrency, 90% and 0% reuse).")


def fig_main_effects(d: Path, out: Path) -> None:
    m = pd.read_csv(d / "clean_main_effects.csv")
    col = "multiplier" if "multiplier" in m else [c for c in m.columns if "mult" in c][0]
    piv = m.pivot_table(index="name", columns="point_id", values=col).reindex(
        ["quant", "kvq", "spec", "prefix", "batch"])[POINT_ORDER]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    norm = TwoSlopeNorm(vmin=0.85, vcenter=1.0, vmax=1.9)
    ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=11.5,
                    color="#ffffff" if v > 1.55 else INK)
    ax.set_xticks(range(len(POINT_ORDER)), [POINT_LABEL[p] for p in POINT_ORDER])
    ax.set_yticks(range(5), ["FP8 weights (quant)", "FP8 KV cache (kvq)", "EAGLE-3 spec", "prefix caching",
                             "tuned batching"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK2)
    ax.set_title("Main effect of each optimization by workload (throughput multiplier, clean data)")
    _finish(fig, out / "06_main_effects_heatmap.png",
            "OLS on log throughput with +-1 coding, averaged over the other optimizations; blue = faster, red = slower.")


def fig_accuracy(d: Path, out: Path) -> None:
    p = pd.read_csv(d / "quality_paired_vs_default.csv", dtype={"setting": str})
    p = p.dropna(subset=["diff_pts"])
    lab = {"100": "FP8 weights", "010": "FP8 KV", "110": "FP8 weights + FP8 KV", "001": "EAGLE-3 spec"}
    p["row"] = p.setting.map(lab) + " | " + p.task.map({"gsm8k": "GSM8K", "humaneval": "HumanEval", "mmlu": "MMLU"})
    p = p.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for i, r in p.iterrows():
        bad = r.verdict == "significant drop beyond gate"
        color = CRITICAL if bad else BLUE
        ax.plot([r.ci95_lo, r.ci95_hi], [i, i], color=color, linewidth=2.2, solid_capstyle="round")
        ax.scatter([r.diff_pts], [i], s=64, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if bad:
            ax.text(r.ci95_hi + 0.25, i, f"significant (p = {r.mcnemar_p:.2f})", color=INK, va="center", fontsize=10)
    ax.axvline(0, color=INK2, linewidth=1.0)
    ax.axvline(-1, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(-1.08, -0.75, "1-point gate", color=INK2, ha="right", va="center", fontsize=10)
    ax.set_ylim(-1.1, len(p) - 0.5)
    ax.set_yticks(range(len(p)), p.row)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy change vs Default (points), 95% CI, paired on the same questions")
    ax.set_title("Each FP8 change alone is within noise; both together lose 2.6 points on GSM8K")
    _finish(fig, out / "07_accuracy_paired.png",
            "GSM8K n = 1319, HumanEval n = 164, MMLU (8 subjects) n = 1687; McNemar test on discordant questions.")


def fig_power(d: Path, out: Path) -> None:
    c = pd.read_csv(d / "cells_combined_clean.csv", dtype={"config_bits": str})
    g = c[c.config_bits == "0000000"].set_index("point_id").reindex(POINT_ORDER)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(POINT_ORDER))
    bars = ax.bar(x, g.sm_clock_mhz, 0.6, color=BLUE, zorder=3)
    for b, v, pw in zip(bars, g.sm_clock_mhz, g.power_w):
        ax.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:.0f} MHz\n{pw:.0f} W", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [POINT_LABEL[p] for p in POINT_ORDER])
    ax.set_ylim(0, 2050)
    ax.set_ylabel("mean SM clock (MHz), Default config")
    ax.set_title("The GPU runs at its 400 W power cap: clocks fall as load rises")
    _finish(fig, out / "08_power_cap_clocks.png",
            "NVML reported clock-throttle reason 'SW power cap' in all 126 investigation runs; "
            "mean power per run 394-400 W against a 400 W limit.")


def fig_outlier(d: Path, out: Path) -> None:
    r = pd.read_csv(d / "outlier_rerun.csv")
    orig = [2715.95, 1762.42, 1774.39]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rng = np.random.default_rng(1)
    groups = [("fast plan\n(GPU 0, 11:21-11:26)", orig, AXIS),
              ("re-run, other GPU busy", list(r[~r.exclusive].output_tok_s), BLUE),
              ("re-run, other GPU idle", list(r[r.exclusive].output_tok_s), ORANGE)]
    for i, (lab, vals, col) in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(vals)), vals, s=70, color=col, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
    ax.annotate("rep 1 before 11:24", (0, orig[0]), xytext=(18, 0), textcoords="offset points", va="center",
                fontsize=10, color=INK2)
    ax.annotate("reps 2-3 after 11:24", (0, 1768), xytext=(18, 0), textcoords="offset points", va="center",
                fontsize=10, color=INK2)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylim(1500, 2900)
    ax.set_ylabel("output tokens/s")
    ax.set_title("The 'outlier' cell is the interference, not the configuration")
    _finish(fig, out / "09_outlier_rerun.png",
            "FP8 KV + EAGLE-3 at 128 requests in flight. 10 clean reps: 2666-2740 tok/s, cv < 1%, "
            "with or without load on the other GPU.")


def build_all(data: Path, out: Path, reported: dict[str, float]) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_contamination, fig_eagle, fig_reuse, fig_composition, fig_main_effects, fig_accuracy,
               fig_power, fig_outlier):
        fn(data, out)
    fig_corrected(data, out, reported)
    return sorted(out.glob("*.png"))
