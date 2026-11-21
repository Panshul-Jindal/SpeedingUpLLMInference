"""Extra report figures (10-12) and the Appendix A table, generated from the report-pack CSVs.

Usage:
    python evaluation/make_report_figures.py [--pack evaluation/report_pack] [--tex docs/report]

Writes figures/10_centre_all_configs.png, 11_shapley_shares.png and 12_latency_vs_gates.png into the pack's
figures/ folder (same style as figures 01-09), and appendix_config_x_regime.tex into --tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BG, BLUE, ORANGE, GREY, INK = "#FCFCFA", "#2B78D6", "#E8663A", "#C9C6BA", "#222222"
POINTS = ["center", "context_low", "context_high", "concurrency_low", "concurrency_high", "reuse_low", "reuse_high"]
LABEL = {"center": "centre", "context_low": "short\nprompts", "context_high": "long\nprompts",
         "concurrency_low": "low\nconcurrency", "concurrency_high": "high\nconcurrency",
         "reuse_low": "no prefix\nreuse", "reuse_high": "90% prefix\nreuse"}
SHORT = {"center": "centre", "context_low": "short\nprompts", "context_high": "long\nprompts",
         "concurrency_low": "2 in\nflight", "concurrency_high": "128 in\nflight", "reuse_low": "0%\nreuse",
         "reuse_high": "90%\nreuse"}
OPT_COLOR = {"quant": "#7A5AC8", "kvq": "#8C8C8C", "spec": BLUE, "prefix": ORANGE, "batch": "#3BA272"}
NAMES = ["quant", "kvq", "spec", "prefix", "batch"]


def style(ax, title):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBBB")
    ax.tick_params(colors="#555555", labelsize=10)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8, color="#777777", ha="left", va="bottom")


def bits(x) -> str:
    return str(int(x)).zfill(7)


def fig_centre(cells: pd.DataFrame, out: Path) -> None:
    g = cells[cells.point_id == "center"].copy().sort_values("speedup")
    g["acc_fail"] = g.failed_gates.fillna("").str.contains("accuracy")
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    y = np.arange(len(g))
    colors = [GREY if f else BLUE for f in g.acc_fail]
    ax.barh(y, g.speedup, color=colors, height=0.72)
    for yi, (s, f) in enumerate(zip(g.speedup, g.acc_fail)):
        ax.text(s + 0.02, yi, f"{s:.2f}x", va="center", fontsize=8, color="#555555" if f else INK)
    ax.set_yticks(y, ["Default" if n == "default" else n for n in g.config_names.fillna("default")], fontsize=8.5)
    ax.axvline(1.0, color="#666666", lw=0.8)
    ax.set_xlabel("throughput speedup over Default (centre: 2048-token prompts, 32 in flight, 50% reuse)")
    ax.set_xlim(0, g.speedup.max() * 1.12)
    ax.grid(axis="x", color="#E3E3E3")
    ax.set_axisbelow(True)
    style(ax, "All 32 configurations at the centre, ranked by speedup over Default")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="passes every gate (policy A)"),
                       Patch(color=GREY, label="fails the original accuracy gate (Value = 1.00)")],
              loc="lower right", frameon=False, fontsize=9)
    note(fig, "Data: cells_combined_clean.csv (geometric mean of 3 reps). No configuration failed a latency gate.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_shapley(sh: pd.DataFrame, cells: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(POINTS))
    for i, p in enumerate(POINTS):
        g = sh[sh.point_id == p]
        gain = round(float(cells[cells.point_id == p].value.max()), 2)
        bottom = 0.0
        for n in NAMES:
            r = g[g.name == n]
            if r.empty:
                continue
            s = float(r.share.iloc[0])
            ax.bar(i, s * 100, bottom=bottom, color=OPT_COLOR[n], width=0.62, edgecolor=BG, lw=1)
            if s > 0.06:
                ax.text(i, bottom + s * 50, f"{n}\n{s * 100:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
            bottom += s * 100
        exact = "exact" if bool(g.exact.iloc[0]) else "approx."
        ax.text(i, 103, f"{g.final_config.map(lambda b: cells_name(cells, p, b)).iloc[0]}\n{gain:.2f}x, {exact}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [LABEL[p] for p in POINTS])
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("share of the final config's log gain (%)")
    ax.grid(axis="y", color="#E3E3E3")
    ax.set_axisbelow(True)
    style(ax, "Who earns the gain: Shapley shares of each final configuration")
    note(fig, "Data: clean_shapley.csv, v(S) = log(T_S / T_0). Exact where every subset was measured; "
              "elsewhere missing subsets come from the fitted model (approx.).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def cells_name(cells: pd.DataFrame, point: str, final_bits) -> str:
    b = bits(final_bits)
    r = cells[(cells.point_id == point) & (cells.config_bits.map(bits) == b)]
    return str(r.config_names.iloc[0]) if len(r) else b


def fig_latency(cells: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(POINTS))
    rows = []
    for p in POINTS:
        g = cells[cells.point_id == p]
        d = g[g.config_bits.map(bits) == "0000000"].iloc[0]
        b = g.loc[g.value.idxmax()]
        rows.append((d.p99_tpot_ms, b.p99_tpot_ms, d.p99_ttft_ms / 1000, b.p99_ttft_ms / 1000))
    rows = np.array(rows)
    for ax, (i0, gate, unit, title) in zip(axes, [(0, 100, "ms", "p99 time per output token"),
                                                  (2, 5, "s", "p99 time to first token")]):
        ax.bar(x - 0.19, rows[:, i0], 0.36, color=GREY, label="Default")
        ax.bar(x + 0.19, rows[:, i0 + 1], 0.36, color=BLUE, label="best config (policy A)")
        ax.axhline(gate, color=ORANGE, ls="--", lw=1.2)
        ax.text(len(POINTS) - 0.5, gate, f"gate {gate} {unit}", color=ORANGE, ha="right", va="bottom", fontsize=9)
        ax.set_xticks(x, [SHORT[p] for p in POINTS], fontsize=8.5)
        ax.set_ylabel(unit)
        ax.set_ylim(0, gate * 1.12)
        ax.grid(axis="y", color="#E3E3E3")
        ax.set_axisbelow(True)
        style(ax, title)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    note(fig, "Data: cells_combined_clean.csv. Gates from configs/experiment.yaml (p99 TPOT <= 100 ms, "
              "p99 TTFT <= 5 s). Every configuration at every point met both gates.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def appendix_table(cells: pd.DataFrame, out: Path) -> None:
    cells = cells.copy()
    cells["b"] = cells.config_bits.map(bits)
    lines = []
    for m in range(32):
        b5 = format(m, "05b")
        b7 = b5 + "00"
        g = cells[cells.b == b7]
        name = "Default" if m == 0 else "+".join(n for n, bit in zip(NAMES, b5) if bit == "1")
        vals = []
        for p in POINTS:
            r = g[g.point_id == p]
            if r.empty:
                vals.append("--")
                continue
            r = r.iloc[0]
            fail = "accuracy" in str(r.failed_gates)
            vals.append(f"{r.speedup:.2f}" + (r"$\dagger$" if fail else ""))
        shade = r"\rowcolor{rowgray} " if b5[1] == "1" else ""
        lines.append(f"{shade}{name} & {b5} & " + " & ".join(vals) + r" \\")
    final = []
    for p in POINTS:
        r = cells.loc[cells[cells.point_id == p].value.idxmax()]
        final.append(r"\makecell[r]{" + str(r.config_names).replace("+batch", r"\\+batch") +
                     r"\\\textbf{" + f"{r.value:.2f}" + "}}")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{@{}l}{\textbf{Final configuration (Value)}} & " + " & ".join(final) + r" \\")
    out.write_text("% Generated by evaluation/make_report_figures.py from cells_combined_clean.csv\n" +
                   "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="evaluation/report_pack")
    ap.add_argument("--tex", default="docs/report")
    a = ap.parse_args()
    pack, figs, tex = Path(a.pack), Path(a.pack) / "figures", Path(a.tex)
    tex.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(pack / "data" / "cells_combined_clean.csv")
    sh = pd.read_csv(pack / "data" / "clean_shapley.csv")
    fig_centre(cells, figs / "10_centre_all_configs.png")
    fig_shapley(sh, cells, figs / "11_shapley_shares.png")
    fig_latency(cells, figs / "12_latency_vs_gates.png")
    appendix_table(cells, tex / "appendix_config_x_regime.tex")
    print("wrote figures 10-12 and", tex / "appendix_config_x_regime.tex")


if __name__ == "__main__":
    main()
