#!/usr/bin/env python3
"""Build the readable edition of the report: docs/Project_Report_readable.pdf.

* Tables are generated from evaluation/report_pack/data/*.csv, so they cannot drift from the data.
* The sketches are (re)drawn by sketches/make_sketches.py.
* After rendering, every number in the PDF is cross-checked against the formal report (docs/Project_Report.pdf),
  whose numbers are verified by evaluation/check_report_numbers.py. Numbers that do not appear there are listed so a
  person can confirm them; the build fails if any are unexplained.

    python docs/report_readable/build.py [--commit SHA]
Needs: pandas, WeasyPrint >= 54 (command line), pdftotext (poppler).
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "evaluation" / "report_pack" / "data"
OUT_PDF = REPO / "docs" / "Project_Report_readable.pdf"
FORMAL_PDF = REPO / "docs" / "Project_Report.pdf"
E = html.escape

POINTS = ["center", "context_low", "context_high", "concurrency_low", "concurrency_high", "reuse_low", "reuse_high"]
LABEL = {"center": "Centre", "context_low": "Short prompts", "context_high": "Long prompts",
         "concurrency_low": "Low load", "concurrency_high": "High load", "reuse_low": "No sharing",
         "reuse_high": "Heavy sharing"}
SETTING = {"center": "2,048 tok · 32 · 50%", "context_low": "256 tok · 32 · 50%", "context_high": "8,192 tok · 32 · 50%",
           "concurrency_low": "2,048 tok · 2 · 50%", "concurrency_high": "2,048 tok · 128 · 50%",
           "reuse_low": "2,048 tok · 32 · 0%", "reuse_high": "2,048 tok · 32 · 90%"}
OPTS = ["quant", "kvq", "spec", "prefix", "batch"]
PRETTY = {"quant": "FP8 weights", "kvq": "FP8 KV", "spec": "EAGLE-3", "prefix": "prefix", "batch": "batching"}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, dtype={"config_bits": str, "best_config": str})


def pretty(names: str) -> str:
    return " + ".join(PRETTY.get(n, n) for n in names.split("+"))


def table(header, rows, cls="t compact num"):
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def tbl_baseline() -> str:
    pol = read("final_configs_by_policy.csv")
    cells = read("cells_combined_clean.csv").set_index(["point_id", "config_bits"])
    rows = []
    for pt in POINTS:
        a = pol[(pol.point_id == pt) & pol.policy.str.startswith("A")].iloc[0]
        b = pol[(pol.point_id == pt) & pol.policy.str.startswith("B")].iloc[0]
        d, w = cells.loc[(pt, "0000000")], cells.loc[(pt, a.best_config)]
        rows.append([
            f"<b>{LABEL[pt]}</b><br><span class='muted'>{SETTING[pt]}</span>",
            f"{d.output_tok_s:,.0f}<br><span class='muted'>{d.p99_tpot_ms:.1f} / {d.p99_ttft_ms:,.0f}</span>",
            f"{E(pretty(a.best_names))}",
            f"{w.output_tok_s:,.0f}<br><span class='muted'>{w.p99_tpot_ms:.1f} / {w.p99_ttft_ms:,.0f}</span>",
            f"<b>{a.speedup:.2f}×</b>",
            f"{int(d.n_reps_ok)} + {int(w.n_reps_ok)} reps<br><span class='muted'>CV ≤ {100 * max(d.tok_s_cv, w.tok_s_cv):.2f}%</span>",
            f"{b.speedup:.2f}×",
        ])
    return table(["Workload<br><span class='muted'>prompt · in flight · sharing</span>",
                  "Default tok/s<br><span class='muted'>p99 TPOT ms / TTFT ms</span>",
                  "Best setup (policy A)",
                  "Its tok/s<br><span class='muted'>p99 TPOT ms / TTFT ms</span>",
                  "Speedup", "Runs / variance", "Policy B<br><span class='muted'>upper bound</span>"], rows)


def tbl_accuracy() -> str:
    p = read("quality_paired_vs_default.csv")
    g = p[p.task == "gsm8k"]
    rows = []
    for r in g.itertuples():
        sig = r.mcnemar_p < 0.05
        d = f"<b>{r.diff_pts:+.2f}</b>" if sig else f"{r.diff_pts:+.2f}"
        rows.append([E(pretty(r.names)), f"{r.acc:.1f}%", d, f"[{r.ci95_lo:+.1f}, {r.ci95_hi:+.1f}]", f"{r.mcnemar_p:.2f}"])
    other = p[p.task != "gsm8k"]
    n_sig = int((other.mcnemar_p < 0.05).sum())
    note = (f"<p class='small'>GSM8K, {int(g.n.iloc[0]):,} questions; Default scores {g.base_acc.iloc[0]:.1f}%. "
            f"On HumanEval ({int(other[other.task == 'humaneval'].n.iloc[0])}) and MMLU "
            f"({int(other[other.task == 'mmlu'].n.iloc[0]):,}) {'no' if n_sig == 0 else n_sig} setting changed "
            f"significantly.</p>")
    return table(["Setting", "Score", "Δ points", "95% interval", "p"], rows) + note


def appendix() -> str:
    t = read("clean_config_x_regime.csv")
    cells = read("cells_combined_clean.csv").set_index(["point_id", "config_bits"])
    rows = []
    for r in t.itertuples():
        if r.config_bits == "FINAL":
            out = []
            for pt in POINTS:
                bits, val = re.match(r"(\d+) \(([\d.]+)\)", getattr(r, pt)).groups()
                on = [PRETTY[OPTS[i]] for i in range(5) if bits[i] == "1"]
                out.append(f"<b>{' + '.join(on)}</b><br>{float(val):.2f}")
            rows.append(["<b>Winner (Value)</b>"] + out)
            continue
        out = []
        for pt in POINTS:
            if (pt, r.config_bits) not in cells.index:
                out.append("<span class='muted'>–</span>")
                continue
            c = cells.loc[(pt, r.config_bits)]
            v = float(c.speedup)
            if isinstance(c.failed_gates, str):
                out.append(f"<span class='gatefail'>{v:.2f}†</span>")
            else:
                out.append(f"<span class='{'hi' if v >= 1.8 else ('mid' if v >= 1.3 else '')}'>{v:.2f}</span>")
        name = "Default" if r.config_bits == "0000000" else E(pretty(r.config_names))
        rows.append([f"{name} <span class='muted'>{r.config_bits[:5]}</span>"] + out)
    return table(["Setup <span class='muted'>(bits)</span>"] + [LABEL[p] for p in POINTS], rows, "t tiny num")


def exec_words(doc: str) -> int:
    m = re.search(r'<div class="exec">(.*?)</div>', doc, re.S)
    return len(re.findall(r"[A-Za-z0-9][\w.×%–\-+,]*", re.sub(r"<[^>]+>", " ", m.group(1))))


NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    out = set()
    for tok in NUM.findall(text):
        tok = tok.replace(",", "").rstrip(".")
        if "." in tok:
            tok = tok.rstrip("0").rstrip(".") if not re.fullmatch(r"\d+\.\d*[1-9]", tok) else tok
        out.add(tok)
    return out


# numbers that are structure, not results: section / figure / step numbers, the commands' arguments, dates, ids
STRUCTURAL = {str(i) for i in range(0, 13)} | {"2026", "21", "14", "16", "25", "29", "30", "120", "100", "5", "2014",
                                                "1.1", "0.31.0", "13", "12", "18", "8", "6", "7", "9", "3", "2", "4"}
# confirmed by hand: values computed by this build from the CSVs, or plain arithmetic restatements
CONFIRMED = {
    "1.5": "1.50x (long prompts) printed without its trailing zero by the number tokenizer",
    "0.11": "rep CV 0.11% in the generated baseline table (cells_combined_clean.csv)",
    "40": "a 40% slowdown = 1 - 0.596 rounded (F1, median 0.596x)",
    "0011000": "the 7-bit harness string for EAGLE-3 + prefix (configuration bits, not a measurement)",
}


def cross_check(pdf: Path) -> list[str]:
    if not FORMAL_PDF.exists():
        return ["formal report not found; cross-check skipped"]
    txt = lambda p: subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout  # noqa: E731
    formal = numbers(txt(FORMAL_PDF))
    mine_text = txt(pdf)
    # generated tables are checked against the CSVs directly; strip appendix A (generated) from the comparison
    mine_text = mine_text.split("Appendix A")[0]
    unknown = sorted(numbers(mine_text) - formal - STRUCTURAL - set(CONFIRMED), key=lambda s: float(s))
    return unknown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default="", help="evidence commit SHA for the cover (set when tagging)")
    args = ap.parse_args(argv)

    subprocess.run([sys.executable, str(HERE / "sketches" / "make_sketches.py")], check=True, capture_output=True)
    doc = (HERE / "report.html").read_text()
    subs = {
        "commit": f"<code>{E(args.commit)}</code>" if args.commit else "<span class='todo'>set when the final tag is created</span>",
        "tbl_baseline": tbl_baseline(), "tbl_accuracy": tbl_accuracy(), "appendix": appendix(),
    }
    doc = re.sub(r"\{\{(\w+)\}\}", lambda m: subs[m.group(1)], doc)
    left = re.findall(r"\{\{\w+\}\}", doc)
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    words = exec_words(doc)
    if words > 200:
        raise SystemExit(f"executive summary has {words} words (limit 200)")
    out_html = HERE / "_report.filled.html"   # next to report.css so relative paths resolve
    out_html.write_text(doc)
    try:
        r = subprocess.run(["weasyprint", "-q", str(out_html), str(OUT_PDF)], capture_output=True, text=True)
    finally:
        out_html.unlink(missing_ok=True)
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        return r.returncode
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(OUT_PDF)], capture_output=True, text=True).stdout)
    print(f"wrote {OUT_PDF.relative_to(REPO)} ({pages.group(1) if pages else '?'} pages; summary {words} words)")
    unknown = cross_check(OUT_PDF)
    if unknown:
        print("numbers not found in the formal report (confirm or explain):", ", ".join(unknown))
        return 2
    print("cross-check: every number appears in the formal report, or is structural / confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
