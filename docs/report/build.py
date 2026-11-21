#!/usr/bin/env python3
"""Build the project report: docs/Project_Report.pdf.

* Tables are generated from evaluation/report_pack/data/*.csv, so they cannot drift from the data.
* The sketches are (re)drawn by sketches/make_sketches.py.
* After rendering, every number printed in the PDF must be one of: a value that evaluation/check_report_numbers.py
  verifies against the data, a number in a table this script generated from the CSVs, a structural number (section,
  figure, date, version) or an explicitly confirmed exception below. Otherwise the build fails and lists it.

    python docs/report/build.py [--commit SHA]
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
OUT_PDF = REPO / "docs" / "Project_Report.pdf"
CHECKER = REPO / "evaluation" / "check_report_numbers.py"
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


# numbers that are structure, not results: small counts used as labels, the dates of the study
STRUCTURAL = {str(i) for i in range(0, 13)} | {"2026", "21", "14", "16", "25", "29", "30", "100", "120"}

# text that contains digits but no measured quantity; removed before the number check
STRUCTURAL_PATTERNS = [
    r"\[\d+(?:,\s*\d+)*\]",                       # citations [5, 6]
    r"(?s)\bReferences\b.*?(?=Appendix A|\Z)",    # the reference list
    r"\bADR-\d+(?:\s*(?:to|,|and)\s*(?:ADR-)?\d+)*", r"\b0\d{2}\b",   # ADR ids, incl. "013, 014" columns
    r"§\s?\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?", r"(?m)^\s*\d+(?:\.\d+)?\s+(?=[A-Z])",  # section refs, headings
    r"\bFigures?\s+\d+", r"\b[SMCFP]\d+(?:\.\d+)?(?:-[\w*]+)*",       # figure, check, milestone, claim, step ids
    r"\b\d+\.\d+\.\d+\S*", r"(?i)llama[- ]?3\.1", r"Opus \d+\.\d+", r"\bRTX \d+", r"H100 NVL \d+ GB",  # names
    r"\b[0-9a-f]{7,40}\b", r"arXiv:\S+",          # commit SHAs, arXiv ids
    r"\b(?:19[4-9]\d|20[0-3]\d)\b", r"\b\d{1,2}:\d{2}\b",              # years, clock times
    r"\b\d{1,2}\s+(?:Oct|October|Nov|November)\b",                     # dates
    r"\S*\d\S*\.(?:png|pdf|csv|svg|yaml|py|md|txt)\b",                 # file names with digits
    r"\bseed \d+", r"16,384", r"\bp99\b", r"\bR²", r"CS\d+", r"Project \d+|Group \d+",
]

# confirmed by hand: quantities the check script does not cover, each with where it comes from
CONFIRMED = {
    "1.5": "1.50x printed without its trailing zero by the tokenizer",
    "40": "a 40% slowdown = 1 - 0.596 (median ratio, S5.2), rounded",
    "4.3": "extra device memory on GPU 0: 91,917 - 87,498 MB (contamination_by_phase.csv)",
    "11.5": "phase-1 wall-clock, 03:23-14:52 UTC on 7 Nov (runs_fast_plan_all.csv timestamps)",
    "10.8": "GPU-minutes per setup = 11.5 h x 2 GPUs x 60 / 128 setups",
    "90": "hours for the 7-trick plan, from the harness RUNBOOK estimate (report section 7)",
    "112": "clean phase-1 setups = 104 kept + 8 re-run (cells_combined_clean.csv, source column)",
    "512": "7-trick plan: 128 + 6 x 64 setups; also the 512-token resolution target in section 9",
    "896": "7 tricks x 128 setups at every one of 7 workloads",
    "1536": "dry-run plan: 512 setups x 3 repeats (SMOKE_TEST_REPORT.md / dry profile)",
    "20": "infeasible setups recorded in the dry run (M2)",
    "68": "unit tests passed (pytest, 21 Nov)",
    "289": "number checks in evaluation/check_report_numbers.py",
    "99.7": "R^2 >= 0.9976 expressed as a percentage",
    "0.99": "R^2 target for future work (section 9), not a result",
    "75": "workload parameter: shared prefixes make up 75% of a reusing prompt (METHODOLOGY.md)",
    "500": "size of the trimmed GSM8K check (quality_trimmed_fast_plan.csv)",
    "4096": "prompt length in the long-context sweep (spec_vs_context.csv)",
    "6144": "prompt length in the long-context sweep (spec_vs_context.csv)",
    "0011000": "the 7-bit harness string for EAGLE-3 + prefix (configuration bits, not a measurement)",
    "00110": "the 5-bit string for EAGLE-3 + prefix (configuration bits)",
}


def verified_numbers() -> set[str]:
    """Every value that check_report_numbers.py compares with the data (both the report and the data side)."""
    r = subprocess.run([sys.executable, "-B", str(CHECKER)], capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0 or "checks passed" not in r.stdout:
        raise SystemExit("evaluation/check_report_numbers.py failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    vals = set()
    for line in r.stdout.splitlines():
        m = re.search(r": report (.*), data (.*)$", line)
        if m:
            vals |= numbers(m.group(1)) | numbers(m.group(2))
    return vals


def cross_check(pdf: Path, generated_html: str) -> list[str]:
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    for pat in STRUCTURAL_PATTERNS:
        txt = re.sub(pat, " ", txt)
    allowed = verified_numbers() | numbers(re.sub(r"<[^>]+>", " ", generated_html)) | STRUCTURAL | set(CONFIRMED)
    return sorted(numbers(txt) - allowed, key=lambda t: float(t))


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
    generated = subs["tbl_baseline"] + subs["tbl_accuracy"] + subs["appendix"]
    unknown = cross_check(OUT_PDF, generated)
    if unknown:
        print("numbers not verified by the check script (confirm or explain):", ", ".join(unknown))
        return 2
    print("number check: every number is verified by check_report_numbers.py, generated from the CSVs, "
          "or structural / confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
