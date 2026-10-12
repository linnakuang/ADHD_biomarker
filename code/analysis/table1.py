from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocessing"))
import config  # noqa: E402
from docx_table import write_styled_table  # noqa: E402
from build_patients import SUBTYPES, NOT_CLASSIFIABLE  # noqa: E402

EDU = [("Less than High School", "Less than High School"),
       ("High School or equivalent", "High School or equivalent"),
       ("College or higher", "College or higher")]

# Sectioned Table-1 spec: (kind, label, col[, levels]).
SECTIONS = [
    ("Child characteristics", [
        ("multi", "Sex", "sex", [("M", "Male"), ("F", "Female")]),
        ("subtype", "ADHD presentation", None),
        ("bin", "Preterm birth (<37 wk)", "preterm"),
    ]),
    ("Maternal characteristics", [
        ("multi", "Race/ethnicity", "race_lbl",
         [("Black", "Non-Hispanic Black"), ("Others", "Others")]),
        ("multi", "Education", "education", EDU),
    ]),
]

def smd_cont(a, b):
    a, b = a.dropna(), b.dropna()
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sp if sp else np.nan

def smd_bin(pa, pb):
    sp = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return (pa - pb) / sp if sp else np.nan

def smd_cat(a, b, levels):
    a, b = a.dropna(), b.dropna()
    pa = np.array([(a == lv).mean() for lv in levels])[:-1]
    pb = np.array([(b == lv).mean() for lv in levels])[:-1]
    k = len(pa)
    if k == 0:
        return np.nan
    S = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            S[i, j] = ((pa[i] * (1 - pa[i]) + pb[i] * (1 - pb[i])) / 2 if i == j
                       else -(pa[i] * pa[j] + pb[i] * pb[j]) / 2)
    d = pa - pb
    try:
        return float(np.sqrt(d @ np.linalg.pinv(S) @ d))
    except Exception:  # noqa: BLE001
        return np.nan

def build_sectioned(NT, ADHD):
    n_nt, n_a = len(NT), len(ADHD)
    rows = []

    def pct(c, denom):
        return f"{int(c)} ({100 * c / denom:.0f})" if denom else "0 (0)"

    def fmt_smd(v):
        return f"{v:+.2f}" if pd.notna(v) else ""

    def cont(label, col):
        if col not in ADHD.columns:
            return
        nt = pd.to_numeric(NT[col], errors="coerce")
        a = pd.to_numeric(ADHD[col], errors="coerce")
        rows.append(("variable", [label, f"{nt.mean():.1f} ({nt.std():.1f})",
                                  f"{a.mean():.1f} ({a.std():.1f})",
                                  fmt_smd(smd_cont(a, nt))]))

    def binary(label, col):
        if col not in ADHD.columns:
            return
        nt = pd.to_numeric(NT[col], errors="coerce")
        a = pd.to_numeric(ADHD[col], errors="coerce")
        rows.append(("variable", [label, pct((nt == 1).sum(), nt.notna().sum()),
                                  pct((a == 1).sum(), a.notna().sum()),
                                  fmt_smd(smd_bin(a.mean(), nt.mean()))]))

    def multi(label, col, levels):
        if col not in ADHD.columns:
            return
        lv = [v for v, _ in levels]
        d_nt, d_a = int(NT[col].notna().sum()), int(ADHD[col].notna().sum())
        rows.append(("variable", [label, "", "", fmt_smd(smd_cat(ADHD[col], NT[col], lv))]))
        for v, disp in levels:
            rows.append(("data", [f"   {disp}", pct((NT[col] == v).sum(), d_nt),
                                  pct((ADHD[col] == v).sum(), d_a), ""]))
        if d_nt < n_nt or d_a < n_a:
            rows.append(("data", ["   Missing", str(n_nt - d_nt), str(n_a - d_a), ""]))

    def subtype():
        if "adhd_subtype" not in ADHD.columns:
            return
        known = ADHD["adhd_subtype"] != NOT_CLASSIFIABLE
        d = int(known.sum())
        rows.append(("variable", ["ADHD presentation", "", "", ""]))
        for st in SUBTYPES:
            rows.append(("data", [f"   {st}", "—",
                                  pct(int((ADHD["adhd_subtype"] == st).sum()), d), ""]))
        if d < n_a:
            rows.append(("data", ["   Missing", "—", str(n_a - d), ""]))

    for si, (sec_label, spec) in enumerate(SECTIONS):
        rows.append(("section", [sec_label, "", "", ""]))
        if si == 0:
            rows.append(("data", ["   N", str(n_nt), str(n_a), ""]))
        for item in spec:
            kind = item[0]
            if kind == "cont":
                cont(item[1], item[2])
            elif kind == "bin":
                binary(item[1], item[2])
            elif kind == "multi":
                multi(item[1], item[2], item[3])
            elif kind == "subtype":
                subtype()
    return rows

def _report(rows, out):
    for kind, cells in rows:
        print(f"  [{kind:<8}] " + " | ".join(f"{c}" for c in cells))
    print(f"-> {out}\n")

def run():
    config.ensure(config.ANALYSIS_RESULTS)
    p = pd.read_csv(config.PATIENTS_ANALYTIC_CSV)
    p["ID"] = p["ID"].astype(str).str.strip()
    p = p[p["label"].isin(["ADHD", "NT"])].copy()
    black = pd.to_numeric(p.get("is_black"), errors="coerce").fillna(0)
    p["race_lbl"] = np.where(black == 1, "Black", "Others")

    NT = p[p["label"] == "NT"]
    ADHD = p[p["label"] == "ADHD"]
    WIDTHS = [3.0, 1.6, 1.6, 1.1]

    rows = build_sectioned(NT, ADHD)
    cols = ["Characteristic", f"NT (N={len(NT)})", f"ADHD (N={len(ADHD)})", "SMD"]
    write_styled_table(config.ANALYSIS_RESULTS / "table1.docx",
                       "Table 1. Cohort characteristics.", cols, rows, col_widths=WIDTHS)
    _report(rows, "table1.docx")
    print(f"Outputs -> {config.ANALYSIS_RESULTS}")

if __name__ == "__main__":
    run()
