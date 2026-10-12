from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocessing"))
import config  # noqa: E402
from build_patients import (SUBTYPES, NOT_CLASSIFIABLE, SUBTYPES_DETAIL,  # noqa: E402
                            DETAIL_TO_MAIN)
from docx_table import write_docx_table  # noqa: E402

# ── sTable3: how the presentation was assigned ──────────────────────────────
def presentation_detail(p):
    """sTable3, over the ADHD rows passed in by run()."""
    n = len(p)
    if "adhd_subtype_detail" not in p.columns:
        raise SystemExit("adhd_subtype_detail is missing; rerun build_patients.py "
                         "so the detailed presentation column is written.")

    codes = pd.to_numeric(p.get("n_adhd_codes"), errors="coerce")
    rows = []
    for level in SUBTYPES_DETAIL:
        m = p["adhd_subtype_detail"] == level
        k = int(m.sum())
        c = codes[m].dropna()
        rows.append([level, DETAIL_TO_MAIN[level], k, f"{100 * k / n:.1f}",
                     "—" if c.empty else f"{int(c.median())}",
                     "—" if c.empty else f"{int(c.quantile(.25))}–{int(c.quantile(.75))}"])
    rows.append(["Total", "—", n, "100.0", "—", "—"])

    write_docx_table(
        config.ANALYSIS_RESULTS / "sTable3_presentation_detail.docx",
        "Supplementary Table S3. Detailed ADHD presentation assignment, and the "
        "number of ADHD diagnosis codes each assignment rests on.",
        ["Detailed assignment", "Reported in Table 1 as", "n", "% of ADHD",
         "ADHD codes, median", "IQR"], rows)

    print(f"[S3] {n} children with ADHD")
    for level, main, k, pct, med, iqr in rows[:-1]:
        print(f"  {level:<56} {k:>5} ({pct:>5}%)  codes med={med} IQR={iqr}  -> {main}")
    collapsed = p["adhd_subtype"].value_counts()
    print("  collapsed to Table 1 categories: "
          + ", ".join(f"{k}={v}" for k, v in collapsed.items()))
    print(f"Output -> {config.ANALYSIS_RESULTS / 'sTable3_presentation_detail.docx'}")

# ── sTable7: presentation composition by clinical domain ────────────────────
def _mix(st, mask):
    """(n classifiable, {presentation: %}, n not classifiable) for a subset."""
    sub = st[mask]
    n_missing = int((sub == NOT_CLASSIFIABLE).sum())
    known = sub[sub != NOT_CLASSIFIABLE]
    n = len(known)
    pct = {s: (100 * (known == s).sum() / n if n else np.nan) for s in SUBTYPES}
    return n, pct, n_missing

def run():
    config.ensure(config.ANALYSIS_RESULTS)
    p = pd.read_csv(config.PATIENTS_ANALYTIC_CSV)
    p["ID"] = p["ID"].astype(str).str.strip()
    p = p[p["label"] == "ADHD"].copy()
    presentation_detail(p)

    p = p.dropna(subset=["adhd_subtype"])
    st = p.set_index("ID")["adhd_subtype"]
    n_adhd = len(p)

    ev = pd.read_csv(config.EVENT_TABLE_CSV)
    ev["ID"] = ev["ID"].astype(str).str.strip()
    ev = ev[ev["ID"].isin(set(p["ID"])) & (ev["age_years"] < config.AGE_CUTOFF)]
    ev = ev.dropna(subset=["Disease_Category"])

    cats = sorted(ev["Disease_Category"].unique())
    # Bonferroni over the domains actually tested; computed up front so the
    # threshold does not depend on the order rows happen to be built in.
    bonf = 0.05 / max(len(cats), 1)

    rows, pvals = [], []
    for cat in cats:
        aff = st.index.isin(set(ev.loc[ev["Disease_Category"] == cat, "ID"]))
        n_aff = int(aff.sum())
        if n_aff == 0:
            continue
        n_known, pct, n_miss = _mix(st, aff)
        if n_known == 0:
            continue
        # presentation x affected/not, among classifiable children only
        ctab = np.array([[int((st[aff] == s).sum()), int((st[~aff] == s).sum())]
                         for s in SUBTYPES])
        try:
            pval = chi2_contingency(ctab)[1] if ctab.sum() and (ctab.sum(1) > 0).all() else np.nan
        except ValueError:
            pval = np.nan
        pvals.append((cat, pval))
        mark = "*" if pd.notna(pval) and pval < bonf else ""
        rows.append([cat, n_aff, round(100 * n_aff / n_adhd, 1)]
                    + [f"{pct[s]:.1f}" for s in SUBTYPES]
                    + [n_miss, ("—" if pd.isna(pval) else
                                f"{'<0.001' if pval < 0.001 else f'{pval:.3f}'}{mark}")])

    rows.sort(key=lambda r: r[2], reverse=True)          # descending by % of ADHD affected

    # Reference row: the presentation mix of the whole ADHD cohort. Every domain
    # row is read against this one.
    all_known, all_pct, all_miss = _mix(st, np.ones(len(st), dtype=bool))
    rows.insert(0, ["All children with ADHD", n_adhd, 100.0]
                + [f"{all_pct[s]:.1f}" for s in SUBTYPES] + [all_miss, "—"])

    write_docx_table(
        config.ANALYSIS_RESULTS / "sTable7_subtype_by_category.docx",
        "Supplementary Table S7. ADHD presentation composition by clinical domain.",
        ["Clinical domain", "n affected", "% of ADHD"]
        + [f"{s}, %" for s in SUBTYPES]
        + ["Not classifiable, n", "p"], rows)

    valid = [pv for _, pv in pvals if pd.notna(pv)]
    sig = [(c, round(pv, 4)) for c, pv in pvals if pd.notna(pv) and pv < bonf]
    print(f"[S7] reference mix (n={all_known} classifiable, {all_miss} not): "
          + ", ".join(f"{s} {all_pct[s]:.1f}%" for s in SUBTYPES))
    print(f"     {len(pvals)} domains tested; Bonferroni threshold {bonf:.4f}; "
          f"min p = {min(valid) if valid else float('nan'):.4f}; "
          f"significant after correction: {sig if sig else 'none'}")
    print(f"Output -> {config.ANALYSIS_RESULTS / 'sTable7_subtype_by_category.docx'}")

if __name__ == "__main__":
    run()
