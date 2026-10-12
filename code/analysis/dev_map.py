from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocessing"))
import config  

F2_MAX_AGE = 16          # the heatmap stops at 16y; the 16-18 tail is too data-sparse

# ── Data + strata ───────────────────────────────────────────────────────────
def load():
    p = pd.read_csv(config.PATIENTS_ANALYTIC_CSV)
    p["ID"] = p["ID"].astype(str).str.strip()
    p = p[p["label"].isin(["ADHD", "NT"])].copy()
    p["is_black"] = pd.to_numeric(p.get("is_black"), errors="coerce").fillna(0).astype(int)
    p["stratum"] = p["sex"].astype(str) + "|" + p["is_black"].astype(str)

    ev = pd.read_csv(config.EVENT_TABLE_CSV)
    ev["ID"] = ev["ID"].astype(str).str.strip()
    ev = ev[ev["ID"].isin(set(p["ID"])) & (ev["age_years"] < config.AGE_CUTOFF)].copy()
    return p, ev

def build_partition(p):
    n = len(p)
    w = p.groupby("stratum").size() / n
    da = p[p["label"] == "ADHD"].groupby("stratum").size()
    dn = p[p["label"] == "NT"].groupby("stratum").size()
    valid = [s for s in w.index if da.get(s, 0) >= config.MIN_STRATUM_N
             and dn.get(s, 0) >= config.MIN_STRATUM_N]
    code = {s: (s if s in valid else "POOLED") for s in w.index}
    cw = {s: float(w[s]) for s in valid}
    pooled = [s for s in w.index if s not in valid]
    if pooled:
        cw["POOLED"] = float(sum(w[s] for s in pooled))
    return code, cw

def _std_curve(A, codes, cw):
    """Directly standardized column means of 0/1 matrix A (n x T)."""
    out = np.zeros(A.shape[1]); wsum = 0.0
    for c, wt in cw.items():
        m = codes == c
        if m.any():
            out += wt * A[m].mean(axis=0)
            wsum += wt
    return out / wsum if wsum > 0 else np.full(A.shape[1], np.nan)

# ── Per-condition summary (feeds codebook refinement) ───────────────────────
def condition_summary(p, ev):
    n_a = int((p["label"] == "ADHD").sum()); n_n = int((p["label"] == "NT").sum())
    lab = p.set_index("ID")["label"]
    ec = ev.dropna(subset=["condition"]).copy()
    ec["label"] = ec["ID"].map(lab)
    first = ec.groupby(["ID", "condition"])["age_years"].min().reset_index(name="onset")
    first["label"] = first["ID"].map(lab)
    rows = []
    for cond, g in ec.groupby("condition"):
        cat = g["Disease_Category"].iloc[0]
        a_ids = g[g["label"] == "ADHD"]["ID"].nunique()
        n_ids = g[g["label"] == "NT"]["ID"].nunique()
        oa = first[(first["condition"] == cond) & (first["label"] == "ADHD")]["onset"].median()
        rows.append((cat, cond, a_ids, 100 * a_ids / n_a, 100 * n_ids / n_n,
                     100 * a_ids / n_a - 100 * n_ids / n_n, oa))
    print(f"[summary] per-condition table written ({ec['condition'].nunique()} conditions)")

def _matrix(p, sub, col):
    """Child x `col` presence matrix (0/1), aligned to p's row order."""
    pres = sub.dropna(subset=[col]).drop_duplicates(["ID", col])
    m = pres.assign(v=1).pivot_table(index="ID", columns=col, values="v", fill_value=0)
    return m.reindex(p["ID"]).fillna(0).astype(float)

# ── F2: condition x developmental-window standardized ADHD-NT difference ─────
def condition_heatmap(p, ev, code, cw, med_dx, top_n=20,
                      out_pdf="F2_condition_diff_heatmap.pdf", whole_pdf=None,
                      tag="F2", col="condition", cbar="ADHD − NT (pp)",
                      out_dir=None):
    out = out_dir if out_dir is not None else config.ANALYSIS_RESULTS
    edges = [0.0, med_dx / 2, med_dx]
    t = med_dx
    while t + config.BIN_WIDTH_YEARS <= F2_MAX_AGE:
        t += config.BIN_WIDTH_YEARS
        edges.append(t)
    if edges[-1] < F2_MAX_AGE:
        edges.append(float(F2_MAX_AGE))
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}y" for i in range(len(edges) - 1)]
    sep_x = 1.5   # boundary at the median ADHD diagnosis age (after the two pre-median bins)

    ev = ev.dropna(subset=[col]).copy()
    ev["time_bin"] = pd.cut(ev["age_years"], bins=edges, labels=labels, right=False)
    ev = ev.dropna(subset=["time_bin"])
    codes = p["stratum"].map(code).values
    a, n = (p["label"] == "ADHD").values, (p["label"] == "NT").values

    adhd_ids, nt_ids = set(p.loc[a, "ID"]), set(p.loc[n, "ID"])
    diffs, counts = {}, {}
    print(f"[{tag}] children with >=1 condition event per time bin:")
    for tb in labels:
        sub = ev[ev["time_bin"] == tb]
        X = _matrix(p, sub, col)
        pa = _std_curve(X.values[a], codes[a], cw)     # directly standardized prevalence, ADHD
        pn = _std_curve(X.values[n], codes[n], cw)     # directly standardized prevalence, NT
        diffs[tb] = pd.Series((pa - pn) * 100, index=X.columns)
        na = sub.loc[sub["ID"].isin(adhd_ids), "ID"].nunique()
        nn = sub.loc[sub["ID"].isin(nt_ids), "ID"].nunique()
        counts[tb] = (na, nn)
        print(f"    {tb}: ADHD {na}, NT {nn}")
    xlab = [f"{tb}\nA={counts[tb][0]} N={counts[tb][1]}" for tb in labels]
    pivot = pd.DataFrame(diffs).reindex(columns=labels).fillna(0)

    cond_cat = (ev.drop_duplicates(col).set_index(col)["Disease_Category"]
                if col == "condition" else {})
    ordered = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    top = ordered.iloc[:top_n]                              # largest mean ADHD-NT gap
    vmax = max(abs(np.nanmin(ordered.values)), abs(np.nanmax(ordered.values)), 10)

    def ylabels(idx):
        return [f"{cond_cat.get(c, '')} | {c}" if col == "condition" else str(c) for c in idx]

    def panel(ax, pv, ttl, fs, annot, ylab=True, yright=False):
        M = pv[labels].values
        im = ax.imshow(M, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(xlab, fontsize=7, rotation=45, ha="right")
        if ylab:
            ax.set_yticks(range(len(pv))); ax.set_yticklabels(ylabels(pv.index), fontsize=fs)
            if yright:
                ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
        else:
            ax.set_yticks([])
        ax.set_title(ttl, fontsize=9)
        ax.axvline(sep_x, color="k", lw=1.5)                # diagnosis boundary (pre | post)
        if annot:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if abs(M[i, j]) >= 1:
                        ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=6)
        return im

    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.62 * len(labels) + 3.4) + 1.2,
                                            0.32 * top_n + 2.4))
    panel(axes[0], ordered, "(A) All conditions", 3.5, False, ylab=False)
    im = panel(axes[1], top, f"(B) Top {top_n} conditions", 7, True, yright=True)
    fig.colorbar(im, ax=axes, location="left", shrink=.5, label=cbar)
    fig.savefig(out / out_pdf, bbox_inches="tight"); plt.close(fig)

    # optional standalone, full-size all-conditions "whole" for the supplement
    if whole_pdf:
        figw, axw = plt.subplots(figsize=(0.9 * len(labels) + 4, 0.20 * len(ordered) + 1.6))
        panel(axw, ordered, "All conditions", 6, False)
        figw.colorbar(axw.images[0], ax=axw, shrink=.5, label=cbar)
        figw.savefig(out / whole_pdf, bbox_inches="tight"); plt.close(figw)

    print(f"[{tag}] {pivot.shape[0]} conditions x {len(labels)} bins; 2-panel all|top"
          + (" + standalone whole" if whole_pdf else ""))

def run():
    config.ensure(config.ANALYSIS_RESULTS)
    p, ev = load()
    med_dx = float(pd.to_numeric(p.loc[p["label"] == "ADHD", "adhd_dx_age"],
                                 errors="coerce").dropna().median())
    code, cw = build_partition(p)
    print(f"ADHD {int((p['label']=='ADHD').sum())} vs NT {int((p['label']=='NT').sum())}; "
          f"median ADHD dx {med_dx:.1f}y")
    condition_heatmap(p, ev, code, cw, med_dx,          # F2: 2-panel + sF3 whole
                      whole_pdf="sF3_condition_diff_heatmap_all.pdf")
    condition_summary(p, ev)
    print(f"\nOutputs -> {config.ANALYSIS_RESULTS}")

if __name__ == "__main__":
    run()
