from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from docx_table import write_docx_table  # noqa: E402
from f1 import (AGES, N_BOOT, ADHD_LINE, ADHD_CI, NT_LINE, NT_CI,  # noqa: E402
                load, administrative_followup, km_curve, bootstrap_bands)

NT_PCTLS = (50, 75, 90)          # thresholds = NT Kaplan-Meier distribution quantiles
MAX_DOMAINS = 16

# ── Time to the k-th distinct affected domain ───────────────────────────────
def domain_onsets(ev):
    first = ev.groupby(["ID", "Disease_Category"])["age_years"].min()
    return {cid: np.sort(g.values) for cid, g in first.groupby(level=0)}

def nth_survival(onsets, followup, ids, k):
    age = np.array([onsets.get(i, np.empty(0))[k - 1]
                    if len(onsets.get(i, np.empty(0))) >= k else np.nan for i in ids])
    fu = followup.reindex(ids).to_numpy()
    d = ~np.isnan(age) & (age <= config.AGE_CUTOFF)
    fu = np.where(d, np.maximum(fu, np.nan_to_num(age)), fu)
    return np.where(d, np.nan_to_num(age), fu), d.astype(np.int64)

def km_at_18(onsets, followup, ids, k):
    t, d = nth_survival(onsets, followup, ids, k)
    return float(km_curve(t, d, AGES)[-1])

def nt_thresholds(onsets, followup, nt_ids):
    reach = {k: km_at_18(onsets, followup, nt_ids, k) for k in range(1, MAX_DOMAINS + 1)}
    out = {}
    for q in NT_PCTLS:
        need = 100 - q
        ks = [k for k, pct in reach.items() if pct >= need]
        out[q] = max(ks) if ks else 1
    return out, reach

def distinct_thresholds(thresholds):
    merged = {}
    for q, k in thresholds.items():
        merged.setdefault(k, []).append(q)
    return [(k, "NT p" + "/p".join(str(q) for q in qs))
            for k, qs in sorted(merged.items())]

# ── sF2: cumulative incidence of reaching each NT-referenced threshold ──────
def draw(curves, panels, n_a, n_n):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.8),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, (thr, lab) in zip(axes, panels):
        c = curves[thr]
        ax.fill_between(AGES, c["n_lo"], c["n_hi"], color=NT_CI, alpha=.2)
        ax.plot(AGES, c["n"], color=NT_LINE, lw=1.5, ls="--", dashes=(5, 3))
        ax.fill_between(AGES, c["a_lo"], c["a_hi"], color=ADHD_CI, alpha=.25)
        ax.plot(AGES, c["a"], color=ADHD_LINE, lw=1.8)
        ax.set_title(f"≥{thr} domains ({lab})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Age (years)")
        ax.set_xlim(0, config.AGE_CUTOFF)
        ax.set_xticks(range(0, config.AGE_CUTOFF + 1, 3))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Cumulative incidence (%)")
    fig.legend(handles=[
        Line2D([0], [0], color=ADHD_LINE, lw=1.8, label=f"ADHD (n={n_a})"),
        Patch(facecolor=ADHD_CI, alpha=.3, label="ADHD 95% CI"),
        Line2D([0], [0], color=NT_LINE, lw=1.5, ls="--", dashes=(5, 3), label=f"NT (n={n_n})"),
        Patch(facecolor=NT_CI, alpha=.3, label="NT 95% CI")],
        loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.06), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = config.ANALYSIS_RESULTS / "sF2_burden_accumulation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[sF2] {len(panels)} threshold(s) -> {out.name}")

def median_age(curve):
    idx = np.where(curve >= 50)[0]
    return float(AGES[idx[0]]) if len(idx) else np.nan

# ── sTable5: KM-estimated burden distribution ───────────────────────────────
def burden_table(onsets, followup, adhd_ids, nt_ids, nt_reach):
    rows = []
    for k in range(1, MAX_DOMAINS + 1):
        a = km_at_18(onsets, followup, adhd_ids, k)
        n = nt_reach[k]
        if a < 0.5 and n < 0.5:
            break
        rows.append([k, f"{a:.1f}", f"{n:.1f}", f"{a - n:+.1f}"])
    write_docx_table(
        config.ANALYSIS_RESULTS / "sTable5_burden_distribution.docx",
        "Supplementary Table S5. Kaplan-Meier cumulative incidence of reaching at least "
        "k affected clinical domains by age 18, children with ADHD versus NT.",
        ["≥ k domains", "ADHD % by 18y", "NT % by 18y", "Difference, pp"], rows)
    print(f"[sTable5] {len(rows)} thresholds tabulated")
    return rows

# ── sTable6: within ADHD, by sex and race ───────────────────────────────────
def subgroup_table(onsets, followup, p, adhd_ids, panels):
    groups = {"Sex": [("Male", "M"), ("Female", "F")],
              "Race": [("Black", 1), ("non-Black", 0)]}
    body, summary = [], []
    for grouping, levels in groups.items():
        col = "sex" if grouping == "Sex" else "is_black"
        for name, val in levels:
            ids = [i for i in adhd_ids if p.at[i, col] == val]
            for thr, lab in panels:
                t, d = nth_survival(onsets, followup, ids, thr)
                curve = km_curve(t, d, AGES)
                med = median_age(curve)
                body.append([grouping, name, lab, f"≥{thr} domains", len(ids),
                             f"{curve[-1]:.1f}",
                             "Not reached" if np.isnan(med) else f"{med:.1f}"])
                summary.append((grouping, name, lab, thr, len(ids), curve[-1], med))
    write_docx_table(
        config.ANALYSIS_RESULTS / "sTable6_threshold_sensitivity.docx",
        "Supplementary Table S6. Kaplan-Meier cumulative incidence of reaching "
        "NT-referenced cross-domain burden thresholds among children with ADHD, "
        "by sex and race/ethnicity.",
        ["Stratification", "Subgroup", "Threshold reference", "Threshold", "n",
         "% by age 18", "Median age, y"], body)
    print("[sTable6] cumulative incidence of NT-referenced high burden, by sex & race")
    print("      grouping subgroup   reference   thr     n  %by18  median_age")
    for g_, nm, lab, thr, nn, pct, med in summary:
        print(f"     {g_:<8} {nm:<10} {lab:<11} >={thr:<4} {nn:>4}  {pct:>5.1f}  "
              f"{'Not reached' if np.isnan(med) else f'{med:.1f}'}")

def run():
    config.ensure(config.ANALYSIS_RESULTS)
    p, ev = load()
    followup, extract = administrative_followup(p, ev)
    adhd_ids = p.index[p["label"] == "ADHD"].tolist()
    nt_ids = p.index[p["label"] == "NT"].tolist()
    onsets = domain_onsets(ev)
    print(f"ADHD n={len(adhd_ids)}, NT n={len(nt_ids)}; extract {extract.date()}")

    thresholds, nt_reach = nt_thresholds(onsets, followup, nt_ids)
    panels = distinct_thresholds(thresholds)
    print("NT Kaplan-Meier domain-count thresholds: "
          + ", ".join(f"p{q}=>={thr}" for q, thr in thresholds.items())
          + (f"  ({len(panels)} distinct)" if len(panels) < len(thresholds) else ""))

    curves = {}
    for thr, lab in panels:
        a_t, a_d = nth_survival(onsets, followup, adhd_ids, thr)
        n_t, n_d = nth_survival(onsets, followup, nt_ids, thr)
        a_lo, a_hi = bootstrap_bands([(1.0, (a_t, a_d))], AGES, N_BOOT, config.SEED)
        n_lo, n_hi = bootstrap_bands([(1.0, (n_t, n_d))], AGES, N_BOOT, config.SEED + 1)
        curves[thr] = dict(a=km_curve(a_t, a_d, AGES), n=km_curve(n_t, n_d, AGES),
                           a_lo=a_lo, a_hi=a_hi, n_lo=n_lo, n_hi=n_hi)
        ma, mn = median_age(curves[thr]["a"]), median_age(curves[thr]["n"])
        print(f"  >={thr} domains ({lab}): ADHD {curves[thr]['a'][-1]:.1f}% by 18 "
              f"(median {'NR' if np.isnan(ma) else f'{ma:.1f}y'}); "
              f"NT {curves[thr]['n'][-1]:.1f}% "
              f"(median {'NR' if np.isnan(mn) else f'{mn:.1f}y'})")

    draw(curves, panels, len(adhd_ids), len(nt_ids))
    burden_table(onsets, followup, adhd_ids, nt_ids, nt_reach)
    subgroup_table(onsets, followup, p, adhd_ids, panels)

    print(f"\nOutputs -> {config.ANALYSIS_RESULTS}")

if __name__ == "__main__":
    run()
