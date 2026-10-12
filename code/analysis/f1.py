from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocessing"))
import config  # noqa: E402
from docx_table import write_docx_table  # noqa: E402

N_BOOT = 2000
AGES = np.arange(0, config.AGE_CUTOFF + 0.5, 0.5)
MILESTONE_AGES = [3, 6, 12, 18]
ADHD_LINE, ADHD_CI = "#C0392B", "#D4726A"
NT_LINE, NT_CI = "#2E7D8C", "#5B9DA8"

# ── Data ────────────────────────────────────────────────────────────────────
def load():
    p = pd.read_csv(config.PATIENTS_ANALYTIC_CSV)
    p["ID"] = p["ID"].astype(str).str.strip()
    p = p[p["label"].isin(["ADHD", "NT"])].copy()
    p["dob_date"] = pd.to_datetime(p["dob_date"], errors="coerce")
    if p["dob_date"].isna().any():
        raise SystemExit("Some children have no birth date; administrative "
                         "follow-up cannot be computed for them.")
    p["is_black"] = pd.to_numeric(p.get("is_black"), errors="coerce").fillna(0).astype(int)
    p["stratum"] = p["sex"].astype(str) + "|" + p["is_black"].astype(str)
    p = p.set_index("ID")

    ev = pd.read_csv(config.EVENT_TABLE_CSV)
    ev["ID"] = ev["ID"].astype(str).str.strip()
    ev = ev[ev["ID"].isin(set(p.index)) & (ev["age_years"] < config.AGE_CUTOFF)]
    ev = ev.dropna(subset=["Disease_Category"])
    return p, ev

def administrative_followup(p, ev):
    last_event = (p["dob_date"].reindex(ev["ID"]).values
                  + pd.to_timedelta(ev["age_years"].values * 365.25, unit="D"))
    extract = pd.Timestamp(np.nanmax(last_event)).normalize()
    fu = ((extract - p["dob_date"]).dt.days / 365.25).clip(upper=float(config.AGE_CUTOFF))
    if (fu <= 0).any():
        raise SystemExit("Some children have non-positive administrative follow-up.")
    return fu, extract

# ── Kaplan-Meier ────────────────────────────────────────────────────────────
def km_curve(t, d, grid):
    order = np.argsort(t, kind="mergesort")
    t_s, d_s = t[order], d[order].astype(np.int64)
    uniq, start = np.unique(t_s, return_index=True)
    deaths = np.add.reduceat(d_s, start)
    at_risk = t_s.size - start
    hit = deaths > 0
    if not hit.any():
        return np.zeros_like(grid)
    surv = np.cumprod(1.0 - deaths[hit] / at_risk[hit])
    idx = np.searchsorted(uniq[hit], grid, side="right") - 1
    return (1.0 - np.where(idx >= 0, surv[np.clip(idx, 0, None)], 1.0)) * 100

def survival_input(first, followup, cat, ids):
    ev = first.loc[first["Disease_Category"] == cat].set_index("ID")["first_age"].reindex(ids)
    age = ev.to_numpy(dtype=float)
    fu = followup.reindex(ids).to_numpy()
    d = (ev.notna() & (ev <= config.AGE_CUTOFF)).to_numpy()

    fu = np.where(d, np.maximum(fu, np.nan_to_num(age)), fu)
    return np.where(d, np.nan_to_num(age), fu), d.astype(np.int64)

def bootstrap_bands(inputs, grid, n_boot, seed):
    rng = np.random.default_rng(seed)
    curves = np.zeros((n_boot, grid.size))
    for b in range(n_boot):
        for w, (t, d) in inputs:
            k = rng.integers(0, t.size, size=t.size)
            curves[b] += w * km_curve(t[k], d[k], grid)
    return np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)

def curves_for(first, followup, strata, weights, cat, ids):
    inputs = []
    by_stratum = {}
    for i in ids:
        by_stratum.setdefault(strata.at[i], []).append(i)
    for s, w in weights.items():
        sids = by_stratum.get(s)
        if sids:
            inputs.append((w, survival_input(first, followup, cat, sids)))
    total = sum(w for w, _ in inputs)
    if not total:
        return np.zeros_like(AGES), []
    inputs = [(w / total, x) for w, x in inputs]
    point = sum(w * km_curve(t, d, AGES) for w, (t, d) in inputs)
    return point, inputs

# ── Figure ──────────────────────────────────────────────────────────────────
def draw(results, n_a, n_n, path, title_suffix=""):
    cats = sorted(results, key=lambda c: np.mean(results[c]["a"] - results[c]["n"]),
                  reverse=True)
    ncol = 4
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, cat in enumerate(cats):
        ax = axes[divmod(i, ncol)]
        r = results[cat]
        ax.fill_between(AGES, r["n_lo"], r["n_hi"], color=NT_CI, alpha=.2)
        ax.plot(AGES, r["n"], color=NT_LINE, lw=1.5, ls="--", dashes=(5, 3))
        ax.fill_between(AGES, r["a_lo"], r["a_hi"], color=ADHD_CI, alpha=.25)
        ax.plot(AGES, r["a"], color=ADHD_LINE, lw=1.8)
        ax.set_title(f"({chr(65 + i)}) {cat}", fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=.2)
        ax.set_xlim(0, config.AGE_CUTOFF)
        ax.set_xticks(range(0, config.AGE_CUTOFF + 1, 3))
    for j in range(len(cats), nrow * ncol):
        axes[divmod(j, ncol)].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Age (years)")
    for r_ in range(nrow):
        axes[r_, 0].set_ylabel("Cumulative incidence (%)")
    fig.legend(handles=[
        Line2D([0], [0], color=ADHD_LINE, lw=1.8, label=f"ADHD (n={n_a})"),
        Patch(facecolor=ADHD_CI, alpha=.3, label="ADHD 95% CI"),
        Line2D([0], [0], color=NT_LINE, lw=1.5, ls="--", dashes=(5, 3), label=f"NT (n={n_n})"),
        Patch(facecolor=NT_CI, alpha=.3, label="NT 95% CI")],
        loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[F1{title_suffix}] {len(cats)} panels -> {path.name}")
    return cats

def milestone_table(results, first, followup, adhd_ids, nt_ids, cats):
    def at(curve, age):
        return float(curve[np.searchsorted(AGES, age, side="right") - 1])

    def at_risk(cat, ids, age):
        t, d = survival_input(first, followup, cat, ids)
        return int(((t >= age) | (d == 1)).sum())

    rows = []
    for cat in cats:
        r = results[cat]
        cells = [cat]
        for age in MILESTONE_AGES:
            cells += [f"{at(r['a'], age):.1f}", f"{at(r['n'], age):.1f}",
                      f"{at(r['a'], age) - at(r['n'], age):+.1f}"]
        cells += [str(at_risk(cat, adhd_ids, 18)), str(at_risk(cat, nt_ids, 18))]
        rows.append(cells)
    cols = ["Clinical domain"]
    for age in MILESTONE_AGES:
        cols += [f"ADHD {age}y %", f"NT {age}y %", f"Diff {age}y"]
    cols += ["ADHD at risk 18y", "NT at risk 18y"]
    out = config.ANALYSIS_RESULTS / "sTable4_cumulative_incidence.docx"
    write_docx_table(out, "Supplementary Table S4. Kaplan-Meier cumulative incidence of "
                          "first diagnosis in each clinical domain, by age.", cols, rows)
    print(f"  Saved: {out.name}")

def run():
    config.ensure(config.ANALYSIS_RESULTS)
    p, ev = load()
    followup, extract = administrative_followup(p, ev)
    adhd_ids = p.index[p["label"] == "ADHD"].tolist()
    nt_ids = p.index[p["label"] == "NT"].tolist()
    print(f"ADHD n={len(adhd_ids)}, NT n={len(nt_ids)}")
    print(f"Record extract date inferred as {extract.date()}")
    for name, ids in (("ADHD", adhd_ids), ("NT", nt_ids)):
        fu = followup.reindex(ids)
        print(f"  {name}: follow-up median {fu.median():.1f}y "
              f"(IQR {fu.quantile(.25):.1f}-{fu.quantile(.75):.1f}); "
              f"reach 6y {100 * (fu >= 6).mean():.0f}%, 12y {100 * (fu >= 12).mean():.0f}%, "
              f"18y {100 * (fu >= 18).mean():.0f}%")

    first = (ev.groupby(["ID", "Disease_Category"])["age_years"].min()
               .rename("first_age").reset_index())
    cats = sorted(ev["Disease_Category"].dropna().unique())

    # Primary: crude KM. Sensitivity: directly standardized over sex x race.
    crude_strata = pd.Series("all", index=p.index)
    specs = [("crude", crude_strata, {"all": 1.0},
              config.ANALYSIS_RESULTS / "F1_cumulative_incidence.pdf"),
             ("standardized", p["stratum"], p["stratum"].value_counts(normalize=True).to_dict(),
              config.ANALYSIS_RESULTS / "sF1_cumulative_incidence_std.pdf")]

    for tag, strata, weights, path in specs:
        results = {}
        for cat in cats:
            a, a_in = curves_for(first, followup, strata, weights, cat, adhd_ids)
            n, n_in = curves_for(first, followup, strata, weights, cat, nt_ids)
            a_lo, a_hi = bootstrap_bands(a_in, AGES, N_BOOT, config.SEED)
            n_lo, n_hi = bootstrap_bands(n_in, AGES, N_BOOT, config.SEED + 1)
            results[cat] = dict(a=a, n=n, a_lo=a_lo, a_hi=a_hi, n_lo=n_lo, n_hi=n_hi)
        ordered = draw(results, len(adhd_ids), len(nt_ids), path, f" {tag}")
        if tag == "crude":
            milestone_table(results, first, followup, adhd_ids, nt_ids, ordered)

    print(f"\nOutputs -> {config.ANALYSIS_RESULTS}")

if __name__ == "__main__":
    run()
