"""D6 — ADHD-only vs ASD+ADHD vs ASD-only (Linna's to-do: "ADHD only vs ASD+ADHD table").

The priority rule hides 61 children who carry BOTH diagnoses inside "ASD". Splitting them out
gives a 2x2 on the NDD side (ADHD yes/no x ASD yes/no, with other-NDD as the neither cell) and
lets us ask three things the main branch cannot:

  (1) Descriptives (Table): sex, race, preterm, LBW, education, ADHD dx age, ASD dx age,
      ADHD diagnostic visits, comorbidity burden. Diagnostic-overshadowing check: is ADHD
      diagnosed LATER in autistic children?
  (2) Shrunk per-condition contrasts (S5 machinery):
        aut_on_adhd  = p(ASD+ADHD) - p(ADHD-only)    what autism adds on top of ADHD
        adhd_on_aut  = p(ASD+ADHD) - p(ASD-only)     what ADHD adds on top of autism
  (3) Additivity: is ADHD's effect the same with and without autism?
        interaction  = [p(ASD+ADHD) - p(ASD-only)] - [p(ADHD) - p(otherNDD)]
      (risk-difference scale; shrunk). Survivors = conditions where ADHD's comorbidity
      footprint changes in the presence of autism.

n(ASD+ADHD)=61, n(ASD-only)=84 -> wide intervals; shrinkage is the multiplicity control.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import adhd_common as ac

OUT = ac.OUT
GROUPS = ["NT", "Other NDD", "ADHD", "ASD+ADHD", "ASD-only"]


def boot_mean_ci(x, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def descriptives(d, conds):
    p = d["patients"].set_index("ID")
    meta = d["meta"]
    rows = []
    for g in GROUPS:
        ids = d["ids"][g]
        sub = p.loc[ids]
        burden = ac.presence(d["df"], ids, conds)[0].sum(1)
        util = ac.utilization(d, ids)
        r = {"group": g, "n": len(ids),
             "male_%": 100 * (sub["sex"] == "M").mean(),
             "black_%": 100 * pd.to_numeric(sub["is_black"], errors="coerce").fillna(0).mean(),
             "preterm_%": 100 * pd.to_numeric(sub["preterm"], errors="coerce").mean(),
             "lbw_%": 100 * pd.to_numeric(sub["lbw"], errors="coerce").mean(),
             "college_%": 100 * (sub["education"].astype(str).str.contains("College")).mean()}
        a = meta.loc[ids, "adhd_dx_age"].dropna()
        r["adhd_dx_age_med"] = a.median() if len(a) else np.nan
        r["adhd_dx_age_iqr"] = f"{a.quantile(.25):.1f}-{a.quantile(.75):.1f}" if len(a) else ""
        v = meta.loc[ids, "adhd_n_visit"].dropna()
        r["adhd_dx_visits_med"] = v.median() if len(v) else np.nan
        s = pd.to_numeric(sub["asd_first_diagnosis_years"], errors="coerce").dropna()
        r["asd_dx_age_med"] = s.median() if len(s) else np.nan
        m, lo, hi = boot_mean_ci(burden)
        r["burden_mean"], r["burden_lo"], r["burden_hi"] = m, lo, hi
        r["event_months_med"] = util["event_months"].median()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ac.tee("d6")
    d = ac.load()
    conds = ac.condition_sets(d)["adhd"]
    df, ids = d["df"], d["ids"]

    desc = descriptives(d, conds)
    desc.round(2).to_csv(os.path.join(OUT, "d6_table.csv"), index=False)
    print("=== D6 descriptives ===")
    print(desc.round(2).to_string(index=False))

    # Diagnostic overshadowing: ADHD dx age, ADHD-only vs ASD+ADHD
    a1 = d["meta"].loc[ids["ADHD"], "adhd_dx_age"].dropna()
    a2 = d["meta"].loc[ids["ASD+ADHD"], "adhd_dx_age"].dropna()
    mw = stats.mannwhitneyu(a2, a1)
    print(f"\nADHD dx age: ADHD-only med {a1.median():.2f} (n={len(a1)}) vs ASD+ADHD med "
          f"{a2.median():.2f} (n={len(a2)}); MWU p={mw.pvalue:.4f}")
    asd_age = pd.to_numeric(d["patients"].set_index("ID").loc[ids["ASD+ADHD"],
                            "asd_first_diagnosis_years"], errors="coerce")
    order = (d["meta"].loc[ids["ASD+ADHD"], "adhd_dx_age"] - asd_age).dropna()
    print(f"ASD+ADHD: ADHD dx minus ASD dx: median {order.median():+.2f}y; ASD first in "
          f"{(order > 0).mean()*100:.0f}% (n={len(order)})")
    # Burden ladder
    for g1, g2 in [("ASD+ADHD", "ADHD"), ("ASD+ADHD", "ASD-only"), ("ADHD", "Other NDD")]:
        b1 = ac.presence(df, ids[g1], conds)[0].sum(1)
        b2 = ac.presence(df, ids[g2], conds)[0].sum(1)
        print(f"burden {g1} {b1.mean():.2f} vs {g2} {b2.mean():.2f}  MWU p="
              f"{stats.mannwhitneyu(b1, b2).pvalue:.4f}")

    p = {g: ac.prevalence(df, ids[g], conds) for g in GROUPS}
    n = {g: len(ids[g]) for g in GROUPS}
    c_aut = ac.shrink_contrast(p["ASD+ADHD"], n["ASD+ADHD"], p["ADHD"], n["ADHD"])
    c_adhd = ac.shrink_contrast(p["ASD+ADHD"], n["ASD+ADHD"], p["ASD-only"], n["ASD-only"])

    # interaction on RD scale with its own DL shrinkage
    def var(pp, nn):
        q = (pp * nn + 1) / (nn + 2)                     # Agresti-Caffo, as in shrink_contrast
        return q * (1 - q) / (nn + 2)
    inter = (p["ASD+ADHD"] - p["ASD-only"]) - (p["ADHD"] - p["Other NDD"])
    se_i = np.sqrt(var(p["ASD+ADHD"], n["ASD+ADHD"]) + var(p["ASD-only"], n["ASD-only"])
                   + var(p["ADHD"], n["ADHD"]) + var(p["Other NDD"], n["Other NDD"]))
    ci = ac.shrink(inter, se_i)
    ish, isurv = ci["shrunk"], ci["surv90"]
    print(f"interaction: tau={ci['tau']:.4f} mu={ci['mu']:+.4f} (se {ci['mu_se']:.4f}), Q={ci['Q']:.1f} df={ci['m']-1}")
    print(f"\ntau: autism-on-ADHD={c_aut['tau']:.4f}  ADHD-on-autism={c_adhd['tau']:.4f}  "
          f"interaction={ci['tau']:.4f}")
    print(f"global offsets mu: autism-on-ADHD={c_aut['mu']:+.4f}  ADHD-on-autism={c_adhd['mu']:+.4f}")

    tab = pd.DataFrame({
        "condition": conds,
        **{f"p_{g}": p[g] for g in GROUPS},
        "aut_on_adhd": c_aut["shrunk"], "aut_on_adhd_surv90": c_aut["surv90"],
        "aut_on_adhd_beyond_mu90": c_aut["surv90_mu"],
        "adhd_on_aut": c_adhd["shrunk"], "adhd_on_aut_surv90": c_adhd["surv90"],
        "adhd_on_aut_beyond_mu90": c_adhd["surv90_mu"],
        "adhd_effect_nonautistic": p["ADHD"] - p["Other NDD"],
        "interaction_raw": inter, "interaction": ish, "interaction_surv90": isurv,
    }).sort_values("aut_on_adhd", key=np.abs, ascending=False)
    tab.round(4).to_csv(os.path.join(OUT, "d6_dual_contrasts.csv"), index=False)

    for col, name in [("aut_on_adhd", "What AUTISM adds on top of ADHD (ASD+ADHD vs ADHD-only)"),
                      ("adhd_on_aut", "What ADHD adds on top of AUTISM (ASD+ADHD vs ASD-only)"),
                      ("interaction", "Non-additive: ADHD's effect differs with vs without autism")]:
        sub = tab[tab[f"{col}_surv90"]].sort_values(col, ascending=False)
        print(f"\n=== {name}: {len(sub)} survive 90% ===")
        for _, r in sub.iterrows():
            print(f"  {r['condition']:<46} {col}={r[col]:+.3f}  "
                  f"oth/ADHD/ASD+ADHD/ASD-only = {r['p_Other NDD']:.3f} {r['p_ADHD']:.3f} "
                  f"{r['p_ASD+ADHD']:.3f} {r['p_ASD-only']:.3f}"
                  + (f"  beyond-mu:{'yes' if r.get(col + '_beyond_mu90', False) else 'no'}"
                     if col != "interaction" else ""))

    with open(os.path.join(OUT, "d6_summary.txt"), "w") as f:
        f.write(desc.round(2).to_string(index=False) + "\n\n")
        f.write(f"ADHD dx age ADHD-only {a1.median():.2f} vs ASD+ADHD {a2.median():.2f}, "
                f"MWU p={mw.pvalue:.4f}; ASD dx precedes ADHD dx in {(order>0).mean()*100:.0f}%\n")
        for col in ["aut_on_adhd", "adhd_on_aut", "interaction"]:
            sub = tab[tab[f"{col}_surv90"]]
            f.write(f"{col} survivors (90%): " + ", ".join(
                f"{r['condition']}({r[col]:+.3f})" for _, r in sub.iterrows()) + "\n")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
