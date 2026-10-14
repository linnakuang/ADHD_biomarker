"""D4 — ADHD presentation as an internal contrast: inattentive vs hyperactive/combined.

Within-ADHD comparison (no autism, no other-NDD in the contrast), so it is free of the
cross-diagnosis ascertainment questions; both arms are ADHD children in the same clinics.

Presentation is assigned with Linna's main-branch rule (build_patients.classify_subtype,
re-implemented verbatim): per child, count ICD-10 F90.0 / ICD-9 314.00 (inattentive) vs
F90.1/F90.2 / 314.01 (hyperactive/combined) codes; majority wins; ties or only F90.8/F90.9/
314.0/314.8/314.9 -> Unspecified; no code -> Not classifiable.

Analyses (adhd78 conditions, ADHD 573):
  - presentation counts + descriptives (sex, ADHD dx age, ADHD codes)
  - shrunk per-condition contrast  d_pres = p(inattentive) - p(hyp/comb)
  - each presentation vs other-NDD: does ADHD-specificity (D1) live in one presentation?
  - code-count sensitivity: restrict to children with >=3 ADHD codes (presentation more certain)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import adhd_common as ac

OUT = ac.OUT
ICD_RDS = "<PROJECT_ROOT>/kid_icdcode_data_Mar2026.RDS"

_ICD10 = {"F90.0": "inattentive", "F90.1": "hyp_comb", "F90.2": "hyp_comb",
          "F90.8": "unspecified", "F90.9": "unspecified"}
_ICD9 = {"314.00": "inattentive", "314.01": "hyp_comb",
         "314.0": "unspecified", "314.8": "unspecified", "314.9": "unspecified"}


def _norm_code(code):
    s = "".join(ch for ch in str(code).upper() if ch.isalnum() or ch == ".")
    if "." not in s and len(s) > 3:
        s = s[:3] + "." + s[3:]
    return s


def classify(ids):
    cache = os.path.join(ac.CACHE, "adhd_presentation.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"ID": str}).set_index("ID")
    import pyreadr
    icd = list(pyreadr.read_r(ICD_RDS).values())[0]
    dx = icd.dropna(subset=["ID"]).copy()
    dx["ID"] = pd.to_numeric(dx["ID"], errors="coerce").astype("Int64").astype(str).str.strip()
    dx = dx[dx["ID"].isin(set(ids))]
    dx["code"] = dx["icdcode"].map(_norm_code)
    dx["sub"] = dx["code"].map(_ICD10).fillna(dx["code"].map(_ICD9))
    dx = dx.dropna(subset=["sub"])
    base = dx.groupby(["ID", "sub"]).size().unstack(fill_value=0).reindex(index=ids, fill_value=0)
    for s in ("inattentive", "hyp_comb", "unspecified"):
        if s not in base.columns:
            base[s] = 0
    ina, hc, un = base["inattentive"], base["hyp_comb"], base["unspecified"]
    ncode = ina + hc + un
    pres = np.where(ncode == 0, "Not classifiable",
           np.where(ina > hc, "Inattentive",
           np.where(hc > ina, "Hyperactive/combined", "Unspecified")))
    out = pd.DataFrame({"presentation": pres, "n_adhd_codes": ncode.values,
                        "n_inatt": ina.values, "n_hypcomb": hc.values}, index=pd.Index(ids, name="ID"))
    out.to_csv(cache)
    return out


def main():
    ac.tee("d4")
    d = ac.load()
    conds = ac.condition_sets(d)["adhd"]
    df = d["df"]
    adhd = d["ids"]["ADHD"]
    pr = classify(adhd)
    print("=== presentation (ADHD 573) ===")
    print(pr["presentation"].value_counts().to_string())
    pt = d["patients"].set_index("ID")
    desc = []
    for g, sub in pr.groupby("presentation"):
        ids = list(sub.index)
        desc.append({"presentation": g, "n": len(ids),
                     "male_%": 100 * (pt.loc[ids, "sex"] == "M").mean(),
                     "adhd_dx_age_med": d["meta"].loc[ids, "adhd_dx_age"].median(),
                     "adhd_codes_med": sub["n_adhd_codes"].median(),
                     "burden_mean": ac.presence(df, ids, conds)[0].sum(1).mean(),
                     "event_months_med": ac.utilization(d, ids)["event_months"].median()})
    desc = pd.DataFrame(desc)
    print(desc.round(2).to_string(index=False))
    desc.round(3).to_csv(os.path.join(OUT, "d4_presentation_table.csv"), index=False)

    I = list(pr.index[pr["presentation"] == "Inattentive"])
    H = list(pr.index[pr["presentation"] == "Hyperactive/combined"])
    O = d["ids"]["Other NDD"]
    pI, pH, pO = (ac.prevalence(df, g, conds) for g in (I, H, O))
    c = ac.shrink_contrast(pI, len(I), pH, len(H))
    cI = ac.shrink_contrast(pI, len(I), pO, len(O))
    cH = ac.shrink_contrast(pH, len(H), pO, len(O))
    print(f"\nInattentive n={len(I)}  Hyp/comb n={len(H)}  tau(I-H)={c['tau']:.4f} mu={c['mu']:+.4f}")

    # sensitivity: >=3 ADHD codes
    I3 = [i for i in I if pr.at[i, "n_adhd_codes"] >= 3]
    H3 = [i for i in H if pr.at[i, "n_adhd_codes"] >= 3]
    c3 = ac.shrink_contrast(ac.prevalence(df, I3, conds), len(I3), ac.prevalence(df, H3, conds), len(H3))
    print(f">=3-code sensitivity: Inattentive n={len(I3)}  Hyp/comb n={len(H3)}")

    tab = pd.DataFrame({"condition": conds, "p_other": pO, "p_inatt": pI, "p_hypcomb": pH,
                        "d_I_minus_H": c["shrunk"], "lo90": c["lo90"], "hi90": c["hi90"],
                        "d_minus_mu": c["shrunk"] - c["mu"], "surv90": c["surv90_mu"], "surv90_3codes": c3["surv90_mu"],
                        "d_I_3codes": c3["shrunk"],
                        "d_inatt_vs_other": cI["shrunk"], "surv_inatt_other": cI["surv90_mu"] & (cI["shrunk"] > cI["mu"]),
                        "d_hyp_vs_other": cH["shrunk"], "surv_hyp_other": cH["surv90_mu"] & (cH["shrunk"] > cH["mu"])})
    tab = tab.sort_values("d_I_minus_H", key=np.abs, ascending=False)
    tab.round(4).to_csv(os.path.join(OUT, "d4_presentation_contrast.csv"), index=False)

    sub = tab[tab["surv90"]]
    print(f"\n=== Inattentive vs Hyp/comb, BEYOND the global offset mu={c['mu']:+.4f} "
          f"(I has uniformly fewer codes): {len(sub)} survive 90% ===")
    for _, r in sub.iterrows():
        print(f"  {r['condition']:<46} d={r['d_I_minus_H']:+.3f} d-mu={r['d_minus_mu']:+.3f} [{r['lo90']:+.3f},{r['hi90']:+.3f}]  "
              f"oth/I/H = {r['p_other']:.3f} {r['p_inatt']:.3f} {r['p_hypcomb']:.3f}  "
              f">=3codes:{'yes' if r['surv90_3codes'] else 'no'}")
    print(f"\nvs other-NDD, excess beyond each arm's global offset (90%): inattentive "
          f"{int(tab['surv_inatt_other'].sum())} (mu={cI['mu']:+.4f}), hyp/comb "
          f"{int(tab['surv_hyp_other'].sum())} (mu={cH['mu']:+.4f})")
    d1p = os.path.join(OUT, "d1_adhd78_adhd573.csv")
    if os.path.exists(d1p):
        d1 = pd.read_csv(d1p).set_index("condition")
        spec = d1.index[d1["label"].isin(["ADHD-SPECIFIC", "ADHD+ASD"])]
        t2 = tab.set_index("condition").loc[spec]
        print("\n=== where does each D1 ADHD-specific condition live? ===")
        for c_, r in t2.iterrows():
            where = ("both" if r["surv_inatt_other"] and r["surv_hyp_other"] else
                     "inattentive" if r["surv_inatt_other"] else
                     "hyp/comb" if r["surv_hyp_other"] else "neither alone")
            print(f"  {c_:<46} I={r['p_inatt']:.3f} H={r['p_hypcomb']:.3f} oth={r['p_other']:.3f} -> {where}")
    with open(os.path.join(OUT, "d4_summary.txt"), "w") as f:
        f.write(desc.round(2).to_string(index=False) + "\n\n")
        f.write(f"tau(I-H)={c['tau']:.4f}; survivors 90%: " + ", ".join(
            f"{r['condition']}({r['d_I_minus_H']:+.3f})" for _, r in sub.iterrows()) + "\n")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
