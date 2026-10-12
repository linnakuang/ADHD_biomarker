from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT / "preprocessing"))
sys.path.append(str(_ROOT / "clustering"))
import config  # noqa: E402
from docx_table import write_docx_table  # noqa: E402
import nmf_analysis as nm  # noqa: E402
from nmf_analysis import THEME_ORDER, label_profiles  # noqa: E402

THR = 9                      # fixed high-burden threshold (primary NT p75 = 9 domains)
MIN_MONTHS = 2               # persistence: >=2 distinct diagnosis months
FOLLOWUP_MIN_Y = 3           # B2: minimum span of EHR records

# A: conditions/categories exempt from the >=2-months rule (valid as one-time events).
EXEMPT_CATEGORIES = {"Congenital", "Neoplastic Process"}
EXEMPT_CONDITIONS = {
    "Fractures (non-skull)", "Dislocations & sprains", "Open wounds / lacerations",
    "Superficial injuries & contusions", "Foreign body (ingested/aspirated/inserted)",
    "Poisoning & toxic effects", "Traumatic brain injury / concussion",
    "Epilepsy & seizures",
}
# C: high-frequency acute, contact-sensitive conditions to exclude.
ACUTE_CONDITIONS = {
    "Acute upper respiratory infection", "Otitis media & middle ear disorders",
    "Influenza & pneumonia", "Conjunctivitis & conjunctival disorders",
    "Pharyngitis & laryngitis", "Superficial injuries & contusions",
}

def summarize(counts):
    return (int(counts.median()), int(counts.quantile(.25)), int(counts.quantile(.75)),
            100 * (counts >= THR).mean())

def dom_counts(ev, ids):
    return ev.groupby("ID")["Disease_Category"].nunique().reindex(list(ids)).fillna(0).astype(int)

def main_profiles(p, ev):
    code, cw = nm.build_partition(p)
    conds, nt_rate = nm.select_conditions(p, ev, code, cw, verbose=False)
    X, _, _ = nm.residual_matrix(p, ev, conds, nt_rate)
    _, (_, H) = nm._fit(X, config.NMF_K, config.SEED)
    Hdf = pd.DataFrame(H, index=[f"P{i+1}" for i in range(config.NMF_K)], columns=conds)
    return Hdf, label_profiles(Hdf)

def refit_cosine(p, ev_f, Hdf_main, lab_main):
    try:
        code, cw = nm.build_partition(p)
        conds, nt_rate = nm.select_conditions(p, ev_f, code, cw, verbose=False)
        X, _, _ = nm.residual_matrix(p, ev_f, conds, nt_rate)
        _, (_, H) = nm._fit(X, config.NMF_K, config.SEED)
        Hdf = pd.DataFrame(H, index=[f"P{i+1}" for i in range(config.NMF_K)], columns=conds)
        lab = label_profiles(Hdf)
        common = [c for c in Hdf_main.columns if c in set(Hdf.columns)]
        sims = [float(cosine_similarity(
            Hdf_main.loc[lab_main[th], common].values.reshape(1, -1),
            Hdf.loc[lab[th], common].values.reshape(1, -1))[0, 0]) for th in THEME_ORDER]
        return min(sims), sims
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] NMF refit failed: {e}")
        return np.nan, []

def standardized_hi(child, thr):
    child = child.copy()
    child["hi"] = (child["ndom"] >= thr).astype(float)
    w = child.groupby("stratum").size() / len(child)
    da = child[child["group"] == "ADHD"].groupby("stratum").size()
    dn = child[child["group"] == "NT"].groupby("stratum").size()
    valid = [s for s in w.index if da.get(s, 0) >= config.MIN_STRATUM_N
             and dn.get(s, 0) >= config.MIN_STRATUM_N]
    code = {s: (s if s in valid else "POOLED") for s in w.index}
    child["st2"] = child["stratum"].map(code)
    cw = {s: float(w[s]) for s in valid}
    pooled = [s for s in w.index if s not in valid]
    if pooled:
        cw["POOLED"] = float(sum(w[s] for s in pooled))

    def stdz(g):
        sub = child[child["group"] == g]
        num = den = 0.0
        for s, wt in cw.items():
            m = sub[sub["st2"] == s]
            if len(m):
                num += wt * m["hi"].mean(); den += wt
        return 100 * num / den if den else np.nan
    return stdz("ADHD"), stdz("NT")

def run():
    config.ensure(config.SEN_RESULTS)

    p, ev_cond = nm.load()                             # condition-mapped, for NMF/persistence
    adhd_ids = set(p.loc[p["label"] == "ADHD", "ID"])
    nt_ids = set(p.loc[p["label"] == "NT", "ID"])
    Hdf_main, lab_main = main_profiles(p, ev_cond)

    # Full category-mapped events -> burden matches the primary S3/S4 definition.
    ev_all = pd.read_csv(config.EVENT_TABLE_CSV)
    ev_all["ID"] = ev_all["ID"].astype(str).str.strip()
    ev_all = ev_all[ev_all["ID"].isin(adhd_ids | nt_ids)
                    & (ev_all["age_years"] < config.AGE_CUTOFF)].dropna(subset=["Disease_Category"])

    rows = []

    def row(label, ev_b, ids_a=adhd_ids, ids_n=nt_ids, cosine=None):
        da, dn = dom_counts(ev_b, ids_a), dom_counts(ev_b, ids_n)
        am, nmn = summarize(da), summarize(dn)
        p75 = int(round(np.percentile(dn.values, 75)))          # sensitivity-specific NT p75
        a_p75 = 100 * (da >= p75).mean()
        cos = "1.00" if cosine == "self" else ("—" if cosine is None else f"{cosine:.2f}")
        rows.append([label, f"{am[0]} ({am[1]}–{am[2]})", f"{nmn[0]} ({nmn[1]}–{nmn[2]})",
                     f"{am[3]:.1f}", f"{nmn[3]:.1f}", f"{am[3]-nmn[3]:+.1f}",
                     str(p75), f"{a_p75:.1f}", cos])
        print(f"  {label:<30} ADHD {am[0]} (%>={THR} {am[3]:.1f}) | NT {nmn[0]} (%>={THR} {nmn[3]:.1f}) | "
              f"Δ={am[3]-nmn[3]:+.1f}  NTp75={p75} ADHD%≥p75={a_p75:.1f}  cos={cos}")

    print("[util] burden vs NT under each definition (any-ADHD vs NT):")
    row("Primary (any-ADHD)", ev_all, cosine="self")

    # ── A. Persistent-condition (>=2 distinct months; episodic/one-time exempt) ──
    ev_cm = ev_all.dropna(subset=["condition"])
    occ = (ev_cm.groupby(["ID", "condition", "Disease_Category"])["month"]
           .nunique().reset_index(name="n_months"))
    occ["exempt"] = (occ["Disease_Category"].isin(EXEMPT_CATEGORIES)
                     | occ["condition"].isin(EXEMPT_CONDITIONS))
    occ["drop"] = ~occ["exempt"] & (occ["n_months"] < MIN_MONTHS)
    drop_pairs = set(map(tuple, occ.loc[occ["drop"], ["ID", "condition"]].values))
    keep = [pd.isna(c) or (i, c) not in drop_pairs for i, c in zip(ev_all["ID"], ev_all["condition"])]
    ev_A = ev_all[keep]
    cosA, simsA = refit_cosine(p, ev_A.dropna(subset=["condition"]), Hdf_main, lab_main)
    row("A. Persistent (≥2 months)", ev_A, cosine=cosA)

    # ── C. Non-acute (exclude high-frequency acute conditions; keep category-only rows) ──
    ev_C = ev_all[~ev_all["condition"].isin(ACUTE_CONDITIONS)]
    cosC, simsC = refit_cosine(p, ev_C.dropna(subset=["condition"]), Hdf_main, lab_main)
    row("C. Exclude acute", ev_C, cosine=cosC)

    # ── B. Encounter-adjusted ──
    enc = ev_all.groupby("ID")["month"].nunique()
    span = ev_all.groupby("ID")["age_years"].max() - ev_all.groupby("ID")["age_years"].min()
    pa = p.set_index("ID")
    child = pd.DataFrame({"ID": list(adhd_ids | nt_ids)})
    child["group"] = np.where(child["ID"].isin(adhd_ids), "ADHD", "NT")
    child["ndom"] = dom_counts(ev_all, child["ID"]).values
    child["enc"] = enc.reindex(child["ID"]).fillna(0).astype(int).values
    child["span"] = span.reindex(child["ID"]).fillna(0).values
    child["encq"] = pd.qcut(child["enc"], 4, labels=False, duplicates="drop").astype(str)
    # sex x race, matching the primary standardization (education is not a stratifier)
    base_str = (pa["sex"].astype(str) + "|"
                + pd.to_numeric(pa["is_black"], errors="coerce").fillna(0).astype(int).astype(str))
    child["base_str"] = base_str.reindex(child["ID"]).values
    child["stratum"] = child["base_str"] + "|" + child["encq"]

    a1, n1 = standardized_hi(child, THR)
    rows.append(["B1. Within encounter quartile (contact-conditioned)",
                 "—", "—", f"{a1:.1f}", f"{n1:.1f}",
                 f"{a1-n1:+.1f}", "—", "—", "—"])
    print(f"  B1 within encounter quartile %>={THR}:  ADHD={a1:.1f}  NT={n1:.1f}  "
          f"Δ={a1-n1:+.1f}   (bounds detection opportunity; not an adjusted estimate)")

    # ── D. Before the diagnosis ──────────────────────────────────────────────
    med_dx = float(pd.to_numeric(pa["adhd_dx_age"], errors="coerce").dropna().median())
    ev_pre = ev_all[ev_all["age_years"] < med_dx]
    dn_pre = dom_counts(ev_pre, nt_ids)
    thr_pre = int(round(np.percentile(dn_pre.values, 75)))
    row(f"D1. Before age {med_dx:.1f}y (median dx), crude", ev_pre)
    print("     ^ the gap predates any ADHD label, so it is not diagnosis-triggered "
          "surveillance")

    child_pre = pd.DataFrame({"ID": list(adhd_ids | nt_ids)})
    child_pre["group"] = np.where(child_pre["ID"].isin(adhd_ids), "ADHD", "NT")
    child_pre["ndom"] = dom_counts(ev_pre, child_pre["ID"]).values
    enc_pre = ev_pre.groupby("ID")["month"].nunique()
    child_pre["enc"] = enc_pre.reindex(child_pre["ID"]).fillna(0).astype(int).values
    child_pre["encq"] = pd.qcut(child_pre["enc"], 4, labels=False,
                                duplicates="drop").astype(str)
    child_pre["stratum"] = (base_str.reindex(child_pre["ID"]).values
                            + "|" + child_pre["encq"])
    a3, n3 = standardized_hi(child_pre, thr_pre)
    rows.append([f"D2. Before age {med_dx:.1f}y, within encounter quartile",
                 "—", "—", f"{a3:.1f}", f"{n3:.1f}", f"{a3-n3:+.1f}",
                 str(thr_pre), "—", "—"])
    print(f"  D2 pre-diagnosis, within encounter quartile %>={thr_pre}:  "
          f"ADHD={a3:.1f}  NT={n3:.1f}  Δ={a3-n3:+.1f}")

    fu = child[child["span"] >= FOLLOWUP_MIN_Y]
    da2 = pd.Series(fu.loc[fu["group"] == "ADHD", "ndom"].values)
    dn2 = pd.Series(fu.loc[fu["group"] == "NT", "ndom"].values)
    a2, n2 = summarize(da2), summarize(dn2)
    p75b = int(round(np.percentile(dn2.values, 75)))
    rows.append([f"B2. ≥{FOLLOWUP_MIN_Y}y follow-up (n={len(da2)}/{len(dn2)})",
                 f"{a2[0]} ({a2[1]}–{a2[2]})", f"{n2[0]} ({n2[1]}–{n2[2]})",
                 f"{a2[3]:.1f}", f"{n2[3]:.1f}", f"{a2[3]-n2[3]:+.1f}",
                 str(p75b), f"{100*(da2>=p75b).mean():.1f}", "—"])
    print(f"  B2 ≥{FOLLOWUP_MIN_Y}y follow-up:  ADHD {a2[0]} (%>={THR} {a2[3]:.1f}) | NT {n2[0]} (%>={THR} {n2[3]:.1f})")

    write_docx_table(
        config.SEN_RESULTS / "sTable11_utilization_sensitivity.docx",
        "Supplementary Table S11. Recorded multimorbidity burden under alternative "
        "event definitions, and within strata of healthcare contact.",
        ["Analysis", "ADHD median (IQR)", "NT median (IQR)", f"ADHD % ≥{THR}",
         f"NT % ≥{THR}", f"Δ % ≥{THR} (pp)", "NT p75", "ADHD % ≥ NT p75",
         "Profile cosine"], rows)

    if simsA:
        print(f"[util] A profile cosines (injury/psychiatric/ENT): {[round(s,3) for s in simsA]}")
    if simsC:
        print(f"[util] C profile cosines (injury/psychiatric/ENT): {[round(s,3) for s in simsC]}")
    print(f"\nOutputs -> {config.SEN_RESULTS}")

if __name__ == "__main__":
    run()
