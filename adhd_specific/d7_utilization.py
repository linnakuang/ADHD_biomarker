"""D7 — do ADHD's comorbidity signals survive a per-child utilization denominator? (M-I for ADHD)

Recurring ceiling from the ASD work: an NDD child is seen more, so more of everything gets
coded. D7 builds the denominator that A3b lacked and asks, per condition, whether the ADHD
effect survives it.

Utilization proxy: distinct event-MONTHS a child has any coded condition (0-18y), computed
LEAVE-CONDITION-OUT (months whose only code is condition j are removed when modelling j), so a
condition never inflates its own denominator. Plus record span (first->last event, years).

Per condition j, logistic regression (own Newton-Raphson, Wald SE):
    M0: y_j ~ ADHD + male + black
    M1: y_j ~ ADHD + male + black + log1p(util_-j) + span
Contrasts: ADHD vs other-NDD (PRIMARY, = D1's d_other) and ADHD vs NT (what main branch
reports). Output: OR0, OR1, attenuation of log-OR, BH q. Headline: which D1 ADHD-specific
conditions survive M1, and do the ADHD-peak deficits/excesses (headache, pain, mood) hold?

NOTE: utilization is partly a CONSEQUENCE of comorbidity (a sick child is seen more), so M1
over-adjusts; read M0->M1 as a bound, not the truth.
"""
import os
import numpy as np
import pandas as pd

import adhd_common as ac

OUT = ac.OUT


def logit_fit(X, y, iters=50):
    """Newton-Raphson logistic. Returns (beta, se). Small ridge for separation safety."""
    b = np.zeros(X.shape[1])
    lam = 1e-6
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1])
        g = X.T @ (y - p) - lam * b
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-eta))
    H = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.eye(X.shape[1])
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def leave_out_util(d, ids, conds):
    """(n x J) event-months excluding months whose ONLY code is condition j; + span."""
    df = d["df"][d["df"]["ID"].isin(set(ids))].assign(m=lambda x: np.floor(x["month"]))
    cm = df.drop_duplicates(["ID", "m", "condition"])
    per_month = cm.groupby(["ID", "m"])["condition"].agg(["nunique", "first"])
    total = per_month.groupby(level=0).size().reindex(ids).fillna(0).values
    solo = per_month[per_month["nunique"] == 1].reset_index()
    solo_ct = solo.groupby(["ID", "first"]).size().unstack(fill_value=0)
    solo_ct = solo_ct.reindex(index=ids, columns=conds, fill_value=0).values
    span = df.groupby("ID")["age_years"].agg(lambda s: s.max() - s.min()).reindex(ids).fillna(0).values
    return total[:, None] - solo_ct, total, span


def run_contrast(d, conds, g_case, g_ref, name):
    ids = d["ids"][g_case] + d["ids"][g_ref]
    case = np.r_[np.ones(len(d["ids"][g_case])), np.zeros(len(d["ids"][g_ref]))]
    pt = d["patients"].set_index("ID").loc[ids]
    male = (pt["sex"] == "M").astype(float).values
    black = pd.to_numeric(pt["is_black"], errors="coerce").fillna(0).values
    Y = ac.presence(d["df"], ids, conds)[0]
    U, total, span = leave_out_util(d, ids, conds)
    print(f"\n[{name}] n={len(ids)}  util(event-months) median case={np.median(total[case==1]):.0f} "
          f"ref={np.median(total[case==0]):.0f}; span median case={np.median(span[case==1]):.1f}y "
          f"ref={np.median(span[case==0]):.1f}y")
    rows = []
    for j, c in enumerate(conds):
        y = Y[:, j]
        X0 = np.c_[np.ones(len(y)), case, male, black]
        X1 = np.c_[X0, np.log1p(U[:, j]), span]
        b0, s0 = logit_fit(X0, y)
        b1, s1_ = logit_fit(X1, y)
        rows.append({"condition": c, "p_case": y[case == 1].mean(), "p_ref": y[case == 0].mean(),
                     "logOR0": b0[1], "se0": s0[1], "logOR1": b1[1], "se1": s1_[1],
                     "util_logOR": b1[4]})
    t = pd.DataFrame(rows)
    from scipy import stats
    t["p0"] = 2 * stats.norm.sf(np.abs(t["logOR0"] / t["se0"]))
    t["p1"] = 2 * stats.norm.sf(np.abs(t["logOR1"] / t["se1"]))
    t["q0"], t["q1"] = bh(t["p0"]), bh(t["p1"])
    t["OR0"], t["OR1"] = np.exp(t["logOR0"]), np.exp(t["logOR1"])
    t["attenuation"] = np.where(np.abs(t["logOR0"]) > 1e-6, 1 - t["logOR1"] / t["logOR0"], np.nan)
    t["contrast"] = name
    return t, (total, span, case)


def main():
    ac.tee("d7")
    d = ac.load()
    conds = ac.condition_sets(d)["adhd"]

    # utilization ladder across all groups
    lad = []
    for g in ["NT", "Other NDD", "ADHD", "ASD"]:
        u = ac.utilization(d, d["ids"][g])
        lad.append({"group": g, "n": len(u), "event_months_med": u["event_months"].median(),
                    "event_months_iqr": f"{u['event_months'].quantile(.25):.0f}-{u['event_months'].quantile(.75):.0f}",
                    "span_med": u["record_span_y"].median()})
    lad = pd.DataFrame(lad)
    print("=== utilization ladder ===\n" + lad.round(2).to_string(index=False))
    lad.to_csv(os.path.join(OUT, "d7_utilization_ladder.csv"), index=False)

    res = []
    for case, ref, name in [("ADHD", "Other NDD", "ADHD_vs_other"), ("ADHD", "NT", "ADHD_vs_NT")]:
        t, _ = run_contrast(d, conds, case, ref, name)
        res.append(t)
        s0 = (t["q0"] < .05); s1 = (t["q1"] < .05)
        print(f"  significant (BH q<.05): unadjusted {int(s0.sum())} -> utilization-adjusted "
              f"{int(s1.sum())}; both {int((s0 & s1).sum())}; median attenuation of sig. log-OR "
              f"{np.nanmedian(t.loc[s0, 'attenuation']):.2f}")
    t = pd.concat(res)
    t.round(4).to_csv(os.path.join(OUT, "d7_util_adjusted.csv"), index=False)

    # Cross with D1 labels
    d1p = os.path.join(OUT, "d1_adhd78_adhd573.csv")
    if os.path.exists(d1p):
        d1 = pd.read_csv(d1p).set_index("condition")
        to = t[t["contrast"] == "ADHD_vs_other"].set_index("condition")
        focus = d1.index[d1["label"].isin(["ADHD-SPECIFIC", "ADHD+ASD", "ADHD-DEFICIT"])]
        print("\n=== D1 ADHD-specific/deficit conditions under utilization adjustment (vs other-NDD) ===")
        out = []
        for c in focus:
            r = to.loc[c]
            ok = (r["q1"] < .05) and np.sign(r["logOR1"]) == np.sign(r["logOR0"])
            out.append({"condition": c, "D1_label": d1.at[c, "label"], "OR0": r["OR0"],
                        "OR1": r["OR1"], "q1": r["q1"], "attenuation": r["attenuation"],
                        "survives_util": ok})
            print(f"  {c:<46} {d1.at[c,'label']:<13} OR {r['OR0']:.2f} -> {r['OR1']:.2f}  "
                  f"q1={r['q1']:.3g}  atten={r['attenuation']:+.2f}  {'SURVIVES' if ok else 'lost'}")
        pd.DataFrame(out).round(4).to_csv(os.path.join(OUT, "d7_d1_survival.csv"), index=False)
        with open(os.path.join(OUT, "d7_summary.txt"), "w") as f:
            f.write(lad.round(2).to_string(index=False) + "\n\n")
            o = pd.DataFrame(out)
            f.write(f"D1 ADHD-specific/deficit conditions surviving utilization adjustment: "
                    f"{int(o['survives_util'].sum())}/{len(o)}\n")
            f.write(o.round(3).to_string(index=False) + "\n")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
