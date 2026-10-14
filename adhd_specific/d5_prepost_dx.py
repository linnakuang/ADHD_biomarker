"""D5 — what changes around the ADHD diagnosis, beyond what age and NDD-care change anyway?

A3/A3b for ADHD, rebuilt to avoid A3's weaknesses:
  * Child-anchored: each ADHD child's OWN first-dx age a_i is the anchor (IQR 5.4-9.3y, so a
    fixed boundary would mix pre and post). Symmetric windows pre=[a-3,a), post=[a,a+3) give
    equal exposure time on each side.
  * Guards get PSEUDO-anchors drawn from the ADHD dx-age distribution: NT (age/ascertainment
    guard, = A3b) AND other-NDD (NDD-care guard: a child newly in developmental care is also
    coded more -- A3b had no such guard). R=20 pseudo-anchor draws, averaged.
  * Eligibility by ADMINISTRATIVE follow-up (age at data extract >= a+3), not by having
    events, so nobody is selected on being coded.
  * Marginal per-condition DiD instead of NMF (A3b showed A3's NMF reshape was mostly age drift).

Per condition j and group g: shift_g = p_post - p_pre.
  DiD_NT    = shift_ADHD - shift_NT      DiD_other = shift_ADHD - shift_other  (bootstrap 90% CI)
A3b's key test: rho(shift_ADHD, shift_NT) and rho(shift_ADHD, shift_other) -- if ADHD's reshape
tracks the guards', the "reshape" is age/care, not ADHD.
Also the PRE-dx window alone: ADHD vs other-NDD (pseudo-anchored), shrunk -> which conditions
precede an ADHD diagnosis more than they precede other NDD (prodromal / early-warning set).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import adhd_common as ac

OUT = ac.OUT
W = 3.0
R_PSEUDO = 20
B = 500
SEED = 20261013


def extract_age(d):
    """Age (y) at data extract per child = (max event date over cohort - dob)."""
    p = d["patients"].set_index("ID")
    dob = pd.to_datetime(p["dob_date"], errors="coerce")
    ev = d["df"][["ID", "age_years"]]
    ev_date = dob.reindex(ev["ID"]).values + pd.to_timedelta(ev["age_years"].values * 365.25, unit="D")
    extract = pd.DatetimeIndex(ev_date).max()
    return ((extract - dob).dt.days / 365.25)


def window_mats(df, ids, anchors, conds):
    """(pre, post) n x J presence in [a-W,a), [a,a+W) per child."""
    ev = df[df["ID"].isin(set(ids)) & df["condition"].isin(conds)][["ID", "condition", "age_years"]]
    a = pd.Series(anchors, index=ids)
    ev = ev.assign(a=a.reindex(ev["ID"]).values)
    def mat(sub):
        m = (sub.drop_duplicates(["ID", "condition"]).assign(v=1.0)
             .pivot_table(index="ID", columns="condition", values="v", fill_value=0)
             .reindex(index=ids, columns=conds, fill_value=0))
        return m.values
    pre = ev[(ev["age_years"] >= ev["a"] - W) & (ev["age_years"] < ev["a"])]
    post = ev[(ev["age_years"] >= ev["a"]) & (ev["age_years"] < ev["a"] + W)]
    return mat(pre), mat(post)


def main():
    ac.tee("d5")
    rng = np.random.default_rng(SEED)
    d = ac.load()
    conds = ac.condition_sets(d)["adhd"]
    df = d["df"]
    xage = extract_age(d)

    adhd = d["ids"]["ADHD"]
    a = d["meta"].loc[adhd, "adhd_dx_age"]
    ok = a.notna() & (a >= W) & (xage.reindex(adhd).values >= a + W) & (a + W <= 18)
    A_ids = list(a.index[ok]); A_anc = a[ok].values
    print(f"ADHD eligible: {len(A_ids)}/{len(adhd)} (dx age >= {W}, follow-up >= dx+{W}); "
          f"anchor median {np.median(A_anc):.2f}y")
    preA, postA = window_mats(df, A_ids, A_anc, conds)

    guards = {}
    for g in ["NT", "Other NDD"]:
        ids_g = d["ids"][g]
        xa = xage.reindex(ids_g).values
        pre_acc, post_acc, n_acc = [], [], []
        mats = []
        for r in range(R_PSEUDO):
            anc = rng.choice(A_anc, size=len(ids_g), replace=True)
            e = (xa >= anc + W)
            ids_e = [i for i, k in zip(ids_g, e) if k]
            pr, po = window_mats(df, ids_e, anc[e], conds)
            mats.append((pr, po))
            n_acc.append(len(ids_e))
        guards[g] = mats
        print(f"{g}: eligible per pseudo-draw median {int(np.median(n_acc))}/{len(ids_g)}")

    def shift(pre, post):
        return post.mean(0) - pre.mean(0)

    sA = shift(preA, postA)
    sG = {g: np.mean([shift(*m) for m in guards[g]], axis=0) for g in guards}
    pre_p = {"ADHD": preA.mean(0), **{g: np.mean([m[0].mean(0) for m in guards[g]], 0) for g in guards}}
    post_p = {"ADHD": postA.mean(0), **{g: np.mean([m[1].mean(0) for m in guards[g]], 0) for g in guards}}

    # bootstrap DiD (resample children; guards use draw 0..B%R)
    did = {g: [] for g in guards}
    nA = len(A_ids)
    for b in range(B):
        ia = rng.integers(0, nA, nA)
        sa = shift(preA[ia], postA[ia])
        for g in guards:
            pr, po = guards[g][b % R_PSEUDO]
            ig = rng.integers(0, len(pr), len(pr))
            did[g].append(sa - shift(pr[ig], po[ig]))
    tab = pd.DataFrame({"condition": conds,
                        "pre_ADHD": pre_p["ADHD"], "post_ADHD": post_p["ADHD"],
                        "pre_NT": pre_p["NT"], "post_NT": post_p["NT"],
                        "pre_other": pre_p["Other NDD"], "post_other": post_p["Other NDD"],
                        "shift_ADHD": sA, "shift_NT": sG["NT"], "shift_other": sG["Other NDD"]})
    for g, k in [("NT", "NT"), ("Other NDD", "other")]:
        arr = np.array(did[g])
        tab[f"DiD_{k}"] = sA - sG[g]
        tab[f"DiD_{k}_lo90"] = np.percentile(arr, 5, axis=0)
        tab[f"DiD_{k}_hi90"] = np.percentile(arr, 95, axis=0)
        tab[f"DiD_{k}_sig"] = (tab[f"DiD_{k}_lo90"] > 0) | (tab[f"DiD_{k}_hi90"] < 0)

    # pre-dx window: ADHD vs other-NDD, shrunk (prodromal set)
    po_pre = pre_p["Other NDD"]
    n_oth = int(np.median([len(m[0]) for m in guards["Other NDD"]]))
    cpre = ac.shrink_contrast(pre_p["ADHD"], nA, po_pre, n_oth)
    cpost = ac.shrink_contrast(post_p["ADHD"], nA, post_p["Other NDD"], n_oth)
    tab["pre_d_vs_other"], tab["pre_surv90"] = cpre["shrunk"] - cpre["mu"], cpre["surv90_mu"]
    tab["post_d_vs_other"], tab["post_surv90"] = cpost["shrunk"] - cpost["mu"], cpost["surv90_mu"]
    tab = tab.sort_values("DiD_other", key=np.abs, ascending=False)
    tab.round(4).to_csv(os.path.join(OUT, "d5_prepost.csv"), index=False)

    rhoNT = stats.spearmanr(sA, sG["NT"]).statistic
    rhoO = stats.spearmanr(sA, sG["Other NDD"]).statistic
    tot = {g: (post_p[g].sum() - pre_p[g].sum()) for g in pre_p}
    print(f"\nTotal burden change post-pre (sum of prevalences): " +
          ", ".join(f"{g}={v:+.3f}" for g, v in tot.items()))
    print(f"Reshape-axis test: rho(shift_ADHD, shift_NT)={rhoNT:.3f}  "
          f"rho(shift_ADHD, shift_other)={rhoO:.3f}  (A3b ASD: 0.80 vs NT)")
    for k in ["NT", "other"]:
        sub = tab[tab[f"DiD_{k}_sig"]].sort_values(f"DiD_{k}", ascending=False)
        print(f"\n=== DiD vs {k}: {len(sub)} conditions with 90% CI excluding 0 ===")
        for _, r in sub.iterrows():
            print(f"  {r['condition']:<46} DiD={r[f'DiD_{k}']:+.3f} "
                  f"[{r[f'DiD_{k}_lo90']:+.3f},{r[f'DiD_{k}_hi90']:+.3f}]  ADHD {r['pre_ADHD']:.3f}->"
                  f"{r['post_ADHD']:.3f}  NT {r['pre_NT']:.3f}->{r['post_NT']:.3f}  "
                  f"oth {r['pre_other']:.3f}->{r['post_other']:.3f}")
    both = tab[tab["DiD_NT_sig"] & tab["DiD_other_sig"] & (np.sign(tab["DiD_NT"]) == np.sign(tab["DiD_other"]))]
    print(f"\nSurvive BOTH guards (same sign): {len(both)}: " + ", ".join(
        f"{c}({v:+.3f})" for c, v in zip(both["condition"], both["DiD_other"])))
    pre_set = tab[tab["pre_surv90"]].sort_values("pre_d_vs_other", ascending=False)
    print(f"\n=== PRE-dx (3y before ADHD dx) ADHD vs other-NDD beyond global offset mu={cpre['mu']:+.4f}, 90%: {len(pre_set)} ===")
    for _, r in pre_set.iterrows():
        print(f"  {r['condition']:<46} d-mu={r['pre_d_vs_other']:+.3f}  ADHD pre {r['pre_ADHD']:.3f} "
              f"other pre {r['pre_other']:.3f}  (post-dx surv: {'yes' if r['post_surv90'] else 'no'})")

    with open(os.path.join(OUT, "d5_summary.txt"), "w") as f:
        f.write(f"ADHD eligible n={nA}; windows +/-{W}y around own dx; guards pseudo-anchored R={R_PSEUDO}\n")
        f.write(f"rho(shift_ADHD, shift_NT)={rhoNT:.3f}; rho(shift_ADHD, shift_other)={rhoO:.3f}\n")
        f.write("burden change: " + ", ".join(f"{g}={v:+.3f}" for g, v in tot.items()) + "\n")
        f.write(f"DiD sig vs NT: {int(tab['DiD_NT_sig'].sum())}; vs other: {int(tab['DiD_other_sig'].sum())}; both: {len(both)}\n")
        f.write("both: " + ", ".join(f"{c}({v:+.3f})" for c, v in zip(both['condition'], both['DiD_other'])) + "\n")
        f.write(f"pre-dx ADHD vs other beyond mu={cpre['mu']:+.4f} (90%): " + ", ".join(
            f"{r['condition']}({r['pre_d_vs_other']:+.3f})" for _, r in pre_set.iterrows()) + "\n")
    _plot(tab, rhoNT, rhoO, os.path.join(OUT, "d5_prepost.pdf"))
    print(f"\nDone. Outputs in {OUT}")


def _plot(tab, rnt, ro, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, k, r, lab in [(axs[0], "shift_NT", rnt, "NT (age guard)"),
                          (axs[1], "shift_other", ro, "other-NDD (care guard)")]:
        ax.scatter(tab[k], tab["shift_ADHD"], s=18, alpha=0.7)
        lim = float(np.nanmax(np.abs(np.r_[tab[k], tab["shift_ADHD"]]))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "--", color="#999", lw=0.8)
        ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
        for _, rr in tab.head(8).iterrows():
            ax.annotate(rr["condition"][:22], (rr[k], rr["shift_ADHD"]), fontsize=6)
        ax.set_xlabel(f"post − pre prevalence, {lab}"); ax.set_ylabel("post − pre prevalence, ADHD")
        ax.set_title(f"ρ = {r:.2f}", fontsize=10)
    fig.suptitle("Change across the ADHD diagnosis (±3y), vs pseudo-anchored guards", fontsize=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
