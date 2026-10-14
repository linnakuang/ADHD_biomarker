"""D1 + D2 — which comorbidities are ADHD-SPECIFIC, and are they steps or gradients?

The ADHD mirror of S5 (shrinkage) + S6 (gradient-vs-step) + S7 (cross-group). An ADHD-vs-NT
design (main branch) can only say "ADHD-associated". With the other NDD groups we decompose
the ADHD elevation over NT, per condition:

    e_ADHD = p_ADHD - NT_std            (what an ADHD-vs-NT paper reports)
           = (p_other - NT_std)  +  (p_ADHD - p_other)
             shared NDD               ADHD-specific delta

Three shrunk contrasts (each its own DerSimonian-Laird tau^2, S5 machinery verbatim):
    d_other = p_ADHD - p_otherNDD   (PRIMARY; 573 vs 930, comparator free of ADHD and autism)
    d_rest  = p_ADHD - p_(ASD+other) (S5 analog; pooled non-ADHD NDD, 1,075)
    d_ASD   = p_ADHD - p_ASD        (vs autism; ASD n=145 so wide)

Labels (d_other survival decided first):
    ADHD-SPECIFIC   d_other>0 survives AND d_ASD>0 survives   (ADHD exceeds both neighbours)
    ADHD+ASD        d_other>0 survives, d_ASD n.s.             (attention+autism share it vs other-NDD)
    ADHD-DEFICIT    d_other<0 survives                         (ADHD below the other-NDD level)
    ASD-PEAK        d_other n.s., d_ASD<0 survives              (an autism signal, not ADHD)
    ABOVE-ASD       d_other n.s., d_ASD>0 survives              (ADHD ~ other-NDD, both > ASD)
    SHARED-NDD      none survive, |e_ADHD| >= MIN_E
    FLAT            none survive, |e_ADHD| <  MIN_E
D2 shape for ADHD-elevated conditions: shared_frac = (p_other-NT)/(p_ADHD-NT)
    <0.5 STEP (ADHD jump), >=0.5 GRADIENT (other-NDD most of the way there); + Cochran-Armitage
    trend z over NT < other-NDD < ADHD.

Run on: condition set {adhd78 (primary), paper51}; ADHD definition {ADHD 573 (primary),
any-ADHD 634}. Aggregate outputs only.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import adhd_common as ac

OUT = ac.OUT
MIN_E = 0.03
LABELS = ["ADHD-SPECIFIC", "ADHD+ASD", "ADHD-DEFICIT", "ASD-PEAK", "ABOVE-ASD",
          "SHARED-NDD", "FLAT"]


def ca_trend_z(counts, ns, scores):
    N = ns.sum(); p = counts.sum() / N
    T = np.sum(scores * (counts - ns * p))
    var = p * (1 - p) * (np.sum(ns * scores**2) - np.sum(ns * scores)**2 / N)
    return float(T / np.sqrt(var)) if var > 0 else 0.0


def label(e, so, do, sa, da):
    if so and do > 0:
        return "ADHD-SPECIFIC" if (sa and da > 0) else "ADHD+ASD"
    if so and do < 0:
        return "ADHD-DEFICIT"
    if sa and da < 0:
        return "ASD-PEAK"
    if sa and da > 0:
        return "ABOVE-ASD"
    return "SHARED-NDD" if abs(e) >= MIN_E else "FLAT"


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def run(d, conds, adhd_key, tag):
    df, ids = d["df"], d["ids"]
    other = ids["Other NDD"]
    rest = ids["rest_noADHD"] if adhd_key == "ADHD" else ids["ASD-only"] + other
    asd = ids["ASD"] if adhd_key == "ADHD" else ids["ASD-only"]
    A = ids[adhd_key]
    nt = ac.nt_rate(d, conds)
    pA, pO, pR, pS = (ac.prevalence(df, g, conds) for g in (A, other, rest, asd))
    pNT_raw = ac.prevalence(df, ids["NT"], conds)
    nA, nO, nR, nS, nN = len(A), len(other), len(rest), len(asd), len(ids["NT"])

    e = pA - nt
    se_e = np.sqrt(pA * (1 - pA) / nA + pNT_raw * (1 - pNT_raw) / nN)
    p_e = 2 * stats.norm.sf(np.abs(pA - pNT_raw) / np.maximum(se_e, 1e-9))
    q_e = bh(p_e)

    co = ac.shrink_contrast(pA, nA, pO, nO)
    # sensitivity: sex x race standardized to the spectrum pool (ADHD 69% male vs other 52%)
    tgt = ids["spectrum"]
    cstd = ac.shrink_contrast(ac.std_prevalence(d, A, conds, tgt), nA,
                              ac.std_prevalence(d, other, conds, tgt), nO)
    cr = ac.shrink_contrast(pA, nA, pR, nR)
    ca = ac.shrink_contrast(pA, nA, pS, nS)

    shared = pO - nt
    sfrac = np.where(np.abs(e) > 1e-9, shared / e, np.nan)
    trend = []
    for j in range(len(conds)):
        cnt = np.array([pNT_raw[j] * nN, pO[j] * nO, pA[j] * nA])
        trend.append(ca_trend_z(cnt, np.array([nN, nO, nA], float), np.array([0., 1., 2.])))
    groups = {"NT": pNT_raw, "Other NDD": pO, "ADHD": pA, "ASD": pS}
    peak = [max(groups, key=lambda g: groups[g][j]) for j in range(len(conds))]

    lab = [label(e[j], co["surv90"][j], co["shrunk"][j], ca["surv90"][j], ca["shrunk"][j])
           for j in range(len(conds))]
    shape = []
    for j in range(len(conds)):
        if lab[j] in ("ADHD-SPECIFIC", "ADHD+ASD"):
            shape.append("STEP" if sfrac[j] < 0.5 else "GRADIENT")
        elif lab[j] == "SHARED-NDD" and e[j] > 0:
            shape.append("PLATEAU" if sfrac[j] >= 0.85 else "GRADIENT-ns")
        else:
            shape.append("")

    cat = ac.category_of(d)
    tab = pd.DataFrame({
        "condition": conds, "category": [cat.get(c, "") for c in conds],
        "p_NT": pNT_raw, "p_otherNDD": pO, "p_ADHD": pA, "p_ASD": pS, "nt_std": nt,
        "e_ADHD_vs_NT": e, "q_e_vs_NT": q_e, "shared_ndd": shared, "shared_frac": sfrac,
        "d_other_raw": co["raw"], "d_other": co["shrunk"], "d_other_lo90": co["lo90"],
        "d_other_hi90": co["hi90"], "surv_other90": co["surv90"], "surv_other95": co["surv95"],
        "d_other_minus_mu": co["shrunk"] - co["mu"], "surv_other_mu90": co["surv90_mu"],
        "surv_other_mu95": co["surv95_mu"],
        "d_other_std": cstd["shrunk"], "d_other_std_minus_mu": cstd["shrunk"] - cstd["mu"],
        "surv_other_std_mu90": cstd["surv90_mu"],
        "d_rest": cr["shrunk"], "surv_rest90": cr["surv90"],
        "d_ASD": ca["shrunk"], "surv_ASD90": ca["surv90"],
        "trend_z": trend, "peak_group": peak, "label": lab, "shape": shape,
    })
    tab = tab.sort_values("d_other", key=np.abs, ascending=False).reset_index(drop=True)
    tab.round(4).to_csv(os.path.join(OUT, f"d1_{tag}.csv"), index=False)

    cnt = pd.Series(lab).value_counts()
    assoc = (q_e < 0.05) & (e > 0)
    spec_among_assoc = int((assoc & co["surv90"] & (co["shrunk"] > 0)).sum())
    summ = {
        "tag": tag, "adhd_def": adhd_key, "n_conditions": len(conds),
        "n_ADHD": nA, "n_other": nO, "n_rest": nR, "n_ASD": nS,
        "tau_other": co["tau"], "mu_other": co["mu"], "tau_rest": cr["tau"], "tau_ASD": ca["tau"],
        "n_assoc_vs_NT_fdr05": int(assoc.sum()),
        "n_specific_vs_other90": int((co["surv90"] & (co["shrunk"] > 0)).sum()),
        "n_specific_vs_other95": int((co["surv95"] & (co["shrunk"] > 0)).sum()),
        "n_deficit_vs_other90": int((co["surv90"] & (co["shrunk"] < 0)).sum()),
        "n_beyond_global_up90": int((co["surv90_mu"] & (co["shrunk"] > co["mu"])).sum()),
        "n_beyond_global_down90": int((co["surv90_mu"] & (co["shrunk"] < co["mu"])).sum()),
        "n_specific_vs_rest90": int((cr["surv90"] & (cr["shrunk"] > 0)).sum()),
        "assoc_that_are_specific": spec_among_assoc,
        **{f"lab_{k}": int(cnt.get(k, 0)) for k in LABELS},
    }
    return tab, summ


def _forest(tab, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = tab.head(25).iloc[::-1]
    fig, (axf, axd) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(top))
    axf.scatter(top["d_other_raw"], y + 0.15, s=20, color="#9aa7b5", label="raw δ")
    col = np.where(top["surv_other_mu90"], "#c1272d", "#555555")
    for yi, (_, r), c in zip(y, top.iterrows(), col):
        axf.plot([r["d_other_lo90"], r["d_other_hi90"]], [yi - 0.15] * 2, color=c, lw=1)
    axf.scatter(top["d_other"], y - 0.15, s=26, c=col, label="shrunk δ (90% CI); red = beyond μ", zorder=3)
    mu = float((top["d_other"] - top["d_other_minus_mu"]).iloc[0])
    axf.axvline(mu, color="#c1272d", lw=0.8, ls="--")
    axf.axvline(0, color="k", lw=0.8)
    axf.set_yticks(y); axf.set_yticklabels(top["condition"], fontsize=7.5)
    axf.set_xlabel("ADHD-specific deviation  δ = p_ADHD − p_otherNDD")
    axf.set_title(title, fontsize=10); axf.legend(fontsize=8, loc="lower right")
    axf.grid(axis="x", alpha=0.3)
    dec = tab.reindex(tab["e_ADHD_vs_NT"].abs().sort_values(ascending=False).index).head(25).iloc[::-1]
    yd = np.arange(len(dec))
    axd.barh(yd, dec["shared_ndd"], color="#4c9f70", label="shared NDD (p_other − NT)")
    axd.barh(yd, dec["d_other"], left=dec["shared_ndd"], color="#c1272d", label="ADHD-specific δ (shrunk)")
    axd.axvline(0, color="k", lw=0.8)
    axd.set_yticks(yd); axd.set_yticklabels(dec["condition"], fontsize=7.5)
    axd.set_xlabel("elevation over NT"); axd.set_title("What the ADHD-vs-NT elevation is made of", fontsize=10)
    axd.legend(fontsize=8, loc="lower right"); axd.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _staircase(tab, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sel = tab[tab["label"].isin(["ADHD-SPECIFIC", "ADHD+ASD", "ADHD-DEFICIT"])].head(12)
    if sel.empty:
        return
    nc = 4; nr = int(np.ceil(len(sel) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), squeeze=False)
    xs = ["NT", "other", "ADHD", "ASD"]
    for ax, (_, r) in zip(axs.flat, sel.iterrows()):
        v = [r["p_NT"], r["p_otherNDD"], r["p_ADHD"], r["p_ASD"]]
        ax.bar(xs, v, color=["#888", "#4c9f70", "#c1272d", "#3b6ea5"])
        ax.set_title(f"{r['condition'][:30]}\n{r['label']} {r['shape']}", fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(sel):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ac.tee("d1")
    d = ac.load()
    cs = ac.condition_sets(d)
    summs = []
    for cset in ("adhd78", "paper51"):
        conds = cs["adhd"] if cset == "adhd78" else cs["paper51"]
        for adef in ("ADHD", "any-ADHD"):
            tag = f"{cset}_{'adhd573' if adef == 'ADHD' else 'anyadhd634'}"
            tab, s = run(d, conds, adef, tag)
            summs.append(s)
            print(f"\n######## {tag}: {len(conds)} conditions, ADHD n={s['n_ADHD']} ########")
            print(f"tau(vs other)={s['tau_other']:.4f}  mu={s['mu_other']:+.4f}  "
                  f"tau(vs rest)={s['tau_rest']:.4f}  tau(vs ASD)={s['tau_ASD']:.4f}")
            print(f"ADHD-associated vs NT (FDR<.05, e>0): {s['n_assoc_vs_NT_fdr05']}; "
                  f"of those ADHD-specific vs other-NDD (90%): {s['assoc_that_are_specific']}")
            print(f"beyond global offset mu (90%): up={s['n_beyond_global_up90']} "
                  f"down={s['n_beyond_global_down90']}")
            print("labels: " + ", ".join(f"{k}={s['lab_'+k]}" for k in LABELS))
            if tag == "adhd78_adhd573":
                primary = tab
                for L in ["ADHD-SPECIFIC", "ADHD+ASD", "ADHD-DEFICIT", "ASD-PEAK", "ABOVE-ASD"]:
                    sub = tab[tab["label"] == L]
                    print(f"\n=== {L} ({len(sub)}) ===")
                    for _, r in sub.iterrows():
                        print(f"  {r['condition']:<46} NT/oth/ADHD/ASD = {r['p_NT']:.3f} "
                              f"{r['p_otherNDD']:.3f} {r['p_ADHD']:.3f} {r['p_ASD']:.3f}  "
                              f"d_oth={r['d_other']:+.3f} d_ASD={r['d_ASD']:+.3f}"
                              f"{'*' if r['surv_ASD90'] else ' '} sf={r['shared_frac']:.2f} {r['shape']}")
                print("\n=== BEYOND the global ADHD-vs-other offset (d_other - mu survives 90%) ===")
                bg = tab[tab["surv_other_mu90"]].sort_values("d_other_minus_mu", ascending=False)
                for _, r in bg.iterrows():
                    print(f"  {r['condition']:<46} d_oth={r['d_other']:+.3f}  d-mu={r['d_other_minus_mu']:+.3f}"
                          f"{'**' if r['surv_other_mu95'] else ''}  {r['label']}  "
                          f"sex/race-std d-mu={r['d_other_std_minus_mu']:+.3f}"
                          f"{' (survives std)' if r['surv_other_std_mu90'] else ' (LOST under std)'}")
                gained = tab[tab["surv_other_std_mu90"] & ~tab["surv_other_mu90"]]
                print("  gained under sex/race standardization: " + ", ".join(
                    f"{c}({v:+.3f})" for c, v in zip(gained["condition"], gained["d_other_std_minus_mu"])))
                print("\n=== Largest ADHD-vs-NT elevations that are NOT ADHD-specific ===")
                gen = tab[~tab["surv_other90"] & (tab["e_ADHD_vs_NT"] > 0)]
                for _, r in gen.sort_values("e_ADHD_vs_NT", ascending=False).head(10).iterrows():
                    print(f"  {r['condition']:<46} e={r['e_ADHD_vs_NT']:+.3f}  shared="
                          f"{r['shared_ndd']:+.3f} ({100*r['shared_frac']:.0f}% of e)  peak={r['peak_group']}")
                _forest(tab, os.path.join(OUT, "d1_forest.pdf"),
                        "ADHD-specific deviation vs other-NDD (78 conditions, shrunk)")
                _staircase(tab, os.path.join(OUT, "d2_staircase.pdf"))

    sdf = pd.DataFrame(summs)
    sdf.round(4).to_csv(os.path.join(OUT, "d1_sensitivity.csv"), index=False)
    # robustness: are primary ADHD-specific conditions stable across sets/definitions?
    prim = set(primary.loc[primary["surv_other_mu90"] & (primary["d_other_minus_mu"] > 0), "condition"])
    print("\n=== Robustness of primary beyond-global ADHD-specific set ===")
    for tag in ["adhd78_anyadhd634", "paper51_adhd573", "paper51_anyadhd634"]:
        t = pd.read_csv(os.path.join(OUT, f"d1_{tag}.csv"))
        s = set(t.loc[t["surv_other_mu90"] & (t["d_other_minus_mu"] > 0), "condition"])
        common = prim & set(t["condition"])
        print(f"  {tag:<22} reproduces {len(common & s)}/{len(common)} of primary "
              f"(present in set); new: {sorted(s - prim)}")
    with open(os.path.join(OUT, "d1_summary.txt"), "w") as f:
        f.write(sdf.round(4).T.to_string() + "\n\n")
        bg = primary[primary["surv_other_mu90"]].sort_values("d_other_minus_mu", ascending=False)
        f.write("BEYOND-GLOBAL (d_other - mu, 90%): " + ", ".join(
            f"{r['condition']}({r['d_other_minus_mu']:+.3f})" for _, r in bg.iterrows()) + "\n")
        for L in ["ADHD-SPECIFIC", "ADHD+ASD", "ADHD-DEFICIT", "ASD-PEAK"]:
            sub = primary[primary["label"] == L]
            f.write(f"{L}: " + ", ".join(f"{r['condition']}({r['d_other']:+.3f},{r['shape']})"
                                         for _, r in sub.iterrows()) + "\n")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
