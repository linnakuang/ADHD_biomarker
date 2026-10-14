"""D3 — do the ADHD comorbidity profiles survive our guards?

Linna's main branch fits K=3 NMF on ADHD (NT-residualized) and finds MSK-injury /
psychiatric-internalizing / atopic-ENT-infectious. Her 132-condition event table (with
injuries) is not on this box, so we refit on OUR event table (adhd78 set, no injury category)
and then apply the three tests the ASD components were put through:

  (a) ADHD-only K=3 NT-residual NMF (paper pipeline, s1 kernels) + §2 bootstrap
      (B=200): reportable? Profile themes by top loadings.
  (b) S2 — shared spectrum latent space (K=3 fit on 1,648 NDD residuals): ADHD vs other-NDD
      CV-AUC on latent W vs raw-78 vs burden-only, permutation p. Is ADHD's distinctness
      (S2 calibration AUC 0.638 on the 51) more than burden?
  (c) B2 — NT-normative basis (K_NT=12, Linna's NT_BASIS_K) projected out of ADHD raw rows;
      K=3 NMF on the residual; cosine of (a)'s profiles vs residual components:
      SURVIVES >=0.80 / PARTIAL >=0.50 / DISSOLVES. Plus r_norm (structural atypicality) for
      NT (5-fold held-out), other-NDD, ADHD, ASD: is the order NT < other < ADHD?
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls, linear_sum_assignment
from scipy.stats import mannwhitneyu
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

import adhd_common as ac
import config

OUT = ac.OUT
s1 = ac.s1
SEED = config.NMF_BOOTSTRAP_SEED
K = 3
K_NT = 12
N_PERM = 200


def top(H, conds, n=6):
    return [", ".join(f"{conds[j]}({H[k, j]:.2f})" for j in np.argsort(-H[k])[:n])
            for k in range(H.shape[0])]


def project(X, H):
    Wp = np.vstack([nnls(H.T, x)[0] for x in X])
    return Wp, Wp @ H


def l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
    return M / n


def main():
    ac.tee("d3")
    d = ac.load()
    conds = ac.condition_sets(d)["adhd"]
    df, ids = d["df"], d["ids"]
    nt = ac.nt_rate(d, conds)

    # ── (a) ADHD-only K=3 ──
    X, hb = s1.residual_matrix(df, ids["ADHD"], conds, nt)
    Xf = X[hb]
    Ha, Wa = s1.fit_nmf_best(Xf, K, 20, SEED)
    cos, worst, conc = s1.bootstrap_stability(Xf, np.ones(len(Xf), bool), K, Ha, Wa,
                                              config.NMF_BOOTSTRAP_B, config.NMF_BOOTSTRAP_RESTARTS, SEED)
    dom = Wa.argmax(1)
    print(f"(a) ADHD-only K={K} NT-residual NMF on {Xf.shape} (of {len(ids['ADHD'])})")
    print(f"    bootstrap cos={cos:.3f} worst={worst:.3f} concordance={conc:.3f} -> "
          f"{'REPORTABLE' if cos >= config.STABILITY_COSINE_FLOOR and worst >= config.STABILITY_COSINE_FLOOR and conc >= config.ASSIGN_CONCORDANCE_FLOOR else 'NOT reportable (some floor missed)'}")
    for k, t in enumerate(top(Ha, conds)):
        print(f"    P{k+1} (n={int((dom == k).sum())}): {t}")
    kscan = []
    for k in [2, 3, 4, 5]:
        Hk, Wk = s1.fit_nmf_best(Xf, k, 12, SEED)
        c_, w_, a_ = s1.bootstrap_stability(Xf, np.ones(len(Xf), bool), k, Hk, Wk, 80, 2, SEED)
        kscan.append({"K": k, "boot_cos": c_, "boot_worst": w_, "concordance": a_})
        print(f"    K={k}: cos={c_:.3f} worst={w_:.3f} conc={a_:.3f}")
    pd.DataFrame(kscan).round(4).to_csv(os.path.join(OUT, "d3_adhd_kscan.csv"), index=False)

    # ── (b) S2: spectrum latent space ──
    sys_s2 = __import__("s2_asd_separation")
    spec = ids["spectrum"]
    lab = d["patients"].set_index("ID")["label"].to_dict()
    Xs, _ = s1.residual_matrix(df, spec, conds, nt)
    Hs, Ws = s1.fit_nmf_best(Xs, K, 20, SEED)
    grp = np.array([lab[i] for i in spec])
    raw = ac.presence(df, spec, conds)[0]
    burden = (Xs > 0).sum(1).astype(float)[:, None]
    rows = []
    for name, (a, b) in {"ADHD_vs_other": ("ADHD", "Other NDD"), "ADHD_vs_ASD": ("ADHD", "ASD"),
                         "ASD_vs_other": ("ASD", "Other NDD")}.items():
        m = (grp == a) | (grp == b)
        y = (grp[m] == a).astype(int)
        r = {"contrast": name, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
        for fn, F in {"latentW": Ws[m], "raw78": Xs[m], "burden": burden[m],
                      "latentW+burden": np.c_[Ws[m], burden[m]]}.items():
            r[fn] = sys_s2.clf_auc(F, y)["logit"].mean()
        rows.append(r)
        print(f"(b) {name:<14} latentW={r['latentW']:.3f} raw78={r['raw78']:.3f} "
              f"burden={r['burden']:.3f} latentW+burden={r['latentW+burden']:.3f}")
    m = (grp == "ADHD") | (grp == "Other NDD")
    y = (grp[m] == "ADHD").astype(int)
    obs = rows[0]["latentW"]
    p = sys_s2.perm_null(Ws[m], y, obs, n_perm=N_PERM)
    print(f"    ADHD_vs_other latentW perm p={p:.4f} (N={N_PERM})")
    auc = pd.DataFrame(rows); auc["perm_p_latent_ADHD_vs_other"] = p
    auc.round(4).to_csv(os.path.join(OUT, "d3_separation_auc.csv"), index=False)
    Wn = Ws / np.where(Ws.sum(1, keepdims=True) > 0, Ws.sum(1, keepdims=True), 1)
    gm = pd.DataFrame({g: Wn[grp == g].mean(0) for g in ["Other NDD", "ADHD", "ASD"]},
                      index=[f"axis{k}" for k in range(K)])
    gm["ADHD_minus_other"] = gm["ADHD"] - gm["Other NDD"]
    gm["top"] = top(Hs, conds, 5)
    gm.to_csv(os.path.join(OUT, "d3_spectrum_axes.csv"))
    print(gm.drop(columns="top").round(3).to_string())
    for i, t in enumerate(gm["top"]):
        print(f"    axis{i}: {t}")
    cs = cosine_similarity(Ha, Hs)
    ri, ci = linear_sum_assignment(-cs)
    print("    ADHD-only profiles vs spectrum axes (matched cosine): " +
          ", ".join(f"P{r+1}~axis{c} {cs[r, c]:.2f}" for r, c in zip(ri, ci)))

    # ── (c) B2: NT-normative basis ──
    Xnt = ac.presence(df, ids["NT"], conds)[0].astype(float)
    m_nt = NMF(n_components=K_NT, init="nndsvda", solver="mu", max_iter=1000, random_state=SEED)
    m_nt.fit(Xnt); Hnt = m_nt.components_
    Xa = ac.presence(df, ids["ADHD"], conds)[0].astype(float)
    _, Xa_hat = project(Xa, Hnt)
    R = np.clip(Xa - Xa_hat, 0, None)
    nz = R.sum(1) > 0
    Hr, Wr = s1.fit_nmf_best(l2(R[nz]), K, 20, SEED)
    rc, rw, rconc = s1.bootstrap_stability(l2(R[nz]), np.ones(int(nz.sum()), bool), K, Hr, Wr,
                                           100, 3, SEED)
    C = cosine_similarity(Ha, Hr)
    print(f"\n(c) B2 residual K={K} on {int(nz.sum())} ADHD rows (K_NT={K_NT}); "
          f"bootstrap cos={rc:.3f} worst={rw:.3f} conc={rconc:.3f}")
    surv = []
    for k in range(K):
        b = float(C[k].max())
        v = "SURVIVES" if b >= .8 else "PARTIAL" if b >= .5 else "DISSOLVES"
        surv.append({"profile": f"P{k+1}", "top": top(Ha, conds, 4)[k], "best_resid_cos": b, "verdict": v})
        print(f"    P{k+1} -> best residual cos {b:.3f}: {v}   [{top(Ha, conds, 4)[k]}]")
    for k, t in enumerate(top(Hr, conds)):
        print(f"    R{k+1}: {t}")
    pd.DataFrame(surv).round(4).to_csv(os.path.join(OUT, "d3_b2_survival.csv"), index=False)

    # r_norm by group (NT held-out 5-fold)
    rn = {}
    kf = KFold(5, shuffle=True, random_state=SEED)
    nt_r = np.zeros(len(Xnt))
    for tr, te in kf.split(Xnt):
        mm = NMF(n_components=K_NT, init="nndsvda", solver="mu", max_iter=1000, random_state=SEED).fit(Xnt[tr])
        _, xh = project(Xnt[te], mm.components_)
        nt_r[te] = np.linalg.norm(Xnt[te] - xh, axis=1)
    rn["NT"] = nt_r
    for g in ["Other NDD", "ADHD", "ASD"]:
        Xg = ac.presence(df, ids[g], conds)[0].astype(float)
        _, xh = project(Xg, Hnt)
        rn[g] = np.linalg.norm(Xg - xh, axis=1)
    burden_g = {g: ac.presence(df, ids[g], conds)[0].sum(1) for g in rn}
    rows = []
    for g in rn:
        b = burden_g[g]
        rows.append({"group": g, "n": len(rn[g]), "r_norm_med": np.median(rn[g]),
                     "r_norm_per_sqrt_burden_med": np.median(rn[g][b > 0] / np.sqrt(b[b > 0])),
                     "burden_mean": b.mean()})
    rtab = pd.DataFrame(rows)
    print("\n" + rtab.round(3).to_string(index=False))
    for a_, b_ in [("ADHD", "Other NDD"), ("ADHD", "ASD"), ("Other NDD", "NT")]:
        ba, bb = burden_g[a_], burden_g[b_]
        xa_, xb_ = rn[a_][ba > 0] / np.sqrt(ba[ba > 0]), rn[b_][bb > 0] / np.sqrt(bb[bb > 0])
        print(f"    burden-adj r_norm {a_} vs {b_}: MWU p={mannwhitneyu(xa_, xb_).pvalue:.4g}")
    rtab.round(4).to_csv(os.path.join(OUT, "d3_rnorm_by_group.csv"), index=False)

    with open(os.path.join(OUT, "d3_summary.txt"), "w") as f:
        f.write(f"(a) ADHD-only K=3: boot cos={cos:.3f} worst={worst:.3f} conc={conc:.3f}\n")
        for k, t in enumerate(top(Ha, conds)):
            f.write(f"    P{k+1}: {t}\n")
        f.write(auc.round(3).to_string(index=False) + "\n")
        f.write(pd.DataFrame(surv).round(3).to_string(index=False) + "\n")
        f.write(rtab.round(3).to_string(index=False) + "\n")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
