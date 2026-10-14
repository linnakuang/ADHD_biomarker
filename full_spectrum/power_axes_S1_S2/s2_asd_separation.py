"""S2 — Does ASD stand out in the shared, well-powered latent space?

Claim (2) of the "include the NDD/ADHD groups" case. S1 shows a K=3 residualized
NMF learned from the whole NDD spectrum (1,648) is highly reproducible where the
145-ASD fit is not. Here we learn that shared latent space (all groups residualized
against the SAME NT baseline as the paper) and ask, agnostically, whether ASD
children occupy a distinguishable region relative to ADHD / other-NDD.

The specificity question the current ASD-vs-NT design cannot ask: does ASD
comorbidity differ from OTHER neurodevelopmental conditions, not just from typical
children? Everything is disciplined against a burden-only baseline (ASD kids simply
carry more conditions) — a real separation must beat burden.

Contrasts (5-fold CV ROC-AUC, LogisticRegression + RandomForest), each on three
feature sets — latent W(K), raw 51-condition residual, burden(1) — with a
permutation null on the key ASD-vs-otherNDD test and an ADHD-vs-otherNDD calibration
baseline (are two non-ASD NDD groups even separable?).

Outputs (results/):
  s2_separation_auc.csv     every contrast × feature set × classifier
  s2_k_sweep.csv            ASD-vs-otherNDD AUC vs latent K
  s2_axis_group_means.csv   per-axis mean loading by group (which axis carries ASD)
  s2_embedding.pdf          PCA of the residual space, groups coloured, ASD highlighted
  s2_summary.txt            headline verdict
  s2_run.log
"""
import os
import sys
import numpy as np
import pandas as pd

import gus_common as gc
import s1_axis_robustness as s1
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy.optimize import nnls

OUT = gc.OUT_DIR
SEED = 20260903
K_LATENT = 3               # primary latent space (paper's K)
K_SWEEP = [2, 3, 4, 5, 6, 8]
N_PERM = 500


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


def clf_auc(X, y, seed=SEED):
    """5-fold CV ROC-AUC for logistic + random forest. Returns dict."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = {}
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    out["logit"] = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    out["rf"] = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    return out


def perm_null(X, y, observed, n_perm=N_PERM, seed=SEED):
    """Permutation p-value for logistic CV-AUC (shuffle labels)."""
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced"))
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        a = cross_val_score(lr, X, yp, cv=cv, scoring="roc_auc").mean()
        ge += (a >= observed)
    return (ge + 1) / (n_perm + 1)


def main():
    sys.stdout = Tee(os.path.join(OUT, "s2_run.log"))
    conditions = sorted(l.strip() for l in open(s1.KEPT) if l.strip())
    patients = gc.load_patients(s1.FULL_PAT)
    df = gc.load_event_data(s1.FULL_EVT, s1.FULL_PAT)
    nt_rate, _ = s1.nt_expected_rate(df, patients, conditions)

    ids = {g: sorted(patients.loc[patients["label"] == g, "ID"])
           for g in ("ASD", "ADHD", "Other NDD", "NT")}
    spectrum = sorted(ids["ASD"] + ids["ADHD"] + ids["Other NDD"])
    lab = patients.set_index("ID")["label"].to_dict()

    # Shared latent space fit on the spectrum residual (NT excluded: its residual is ~0).
    Xspec, _ = s1.residual_matrix(df, spectrum, conditions, nt_rate)     # rows align to `spectrum`
    Hs, Ws = s1.fit_nmf_best(Xspec, K_LATENT, restarts=20, seed=SEED)
    burden = (Xspec > 0).sum(axis=1).astype(float)                       # retained-cond count (excess-over-NT)
    grp = np.array([lab[i] for i in spectrum])
    print(f"Latent space: NMF K={K_LATENT} on spectrum residual {Xspec.shape}; "
          f"groups {dict(pd.Series(grp).value_counts())}")

    def subset(mask_a, mask_b):
        m = mask_a | mask_b
        return Xspec[m], Ws[m], burden[m][:, None], (mask_a[m]).astype(int)

    is_asd, is_adhd, is_other = grp == "ASD", grp == "ADHD", grp == "Other NDD"
    contrasts = {
        "ASD_vs_otherNDD":   (is_asd, is_other),      # the specificity test
        "ASD_vs_ADHD":       (is_asd, is_adhd),
        "ASD_vs_ADHD+other": (is_asd, is_adhd | is_other),
        "ADHD_vs_otherNDD":  (is_adhd, is_other),     # calibration baseline
    }

    rows = []
    print("\n=== CV ROC-AUC by contrast × feature set ===")
    for name, (ma, mb) in contrasts.items():
        Xr, Wr, Br, y = subset(ma, mb)
        feats = {"latentW": Wr, "raw51": Xr, "burden": Br}
        line = {"contrast": name, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
        for fname, Xf in feats.items():
            a = clf_auc(Xf, y)
            line[f"{fname}_logit"] = a["logit"].mean()
            line[f"{fname}_logit_sd"] = a["logit"].std()
            line[f"{fname}_rf"] = a["rf"].mean()
        rows.append(line)
        print(f"  {name:<20} latentW logit={line['latentW_logit']:.3f}±{line['latentW_logit_sd']:.3f}"
              f"  raw51={line['raw51_logit']:.3f}  burden={line['burden_logit']:.3f}"
              f"  (rf latentW={line['latentW_rf']:.3f})")
    auc_df = pd.DataFrame(rows)
    auc_df.to_csv(os.path.join(OUT, "s2_separation_auc.csv"), index=False)

    # Permutation null on the key specificity test (latent W).
    ma, mb = contrasts["ASD_vs_otherNDD"]
    Xr, Wr, Br, y = subset(ma, mb)
    obs = clf_auc(Wr, y)["logit"].mean()
    p = perm_null(Wr, y, obs)
    print(f"\nASD_vs_otherNDD latentW: AUC={obs:.3f}  permutation p={p:.4f}  (N={N_PERM})")

    # K-sweep: does a richer latent space separate ASD from other-NDD better?
    print("\n=== ASD-vs-otherNDD AUC vs latent K ===")
    ksweep = []
    for k in K_SWEEP:
        Hk, Wk = s1.fit_nmf_best(Xspec, k, restarts=12, seed=SEED)
        m = is_asd | is_other
        a = clf_auc(Wk[m], is_asd[m].astype(int))["logit"]
        ksweep.append({"K": k, "auc_logit": a.mean(), "auc_sd": a.std()})
        print(f"  K={k}: AUC={a.mean():.3f}±{a.std():.3f}")
    pd.DataFrame(ksweep).to_csv(os.path.join(OUT, "s2_k_sweep.csv"), index=False)

    # Per-axis group means (which latent axis carries the ASD signal) + top loadings.
    Wn = Ws / np.where(Ws.sum(1, keepdims=True) > 0, Ws.sum(1, keepdims=True), 1)  # compositional
    axis_means = pd.DataFrame({g: Wn[grp == g].mean(0) for g in
                               ("ASD", "ADHD", "Other NDD")},
                              index=[f"axis{c}" for c in range(K_LATENT)])
    axis_means["ASD_minus_other"] = axis_means["ASD"] - axis_means["Other NDD"]
    axis_means.to_csv(os.path.join(OUT, "s2_axis_group_means.csv"))
    print("\n=== latent-axis mean loading by group (compositional) ===")
    print(axis_means.round(3).to_string())
    top = axis_means["ASD_minus_other"].abs().idxmax()
    a_idx = int(top.replace("axis", ""))
    order = np.argsort(-Hs[a_idx])[:8]
    print(f"\nMost ASD-vs-other axis = {top} (ASD−other={axis_means.loc[top,'ASD_minus_other']:+.3f}); "
          f"top conditions:")
    for j in order:
        print(f"    {conditions[j]:<45s} {Hs[a_idx, j]:.3f}")

    _embed(df, ids, conditions, nt_rate, patients, os.path.join(OUT, "s2_embedding.pdf"))

    with open(os.path.join(OUT, "s2_summary.txt"), "w") as f:
        f.write(f"ASD vs other-NDD (latent W, K={K_LATENT}): AUC={obs:.3f}, perm p={p:.4f}\n")
        f.write(f"  burden-only baseline: AUC={auc_df.set_index('contrast').loc['ASD_vs_otherNDD','burden_logit']:.3f}\n")
        f.write(f"  raw-51 residual:      AUC={auc_df.set_index('contrast').loc['ASD_vs_otherNDD','raw51_logit']:.3f}\n")
        f.write(f"  ADHD-vs-other calibration (latent W): "
                f"AUC={auc_df.set_index('contrast').loc['ADHD_vs_otherNDD','latentW_logit']:.3f}\n")
    print(f"\nDone. Outputs in {OUT}")


def _embed(df, ids, conditions, nt_rate, patients, path):
    """PCA of the residual space with all four groups; NT projected for context."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    allids = sorted(ids["ASD"] + ids["ADHD"] + ids["Other NDD"] + ids["NT"])
    X, _ = s1.residual_matrix(df, allids, conditions, nt_rate)
    lab = patients.set_index("ID")["label"].to_dict()
    grp = np.array([lab[i] for i in allids])
    Z = PCA(2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    style = {"NT": ("#bbbbbb", 8, 0.3, "o"), "Other NDD": ("#4c9f70", 12, 0.5, "o"),
             "ADHD": ("#3b6ea5", 12, 0.5, "o"), "ASD": ("#c1272d", 26, 0.9, "*")}
    for g in ("NT", "Other NDD", "ADHD", "ASD"):
        m = grp == g; c, sz, al, mk = style[g]
        ax.scatter(Z[m, 0], Z[m, 1], s=sz, c=c, alpha=al, marker=mk,
                   label=f"{g} (n={m.sum()})", edgecolors="none")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Residual comorbidity space (residualized vs NT)\nASD within the NDD spectrum")
    ax.legend(markerscale=1.5, fontsize=8); fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
