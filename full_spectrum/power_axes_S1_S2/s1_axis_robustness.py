"""S1 — Do residualized comorbidity axes become more robust with more data?

The central premise of the "include the NDD/ADHD groups" case. The paper learns
K=3 NMF axes from 145 ASD children residualized against the NT-expected rate; those
axes sit at / below the pre-registered §2 stability floor (bootstrap matched cosine
~0.80, dominant-assignment concordance ~0.66). Here we residualize the WHOLE NDD
spectrum (ASD + ADHD + other-NDD = 1,648 children) against the SAME NT strata and
ask, as a measured learning curve, whether axis reproducibility rises with n and
crosses the §2 0.80 floor.

Feature space is fixed to the paper's 51 conditions and the NT-expected rate is the
one stratum-standardized vector (sex × is_black × education, target = full cohort,
strata with <5 NT pooled) subtracted from every group — so the ONLY thing that
changes across the curve is how many children the axes are estimated from.

Outputs (results/):
  s1_learning_curve.csv     stability vs n (mean over draws) for K in {2,3}
  s1_learning_curve.pdf     the money plot with the 0.80 floor drawn on
  s1_anchors.csv            ASD-only(145) vs full-spectrum(1648) at full bootstrap
  s1_run.log                console log
"""
import os
import sys
import time
import numpy as np
import pandas as pd

import gus_common as gc
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

OUT = gc.OUT_DIR
FULL_EVT = "<PROJECT_ROOT>/BBC_Dataset/preprocessing/results/BBC_event_table_full.csv"
FULL_PAT = "<PROJECT_ROOT>/BBC_Dataset/preprocessing/results/BBC_patients_analytic_full.csv"
KEPT = "<PROJECT_ROOT>/BBC_Dataset/clustering/results/nmf_kept_conditions.txt"
MIN_STRATUM_N = 5
SEED = 20260903

# knobs (learning-curve = fast bootstrap; anchors = full §2 bootstrap)
LC_GRID = [145, 200, 300, 450, 650, 900, 1200, 1648]
LC_DRAWS = 4          # random subsamples per grid point
LC_BOOT_B = 80        # bootstrap resamples inside each stability eval
LC_BOOT_R = 2         # restarts per bootstrap resample
FIT_RESTARTS = 12     # restarts for the point (reference) fit
ANCHOR_BOOT_B = 200
ANCHOR_BOOT_R = 3
KS = [2, 3]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


def stratum_series(patients):
    p = patients.copy()
    p["is_black"] = pd.to_numeric(p["is_black"], errors="coerce").fillna(0).astype(int)
    p["education"] = p["education"].fillna("Missing").astype(str)
    return (p["sex"].astype(str) + "_" + p["is_black"].astype(str) + "_" + p["education"]), p


def nt_expected_rate(df, patients, conditions):
    """The single stratum-standardized NT-rate vector subtracted from every group.

    Mirrors build_residual_matrix: NT rate per condition, standardized to the full
    cohort stratum distribution, strata with <5 NT pooled. Returns a fraction (0-1)
    aligned to sorted(conditions).
    """
    strat, p = stratum_series(patients)
    p = p.assign(stratum=strat.values).set_index("ID")
    strata = sorted(p["stratum"].unique())
    weights = {s: float((p["stratum"] == s).mean()) for s in strata}   # target = full cohort

    nt_ids = sorted(p.index[p["label"] == "NT"])
    raw_nt, conds = gc.raw_binary_matrix(df, nt_ids, conditions)        # n_nt x J (0/1)
    nt_strata = p.loc[nt_ids, "stratum"].values
    nt_n = {s: int((nt_strata == s).sum()) for s in strata}
    valid = [s for s in strata if nt_n[s] >= MIN_STRATUM_N]
    pooled = [s for s in strata if nt_n[s] < MIN_STRATUM_N]
    pooled_n = sum(nt_n[s] for s in pooled)
    pooled_w = sum(weights[s] for s in pooled)

    rate = np.zeros(len(conds), dtype=np.float64)
    for s in valid:
        m = nt_strata == s
        if m.sum():
            rate += raw_nt[m].mean(axis=0) * weights[s]
    if pooled_n > 0:
        m = np.isin(nt_strata, pooled)
        if m.sum():
            rate += raw_nt[m].mean(axis=0) * pooled_w
    return rate, conds


def residual_matrix(df, ids, conditions, nt_rate):
    """raw presence - NT rate, clipped at 0, unit-L2 rows (paper's normalize_for_nmf)."""
    raw, _ = gc.raw_binary_matrix(df, ids, conditions)
    resid = raw - nt_rate[None, :]
    X, has_burden = gc.normalize_for_nmf(resid)
    return X, has_burden


def fit_nmf_best(X, k, restarts, seed):
    best = (np.inf, None, None)
    for s in range(restarts):
        m = NMF(n_components=k, init="random", max_iter=1000, solver="mu",
                beta_loss="frobenius", random_state=seed + s)
        W = m.fit_transform(X)
        if m.reconstruction_err_ < best[0]:
            best = (m.reconstruction_err_, m.components_, W)
    return best[1], best[2]                                             # H, W


def bootstrap_stability(X, has_burden, k, H_ref, W_ref, B, restarts, seed):
    """Resample children, refit best-of-restarts, Hungarian-match to H_ref.

    Returns (mean matched cosine, weakest-component mean, dominant-assignment
    concordance) — identical construction to nmf_analysis.bootstrap_stability.
    """
    rng = np.random.default_rng(seed + k)
    ref_dom = W_ref.argmax(axis=1)
    cos_runs, agree = [], []
    n = len(X)
    for b in range(B):
        idx = rng.integers(0, n, n)
        best = (np.inf, None, None)
        for s_ in range(restarts):
            m = NMF(n_components=k, init="random", max_iter=1000, solver="mu",
                    beta_loss="frobenius", random_state=seed + 1000 * b + s_)
            W = m.fit_transform(X[idx])
            if m.reconstruction_err_ < best[0]:
                best = (m.reconstruction_err_, m.components_, W)
        H_b, W_b = best[1], best[2]
        cm = cosine_similarity(H_ref, H_b)
        r_i, c_i = linear_sum_assignment(-cm)
        cos_runs.append(np.array([cm[r, c] for r, c in zip(r_i, c_i)]))
        back = {c: r for r, c in zip(r_i, c_i)}
        dom_b = np.array([back[x] for x in W_b.argmax(axis=1)])
        keep = has_burden[idx]
        agree.append(float((dom_b[keep] == ref_dom[idx][keep]).mean()) if keep.any() else np.nan)
    cos_runs = np.array(cos_runs)
    return float(cos_runs.mean()), float(cos_runs.mean(axis=0).min()), float(np.nanmean(agree))


def stability_of_ids(df, ids, conditions, nt_rate, k, boot_b, boot_r, seed):
    X, hb = residual_matrix(df, ids, conditions, nt_rate)
    Xf, hbf = X[hb], hb[hb]                                             # drop all-zero rows
    if len(Xf) < k * 5:
        return dict(n=len(ids), n_burden=int(hb.sum()), boot_cos=np.nan,
                    boot_worst=np.nan, boot_assign=np.nan)
    H, W = fit_nmf_best(Xf, k, FIT_RESTARTS, seed)
    cos, worst, assign = bootstrap_stability(Xf, np.ones(len(Xf), bool), k, H, W,
                                             boot_b, boot_r, seed)
    return dict(n=len(ids), n_burden=int(hb.sum()), boot_cos=cos,
                boot_worst=worst, boot_assign=assign)


def main():
    sys.stdout = Tee(os.path.join(OUT, "s1_run.log"))
    t0 = time.time()
    conditions = sorted(l.strip() for l in open(KEPT) if l.strip())
    patients = gc.load_patients(FULL_PAT)
    df = gc.load_event_data(FULL_EVT, FULL_PAT)
    nt_rate, conds = nt_expected_rate(df, patients, conditions)
    assert conds == conditions

    ids = {g: sorted(patients.loc[patients["label"] == g, "ID"])
           for g in ("ASD", "ADHD", "Other NDD", "NT")}
    spectrum = sorted(ids["ASD"] + ids["ADHD"] + ids["Other NDD"])
    print(f"NT-expected rate computed over {len(ids['NT'])} NT; spectrum pool = "
          f"{len(spectrum)} (ASD {len(ids['ASD'])} + ADHD {len(ids['ADHD'])} + "
          f"OtherNDD {len(ids['Other NDD'])})")
    print(f"§2 floors: boot cosine >= {gc.config.STABILITY_COSINE_FLOOR}, "
          f"concordance >= {gc.config.ASSIGN_CONCORDANCE_FLOOR}\n")

    rng = np.random.default_rng(SEED)
    rows = []
    for k in KS:
        print(f"===== K={k} learning curve =====")
        for n in LC_GRID:
            draw_stats = []
            for d in range(LC_DRAWS):
                if n >= len(spectrum):
                    sub = spectrum
                else:
                    sub = list(rng.choice(spectrum, size=n, replace=False))
                s = stability_of_ids(df, sub, conditions, nt_rate, k,
                                     LC_BOOT_B, LC_BOOT_R, SEED + 7 * d)
                draw_stats.append(s)
                if n >= len(spectrum):
                    break                                              # full pool: 1 draw
            cos = np.nanmean([s["boot_cos"] for s in draw_stats])
            cos_sd = np.nanstd([s["boot_cos"] for s in draw_stats])
            asg = np.nanmean([s["boot_assign"] for s in draw_stats])
            asg_sd = np.nanstd([s["boot_assign"] for s in draw_stats])
            wor = np.nanmean([s["boot_worst"] for s in draw_stats])
            rows.append(dict(K=k, n=n, draws=len(draw_stats),
                             boot_cos=cos, boot_cos_sd=cos_sd,
                             boot_worst=wor, boot_assign=asg, boot_assign_sd=asg_sd))
            print(f"  n={n:>5}: boot_cos={cos:.3f}±{cos_sd:.3f}  worst={wor:.3f}  "
                  f"assign={asg:.3f}±{asg_sd:.3f}  ({len(draw_stats)} draw)")
    lc = pd.DataFrame(rows)
    lc.to_csv(os.path.join(OUT, "s1_learning_curve.csv"), index=False)

    # ── anchors at full §2 bootstrap ──
    print("\n===== Anchors (full §2 bootstrap) =====")
    anchors = []
    for k in KS:
        a = stability_of_ids(df, ids["ASD"], conditions, nt_rate, k,
                             ANCHOR_BOOT_B, ANCHOR_BOOT_R, SEED)
        a.update(group="ASD-only", K=k)
        anchors.append(a)
        print(f"  ASD-only  K={k}: cos={a['boot_cos']:.3f} worst={a['boot_worst']:.3f} "
              f"assign={a['boot_assign']:.3f}")
        b = stability_of_ids(df, spectrum, conditions, nt_rate, k,
                             ANCHOR_BOOT_B, ANCHOR_BOOT_R, SEED)
        b.update(group="full-spectrum", K=k)
        anchors.append(b)
        print(f"  spectrum  K={k}: cos={b['boot_cos']:.3f} worst={b['boot_worst']:.3f} "
              f"assign={b['boot_assign']:.3f}")
    pd.DataFrame(anchors).to_csv(os.path.join(OUT, "s1_anchors.csv"), index=False)

    _plot(lc, os.path.join(OUT, "s1_learning_curve.pdf"))
    print(f"\nDone in {time.time()-t0:.0f}s. Outputs in {OUT}")


def _plot(lc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    floors = (gc.config.STABILITY_COSINE_FLOOR, gc.config.ASSIGN_CONCORDANCE_FLOOR)
    for ax, (col, sd, floor, title) in zip(axes, [
            ("boot_cos", "boot_cos_sd", floors[0], "Bootstrap matched cosine"),
            ("boot_assign", "boot_assign_sd", floors[1], "Dominant-assignment concordance")]):
        for k in sorted(lc["K"].unique()):
            s = lc[lc["K"] == k]
            ax.errorbar(s["n"], s[col], yerr=s[sd], marker="o", capsize=3, label=f"K={k}")
        ax.axhline(floor, ls="--", color="crimson", lw=1)
        ax.text(lc["n"].max(), floor, f" §2 floor {floor}", color="crimson",
                va="bottom", ha="right", fontsize=8)
        ax.axvline(145, ls=":", color="gray", lw=1)
        ax.text(145, ax.get_ylim()[0], " n=145 (paper)", color="gray", fontsize=8,
                rotation=90, va="bottom")
        ax.set_xlabel("children the axes are estimated from (n)")
        ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Residualized-axis reproducibility rises with spectrum sample size", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
