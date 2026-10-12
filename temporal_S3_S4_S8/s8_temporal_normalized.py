"""S8 (M-II) — temporal factorization done right, or a clean negative.

S4 established that a plain nonnegative CP on the spectrum tensor is *estimable* (inclusion lifts
the density ceiling) but *not valid*: the temporal factor tracks the per-window event-density
envelope (density-artifact guard |r| > 0.7). The envelope is a pure ascertainment signal -- how
much any child is coded at each age -- not developmental biology.

This script attacks that artifact three ways, all built around it, and compares each against the
S4 baseline on the SAME four criteria (estimable / stable / artifact-free / group-separating):

  BASELINE          absolute-age 6-window NT-residual tensor (reproduces S4's artifact).
  (a) REL-TIMING    within-child relative timing: each event placed in a quartile of THAT child's
                    own first->last-code span. The population density envelope is removed by
                    construction -- every child contributes across their own developmental span.
  (b) DENS-EQUAL    the baseline tensor with each age-slice rescaled to equal total mass, so no
                    factor can be driven by high-volume windows; guard re-checked vs the original
                    envelope.
  (c) UTIL-NORM     each (child x window x condition) cell = that condition's SHARE of the child's
                    coding activity in that window (count / child's total events in window),
                    NT-residualized. Raw coding volume is divided out cell-by-cell.

For each tensor we fit best-of-N_RESTARTS nonnegative CP at ranks 2-3 (kernel + bootstrap reused
verbatim from s4) and report: explained variance; restart + bootstrap stability against the §2
floors (cosine >= 0.80, concordance >= 0.60); the density-artifact guard max|r(temporal factor,
per-window density)|; and a NEW group-separation score -- the best per-component ROC-AUC of the
child loadings discriminating ASD from the rest of the spectrum. A genuine temporal finding must
clear ALL FOUR. A method that is stable and artifact-free but does NOT separate groups is a clean,
publishable negative: "once ascertainment is removed, developmental *timing* carries no additional
ASD signal beyond the marginal comorbidity rates S5-S7 already capture."

Outputs (results/):
  s8_temporal_normalized.csv   pool x approach x rank: expl, stability, guard, separation, verdict
  s8_summary.txt               verdict table + interpretation
  s8_run.log
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

import gus_common as gc
import s1_axis_robustness as s1
import s4_spectrum_tensor as s4     # reuse ncp, _khatri_rao, build_tensor, boot_cp (main under __main__)

EPS = 1e-10
OUT = gc.OUT_DIR
RANKS = [2, 3]
N_RESTARTS = 10
BOOT_B = 60
BOOT_RESTARTS = 2
ARTIFACT_R = gc.config.DENSITY_ARTIFACT_R          # 0.7
COS_FLOOR = gc.config.STABILITY_COSINE_FLOOR       # 0.80
CONC_FLOOR = gc.config.ASSIGN_CONCORDANCE_FLOOR    # 0.60
SEP_FLOOR = 0.60                                    # child-loading ASD-vs-rest AUC to count as "separating"
FULL_EDGES, FULL_LABELS = gc.truncated_bins(18, 3)  # S4's "Full 0-18 6w"


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


# ── Tensor builders (each returns X: I x J x K, and dens: per-window density vector) ──
def build_relative(df, ids, conditions, K=4):
    """Within-child relative-timing tensor: event -> quartile of that child's own code span."""
    conds = sorted(conditions); cidx = {c: j for j, c in enumerate(conds)}
    pos = {i: r for r, i in enumerate(ids)}; idset = set(ids)
    X = np.zeros((len(ids), len(conds), K), dtype=np.float32)
    sub = df[df["condition"].isin(conds) & df["ID"].isin(idset)]
    span = sub.groupby("ID")["age_years"].agg(["min", "max"])
    for r in sub.itertuples(index=False):
        i = pos.get(r.ID)
        if i is None:
            continue
        lo, hi = span.at[r.ID, "min"], span.at[r.ID, "max"]
        rel = 0.0 if hi <= lo else (r.age_years - lo) / (hi - lo)
        k = min(int(rel * K), K - 1)
        X[i, cidx[r.condition], k] = 1.0
    dens = (X > 0).sum(axis=(0, 1)).astype(float); dens = dens / (dens.sum() + EPS)
    return X, dens


def build_dens_equalized(df, ids, patients, conditions, edges, labels):
    """S4 baseline tensor with each age-slice rescaled to equal total mass; dens = ORIGINAL envelope."""
    X, dens = s4.build_tensor(df, ids, patients, conditions, edges, labels)  # dens = nonzero-frac envelope
    mass = X.sum(axis=(0, 1))                                                 # per-window residual mass
    scale = np.where(mass > 0, mass.mean() / mass, 1.0)
    Xe = X * scale[None, None, :]
    return Xe, dens                                                           # test factors vs the SAME envelope


def build_util_norm(df, ids, patients, conditions, edges, labels):
    """Utilization-normalized: cell = condition's share of the child's coding in the window, NT-residualized."""
    conds = sorted(conditions); cidx = {c: j for j, c in enumerate(conds)}
    pos = {i: r for r, i in enumerate(ids)}
    nt_ids = set(patients.loc[patients["label"] == "NT", "ID"])
    K = len(edges) - 1
    X = np.zeros((len(ids), len(conds), K), dtype=np.float32)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        dfw = df[(df["age_years"] >= lo) & (df["age_years"] < hi)]
        tot = dfw.groupby("ID").size()                                        # utilization = all events in window
        cc = (dfw[dfw["condition"].isin(conds)].groupby(["ID", "condition"]).size())
        # per-child condition share
        frac = {}
        for (cid, cond), cnt in cc.items():
            t = tot.get(cid, 0)
            if t > 0:
                frac.setdefault(cid, {})[cond] = cnt / t
        # NT expected share per condition (mean over NT children present in window)
        nt_present = [c for c in frac if c in nt_ids]
        nt_share = np.zeros(len(conds))
        if nt_present:
            for cid in nt_present:
                for cond, v in frac[cid].items():
                    nt_share[cidx[cond]] += v
            nt_share /= len(nt_present)
        for cid, d in frac.items():
            i = pos.get(cid)
            if i is None:
                continue
            for cond, v in d.items():
                X[i, cidx[cond], k] = max(0.0, v - nt_share[cidx[cond]])       # residual share, clip 0
    dens = (X > 0).sum(axis=(0, 1)).astype(float); dens = dens / (dens.sum() + EPS)
    return X, dens


# ── Evaluate one prebuilt tensor exactly as S4 did, plus group separation ──
def eval_tensor(X, dens, is_asd, approach, pool):
    I, J, K = X.shape
    X_fit, has_burden = gc.normalize_tensor(X)
    SS_TOT = float(np.sum((X_fit - X_fit.mean()) ** 2))
    rows = []
    for rank in RANKS:
        runs = [s4.ncp(X_fit, rank, s) for s in range(N_RESTARTS)]
        factors, err = min(runs, key=lambda t: t[1])
        A, B, C = factors
        approx = np.reshape(A @ s4._khatri_rao([B, C]).T, X_fit.shape)
        expl = 1 - np.sum((X_fit - approx) ** 2) / SS_TOT if SS_TOT > 0 else 0.0
        sims = []
        for f_run, _ in runs:
            if f_run is factors:
                continue
            cm = cosine_similarity(B.T, f_run[1].T)
            ri, ci = linear_sum_assignment(-cm)
            sims.append(np.mean([cm[r, c] for r, c in zip(ri, ci)]))
        restart_stab = float(np.mean(sims)) if sims else 1.0
        Cn = C / (C.sum(axis=0, keepdims=True) + EPS)
        r_dens = [float(np.corrcoef(Cn[:, c], dens)[0, 1]) for c in range(rank)]
        max_abs_r = max(r_dens, key=abs)
        # group separation: best per-component ASD-vs-rest AUC on child loadings
        seps = []
        for c in range(rank):
            a = roc_auc_score(is_asd, A[:, c])
            seps.append(max(a, 1 - a))
        sep_auc = float(max(seps))
        bc, bw, bconc = s4.boot_cp(X_fit, has_burden, rank, A, B, gc.config.NMF_BOOTSTRAP_SEED)
        stable = bool(bc >= COS_FLOOR and bconc >= CONC_FLOOR)
        clean = bool(abs(max_abs_r) <= ARTIFACT_R)
        separates = bool(sep_auc >= SEP_FLOOR)
        verdict = "REAL-TEMPORAL" if (stable and clean and separates) else (
            "clean-negative" if (stable and clean) else "unusable")
        rows.append(dict(pool=pool, approach=approach, windows=K, R=rank,
                         expl_var=round(float(expl), 3), restart_stab=round(restart_stab, 3),
                         boot_cos=round(bc, 3), boot_worst=round(bw, 3), boot_conc=round(bconc, 3),
                         max_r_density=round(max_abs_r, 3), sep_auc=round(sep_auc, 3),
                         stable=stable, artifact_free=clean, separates=separates, verdict=verdict))
        v = rows[-1]
        print(f"  [{approach:<10} R={rank}] expl={v['expl_var']:.3f} restart={v['restart_stab']:.3f} "
              f"boot_cos={v['boot_cos']:.3f} conc={v['boot_conc']:.3f} |r(dens)|={abs(max_abs_r):.3f} "
              f"sepAUC={v['sep_auc']:.3f} -> {v['verdict']}")
    return rows


def main():
    sys.stdout = Tee(os.path.join(OUT, "s8_run.log"))
    t0 = time.time()
    conditions = sorted(l.strip() for l in open(s1.KEPT) if l.strip())
    patients = gc.load_patients(s1.FULL_PAT)
    df = gc.load_event_data(s1.FULL_EVT, s1.FULL_PAT)
    ids_g = {g: sorted(patients.loc[patients["label"] == g, "ID"])
             for g in ("ASD", "ADHD", "Other NDD")}
    spectrum = sorted(ids_g["ASD"] + ids_g["ADHD"] + ids_g["Other NDD"])
    lab = patients.set_index("ID")["label"].to_dict()
    is_asd = np.array([1 if lab[i] == "ASD" else 0 for i in spectrum])
    print(f"spectrum n={len(spectrum)} (ASD={is_asd.sum()}); ranks {RANKS}; restarts {N_RESTARTS}; "
          f"bootstrap {BOOT_B}x{BOOT_RESTARTS}\n"
          f"floors: cos>={COS_FLOOR} conc>={CONC_FLOOR}; artifact |r(dens)|>{ARTIFACT_R}; "
          f"separation AUC>={SEP_FLOOR}\n")
    s4.BOOT_B = BOOT_B; s4.BOOT_RESTARTS = BOOT_RESTARTS      # make s4.boot_cp use our lighter settings

    builders = [
        ("BASELINE", lambda: s4.build_tensor(df, spectrum, patients, conditions, FULL_EDGES, FULL_LABELS)),
        ("REL-TIMING", lambda: build_relative(df, spectrum, conditions, K=4)),
        ("DENS-EQUAL", lambda: build_dens_equalized(df, spectrum, patients, conditions, FULL_EDGES, FULL_LABELS)),
        ("UTIL-NORM", lambda: build_util_norm(df, spectrum, patients, conditions, FULL_EDGES, FULL_LABELS)),
    ]
    all_rows = []
    for name, build in builders:
        print(f"===== approach: {name} =====")
        X, dens = build()
        print(f"  tensor {X.shape}  density-envelope={np.round(dens, 3)}")
        all_rows += eval_tensor(X, dens, is_asd, name, "spectrum(1648)")

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(OUT, "s8_temporal_normalized.csv"), index=False)

    with open(os.path.join(OUT, "s8_summary.txt"), "w") as f:
        f.write(f"Temporal factorization, ascertainment-aware (spectrum n={len(spectrum)}, "
                f"ASD={int(is_asd.sum())})\n")
        f.write(f"floors: boot_cos>={COS_FLOOR}, conc>={CONC_FLOOR}; artifact |r(dens)|>{ARTIFACT_R}; "
                f"separation AUC>={SEP_FLOOR}\n\n")
        f.write(out.to_string(index=False) + "\n\n")
        base_art = out[(out.approach == "BASELINE")]["artifact_free"].eq(False).any()
        fixed = out[out.approach != "BASELINE"]
        any_real = fixed["verdict"].eq("REAL-TEMPORAL").any()
        cleaned = fixed["artifact_free"].all()
        f.write(f"Baseline shows density artifact: {bool(base_art)}\n")
        f.write(f"All normalized approaches artifact-free: {bool(cleaned)}\n")
        f.write(f"Any normalized approach yields a REAL (stable+clean+separating) temporal factor: "
                f"{bool(any_real)}\n")
        if not any_real:
            f.write("=> CLEAN NEGATIVE: after removing ascertainment, developmental timing adds no "
                    "ASD signal beyond the marginal comorbidity rates (S5-S7).\n")
    print(open(os.path.join(OUT, "s8_summary.txt")).read())
    print(f"Done in {time.time()-t0:.0f}s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
