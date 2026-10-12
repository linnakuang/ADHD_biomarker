"""S4 — Spectrum CP/PARAFAC: does inclusion turn an unestimable temporal model VALID?

S3 showed the density ceiling that made A1/A2 data-availability artifacts (thin slices)
collapses on the spectrum (1,648): the full six-window 0-18 factorization goes from 1/6 to
6/6 windows clearing the §2 rule. But estimability is not validity. A1's failure mode was
the temporal factor tracking the per-window event-density envelope (|r| up to ~0.99), and
that envelope's SHAPE is roughly n-invariant (10x children scales every window ~equally).
So more data fixes thin slices but need NOT decouple the temporal factor from density.

This script runs the SAME nonnegative CP as A1 (kernel copied — a1_dx_anchored_cp.py runs
on import) on both pools (ASD-only 145, spectrum 1,648) across two binnings, and reports
side by side: explained variance, restart stability, bootstrap matched-cosine / worst /
concordance (§2 floors), and the density-artifact guard max|r(temporal, density)|. Verdict
per (pool × scheme × R): estimable? stable (§2)? artifact-free (guard)?

Outputs: results/s4_spectrum_tensor.csv, results/s4_run.log
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

import gus_common as gc
import s1_axis_robustness as s1

EPS = 1e-10
OUT = gc.OUT_DIR
MEDIAN_DX = 3.253
RANKS = [2, 3, 4]
N_RESTARTS = 15
BOOT_B = 120
BOOT_RESTARTS = 2
BOOT_SEED = gc.config.NMF_BOOTSTRAP_SEED
ARTIFACT_R = gc.config.DENSITY_ARTIFACT_R          # 0.7
COS_FLOOR = gc.config.STABILITY_COSINE_FLOOR       # 0.80
CONC_FLOOR = gc.config.ASSIGN_CONCORDANCE_FLOOR    # 0.60


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


# ── Non-negative CP (PARAFAC) by multiplicative updates — verbatim from a1 ──
def _unfold(X, n):
    return np.reshape(np.moveaxis(X, n, 0), (X.shape[n], -1))

def _khatri_rao(mats):
    r = mats[0].shape[1]; out = mats[0]
    for m in mats[1:]:
        out = np.einsum("ir,jr->ijr", out, m).reshape(-1, r)
    return out

def ncp(X, rank, seed, max_iter=500, tol=1e-6):
    rng = np.random.default_rng(seed)
    factors = [rng.random((dim, rank)) + 0.1 for dim in X.shape]
    normX = np.linalg.norm(X); prev = None
    for _ in range(max_iter):
        for n in range(X.ndim):
            others = [factors[m] for m in range(X.ndim) if m != n]
            gram = np.ones((rank, rank))
            for m in others:
                gram *= m.T @ m
            numer = _unfold(X, n) @ _khatri_rao(others)
            factors[n] *= numer / (factors[n] @ gram + EPS)
        approx = np.reshape(factors[0] @ _khatri_rao(factors[1:]).T, X.shape)
        err = np.linalg.norm(X - approx) / (normX + EPS)
        if prev is not None and abs(prev - err) < tol:
            break
        prev = err
    return factors, err


def build_tensor(df, ids, patients, conditions, edges, labels):
    """I x J x K within-window NT-residual tensor (clip at 0), + per-window density vector."""
    slices, dens = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        df_w = df[(df["age_years"] >= lo) & (df["age_years"] < hi)]
        nt_rate, _ = s1.nt_expected_rate(df_w, patients, conditions)
        raw, _ = gc.raw_binary_matrix(df_w, ids, conditions)
        clipped = np.clip(raw - nt_rate[None, :], 0, None)
        slices.append(clipped)
        dens.append(float((clipped > 0).sum()))
    X = np.stack(slices, axis=2)                       # I x J x K
    dens = np.array(dens); dens = dens / (dens.sum() + EPS)
    return X, dens


def boot_cp(X_fit, has_burden, rank, A_ref, B_ref, seed):
    rng = np.random.default_rng(seed + rank)
    ref_dom = A_ref.argmax(axis=1); H_ref = B_ref.T
    cos_runs, agree = [], []; n = X_fit.shape[0]
    for b in range(BOOT_B):
        idx = rng.integers(0, n, n); Xb = X_fit[idx]
        best_err, fb = np.inf, None
        for s_ in range(BOOT_RESTARTS):
            f_, e_ = ncp(Xb, rank, seed + 1000 * b + s_)
            if e_ < best_err:
                best_err, fb = e_, f_
        A_b, B_b = fb[0], fb[1]
        cm = cosine_similarity(H_ref, B_b.T)
        r_i, c_i = linear_sum_assignment(-cm)
        cos_runs.append(np.array([cm[r, c] for r, c in zip(r_i, c_i)]))
        back = {c: r for r, c in zip(r_i, c_i)}
        dom_b = np.array([back[x] for x in A_b.argmax(axis=1)])
        keep = has_burden[idx]
        agree.append(float((dom_b[keep] == ref_dom[idx][keep]).mean()) if keep.any() else np.nan)
    cr = np.array(cos_runs)
    return float(cr.mean()), float(cr.mean(axis=0).min()), float(np.nanmean(agree))


def eval_pool_scheme(df, patients, conditions, ids, pool, sch_name, edges, labels):
    X, dens = build_tensor(df, ids, patients, conditions, edges, labels)
    I, J, K = X.shape
    X_fit, has_burden = gc.normalize_tensor(X)
    SS_TOT = float(np.sum((X_fit - X_fit.mean()) ** 2))
    thin = int((( X > 0).sum(axis=0) < gc.config.DENSITY_FLOOR_CHILDREN).sum())
    thin_pct = round(thin / (J * K) * 100)
    rows = []
    for rank in RANKS:
        runs = [ncp(X_fit, rank, s) for s in range(N_RESTARTS)]
        factors, err = min(runs, key=lambda t: t[1])
        A, B, C = factors
        approx = np.reshape(A @ _khatri_rao([B, C]).T, X_fit.shape)
        expl = 1 - np.sum((X_fit - approx) ** 2) / SS_TOT if SS_TOT > 0 else 0.0
        # restart stability (condition mode)
        sims = []
        for f_run, _ in runs:
            if f_run is factors:
                continue
            cm = cosine_similarity(B.T, f_run[1].T)
            ri, ci = linear_sum_assignment(-cm)
            sims.append(np.mean([cm[r, c] for r, c in zip(ri, ci)]))
        restart_stab = float(np.mean(sims)) if sims else 1.0
        # density-artifact guard
        Cn = C / (C.sum(axis=0, keepdims=True) + EPS)
        r_dens = [float(np.corrcoef(Cn[:, c], dens)[0, 1]) for c in range(rank)]
        max_abs_r = max(r_dens, key=abs)
        bc, bw, bconc = boot_cp(X_fit, has_burden, rank, A, B, BOOT_SEED)
        rows.append(dict(pool=pool, scheme=sch_name, windows=K, R=rank,
                         thin_pct=thin_pct, expl_var=round(float(expl), 3),
                         restart_stab=round(restart_stab, 3),
                         boot_cos=round(bc, 3), boot_worst=round(bw, 3),
                         boot_conc=round(bconc, 3),
                         max_r_density=round(max_abs_r, 3),
                         stable=bool(bc >= COS_FLOOR and bconc >= CONC_FLOOR),
                         artifact=bool(abs(max_abs_r) > ARTIFACT_R)))
        v = rows[-1]
        print(f"  [{pool:<14} {sch_name:<16} R={rank}] thin={thin_pct:>2}%  "
              f"expl={v['expl_var']:.3f}  restart={v['restart_stab']:.3f}  "
              f"boot_cos={v['boot_cos']:.3f} worst={v['boot_worst']:.3f} "
              f"conc={v['boot_conc']:.3f}  |r(dens)|={abs(max_abs_r):.3f}  "
              f"-> {'STABLE' if v['stable'] else 'unstable'},"
              f"{'ARTIFACT' if v['artifact'] else ' clean'}")
    return rows


def main():
    sys.stdout = Tee(os.path.join(OUT, "s4_run.log"))
    t0 = time.time()
    conditions = sorted(l.strip() for l in open(s1.KEPT) if l.strip())
    patients = gc.load_patients(s1.FULL_PAT)
    df = gc.load_event_data(s1.FULL_EVT, s1.FULL_PAT)
    ids = {g: sorted(patients.loc[patients["label"] == g, "ID"])
           for g in ("ASD", "ADHD", "Other NDD")}
    pools = {"ASD(145)": ids["ASD"],
             "spectrum(1648)": sorted(ids["ASD"] + ids["ADHD"] + ids["Other NDD"])}
    schemes = [("Dx-anchored 3w", *gc.dx_anchored_bins(MEDIAN_DX, 3, 3.0)),
               ("Full 0-18 6w", *gc.truncated_bins(18, 3))]
    print(f"CP rank scan {RANKS}; restarts={N_RESTARTS}; bootstrap {BOOT_B}x{BOOT_RESTARTS}. "
          f"§2 floors cos>={COS_FLOOR} conc>={CONC_FLOOR}; artifact |r(density)|>{ARTIFACT_R}\n")
    all_rows = []
    for sch_name, edges, labels in schemes:
        print(f"===== scheme: {sch_name} ({len(labels)} windows) =====")
        for pool, pids in pools.items():
            all_rows += eval_pool_scheme(df, patients, conditions, pids, pool,
                                         sch_name, edges, labels)
    out = pd.DataFrame(all_rows)
    path = os.path.join(OUT, "s4_spectrum_tensor.csv")
    out.to_csv(path, index=False)
    print(f"\nDone in {time.time()-t0:.0f}s. Saved {path}")


if __name__ == "__main__":
    main()
