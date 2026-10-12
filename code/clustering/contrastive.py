from __future__ import annotations
import sys
from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
from scipy.optimize import nnls, linear_sum_assignment
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from docx_table import write_docx_table  # noqa: E402
import nmf_analysis as nm  # noqa: E402
from nmf_analysis import label_profiles, SHORT  # noqa: E402

# ── Matrices ────────────────────────────────────────────────────────────────
def pattern_matrix(p, ev, group, conds):
    M = nm.condition_matrix(p, ev, group, conds)
    X = M.values.astype(float)
    keep = X.sum(axis=1) > 0
    n_drop = int((~keep).sum())
    X = X[keep]
    return X / np.linalg.norm(X, axis=1, keepdims=True), M.index[keep].tolist(), n_drop

def fit_basis(X, k, restarts=None):
    _, _, H, _, _ = nm._fit_restarts(X, k, config.SEED, n_restarts=restarts)
    return H

def project(X, H):
    W = np.vstack([nnls(H.T, x)[0] for x in X])
    fitted = W @ H
    ss = np.sum((X - fitted) ** 2, axis=1) / np.sum(X ** 2, axis=1)
    return W, np.clip(X - fitted, 0, None), 1 - ss

def shape_components(R, k, min_norm, restarts=None):
    norms = np.linalg.norm(R, axis=1)
    keep = norms > max(min_norm, 1e-12)
    if keep.sum() < k * 10:
        return None, int(keep.sum())
    Xr = R[keep] / norms[keep, None]
    _, _, H, _, _ = nm._fit_restarts(Xr, k, config.SEED, n_restarts=restarts)
    return H, int(keep.sum())

def best_cosine(H_primary, H_resid):
    return cosine_similarity(H_primary, H_resid).max(axis=1)

def align_to_primary(H_primary, H_resid):
    if H_primary.shape[0] != H_resid.shape[0]:
        return H_resid, False, None
    cm = cosine_similarity(H_primary, H_resid)
    rows, cols = linear_sum_assignment(-cm)
    order = np.arange(H_resid.shape[0])   # not np.empty: an unassigned slot
    order[rows] = cols                    # would index with uninitialised memory
    clean = bool((order == cm.argmax(axis=1)).all())
    assigned = np.array([cm[r, order[r]] for r in range(len(order))])
    return H_resid[order], clean, assigned

# ── One cross-fit fold ──────────────────────────────────────────────────────
def one_fold(X_nt, train, held, X_adhd, H_primary, k_nt, k_res, q, restarts=None):  # noqa: PLR0913
    H = fit_basis(X_nt[train], k_nt, restarts)
    _, R_nt, frac_nt = project(X_nt[held], H)
    _, R_a, frac_a = project(X_adhd, H)

    thr = np.quantile(np.linalg.norm(R_nt, axis=1), q) if q > 0 else 0.0
    H_a, n_a = shape_components(R_a, k_res, thr, restarts)
    H_n, n_n = shape_components(R_nt, k_res, thr, restarts)
    if H_a is None or H_n is None:
        return None
    cm_a = cosine_similarity(H_primary, H_a)
    cm_n = cosine_similarity(H_primary, H_n)
    cos_a, cos_n = cm_a.max(axis=1), cm_n.max(axis=1)
    H_a_aligned, clean, assigned = align_to_primary(H_primary, H_a)
    return dict(H=H, R_a=R_a, R_nt=R_nt, frac_a=frac_a, frac_nt=frac_nt,
                H_a=H_a_aligned, H_a_raw=H_a, H_n=H_n,
                best_a=cm_a.argmax(axis=1), best_n=cm_n.argmax(axis=1),
                aligned_clean=clean, assigned=assigned,
                cos_a=cos_a, cos_n=cos_n,
                gap=cos_a - cos_n, thr=thr, n_a=n_a, n_n=n_n, held=held)

def crossfit(X_nt, X_adhd, H_primary, k_nt, k_res, q, restarts=None, seed=None):
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    idx = rng.permutation(len(X_nt))
    h = len(idx) // 2
    folds = [(idx[:h], idx[h:]), (idx[h:], idx[:h])]
    return [one_fold(X_nt, tr, he, X_adhd, H_primary, k_nt, k_res, q, restarts)
            for tr, he in folds]

# ── Empirical null for the cosine gap ───────────────────────────────────────
def _null_once(seed, X_nt, H_primary, k_nt, k_res, q):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_nt))
    t = len(idx) // 3
    tr, case, ctrl = idx[:t], idx[t:2 * t], idx[2 * t:]
    H = fit_basis(X_nt[tr], k_nt, config.CONTRAST_NULL_RESTARTS)
    _, R_c, _ = project(X_nt[case], H)
    _, R_k, _ = project(X_nt[ctrl], H)
    thr = np.quantile(np.linalg.norm(R_k, axis=1), q) if q > 0 else 0.0
    H_c, _ = shape_components(R_c, k_res, thr, config.CONTRAST_NULL_RESTARTS)
    H_k, _ = shape_components(R_k, k_res, thr, config.CONTRAST_NULL_RESTARTS)
    if H_c is None or H_k is None:
        return None
    return best_cosine(H_primary, H_c) - best_cosine(H_primary, H_k)

def null_gaps(X_nt, H_primary, k_nt, k_res, q):
    seeds = np.random.SeedSequence(config.NMF_BOOTSTRAP_SEED).spawn(
        config.CONTRAST_NULL_ITER)
    out = Parallel(n_jobs=config.CONTRAST_NULL_JOBS, verbose=1)(
        delayed(_null_once)(sd, X_nt, H_primary, k_nt, k_res, q) for sd in seeds)
    out = [o for o in out if o is not None]
    return np.array(out)

def percentile_ci(null, pct=95, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    draws = np.array([np.percentile(null[rng.integers(0, len(null), len(null))],
                                    pct, axis=0) for _ in range(n_boot)])
    return np.percentile(draws, [2.5, 97.5], axis=0)

# ── Figure ────────────────────────────────────────────────────────────────
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v

def tick_label(name, width=18):
    """Condition name shortened for an axis tick, without truncation.

    The parenthetical in these names is a gloss, not part of the identity --
    "Mood disorders (depression, bipolar)" and "Sleep disorders (organic)" lose
    nothing by dropping it, and across the retained conditions doing so produces
    no duplicate labels. What remains is wrapped rather than cut, so no label
    ends in an ellipsis the reader cannot resolve.
    """
    return "\n".join(textwrap.wrap(re.sub(r"\s*\([^)]*\)", "", name).strip(), width))

def draw(r_a, r_nt, folds, H_primary, conds, labels, null_p95, path):
    k = H_primary.shape[0]
    fig = plt.figure(figsize=(14.5, 13.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.56, 0.56],
                          hspace=0.95, wspace=0.28)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC = [fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])]

    # (A) ECDF with the NT upper-tail cut-points
    for v, col, lab in ((r_nt, "#2E7D8C", "NT held out"), (r_a, "#C0392B", "ADHD")):
        x = np.sort(v)
        axA.step(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=2,
                 label=f"{lab}  median {np.median(v):.3f}, n={len(v)}")
    tail = []
    for (pct, ls), ytxt in zip(((90, ":"), (95, "--")), (0.04, 0.13)):
        cut = np.percentile(r_nt, pct)
        axA.axvline(cut, color="#555", lw=1, ls=ls)
        axA.text(cut, ytxt, f" NT p{pct}", fontsize=8, color="#555",
                 ha="left", va="bottom")
        tail.append(f"above NT p{pct}: {100 * (r_a > cut).mean():.1f}% of ADHD "
                    f"({100 - pct}% expected)")
    axA.text(0.03, 0.97, "\n".join(tail), transform=axA.transAxes, fontsize=8,
             ha="left", va="top", color="#333",
             bbox=dict(fc="white", ec="#ccc", lw=.6, pad=4))
    axA.set_xlabel("Positive residual magnitude  ‖r⁺‖")
    axA.set_ylabel("Cumulative proportion of children")
    axA.set_ylim(0, 1.0)
    axA.set_title("(A) Residual magnitude is not elevated in ADHD",
                  fontsize=11, fontweight="bold", loc="left")
    axA.legend(frameon=False, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=1)
    axA.spines[["top", "right"]].set_visible(False)

    # (B) ADHD-residual vs NT-residual cosine, per profile
    y = np.arange(k)[::-1]
    for c in range(k):
        ca = np.mean([f["cos_a"][c] for f in folds])
        cn = np.mean([f["cos_n"][c] for f in folds])
        axB.plot([cn, ca], [y[c], y[c]], color="#999", lw=2, zorder=1)
        axB.scatter(cn, y[c], s=95, color="#2E7D8C", zorder=2,
                    label="nearest NT residual component" if c == 0 else None)
        axB.scatter(ca, y[c], s=95, color="#C0392B", zorder=2,
                    label="nearest ADHD residual component" if c == 0 else None)
        axB.text(1.02, y[c], f"{ca - cn:+.2f}", va="center", fontsize=9,
                 fontweight="bold" if (ca - cn) > null_p95[c] else "normal")
    axB.text(1.02, 1.01, "gap", transform=axB.get_xaxis_transform(),
             fontsize=9, style="italic", color="#555", va="bottom")
    axB.set_ylim(-0.5, k - 0.5)
    axB.set_yticks(y)
    axB.set_yticklabels(labels, fontsize=9)
    axB.set_xlim(0, 1.22)
    axB.set_xticks(np.arange(0, 1.01, 0.2))
    axB.set_xlabel("Cosine similarity to the nearest residual component")
    axB.set_title("(B) Only the psychiatric profile is enriched in the ADHD residual",
                  fontsize=11, fontweight="bold", loc="left")
    axB.legend(frameon=False, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=1)
    axB.grid(axis="x", alpha=.25)
    axB.spines[["top", "right"]].set_visible(False)

    # (C) the two largest-gap profiles, weaker first so the contrast builds
    mean_gap = np.array([np.mean([f["gap"][c] for f in folds]) for c in range(k)])
    show = sorted(np.argsort(mean_gap)[-2:], key=lambda c: mean_gap[c])
    f0 = folds[0]
    for ax, c in zip(axC, show):
        src = {"Primary profile": unit(H_primary[c]),
               "ADHD residual": unit(f0["H_a_raw"][f0["best_a"][c]]),
               "NT residual": unit(f0["H_n"][f0["best_n"][c]])}
        rows = list(pd.Series(H_primary[c], index=conds)
                    .sort_values(ascending=False)
                    .head(config.NMF_TOP_CONDITIONS * 2).index)
        mat = np.array([[src[s_][conds.index(r)] for r in rows] for s_ in src])
        ax.imshow(mat, aspect="auto", cmap="Purples", vmin=0, vmax=mat.max())
        ax.set_yticks(range(len(src)))
        ax.set_yticklabels(list(src), fontsize=9)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([tick_label(r) for r in rows], fontsize=7, ha="center")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if mat[i, j] > mat.max() * 0.55 else "#333")
        verdict = ("reproduced in the ADHD residual but also present in NT"
                   if mean_gap[c] <= config.CONTRAST_GAP_FLOOR else
                   "reproduced in the ADHD residual, weak in NT")
        ax.set_title(f"(C{list(show).index(c) + 1}) {labels[c]} — gap "
                     f"{mean_gap[c]:+.2f}: {verdict}",
                     fontsize=11, fontweight="bold", loc="left")

    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

def draw_residual_heatmap(H_resid, conds, xlab, path):
    """sF6: the full residual-component loadings, moved out of the main figure."""
    rows = []
    for c in range(H_resid.shape[0]):
        top = pd.Series(H_resid[c], index=conds).sort_values(ascending=False)
        rows += [x for x in top.head(config.NMF_TOP_CONDITIONS).index if x not in rows]
    mat = np.array([[H_resid[c, conds.index(r)] for c in range(H_resid.shape[0])]
                    for r in rows])
    fig, ax = plt.subplots(figsize=(1.6 * H_resid.shape[0] + 4, 0.30 * len(rows) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="Purples", vmin=0)
    ax.set_xticks(range(H_resid.shape[0]))
    ax.set_xticklabels(xlab, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title("ADHD positive-residual components. Columns are a one-to-one "
                 "assignment to the primary profiles, not a demonstrated "
                 "correspondence; see the cosine on each header.", fontsize=9)
    fig.colorbar(im, ax=ax, label="Residual component weight", fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

def run():
    config.ensure(config.CLUSTER_RESULTS)
    p, ev = nm.load()
    code, cw = nm.build_partition(p)
    conds, nt_rate = nm.select_conditions(p, ev, code, cw, verbose=False)
    k_nt, k_res, q = config.NT_BASIS_K, config.NMF_K, config.CONTRAST_MIN_RESID_Q

    X_a, ids_a, drop_a = pattern_matrix(p, ev, "ADHD", conds)
    X_nt, _, drop_nt = pattern_matrix(p, ev, "NT", conds)
    print(f"Contrastive: {len(X_a)} ADHD, {len(X_nt)} NT, {len(conds)} conditions, "
          f"NT basis K={k_nt}")
    print(f"  dropped as all-zero (no retained condition): {drop_a} ADHD, {drop_nt} NT")

    X_p, _, _ = nm.residual_matrix(p, ev, conds, nt_rate)
    _, (_, H_primary) = nm._fit(X_p, k_res, config.SEED)

    # ── primary: two-fold cross-fit, same basis for both arms ──
    folds = crossfit(X_nt, X_a, H_primary, k_nt, k_res, q)
    if any(f is None for f in folds):
        raise SystemExit("A fold had too few children above the residual threshold.")
    r_a = np.linalg.norm(folds[0]["R_a"], axis=1)
    r_nt = np.linalg.norm(folds[0]["R_nt"], axis=1)
    mw = mannwhitneyu(r_a, r_nt, alternative="two-sided")
    print("\n  Positive residual magnitude (fold A basis, both arms):")
    print(f"    ADHD median {np.median(r_a):.3f}   NT held out {np.median(r_nt):.3f}   "
          f"difference {np.median(r_a) - np.median(r_nt):+.3f}, MWU p={mw.pvalue:.2g}")
    print("  NT-reconstructable fraction (squared pattern magnitude recovered):")
    print(f"    ADHD median {np.median(folds[0]['frac_a']):.1%}   "
          f"NT held out {np.median(folds[0]['frac_nt']):.1%}")
    nt_held = X_nt[folds[0]["held"]]
    rho_a = spearmanr(r_a, (X_a > 0).sum(1)).correlation
    rho_n = spearmanr(r_nt, (nt_held > 0).sum(1)).correlation
    print(f"    Spearman of ‖r⁺‖ with condition count: "
          f"ADHD {rho_a:+.3f}, NT {rho_n:+.3f}  "
          + ("(same sign in NT -> a property of the projection, not of ADHD)"
             if np.sign(rho_a) == np.sign(rho_n) else "(differs in sign from NT)"))

    # ||r+|| falls with condition count in BOTH groups, and the ADHD children
    # carry more conditions, so a crude group comparison of residual magnitude is
    # confounded by burden. Compare within strata of condition count instead.
    n_a_cond, n_nt_cond = (X_a > 0).sum(1), (nt_held > 0).sum(1)
    edges = np.unique(np.quantile(np.concatenate([n_a_cond, n_nt_cond]),
                                  [0, .2, .4, .6, .8, 1.0]))
    print("\n  Residual magnitude within strata of condition count:")
    print(f"    {'conditions':<14}{'n ADHD':>8}{'n NT':>7}{'ADHD':>9}{'NT':>8}{'diff':>9}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        ma = (n_a_cond >= lo) & (n_a_cond < hi if hi < edges[-1] else n_a_cond <= hi)
        mn = (n_nt_cond >= lo) & (n_nt_cond < hi if hi < edges[-1] else n_nt_cond <= hi)
        if ma.sum() < 10 or mn.sum() < 10:
            continue
        da, dn = np.median(r_a[ma]), np.median(r_nt[mn])
        print(f"    {f'{lo:.0f}-{hi:.0f}':<14}{ma.sum():>8}{mn.sum():>7}"
              f"{da:>9.3f}{dn:>8.3f}{da - dn:>+9.3f}")
    print(f"    (crude difference {np.median(r_a) - np.median(r_nt):+.3f}; if the "
          f"within-stratum differences are near zero, the crude deficit is a "
          f"burden-composition effect)")

    print(f"\n  Cosine gap by fold (residual K={k_res}, "
          f"threshold = NT p{int(q * 100)} of ‖r⁺‖):")
    print(f"    {'profile':<9}{'foldA ADHD':>11}{'foldA NT':>10}{'gap':>8}"
          f"{'foldB ADHD':>12}{'foldB NT':>10}{'gap':>8}")
    for c in range(k_res):
        a, b = folds
        print(f"    P{c + 1:<8}{a['cos_a'][c]:>11.3f}{a['cos_n'][c]:>10.3f}"
              f"{a['gap'][c]:>+8.3f}{b['cos_a'][c]:>12.3f}{b['cos_n'][c]:>10.3f}"
              f"{b['gap'][c]:>+8.3f}")

    # ── empirical null for the gap ──
    print(f"\n  Empirical NT null for the gap ({config.CONTRAST_NULL_ITER} splits, "
          f"{config.CONTRAST_NULL_JOBS} jobs):")
    null = null_gaps(X_nt, H_primary, k_nt, k_res, q)
    null_p95 = np.percentile(null, 95, axis=0)
    lo, hi = percentile_ci(null, 95)
    print(f"    {len(null)} usable replicates")
    for c in range(k_res):
        obs = min(folds[0]["gap"][c], folds[1]["gap"][c])
        resolved = "resolved" if (obs < lo[c] or obs > hi[c]) else \
                   "NOT resolved at this number of splits"
        print(f"    P{c + 1}: null p95 {null_p95[c]:+.3f} "
              f"[{lo[c]:+.3f}, {hi[c]:+.3f}]   weaker-fold gap {obs:+.3f}  -> {resolved}")

    # ── robustness: the VERDICT across K_NT, not just the score ranking ──
    print("\n  Cosine gap across normative rank (mean of the two folds):")
    print("    " + "profile".ljust(9) + "".join(f"K_NT={kk}".rjust(10)
                                                for kk in config.NT_BASIS_K_ROBUST))
    gaps_by_k = {}
    for kk in config.NT_BASIS_K_ROBUST:
        fs = crossfit(X_nt, X_a, H_primary, kk, k_res, q)
        gaps_by_k[kk] = (np.mean([f["gap"] for f in fs], axis=0)
                         if all(f is not None for f in fs) else np.full(k_res, np.nan))
    for c in range(k_res):
        print(f"    P{c + 1:<8}" + "".join(f"{gaps_by_k[kk][c]:>+10.3f}"
                                           for kk in config.NT_BASIS_K_ROBUST))

    print("\n  Cosine gap across residual rank (mean of the two folds):")
    print("    " + "profile".ljust(9) + "".join(f"K_res={kr}".rjust(10)
                                                for kr in config.CONTRAST_RESID_K_SCAN))
    gaps_by_kr = {}
    for kr in config.CONTRAST_RESID_K_SCAN:
        fs = crossfit(X_nt, X_a, H_primary, k_nt, kr, q)
        gaps_by_kr[kr] = (np.mean([f["gap"] for f in fs], axis=0)
                          if all(f is not None for f in fs) else np.full(k_res, np.nan))
    for c in range(k_res):
        print(f"    P{c + 1:<8}" + "".join(f"{gaps_by_kr[kr][c]:>+10.3f}"
                                           for kr in config.CONTRAST_RESID_K_SCAN))

    print("\n  Cosine gap across the residual-magnitude threshold:")
    print("    " + "profile".ljust(9) + "".join(f"NT p{int(qq * 100)}".rjust(10)
                                                for qq in config.CONTRAST_MIN_RESID_Q_SENS))
    gaps_by_q = {}
    for qq in config.CONTRAST_MIN_RESID_Q_SENS:
        fs = crossfit(X_nt, X_a, H_primary, k_nt, k_res, qq)
        gaps_by_q[qq] = (np.mean([f["gap"] for f in fs], axis=0)
                         if all(f is not None for f in fs) else np.full(k_res, np.nan))
    for c in range(k_res):
        print(f"    P{c + 1:<8}" + "".join(f"{gaps_by_q[qq][c]:>+10.3f}"
                                           for qq in config.CONTRAST_MIN_RESID_Q_SENS))

    # ── verdict: replicate in both folds, clear the floors, beat the null ──
    print("\n  Verdict (must hold in BOTH folds):")
    body = []
    for c in range(k_res):
        a, b = folds
        mean_cos = (a["cos_a"][c] + b["cos_a"][c]) / 2
        both = (min(a["cos_a"][c], b["cos_a"][c]) >= config.CONTRAST_STABILITY_COSINE
                and min(a["gap"][c], b["gap"][c]) >= config.CONTRAST_GAP_FLOOR
                and min(a["gap"][c], b["gap"][c]) > null_p95[c])
        verdict = "ADHD-ENRICHED" if both else (
            "reconstructable from NT structure"
            if max(a["cos_n"][c], b["cos_n"][c]) >= config.CONTRAST_STABILITY_COSINE
            else "inconclusive")
        top = ", ".join(pd.Series(H_primary[c], index=conds)
                        .sort_values(ascending=False).head(3).index)
        print(f"    P{c + 1}: cos {mean_cos:.3f}, gap "
              f"{a['gap'][c]:+.3f}/{b['gap'][c]:+.3f} vs null {null_p95[c]:+.3f}  "
              f"-> {verdict}   [{top}]")
        body.append([f"P{c + 1}", f"{a['cos_a'][c]:.3f} / {b['cos_a'][c]:.3f}",
                     f"{a['cos_n'][c]:.3f} / {b['cos_n'][c]:.3f}",
                     f"{a['gap'][c]:+.3f} / {b['gap'][c]:+.3f}",
                     f"{null_p95[c]:+.3f}", verdict])

    comp = [f"P{i + 1}" for i in range(k_res)]
    theme_of = {pn: SHORT[th] for th, pn in
                label_profiles(pd.DataFrame(H_primary, index=comp,
                                            columns=conds)).items()}
    asg = folds[0]["assigned"]
    xlab = [f"{c}\n{theme_of.get(c, '')}\ncos {asg[i]:.2f}"
            for i, c in enumerate(comp)]
    labels = [f"{c} {theme_of.get(c, '')}" for c in comp]
    for i, f in enumerate(folds):
        if not f["aligned_clean"]:
            print(f"  note: in fold {'AB'[i]} two profiles share a closest residual "
                  f"component, so no one-to-one descent from the primary profiles "
                  f"exists; the sF6 columns are an assignment, not a correspondence")
    for pct in (90, 95):
        cut = np.percentile(r_nt, pct)
        print(f"  ADHD above the NT p{pct} of ‖r⁺‖ ({cut:.3f}): "
              f"{100 * (r_a > cut).mean():.1f}%  (expected {100 - pct}% under the null)")
    draw(r_a, r_nt, folds, H_primary, conds, labels, null_p95,
         config.CLUSTER_RESULTS / "F4_contrastive.pdf")
    draw_residual_heatmap(folds[0]["H_a"], conds, xlab,
                          config.CLUSTER_RESULTS / "sF6_adhd_residual_components.pdf")
    write_docx_table(
        config.CLUSTER_RESULTS / "sTable12_contrastive.docx",
        "Supplementary Table S12. Correspondence of each multimorbidity profile "
        "with the ADHD and the held-out neurotypical positive residual, under a "
        "shared normative basis, in two cross-fitted folds.",
        ["Profile", "Cosine to ADHD residual (fold A / B)",
         "Cosine to NT residual (fold A / B)", "Gap (fold A / B)",
         "95th percentile of NT null gap", "Verdict"], body)
    print(f"\nOutputs -> {config.CLUSTER_RESULTS}")

if __name__ == "__main__":
    run()
