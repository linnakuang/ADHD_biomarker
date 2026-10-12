from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / "preprocessing"))
import config  # noqa: E402
from docx_table import write_docx_table  # noqa: E402
from build_patients import SUBTYPES, NOT_CLASSIFIABLE  # noqa: E402

TAG = ""

# ── Profile themes ──────────────────────────────────────
THEME_SIGNS = {
    "Injury–musculoskeletal / soft tissue":
        ["Soft tissue & muscle disorders", "Arthropathies & joint disorders (arthritis)",
         "Superficial injuries & contusions", "Dislocations & sprains", "Fractures (non-skull)"],
    "Psychiatric–congenital–neurologic":
        ["Anxiety, stress & somatoform disorders", "Mood disorders (depression, bipolar)",
         "Movement & other episodic disorders", "Personality disorders", "Epilepsy & seizures"],
    "ENT–infectious–sensory / dermatologic":
        ["Otitis media & middle ear disorders", "Hearing loss & other ear disorders",
         "Acute upper respiratory infection", "Dermatitis & eczema (atopic)",
         "Conjunctivitis & conjunctival disorders"],
}
THEME_ORDER = list(THEME_SIGNS)
SHORT = dict(zip(THEME_ORDER, ["Injury/MSK", "Psychiatric", "ENT/atopic"]))

def label_profiles(Hdf):
    pairs = []
    for th, sig in THEME_SIGNS.items():
        s = Hdf.reindex(columns=sig).fillna(0).sum(axis=1)
        pairs += [(float(val), th, pname) for pname, val in s.items()]
    pairs.sort(reverse=True)
    out, used_th, used_p = {}, set(), set()
    for _, th, pname in pairs:
        if th in used_th or pname in used_p:
            continue
        out[th] = pname; used_th.add(th); used_p.add(pname)
    return out

# ── Data + strata standardization (shared with Stage 1) ─────────────────────
def load():
    p = pd.read_csv(config.PATIENTS_ANALYTIC_CSV)
    p["ID"] = p["ID"].astype(str).str.strip()
    p = p[p["label"].isin(["ADHD", "NT"])].copy()
    p["is_black"] = pd.to_numeric(p.get("is_black"), errors="coerce").fillna(0).astype(int)
    p["stratum"] = p["sex"].astype(str) + "|" + p["is_black"].astype(str)
    ev = pd.read_csv(config.EVENT_TABLE_CSV)
    ev["ID"] = ev["ID"].astype(str).str.strip()
    ev = ev[ev["ID"].isin(set(p["ID"])) & (ev["age_years"] < config.AGE_CUTOFF)].copy()
    ev = ev.dropna(subset=["condition"])
    return p, ev

def build_partition(p):
    w = p.groupby("stratum").size() / len(p)
    da = p[p["label"] == "ADHD"].groupby("stratum").size()
    dn = p[p["label"] == "NT"].groupby("stratum").size()
    valid = [s for s in w.index if da.get(s, 0) >= config.MIN_STRATUM_N
             and dn.get(s, 0) >= config.MIN_STRATUM_N]
    code = {s: (s if s in valid else "POOLED") for s in w.index}
    cw = {s: float(w[s]) for s in valid}
    pooled = [s for s in w.index if s not in valid]
    if pooled:
        cw["POOLED"] = float(sum(w[s] for s in pooled))
    return code, cw

def _std(M, codes, cw):
    out = np.zeros(M.shape[1]); wsum = 0.0
    for c, wt in cw.items():
        m = codes == c
        if m.any():
            out += wt * M[m].mean(axis=0); wsum += wt
    return out / wsum if wsum > 0 else np.full(M.shape[1], np.nan)

def condition_matrix(p, ev, group, conditions=None):
    ids = p.loc[p["label"] == group, "ID"]
    pres = ev.drop_duplicates(["ID", "condition"]).assign(v=1)
    M = pres.pivot_table(index="ID", columns="condition", values="v", fill_value=0)
    if conditions is not None:
        M = M.reindex(columns=conditions, fill_value=0)
    return M.reindex(ids).fillna(0).astype(float)

# ── Condition selection ─────────────────────────────────────────────────────
def nt_rates(p, ev, conds, code, cw):
    nt = condition_matrix(p, ev, "NT", conds)
    nc = nt.index.map(p.set_index("ID")["stratum"]).map(code).values
    return pd.Series(_std(nt.values, nc, cw), index=conds)

def select_conditions(p, ev, code, cw, verbose=True):
    conds = sorted(ev["condition"].unique())
    adhd = condition_matrix(p, ev, "ADHD", conds)
    ac = adhd.index.map(p.set_index("ID")["stratum"]).map(code).values
    adhd_std = _std(adhd.values, ac, cw)
    nt_rate = nt_rates(p, ev, conds, code, cw)
    n_adhd = len(adhd)
    keep = []
    if verbose:
        print(f"  Condition filter (ADHD n >= {config.NMF_MIN_ADHD_COUNT}):")
    for j, c in enumerate(conds):
        cnt = int(adhd[c].sum())
        if cnt < config.NMF_MIN_ADHD_COUNT:
            continue
        keep.append(c)
        if verbose:
            print(f"    {c:<50s} adhd_n={cnt:>4d}  prev={cnt / n_adhd:>5.1%}  "
                  f"gap={(adhd_std[j] - nt_rate.iloc[j]) * 100:+6.1f}pp")
    if not verbose:
        print(f"  Condition filter: {len(keep)}/{len(conds)} retained "
              f"(ADHD n >= {config.NMF_MIN_ADHD_COUNT})")
    return keep, nt_rate

def residual_matrix(p, ev, conds, nt_rate):
    adhd = condition_matrix(p, ev, "ADHD", conds)
    X = np.clip(adhd.values - nt_rate[conds].values[None, :], 0, None)
    has_burden = X.sum(axis=1) > 0
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1
    return X / norms, adhd.index.tolist(), has_burden

def dominant(W, has_burden):
    return np.where(has_burden, W.argmax(axis=1), -1)

# ── NMF fit + stability ─────────────────────────────────────────────────────
def _fit_restarts(X, k, seed, n_restarts=None):
    n_restarts = config.NMF_N_RESTARTS if n_restarts is None else n_restarts
    best_err, best, all_H = np.inf, None, []
    for r in range(n_restarts):
        m = NMF(n_components=k, init="random", solver="mu", beta_loss="frobenius",
                max_iter=1000, random_state=seed + r)
        W = m.fit_transform(X)
        all_H.append(m.components_.copy())
        if m.reconstruction_err_ < best_err:
            best_err, best = m.reconstruction_err_, (W, m.components_, len(all_H) - 1)
    W, H, best_idx = best
    return best_err, W, H, all_H, best_idx

def _fit(X, k, seed):
    err, W, H, _, _ = _fit_restarts(X, k, seed)
    return err, (W, H)

def _match(H1, H2):
    C = cosine_similarity(H1, H2)
    r, c = linear_sum_assignment(-C)
    return C[r, c].mean()

def stability(X, k, rng):
    sims = []
    n = X.shape[0]
    for _ in range(config.NMF_STABILITY_REPEATS):
        idx = rng.permutation(n)
        h = n // 2
        s = int(rng.integers(0, 1_000_000))
        _, (_, H1) = _fit(X[idx[:h]], k, s)
        _, (_, H2) = _fit(X[idx[h:]], k, s + 1)
        sims.append(_match(H1, H2))
    return float(np.mean(sims))

def loading_stats(W, has_burden):
    Wb = W[has_burden]
    if not len(Wb):
        return float("nan"), float("nan")
    tot = Wb.sum(axis=1, keepdims=True)
    share = np.where(tot > 0, Wb / np.maximum(tot, 1e-12), 0.0).max(axis=1)
    return float(np.median(share)), float(100 * (share >= 0.5).mean())

def rank_correspondence(models, conds):
    ks = sorted(models)
    print("\n  Rank-to-rank correspondence (each component vs its best match one rank down):")
    body = []
    for a, b in zip(ks[:-1], ks[1:]):
        Ha, Hb = models[a], models[b]
        cm = cosine_similarity(Hb, Ha)
        print(f"    K={a} -> K={b}")
        for c in range(Hb.shape[0]):
            j = int(cm[c].argmax())
            top = ", ".join(pd.Series(Hb[c], index=conds)
                            .sort_values(ascending=False).head(4).index)
            tag = "Split or copy" if cm[c, j] >= 0.80 else (
                  "Shifted" if cm[c, j] >= 0.60 else "New pattern")
            print(f"      P{c + 1} -> K={a} P{j + 1}  cos={cm[c, j]:.3f}  {tag:<13} {top}")
            body.append([b, f"P{c + 1}", top, f"K={a} P{j + 1}",
                         f"{cm[c, j]:.3f}", tag])
    write_docx_table(
        config.CLUSTER_RESULTS / f"sTable9_rank_correspondence{TAG}.docx",
        "Supplementary Table S9. Correspondence between each NMF component and its "
        "closest counterpart one rank lower, showing which components persist as the "
        "rank increases and which are subdivisions of an existing component.",
        ["K", "Component", "Top conditions", "Closest component at K-1",
         "Cosine similarity", "Interpretation"], body)

def kscan(X, has_burden, conds):
    rng = np.random.default_rng(config.SEED)
    total = float((X ** 2).sum())
    rows, models = [], {}
    for k in config.NMF_K_SCAN:
        err, (W, H_k) = _fit(X, k, config.SEED)
        models[k] = H_k
        stab = stability(X, k, rng)
        expl = 1 - err ** 2 / total if total else np.nan
        dom = dominant(W, has_burden)
        sizes = "/".join(str(int((dom == c).sum())) for c in range(k))
        top_share, pct_major = loading_stats(W, has_burden)
        rows.append({"K": k, "reconstruction_err": err, "explained_var": expl,
                     "component_sizes": sizes, "split_half_stability": stab,
                     "median_top_share": top_share, "even_split": 1 / k,
                     "pct_majority_component": pct_major})
        print(f"  K={k}: recon={err:.3f}  expl_var={expl:.3f}  "
              f"stability={stab:.3f}  sizes={sizes}")
        print(f"        median top share={top_share:.3f} (even split {1 / k:.3f})  "
              f"{pct_major:.0f}% of children have a majority component")
        for c in range(k):
            top = ", ".join(pd.Series(H_k[c], index=conds)
                            .sort_values(ascending=False).head(5).index)
            print(f"          P{c + 1} (n={int((dom == c).sum())}): {top}")
    tab = pd.DataFrame(rows)
    write_docx_table(
        config.CLUSTER_RESULTS / f"sTable8_model_selection{TAG}.docx",
        "Supplementary Table S8. NMF model selection (K-scan).",
        ["K", "Component sizes (n)", "Reconstruction error", "Explained variance",
         "Split-half stability", "Median top share (even split)",
         "% with majority component"],
        [[int(r["K"]), r["component_sizes"], round(r["reconstruction_err"], 3),
          round(r["explained_var"], 3), round(r["split_half_stability"], 3),
          f"{r['median_top_share']:.3f} ({r['even_split']:.3f})",
          f"{r['pct_majority_component']:.1f}"] for r in rows])
    rank_correspondence(models, conds)
    return tab

# ── Outputs ─────────────────────────────────────────────────────────────────
def save_model(p, ids, conds, has_burden, W, H):
    k = H.shape[0]
    comp = [f"P{i+1}" for i in range(k)]
    Hdf = pd.DataFrame(H, index=comp, columns=conds)
    Wn = W / W.sum(axis=1, keepdims=True).clip(min=1e-9)
    Wdf = pd.DataFrame(Wn, index=ids, columns=comp)
    Wdf["dominant"] = [comp[c] if c >= 0 else "" for c in dominant(W, has_burden)]
    Wdf.to_csv(config.CLUSTER_RESULTS / f"nmf_loadings{TAG}.csv")

    # n = ADHD children whose dominant profile is this component ("" = no excess burden)
    n_dom = (Wdf.loc[Wdf["dominant"] != "", "dominant"]
             .value_counts().reindex(comp).fillna(0).astype(int))
    disp = [f"Profile {i + 1}\n(n={n_dom[c]})" for i, c in enumerate(comp)]

    # F3 itself is drawn in draw_f3() once the window refits exist -- its panel B is
    # the window-similarity trajectory. Here we only write the supplementary whole.
    order_c = sorted(conds, key=lambda c: (comp.index(Hdf[c].idxmax()), -Hdf[c].max()))
    vmax = float(Hdf.values.max())

    def nmf_panel(ax, cs, ttl, fs, ylab=True, yright=False):
        # Sequential light->dark blue: NMF weights are non-negative (no diverging scale).
        im = ax.imshow(Hdf.loc[:, cs].T.values, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(k)); ax.set_xticklabels(disp, fontsize=9)
        if ylab:
            ax.set_yticks(range(len(cs))); ax.set_yticklabels(cs, fontsize=fs)
            if yright:
                ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
        else:
            ax.set_yticks([])
        ax.set_title(ttl, fontsize=9)
        return im

    # sF4: standalone full-size all-conditions "whole" for the supplement
    figw, axw = plt.subplots(figsize=(1.1 * k + 4, 0.20 * len(order_c) + 1.6))
    nmf_panel(axw, order_c, "All conditions", 6)
    figw.colorbar(axw.images[0], ax=axw, shrink=.5, label="Component weight")
    figw.savefig(config.CLUSTER_RESULTS / f"sF4_nmf_components_all{TAG}.pdf", bbox_inches="tight")
    plt.close(figw)

    print(f"[NMF] K={k}: components + loadings saved.")
    for i, c in enumerate(comp):
        tops = ", ".join(Hdf.iloc[i].nlargest(config.NMF_TOP_CONDITIONS).index)
        print(f"  Profile {i + 1} (n={n_dom[c]}): {tops}")

    # How concentrated each child's loading is -- the same read as the K-scan's
    # top-share column, broken out per component for the retained solution.
    shares = Wn.max(axis=1)
    med_share, pct_major = loading_stats(W, has_burden)
    print(f"  {'component':<12}{'n dominant':>12}{'% cohort':>10}"
          f"{'median load':>14}{'median top share':>18}")
    for i, c in enumerate(comp):
        m = (Wdf["dominant"] == c).values
        print(f"  Profile {i + 1:<4}{n_dom[c]:>12d}{100 * n_dom[c] / len(Wdf):>9.1f}%"
              f"{Wdf.loc[m, c].median() if m.any() else float('nan'):>14.3f}"
              f"{np.median(shares[m]) if m.any() else float('nan'):>18.3f}")
    print(f"  median top share {med_share:.3f} (even split {1 / k:.3f}); "
          f"{pct_major:.1f}% of children with excess burden have a majority component")

    # ADHD subtype composition of each dominant profile (descriptive)
    if "adhd_subtype" in p.columns:
        st_order = SUBTYPES + [NOT_CLASSIFIABLE]
        st = p.set_index("ID")["adhd_subtype"].reindex(Wdf.index)
        ct = (pd.crosstab(Wdf["dominant"], st).reindex(index=comp, fill_value=0)
              .reindex(columns=st_order, fill_value=0))
        print("[NMF] ADHD presentation composition by dominant profile (row %):")
        body = []
        for i, c in enumerate(comp):
            tot = int(ct.loc[c].sum())
            cells = [f"Profile {i + 1}", tot]
            parts = []
            for s in st_order:
                nsub = int(ct.loc[c, s])
                cell = f"{nsub} ({100*nsub/max(tot,1):.0f}%)"
                cells.append(cell); parts.append(f"{s} {cell}")
            body.append(cells)
            print(f"  Profile {i + 1} (n={tot}): {', '.join(parts)}")

# ── Developmental windows: F3 panel B, sF5 + sTable11 ──────────────────────
WINDOW_PANEL_COLS = 2

def _assign(H_ref, H_new):
    cm = cosine_similarity(H_ref, H_new)
    rows, cols = linear_sum_assignment(-cm)
    order = np.empty(len(H_ref), dtype=int)
    for r, c in zip(rows, cols):
        order[r] = c
    return order, [float(cm[r, order[r]]) for r in range(len(H_ref))]

def _restart_sims(H_ref, all_H, best_idx, k):
    sims = []
    for i, H_run in enumerate(all_H):
        if i == best_idx:
            continue
        cm = cosine_similarity(H_ref, H_run)
        r, c = linear_sum_assignment(-cm)
        sims.append(cm[r, c])
    return np.mean(sims, axis=0) if sims else np.ones(k)

def refit_window(p, ev, code, cw, conds, cutoff, label, H_p, dom_p, adhd_ids):
    k = H_p.shape[0]
    ev_w = ev[ev["age_years"] < cutoff]
    lab = p.set_index("ID")["label"]
    seen = ev_w.drop_duplicates("ID")["ID"].map(lab).value_counts()

    # NT rates are recomputed within the window: an NT child with no in-window event
    # contributes a zero, the same way the ADHD rows do.
    nt_rate_w = nt_rates(p, ev_w, conds, code, cw)
    X_w, ids_w, burden_w = residual_matrix(p, ev_w, conds, nt_rate_w)
    assert ids_w == adhd_ids, f"cohort changed in {label}"

    err_w, W_w, H_w, all_H_w, idx_w = _fit_restarts(X_w, k, config.SEED)
    stab_w = _restart_sims(H_w, all_H_w, idx_w, k)
    order, sims = _assign(H_p, H_w)

    back = {c: r for r, c in enumerate(order)}
    dom_w = np.where(burden_w, [back[c] for c in W_w.argmax(axis=1)], -1)
    scored = (dom_w >= 0) & (dom_p >= 0)
    concordance = float((dom_w[scored] == dom_p[scored]).mean()) if scored.sum() else np.nan
    sizes_w = [int((dom_w == c).sum()) for c in range(k)]

    counts_w = (ev_w[ev_w["ID"].isin(set(adhd_ids))]
                .drop_duplicates(["ID", "condition"])["condition"].value_counts())
    observable = np.array([counts_w.get(c, 0) >= config.NMF_MIN_OBSERVABLE for c in conds])
    ceiling = [float(H_p[c, observable].sum() / H_p[c].sum()) if H_p[c].sum() else 0.0
               for c in range(k)]

    rng = np.random.default_rng(config.NMF_BOOTSTRAP_SEED + int(round(cutoff * 100)))
    n_sub = max(k * 10, int(round(config.NMF_SUBSAMPLE_FRAC * len(X_w))))
    sub_sims, sub_conc, n_skipped = [], [], 0
    for _ in range(config.NMF_BOOTSTRAP_B):
        ib = rng.choice(len(X_w), size=n_sub, replace=False)
        _, W_b, H_b, _, _ = _fit_restarts(X_w[ib], k, config.SEED,
                                          n_restarts=config.NMF_BOOTSTRAP_RESTARTS)
        # A diverged fit yields non-finite loadings and an all-but-random match;
        # dropping it is honest, and the count is reported.
        if not (np.all(np.isfinite(H_b)) and np.all(np.isfinite(W_b))):
            n_skipped += 1
            continue
        ord_b, sims_b = _assign(H_p, H_b)
        sub_sims.append(sims_b)
        back_b = {c: r for r, c in enumerate(ord_b)}
        dom_b = np.where(burden_w[ib], [back_b[x] for x in W_b.argmax(axis=1)], -1)
        sc = (dom_b >= 0) & (dom_p[ib] >= 0)
        sub_conc.append(float((dom_b[sc] == dom_p[ib][sc]).mean()) if sc.sum() else np.nan)

    if n_skipped:
        print(f"    ({n_skipped}/{config.NMF_BOOTSTRAP_B} subsample fits dropped as non-finite)")
    sub_sims = np.array(sub_sims) if sub_sims else np.full((1, k), np.nan)
    ci_lo = np.percentile(sub_sims, 2.5, axis=0)
    ci_hi = np.percentile(sub_sims, 97.5, axis=0)
    mean_lo, mean_hi = np.percentile(sub_sims.mean(axis=1), [2.5, 97.5])
    conc_lo, conc_hi = (np.nanpercentile(sub_conc, [2.5, 97.5]) if sub_conc else (np.nan, np.nan))

    return {"label": label, "H": H_w[order], "sizes": sizes_w,
            "n_adhd": int(seen.get("ADHD", 0)), "n_nt": int(seen.get("NT", 0)),
            "n_observable": int(observable.sum()), "sims": sims,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "mean_cos": float(np.mean(sims)),
            "mean_ci": (mean_lo, mean_hi), "concordance": concordance,
            "conc_ci": (conc_lo, conc_hi), "ceiling": ceiling,
            "stab": stab_w, "err": float(err_w)}

def _report_window(r, k):
    print(f"\n  {r['label']}: {r['n_adhd']} ADHD, {r['n_nt']} NT with >=1 event; "
          f"observable {r['n_observable']}; sizes={r['sizes']}")
    print(f"    {'component':<12s}{'cosine':>9s}{'subsample range':>18s}"
          f"{'ceiling':>10s}{'stab(win)':>11s}")
    for c in range(k):
        rng = "({:.3f}, {:.3f})".format(r["ci_lo"][c], r["ci_hi"][c])
        print("    P{:<11d}{:>9.3f}{:>18s}{:>10.2f}{:>11.3f}".format(
            c + 1, r["sims"][c], rng, r["ceiling"][c], r["stab"][c]))
    print(f"    mean cos {r['mean_cos']:.3f} ({r['mean_ci'][0]:.3f}, {r['mean_ci'][1]:.3f})   "
          f"concordance {r['concordance']*100:.1f}% "
          f"({r['conc_ci'][0]*100:.1f}, {r['conc_ci'][1]*100:.1f})")

def top_rows(H_p, conds):
    rows = []
    for c in range(H_p.shape[0]):
        top = pd.Series(H_p[c], index=conds).sort_values(ascending=False)
        rows += [x for x in top.head(config.NMF_TOP_CONDITIONS).index if x not in rows]
    return rows

# Profile colours for F3 panel B; extended by cycling if K exceeds the list.
COMP_COLOR = ["#C0392B", "#2E7D8C", "#B07A2A", "#5B4A8A", "#4C78A8", "#5A8F5A"]

def draw_f3(results, H_p, sizes_p, conds, n_adhd, xlab):
    k = H_p.shape[0]
    rows = top_rows(H_p, conds)
    idx = [conds.index(c) for c in rows]
    mat = H_p[:, idx].T

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(max(7.5, k * 1.6 + 5), len(rows) * 0.34 + 6),
        gridspec_kw={"height_ratios": [len(rows) * 0.34 + 1.2, 3.4]})

    imA = axA.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=float(mat.max()))
    axA.set_xticks(range(k))
    axA.set_xticklabels([f"{xlab[c]}\nn={sizes_p[c]}" for c in range(k)],
                        fontsize=9, fontweight="bold")
    axA.set_yticks(range(len(rows)))
    axA.set_yticklabels(rows, fontsize=9)
    for r_ in range(mat.shape[0]):
        for j in range(k):
            if mat[r_, j] > 0.001:
                axA.text(j, r_, f"{mat[r_, j]:.3f}", ha="center", va="center", fontsize=7,
                         color="black" if mat[r_, j] < mat.max() * 0.7 else "white")
    for c in range(1, k):
        axA.axvline(x=c - 0.5, color="white", linewidth=3)
    axA.set_title(f"(A) Multimorbidity profiles, birth to age {config.AGE_CUTOFF} "
                  f"(n={n_adhd} children with ADHD)", fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(imA, ax=axA, label="Component weight", fraction=0.04, pad=0.02)

    panelB = results + [{"label": f"Birth to {config.AGE_CUTOFF}y",
                         "sims": [1.0] * k, "ci_lo": [np.nan] * k, "ci_hi": [np.nan] * k}]
    xs = np.arange(len(panelB))
    for c in range(k):
        col = COMP_COLOR[c % len(COMP_COLOR)]
        y = [w["sims"][c] for w in panelB]
        axB.plot(xs, y, "-o", color=col, lw=2, ms=6, label=xlab[c].replace("\n", " "))
        axB.fill_between(xs, [w["ci_lo"][c] for w in panelB],
                         [w["ci_hi"][c] for w in panelB], color=col, alpha=.15)
        for x, v in zip(xs, y):
            axB.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color=col)
    axB.set_xticks(xs)
    axB.set_xticklabels([w["label"].replace(" (<", "\n(<") for w in panelB], fontsize=9)
    axB.set_ylim(0, 1.08)
    axB.set_ylabel(f"Cosine similarity to the\nfull birth-to-{config.AGE_CUTOFF} profile")
    axB.grid(axis="y", alpha=.25)
    axB.legend(frameon=False, loc="lower right", fontsize=9)
    axB.set_title("(B) Profile similarity across developmental windows",
                  fontsize=11, fontweight="bold", loc="left")
    axB.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = config.CLUSTER_RESULTS / f"F3_nmf{TAG}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\n  Saved: {out}")
    print("  Panel B trajectories:")
    for c in range(k):
        print(f"    {xlab[c].replace(chr(10), ' '):<28s} "
              + "  ".join(f"{w['sims'][c]:.3f}" for w in panelB))

def draw_windows(results, H_p, sizes_p, conds, n_adhd, xlab):
    """sF5: one loading heatmap per window, windows ascending, primary last."""
    k = H_p.shape[0]
    rows = top_rows(H_p, conds)
    idx = [conds.index(c) for c in rows]

    panels = [(r["H"][:, idx].T, r["sizes"],
               f"{r['label']}\nADHD n={r['n_adhd']}, mean cos={r['mean_cos']:.2f}, "
               f"{r['concordance']*100:.0f}% same profile")
              for r in results]
    panels.append((H_p[:, idx].T, sizes_p,
                   f"All events to age {config.AGE_CUTOFF} (primary)\nADHD n={n_adhd}"))

    vmax = max(float(m.max()) for m, _, _ in panels)
    ncols = min(WINDOW_PANEL_COLS, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, sharey=True,
                             figsize=(k * 1.6 * ncols + 5, nrows * (len(rows) * 0.42 + 1.8) + 1))
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, (mat, sizes, title)) in enumerate(zip(axes, panels)):
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"{xlab[c]}\nn={sizes[c]}" for c in range(k)],
                           fontsize=9, fontweight="bold")
        ax.set_title(f"{'ABCDEFGH'[i]}. {title}", fontsize=10, fontweight="bold")
        for r_ in range(mat.shape[0]):
            for j in range(k):
                if mat[r_, j] > 0.001:
                    ax.text(j, r_, f"{mat[r_, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="black" if mat[r_, j] < vmax * 0.7 else "white")
        for c in range(1, k):
            ax.axvline(x=c - 0.5, color="white", linewidth=3)
        if i % ncols == 0:                      # sharey hides the inner tick labels
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows, fontsize=9)

    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), label="Component weight (residualized)",
                 fraction=0.02, pad=0.02)
    out = config.CLUSTER_RESULTS / f"sF5_nmf_window_loadings{TAG}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\n  Saved: {out}")

def write_window_table(results, n_conds, n_adhd, xlab, k):
    cols = (["Window", "ADHD n", "Conditions observable"]
            + [f"{xlab[c].splitlines()[0]} cosine (subsample range)" for c in range(k)]
            + ["Mean cosine (subsample range)", "Dominant concordance (subsample range)"])
    body = [[r["label"], r["n_adhd"], f"{r['n_observable']}/{n_conds}"]
            + [f"{r['sims'][c]:.3f} ({r['ci_lo'][c]:.3f}, {r['ci_hi'][c]:.3f})" for c in range(k)]
            + [f"{r['mean_cos']:.3f} ({r['mean_ci'][0]:.3f}, {r['mean_ci'][1]:.3f})",
               f"{r['concordance']*100:.1f}% ({r['conc_ci'][0]*100:.1f}, "
               f"{r['conc_ci'][1]*100:.1f})"]
            for r in results]
    body.append([f"All events to age {config.AGE_CUTOFF} (primary)", n_adhd,
                 f"{n_conds}/{n_conds}"] + ["—"] * (k + 2))
    out = config.CLUSTER_RESULTS / f"sTable10_nmf_window_agreement{TAG}.docx"
    write_docx_table(out, "Supplementary Table S10. Agreement between multimorbidity profiles "
                          "refit within cumulative developmental windows and the primary "
                          "solution.", cols, body)
    print(f"  Saved: {out}")


def window_refits(p, ev, code, cw, conds, adhd_ids, has_burden, W_p, H_p):
    k = H_p.shape[0]
    dom_p = dominant(W_p, has_burden)
    sizes_p = [int((dom_p == c).sum()) for c in range(k)]
    comp = [f"P{i+1}" for i in range(k)]

    # Profiles carry their F5 theme names so the windows read against the primary figure.
    theme_of = {pn: SHORT[th] for th, pn in
                label_profiles(pd.DataFrame(H_p, index=comp, columns=conds)).items()}
    xlab = [f"{c}\n{theme_of.get(c, '')}" for c in comp]

    med_dx = float(pd.to_numeric(p.loc[p["label"] == "ADHD", "adhd_dx_age"],
                                 errors="coerce").dropna().median())
    mid = (med_dx + config.AGE_CUTOFF) / 2
    wins = [(med_dx, f"Pre-diagnosis (<{med_dx:.1f}y)"),
            (mid, f"Birth to {mid:.1f}y")]
    print(f"\nProfiles refit within cumulative windows "
          f"({[f'{c:.2f}' for c, _ in wins]}); median ADHD dx {med_dx:.2f}y")

    results = []
    for cutoff, label in wins:
        r = refit_window(p, ev, code, cw, conds, cutoff, label, H_p, dom_p, adhd_ids)
        _report_window(r, k)
        results.append(r)

    draw_f3(results, H_p, sizes_p, conds, len(adhd_ids), xlab)
    draw_windows(results, H_p, sizes_p, conds, len(adhd_ids), xlab)
    write_window_table(results, len(conds), len(adhd_ids), xlab, k)

# ── Driver ──────────────────────────────────────────────────────────────────
def prepare():
    p, ev = load()
    code, cw = build_partition(p)
    conds, nt_rate = select_conditions(p, ev, code, cw)
    return p, ev, code, cw, conds, nt_rate

def run():
    config.ensure(config.CLUSTER_RESULTS)
    p, ev, code, cw, conds, nt_rate = prepare()
    (config.CLUSTER_RESULTS / f"nmf_kept_conditions{TAG}.txt").write_text("\n".join(conds))
    print(f"ADHD n={int((p['label']=='ADHD').sum())}; conditions kept: {len(conds)}")

    X, ids, has_burden = residual_matrix(p, ev, conds, nt_rate)
    print(f"NMF input: {X.shape[0]} children x {X.shape[1]} conditions "
          f"({int(has_burden.sum())} carry >=1 retained condition, "
          f"{int((~has_burden).sum())} left unassigned)")
    print("Model selection (K-scan):")
    kscan(X, has_burden, conds)
    _, W, H, _, _ = _fit_restarts(X, config.NMF_K, config.SEED)
    save_model(p, ids, conds, has_burden, W, H)
    window_refits(p, ev, code, cw, conds, ids, has_burden, W, H)
    print(f"\nOutputs -> {config.CLUSTER_RESULTS}")

if __name__ == "__main__":
    run()
