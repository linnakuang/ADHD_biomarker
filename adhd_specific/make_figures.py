"""Report figures for RESULTS.md, built only from the aggregate CSVs in results/.

    python make_figures.py        # writes results/fig_*.png

Figures carry no analysis IDs (D1 ... D7) in titles; captions live in RESULTS.md. Every input
is aggregate, so this never touches per-child data (no _cache/ or adhd_common import needed;
runs in seconds rather than minutes on the slow filesystem).
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
DPI = 150

C = {"NT": "#8c8c8c", "Other NDD": "#4c9f70", "ADHD": "#c1272d", "ASD": "#3b6ea5",
     "ASD+ADHD": "#7b3294", "ASD-only": "#6fa8dc"}
ACC, GREY = "#c1272d", "#555555"
N = {"NT": 1851, "Other NDD": 930, "ADHD": 573, "ASD": 145}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.frameon": False})


def short(c, n=34):
    s = re.sub(r"\s*\(.*?\)", "", c).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def wilson(p, n, z=1.96):
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return ctr - h, ctr + h


def save(fig, name):
    fig.savefig(os.path.join(R, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


d1 = pd.read_csv(os.path.join(R, "d1_adhd78_adhd573.csv"))
d1a = pd.read_csv(os.path.join(R, "d1_adhd78_anyadhd634.csv")).set_index("condition")
d1p = pd.read_csv(os.path.join(R, "d1_paper51_adhd573.csv")).set_index("condition")
MU = float((d1["d_other"] - d1["d_other_minus_mu"]).iloc[0])
d7 = pd.read_csv(os.path.join(R, "d7_util_adjusted.csv"))
d7o = d7[d7["contrast"] == "ADHD_vs_other"].set_index("condition")
d5 = pd.read_csv(os.path.join(R, "d5_prepost.csv")).set_index("condition")
CORE = list(d1.loc[d1["surv_other_mu90"] & (d1["d_other_minus_mu"] > 0)]
            .sort_values("d_other_minus_mu", ascending=False)["condition"])
UTIL_SURV = list(d7o.index[(d7o["q1"] < .05)])
ROWS = CORE + [c for c in UTIL_SURV if c not in CORE]


# ── Fig 1: utilization and burden ladder ──
def fig_ladder():
    t = pd.read_csv(os.path.join(R, "d6_table.csv"))
    lad = pd.read_csv(os.path.join(R, "d7_utilization_ladder.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(t))
    a.bar(x, t["burden_mean"], color=[C[g] for g in t["group"]])
    a.errorbar(x, t["burden_mean"], yerr=[t["burden_mean"] - t["burden_lo"], t["burden_hi"] - t["burden_mean"]],
               fmt="none", ecolor="k", capsize=3, lw=0.8)
    a.set_xticks(x); a.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(t["group"], t["n"])], fontsize=8)
    a.set_ylabel("conditions per child (of 78), mean ± 95% CI")
    a.set_title("a  Comorbidity burden")
    iq = lad["event_months_iqr"].str.split("-", expand=True).astype(float)
    x2 = np.arange(len(lad))
    b.bar(x2, lad["event_months_med"], color=[C[g] for g in lad["group"]])
    b.errorbar(x2, lad["event_months_med"], yerr=[lad["event_months_med"] - iq[0], iq[1] - lad["event_months_med"]],
               fmt="none", ecolor="k", capsize=3, lw=0.8)
    b.set_xticks(x2); b.set_xticklabels(lad["group"], fontsize=8)
    b.set_ylabel("distinct months with ≥1 coded condition\n(median, IQR)")
    b.set_title("b  Healthcare-contact proxy")
    fig.tight_layout(); save(fig, "fig_ladder.png")


# ── Fig 2: shrunk ADHD − other-NDD forest ──
def fig_forest():
    t = d1.reindex(d1["d_other_minus_mu"].abs().sort_values(ascending=False).index).head(30)
    t = t.sort_values("d_other")
    fig, ax = plt.subplots(figsize=(7.2, 8))
    y = np.arange(len(t))
    ax.axvspan(0, MU, color=ACC, alpha=0.07)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(MU, color=ACC, lw=1, ls="--")
    ax.text(MU, len(t) - 0.2, f" μ = {MU:+.3f}\n (global offset)", color=ACC, fontsize=7.5, va="top")
    col = np.where(t["surv_other_mu90"], ACC, GREY)
    ax.scatter(t["d_other_raw"], y, s=14, facecolors="none", edgecolors="#9aa7b5", label="raw δ", zorder=2)
    for yi, lo, hi, c in zip(y, t["d_other_lo90"], t["d_other_hi90"], col):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.2)
    ax.scatter(t["d_other"], y, s=24, c=col, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([short(c, 40) for c in t["condition"]], fontsize=7.5)
    ax.set_xlabel("prevalence difference, ADHD − other NDD (shrunk, 90% interval)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#9aa7b5", label="raw δ"),
                       plt.Line2D([], [], marker="o", color=ACC, label="shrunk δ, deviates from μ (90%)"),
                       plt.Line2D([], [], marker="o", color=GREY, label="shrunk δ, consistent with μ")],
              loc="lower right", fontsize=7.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); save(fig, "fig_forest.png")


# ── Fig 3: decomposition of the ADHD-vs-NT elevation ──
def fig_decomp():
    t = d1[d1["e_ADHD_vs_NT"] > 0].sort_values("e_ADHD_vs_NT", ascending=False).head(22).iloc[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(t))
    spec = t["p_ADHD"] - t["p_otherNDD"]
    a.barh(y, t["shared_ndd"], color=C["Other NDD"], label="shared NDD  (other NDD − NT)")
    a.barh(y, spec, left=t["shared_ndd"], color=np.where(t["surv_other_mu90"], ACC, "#e8a3a6"),
           label="ADHD beyond other NDD  (ADHD − other NDD)")
    a.axvline(0, color="k", lw=0.8)
    a.set_yticks(y); a.set_yticklabels([short(c, 38) for c in t["condition"]], fontsize=7.5)
    a.set_xlabel("ADHD − NT (stratum-standardized), prevalence difference")
    a.legend(handles=[Patch(color=C["Other NDD"], label="shared NDD (other NDD − NT)"),
                      Patch(color=ACC, label="ADHD − other NDD, deviates from μ"),
                      Patch(color="#e8a3a6", label="ADHD − other NDD, within μ")],
             loc="lower right", fontsize=7.5)
    a.set_title("a  22 largest ADHD-vs-NT elevations")
    a.grid(axis="x", alpha=0.25)
    s = d1
    sig = s["q_e_vs_NT"] < .05
    b.scatter(s.loc[~sig, "e_ADHD_vs_NT"], s.loc[~sig, "shared_ndd"], s=14, c="#bbbbbb", label="ADHD vs NT n.s.")
    b.scatter(s.loc[sig, "e_ADHD_vs_NT"], s.loc[sig, "shared_ndd"], s=16, c=C["Other NDD"],
              label="ADHD vs NT, FDR < .05")
    cm = s["condition"].isin(CORE)
    b.scatter(s.loc[cm, "e_ADHD_vs_NT"], s.loc[cm, "shared_ndd"], s=34, facecolors="none", edgecolors=ACC,
              lw=1.3, label="deviates from μ (ADHD-specific)")
    lim = max(s["e_ADHD_vs_NT"].max(), s["shared_ndd"].max()) * 1.08
    lo = min(s["e_ADHD_vs_NT"].min(), s["shared_ndd"].min(), 0) - 0.01
    b.plot([lo, lim], [lo, lim], "--", color="#999", lw=0.8)
    b.plot([lo, lim], [lo / 2, lim / 2], ":", color="#999", lw=0.8)
    b.text(lim, lim, " all shared", fontsize=7, color="#777", va="bottom", ha="right")
    b.text(lim, lim / 2, " half shared", fontsize=7, color="#777", va="bottom", ha="right")
    for c in CORE[:5]:
        r = s.set_index("condition").loc[c]
        b.annotate(short(c, 18), (r["e_ADHD_vs_NT"], r["shared_ndd"]), fontsize=6.5, xytext=(3, -8),
                   textcoords="offset points", color=ACC)
    b.set_xlabel("ADHD − NT"); b.set_ylabel("other NDD − NT")
    b.set_title("b  All 78 conditions: elevation that is shared")
    b.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "fig_decomposition.png")


# ── Fig 4: prevalence ladder for the robust conditions ──
def fig_ladder_core():
    rows = ROWS
    t = d1.set_index("condition").loc[rows]
    nc = 5; nr = int(np.ceil(len(rows) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.2 * nr), squeeze=False)
    gs = ["NT", "Other NDD", "ADHD", "ASD"]
    cols = ["p_NT", "p_otherNDD", "p_ADHD", "p_ASD"]
    for ax, (c, r) in zip(axs.flat, t.iterrows()):
        p = np.array([r[k] for k in cols]) * 100
        lo, hi = wilson(p / 100, np.array([N[g] for g in gs]))
        ax.bar(range(4), p, color=[C[g] for g in gs])
        ax.errorbar(range(4), p, yerr=[p - lo * 100, hi * 100 - p], fmt="none", ecolor="k", lw=0.7, capsize=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(["NT", "Oth", "ADHD", "ASD"], fontsize=7)
        ax.set_title(short(c, 30), fontsize=7.8)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("% of children", fontsize=8)
    for ax in list(axs.flat)[len(rows):]:
        ax.axis("off")
    fig.tight_layout(); save(fig, "fig_core_prevalence.png")


# ── Fig 5: latent profiles ──
def fig_profiles():
    ks = pd.read_csv(os.path.join(R, "d3_adhd_kscan.csv"))
    auc = pd.read_csv(os.path.join(R, "d3_separation_auc.csv"))
    b2 = pd.read_csv(os.path.join(R, "d3_b2_survival.csv"))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    a.plot(ks["K"], ks["boot_cos"], "o-", color="#333", label="mean matched cosine")
    a.plot(ks["K"], ks["boot_worst"], "s-", color=ACC, label="weakest component")
    a.plot(ks["K"], ks["concordance"], "^-", color=C["ASD"], label="assignment concordance")
    a.axhline(0.80, color=ACC, ls="--", lw=0.8); a.axhline(0.60, color=C["ASD"], ls="--", lw=0.8)
    a.set_xticks(ks["K"]); a.set_xlabel("number of profiles K"); a.set_ylabel("bootstrap stability")
    a.set_ylim(0.5, 1); a.legend(fontsize=7, loc="lower left")
    a.set_title("a  Profile stability, ADHD only")
    feats = ["burden", "latentW", "raw78"]
    lab = {"burden": "burden only", "latentW": "3 latent axes", "raw78": "78 conditions"}
    w = 0.26; x = np.arange(len(auc))
    for i, f in enumerate(feats):
        b.bar(x + (i - 1) * w, auc[f], w, label=lab[f], color=["#bbbbbb", ACC, "#333333"][i])
    b.axhline(0.5, color="k", lw=0.6)
    b.set_xticks(x); b.set_xticklabels([s.replace("_vs_", " vs\n").replace("other", "other NDD")
                                        for s in auc["contrast"]], fontsize=8)
    b.set_ylim(0.45, 0.7); b.set_ylabel("5-fold CV ROC-AUC")
    b.legend(fontsize=7, loc="upper right", ncol=3)
    b.set_title("b  Separation in a shared NDD latent space")
    names = [f"{p}: " + " / ".join(re.split(r"[,(]| & ", x)[0].strip() for x in t.split("), ")[:2])
             for p, t in zip(b2["profile"], b2["top"])]
    cc = [ACC if v >= .8 else "#e8a3a6" if v >= .5 else "#bbbbbb" for v in b2["best_resid_cos"]]
    c.barh(range(len(b2)), b2["best_resid_cos"], color=cc)
    c.axvline(0.8, color="k", ls="--", lw=0.8); c.axvline(0.5, color="k", ls=":", lw=0.8)
    c.set_yticks(range(len(b2))); c.set_yticklabels(names, fontsize=7.5); c.invert_yaxis()
    c.set_xlim(0, 1); c.set_xlabel("cosine to best NT-projected residual profile")
    c.set_title("c  Survival after removing NT structure")
    fig.tight_layout(); save(fig, "fig_profiles.png")


# ── Fig 6: presentation ──
def fig_presentation():
    t = pd.read_csv(os.path.join(R, "d4_presentation_contrast.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4))
    m = t["condition"].isin(CORE)
    a.scatter(t["p_hypcomb"] * 100, t["p_inatt"] * 100, s=14, c="#999")
    a.scatter(t.loc[m, "p_hypcomb"] * 100, t.loc[m, "p_inatt"] * 100, s=26, c=ACC, label="ADHD-specific conditions")
    lim = max(t["p_hypcomb"].max(), t["p_inatt"].max()) * 105
    a.plot([0, lim], [0, lim], "--", color="#999", lw=0.8)
    for _, r in t[m].iterrows():
        a.annotate(short(r["condition"], 16), (r["p_hypcomb"] * 100, r["p_inatt"] * 100), fontsize=6.3,
                   xytext=(3, 2), textcoords="offset points", color=ACC)
    a.set_xlabel("% hyperactive/combined (n=432)"); a.set_ylabel("% inattentive (n=60)")
    a.legend(fontsize=7.5, loc="upper left"); a.set_title("a  Per-condition prevalence by presentation")
    s = t[m].set_index("condition")
    y = np.arange(len(s))
    b.barh(y - 0.2, s["d_hyp_vs_other"], 0.4, color=ACC, label="hyp/combined − other NDD")
    b.barh(y + 0.2, s["d_inatt_vs_other"], 0.4, color="#e8a3a6", label="inattentive − other NDD")
    b.axvline(0, color="k", lw=0.8)
    b.set_yticks(y); b.set_yticklabels([short(c, 28) for c in s.index], fontsize=7.5); b.invert_yaxis()
    b.set_xlabel("shrunk prevalence difference vs other NDD")
    b.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    b.set_title("b  Where the ADHD-specific excess lives")
    fig.tight_layout(); save(fig, "fig_presentation.png")


# ── Fig 7: around the diagnosis ──
def fig_prepost():
    t = d5.reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    both = t["DiD_NT_sig"] & t["DiD_other_sig"] & (np.sign(t["DiD_NT"]) == np.sign(t["DiD_other"]))
    for ax, k, lab, tag in [(axs[0], "shift_NT", "NT (pseudo-anchored)", "a"),
                            (axs[1], "shift_other", "other NDD (pseudo-anchored)", "b")]:
        ax.scatter(t.loc[~both, k], t.loc[~both, "shift_ADHD"], s=14, c="#999")
        ax.scatter(t.loc[both, k], t.loc[both, "shift_ADHD"], s=26, c=ACC)
        lim = float(np.nanmax(np.abs(np.r_[t[k], t["shift_ADHD"]]))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "--", color="#999", lw=0.8)
        ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
        for _, r in t[both].iterrows():
            ax.annotate(short(r["condition"], 16), (r[k], r["shift_ADHD"]), fontsize=6.5, color=ACC,
                        xytext=(3, 2), textcoords="offset points")
        rho = pd.Series(t[k]).rank().corr(pd.Series(t["shift_ADHD"]).rank())
        ax.set_xlabel(f"post − pre prevalence, {lab}"); ax.set_ylabel("post − pre prevalence, ADHD")
        ax.set_title(f"{tag}  vs {lab.split(' (')[0]}: Spearman ρ = {rho:.2f}")
    s = t[both].sort_values("DiD_other")
    y = np.arange(len(s))
    c = axs[2]
    c.errorbar(s["DiD_NT"], y + 0.15, xerr=[s["DiD_NT"] - s["DiD_NT_lo90"], s["DiD_NT_hi90"] - s["DiD_NT"]],
               fmt="o", color=C["NT"], ms=4, capsize=2, label="vs NT")
    c.errorbar(s["DiD_other"], y - 0.15, xerr=[s["DiD_other"] - s["DiD_other_lo90"], s["DiD_other_hi90"] - s["DiD_other"]],
               fmt="s", color=C["Other NDD"], ms=4, capsize=2, label="vs other NDD")
    c.axvline(0, color="k", lw=0.8)
    c.set_yticks(y); c.set_yticklabels([short(x, 26) for x in s["condition"]], fontsize=7.5)
    c.set_xlabel("difference-in-differences (90% bootstrap CI)")
    c.legend(fontsize=7.5, loc="lower right"); c.set_title("c  Change beyond both comparison groups")
    fig.tight_layout(); save(fig, "fig_prepost.png")


# ── Fig 8: additivity with autism ──
def fig_additive():
    t = pd.read_csv(os.path.join(R, "d6_dual_contrasts.csv"))
    x = t["p_ADHD"] - t["p_Other NDD"]
    y = t["p_ASD+ADHD"] - t["p_ASD-only"]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    m = t["condition"].isin(CORE)
    ax.scatter(x[~m], y[~m], s=14, c="#999")
    ax.scatter(x[m], y[m], s=26, c=ACC, label="ADHD-specific conditions")
    lim = float(np.nanmax(np.abs(np.r_[x, y]))) * 1.08
    ax.plot([-lim, lim], [-lim, lim], "--", color="#999", lw=0.8, label="equal effect (additive)")
    ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
    for c, xi, yi in zip(t.loc[m, "condition"], x[m], y[m]):
        ax.annotate(short(c, 16), (xi, yi), fontsize=6.3, color=ACC, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("ADHD effect without autism\n(ADHD − other NDD, n=573 vs 930)")
    ax.set_ylabel("ADHD effect with autism\n(ASD+ADHD − ASD-only, n=61 vs 84)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); save(fig, "fig_additivity.png")


# ── Fig 9: utilization adjustment ──
def fig_util():
    t = d7o[(d7o["q0"] < .05) | (d7o["q1"] < .05)].sort_values("OR0")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(t))
    z = 1.96
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.plot([r["OR0"], r["OR1"]], [yi, yi], color="#ccc", lw=2, zorder=1)
    ax.errorbar(t["OR0"], y + 0.12, xerr=[t["OR0"] - np.exp(t["logOR0"] - z * t["se0"]),
                                          np.exp(t["logOR0"] + z * t["se0"]) - t["OR0"]],
                fmt="o", color="#999", ms=4, capsize=0, lw=0.8, label="M0: sex + race")
    col = np.where(t["q1"] < .05, ACC, "#333")
    ax.errorbar(t["OR1"], y - 0.12, xerr=[t["OR1"] - np.exp(t["logOR1"] - z * t["se1"]),
                                          np.exp(t["logOR1"] + z * t["se1"]) - t["OR1"]],
                fmt="none", ecolor=col, lw=0.8)
    ax.scatter(t["OR1"], y - 0.12, c=col, s=22, zorder=3, label="M1: + utilization, record span")
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_yticks(y); ax.set_yticklabels([short(c, 38) for c in t.index], fontsize=7.5)
    ax.set_xlabel("odds ratio, ADHD vs other NDD (95% Wald CI); red = BH q < .05 after adjustment")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); save(fig, "fig_utilization.png")


# ── Fig 10: robustness matrix ──
def fig_matrix():
    rows = ROWS
    up = lambda tab, c, col, dcol: (c in tab.index) and bool(tab.at[c, col]) and tab.at[c, dcol] > 0
    p = d1.set_index("condition")
    tests = {
        "beyond μ vs\nother NDD": lambda c: 1 if up(p, c, "surv_other_mu90", "d_other_minus_mu")
        else (-1 if p.at[c, "surv_other_mu90"] else 0),
        "sex × race\nstandardized": lambda c: 1 if (p.at[c, "surv_other_std_mu90"] and p.at[c, "d_other_std_minus_mu"] > 0)
        else (-1 if p.at[c, "surv_other_std_mu90"] else 0),
        "any-ADHD\n(n=634)": lambda c: 1 if up(d1a, c, "surv_other_mu90", "d_other_minus_mu")
        else (-1 if d1a.at[c, "surv_other_mu90"] else 0),
        "paper's 51\nconditions": lambda c: np.nan if c not in d1p.index else
        (1 if up(d1p, c, "surv_other_mu90", "d_other_minus_mu") else (-1 if d1p.at[c, "surv_other_mu90"] else 0)),
        "utilization-\nadjusted": lambda c: (1 if d7o.at[c, "logOR1"] > 0 else -1) if d7o.at[c, "q1"] < .05 else 0,
        "higher in 3 y\nbefore dx": lambda c: (1 if d5.at[c, "pre_d_vs_other"] > 0 else -1) if d5.at[c, "pre_surv90"] else 0,
        "rises after dx\n(both guards)": lambda c: (1 if d5.at[c, "DiD_other"] > 0 else -1)
        if (d5.at[c, "DiD_NT_sig"] and d5.at[c, "DiD_other_sig"]) else 0,
    }
    M = np.array([[tests[k](c) for k in tests] for c in rows], float)
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.6))
    cmap = {1: ACC, -1: C["ASD"], 0: "#f0f0f0"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            fc = "#ffffff" if np.isnan(v) else cmap[int(v)]
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.88, color=fc, ec="#ddd"))
            txt = "–" if np.isnan(v) else {1: "↑", -1: "↓", 0: ""}[int(v)]
            ax.text(j + 0.47, i + 0.46, txt, ha="center", va="center", fontsize=10,
                    color="white" if not np.isnan(v) and v != 0 else "#999")
    ax.set_xlim(0, M.shape[1]); ax.set_ylim(M.shape[0], 0)
    ax.set_xticks(np.arange(M.shape[1]) + 0.47); ax.set_xticklabels(list(tests), fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.46); ax.set_yticklabels([short(c, 40) for c in rows], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(color=ACC, label="ADHD higher (passes)"), Patch(color=C["ASD"], label="ADHD lower"),
                       Patch(color="#f0f0f0", label="not significant"), Patch(color="white", ec="#ccc", label="not in set")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_robustness.png")


if __name__ == "__main__":
    import sys
    only = set(sys.argv[1:])          # e.g. `python make_figures.py fig_profiles`
    for f in (fig_ladder, fig_forest, fig_decomp, fig_ladder_core, fig_profiles, fig_presentation,
              fig_prepost, fig_additive, fig_util, fig_matrix):
        if not only or f.__name__ in only:
            f()
    print("core:", CORE)
    print("util survivors:", UTIL_SURV)
