"""S7 (cross-group contrast) — is ASD specific even against ADHD, not just the pooled rest?

S5/S6 contrast ASD against "the rest" = ADHD + other-NDD pooled. Pooling can *manufacture*
apparent ASD-specificity: a condition equally common in ASD and ADHD but rare in other-NDD looks
"ASD-specific vs rest" only because other-NDD dilutes the pool. The sharpest specificity test is
ASD against its nearest neighbour, ADHD alone — it isolates *autism* from *attention*.

For every condition we compute three shrunk contrasts (each its own DerSimonian-Laird tau^2,
reusing the S5 machinery verbatim):

  delta_rest = p_ASD - p_rest    (== S5; ASD vs pooled ADHD+other)
  delta_AA   = p_ASD - p_ADHD    (NEW; ASD vs ADHD alone -- the hard specificity test)
  delta_AO   = p_ADHD - p_other  (calibration; are the two non-ASD NDD groups even separable?)

Crossing "survives vs rest" with "survives vs ADHD" tells us, per condition, whether an
ASD-specific signal is truly autism-specific (survives vs ADHD too) or shared with ADHD (survives
vs the pool but not vs ADHD). We also record the peak group (argmax prevalence) so ADHD-peaked
conditions -- where ADHD, not ASD, is the most affected group -- are surfaced explicitly.

Outputs (results/):
  s7_cross_group_contrast.csv   per-condition: 4 prevalences, 3 shrunk deltas + survival, peak, label
  s7_summary.txt                counts + the autism-specific / shared-with-ADHD / ADHD-peak lists
  s7_contrast.pdf               delta_rest (x) vs delta_AA (y) scatter; quadrants = the specificity story
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

import gus_common as gc
import s1_axis_robustness as s1
import s5_hierarchical_shrinkage as s5      # reuse prevalence() + dersimonian_laird()

OUT = gc.OUT_DIR
MIN_E = 0.03
GROUPS = ["NT", "Other NDD", "ADHD", "ASD"]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


def shrink_contrast(pa, na, pb, nb):
    """Shrunk per-condition difference pa-pb with 90/95% survival, S5-style EB shrinkage."""
    delta = pa - pb
    se = np.maximum(np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb), 1e-6)
    tau2, mu_re, _, _ = s5.dersimonian_laird(delta, se)
    B = tau2 / (tau2 + se**2)
    d_shrunk = mu_re + B * (delta - mu_re)
    post_sd = np.sqrt(B * se**2)
    z90, z95 = stats.norm.ppf(0.95), stats.norm.ppf(0.975)
    surv90 = (d_shrunk - z90 * post_sd > 0) | (d_shrunk + z90 * post_sd < 0)
    surv95 = (d_shrunk - z95 * post_sd > 0) | (d_shrunk + z95 * post_sd < 0)
    return d_shrunk, surv90, surv95, np.sqrt(tau2)


def label_row(e_asd, dr_s, dr_surv, da_s, da_surv, peak):
    """Cross 'survives vs rest' x 'survives vs ADHD' into a specificity label."""
    if abs(e_asd) < MIN_E and not dr_surv and not da_surv:
        return "FLAT"
    if da_surv and da_s > 0:
        return "AUTISM-SPECIFIC"          # ASD exceeds even ADHD
    if da_surv and da_s < 0:
        return "ADHD-PEAK"                # ADHD exceeds ASD (survives)
    if dr_surv and dr_s > 0:
        return "SHARED-WITH-ADHD"         # ASD-specific vs pool, but not vs ADHD alone
    if dr_surv and dr_s < 0:
        return "ASD-DEFICIT"             # below the pooled spectrum
    return "SHARED-NDD"                   # elevated but not specific in any contrast


def main():
    sys.stdout = Tee(os.path.join(OUT, "s7_run.log"))
    conditions = sorted(l.strip() for l in open(s1.KEPT) if l.strip())
    patients = gc.load_patients(s1.FULL_PAT)
    df = gc.load_event_data(s1.FULL_EVT, s1.FULL_PAT)
    ids = {g: sorted(patients.loc[patients["label"] == g, "ID"]) for g in GROUPS}
    rest = sorted(ids["ADHD"] + ids["Other NDD"])

    p = {g: s5.prevalence(df, ids[g], conditions)[0] for g in GROUPS}
    n = {g: len(ids[g]) for g in GROUPS}
    p_rest, n_rest = s5.prevalence(df, rest, conditions)
    e_asd = p["ASD"] - p["NT"]
    print(f"n: " + ", ".join(f"{g}={n[g]}" for g in GROUPS) + f"  rest={n_rest}")

    dr, dr90, dr95, tau_r = shrink_contrast(p["ASD"], n["ASD"], p_rest, n_rest)
    da, da90, da95, tau_a = shrink_contrast(p["ASD"], n["ASD"], p["ADHD"], n["ADHD"])
    do, do90, do95, tau_o = shrink_contrast(p["ADHD"], n["ADHD"], p["Other NDD"], n["Other NDD"])
    print(f"tau: vs-rest={tau_r:.4f}  vs-ADHD={tau_a:.4f}  ADHD-vs-other(calib)={tau_o:.4f}")

    peak = [GROUPS[int(np.argmax([p[g][j] for g in GROUPS]))] for j in range(len(conditions))]
    lab = [label_row(e_asd[j], dr[j], dr90[j], da[j], da90[j], peak[j])
           for j in range(len(conditions))]

    tab = pd.DataFrame({
        "condition": conditions,
        "p_NT": p["NT"], "p_otherNDD": p["Other NDD"], "p_ADHD": p["ADHD"], "p_ASD": p["ASD"],
        "e_ASD_vs_NT": e_asd,
        "delta_vs_rest": dr, "surv_rest90": dr90,
        "delta_vs_ADHD": da, "surv_ADHD90": da90, "surv_ADHD95": da95,
        "delta_ADHD_vs_other": do, "surv_calib90": do90,
        "peak_group": peak, "label": lab,
    }).sort_values(["label", "delta_vs_ADHD"],
                   key=lambda s: s if s.name == "label" else s.abs(),
                   ascending=[True, False]).reset_index(drop=True)
    tab.round(4).to_csv(os.path.join(OUT, "s7_cross_group_contrast.csv"), index=False)

    counts = pd.Series(lab).value_counts()
    print("\n=== label counts ===")
    for k in ["AUTISM-SPECIFIC", "SHARED-WITH-ADHD", "ADHD-PEAK", "ASD-DEFICIT",
              "SHARED-NDD", "FLAT"]:
        print(f"  {k:<18} {int(counts.get(k, 0))}")

    # The headline cross-tab: of the S5 ASD-specific set (survives vs rest), how many survive vs ADHD?
    spec_rest = tab[tab["surv_rest90"]]
    kept = int((spec_rest["surv_ADHD90"] & (spec_rest["delta_vs_ADHD"] > 0)).sum())
    print(f"\nOf {len(spec_rest)} conditions ASD-specific vs the pooled rest (S5), "
          f"{kept} remain ASD>ADHD (survive the harder ADHD-alone contrast).")

    def show(label):
        sub = tab[tab["label"] == label]
        for _, r in sub.iterrows():
            print(f"  {r['condition']:<44} NT->oth->ADHD->ASD = {r['p_NT']:.3f} "
                  f"{r['p_otherNDD']:.3f} {r['p_ADHD']:.3f} {r['p_ASD']:.3f}  "
                  f"d(vs rest)={r['delta_vs_rest']:+.3f}{'*' if r['surv_rest90'] else ' '} "
                  f"d(vs ADHD)={r['delta_vs_ADHD']:+.3f}{'*' if r['surv_ADHD90'] else ' '} "
                  f"peak={r['peak_group']}")

    print("\n=== AUTISM-SPECIFIC (ASD > ADHD, survives) ===");   show("AUTISM-SPECIFIC")
    print("\n=== SHARED-WITH-ADHD (ASD-specific vs pool, but not vs ADHD) ==="); show("SHARED-WITH-ADHD")
    print("\n=== ADHD-PEAK (ADHD exceeds ASD, survives) ===");   show("ADHD-PEAK")
    print("\n=== ASD-DEFICIT ===");                               show("ASD-DEFICIT")

    with open(os.path.join(OUT, "s7_summary.txt"), "w") as f:
        f.write(f"Cross-group contrast (n_ASD={n['ASD']}, ADHD={n['ADHD']}, "
                f"other={n['Other NDD']}, rest={n_rest}, NT={n['NT']})\n")
        f.write(f"tau: vs-rest={tau_r:.4f}, vs-ADHD={tau_a:.4f}, ADHD-vs-other={tau_o:.4f}\n")
        f.write("label counts: " + ", ".join(
            f"{k}={int(counts.get(k,0))}" for k in
            ["AUTISM-SPECIFIC", "SHARED-WITH-ADHD", "ADHD-PEAK", "ASD-DEFICIT",
             "SHARED-NDD", "FLAT"]) + "\n")
        f.write(f"Of {len(spec_rest)} ASD-specific-vs-rest conditions, {kept} remain ASD>ADHD.\n")
        for label in ["AUTISM-SPECIFIC", "SHARED-WITH-ADHD", "ADHD-PEAK", "ASD-DEFICIT"]:
            sub = tab[tab["label"] == label]
            f.write(f"\n{label}:\n")
            for _, r in sub.iterrows():
                f.write(f"  {r['condition']}: d_rest={r['delta_vs_rest']:+.3f}"
                        f"{'*' if r['surv_rest90'] else ''}, d_ADHD={r['delta_vs_ADHD']:+.3f}"
                        f"{'*' if r['surv_ADHD90'] else ''}, peak={r['peak_group']}\n")

    _scatter(tab, os.path.join(OUT, "s7_contrast.pdf"))
    print(f"\nDone. Outputs in {OUT}")


def _scatter(tab, path):
    """delta_vs_rest (x) vs delta_vs_ADHD (y): quadrants tell the specificity story."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cmap = {"AUTISM-SPECIFIC": "#c1272d", "SHARED-WITH-ADHD": "#e08214",
            "ADHD-PEAK": "#3b6ea5", "ASD-DEFICIT": "#5b3a8c",
            "SHARED-NDD": "#4c9f70", "FLAT": "#bbbbbb"}
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for label, c in cmap.items():
        s = tab[tab["label"] == label]
        ax.scatter(s["delta_vs_rest"], s["delta_vs_ADHD"], s=34, c=c, alpha=0.85,
                   edgecolors="none", label=f"{label} ({len(s)})")
    # annotate the conditions that survive the ADHD-alone test, plus ADHD-peaks
    for _, r in tab.iterrows():
        if r["surv_ADHD90"] or r["label"] == "ADHD-PEAK":
            ax.annotate(r["condition"][:24], (r["delta_vs_rest"], r["delta_vs_ADHD"]),
                        fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    lim = float(np.nanmax(np.abs(np.r_[tab["delta_vs_rest"], tab["delta_vs_ADHD"]]))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "--", color="#999", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("ASD-specific deviation vs POOLED rest  (delta_vs_rest, = S5)")
    ax.set_ylabel("ASD deviation vs ADHD alone  (delta_vs_ADHD)")
    ax.set_title("Cross-group contrast: does ASD-specificity survive the ADHD-alone test?\n"
                 "upper-right = autism-specific beyond attention · below 0 = ADHD-peaked")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
