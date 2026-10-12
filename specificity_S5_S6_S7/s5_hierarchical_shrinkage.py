"""S5 — Hierarchical shared + ASD-specific deviation, with shrinkage.

The S2 structural follow-on: borrow power from the whole spectrum for ASD-SPECIFIC
inference. The current study reports, per condition, the ASD elevation over typical
children  e_ASD = p_ASD - NT. With the excluded groups we decompose it:

    e_ASD(j)  =  s(j)         +  delta(j)
                 shared NDD       ASD-specific
                 = p_rest - NT     = p_ASD - p_rest

  - s(j)      : shared neurodevelopmental elevation, estimated from the 1,503 excluded
                children (ADHD + other-NDD) vs 1,851 NT  -> tight.
  - delta(j)  : how much ASD departs from the rest of the NDD spectrum, estimated from
                145 ASD vs 1,503 non-ASD-NDD -> noisy -> shrunk.

Two borrowings of power, both from including the excluded groups:
  (a) the shared level s(j) is pinned down by the whole spectrum;
  (b) the 51 noisy delta(j) are stabilized by empirical-Bayes shrinkage across conditions
      (DerSimonian-Laird tau^2; normal-normal posterior).

S2 (ASD-vs-other AUC ~0.59) predicts most delta shrink toward ~0. The deliverable is the
honest per-condition verdict: which apparent ASD-vs-typical associations are really generic
NDD (large e_ASD, delta ~ 0) vs genuinely ASD-specific (delta credibly != 0) — a question
the ASD-vs-NT design cannot pose.

Outputs (results/):
  s5_asd_specific.csv   per-condition decomposition + raw/shrunk delta + 90/95% flags
  s5_summary.txt        global ASD-vs-rest shift, tau, #surviving, headline
  s5_forest.pdf         forest (raw vs shrunk delta) + decomposition bars
  s5_run.log
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

import gus_common as gc
import s1_axis_robustness as s1

OUT = gc.OUT_DIR


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


def prevalence(df, ids, conditions):
    """Directly-observed presence rate per condition over `ids` (rows 0/1)."""
    raw, _ = gc.raw_binary_matrix(df, ids, conditions)
    p = raw.mean(axis=0)
    n = len(ids)
    return p, n


def dersimonian_laird(delta, se):
    """Random-effects tau^2 (DL) + RE global mean over the per-condition deltas."""
    w = 1.0 / se**2
    mu_fe = np.sum(w * delta) / np.sum(w)
    Q = np.sum(w * (delta - mu_fe) ** 2)
    m = len(delta)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (Q - (m - 1)) / c) if c > 0 else 0.0
    wstar = 1.0 / (se**2 + tau2)
    mu_re = np.sum(wstar * delta) / np.sum(wstar)
    return tau2, mu_re, Q, m


def main():
    sys.stdout = Tee(os.path.join(OUT, "s5_run.log"))
    conditions = sorted(l.strip() for l in open(s1.KEPT) if l.strip())
    patients = gc.load_patients(s1.FULL_PAT)
    df = gc.load_event_data(s1.FULL_EVT, s1.FULL_PAT)

    ids = {g: sorted(patients.loc[patients["label"] == g, "ID"])
           for g in ("ASD", "ADHD", "Other NDD", "NT")}
    rest = sorted(ids["ADHD"] + ids["Other NDD"])                 # non-ASD NDD spectrum

    nt_rate, _ = s1.nt_expected_rate(df, patients, conditions)    # standardized NT (1,851)
    p_asd, n_asd = prevalence(df, ids["ASD"], conditions)         # 145
    p_rest, n_rest = prevalence(df, rest, conditions)             # 1,503
    print(f"ASD n={n_asd}  non-ASD-NDD n={n_rest}  NT (standardized) baseline; "
          f"{len(conditions)} conditions")

    e_asd = p_asd - nt_rate                # ASD elevation over NT (what the study reports)
    shared = p_rest - nt_rate             # shared NDD elevation over NT
    delta = p_asd - p_rest                # ASD-specific deviation (raw)
    se = np.sqrt(p_asd * (1 - p_asd) / n_asd + p_rest * (1 - p_rest) / n_rest)
    se = np.maximum(se, 1e-6)

    tau2, mu_re, Q, m = dersimonian_laird(delta, se)
    tau = np.sqrt(tau2)
    B = tau2 / (tau2 + se**2)                                      # shrinkage weight (0..1)
    delta_shrunk = mu_re + B * (delta - mu_re)
    post_sd = np.sqrt(B * se**2)                                   # normal-normal posterior sd
    z90, z95 = stats.norm.ppf(0.95), stats.norm.ppf(0.975)
    lo90, hi90 = delta_shrunk - z90 * post_sd, delta_shrunk + z90 * post_sd
    lo95, hi95 = delta_shrunk - z95 * post_sd, delta_shrunk + z95 * post_sd
    surv90 = (lo90 > 0) | (hi90 < 0)
    surv95 = (lo95 > 0) | (hi95 < 0)

    tab = pd.DataFrame({
        "condition": conditions,
        "nt_rate": nt_rate, "p_asd": p_asd, "p_rest": p_rest,
        "e_asd_vs_nt": e_asd, "shared_ndd": shared,
        "delta_raw": delta, "se": se, "shrink_B": B,
        "delta_shrunk": delta_shrunk, "post_lo90": lo90, "post_hi90": hi90,
        "surv90": surv90, "surv95": surv95,
        "shared_frac_of_e": np.where(np.abs(e_asd) > 1e-9, shared / e_asd, np.nan),
    }).sort_values("delta_shrunk", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    tab.round(4).to_csv(os.path.join(OUT, "s5_asd_specific.csv"), index=False)

    print(f"\nGlobal ASD-vs-rest shift (RE mean mu) = {mu_re:+.4f}  "
          f"(ASD {'more' if mu_re>0 else 'less'} elevated than non-ASD NDD on average)")
    print(f"Between-condition SD of ASD-specific deviation tau = {tau:.4f}  "
          f"(Q={Q:.1f}, df={m-1})")
    print(f"Mean shrinkage weight B = {B.mean():.3f}  "
          f"(0=fully shrunk to mu, 1=data trusted); "
          f"mean |delta_raw|={np.abs(delta).mean():.4f} -> "
          f"|delta_shrunk|={np.abs(delta_shrunk).mean():.4f}")
    print(f"ASD-specific conditions surviving shrinkage: "
          f"{int(surv90.sum())}/{len(conditions)} at 90%, "
          f"{int(surv95.sum())}/{len(conditions)} at 95%")

    print("\n=== Conditions with a credible ASD-SPECIFIC deviation (90%) ===")
    surv = tab[tab["surv90"]]
    if len(surv):
        for _, r in surv.iterrows():
            direc = "excess" if r["delta_shrunk"] > 0 else "deficit"
            print(f"  {r['condition']:<46} delta={r['delta_shrunk']:+.3f} "
                  f"[{r['post_lo90']:+.3f},{r['post_hi90']:+.3f}]  {direc} vs rest-of-spectrum "
                  f"(raw ASD-vs-NT e={r['e_asd_vs_nt']:+.3f})")
    else:
        print("  none")

    print("\n=== 'Looks ASD-associated vs typical, but is generic NDD' "
          "(largest e_asd_vs_nt with delta NOT surviving) ===")
    generic = tab[~tab["surv90"]].reindex(
        tab[~tab["surv90"]]["e_asd_vs_nt"].abs().sort_values(ascending=False).index).head(8)
    for _, r in generic.iterrows():
        sf = r["shared_frac_of_e"]
        print(f"  {r['condition']:<46} e(ASD-vs-NT)={r['e_asd_vs_nt']:+.3f}  "
              f"shared={r['shared_ndd']:+.3f} ({sf*100:.0f}% of e)  "
              f"ASD-specific delta_shrunk={r['delta_shrunk']:+.3f} (n.s.)")

    with open(os.path.join(OUT, "s5_summary.txt"), "w") as f:
        f.write(f"Hierarchical shared+ASD-specific decomposition (n_ASD={n_asd}, "
                f"n_rest={n_rest}, {len(conditions)} conditions)\n")
        f.write(f"Global ASD-vs-rest shift mu = {mu_re:+.4f}; tau = {tau:.4f}\n")
        f.write(f"Mean shrinkage |delta_raw| {np.abs(delta).mean():.4f} -> "
                f"|delta_shrunk| {np.abs(delta_shrunk).mean():.4f}\n")
        f.write(f"ASD-specific conditions surviving: {int(surv90.sum())}/{len(conditions)} "
                f"(90%), {int(surv95.sum())}/{len(conditions)} (95%)\n")
        f.write("Surviving (90%): " + ", ".join(
            f"{r['condition']}({r['delta_shrunk']:+.3f})"
            for _, r in surv.iterrows()) + "\n")

    _plot(tab, os.path.join(OUT, "s5_forest.pdf"))
    print(f"\nDone. Outputs in {OUT}")


def _plot(tab, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = tab.head(20).iloc[::-1]                                  # 20 largest |delta_shrunk|
    fig, (axf, axd) = plt.subplots(1, 2, figsize=(13, 7),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(top))
    # forest: raw (light) vs shrunk (dark) with 90% CI
    axf.scatter(top["delta_raw"], y + 0.15, s=22, color="#9aa7b5", label="raw δ")
    axf.errorbar(top["delta_shrunk"], y - 0.15, xerr=[top["delta_shrunk"] - top["post_lo90"],
                 top["post_hi90"] - top["delta_shrunk"]], fmt="o", ms=5, color="#c1272d",
                 ecolor="#c1272d", capsize=2, label="shrunk δ (90% CI)")
    axf.axvline(0, color="k", lw=0.8)
    axf.set_yticks(y); axf.set_yticklabels(top["condition"], fontsize=8)
    axf.set_xlabel("ASD-specific deviation  (p_ASD − p_rest-of-spectrum)")
    axf.set_title("Shrinkage pulls noisy per-condition ASD deviations toward 0", fontsize=10)
    axf.legend(fontsize=8, loc="lower right"); axf.grid(axis="x", alpha=0.3)
    # decomposition: shared NDD vs ASD-specific for the largest ASD-vs-NT elevations
    dec = tab.reindex(tab["e_asd_vs_nt"].abs().sort_values(ascending=False).index).head(20).iloc[::-1]
    yd = np.arange(len(dec))
    axd.barh(yd, dec["shared_ndd"], color="#4c9f70", label="shared NDD  (p_rest − NT)")
    axd.barh(yd, dec["delta_shrunk"], left=dec["shared_ndd"], color="#c1272d",
             label="ASD-specific δ (shrunk)")
    axd.axvline(0, color="k", lw=0.8)
    axd.set_yticks(yd); axd.set_yticklabels(dec["condition"], fontsize=8)
    axd.set_xlabel("elevation over NT")
    axd.set_title("What the ASD-vs-NT elevation is made of", fontsize=10)
    axd.legend(fontsize=8, loc="lower right"); axd.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
