"""S6 (research agenda B-I) — gradient vs. step, per condition.

The direct follow-up to S5, sitting at the intersection of the two headline spectrum
opportunities: ASD-specific per-condition inference (S5) and cross-group contrast (S2).

S5 gave one number per condition — delta = p_ASD - p_rest, the ASD-specific deviation. S6
asks the *shape* question the user endorsed ("both framings, let data decide"): across the
ordered groups NT -> other-NDD -> ADHD -> ASD, is a condition's ASD signal

  (a) a monotone GRADIENT  — a shared neurodevelopmental liability continuum ASD sits atop, or
  (b) an ASD-specific STEP — a jump concentrated at ASD, the rest of the spectrum near NT.

We report, per condition, the honest evidence (all four raw group prevalences), two continuous
descriptors, and a transparent categorical label from their cross:

  - delta_shrunk (+ 90% survival): ASD-specific excess/deficit, reused verbatim from S5.
  - shared_frac  = (p_rest - p_NT) / (p_ASD - p_NT): how far the rest-of-spectrum sits between
                   NT and ASD. ~0 => rest near NT (abrupt/step); ~0.5 => halfway (gradient);
                   >=~0.85 => rest ~ ASD (generic NDD, ASD adds nothing).
  - CA_trend_z   : Cochran-Armitage trend z across the four groups (fixed clinical order
                   NT<other<ADHD<ASD; the order is an assumption, so the four raw prevalences
                   are carried alongside so the actual shape is always visible).

Categories (delta survival is checked first, so an ASD-specific signal is never hidden by a
small ASD-vs-NT elevation — the deficits sit near NT precisely because the rest of the
spectrum is elevated and ASD is not):
  STEP        : delta survives (90%), delta>0, shared_frac < 0.5   (ASD-specific excess jump)
  GRADIENT+   : delta survives (90%), delta>0, shared_frac >= 0.5  (graded, ASD boosted on top)
  STEP-DOWN   : delta survives (90%), delta<0                      (ASD-specific deficit: ASD
                                                                    below the elevated spectrum)
  SHARED      : delta n.s., |e_ASD|>=MIN_E, shared_frac >= 0.5     (generic NDD gradient/plateau)
  ASD-LEAN    : delta n.s., |e_ASD|>=MIN_E, shared_frac < 0.5      (ASD higher than rest, n.s.)
  FLAT        : delta n.s. AND |e_ASD| < MIN_E

Outputs (results/):
  s6_gradient_vs_step.csv   all 51 conditions: 4 group prevalences, elevations, delta, shared_frac, CA z, category
  s6_summary.txt            category counts + the STEP and SHARED headline lists
  s6_staircase.pdf          four-group prevalence "staircase" for curated exemplars
  s6_run.log
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
MIN_E = 0.03                                 # min |ASD-vs-NT elevation| to be "notable"
GROUP_ORDER = ["NT", "Other NDD", "ADHD", "ASD"]
GROUP_SCORE = {"NT": 0, "Other NDD": 1, "ADHD": 2, "ASD": 3}


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


def ca_trend_z(counts, ns, scores):
    """Cochran-Armitage trend z-statistic. counts/ns/scores are per-group arrays."""
    N = ns.sum(); R = counts.sum()
    p = R / N
    t = scores
    T = np.sum(t * (counts - ns * p))
    var = p * (1 - p) * (np.sum(ns * t**2) - (np.sum(ns * t))**2 / N)
    return float(T / np.sqrt(var)) if var > 0 else 0.0


def classify(e_asd, delta_surv, shared_frac, delta_shrunk):
    # ASD-specific signal (delta survives) is decided first, so a deficit whose ASD-vs-NT
    # elevation is ~0 (because the rest of the spectrum is elevated and ASD is not) is never
    # mislabeled FLAT — it is the purest ASD-specific signal there is.
    if delta_surv:
        if delta_shrunk < 0:
            return "STEP-DOWN"
        return "STEP" if (shared_frac < 0.5) else "GRADIENT+"
    if abs(e_asd) < MIN_E:
        return "FLAT"
    return "SHARED" if (shared_frac >= 0.5) else "ASD-LEAN"


def main():
    sys.stdout = Tee(os.path.join(OUT, "s6_run.log"))
    conditions = sorted(l.strip() for l in open(s1.KEPT) if l.strip())
    patients = gc.load_patients(s1.FULL_PAT)
    df = gc.load_event_data(s1.FULL_EVT, s1.FULL_PAT)
    ids = {g: sorted(patients.loc[patients["label"] == g, "ID"]) for g in GROUP_ORDER}
    rest = sorted(ids["ADHD"] + ids["Other NDD"])

    # raw per-group prevalence (the honest four-group evidence)
    p = {g: s5.prevalence(df, ids[g], conditions)[0] for g in GROUP_ORDER}
    n = {g: len(ids[g]) for g in GROUP_ORDER}
    p_rest, n_rest = s5.prevalence(df, rest, conditions)
    n_asd = n["ASD"]
    print(f"n: " + ", ".join(f"{g}={n[g]}" for g in GROUP_ORDER) + f"  (rest={n_rest})")

    e = {g: p[g] - p["NT"] for g in GROUP_ORDER}                 # elevation vs NT (raw)
    delta = p["ASD"] - p_rest                                    # ASD-specific (== S5)
    se = np.sqrt(p["ASD"] * (1 - p["ASD"]) / n_asd + p_rest * (1 - p_rest) / n_rest)
    se = np.maximum(se, 1e-6)
    tau2, mu_re, Q, m = s5.dersimonian_laird(delta, se)          # same shrinkage as S5
    B = tau2 / (tau2 + se**2)
    delta_shrunk = mu_re + B * (delta - mu_re)
    post_sd = np.sqrt(B * se**2)
    z90 = stats.norm.ppf(0.95)
    surv90 = (delta_shrunk - z90 * post_sd > 0) | (delta_shrunk + z90 * post_sd < 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        shared_frac = np.where(np.abs(e["ASD"]) > 1e-9,
                               (p_rest - p["NT"]) / e["ASD"], np.nan)

    ca = np.array([ca_trend_z(
        np.array([p[g][j] * n[g] for g in GROUP_ORDER]),
        np.array([n[g] for g in GROUP_ORDER]),
        np.array([GROUP_SCORE[g] for g in GROUP_ORDER], float)) for j in range(len(conditions))])

    cat = [classify(e["ASD"][j], bool(surv90[j]), shared_frac[j], delta_shrunk[j])
           for j in range(len(conditions))]

    tab = pd.DataFrame({
        "condition": conditions,
        "p_NT": p["NT"], "p_otherNDD": p["Other NDD"], "p_ADHD": p["ADHD"], "p_ASD": p["ASD"],
        "e_otherNDD": e["Other NDD"], "e_ADHD": e["ADHD"], "e_ASD": e["ASD"],
        "delta_raw": delta, "delta_shrunk": delta_shrunk, "surv90": surv90,
        "shared_frac": shared_frac, "CA_trend_z": ca, "category": cat,
    }).sort_values(["category", "e_ASD"], key=lambda s: s if s.name == "category" else s.abs(),
                   ascending=[True, False]).reset_index(drop=True)
    tab.round(4).to_csv(os.path.join(OUT, "s6_gradient_vs_step.csv"), index=False)

    counts = pd.Series(cat).value_counts()
    print("\n=== category counts ===")
    for k in ["STEP", "STEP-DOWN", "GRADIENT+", "SHARED", "ASD-LEAN", "FLAT"]:
        print(f"  {k:<10} {int(counts.get(k, 0))}")

    def show(label):
        sub = tab[tab["category"] == label]
        for _, r in sub.iterrows():
            print(f"  {r['condition']:<44} e_ASD={r['e_ASD']:+.3f}  "
                  f"NT->oth->ADHD->ASD = {r['p_NT']:.3f} {r['p_otherNDD']:.3f} "
                  f"{r['p_ADHD']:.3f} {r['p_ASD']:.3f}  "
                  f"delta*={r['delta_shrunk']:+.3f}{'*' if r['surv90'] else ' '}  "
                  f"shared={r['shared_frac']:.2f}  CAz={r['CA_trend_z']:+.1f}")

    print("\n=== STEP (ASD-specific jump; rest-of-spectrum near NT) ===");     show("STEP")
    print("\n=== STEP-DOWN (ASD-specific deficit; ASD below elevated spectrum) ==="); show("STEP-DOWN")
    print("\n=== GRADIENT+ (graded, ASD boosted on top) ===");                 show("GRADIENT+")
    print("\n=== SHARED (generic NDD gradient/plateau; ASD not specific) ==="); show("SHARED")

    with open(os.path.join(OUT, "s6_summary.txt"), "w") as f:
        f.write(f"Gradient vs step, per condition (n_ASD={n_asd}, rest={n_rest}, "
                f"NT={n['NT']}). tau={np.sqrt(tau2):.4f}\n")
        f.write("category counts: " + ", ".join(
            f"{k}={int(counts.get(k,0))}" for k in
            ["STEP", "STEP-DOWN", "GRADIENT+", "SHARED", "ASD-LEAN", "FLAT"]) + "\n")
        for label in ["STEP", "STEP-DOWN", "GRADIENT+", "SHARED"]:
            sub = tab[tab["category"] == label]
            f.write(f"\n{label}:\n")
            for _, r in sub.iterrows():
                f.write(f"  {r['condition']}: e_ASD={r['e_ASD']:+.3f}, "
                        f"delta_shrunk={r['delta_shrunk']:+.3f}"
                        f"{'*' if r['surv90'] else ''}, shared_frac={r['shared_frac']:.2f}\n")

    _staircase(tab, os.path.join(OUT, "s6_staircase.pdf"))
    print(f"\nDone. Outputs in {OUT}")


def _staircase(tab, path):
    """Four-group prevalence staircase for curated exemplars: gradient vs step made visual."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = pd.concat([
        tab[tab["category"] == "STEP"].head(6),
        tab[tab["category"] == "STEP-DOWN"],
        tab[tab["category"] == "GRADIENT+"].reindex(
            tab[tab["category"] == "GRADIENT+"]["e_ASD"].abs().sort_values(ascending=False).index).head(4),
        tab[tab["category"] == "SHARED"].reindex(
            tab[tab["category"] == "SHARED"]["e_ASD"].abs().sort_values(ascending=False).index).head(4),
    ])
    cmap = {"STEP": "#c1272d", "STEP-DOWN": "#5b3a8c", "GRADIENT+": "#e08214", "SHARED": "#4c9f70"}
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in pick.iterrows():
        ys = [r["p_NT"], r["p_otherNDD"], r["p_ADHD"], r["p_ASD"]]
        ax.plot(x, ys, "-o", ms=4, lw=1.6, color=cmap[r["category"]], alpha=0.85)
        ax.annotate(r["condition"][:26], (3, ys[3]), fontsize=7,
                    xytext=(4, 0), textcoords="offset points", va="center",
                    color=cmap[r["category"]])
    ax.set_xticks(x); ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylabel("condition prevalence (raw)")
    ax.set_title("Gradient vs step: per-condition prevalence across the spectrum\n"
                 "red=ASD-specific step · purple=ASD-specific deficit · "
                 "orange=graded+ASD-boost · green=shared/generic NDD")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=c, marker="o", label=k)
                       for k, c in cmap.items()], fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
