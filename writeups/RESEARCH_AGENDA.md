# Research agenda — questions to investigate on the spectrum cohort

Proposed 2026-09-16 after the S1–S5 spectrum analyses (see `SPECTRUM_REPORT.md`,
`RESULTS_Spectrum.md`). The data is imperfect (ascertainment/burden confounds, temporal
artifact, label overlap) but the spectrum pool opens genuinely new questions. Split into
biological and methodological. **Recurring ceiling to respect:** utilization/ascertainment is
not divided out, so every candidate finding needs a reason it isn't just "ASD kids get coded
more."

Priority spine: **M-I first** (it de-confounds everything downstream), with **B-I** run
immediately in parallel (cheap, reuses S5 machinery). B-I + B-II are the biological results most
likely to land.

---

## Biological

**B-I. Gradient vs. step, per condition (highest promise, cheap).**
For each of the 51 conditions, is the signal a monotone severity gradient
NT → other-NDD → ADHD → ASD, or an ASD-specific step that breaks the gradient? Gradient ⇒ shared
neurodevelopmental liability continuum; step ⇒ ASD-specific mechanism. Generalizes S5's
ASD-vs-rest δ to the full ordering. Reuses the shrinkage machinery. This is the user's endorsed
"both framings, let data decide" question.

**B-II. A syndromic/congenital ASD subtype.**
S5's ASD-specific set is dominated by a congenital-anomaly + epilepsy + movement cluster that
converges with Track B (B1). Do these co-occur within the *same* children? If yes → candidate
early-developmental/genetic ASD subgroup distinct from the psychiatric-tilt majority. Test via
within-ASD co-occurrence + latent-class/mixture modeling on ASD residuals (now stably estimable
via borrowed axes). Most likely route to a genuinely biological subtyping result.

**B-III. The headache/pain deficit.**
ASD shows an ASD-specific *deficit* in headache/migraine (−0.049) and pain (−0.032) — rare
negative signal, mechanistically interesting (altered interoception / pain expression /
reporting in ASD). Does it survive within high-utilization children (⇒ real) or vanish
(⇒ reporting artifact)? A deficit is hard to explain by the usual "more coding" confound, which
makes it attractive.

**B-IV. Sex-stratified comorbidity (female protective effect).**
Female-protective-effect predicts female ASD carries higher comorbidity burden. Spectrum's larger
female-NDD counts make sex × group interactions estimable. Risk: female ASD n still small — power
check first.

**B-V. Differential comorbidity network topology.**
Beyond marginal rates (B1), does ASD have distinct comorbidity *wiring* (hubs, modularity) vs
other NDD? Network structure can differ even when marginals match. Condition mode is reproducible
on the spectrum (S4), so edges are more trustworthy. Moderate risk (co-occurrence sparse).

---

## Methodological

**M-I. Ascertainment normalization — the master key (highest promise, unlocks the rest).**
Build a per-child utilization denominator (total visits / codes / observation span); redo core
residualization as rates given opportunity to be coded. A3b set the ceiling; this attempts to
break it. Pair with a **negative-control-condition calibration set** (conditions expected
group-invariant) to *bound* residual ascertainment. If it works, every downstream claim (S2, S5,
B-I…B-V) gets a confound-adjusted version. Single most valuable methods investment. First step:
check whether a usable utilization denominator exists in the tables.

**M-II. Temporal modeling done right (or a clean "why it can't be").**
Plain CP fails structurally (S4: temporal factor tracks the n-invariant density envelope). Fix or
rigorously document: (a) utilization-normalized tensor; (b) explicitly regress out per-window
density envelope, re-check guard; (c) **within-child relative timing** (age-at-first-code relative
to that child's own code density) — removes the population envelope by construction. Even a clean
negative is publishable methodology.

**M-III. Full Bayesian hierarchical model.**
S5 used method-of-moments shrinkage (DerSimonian–Laird). A proper hierarchical model with
child-level + condition-level random effects and the *ordered* group structure baked in gives
principled per-condition ASD-specificity posteriors and expresses B-I's gradient-vs-step as a
parameter. Cleaner, more reviewer-defensible inference.

**M-IV. `‖r_i‖` atypicality as a continuous phenotype.**
Carry Track B's robust per-child structural-atypicality score into the spectrum: does it order
NT < other-NDD < ADHD < ASD? Predict anything out-of-sample? Turns a categorical contrast into a
continuous individual-level measure usable as a covariate everywhere else.

---

## Status
**B-I — DONE** (2026-09-17). `s6_gradient_vs_step.py` → `results/s6_gradient_vs_step.csv`,
`s6_summary.txt`, `s6_staircase.pdf`. Result: of the 11 ASD-specific conditions (δ survives at
90%), 8 are ASD-specific **STEP**s up (epilepsy, movement, congenital musculoskeletal/digestive,
behavioral eating/sleep, personality, diabetes, glaucoma), 2 are ASD-specific **STEP-DOWN**
deficits (headache, pain — B-III's targets, ASD sits *below* the elevated spectrum), and only 1
(organic sleep, shared_frac=0.51) is a graded gradient ASD sits atop. The other 34 elevated
conditions are **SHARED** generic-NDD gradients where ASD is not specific (incl. the whole
somatic/sensory cluster; anxiety and mood actually peak in ADHD, not ASD). Headline: ASD's
*specific* signal is concentrated and largely step-shaped (congenital/neurological/behavioral),
while its *largest* elevations (anxiety, sensory, somatic) are shared spectrum-wide.

**Cross-group contrast (S7) — DONE** (2026-09-18). `s7_cross_group_contrast.py` →
`results/s7_cross_group_contrast.csv`, `s7_summary.txt`, `s7_contrast.pdf`. Extends B-I: reruns the
S5 shrinkage on three contrasts (ASD−rest, ASD−ADHD, ADHD−other). Headline: of the 11 conditions
ASD-specific vs the pooled rest, only **2 survive the ADHD-alone test** — AUTISM-SPECIFIC = epilepsy
& glaucoma. 7 SHARED-WITH-ADHD (survive vs pool, not vs ADHD alone: congenital digestive/musculo,
movement, behavioral, organic sleep, diabetes, personality), 4 ADHD-PEAK (headache, mood, acute URI,
pain — ADHD is the peak group; headache/pain = same B-III deficits). Pooling other-NDD had inflated
the apparent specificity of 9 conditions. τ: vs-rest 0.033, vs-ADHD 0.035, ADHD-vs-other 0.030
(calibration non-zero → the two non-ASD NDD groups are themselves distinguishable).

**M-II temporal — DONE, CLEAN NEGATIVE** (2026-09-18). `s8_temporal_normalized.py` →
`results/s8_temporal_normalized.csv`, `s8_summary.txt`, `s8_run.log`. Ran all three ascertainment
fixes vs the S4 baseline, scored on estimable/stable/artifact-free/**group-separating** (new: best
per-component ASD-vs-rest child-loading ROC-AUC ≥0.60). BASELINE reproduces the artifact
(|r(dens)|=0.90/0.88). REL-TIMING still artifact (|r|=0.974 — within-child relative timing did NOT
decouple; first quartile still carries disproportionate coding). DENS-EQUAL still artifact
(|r|=0.96/0.97). **UTIL-NORM R=2 is the only one clearing the guard** (|r|=0.684<0.70, stable
cos=0.998/conc=0.988) — but sepAUC=0.540 (coin-flip) → clean-negative; at R=3 it drifts back
(|r|=0.781). No approach yields a REAL (stable+clean+separating) factor. Verdict: once ascertainment
is removed, developmental *timing* adds no ASD signal beyond the marginal rates (S5–S7). Honest
limit: de-confounding is fragile (clears guard only at R=2, narrowly), but the negative is robust —
sepAUC never exceeds ~0.57 anywhere.

**Writeup:** `GRADIENT_VS_STEP.md` / `.html` now spans S6 (specificity) + S7 (cross-group) + S8
(temporal), self-contained with all terms defined and the earlier methodology questions framed.

**M-I scoped** (2026-09-17): usable per-child utilization denominator confirmed to exist
(per-child event count median 26, IQR 12–51; distinct months; distinct codes) — buildable.
Not yet implemented; now the top remaining priority (S8's UTIL-NORM shows cell-by-cell divide-out
works but is fragile → M-I is the full marginal de-confounder).

Remaining: M-I (proper build), M-III, M-IV; B-II, B-III, B-IV, B-V. Awaiting user selection.
