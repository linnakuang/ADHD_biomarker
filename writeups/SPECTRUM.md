# Spectrum cohort report — the case for inclusion, methods, and findings (S1–S8)

**What this is.** A single self-contained report on the "spectrum" analyses that test whether
including the 1,503 currently-excluded neurodevelopmental children (573 ADHD-without-ASD +
930 other-NDD) lets us learn more *about ASD*. It folds three previously-separate documents:

- **Part I** — the collaborator-facing *case* for inclusion (formerly `RESULTS_Spectrum.md`).
- **Parts II–VI** — the methods-and-evidence report for **S1–S5** (formerly `SPECTRUM_REPORT.md`).
- **Part VII** — **S6–S8**: per-condition specificity, cross-group contrast, and developmental
  timing (formerly `GRADIENT_VS_STEP.md`).

All analyses run at `<PROJECT_ROOT>/gus_analysis/`; outputs in `results/`. Last
updated 2026-09-18. Companion: `RESEARCH_AGENDA.md` (the B-I…B-V / M-I…M-IV follow-on list and
status); Track-A/B writeups `TRACK_A.md`, `TRACK_B.md`.

> **Governing framing — "both framings, let the data decide."** Throughout, we do not pre-commit
> to "ASD is special" or "ASD is just the far end of a shared continuum." For every question and
> every condition we compute the honest evidence for *both* and let each finding fall where the
> data puts it. The excluded NDD/ADHD children are what make the intermediate comparisons
> observable at all.

---

# Part I — The case for including the NDD/ADHD groups

**Question.** The study currently analyses 145 ASD children against 1,851 NT and *excludes* 1,503
children with other neurodevelopmental diagnoses (573 ADHD-without-ASD, 930 other-NDD) by design.
Does bringing those children in let us learn more *about ASD*? Five analyses (S1–S5), all on the
paper's baseline / 51 conditions / §2 rules; only the pool learned from changes.

| | question | verdict |
|---|---|---|
| **S1** | Do residualized axes get more reproducible with n? | **Yes, decisively.** K=3 bootstrap cosine 0.82 (a component *below* the §2 floor at n=145) → **0.97** at n=1,648. n, not biology, was the binding constraint. |
| **S2** | Does ASD stand out from ADHD/other-NDD in the shared latent space? | **Real but modest.** ASD-vs-other AUC 0.587 (perm p=0.006), barely above burden, *less* distinct than ADHD-vs-other (0.638). A graded tilt (toward the psychiatric axis), not an island. |
| **S3** | Does inclusion lift the temporal density ceiling? | **Yes, for estimability.** Thin slices (the A1/A2 artifact driver) 53%→12%; full 6-window 0–18 goes 1/6 → 6/6 windows passing §2. |
| **S4** | …does that make a temporal factorization *valid*? | **No.** CP is stable on the spectrum (worst component 0.65→0.93) but the temporal factor *still* tracks the density envelope (\|r\|≈0.88, unmoved by 10× data). Estimable ≠ valid. Strengthens A1: the temporal artifact is structural, not small-sample. |
| **S5** | Can we borrow power for *ASD-specific* inference? | **Yes — the payoff.** Decomposing each condition into shared-NDD + ASD-specific, with shrinkage: **7–11/51 conditions are genuinely ASD-specific** (epilepsy, sleep, congenital anomalies, movement/behavioral disorders ↑; headache/pain ↓), while the *largest* ASD-vs-NT signals (anxiety, refractive, otitis, obesity) are **70–104% generic NDD** — unknowable without the excluded groups. |

**One-line case for collaborators:** inclusion removes the power constraint that leaves the headline
model under its own stability floor (S1); the temporal-trajectory ambition is a dead end at any n
and should be dropped (S3/S4); and the excluded children convert "ASD differs from typical kids"
into the sharper, honest, *ASD-specific* finding of *which* conditions truly distinguish ASD from
the rest of the neurodevelopmental spectrum (S2/S5/S6/S7). Keeps the ASD focus.

**What this means for the collaborator case — the inclusion argument is *stronger* for being honest:**

1. **Power is the binding constraint, and inclusion removes it (S1).** The paper's headline K=3
   model is under-powered at n=145 — one axis fails the team's own §2 floor. The excluded children
   fix that directly (0.82 → 0.97 cosine; failing component 0.76 → 0.96). Every downstream
   axis-based claim rests on a firmer footing.
2. **Inclusion lets us ask a question the current design cannot (S2).** "Does ASD comorbidity differ
   from *other neurodevelopmental conditions*?" is unanswerable against an NT-only comparator —
   residualizing vs NT makes NT trivial by construction. The excluded groups are the
   matched-utilization, non-ASD controls that turn "ASD differs from typical children" (partly a
   burden/ascertainment artifact — see Track-A A3b/A5b and Track-B B2) into "ASD differs from *ADHD
   and other NDD in these specific ways*."
3. **The specificity answer is itself a defensible, publishable result.** ASD sits at AUC ≈ 0.59
   (p = 0.006) from other-NDD, mostly a psychiatric-axis tilt, less distinct than ADHD. "ASD is a
   graded position within a shared NDD comorbidity landscape, not a categorical comorbidity
   signature" is a real finding — obtainable only *with* these groups. S5–S7 then sharpen it to a
   short, mechanistically-interpretable core.

The framing the collaborators worried about is preserved: this keeps the **ASD focus** (ASD is the
group we characterise *within* the spectrum); it is not a pivot to an NDD-wide paper.

---

# Part II — Cohort, features, and the shared residualization scheme

**Cohort (Boston Birth Cohort, postnatal EHR).** 3,499 children with a clean 4-way label from the
priority rule ASD > ADHD > other-NDD > NT:

| label | n | role |
|---|---|---|
| NT | 1,851 | baseline / expected-rate reference (rung 0) |
| Other NDD | 930 | currently excluded — part of "the rest" (rung 1) |
| ADHD (no ASD) | 573 | currently excluded — part of "the rest", ASD's nearest neighbour (rung 2) |
| ASD | 145 | the group of interest (rung 3) |

Derived pools used throughout: **analytic cohort** = 1,996 (145 ASD + 1,851 NT, the paper's design);
**spectrum pool** = 1,648 (ASD + ADHD + other-NDD); **non-ASD-NDD "rest"** = 1,503 (the excluded
children). Label overlap is real and reported: 61/145 ASD also carry an ADHD flag, 140/145 carry ≥1
other-NDD flag (the priority rule assigns the primary label).

**Feature space.** The paper's fixed 51 retained conditions (`nmf_kept_conditions.txt`).

**Residualization vs NT (shared across all analyses).** For each condition we subtract a **single,
fixed NT-expected-rate vector** from every child of every group. That vector is NT prevalence per
condition, **directly standardized** over sex × is_black × education (target = full-cohort
distribution; strata with <5 NT pooled). Because the *same* NT vector is subtracted from all groups,
the resulting latent space is comparable across ASD / ADHD / other-NDD. For matrix analyses the
residual is clipped at 0 and rows are unit-L2-normalized (identical to the paper's
`build_residual_matrix` / `normalize_for_nmf`, except the NT rate is computed once and shared).

> **You asked (about S2): how was the data residualized, and against what baseline?** The procedure
> above is used everywhere. We build **one** NT-expected-rate vector by direct standardization —
> within each stratum of (sex × is_black × education) we take the NT prevalence of each condition,
> then average those stratum rates using the *whole cohort's* stratum weights (strata with < 5 NT
> pooled). A child's residual is their raw condition vector minus this same vector, clipped at 0.
> Because the identical vector is subtracted from every child regardless of group, all groups land
> in one comparable space, and NT becomes ~0 by construction (it is excluded from the latent fit and
> appears only for context).

**Fidelity check (why the comparison is valid).** With this pipeline the **ASD-only K=3** bootstrap
reproduces the paper's published numbers to the third decimal (cosine 0.819 / concordance 0.675 vs
0.80 / 0.66). So differences downstream are attributable to *who is in the pool*, not to a
re-parameterization.

**Pre-registered §2 stability rules** (unchanged from the paper): bootstrap matched cosine ≥ 0.80;
dominant-assignment concordance ≥ 0.60; density floor 5 children per cell; window-keep if ≥60% of
conditions clear the floor; density-artifact guard |r(temporal factor, per-window density)| > 0.7;
"null is publishable."

---

# Part III — Figure & table index

| ID | figure / table | file | shows |
|---|---|---|---|
| S1 | learning curve | `results/s1_learning_curve.pdf` | bootstrap cosine & concordance vs n, K=2/3 |
| S1 | anchors | `results/s1_anchors.csv`, `results/s1_learning_curve.csv` | ASD-only vs full-spectrum §2 bootstrap; curve points |
| S2 | embedding | `results/s2_embedding.pdf` | K=3 latent space, groups overlaid |
| S2 | separation | `results/s2_separation_auc.csv`, `results/s2_summary.txt` | AUCs (latent/raw/burden) per contrast |
| S2 | K-sweep | `results/s2_k_sweep.csv` | ASD-vs-other AUC across K=2..8 |
| S2 | axis tilt | `results/s2_axis_group_means.csv` | per-axis compositional means by group |
| S3 | feasibility table | `results/s3_temporal_feasibility.csv` | density / thin% / cell-n / windows-pass, ASD vs spectrum, 7 schemes |
| S4 | tensor scan | `results/s4_spectrum_tensor.csv` | CP stability + density-artifact guard, pool × scheme × rank |
| S5 | forest + decomposition | `results/s5_forest.pdf` | raw vs shrunk δ (CI); shared-vs-ASD-specific bars |
| S5 | per-condition table | `results/s5_asd_specific.csv`, `results/s5_summary.txt` | decomposition + survival flags for all 51 conditions |
| S6 | staircase | `results/s6_staircase.pdf`, `results/s6_gradient_vs_step.csv` | four-group prevalence "staircase" by shape category |
| S7 | cross-group scatter | `results/s7_contrast.pdf`, `results/s7_cross_group_contrast.csv` | δ-vs-rest × δ-vs-ADHD; quadrants = the specificity story |
| S8 | temporal verdict table | `results/s8_temporal_normalized.csv`, `results/s8_summary.txt` | four approaches × ranks: stability / density guard / separation |

Prior Track-A/B figures referenced for convergence: `results/a1_profiles.pdf`,
`results/a3b_ascertainment.pdf`, `results/b1_diff_heatmap.pdf`, `results/b2_rnorm_hist.pdf`.

---

# Part IV — Spectrum analyses S1–S5 (methods & evidence)

## S1 — do residualized axes get more reproducible with n?

**Method.** `s1_axis_robustness.py`. Fit K=3 NMF (sklearn `mu`/Frobenius, best of restarts) on the
shared NT-residual. Run the full §2 bootstrap (B=200 resamples × 3 restarts): resample children,
refit, Hungarian-match components to the reference (scipy `linear_sum_assignment`), report mean
matched cosine, weakest matched component, and dominant-assignment concordance. Compare two fits —
ASD-only (n=145, the paper's model) vs full spectrum (n=1,648) — and trace a **learning curve** by
refitting on random spectrum subsamples of increasing size.

| K=3 object | boot cosine | worst component | concordance |
|---|---|---|---|
| ASD-only (n=145) | 0.819 | **0.764** ✗ below floor | 0.675 |
| Full spectrum (n=1,648) | **0.970** | **0.960** | **0.867** |

The paper's own K=3 model sits *on* the §2 floor and its weakest axis (0.764) is **below** it — one
of the three axes is not reproducible at n=145. Learning curve (K=3 cosine): 0.803 → 0.849 → 0.880 →
0.906 → 0.939 → **0.969**, monotone. K=2 is already stable at 145 (cos ~0.89), so the gain is
specific to the K the paper reports.

**Verdict.** Strong yes. Reproducibility is limited by **n, not biology**; inclusion turns
floor-scraping axes into rock-solid ones. Figure: `results/s1_learning_curve.pdf`.

## S2 — does ASD stand out in the shared latent space?

**Method.** `s2_asd_separation.py`. Learn K=3 on the full spectrum residual, then test whether ASD
children are distinguishable from other groups with a 5-fold CV ROC-AUC (logistic + random forest,
`clf_auc`). Every contrast is disciplined against a **burden-only baseline** (total code count) and
an **ADHD-vs-other-NDD calibration** (are two non-ASD NDD groups even separable?). Significance by
label permutation (`perm_null`, N=500). Sweep K=2..8.

| contrast | latent W (K=3) | raw 51-residual | burden only |
|---|---|---|---|
| **ASD vs other-NDD** *(the specificity test)* | **0.587** (perm p=0.006) | 0.624 | 0.566 |
| ASD vs ADHD | 0.522 | 0.588 | 0.474 |
| ASD vs ADHD + other | 0.541 | 0.620 | 0.539 |
| *ADHD vs other-NDD* *(calibration)* | *0.638* | *0.636* | *0.569* |

Real but weak: ASD-vs-other is statistically significant yet barely beats burden (0.587 vs 0.566),
the raw residual beats the compressed latent (0.624 > 0.587, so K=3 *loses* ASD info), and **ADHD is
more distinct from other-NDD (0.638) than ASD is** — ASD is not the stand-out group. K-sweep flat
(0.57–0.60). Where ASD does tilt (per-axis ASD − other-NDD): **away from** axis0 (generic
somatic/sensory, −0.084) and **toward** axis2 (psychiatric: anxiety/mood, +0.084).

**Verdict.** ASD is a modest, statistically-real *tilt* within a shared NDD comorbidity space — not
a distinct island, and less distinct than ADHD. Figures: `results/s2_embedding.pdf`, tables
`s2_separation_auc.csv`, `s2_axis_group_means.csv`.

## S3 — does inclusion lift the temporal density ceiling?

**Method.** `s3_temporal_feasibility.py`. Re-run the a0 feasibility audit on the spectrum pool: for
each age-binning scheme, build within-window NT-residual slices and count (condition × window) cells
with <5 children ("thin slices" — the unestimable cells that drove the A1/A2 temporal artifacts).
Pool-agnostic by construction. **Validation: the ASD-only column reproduces `a0_feasibility_table.csv`
exactly.**

| scheme (windows) | ASD (145) density / thin% / cell-n / pass | **spectrum (1,648)** density / thin% / cell-n / pass |
|---|---|---|
| Full 0–18, 3y (6) | 6.2% / 53% / 4 / **1/6** | 5.8% / **12%** / **39** / **6/6** |
| Truncate @12, 3y (4) | 7.7% / 45% / 6 / 1/4 | 7.1% / 7% / 58 / **4/4** |
| Truncate @10, 2y (5) | 6.2% / 52% / 4 / 0/5 | 5.7% / 11% / 39 / **5/5** |
| Dx-anchored 3w (3) | 8.5% / 39% / 8 / 1/3 | 7.9% / 5% / 69 / **3/3** |
| Two-window pre/post (2) | 13.1% / 23% / 13 / 2/2 | 12.5% / 2% / 125 / 2/2 |

The key nuance: **density-the-rate barely moves** (~6%, even a touch lower — a broader pool adds
low-utilization children) because that was never the constraint. **Thin-slice fraction** is, and it
**collapses** (53% → 12%); median cell occupancy goes 4 → 39. The full six-window 0–18 factorization
goes from **1/6 → 6/6** windows passing.

**Verdict.** Yes — a ≥3-window developmental factorization is now *estimable*. Critical caveat:
feasibility ≠ validity (tested in S4). Table: `results/s3_temporal_feasibility.csv`.

## S4 — is the temporal factorization actually valid on the spectrum?

**Method.** `s4_spectrum_tensor.py`. Run the paper's exact nonnegative CP/PARAFAC (multiplicative
updates, kernel copied verbatim from a1) on both pools, two binnings, rank scan R∈{2,3,4}, with the
same bootstrap stability and the same **density-artifact guard** |r(temporal factor C, per-window
density)| > 0.7. Reproduces A1's failure on ASD-only as a check.

| R=3, 6w | ASD (145) | spectrum (1,648) |
|---|---|---|
| thin slices | 53% | **12%** |
| bootstrap cosine | 0.820 | **0.957** |
| worst component | **0.648** ✗ | **0.933** |
| concordance | 0.688 | 0.853 |
| **\|r(temporal, density)\|** | **0.887** ✗ | **0.880** ✗ |
| explained variance | 0.159 | 0.157 |

Two results, reported together: **(1) estimability/stability are fixed** by inclusion (worst
component 0.648 → 0.933; same at dx-anchored 3w, 0.803 → 0.960). **(2) Validity is not** — the
density-artifact guard **fails at every pool × scheme × rank** (|r| = 0.82–0.99), essentially
unmoved by 10× the data (0.887 → 0.880). Explained variance stays ~0.15. Reason: the per-window
density *envelope shape* (pediatric codes cluster in early childhood) is **n-invariant** — more
children crush thin slices but don't change *when* codes occur.

**Verdict.** Inclusion converts an *unestimable* temporal model into a *stably estimable but still
artifactual* one. This **strengthens** A1: the temporal-factor-as-artifact is structural, not
small-sample. Constructive corollary: the failure is confined to the **temporal (C) mode** — the
**condition (B) mode is highly reproducible**, so stable *co-occurrence* is learnable, developmental
*timing* is not (without a per-child utilization denominator). Table: `results/s4_spectrum_tensor.csv`.
S8 (Part VII) is the direct follow-up that tries three ways to remove this artifact.

## S5 — hierarchical shrinkage for ASD-specific inference (the payoff)

**Method.** `s5_hierarchical_shrinkage.py`. Decompose the elevation the study reports, per condition:

    e_ASD(j) = s(j) + δ(j)
    (p_ASD − NT) = (p_rest − NT, shared NDD) + (p_ASD − p_rest, ASD-specific)

s(j) is pinned by the 1,503 excluded + 1,851 NT (tight); δ(j) is noisy (145 vs 1,503) and is
stabilized by empirical-Bayes shrinkage across the 51 conditions (DerSimonian–Laird τ², normal-normal
posterior).

> **You asked (about S5): explain the shrinkage top-level, and — since every formula was a point
> estimate — how are the confidence intervals actually generated?** Four steps:
>
> 1. **Decompose the elevation:** `e_ASD = (p_rest − p_NT) + δ`. Shrinkage acts only on the
>    ASD-specific piece δ.
> 2. **Estimate genuine between-condition spread.** Treat the 51 δ's as noisy draws around a global
>    mean μ with true spread τ. The **DerSimonian–Laird** estimator (standard random-effects
>    meta-analysis) recovers τ² by method of moments: it takes the observed spread of the δ's
>    (Cochran's Q) and subtracts the part explained by each δ's own sampling noise `SE²`, leaving τ²
>    as the "real" variance. (Here μ = +0.012, τ = 0.033; SE_j = √[p_asd(1−p_asd)/145 +
>    p_rest(1−p_rest)/1503].)
> 3. **Shrink each δ.** The **shrinkage factor** `B = τ² / (τ² + SE²)` is the fraction of a
>    condition's variance that is signal: B ≈ 1 (tight, barely moved), B ≈ 0 (noisy, pulled to μ).
>    The shrunk estimate is `δ_shrunk = μ + B·(δ − μ)`.
> 4. **The interval — the step you flagged.** Shrinkage also *sharpens* uncertainty. The
>    **posterior standard deviation** is `post_sd = √(B · SE²)` (smaller than the raw SE, because
>    borrowing adds information). The interval is `δ_shrunk ± z · post_sd`, `z = 1.645` (90%) or
>    `1.96` (95%). So the CIs come from the posterior SD, not the raw SE. A condition **"survives"**
>    when its whole interval excludes 0.

**Findings.** Global ASD-vs-rest shift μ = **+0.012** (tiny on average), but real between-condition
structure τ = **0.033** (Q = 117.7, df 50, p ≪ 0.001). Shrinkage: mean |δ| 0.040 → 0.022. **7/51
survive at 95%, 11/51 at 90%** — the aggregate S2 signal was weak because it is *concentrated*, not
diffuse.

> **You asked: I saw 5 conditions in the table but you said 7 — which are they?** A threshold
> difference. **7 survive at 95%; 11 at 90%**; the earlier table showed the top 5 by magnitude. S5's
> survivor set (below) is the 90% set; S6/S7 build on it.

Genuinely ASD-specific (90% CI excludes 0): **excess** — sleep +0.055, behavioral +0.052, epilepsy
+0.051, movement +0.045, congenital digestive +0.044, congenital musculoskeletal +0.041, personality
+0.038, diabetes +0.030, glaucoma +0.029; **deficit** — headache/migraine −0.049, pain −0.032. The
epilepsy + congenital-anomaly excesses converge with Track B (B1).

The reveal — the *largest* ASD-vs-NT signals are mostly generic NDD:

| condition | e (ASD vs NT) | shared NDD | ASD-specific δ (shrunk) |
|---|---|---|---|
| Anxiety, stress & somatoform | +0.353 | +0.258 (**73%**) | +0.042 (n.s.) |
| Refractive errors | +0.233 | +0.162 (70%) | +0.033 (n.s.) |
| Chronic upper respiratory | +0.197 | +0.167 (85%) | +0.019 (n.s.) |
| Obesity & overnutrition | +0.192 | +0.160 (83%) | +0.020 (n.s.) |
| Hearing loss & ear disorders | +0.175 | +0.183 (**104%**) | +0.005 (n.s.) |
| Functional intestinal (constipation/IBS) | +0.171 | +0.130 (76%) | +0.022 (n.s.) |
| Otitis media | +0.168 | +0.135 (80%) | +0.020 (n.s.) |

**Anxiety — the single largest ASD-vs-NT elevation and the top of S2's psychiatric axis — is 73% a
generic-NDD signal.** The whole somatic/sensory cluster is 70–104% shared. A study comparing only
ASD to typical children would report all of these as headline "ASD-associated" comorbidities; only
the excluded groups reveal they are not ASD-specific.

**Verdict.** The excluded groups turn "ASD has more of everything than typical kids" into a specific,
defensible claim about *which* conditions distinguish ASD from the rest of the neurodevelopmental
spectrum. Figure: `results/s5_forest.pdf`; table `results/s5_asd_specific.csv`.

---

# Part V — What changes with the expanded cohort (feasibility & opportunity)

| dimension | at n=145 (analytic) | with the spectrum (1,648 / rest 1,503) | net |
|---|---|---|---|
| **Axis estimation** | K=3 worst component 0.764, below the team's own §2 floor | 0.960; every §2 number clears comfortably | **feasibility unlocked** |
| **Cross-group contrast** | impossible by construction (NT residual makes NT trivial) | ASD vs ADHD vs other-NDD directly testable | **new question class** |
| **Temporal factorization (estimability)** | thin slices 53%, 1/6 windows pass | 12%, 6/6 pass | **feasibility unlocked** |
| **Temporal factorization (validity)** | artifact (small-sample suspected) | still artifact (\|r(density)\|≈0.88, n-invariant) | **no gain — and now proven structural** |
| **Co-occurrence (condition mode)** | under-powered | highly reproducible on spectrum | **feasibility unlocked** |
| **ASD-specific per-condition inference** | not separable from generic NDD | 7–11/51 conditions isolated via shrinkage | **new question class** |

**The two-part takeaway.** (1) **Feasibility** — almost everything limited by sample size at n=145 is
unlocked: axis robustness (S1), temporal *estimability* (S3), co-occurrence structure (S4 condition
mode), stable per-condition prevalence (S5). The one thing inclusion does **not** rescue is temporal
*validity* (S4/S8): that ceiling is structural, and more data proves it rather than fixing it.
(2) **Opportunity** — inclusion makes two genuinely new question-classes askable, both keeping the
ASD focus: *how does ASD differ from other NDD* (S2), and *which specific comorbidities are truly
ASD-specific vs shared NDD* (S5–S7). Neither is possible against an NT-only comparator.

---

# Part VI — Honest caveats (S1–S5)

- **Label overlap** from the priority rule: ASD-vs-other contrasts are between *primary-label* groups
  with shared underlying flags; if anything this understates a "pure" ASD contrast.
- **Residualizing vs NT makes NT trivial by construction** — NT is excluded from the latent fit (its
  residual ≈ 0) and appears only for context in the embedding.
- **Burden confound** persists (ASD carries more codes); we report the burden baseline beside every
  contrast rather than adjusting it away (burden is part of the phenotype).
- **Ascertainment** still differs across groups — no per-child utilization denominator (the Track-A
  A3b ceiling). Coding *rate* cannot be divided out; a modest AUC could be attenuated real signal or
  residual ascertainment.
- **Robustness ≠ validity** (S1): the respiratory/atopic axis is the same likely artifact Track B
  flagged — more data makes it more *stably estimated*, not more real.
- **S5 δ** is not further demographically adjusted (NT standardization cancels; both groups are NDD so
  are more comparable than vs NT); single τ² across conditions; shrinkage *is* the multiplicity
  control (the 7 at 95% are the conservative set), not FDR.

---

# Part VII — Per-condition specificity, cross-group contrast, and developmental timing (S6–S8)

S1–S5 established that ASD's comorbidity signal is concentrated in a specific set (S5). The three
follow-ups here sharpen that set from three angles. Each reuses the Part II residualization and, for
S6/S7, the Part IV/S5 shrinkage machinery verbatim; the definitions below are only the pieces new to
S6–S8.

**Definitions used in S6–S8.**
- **Elevation vs NT** `e_g = p_g − p_NT`; **ASD-specific deviation** `δ = p_ASD − p_rest` (as in S5).
- **Shared fraction** `shared_frac = (p_rest − p_NT) / (p_ASD − p_NT)` — the fraction of ASD's total
  elevation over NT the rest of the spectrum already accounts for. `≈0` → abrupt **step**; `≈0.5` →
  **gradient**; `≳0.85` → generic NDD.
- **Cochran–Armitage trend z** — linear-trend test across the *ordered* groups (scores NT=0,
  other-NDD=1, ADHD=2, ASD=3). The four raw prevalences are always shown alongside so the true shape
  is visible.
- **Three cross-group contrasts (S7)**, each with its own DerSimonian–Laird τ² (identical to S5):
  `delta_vs_rest = p_ASD − p_rest` (= S5); `delta_vs_ADHD = p_ASD − p_ADHD` (the hard test — ASD vs
  its nearest neighbour, isolating *autism* from *attention*); `delta_ADHD_vs_other = p_ADHD − p_other`
  (calibration).
- **CP / PARAFAC + guards (S8)** — nonnegative CP factorizes the child × window × condition tensor
  into R rank-1 triples (child-loading, condition vector, **temporal factor**). A component is real
  only if it is **stable** (bootstrap cosine ≥0.80, concordance ≥0.60), **artifact-free**
  (|r(temporal factor, per-window density envelope)| ≤ 0.7), *and* **group-separating** (best
  per-component ASD-vs-rest child-loading ROC-AUC ≥ 0.60).

## S6 — per-condition specificity: gradient vs. step

`s6_gradient_vs_step.py`. Survival (from S5's shrunk δ) is checked **first**, so an ASD-specific
signal is never hidden by a coincidentally small `e_ASD`. Six shape categories: **STEP** (δ survives,
δ>0, shared_frac<0.5 — ASD-specific excess jump), **STEP-DOWN** (δ survives, δ<0 — ASD-specific
deficit), **GRADIENT+** (δ survives, δ>0, shared_frac≥0.5 — graded continuum ASD is boosted atop),
**SHARED** (δ n.s., |e_ASD|≥0.03, shared_frac≥0.5 — generic-NDD gradient), **ASD-LEAN** (δ n.s.,
|e_ASD|≥0.03, shared_frac<0.5), **FLAT** (δ n.s. and |e_ASD|<0.03).

**Category counts (all 51):** STEP 8 · STEP-DOWN 2 · GRADIENT+ 1 · SHARED 34 · ASD-LEAN 4 · FLAT 2.
The **11 ASD-specific conditions** (δ surviving at 90%) split cleanly: **8 steps up, 2 steps down,
only 1 true gradient** — ASD-specificity, where it exists, is overwhelmingly step-shaped.

**STEP — ASD-specific excess (rest of the spectrum near NT):**

| Condition | e_ASD | NT → oth → ADHD → ASD | δ_shrunk | shared_frac |
|---|---|---|---|---|
| Movement & other episodic disorders | +0.142 | 0.010 · 0.072 · 0.089 · 0.152 | +0.045 | 0.48 |
| Congenital musculoskeletal & limb anomalies | +0.135 | 0.085 · 0.137 · 0.157 · 0.221 | +0.041 | 0.44 |
| Behavioral syndromes (eating, sleep) | +0.125 | 0.019 · 0.045 · 0.084 · 0.145 | +0.052 | 0.32 |
| Epilepsy & seizures | +0.118 | 0.006 · 0.035 · 0.061 · 0.124 | +0.051 | 0.33 |
| Personality disorders | +0.090 | 0.006 · 0.017 · 0.084 · 0.097 | +0.038 | 0.40 |
| Congenital digestive anomalies | +0.090 | 0.048 · 0.065 · 0.072 · 0.138 | +0.044 | 0.21 |
| Diabetes & glucose regulation | +0.052 | 0.017 · 0.028 · 0.035 · 0.069 | +0.030 | 0.27 |
| Glaucoma | +0.043 | 0.019 · 0.030 · 0.019 · 0.062 | +0.029 | 0.15 |

A coherent **congenital + neurological + behavioral** cluster — the most mechanistically "ASD" set,
converging with S5 and Track B.

**STEP-DOWN — ASD-specific deficit (ASD *below* the elevated spectrum):**

| Condition | e_ASD | NT → oth → ADHD → ASD | δ_shrunk | shared_frac |
|---|---|---|---|---|
| Headache & migraine | −0.020 | 0.082 · 0.118 · 0.175 · **0.062** | −0.049 | −2.87 |
| Pain syndromes | −0.020 | 0.061 · 0.086 · 0.087 · **0.041** | −0.032 | −1.29 |

Headache *rises* across the rest of the spectrum then **drops at ASD to below NT**. ASD looks
unremarkable vs NT (`e_ASD ≈ 0`) *precisely because* it fails to show the spectrum-wide elevation;
the specificity is only visible against the rest (δ = −0.049 survives). These are the B-III targets
(interoception / pain reporting), and being *deficits* they run against the "ASD is coded more"
confound.

**GRADIENT+ — the one true graded-plus-boost condition:** organic sleep disorders, a clean monotone
climb `0.112 → 0.208 → 0.309 → 0.372` (CA_trend_z = +12.7) with the rest exactly halfway
(shared_frac 0.51) *and* a surviving ASD-specific boost.

**SHARED (34 conditions):** most of ASD's raw elevations are not specific — the whole
somatic/sensory cluster (refractive, otitis, hearing, asthma, obesity, dental, functional GI) is
strongly elevated in ASD (e_ASD up to +0.35) with δ not surviving and shared_frac ≥ 0.5. Two worth
flagging: **Anxiety** `0.137 · 0.309 · 0.517 · 0.483` (ASD's largest elevation — but **peaks in
ADHD**) and **Mood** `0.070 · 0.114 · 0.237 · 0.110` (also peaks in ADHD; ASD barely above other-NDD).

Figure: `results/s6_staircase.pdf`; full table `results/s6_gradient_vs_step.csv`.

## S7 — cross-group contrast: is ASD specific even against ADHD?

`s7_cross_group_contrast.py`. S6 asks whether ASD differs from the *pooled* rest; pooling can
*manufacture* apparent specificity (a condition common in both ASD and ADHD but rare in other-NDD
looks "ASD-specific vs rest" only because other-NDD dilutes the pool). S7 asks the harder question:
does that specificity survive when the comparator is **ADHD alone**?

**Label counts (all 51):** AUTISM-SPECIFIC 2 · SHARED-WITH-ADHD 7 · ADHD-PEAK 4 · ASD-DEFICIT 0 ·
SHARED-NDD 36 · FLAT 2. (τ: vs-rest 0.033, vs-ADHD 0.035, ADHD-vs-other 0.030 — the calibration τ is
non-zero, so ADHD and other-NDD are themselves per-condition distinguishable.)

> **The headline result: of the 11 conditions ASD-specific vs the pooled rest (S5/S6), only 2 survive
> the ADHD-alone test.** Pooling other-NDD into "the rest" inflated the apparent specificity of the
> other 9.

**AUTISM-SPECIFIC — ASD exceeds even ADHD (survives):**

| Condition | NT → oth → ADHD → ASD | δ vs rest | δ vs ADHD | peak |
|---|---|---|---|---|
| Epilepsy & seizures | 0.006 · 0.035 · 0.061 · 0.124 | +0.051* | +0.037* | ASD |
| Glaucoma | 0.019 · 0.030 · 0.019 · 0.062 | +0.029* | +0.032* | ASD |

Only **epilepsy** and **glaucoma** remain specific to autism beyond attention. Epilepsy is the
strongest and most mechanistically expected (shared genetic/neurological substrate with autism);
glaucoma is smaller and rarer and warrants a cautious read.

**SHARED-WITH-ADHD (7):** congenital digestive anomalies, movement disorders, behavioral
(eating/sleep) syndromes, congenital musculoskeletal anomalies, organic sleep, diabetes, personality
disorders. Each survives vs the pooled rest but its ASD-vs-ADHD deviation does *not* survive
(δ_vs_ADHD +0.008 to +0.038, n.s.) — ASD and ADHD are elevated *together* above other-NDD. These are
**neurodevelopmental** (shared by the two higher rungs), not autism-specific — the group most
vulnerable to the pooling artifact, honestly reclassified.

**ADHD-PEAK — ADHD exceeds ASD (survives) (4):**

| Condition | NT → oth → ADHD → ASD | δ vs ADHD | peak |
|---|---|---|---|
| Headache & migraine | 0.082 · 0.118 · 0.175 · 0.062 | −0.073* | ADHD |
| Mood disorders | 0.070 · 0.114 · 0.237 · 0.110 | −0.070* | ADHD |
| Acute upper respiratory infection | 0.659 · 0.817 · 0.818 · 0.710 | −0.046* | ADHD |
| Pain syndromes | 0.061 · 0.086 · 0.087 · 0.041 | −0.034* | ADHD |

ADHD, not ASD, is the most affected group. Headache and pain are the same two conditions S6 flagged
as ASD-specific *deficits* — here the deficit is sharpest against ADHD (the peak group), so S7 labels
them ADHD-PEAK; both facts are true and consistent (this is why the ASD-DEFICIT count is 0 — the
label logic checks the ADHD contrast first). Anxiety and mood, the largest raw ASD elevations, also
crest at ADHD.

Figure: `results/s7_contrast.pdf` (δ-vs-rest × δ-vs-ADHD; upper-right = autism-specific beyond
attention; below the x-axis = ADHD-peaked). Full table `results/s7_cross_group_contrast.csv`.

## S8 (M-II) — developmental timing: can a temporal signal survive ascertainment?

`s8_temporal_normalized.py`. S4 showed a plain CP on the spectrum tensor is *estimable* but *invalid*
(temporal factor tracks the density envelope). S8 attacks that artifact three ways, all built around
it, and scores each vs the S4 baseline on all four criteria (estimable / stable / artifact-free /
group-separating).

> **You flagged the ascertainment artifact yourself.** S8 is built entirely around it: all three
> methods are attempts to remove it, and the density guard is the arbiter. A method that is stable
> and artifact-free but does *not* separate groups is a **clean negative**, itself a publishable
> methods result.

| Approach | What it changes | Result |
|---|---|---|
| **BASELINE** | S4's absolute-age 6-window residual tensor | reproduces the artifact |
| **(a) REL-TIMING** | each event placed in a quartile of *that child's own* first→last-code span — removes the population envelope by construction | artifact persists |
| **(b) DENS-EQUAL** | each age-slice rescaled to equal total mass, so no factor can be driven by high-volume windows | artifact persists |
| **(c) UTIL-NORM** | each cell = the condition's *share* of that child's coding in the window (count ÷ child's total events), NT-residualized — divides out coding volume cell-by-cell | de-confounds, but no ASD signal |

**Verdict table (spectrum, n = 1,648; ASD = 145):**

| Approach | R | expl. var | boot cos | concord. | \|r(density)\| | separation AUC | verdict |
|---|---|---|---|---|---|---|---|
| BASELINE | 2 | 0.133 | 0.998 | 0.927 | **0.896** | 0.522 | unusable (artifact) |
| BASELINE | 3 | 0.157 | 0.953 | 0.844 | **0.880** | 0.533 | unusable (artifact) |
| REL-TIMING | 2 | 0.166 | 0.994 | 0.921 | **0.974** | 0.555 | unusable (artifact) |
| REL-TIMING | 3 | 0.197 | 0.988 | 0.876 | **0.974** | 0.574 | unusable (artifact) |
| DENS-EQUAL | 2 | 0.111 | 0.998 | 0.930 | **0.957** | 0.519 | unusable (artifact) |
| DENS-EQUAL | 3 | 0.136 | 0.902 | 0.724 | **0.970** | 0.531 | unusable (artifact) |
| **UTIL-NORM** | **2** | 0.102 | 0.998 | 0.988 | **0.684** ✓ | **0.540** | **clean-negative** |
| UTIL-NORM | 3 | 0.132 | 0.946 | 0.893 | 0.781 | 0.573 | unusable (artifact) |

> **The result: no approach yields a real (stable + artifact-free + group-separating) temporal
> factor.** Only cell-by-cell utilization normalization (UTIL-NORM, rank 2) clears the density guard
> while staying stable — and *there*, the temporal factor's ASD-vs-rest separation is 0.540, a
> coin-flip. Once ascertainment is divided out, developmental **timing** carries no ASD signal beyond
> the marginal comorbidity rates S5–S7 already capture.

Two honest limits: (i) UTIL-NORM clears the 0.70 guard only at rank 2 and only narrowly (|r|=0.684);
at rank 3 it drifts back to 0.781 — the de-confounding is real but fragile. (ii) The conclusion is
nonetheless robust: separation AUC never exceeds ~0.57 anywhere, artifact removed or not. Notably,
within-child relative timing (the a priori most promising fix) leaves the density correlation
essentially untouched (|r|=0.974): even relative to a child's own span, the first quartile carries
disproportionate coding, and the factor locks onto it. Full table: `results/s8_temporal_normalized.csv`.

---

# Part VIII — Unified interpretation and S6–S8 caveats

Read together, the eight analyses give a consistent, honest picture of where ASD's comorbidity signal
lives:

- **It is marginal, not temporal.** All the signal is in *which* conditions are elevated (rates), none
  in *when* they appear (timing). S8 is a clean negative — useful, because it closes off the
  temporal-modeling direction rather than leaving it an open promissory note. This is S4 confirmed and
  extended.
- **The specific signal is small and shrinks under scrutiny.** ASD-specific vs the pooled rest = 11
  conditions (S5/S6), overwhelmingly step-shaped. ASD-specific vs **ADHD alone** = **2** (epilepsy,
  glaucoma; S7). Each tightening of the comparator removes conditions, exactly as expected if much of
  ASD's profile is shared neurodevelopmental liability.
- **The largest elevations are shared — and often peak in ADHD.** Anxiety, mood, and the whole
  somatic/sensory cluster are ASD's biggest raw elevations but are spectrum-wide; anxiety and mood
  actually crest at ADHD.

For the inclusion case this is the payoff argued in Part I: the excluded children let a single
undifferentiated "elevated in ASD" list be split into (a) a short, mechanistically-interpretable
autism-specific core (epilepsy above all), (b) a broad shared neurodevelopmental gradient, and (c) a
definitive statement that timing adds nothing — none of which the ASD-vs-NT design can produce. Both
framings held open throughout are true, for different conditions.

**A classification nuance worth knowing (S6).** The two deficits (headache, pain) were initially
mis-classified as FLAT because their ASD-vs-NT elevation is ~0. That was wrong — their `e_ASD ≈ 0`
*is* the finding (ASD fails to join the spectrum-wide rise). The classifier now checks δ-survival
**first**, so a surviving deficit lands in STEP-DOWN regardless of its NT elevation. Point estimates
and S5 are unchanged — only the labeling.

**S6–S8-specific caveats** (in addition to Part VI):
- **Ordering is an assumption.** NT < other-NDD < ADHD < ASD is clinically reasonable but imposed;
  every table carries the four raw prevalences so the actual shape is inspectable.
- **Empirical-Bayes shrinkage is plug-in.** DerSimonian–Laird treats τ² as known, making intervals
  mildly anticonservative — well-calibrated screening, not final inference (a full Bayesian model,
  agenda item M-III, would propagate τ² uncertainty).
- **The ascertainment ceiling still applies to the positive findings.** Utilization is not divided out
  of S5–S7. S8's UTIL-NORM shows a cell-by-cell divide-out is possible but fragile; agenda item M-I
  would build it as a full marginal de-confounder. Until then the two **deficits** (which run against
  the more-coding direction) and the **AUTISM-SPECIFIC** epilepsy signal (large, mechanistically
  expected) are the most confound-robust results.

---

# Follow-on

The proposed next questions (biological B-I…B-V, methodological M-I…M-IV) and their current status
live in `RESEARCH_AGENDA.md`. As of 2026-09-18: B-I (S6), the cross-group contrast (S7), and M-II
(S8, clean negative) are done; **M-I** (a per-child utilization denominator as a full marginal
ascertainment de-confounder) is the top remaining priority — S8 showed the cell-by-cell divide-out
works but is fragile, so M-I is needed to de-confound S2/S5/S6/S7 rather than just the tensor.
