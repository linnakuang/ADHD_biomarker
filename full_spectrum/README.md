# Full-spectrum NDD comorbidity analyses, S1–S8 (`full_spectrum/`)

Postnatal-EHR comorbidity analyses on the Boston Birth Cohort, contributed by Gus Fridell.
This was the first body of work on the `gus` branch. It has been **moved and consolidated into this folder**; the
front-page README now describes the ADHD-focused follow-up in [`../adhd_specific/`](../adhd_specific/). The
files are unchanged apart from the move, and the paths below are relative to this folder.

> ## ⚠️ Read this first: what cohort these analyses actually use
>
> **None of the work in this folder is ADHD-only.** Despite the repository name, every
> analysis here operates on the **full neurodevelopmental spectrum**, using a single
> mutually-exclusive 4-way label assigned by the priority rule **ASD > ADHD > other-NDD > NT**.
> **ADHD (n = 573, meaning *ADHD-without-ASD*)** appears only as **one of four groups** or as a
> **comparator** — never as the sole study population. The ADHD-centred reanalysis is new work
> built on top of this (`../adhd_specific/`); these files are the full-spectrum baseline it started from.

## Cohort legend

| Group (priority-labeled, mutually exclusive) | n | Role |
|---|---|---|
| **NT** (neurotypical) | 1,851 | Baseline / expected-rate reference |
| **Other NDD** (non-ADHD, non-ASD) | 930 | Spectrum rung 1 |
| **ADHD** (ADHD *without* ASD) | 573 | Spectrum rung 2 — ASD-comorbid ADHD children are pulled up to ASD |
| **ASD** | 145 | Spectrum rung 3 |
| **Total labeled** | **3,499** | |

Derived pools: **analytic cohort** = ASD + NT = 1,996 (the original paper's design) · **spectrum
pool** = ASD + ADHD + other-NDD = 1,648 · **"the rest"** = ADHD + other-NDD = 1,503.

## Which individuals each analysis uses

| Analysis | Individuals used |
|---|---|
| **S1** axis robustness | ASD-only (145) **vs** full spectrum (1,648) |
| **S2** ASD separation in NMF space | spectrum (1,648); NT (1,851) as residual baseline only |
| **S3** temporal feasibility | ASD-only (145) vs spectrum (1,648) |
| **S4** spectrum tensor | ASD-only vs spectrum pools |
| **S5** hierarchical shrinkage | ASD (145) vs rest (1,503); NT (1,851) baseline |
| **S6** gradient-vs-step | all four groups (NT 1,851 / other-NDD 930 / ADHD 573 / ASD 145) |
| **S7** cross-group contrast | all four groups (ASD vs rest, **ASD vs ADHD**, ADHD vs other) |
| **S8** temporal normalized | spectrum (1,648), ASD vs rest |

## Directory map

```
shared_infra/            config + shared bootstrap/helpers used by every S-script
power_axes_S1_S2/        S1 axis robustness, S2 ASD detectability in NMF latent space
temporal_S3_S4_S8/       S3 feasibility, S4 tensor validity, S8 ascertainment-normalized timing
specificity_S5_S6_S7/    S5 shrinkage, S6 gradient-vs-step, S7 cross-group contrast
writeups/                SPECTRUM.md (full methods+results), RESEARCH_AGENDA.md, analysis_plan/memo
```
Each analysis directory has its own `README.md` stating the exact groups it uses and its finding.
The complete, self-contained write-up is **`writeups/SPECTRUM.md`**.

## Results in one line each

- **S1 — power, not biology, was the binding constraint.** The ASD-only K=3 model sits below its own
  stability floor (worst component 0.76); on the full spectrum it is 0.96.
- **S2 — ASD is a modest graded tilt, not an island.** ASD-vs-other-NDD AUC 0.587 (p = 0.006), barely
  above a burden-only baseline, and *less* distinct than ADHD-vs-other (0.638).
- **S3 — inclusion makes a temporal factorization *estimable*.** Thin slices 53% → 12%; windows
  passing 1/6 → 6/6.
- **S4 — but not *valid*.** The temporal factor still tracks the per-window density envelope
  (|r| ≈ 0.88), essentially unmoved by 10× data — estimable ≠ valid.
- **S5 — the payoff: 7–11/51 conditions are genuinely ASD-specific**, while ASD's *largest* raw
  elevations (anxiety, refractive, otitis, obesity) are 70–104% generic NDD.
- **S6 — ASD-specificity is step-shaped.** Of the 11 ASD-specific conditions, 8 step up, 2 step down,
  1 is a true gradient; the largest elevations are shared spectrum-wide.
- **S7 — against ADHD alone, only 2 survive** (epilepsy, glaucoma); pooling had inflated the apparent
  specificity of 9. Headache, mood, acute URI and pain actually peak in ADHD.
- **S8 — a clean negative.** No ascertainment correction yields a temporal factor that is stable,
  artifact-free *and* group-separating; once coding volume is divided out, timing adds no ASD signal.

## Conventions, data policy, and reproducibility

- **No individual-level data is included, by design.** Every file here is aggregate: result tables are
  per-condition (51 rows) or per-configuration, and figures are condition/n-level (no per-child
  points). Raw EHR, demographics, and any per-child outputs are excluded.
- **Paths are obfuscated.** The real compute-node location has been replaced throughout by the
  placeholder **`<PROJECT_ROOT>`**; only the directory structure below that level is shown.
- **Archival, not runnable standalone.** The scripts were authored in one flat working directory and
  cross-import by module name (e.g. `import s1_axis_robustness as s1`, `import gus_common as gc`).
  Running them requires (a) placing all `.py` on one `PYTHONPATH`, (b) the upstream BBC preprocessing
  / clustering pipeline that provides `clustering_utils` (`load_event_data`, `load_patients`,
  `build_residual_matrix`, `select_nmf_conditions`), and (c) the restricted source data — none of
  which are in this repository. The thematic directory grouping is for reading, not execution.
