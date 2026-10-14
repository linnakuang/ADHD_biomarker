# ADHD-focused analyses (`adhd_analysis/`)

## Start here (for collaborators)

- **Results, with full methods, formulas and figures:** [`RESULTS.md`](RESULTS.md), or the self-contained `RESULTS.html` (figures embedded; equations need internet for MathJax).
- **Sample, filtering and ICD grouping:** RESULTS.md §1.1–1.4 and Appendix A (ICD-9/10 definition of every condition).
- **Open issues and pending checks:** RESULTS.md §9 *Caveats* and §10 *Next*; condition-code audit in §8.1.
- **Data policy:** every file in `results/` is aggregate (per-condition or per-group). Per-child caches live in `_cache/`, which is git-ignored and must never be pushed.

### Layout

| Path | What |
|---|---|
| `config.py` | Input paths (absolute, for the `/dcs07` cluster) and pre-registered floors |
| `adhd_common.py` | Loader (caches to `_cache/base.pkl`), group ids, condition sets, prevalence, standardization, EB shrinkage (`shrink`, `shrink_contrast`), utilization proxy |
| `d1_adhd_specificity.py` | D1 and D2: ADHD-specific vs shared NDD; step vs gradient |
| `d3_profiles.py` | D3: NMF profile stability, shared-space separation, NT-projection survival |
| `d4_presentation.py` | D4: inattentive vs hyperactive/combined |
| `d5_prepost_dx.py` | D5: pre/post-diagnosis DiD with pseudo-anchored guards |
| `d6_dual_diagnosis.py` | D6: ADHD with and without autism (additivity) |
| `d7_utilization.py` | D7: utilization-adjusted logistic models |
| `make_condition_table.py` | Appendix A table: ICD definitions + group prevalences of the 78 conditions |
| `make_figures.py` | `results/fig_*.png` from aggregate CSVs only (`python make_figures.py fig_profiles` redraws one) |
| `md_to_html.py` | RESULTS.md → self-contained RESULTS.html |
| `run_all.sh` | Runs D1, then D4–D7 in parallel (D3 separately) |

### Rerunning elsewhere

- **Edit `config.py` paths** to point at your copies of `BBC_event_table_full.csv`, `BBC_patients_analytic_full.csv`, the NDD label CSV and (for D4) `kid_icdcode_data_Mar2026.RDS`.
- **Shared code dependencies:** `adhd_common.py` imports `gus_common`, `s1_axis_robustness`, `s5_hierarchical_shrinkage` (and D3 imports `s2_asd_separation`) from `config.GUS_DIR`. `gus_common` in turn needs `BBC_Dataset/clustering/clustering_utils.py`. On the `gus` branch these live under `shared_infra/` and the S-folders, so point `GUS_DIR` / `sys.path` there.
- **Python:** pandas, numpy, scipy, scikit-learn, matplotlib, pyreadr (D4), markdown (HTML). The first `load()` builds `_cache/base.pkl`; later runs are fast.

---

> **Status 2026-10-14: D1–D7 all run. Results with full methods + figures: [`RESULTS.md`](RESULTS.md) / `RESULTS.html` (self-contained).** Rerun: `./run_all.sh` (+ `d3_profiles.py`), then `make_figures.py` and `md_to_html.py`.

Started 2026-10-13. Goal: **ADHD-specific comorbidity signal** in the Boston Birth Cohort,
reusing the methods developed for ASD (`gus_analysis/`: Track A temporal, Track B contrastive,
S1–S8 spectrum) with ADHD as the focal group.

## Original plan and context (2026-10-13)


### `main` branch of `linnakuang/ADHD_biomarker` (Linna's ADHD-vs-NT pipeline)
- **Cohort:** ADHD vs NT (1,851). The diary says "ADHD only: 573", but
  `code/preprocessing/build_patients.py` sets `label[adhd] = "ADHD"` without excluding ASD, so the
  code actually builds **any-ADHD = 634** (573 + 61 ASD+ADHD). Needs reconciling with Linna.
- **Conditions:** 132 granular conditions (`codebook/sTable2.xlsx`), not the paper's 51.
- **Done:** Table 1 + ADHD presentation from ICD (inattentive / hyp-combined / unspecified;
  `table1.py`, `subtype_by_category.py`); F1 KM time-to-first-condition and burden (time to k-th
  domain vs NT percentiles; `f1.py`, `burden.py`); developmental heatmap (`dev_map.py`);
  K=3 NMF → **MSK–injury / psychiatric–internalizing / atopic-ENT–infectious**
  (`nmf_analysis.py`); contrastive NT-basis residual (F5, `contrastive.py`, the same idea as our
  B2); utilization sensitivity (≥2-month persistence, ≥3y follow-up, drop acute codes;
  `sen/utilization_sensitivity.py`); prenatal panel.
- **To-do listed in the diary (7/20):** ADHD-only vs ASD+ADHD table, NMF with any-ADHD,
  developmental figure (any ADHD).
- **Not done:** any comparison against other-NDD or ASD. Every ADHD-vs-NT signal there is
  "ADHD-associated," not "ADHD-specific."

### `gus` branch (our full-spectrum work, S1–S8)
ADHD shows up only as a comparator, but there are already ADHD findings to build on:
- **S2:** ADHD-vs-other-NDD AUC **0.638**, higher than ASD-vs-other (0.587). ADHD is the
  *more* distinct group in the shared latent space.
- **S6/S7:** **headache/migraine, mood, acute URI and pain peak in ADHD** (δ ADHD−ASD −0.07 for
  headache and mood). Anxiety and mood peak in ADHD, not ASD.
- **S7:** an `ADHD−other` contrast was computed only as a calibration (`delta_ADHD_vs_other`,
  `surv_calib90` in `s7_cross_group_contrast.csv`). It hasn't been read as a result yet.

## Cohort (verified 2026-10-13 from `*_full.csv` + label CSV)

| Group (priority ASD > ADHD > other-NDD > NT) | n |
|---|---|
| NT | 1,851 |
| Other NDD | 930 |
| **ADHD (no ASD)**: primary | **573** |
| ASD (61 of them also ADHD) | 145 |

ADHD first-dx age (n=634 any-ADHD): **median 7.1 y** (IQR 5.4–9.3), compared with 3.25 y for ASD.
That leaves about 4 extra years of pre-dx record per child, which matters for the temporal questions.
The label CSV also has `ADHD_diag_n_of_visit` (diagnostic certainty / utilization proxy).

## Proposed analyses (ADHD-flavored S-series → "D-series")

Ordered by value ÷ cost. Same §2 floors, same stratum-standardized NT baseline as `gus_analysis`.

| ID | Question | Method reused | Notes |
|---|---|---|---|
| **D1** | Which conditions are **ADHD-specific** vs generic NDD? | S5 shrinkage, ADHD vs (ASD+other) and ADHD vs other-NDD | Cheapest, highest yield. Mirror of the S5 "reveal" |
| **D2** | Gradient vs step with ADHD as the focal rung | S6/S7, 3 contrasts | Is ADHD's psychiatric/pain excess a step or a gradient? |
| **D3** | Do Linna's 3 NMF profiles survive NT-residualization **and** an other-NDD contrast? | B2 + S2 | Bridges her main-branch results to ours |
| **D4** | ADHD presentation subtypes (inattentive vs hyp/comb) as an internal contrast | S5 shrinkage within ADHD | Uses her ICD subtype classifier, ASD-free comparison |
| **D5** | Pre- vs post-ADHD-dx reshape, with NT age-matched guard | A3 + A3b DiD | Later dx (7.1 y) gives denser pre-dx window; guard is essential |
| **D6** | ADHD-only vs ASD+ADHD (61) | S7-style contrast | On Linna's to-do list. Small n, so shrinkage needed |
| **D7** | Headache/pain excess in ADHD: robust to utilization? | M-I denominator (events/child) | Opposite sign to ASD's deficit, so a good paired story (B-III) |

Ceilings inherited from the ASD work: there's no per-child visit denominator (A3b), and temporal CP
tracks the density envelope at any n (S4/S8). We won't sell trajectory factorizations.

## Infra
- `config.py`: real paths, `*_full` tables, same §2 floors.
- Reuse `gus_analysis/gus_common.py` and the S-script kernels via `sys.path` (see `GUS_DIR`).
- Outputs go in `results/` (aggregate only; nothing per-child leaves this box).
