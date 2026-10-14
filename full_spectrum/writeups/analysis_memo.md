# ASD Comorbidity Study — Analysis & Data Memo

**Scope.** A methods-style inventory of the *current* code and data state in
`<PROJECT_ROOT>` (the revision working copy), independent of the
June manuscript draft. It catalogs (1) the composition of the data, (2) how the
modeling matrix is built and how sparse it is, (3) every analysis that has been
run, and (4) every method that has been tried. All counts below were recomputed
directly from the data files (the study's own scripts hardcode Windows paths and
do not run as-is here); figures were derived with pandas/numpy against the
`preprocessing/results/` and `NDD datasets` files.

*Prepared 2026-08-28.*

[TOC]

---

## 1. Data composition

### 1.1 Cohort structure and ASD ascertainment

The Boston Birth Cohort postnatal follow-up has EHR pulls for **3,500 children**.
One child (ID 3490, a questionable ASD case with a missing NDD value) is dropped,
leaving **3,499** with a definite label.

**Two variables must be kept separate here.** (i) The **category** (`ndd_status` /
`label`) is a single-label partition — every child sits in **exactly one** category,
so the table below is mutually exclusive and no child is double-counted. (ii) The
**underlying diagnosis flags** (the binary `ASD`, `ADHD`, `ID`, … columns) are
**not** mutually exclusive — one child can carry several. The category is derived
from the flags by an **ASD > ADHD > other-NDD > neurotypical priority rule**: a child
positive for *both* ASD and ADHD is assigned to ASD and counted only there.

| Category (single-label) | n | Definition |
|---|---:|---|
| Neurotypical (NT) | 1,851 | No ASD and no other recorded NDD |
| ASD | 145 | Chart-confirmed ASD — **wins over any co-occurring NDD flag** |
| ADHD (no ASD) | 573 | ADHD flag present, ASD absent |
| Other DSM-5 NDD (no ASD/ADHD) | 930 | Conduct, ID, communication, learning, motor, tic, or misc. DD |
| **Total** | **3,499** | |

*Worked example of the two-variable distinction:* **634** children carry the ADHD
diagnosis flag, but **61** of them are also ASD-positive. Those 61 are counted in
the ASD row (145), so the ADHD **category** row is the remaining **634 − 61 = 573**.
The flag "ADHD" (634) is therefore not the same as the category "ADHD, no ASD"
(573). This is also why 140 of the 145 ASD children carry ≥ 1 additional NDD flag
(§1.3) despite ASD being a single, exclusive *category*.

**ASD ascertainment.** 198 children were flagged with any ASD ICD code; chart
review by a developmental pediatrician confirmed 139, and 6 more were added (5
from the BMC ASD registry, 1 from a prior EHR pull), for **145 confirmed ASD
cases**. This is the number that flows through every ASD-side analysis.

### 1.2 Sample attrition: full (3,499) → analytic (1,996)

The analytic cohort (`BBC_patients_analytic.csv`) is **1,996 children = 145 ASD +
1,851 NT**. The move from 3,499 to 1,996 drops **exactly 1,503 children — the
entire NDD-but-not-ASD population** (573 ADHD-without-ASD + 930 other-NDD):

- The exclusion is **by design**, not data quality. **Every ASD (145/145) and
  every NT (1,851/1,851) is retained** — none lost to missing covariates, missing
  events, or short records.
- It happens at label assignment in `preprocessing.py`: only ASD cases and pure-NT
  cases receive a label; ADHD-without-ASD and other-NDD-without-ASD keep
  `label = NA` and are removed by `dropna`. The study is deliberately an
  **ASD-vs-neurotypical contrast**, with the NDD spectrum held out as a
  comparison-defining exclusion rather than analyzed.

A parallel `_full` cohort (`BBC_patients_analytic_full.csv`, 3,499 rows, keeps the
4-way label) is produced by re-running the same pipeline with an environment flag,
so the NDD groups remain available for future analysis without rebuilding.

### 1.3 NDD labels: encoding and gradation

The NDD/mental-health file (`chs_ndd_mental_03232026.csv`, one row per child) is
the labeling source. **Every diagnosis is a binary presence flag (0/1)** — there
is **no severity or gradation** on any individual condition. For each flag there is
a matched first-diagnosis year, first-diagnosis date, and diagnosis visit-count
(a burden proxy, not a severity grade). First-diagnosis year is present for 100% of
flag-positive individuals.

| Flag | Meaning | n (flag = 1) |
|---|---|---:|
| ASD | Autism spectrum disorder | 145 |
| ADHD | Attention-deficit/hyperactivity | 634 |
| ID | Intellectual disability | 83 |
| CD | Communication disorder | 979 |
| LD | Learning disorder | 574 |
| MD | Motor disorder | 55 |
| TD | Tic disorder | 55 |
| conduct | Conduct disorder | 577 |
| misc_DD | Other misc. developmental disorder | 640 |
| other_NDD_DSM5 | OR of the seven non-ASD/ADHD NDDs | 1,544 |
| MDD | Major depressive disorder (mental-health tier) | 266 |
| PTSD | Post-traumatic stress (mental-health tier) | 166 |
| AD | Anxiety disorder (mental-health tier) | 496 |

The only tiered variable, **`NDD ∈ {0,1,2,3}`, is not a severity scale** — it is
the mutually-exclusive priority ranking above (0 = NT, 1 = ASD, 2 = ADHD-no-ASD,
3 = other-NDD).

**NDD multimorbidity is the norm, not the exception.** Of the 1,648 NDD-positive
children, 1,468 carry ≥2 NDD labels and only 180 carry exactly one. Among the 145
ASD children, **only 5 are "ASD-only"; 140 carry ≥1 additional NDD label.** Any
future attempt to model NDD subtypes would be working with heavily overlapping
labels, not clean strata.

### 1.4 What EHR data is actually measured

**Only ICD diagnosis codes.** There are no labs, medications, procedures, or
vitals in the modeling pipeline. The single exception is child **BMI z-score**
(`BBC_underage20_HTWT_BMI_clean031626.csv`), used only as a covariate, never in the
factorization.

- Raw encounter-level ICD data (`chs_ndd_mental_longformat`): **600,271 code rows
  across 3,500 children** (all four groups).
- Analytic mapped event table (`BBC_event_table.csv`, ASD+NT, age < 18, mapped to
  conditions): **62,882 condition-events**, over **123 distinct condition labels**
  in **15 organ-system categories**.
- Each event row carries `ID`, `month`, `Disease_Category`, `condition`, `codes`,
  `age_years`. Timing is therefore available at encounter granularity — the time
  axis is present in the raw data and is only collapsed downstream (§2).

**Code exclusions applied during mapping (`event.py`):**

- V codes (encounter types) and ICD-9 E000–E999 (external causes) are dropped, per
  the diagnosis-only design.
- NDD-*defining* ICD codes are removed **as events** (stems `299, 312, 314, 315,
  317, 318, 319, 307.2`; ICD-10 `F70–F79, F80, F81, F82, F84, F88, F89, F90, F91,
  F95`) so an NDD code cannot itself become a comorbidity outcome. This is an
  event-level filter, orthogonal to the person-level cohort exclusion in §1.2.
- Missing maternal education is imputed to the modal category ("College or
  higher").

### 1.5 Condition mapping and codebook

- **Category level (15 systems):** `ICD_Codes_Extension.xlsx` (= Supp. Table 1)
  maps ICD-9/10 codes to organ-system categories.
- **Condition level:** a condition codebook (`conditions/icd_condition_mapped.xlsx`;
  Supp. Table 2 family) maps codes to the 123 detailed condition labels used for
  window and NMF analyses.
- Congenital conditions are treated as **ever/never** (persistent anomalies), all
  other categories as recurrent events — a modeling distinction worth keeping in
  mind when reading burden counts.

### 1.6 Maternal and demographic covariates

Covariates come from the demographics source (`primary_teleform03202026.csv`) plus a
separate child BMI file. There are **two tiers**: a reduced set carried into the
analytic file, and a substantially richer set that stays in the raw teleform and is
never pulled through. All coverage figures are within the 1,996 analytic children.

**(a) Maternal covariates present in `BBC_patients_analytic.csv` (9):**

| Variable | Type | Coverage |
|---|---|---|
| `maternal_age` | continuous | median 28.3 y; ~0% missing |
| `is_black` | binary race | 56.1% Black / 43.9% not; 0% missing |
| `education` | 3 levels | College 38.8% / HS 34.1% / <HS 27.1%; **0% missing — missing was imputed to "College or higher"** (see §1.4) |
| `marital_status` | Married / Other | 34.0% married; 1.9% missing |
| `maternal_bmi` (+ `_category`) | continuous + 4-cat | median 25.0; 5.8% missing |
| `mat_htn` | binary | 13.1% positive; 0% missing (derived composite) |
| `maternal_diabetes` | No / GDM / DM | GDM 4.9%, DM 2.7%; 1.5% missing |
| `inflam` | binary | 12.5% positive; 9.2% missing (maternal inflammation composite, likely) |

Birth/infant variables also in the file — `birth_weight`, `gest_age`, `preterm`,
`lbw`, `birth_year`, `sex` — are child-level, not maternal.

**(b) The adjustment uses only a subset.** `standardize.py` /
`build_residual_matrix` standardize on **sex × `is_black` × `education`** strata and
nothing else. `maternal_age`, BMI, diabetes, hypertension, marital status, and
inflammation sit in the file but do **not** enter the residualization or direct
standardization.

**(c) Maternal covariates available in the raw teleform but NOT carried through** —
mostly well-populated, and several are classic prenatal exposures:

| Variable(s) | What it is | Coverage |
|---|---|---|
| `race_update` | **Full 9-category** race/ethnicity | 0% missing — `is_black` is a binary collapse of this |
| `yrincom` | Household **income** (12 brackets) | 11.3% missing |
| `PARITY` | **Parity**; 42.8% nulliparous | 0.1% missing |
| `DELTYPE` | **Delivery mode** (cesarean vs vaginal); one category 31.9% | 0.6% missing |
| `smoke`, `ever_smk`, `cont_smk`, `smkreg`, `smk_pres`, `smk_s1/s2/s3` | **Maternal smoking** — any (22.8%), ever, continued, **and by trimester** | ~1–1.3% missing |
| `etoh` | **Alcohol** in pregnancy | 7.3% positive; 3.9% missing |
| `PREECLAM`, `ECLAMP`, `HELLP`, `GHTN`, `CHTN` | **Hypertensive disorders of pregnancy** — pre-eclampsia, eclampsia (0.2%), HELLP (0.9%), gestational HTN, chronic HTN (6.1%) | mostly <1% missing, **except `GHTN` at 40.8%** |
| `GEAA`, `geaa_o`, `GEAL`, `GEAU` | Gestational-age estimates | ~0.2% missing |
| `familyid` | Family/sibling linkage ID | 0% missing |
| `month_dod`, `year_dod` | Date of **delivery** (1998–2019; = birth date, not death) — already reflected in `birth_year` | 0% missing |

**Implication.** The adjustment rests on three variables (sex, binary race,
education), while several well-covered prenatal confounders/exposures — smoking
(with trimester resolution), income, parity, delivery mode, pre-eclampsia, and full
race granularity — sit unused. An ASD−NT difference attributed to ASD could be
riding on an unmodeled SES or smoking gradient. Two caveats before leaning on the
unused set: `yrincom` (11%) and `GHTN` (41%) are meaningfully incomplete, and
`education`'s "0% missing" is an impute-to-college artifact — the raw `edu` field
should be checked for how many values were filled. Expanding the confounder set is
carried as a recommendation in the contrastive section of `analysis_plan.md`.

---

## 2. The modeling matrix: construction and sparsity

### 2.1 Feature selection, residualization, normalization

**Condition selection (`select_nmf_conditions`).** Restricted to seven target
categories (Congenital, Psychiatric, Neurologic, Sensory, Endocrine–Metabolic,
Digestive, Respiratory). A condition is kept if it appears in **≥ 5 ASD children**
AND (**overall ASD prevalence ≥ 3%** OR **max standardized ASD−NT difference ≥ 5
pp** across developmental windows). The current thresholds retain **~50
conditions**.

**Residualization (`build_residual_matrix`).** For each ASD child × condition, the
value is `ASD_present(0/1) − NT_expected_rate`, where the NT rate is directly
standardized over sex × race(is_black) × education strata (strata with < 5 NT
children pooled).

**Nonnegativity + normalization (`nmf_analysis.py`).** The residual is then
**clipped at 0** (`np.clip(residual, 0, None)`) and each row is scaled to **unit L2
norm**. Two consequences worth flagging explicitly:

- Clipping **discards all negative residuals**, so the matrix fed to NMF has the
  *same nonzero pattern as the raw binary presence matrix*. The residualization
  functions as an **IDF-style down-weighting** of conditions common in NT
  (present cells carry weight `1 − NT_rate`), not as a two-sided residual.
- L2 row-normalization (not sum-to-1) is a deliberate choice so that total
  documentation intensity drops out and every child weighs equally.

### 2.2 Sparsity — measured

| Matrix | Shape | Density | Detail |
|---|---|---:|---|
| **Primary (2-D, ever-diagnosed < 18)** | 145 × ~50 | **22.1%** | median 11 conditions/child (IQR 7–15, range 0–26); every retained condition ≥ 5 ASD children; **1 child has 0 retained conditions** (left unassigned) |
| **Tensor (× 6 time windows)** | 145 × ~50 × 6 | **6.1%** | see per-window decay below |

**Per-window density in the tensor** (windows: Pre-dx, then 3-year blocks to 18;
median ASD dx = 3.25 y):

| Window | ASD children with any event | Density |
|---|---:|---:|
| Pre-dx (0–3.25 y) | 132 | 10.9% |
| 3.3–6.3 y | 127 | 8.0% |
| 6.3–9.3 y | 116 | 6.7% |
| 9.3–12.3 y | 99 | 5.1% |
| 12.3–15.3 y | 88 | 4.0% |
| 15.3–18.0 y | 50 | 1.9% |

**53% of condition-window slices rest on fewer than 5 children.** This is the
crux of the sparsity problem: the 2-D collapse is workable at 22% density, but any
time-resolved representation falls to ~6% and thins further with age as coverage
attrites (132 → 50 children). The code's own tensor routine prints a warning at
exactly this threshold.

---

## 3. What has been run (pipeline inventory)

Data flows raw → cleaned → analytic, then into five analysis families. Directories
are under `BBC_Dataset/`.

| Stage | Script(s) | Output(s) | Status |
|---|---|---|---|
| ICD cleaning & merge | `preprocessing/preprocessing.py` | `BBC_patients.csv`, `_full` | run |
| Event mapping (codes→conditions) | `preprocessing/event.py` | `BBC_event_table.csv`, `BBC_patients_analytic.csv`, `unmapped_codes.csv`, `events_per_category_per_label.xlsx` | run |
| Direct standardization | `analysis/standardize.py` | (library used by others) | run |
| Association / window analysis | `analysis/association_analysis.py` | `sT7_condition_diff_with_ci.csv`, `condition_diff_heatmap_*` (Fig 3) | run |
| Fig 3 sensitivity (preterm/LBW) | `analysis/sensitivity_f3.py` | `sF_condition_diff_sensitivity_*` | run |
| **Primary NMF** | `clustering/nmf_analysis.py` | `nmf_model_selection.docx`, `nmf_condition_excess.docx`, `nmf_component_heatmap_k3*.pdf`, `nmf_subject_profile_distribution_k3.pdf`, `nmf_kept_conditions.txt` | run |
| **Tensor NMF (time)** | `clustering/tensor_nmf.py` | `tensor_nmf_rank_scan.csv`, `tensor_nmf_components.xlsx`, `tensor_nmf_profiles.pdf` | run |
| NMF sensitivity (perinatal) | `clustering/sensitivity_nmf.py` | `sF_nmf_sensitivity_preterm_lbw.pdf`, `_summary.csv` | run |
| NMF pre-diagnosis check | `clustering/nmf_prediagnosis.py` | `sF_nmf_prediagnosis.pdf`, `nmf_prediagnosis_summary.csv` | run |
| Prediction models | `clustering/predict_train.py`, `predict_analysis.py` | (in-dir) | present |
| Descriptive figures | `visualization/*.py` | Table 1, cumulative proportion, forest onset, categorical heatmap | run |
| EHR connectedness exploration | `gus/` (this folder) | timepoint coverage, occupancy, continuity/connectedness, burden-by-age | run |

Note: `BBC_Dataset/conditions/` is an exploratory output directory; per the README
nothing downstream reads from it.

---

## 4. What has been tried (method catalogue)

### 4.1 Primary NMF — latent comorbidity profiles within ASD

- **Estimator:** `sklearn.decomposition.NMF`, multiplicative updates
  (`solver="mu"`), **Frobenius loss**, `max_iter=1000`, best-of-**20 random
  restarts** (lowest reconstruction error kept).
- **Input:** the 145 × ~50 residualized, clipped, L2-normalized matrix (§2). A
  child with no retained condition contributes an all-zero row and is left
  unassigned (dominant component = −1), rather than being dropped or forced into a
  profile.
- **Model selection:** K scanned 2–6. Selection is driven by **stability**, not
  fit — the code explicitly notes reconstruction error and explained variance
  improve with K by construction. Current model-selection table:

| K | Recon err | Expl. var | Component sizes | Restart stab. | Bootstrap cosine | Weakest comp. | Assignment concordance |
|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 9.7101 | 0.184 | [72, 72] | 0.986 | 0.894 | 0.887 | 0.781 |
| **3** | **9.4305** | **0.230** | **[41, 55, 48]** | **0.914** | **0.803** | **0.751** | **0.657** |
| 4 | 9.1694 | 0.273 | [33, 35, 31, 45] | 0.947 | 0.788 | 0.746 | 0.612 |
| 5 | 8.9462 | 0.308 | [34, 17, 35, 32, 26] | 0.807 | 0.724 | 0.570 | 0.527 |
| 6 | 8.7238 | 0.342 | [26, 27, 19, 21, 15, 36] | 0.843 | 0.722 | 0.637 | 0.529 |

- **Stability machinery (notably thorough):**
    - *Restart stability* — mean Hungarian-matched cosine of component loadings
      across the 20 restarts (per-component, so an intrinsically hard component is
      distinguishable from a sample-sensitive one).
    - *Bootstrap stability* — 200 patient resamples × 3 restarts each, refit at
      every K, components matched back to the full-sample solution. Reports mean
      matched cosine, weakest-component cosine, and **dominant-assignment
      concordance** (share of children whose top component is unchanged under
      resampling).
    - *Burden diagnostic* — correlation of each component's loading with a child's
      total burden, flagging any component that is really a severity axis rather
      than a distinct pattern.
- **Read on K=3:** loadings are reasonably reproducible (bootstrap cosine 0.803),
  but **dominant-component membership is fragile — concordance 0.657**, i.e. ~1 in
  3 children would switch profile under resampling. Loadings support a stronger
  claim than hard membership counts do.

### 4.2 Tensor NMF — the time axis (tried, appropriately relegated)

- **Estimator:** a hand-rolled **nonnegative CP / PARAFAC** decomposition
  (multiplicative updates, Khatri–Rao without materialization). Mode 3 is
  developmental time (6 windows).
- **Rank scan** (`tensor_nmf_rank_scan.csv`):

| R | Rel. error | Expl. var | Sizes | Restart stab. |
|---:|---:|---:|---|---:|
| 2 | 0.9057 | 0.133 | [74, 70] | 1.000 |
| 3 | 0.8916 | 0.159 | [32, 56, 56] | 0.884 |
| 4 | 0.8783 | 0.184 | [28, 57, 24, 35] | 0.859 |
| 5 | 0.8659 | 0.207 | [33, 20, 34, 30, 27] | 0.969 |

- **Built-in honesty checks:** the routine reports (a) density and a warning when
  most condition-window slices have < 5 children — which fires here (53%); (b)
  correlation of each temporal factor with per-window event *density*, flagging a
  factor that merely tracks data availability; (c) pairwise temporal-factor
  correlation, flagging when extra rank separates conditions rather than timing;
  and (d) cosine agreement of the condition mode against the primary 2-D fit.
- **Verdict embedded in the code and confirmed by the numbers:** explained
  variance is low (0.13–0.21), density is ~6%, and slices are underpowered — so
  the tensor is a sensitivity/robustness object, not a basis for time-resolved
  phenotype claims at n = 145. The time axis *was* tried honestly; the data cannot
  yet support it.

### 4.3 NMF sensitivity — perinatal confounding

`sensitivity_nmf.py` refits K=3 on three restricted samples (term-only, normal
birth-weight-only, both), Hungarian-matches each refit to the primary components,
and reports matched cosine and dominant-concordance per panel. Components below
n = 25 are flagged as underpowered rather than interpreted. Purpose: confirm the
profiles are not artifacts of prematurity / low birth weight.

### 4.4 NMF pre-diagnosis check — early-recognition support

`nmf_prediagnosis.py` refits on events recorded **before the median ASD diagnosis
age (3.25 y)** and matches to the full-window solution. It adds an **observability
ceiling** — the fraction of each component's weight that comes from conditions
already observable pre-diagnosis — quantifying how much of each profile is in
principle detectable before the ASD label. Purpose: support the claim that
comorbidity structure is present early.

### 4.5 Association / developmental-window analysis

`association_analysis.py` + `standardize.py` produce the adjusted ASD−NT
prevalence differences by developmental window (Fig 3 / Supp. Fig 1), directly
standardized over sex × race × education (small strata pooled), with a fast
array-form estimator for bootstrap CIs that is asserted equal to the reference
estimator. Also computes the Spearman ordering correlation showing ASD/NT
condition rankings diverge with age.

### 4.6 Descriptive / connectivity exploration (this `gus/` folder)

Independent EHR-coverage diagnostics already generated here:

- **Timepoint coverage** (`ehr_timepoint_distribution`, `asd_ehr_timepoint_coverage`):
  ASD children have *more* distinct visit-years than NT — sparsity is not driven by
  ASD under-documentation.
- **Occupancy** (`asd_occupancy_matrix`, `asd_occupancy_heatmap`): child × year
  (0–23) presence grid.
- **Continuity / connectedness** (`ehr_continuity_per_individual`,
  `ehr_connectedness_maxspan`, `ehr_continuity_sufficiency`): per-child span,
  continuity (fraction of years between first and last visit that are observed),
  and a connectedness score.
- **Burden by age** (`comorbidity_burden_by_age`): ASD ~3–4.7 conditions/person vs
  NT ~2.5–3.7, with `covered` counts attriting from 121 (age 0) to 20 (age 18) —
  the same attrition that thins the tensor.
- **Category counts** (`comorbidity_counts_by_category`, `asd_timepoints_vs_comorbidities`).

---

## 5. Observations and opportunities

Brief, and separate from the descriptive record above.

1. **Loss function.** Frobenius NMF treats near-binary incidence as Gaussian. For
   presence/count EHR data, **KL-divergence NMF** (`beta_loss="kullback-leibler"`,
   ≈ Poisson) or **logistic/Bernoulli** factorization are more principled — and
   since counts are currently binarized away, feeding **event counts with KL** would
   recover recurrence intensity at no sparsity cost.
2. **Hard vs soft membership.** With dominant-assignment concordance ~0.66, the
   per-profile counts (n = 41/55/48) are the fragile part. The loading matrix and
   partial memberships carry the stable signal; profile *counts* are best treated as
   descriptive.
3. **Time, realistically.** The 6-window tensor is too sparse. **2–3 coarse windows**
   (pre-dx / childhood / adolescence) is the only time-resolved cut n = 145 can bear
   and would keep slices populated.
4. **Sample size is the binding constraint**, not connectivity per se: ~48
   children/component at K=3, and coverage that halves by adolescence. This bounds
   what any unsupervised method can recover and is worth stating as a first-order
   limitation.
5. **Reproducibility (minor).** `config.py` and `event.py` hardcode Windows paths
   (`/mnt/c/Users/lkuang/...`), so the pipeline does not run as-is off the author's
   machine; a relative-path config would make it portable.

---

## Appendix — file map

```
ASD_revision/
├─ kid_icdcode_data_Mar2026.RDS            raw EHR ICD (source)
├─ BBC_underage20_HTWT_BMI_clean031626.csv child BMI z-score (covariate)
├─ NDD datasets 3-24-2026/
│  ├─ chs_ndd_mental_03232026.csv          wide NDD labels (1 row/child)
│  ├─ chs_ndd_mental_longformat_*.csv      encounter-level ICD (600,271 rows)
│  └─ ASD_NDD_Flowchart.pptx               consort flow
├─ ICD Codebook/  s_table_1.xlsx, s_table_2.xlsx
└─ BBC_Dataset/
   ├─ preprocessing/  preprocessing.py, event.py, ICD_Codes_Extension.xlsx
   │  └─ results/  BBC_event_table.csv, BBC_patients_analytic.csv (+_full)
   ├─ analysis/      association_analysis.py, standardize.py, sensitivity_f3.py
   ├─ clustering/    nmf_analysis.py, tensor_nmf.py, sensitivity_nmf.py,
   │                 nmf_prediagnosis.py, predict_*.py, config.py, clustering_utils.py
   │  └─ results/    nmf_*, tensor_nmf_*
   ├─ visualization/ table1.py, cum_prop.py, forest_onset_plot.py, categorical_heatmap.py
   └─ conditions/    exploratory outputs (not read downstream)
```

*All counts recomputed from source on 2026-08-28; the study scripts were read, not
executed (hardcoded external paths).*
