# ADHD-specific comorbidity in the Boston Birth Cohort (`gus` branch)

Postnatal-EHR comorbidity analyses contributed by Gus Fridell. They ask which comorbidities are specific to ADHD and which are
shared with other neurodevelopmental disorders (NDD). An ADHD-vs-neurotypical comparison cannot answer that on its own: it shows
which conditions are *ADHD-associated*, not which are *ADHD-specific*.

**Start here: [`adhd_specific/`](adhd_specific/)**
- [`README.md`](adhd_specific/README.md): layout, how to rerun, data policy.
- [`RESULTS.md`](adhd_specific/RESULTS.md): the full write-up. It covers cohort construction and filtering, ICD processing and condition grouping, the ICD definition of every condition (Appendix A), statistical formulation, results, figures and caveats.
- `RESULTS.html`: the same document as one file with the figures embedded. Download it and open it in a browser.

> **Moved:** the earlier full-spectrum work (S1–S8: NT / other-NDD / ADHD / ASD treated as a spectrum, ASD as the focus) has
> been consolidated into [`full_spectrum/`](full_spectrum/), with its own [README](full_spectrum/README.md). Nothing in it
> changed apart from the move.

## Cohort

Children are given mutually exclusive groups by the priority rule ASD > ADHD > other-NDD > NT, using the BBC NDD label file
(3,500 children; 3,499 analysed).

| Group | n | Role |
|---|---|---|
| **ADHD** (ADHD without ASD) | **573** | Focal group |
| Other NDD (no ASD, no ADHD) | 930 | Primary comparator: separates ADHD-specific from shared-NDD signal |
| ASD | 145 | Comparator; split into ASD+ADHD 61 / ASD-only 84 for the dual-diagnosis analysis |
| NT (no neurodevelopmental diagnosis) | 1,851 | Reference |

The outcomes are 78 conditions selected for ADHD: at least 5 ADHD children and at least 3% ADHD prevalence, across all disease categories. Each is
coded as ever present before age 18. Sensitivity runs also use any-ADHD (634 = 573 + 61), the definition used by the `main`-branch code.

## Analyses (D1–D7)

| | Question |
|---|---|
| D1 | Which conditions are ADHD-specific vs shared with other NDD? Empirical-Bayes shrinkage, tested *beyond* ADHD's uniform offset |
| D2 | Is each ADHD elevation a step (ADHD-only) or a gradient (other-NDD part-way there)? |
| D3 | Do ADHD NMF comorbidity profiles survive stability floors and removal of NT co-occurrence structure? |
| D4 | Inattentive vs hyperactive/combined presentation |
| D5 | What changes around the ADHD diagnosis, beyond age and NDD-care trends (pre/post difference-in-differences)? |
| D6 | ADHD with vs without autism: are the effects additive? |
| D7 | Do the signals survive adjustment for healthcare utilization? |

## Main findings

1. **Most ADHD-vs-NT comorbidity is not ADHD-specific.** 68 of 78 conditions are elevated vs NT, but ADHD sits a uniform ~2.4 percentage points above other NDD on nearly everything. Only 8 conditions deviate beyond that offset, and 6 survive utilization adjustment.
2. **The robust ADHD-specific core:**
   - anxiety, mood and personality-disorder codes, plus asthma, the one non-psychiatric condition that survives every check;
   - substance use, which is provisional because part of it is neonatal or exposure coding;
   - constipation, which is *lower* in ADHD after adjustment.
3. **ADHD's footprint is the same with and without autism** (no interaction). ADHD is diagnosed *earlier* in autistic children.
4. **Anxiety is elevated before the ADHD diagnosis.** Anxiety, mood and sleep codes rise further after it, beyond age and NDD-care trends. Co-coding at the ADHD workup is not yet ruled out.

Details, numbers and limitations are in [`adhd_specific/RESULTS.md`](adhd_specific/RESULTS.md).

## Repository layout

```
adhd_specific/   D1–D7 ADHD-focused analyses: code, RESULTS.md / RESULTS.html, aggregate results/  ← current work
full_spectrum/   S1–S8 full-spectrum (ASD-focused) analyses and write-ups, moved here and consolidated
code/, codebook/, ADHD_multimorbidity_project_diary.pptx   from the main branch (Linna Kuang's ADHD-vs-NT pipeline)
```

## Data policy

- **No individual-level data is included.** Every file is aggregate: per-condition or per-group tables, and figures with no per-child points.
- **Paths:** the real compute location is replaced by `<PROJECT_ROOT>`.
- **Running the code** requires the restricted BBC source data and the upstream preprocessing pipeline; see `adhd_specific/README.md`.
