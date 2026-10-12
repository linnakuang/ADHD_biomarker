# ASD Comorbidity — Analysis Opportunities & Strategy Plan

**Purpose.** A planning document for exploratory analyses beyond the primary
residualized NMF. Two tracks: (A) a **longitudinal/temporal factorization** built
under the most permissive scheme the sample can bear, and (B) **contrastive
factorization** that corrects a specific weakness in the current residualization.
The governing reality is n = 145 ASD children, so several of these are expected to
return null or marginal results. That is acceptable and, stated correctly, useful:
a *pre-registered* negative ("the time-resolved structure is not estimable at this
n, and here is the density that shows why") is a defensible contribution. This plan
therefore fixes decision rules **before** running anything, so a null is credible
rather than a failed search.

*Prepared 2026-08-28. Companion to `analysis_memo.md`. Execution status added 2026-09-01.*

[TOC]

---

## 0. Execution status (updated 2026-09-01)

Both tracks have now been run in `<PROJECT_ROOT>/gus_analysis/`. Full
write-ups: **Track A** → `RESULTS.md` + `WRITEUP_TrackA.md` (+ `.html`); **Track B** →
`RESULTS_TrackB.md`. All verdicts applied the §2 rules as fixed (§7: a null that clears
§2 is reported).

| Analysis | Status | §2 verdict | One-line outcome |
|---|---|---|---|
| A1 dx-anchored 3-window CP | done | **NOT reportable** | temporal factor = data-availability artifact (r up to 0.996) |
| A2 smoothness-regularized CP | done | **NOT reportable** | density artifact survives every λ |
| A3 pre/post-dx two-window NMF | done | **REPORTABLE** (stability) | structure reshapes across dx boundary; C1 metabolic→psychiatric post-dx |
| A3b ascertainment guard | done | **guard** | reshape *shape* NOT ASD-specific (NT ρ 0.80 vs ASD 0.91); only a detectability-locked magnitude excess (top-8 p=0.010) |
| A4 timing-encoded matrix | done | **REPORTABLE** | early-onset acute/GI vs late-onset psychiatric/sensory split |
| A5 burden-trajectory clustering | done | **LEVEL reportable, SHAPE null** | graded severity gradient; timing matches NT schedule (A5b) |
| B1 differential co-occurrence | done | **directional** (as designed) | 0 infectious / 59 congenital stable pairs → §4.1 confirmed |
| B2 residual-of-projection | done | **`‖r_i‖` REPORTABLE**, components not | per-child atypicality score is the deliverable; all 3 published components dissolve/partial |
| B3 contrastive β-spectrum | done | **corroborates** (parametric) | C2 infectious decays fastest; max-β basis congenital/psychiatric |

**Headline.** *Track A* — temporal signal is recoverable only in forms that avoid the
sparse third mode (A3/A4/A5); a genuine ≥3-window tensor is not estimable at n=145 (the
quantified ceiling is itself the result, §3.2 / §7). The A3 pre/post reshape, tested
against NT (A3b), is mostly age/ascertainment: NT reshapes toward A3's own top-loaded
conditions nearly as strongly as ASD, leaving only a magnitude excess that is
age-locked-late in NT and so unidentifiable from earlier ascertainment. *Track B* — three
independent contrastive methods converge: the published components are largely generic
pediatric co-occurrence residualization never removed; the ASD-differential *second-moment*
structure is **congenital, not infectious**. The robust quantitative deliverable is B2's
per-child **`‖r_i‖` structural-atypicality score** (rank covariate; Spearman ≥0.95 over
K_NT 12–20).

### 0.1 Open contrastive threads — the shift-gears target for the next chat

Track B answered the §4.1 methodological question; what remains open on the contrastive
side, in rough priority:

1. **Confounder-set expansion (§4.1, applies to B1–B3 + primary standardization).** Every
   run so far used only **sex × is_black × education**. The teleform carries maternal
   smoking (trimester resolution), household income, parity, delivery mode, pre-eclampsia,
   and full 9-category race — re-derive the NT-normative model (`H_NT`) and the
   standardization strata on these before any ASD-specific claim is finalized. `yrincom`
   (~11% missing) and `GHTN` (~41%) need missing indicators. **This is the main
   not-yet-done contrastive item.**
2. **B2 `‖r_i‖` as a downstream covariate.** The score is the intended input to the
   molecular / outcome analysis; nothing downstream has consumed it yet.
3. **Track C combined (§5).** `‖r_i‖` within the diagnosis-anchored early window —
   exploratory, density-limited on the A-side; reasonable now B2 is positive, deferred
   otherwise.
4. **B3 manuscript polish** if it ships (the β=1–2 transient collapse is noted in
   `RESULTS_TrackB.md`; ranking/endpoints carry the claim).

---

## 1. Governing constraints (what bounds every design below)

From the data audit (`analysis_memo.md`), recomputed on source:

- **n = 145 ASD** children; ~48 per component at K=3; dominant-component assignment
  concordance under resampling is only 0.66.
- The **primary 2-D matrix** (145 × ~50, ever-diagnosed) sits at **22% density** —
  workable.
- The **six-window tensor collapses to 5–6% density**, and **~55% of
  condition-window slices rest on fewer than five children**. That is the wall.
- Coverage **attrites with age** (133 children with any event in [0,3) → 56 in
  [15,18)), and the tensor code's own diagnostic flags temporal factors that merely
  "track data availability." Right-censoring at older ages is not signal.

**Design consequence.** The binding constraint is sample size expressed as *slice
occupancy*, not connectivity per se. Every design below either (i) shrinks the
number of things being estimated (fewer windows, fewer parameters, a scalar
outcome), (ii) borrows strength across cells (smoothness, shrinkage, a normative
prior), or (iii) trades the tensor for a denser representation of the same timing
information. Anything that does none of these will factorize sampling noise.

---

## 2. Pre-registered decision rules (apply to all analyses)

Set once, here, so results are interpretable either way:

| Rule | Threshold | Rationale |
|---|---|---|
| **Density floor** | A (condition × window) slice enters a model only with **≥ 5 children**; a window enters only if **≥ 60% of retained conditions** clear the floor | prevents estimating structure from empty cells |
| **Stability floor** | Bootstrap (B = 200) matched cosine **≥ 0.80** *and* dominant-assignment concordance **≥ 0.60** for a solution to be "reportable" | matches the machinery already in `nmf_analysis.py` |
| **Min component size** | A component with **< 20 children** is flagged provisional, not interpreted | mirrors `sensitivity_nmf.py`'s existing rule |
| **Data-availability guard** | Any temporal factor with **r > 0.7** vs per-window event density is discarded as an artifact | already implemented in `tensor_nmf.py` |
| **Null is publishable** | If a track fails the stability floor, the reported result is the **feasibility table + the failure**, not a lowered threshold | keeps the negative honest |

---

## 3. Track A — Longitudinal / temporal factorization

### 3.1 Why the full six-window tensor cannot work (measured)

The production tensor is 145 × ~50–80 × 6. Recomputed density by window (ASD, ≥5
child condition filter, `/tmp/temporal_feasibility.py`):

| Window | Children w/ event | Window density |
|---|---:|---:|
| [0,3) | 133 | 9.5% |
| [3,6) | 130 | 6.8% |
| [6,9) | 119 | 5.8% |
| [9,12) | 108 | 4.4% |
| [12,15) | 93 | 3.8% |
| [15,18) | 56 | 1.9% |

Overall 5.4% density; **55% of slices < 5 children; median slice n = 4.** Not
estimable.

### 3.2 The truncation idea, quantified — "stop the endpoints early"

The user's instinct is correct and it measurably helps. Truncating the age axis to
the densely-sampled early-childhood window (where comorbidities present *and* where
children are actually seen) roughly halves the thin-slice fraction and lifts median
slice occupancy. Candidate schemes:

| Scheme | Windows | Conditions | Density | Slices < 5 | Median slice n |
|---|---:|---:|---:|---:|---:|
| Full 0–18, 3y | 6 | 81 | 5.4% | **55%** | 4 |
| Truncate @12, 3y | 4 | 77 | 6.9% | 44% | 6 |
| Truncate @10, 2y | 5 | 76 | 5.7% | 51% | 4 |
| Truncate @8, 2y | 4 | 72 | 6.3% | 45% | 6 |
| **Truncate @9, 3y** | **3** | **74** | **8.0%** | **37%** | **7** |
| **Dx-anchored: [0,3.25),[3.25,6.25),[6.25,9.25)** | **3** | **75** | **7.9%** | **38%** | **7** |
| Truncate @6, 2y | 3 | 65 | 7.3% | 37% | 7 |
| **Two-window pre/post-dx** | **2** | **81** | **11.8%** | **25%** | **12** |

**Reading of the table.** Truncation genuinely moves the analysis from *hopeless*
(55% thin) to *marginal-but-defensible* (37% thin). It does **not** reach
comfortable territory — even the best 3-window scheme leaves ~1/3 of slices thin —
so the honest expectation is a provisional result that likely fails the 0.80
stability floor. That expectation is the point: the truncated tensor is the object
that lets us *demonstrate* the ceiling rather than assert it.

### 3.3 A1 — Diagnosis-anchored 3-window CP (recommended temporal primary)

- **Object:** 145 × ~75 × 3 tensor, windows `[0, 3.25) / [3.25, 6.25) / [6.25,
  9.25)` — anchored on the median diagnosis age rather than arbitrary calendar
  cuts, so the pre-diagnosis window is developmentally meaningful.
- **Method:** the existing `ncp()` non-negative CP (multiplicative updates) in
  `tensor_nmf.py` — no new machinery, just change `_build_time_bins` and truncate.
- **Rank:** scan R = 2–4 only (R ≥ 5 is unsupportable at this density).
- **Read-outs already coded:** restart + bootstrap stability, temporal-factor vs
  density correlation, condition-mode agreement with the 2-D fit.
- **Decide by §2.** Expect it to clear density but sit near the stability floor.
- **Effort:** low (hours). **Power:** low–marginal. **Run first in Track A.**

### 3.4 A2 — Smoothness-regularized temporal mode (the permissive extension)

The single change that most extends what the tensor can bear: **penalize roughness
in the temporal factor.** Add `λ · Σ_t (C[t+1] − C[t])²` to the CP objective so a
thin slice borrows strength from its neighbors instead of being fit independently.
This is the disciplined way to be permissive — it lets you keep more windows (even
the full 0–18) because adjacent windows are no longer estimated in isolation.

- **Implementation:** the temporal-mode multiplicative update in `ncp()` gains a
  tridiagonal smoothing term; ~20 lines. Choose λ by the stability of the temporal
  factor, not reconstruction error.
- **Payoff:** turns "each window is its own tiny sample" into "one smooth
  trajectory with ~6 knots," which is far fewer effective parameters.
- **Effort:** medium. **Power:** the best shot at *keeping time resolution*.

### 3.5 A3 — Two-window pre/post-diagnosis contrast (the robust anchor)

At **11.8% density and only 25% thin slices**, the pre-dx vs post-dx object is the
one temporal design comfortably above the floor. It answers a narrower question —
*does comorbidity structure differ before vs after the ASD label?* — but answers it
credibly. This generalizes the existing `nmf_prediagnosis.py` (which fits the pre-dx
slice alone) into an explicit two-slab comparison with Hungarian-matched
components. **Run this as the temporal result you can actually stand behind**, with
A1/A2 as the "we also tried finer time resolution and here is why it doesn't hold."

### 3.6 A4 — Timing-encoded matrix (temporal signal without a tensor)

Sidestep the third mode entirely. Keep a **145 × ~50 matrix at 22% density**, but
make each nonzero entry carry *timing* instead of 0/1:

- **age-at-first-event** (earlier onset → larger value, e.g. `1/(age+1)` or a
  z-scored recency), or
- **a two-column split** per condition (early/late indicator), doubling columns but
  staying dense.

NMF then discovers components that mix *what* and *when* without ever facing a 6%
tensor. Cost: it cannot recover free-form temporal trajectories the way CP can, but
it uses onset timing at full power. **Effort: low. Power: good** — a strong
complement to A3.

### 3.7 A5 — Burden-trajectory clustering (well-powered fallback)

If the condition × time structure is unrecoverable (likely), collapse conditions to
a **cumulative comorbidity count per age** and cluster the 145 *trajectories*
(latent-class growth / k-means on splines). One value per child per age is dense and
well-powered; 2–3 trajectory classes (e.g. early-plateau vs escalating burden) are
realistic at n = 145. This is the temporal analysis most likely to yield a stable,
reportable result, at the cost of discarding condition identity. **Effort: low.
Power: highest in Track A.**

### 3.8 Count/KL variant (applies to A1–A3)

The conditions are currently binarized. Feeding **event counts** with a
**KL-divergence (Poisson) loss** (`beta_loss="kullback-leibler"`) recovers
recurrence intensity — how many otitis episodes, not just ever/never — at **no
sparsity cost**, and better matches count data than Frobenius. Cheap to add and
worth running as a variant of whichever temporal object survives §2.

---

## 4. Track B — Contrastive factorization

### 4.1 The core critique (why residualization is not enough)

**Moments.** The mean is the first moment; variance/covariance is the second. The
current pipeline residualizes each condition against its demographically-expected
NT rate — an adjustment to the **first moment**, one condition at a time,
independently. But NMF builds components out of **co-occurrence**, which is
**second-moment** information. The correction therefore operates on a quantity NMF
does not consume and leaves untouched the quantity it does.

**Sharpened by this pipeline's clip step.** The residual is then `clip(·, 0)`
(memo §2.1), discarding negative residuals. So the co-occurrence structure actually
fed to NMF is essentially the **raw, demographically-unadjusted ASD co-occurrence**
— the residualization down-weights common conditions cell-by-cell (IDF-style) but
never removes the *joint* structure that ordinary children also share.

**The concrete failure.** URI and otitis media co-occur in nearly all young
children (shared anatomy, season, daycare). Residualization removes the NT
*baseline rates* but the *correlation* between them is identical in ASD and NT and
passes through untouched. NMF then assembles a component from that surviving
co-occurrence — real, but not ASD-specific. The reframed question contrastive
methods ask is **"which conditions co-occur *differently* in ASD than in ordinary
children,"** not "which are more prevalent."

**Standing prediction (pre-registered).** Under any contrastive treatment,
**Component 3 (infectious–inflammatory: otitis, URI, LRI, pharyngitis) should
largely dissolve** — it is the canonical generic pediatric cluster — while
**Component 2 (congenital cardiac, GERD, retinal, asthma) should survive**, because
rare congenital anomalies have genuinely unusual joint structure. If that is the
outcome, it is a methodological contribution independent of any downstream result.

> **CONFIRMED (2026-08-31, see `RESULTS_TrackB.md`).** The infectious–respiratory
> component dissolved on all three methods — B2 projection cosine 0.467, **0** stable
> differential pairs in B1, fastest β-decay in B3 — while congenital/chromosomal
> structure survived (59 stable B1 pairs, 37%; dominates the max-β basis). The
> ASD-differential joint structure is congenital, not infectious.

**Expand the confounder set (applies to B1–B3 and to the primary standardization).**
The current adjustment uses only **sex × is_black × education** (memo §1.6), yet the
raw teleform carries well-covered prenatal confounders that go unused — **maternal
smoking (with trimester resolution), household income, parity, delivery mode,
pre-eclampsia, and full 9-category race**. Any "ASD-specific" claim is stronger if
the NT-normative model (B2's `H_NT`) and the standardization strata condition on
these rather than on three variables — otherwise a residual attributed to ASD may
be riding an unmodeled SES or smoking gradient. Add them as strata (or as NT-side
model covariates), noting `yrincom` (~11% missing) and `GHTN` (~41%) need a missing
indicator.

### 4.2 B1 — Differential co-occurrence + symmetric NMF (sanity check, run first)

- Compute the condition × condition co-occurrence separately in ASD and NT
  (**tetrachoric** or partial correlations, since binary), form `D = C_ASD − C_NT`,
  and factorize/cluster the **positive part** with symmetric NMF.
- **Interpretation is trivial:** these condition *pairs* co-occur more in ASD than
  in NT.
- **Essential caveat:** `C_NT` is well estimated (n = 1,851), but `C_ASD` is ~2,500
  pairs from 145 children — pure noise without **shrinkage**. Use Ledoit–Wolf or
  **graphical lasso**, and **bootstrap the edges** so only stable differential pairs
  are reported.
- **Effort: low. Power: noisy but directional.** Best used to confirm the §4.1
  prediction before building anything heavier.

### 4.3 B2 — Residual-of-projection (recommended contrastive primary)

The strongest option, because it reuses tooling already in the repo and yields a
well-powered scalar.

1. Fit NMF on the **1,851 NT children** → `H_NT`, a basis of ordinary pediatric
   multimorbidity. Choose `K_NT` generously (**8–12**, by NT-side cross-validation)
   so the normative model is rich, not impoverished.
2. Project each ASD child by NNLS: `w_i = argmin_{w≥0} ‖x_i − w·H_NT‖²`.
3. Reconstruct `x̂_i = w_i·H_NT` — the part of this child's comorbidity that
   *ordinary* pediatric structure explains, **conditioned on the rest of their
   profile** (the key advance over demographic-only residualization).
4. Residual `R = X_ASD − X̂`. Positive entries are conditions present *in excess of
   what the normative pattern predicts*. Shift non-negative and factorize `R` →
   ASD-specific components.
5. **Bonus, and arguably the headline:** `‖r_i‖` is a **per-child scalar** — how
   structurally atypical that child's comorbidity profile is. One **continuous,
   well-powered outcome** is a far better target at n = 145 than three compositional
   loadings, and it is a natural covariate for any downstream (molecular) analysis.

- **Power note:** the NT-side fit is well-powered; the residual factorization
  inherits ASD-side sparsity, so treat the **components** as provisional but the
  **`‖r_i‖` score** as the robust deliverable.
- **Effort: medium. Power: best available.** **Run as the primary contrastive
  analysis.**

### 4.4 B3 — Contrastive objective / β-spectrum (most ambitious, run last)

Solve `min ‖X_A − W_A H‖² − β‖X_N − W_N H‖²` with `H` **shared**: β = 0 recovers the
current paper; large β pushes `H` toward structure that explains ASD but *not* NT.

- The negative term is **unbounded below** — you must constrain `‖h_r‖ = 1` and use
  projected gradient, or it diverges.
- Do not pick a single β. **Report the components as a function of β** (a
  "contrastive spectrum"), the way the contrastive-PCA literature (Abid et al.)
  handles the identical problem.
- **Effort: high (custom optimizer + tuning). Power: hardest to validate at n =
  145.** Attempt only if B1/B2 are encouraging.

### 4.5 Deliverable for Track B

Across B1–B3, the reportable question is: **which of the published components
survive a contrastive treatment?** The pre-registered prediction (§4.1) is
infectious–inflammatory dissolves, congenital survives. Either outcome is
informative; a component that dissolves was never ASD-specific, and saying so
crisply is a genuine methodological result for the ASD comorbidity literature.

---

## 5. Track C — Combined (contrastive within the early window)

The two tracks compose. The residual-of-projection (B2) can be run **within the
diagnosis-anchored early window (A1)**: fit `H_NT` per early window, project the ASD
early-window slices, and read `‖r_i‖` as an **early-childhood structural-atypicality
trajectory**. This is speculative and density-limited, but it is the natural place a
positive B2 result and a marginal A1 result would meet. Deprioritized until both
parents show signal.

---

## 6. Prioritization and sequencing

Ordered by (power × interpretability) / effort — run top-down, stop where the
stability floor stops being cleared. *(All of #1–#8 have now been run; see §0 for
outcomes. The Run? column below is the original pre-execution plan, kept as the record
of what was committed before results.)*

| # | Analysis | Track | Effort | Expected power | Run? |
|---|---|---|---|---|---|
| 1 | **B2 residual-of-projection** (+ `‖r_i‖` score) | B | Med | **Best** | **Yes — primary** |
| 2 | **A3 pre/post-dx two-window** | A | Low | Good | **Yes — robust temporal anchor** |
| 3 | **A5 burden-trajectory clustering** | A | Low | **Best in A** | **Yes — well-powered fallback** |
| 4 | B1 differential co-occurrence (shrunk) | B | Low | Directional | Yes — sanity check |
| 5 | A4 timing-encoded matrix | A | Low | Good | Yes |
| 6 | A1 dx-anchored 3-window CP | A | Low | Marginal | Yes — to demonstrate the ceiling |
| 7 | A2 smoothness-regularized CP | A | Med | Marginal | If A1 is borderline |
| 8 | KL/count variant | A | Low | — | On whichever A object survives |
| 9 | B3 contrastive β-spectrum | B | High | Hardest | Only if B1/B2 encourage |
| 10 | C combined | A×B | High | Speculative | Only if A1 + B2 both signal |

**Fastest path to a defensible result:** #1 (B2's `‖r_i‖`) and #3 (burden
trajectories) are both well-powered and low-effort — start there. #2 is the temporal
result you can publish. #6/#7 exist primarily to *document the ceiling* the user
already anticipates.

---

## 7. What counts as "it didn't work" (so the negative is credible)

Pre-committed, per §2:

- **Temporal (A1/A2):** a solution that fails bootstrap cosine ≥ 0.80 **or** has any
  temporal factor with r > 0.7 vs density is reported as *not estimable*, with the
  feasibility table (§3.2) as the evidence — **not** re-tuned to pass.
- **Contrastive (B1/B2):** if the ASD-specific residual components fail the
  stability floor, `‖r_i‖` is still reported as a continuous outcome (it does not
  depend on component stability), and the component-level null is stated plainly.
- **The feasibility table itself is a result.** "A time-resolved factorization is
  not supported below ~8% density / above ~37% thin slices at n = 145" is a
  quantitative, reusable statement about this class of cohort.

---

## 8. Implementation notes

- **Environment gap:** `~/miniforge/bin/python` has pandas/numpy/scipy but **no
  scikit-learn**; NMF and NNLS need it. Install into the miniforge env
  (`~/miniforge/bin/pip install scikit-learn`) before running B2/A1, or hand-roll
  NNLS via `scipy.optimize.nnls` (already available).
- **Reuse, don't rebuild:** `ncp()` (tensor), `build_residual_matrix`,
  `select_nmf_conditions`, `_build_time_bins`, and the bootstrap/Hungarian stability
  code in `nmf_analysis.py` cover most of Track A. B2 needs only an NT-side NMF fit +
  a per-child `scipy.optimize.nnls` loop.
- **Paths:** `config.py` hardcodes Windows paths; point it at
  `<PROJECT_ROOT>/BBC_Dataset/preprocessing/results/` before any run.
- **Feasibility script:** `/tmp/temporal_feasibility.py` reproduces §3.2 and is the
  place to add new truncation schemes.

*All density and slice figures recomputed from `BBC_event_table.csv` /
`BBC_patients_analytic.csv` on 2026-08-28.*
