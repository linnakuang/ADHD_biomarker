# Power & axes (S1, S2)

Do the comorbidity axes get more reproducible with the larger spectrum cohort (S1), and is ASD
detectable within the shared NMF latent space (S2)?

**Individuals used**
- **S1 (`s1_axis_robustness.py`)** — compares the **ASD-only** cohort (145) against the **full
  spectrum pool** (1,648 = ASD + ADHD + other-NDD) via a learning curve over spectrum subsamples.
- **S2 (`s2_asd_separation.py`)** — fits the latent space on the **spectrum pool** (1,648) and
  classifies **ASD vs other-NDD / ADHD / pooled rest**. NT (1,851) is used only as the residual
  baseline (it is ~0 by construction and excluded from the fit).

**Findings**
- **S1:** the paper's ASD-only K=3 model sits below its own §2 stability floor (worst component 0.76);
  on the spectrum it rises to 0.96. Reproducibility was limited by **n, not biology**.
- **S2:** ASD-vs-other-NDD latent AUC = 0.587 (permutation p = 0.006) — statistically real but barely
  above a burden-only baseline (0.566), and **less distinct than ADHD-vs-other-NDD (0.638)**. ASD is
  a graded tilt toward the psychiatric axis, not a separable island.

**Files** — `results/`: `s1_anchors.csv`, `s1_learning_curve.csv`, `s1_learning_curve.pdf`;
`s2_separation_auc.csv`, `s2_k_sweep.csv`, `s2_axis_group_means.csv`, `s2_summary.txt`.
