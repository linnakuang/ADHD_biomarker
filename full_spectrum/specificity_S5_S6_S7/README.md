# ASD-specificity within the spectrum (S5, S6, S7)

Which comorbidities are genuinely specific to ASD versus shared across the neurodevelopmental
spectrum — and do they survive comparison against ADHD alone?

**Individuals used** — all three use the four priority-labeled groups (NT 1,851 / other-NDD 930 /
ADHD 573 / ASD 145); differences are in the comparator:
- **S5 (`s5_hierarchical_shrinkage.py`)** — ASD (145) vs the pooled **rest** (1,503), NT as baseline;
  empirical-Bayes (DerSimonian–Laird) shrinkage decomposes each condition into shared-NDD + ASD-specific.
- **S6 (`s6_gradient_vs_step.py`)** — reads the full four-group "staircase" (NT → other-NDD → ADHD →
  ASD) and classifies each condition's shape (STEP / STEP-DOWN / GRADIENT+ / SHARED / …).
- **S7 (`s7_cross_group_contrast.py`)** — three shrunk contrasts: ASD vs rest, **ASD vs ADHD alone**
  (the hard specificity test), and ADHD vs other-NDD (calibration).

**Findings**
- **S5:** 7/51 conditions survive as ASD-specific at 95% (11 at 90%); the *largest* ASD-vs-NT signals
  (anxiety, refractive, otitis, obesity) are 70–104% generic NDD.
- **S6:** of the 11 ASD-specific conditions, **8 step up, 2 step down, only 1 is a true gradient** —
  ASD-specificity, where it exists, is overwhelmingly step-shaped; the biggest elevations are SHARED.
- **S7:** against **ADHD alone**, only **2** remain AUTISM-SPECIFIC — **epilepsy** and **glaucoma**.
  Seven of the S6 set are reclassified SHARED-WITH-ADHD; four conditions (headache, mood, acute URI,
  pain) actually **peak in ADHD**, not ASD. Pooling other-NDD into "the rest" had inflated the
  apparent specificity of 9 conditions. (Note: the ASD group still contains ~61 ADHD-comorbid
  children by the priority rule, which makes the ASD-vs-ADHD test *conservative*.)

**Files** — `results/`: `s5_asd_specific.csv`, `s5_summary.txt`, `s5_forest.pdf`;
`s6_gradient_vs_step.csv`, `s6_summary.txt`, `s6_staircase.pdf`;
`s7_cross_group_contrast.csv`, `s7_summary.txt`, `s7_contrast.pdf`.
