# Developmental timing (S3, S4, S8)

Can the larger cohort support a *developmental-trajectory* factorization of comorbidity — and does
correcting for ascertainment ever make it valid?

**Individuals used**
- **S3 (`s3_temporal_feasibility.py`)** — density/feasibility audit, **ASD-only (145) vs spectrum
  (1,648)**.
- **S4 (`s4_spectrum_tensor.py`)** — nonnegative CP/PARAFAC on the child × window × condition tensor,
  **ASD-only vs spectrum pools**.
- **S8 (`s8_temporal_normalized.py`)** — three ascertainment corrections vs baseline on the
  **spectrum pool (1,648)**, scored on whether the temporal factor separates **ASD vs the rest**.

**Findings**
- **S3:** inclusion makes a ≥3-window factorization *estimable* — thin (under-populated) slices fall
  53% → 12%; windows passing the density floor go 1/6 → 6/6.
- **S4:** but it is not *valid* — the temporal factor still tracks the per-window density envelope
  (|r| ≈ 0.88), essentially unmoved by 10× the data, because the envelope shape is n-invariant.
  Estimable ≠ valid; this strengthens the "temporal factor = ascertainment artifact" conclusion.
- **S8 — clean negative:** of BASELINE / REL-TIMING / DENS-EQUAL / UTIL-NORM, only utilization
  normalization at rank 2 clears the density-artifact guard (|r| = 0.684) while staying stable — and
  there its ASD-vs-rest separation is AUC 0.540 (a coin flip). Once coding volume is divided out,
  developmental *timing* carries no ASD signal beyond the marginal rates (see S5–S7).

**Files** — `results/`: `s3_temporal_feasibility.csv`; `s4_spectrum_tensor.csv`;
`s8_temporal_normalized.csv`, `s8_summary.txt`.
