# ── ADHD-focused analyses: paths + knobs ──
# Mirrors gus_analysis/config.py (real Linux paths) but defaults to the *_full tables,
# which carry all 3,499 labeled children (NT / ASD / ADHD / Other NDD).
_ROOT = "<PROJECT_ROOT>"
_BASE = f"{_ROOT}/BBC_Dataset/preprocessing/results"
EVENT_PATH = f"{_BASE}/BBC_event_table_full.csv"
PATIENTS_PATH = f"{_BASE}/BBC_patients_analytic_full.csv"
LABEL_CSV = f"{_ROOT}/NDD datasets 3-24-2026/chs_ndd_mental_03232026.csv"
OUT_DIR = f"{_ROOT}/adhd_analysis/results"
GUS_DIR = f"{_ROOT}/gus_analysis"          # reuse gus_common / S-script kernels from here

# ── Cohort ──
# Primary ADHD group = priority-rule "ADHD" label (ADHD without ASD, n=573), matching the
# diary's case definition. Sensitivity = any-ADHD (634 = 573 + 61 ASD+ADHD), which is what
# the main-branch build_patients.py actually labels "ADHD".
ADHD_DX_AGE_COL = "ADHD_firstdiag_years"
AGE_CUTOFF = 18

# ── NMF ──
NMF_K = 3
NMF_N_RESTARTS = 20
NMF_TOP_CONDITIONS = 5
NMF_MIN_CASE_COUNT = 5
NMF_MIN_PREV = 0.03
NMF_MIN_DIFF_PP = 5.0

NMF_BOOTSTRAP_B = 200
NMF_BOOTSTRAP_RESTARTS = 3
NMF_BOOTSTRAP_SEED = 20261013

# ── Pre-registered decision rules (same §2 floors as gus_analysis) ──
DENSITY_FLOOR_CHILDREN = 5
WINDOW_KEEP_FRAC = 0.60
STABILITY_COSINE_FLOOR = 0.80
ASSIGN_CONCORDANCE_FLOOR = 0.60
MIN_COMPONENT_SIZE = 20
DENSITY_ARTIFACT_R = 0.70
