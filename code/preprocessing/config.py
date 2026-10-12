from pathlib import Path

# ── Code directories ────────────────────────────────────────────────────────
PREP_DIR = Path(__file__).resolve().parent          # code/preprocessing
CODE_DIR = PREP_DIR.parent                           # code

# ── Raw data (absolute paths on the analysis box) ───────────────────────────
DOCS_DIR = Path("/mnt/c/Users/lkuang/Documents")            # raw source files
BASE_DIR = Path("/mnt/c/Users/lkuang/ADHD_comorbidity_project")
CODEBOOK_XLSX = BASE_DIR / "codebook" / "sTable1.xlsx"       # single sheet: category (chapter) map
CONDITION_MAP_CSV = BASE_DIR / "codebook" / "sTable2.xlsx"   # granular condition map

TELEFORM_CSV = DOCS_DIR / "primary_teleform03202026.csv"
LABEL_CSV = DOCS_DIR / "NDD datasets 3-24-2026" / "chs_ndd_mental_03232026.csv"
INFLAM_PATH = DOCS_DIR / "inflam.xls"
ICD_RDS = DOCS_DIR / "kid_icdcode_data_Mar2026.RDS"
CHILD_BMI_CSV = DOCS_DIR / "BBC_underage20_HTWT_BMI_clean031626.csv"
SBWT_CSV = DOCS_DIR / "sbwt10032018.csv"
STRESS_XLSX = DOCS_DIR / "stress_data_alltress_data_all05202019_update.xlsx"
PM25_CSV = DOCS_DIR / "pm2.5_eachtrimester_GM_update1.csv"
DIET_XLSX = DOCS_DIR / "Mediterranean_diet_score.xlsx"
CONGENITAL_CSV = DOCS_DIR / "congenital_flag.csv"

# ── Per-stage results (a results/ folder inside each stage directory) ───────
PREP_RESULTS = PREP_DIR / "results"
ANALYSIS_RESULTS = CODE_DIR / "analysis" / "results"
CLUSTER_RESULTS = CODE_DIR / "clustering" / "results"
SEN_RESULTS = CODE_DIR / "sen" / "results"

# ── Derived analytic tables (preprocessing outputs) ─────────────────────────
PATIENTS_CSV = PREP_RESULTS / "ADHD_patients.csv"
PATIENTS_ANALYTIC_CSV = PREP_RESULTS / "ADHD_patients_analytic.csv"
EVENT_TABLE_CSV = PREP_RESULTS / "ADHD_event_table.csv"
UNMAPPED_CODES_CSV = PREP_RESULTS / "unmapped_codes.csv"

# ── Labels ──────────────────────────────────────────────────────────────────
# Two groups only, cohort-wide: any ADHD (with or without co-occurring ASD) vs NT.
DIAG_AGE_COL = "ADHD_firstdiag_years"

# Children excluded from the cohort (e.g., records with missing key values).
EXCLUDE_IDS = ["3490"]

# ── Analysis window ─────────────────────────────────────────────────────────
AGE_CUTOFF = 18          # years; events at/after this age are dropped
BIN_WIDTH_YEARS = 3      # post-diagnosis bin width
SEED = 42

# ── Standardization / strata ────────────────────────────────────────────────
MIN_STRATUM_N = 5        # min per-stratum N before pooling

# Race coding. Matches the ASD project: Non-Hispanic Black vs all others.
RACE_BLACK_CODES = [1, 5]

# ── NMF (Stage 2) ───────────────────────────────────────────────────────────
NMF_K = 3
NMF_K_SCAN = range(2, 7)
NMF_N_RESTARTS = 20
NMF_TOP_CONDITIONS = 5
NMF_MIN_ADHD_COUNT = 5
NMF_STABILITY_REPEATS = 50

# ── NMF developmental windows (Stage 2b) ────────────────────────────────────
NMF_MIN_OBSERVABLE = 5             # ADHD children carrying a condition for it to be recoverable

# ── Contrastive analysis (F5: shared vs ADHD-specific) ─────────────────────
NT_BASIS_K = 12
NT_BASIS_K_ROBUST = [8, 12, 16, 20]
CONTRAST_RESID_K_SCAN = [2, 3, 4, 5]   # forcing the residual to K=3 would itself
                                       # encourage correspondence with 3 profiles
CONTRAST_STABILITY_COSINE = 0.80       # floor the residual components must clear
CONTRAST_GAP_FLOOR = 0.15              # minimum ADHD - NT cosine gap

CONTRAST_MIN_RESID_Q = 0.50
CONTRAST_MIN_RESID_Q_SENS = [0.00, 0.50, 0.75]

CONTRAST_NULL_ITER = 1000
CONTRAST_NULL_RESTARTS = 3
CONTRAST_NULL_JOBS = -1            # -1 = all cores
NMF_BOOTSTRAP_B = 200
NMF_BOOTSTRAP_RESTARTS = 3
NMF_BOOTSTRAP_SEED = 20260831
NMF_SUBSAMPLE_FRAC = 0.8

def ensure(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
 