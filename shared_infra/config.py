# ── Paths (fixed to the real Linux compute-node locations) ──
# This mirrors BBC_Dataset/clustering/config.py but points EVENT_PATH / PATIENTS_PATH
# at the actual preprocessing/results/ dir, and OUT_DIR at gus_analysis/results.
_BASE = "<PROJECT_ROOT>/BBC_Dataset/preprocessing/results"
EVENT_PATH = f"{_BASE}/BBC_event_table.csv"
PATIENTS_PATH = f"{_BASE}/BBC_patients_analytic.csv"
OUT_DIR = "<PROJECT_ROOT>/gus_analysis/results"

# ── Data ──
AGE_CUTOFF = 18

# ── NMF ──
NMF_K = 3
NMF_N_RESTARTS = 20
NMF_TOP_CONDITIONS = 5
# Condition-selection rule for NMF input (see select_nmf_conditions).
NMF_MIN_ASD_COUNT = 5
NMF_MIN_PREV = 0.03
NMF_MIN_DIFF_PP = 5.0

NMF_BOOTSTRAP_B = 200
NMF_BOOTSTRAP_RESTARTS = 3
NMF_BOOTSTRAP_SEED = 20260824

# Disease categories the NMF condition filter is restricted to.
TARGET_CATEGORIES = [
    "Congenital", "Psychiatric", "Neurologic", "Sensory",
    "Endocrine - Metabolic", "Digestive", "Respiratory",
]

# ── Pre-registered decision rules (analysis_plan.md §2) ──
DENSITY_FLOOR_CHILDREN = 5      # a (condition x window) slice needs >= this many children
WINDOW_KEEP_FRAC = 0.60        # a window enters only if >=60% of retained conditions clear the floor
STABILITY_COSINE_FLOOR = 0.80  # bootstrap matched cosine >= this to be "reportable"
ASSIGN_CONCORDANCE_FLOOR = 0.60
MIN_COMPONENT_SIZE = 20        # component with < this many children is provisional
DENSITY_ARTIFACT_R = 0.70      # temporal factor with r > this vs per-window density is an artifact
