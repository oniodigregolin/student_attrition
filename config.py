"""Study settings. Every script reads from here, so changing a value here changes the study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "synthetic_data.xlsx"
OUTPUT_DIR = ROOT / "outputs"

# Temporal protocol
TRAIN_YEAR = 2023          # hyperparameter search: train
VALIDATION_YEAR = 2024     # hyperparameter search: validate
FINAL_TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025           # never used for any choice
YEARS = [2023, 2024, 2025]

# Population
EXCLUDED_SCHOOLS = [120, 260, 270]   # 120 joined in 2024; 260 and 270 only exist in 2023
CUTOFF_STAGE = 2                      # end of the 2nd assessment stage (late August)
LAST_INSTALMENT_BEFORE_CUTOFF = 8     # August
N_INSTALMENTS = 12

FEATURES = ["T1_MEAN", "T2_MEAN", "PCT_T2", "T2_BELOW6", "DECLINE_PTS", "DECLINE_PCT",
            "DECLINE_Z", "CLASS_SIZE", "REL_DECLINE", "NEW", "YEARS_NET", "UPPER_SEC",
            "AFTERNOON", "FULLTIME", "ENROL_MONTH", "NO_IMAGE_CONSENT",
            "SCHOOL_EXIT_RATE", "CHAPLAINCY"]

# Alternative predictor sets (robustness analyses; not run by default)
FEATURE_SETS = {
    "full": FEATURES,
    "without_image_consent": [f for f in FEATURES if f != "NO_IMAGE_CONSENT"],
    "without_religious": [f for f in FEATURES if f not in ("REL_DECLINE", "CHAPLAINCY")],
    "academic_admin_only": ["T1_MEAN", "T2_MEAN", "PCT_T2", "T2_BELOW6", "DECLINE_PTS",
                            "DECLINE_PCT", "DECLINE_Z", "NEW", "YEARS_NET"],
}

# Columns that are never predictors: LAST_BILLED only defines the cohort; ARREARS_TO_AUG feeds the
# additive baseline score; SCHOOL_EXIT_RATE_PAST is used only in a robustness analysis.
COHORT_COLUMNS = ["LAST_BILLED"]
AUXILIARY_COLUMNS = ["ARREARS_TO_AUG", "SCHOOL_EXIT_RATE_PAST"]
ID_COLUMNS = ["RA", "ANO", "SCHOOL_ID"]

# Models
MODELS = ["LR", "DT", "SVM", "RF", "XGB", "BRF", "RUSBoost"]
GENERAL_MODELS = ["LR", "DT", "SVM", "RF", "XGB"]
BUILTIN_MODELS = ["BRF", "RUSBoost"]
STRATEGIES = ["none", "weight", "RUS", "ROS", "SMOTE"]
N_SETTINGS = 25
SEEDS = [0, 1, 2, 3, 4]
BASE_SEED = 0
N_JOBS = 2
N_TREES = 300
NYSTROEM_COMPONENTS = 300

# Search spaces. Values are drawn in the order listed, which matters for reproducibility.
SEARCH_SPACES = {
    "LR": {"C": ("loguniform", 1e-3, 10)},
    "DT": {"max_depth": [2, 3, 4, 5, 6, 8, 10], "msl": [10, 20, 50, 100, 200]},
    "SVM": {"gamma": [0.01, 0.03, 0.1], "C": ("loguniform", 1e-4, 1)},
    "RF": {"max_depth": [4, 6, 8, 12, None], "msl": [5, 10, 20, 50], "mf": ["sqrt", 0.5]},
    "XGB": {"n": [200, 400, 800], "depth": [2, 3, 4, 5, 6], "lr": [0.01, 0.03, 0.05, 0.1],
            "mcw": [1, 5, 10, 20, 50], "ss": [0.6, 0.8, 1.0], "cs": [0.6, 0.8, 1.0]},
    "BRF": {"max_depth": [4, 6, 8, 12, None], "msl": [1, 5, 10, 20], "mf": ["sqrt", 0.5],
            "ratio": [0.25, 0.5, 1.0]},
    "RUSBoost": {"n": [50, 100, 200], "lr": [0.05, 0.1, 0.5, 1.0], "depth": [1, 2, 3],
                 "ratio": [0.25, 0.5, 1.0]},
}
STRATEGY_SPACES = {
    "weight": {"w_mult": [0.25, 0.5, 1.0], "w_mode": ["sqrt", "frac"]},
    "resampling": {"ratio_s": [0.05, 0.1, 0.25, 0.5, 1.0]},
    "smote": {"k": [3, 5, 10]},
}

# Evaluation
N_BOOT = 1000
BOOTSTRAP_SEED = 2025
LIST_SHARES = [0.005, 0.01, 0.02, 0.05, 0.10]
MAIN_LIST_SHARE = 0.05       # 1,171 students in 2025
SHORT_LIST_SHARE = 0.01      # 234 students in 2025
MIN_SCHOOL_POSITIVES = 3

# Rarity experiment
RARITY_LEVELS = [1.0, 0.5, 0.25, 0.10]
RARITY_REPS = 10
