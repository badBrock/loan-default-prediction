"""Central configuration for paths, constants, and model params."""
from pathlib import Path

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw source files — UPDATE THESE to point at your data.
# (Originals lived under E:\senior_ds_test\... — kept configurable here.)
RAW_ACCOUNTS_TRAIN = RAW_DIR / "accounts_data_train.json"
RAW_ENQUIRY_TRAIN = RAW_DIR / "enquiry_data_train.json"
RAW_FLAG_TRAIN = RAW_DIR / "train_flag.csv"

RAW_ACCOUNTS_TEST = RAW_DIR / "accounts_data_test.json"
RAW_ENQUIRY_TEST = RAW_DIR / "enquiry_data_test.json"
RAW_FLAG_TEST = RAW_DIR / "test_flag.csv"

# Engineered feature outputs
FEATURES_TRAIN = PROCESSED_DIR / "features_train.csv"
FEATURES_TEST = PROCESSED_DIR / "features_test.csv"

# Model artifacts
BASELINE_MODEL_PATH = MODELS_DIR / "baseline.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Submission
SUBMISSION_PATH = PROCESSED_DIR / "final_submission.csv"

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
import pandas as pd

# Snapshot / application reference date (confirmed = max date across sources)
REF_DATE = pd.Timestamp("2021-01-01")

# Target / id / meta columns (never used as features)
TARGET_COL = "TARGET"
ID_COL = "uid"
META_COLS = [ID_COL, TARGET_COL, "NAME_CONTRACT_TYPE"]

RANDOM_STATE = 42
N_SPLITS = 5

# Credit types kept separate; the rest → "Other"
KEEP_CREDIT_TYPES = [
    "Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan",
]

# Enquiry types that looked "real"; rest → "Other"
KEEP_ENQ_TYPES = ["Cash loans", "Revolving loans"]

# ----------------------------------------------------------------------
# MODEL HYPERPARAMETERS (best from Optuna)
# ----------------------------------------------------------------------
BASELINE_XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", n_jobs=-1, random_state=RANDOM_STATE,
)

BEST_XGB_PARAMS = dict(
    n_estimators=481, max_depth=3, learning_rate=0.05961,
    subsample=0.74043, colsample_bytree=0.64632,
    min_child_weight=8, gamma=3.85739,
    reg_alpha=3.66946, reg_lambda=8.80245,
    eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1,
)


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)