from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"


# ============================================================
# RAW DATA PATHS
# ============================================================

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

TRAIN_IMAGE_DIR = RAW_DATA_DIR / "train"
TEST_IMAGE_DIR = RAW_DATA_DIR / "test"


# ============================================================
# PROCESSED DATA DIRECTORIES
# ============================================================

FEATURE_TABLE_DIR = PROCESSED_DATA_DIR / "feature_tables"
FOLDS_DIR = PROCESSED_DATA_DIR / "folds"
CACHED_FEATURES_DIR = PROCESSED_DATA_DIR / "cached_features"


# ============================================================
# MODEL DIRECTORIES
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"

BASELINE_RF_DIR = MODELS_DIR / "baseline_rf"
BASELINE_LGBM_DIR = MODELS_DIR / "baseline_lgbm"
CNN_MODEL_DIR = MODELS_DIR / "cnn"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_FOLDS = 5

# ============================================================
# FEATURE EXTRACTION CONFIGURATION
# ============================================================

# Pretrained CNN backbone used for image feature extraction
CNN_BACKBONE = "EfficientNetB0"

# EfficientNetB0 default input dimensions
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Number of images processed per batch during feature extraction
CNN_BATCH_SIZE = 32

# Output dimension after EfficientNetB0 GlobalAveragePooling2D
CNN_FEATURE_DIM = 1280

# ============================================================
# BASELINE MODEL CONFIGURATION
# ============================================================

# Ridge regression regularization strength
RIDGE_ALPHA = 1.0

# Random Forest configuration
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1
RF_N_JOBS = -1

# ============================================================
# VISUAL FEATURE CONFIGURATION
# ============================================================

VISUAL_FEATURE_DIR = PROCESSED_DATA_DIR / "visual_features"

VISUAL_IMAGE_SIZE = (512, 512)

COLOR_HIST_BINS = 16

GLCM_DISTANCES = [1, 3, 5]

GLCM_ANGLES = [0]

EXG_THRESHOLD = 0.0

# ============================================================
# ADVANCED MODEL DIRECTORIES
# ============================================================

ADVANCED_MODEL_DIR = MODELS_DIR / "advanced"

EXTRA_TREES_MODEL_DIR = ADVANCED_MODEL_DIR / "extra_trees"
ADVANCED_LGBM_MODEL_DIR = ADVANCED_MODEL_DIR / "lightgbm"
XGBOOST_MODEL_DIR = ADVANCED_MODEL_DIR / "xgboost"
CATBOOST_MODEL_DIR = ADVANCED_MODEL_DIR / "catboost"


# ============================================================
# DEEP LEARNING AND ENSEMBLE DIRECTORIES
# ============================================================

DEEP_LEARNING_MODEL_DIR = MODELS_DIR / "deep_learning"
ENSEMBLE_MODEL_DIR = MODELS_DIR / "ensemble"


# ============================================================
# FIGURE SUBDIRECTORIES
# ============================================================

EXPLORATION_FIGURES_DIR = FIGURES_DIR / "exploration"
BASELINE_FIGURES_DIR = FIGURES_DIR / "baseline"
VISUAL_FIGURES_DIR = FIGURES_DIR / "visual_features"
ADVANCED_FIGURES_DIR = FIGURES_DIR / "advanced_modeling"
DEEP_LEARNING_FIGURES_DIR = FIGURES_DIR / "deep_learning"
FINAL_EVALUATION_FIGURES_DIR = FIGURES_DIR / "final_evaluation"

# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

DIRECTORIES = [
    PROCESSED_DATA_DIR,
    FEATURE_TABLE_DIR,
    FOLDS_DIR,
    CACHED_FEATURES_DIR,
    MODELS_DIR,
    BASELINE_RF_DIR,
    BASELINE_LGBM_DIR,
    CNN_MODEL_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    SUBMISSIONS_DIR,
    VISUAL_FEATURE_DIR,
    ADVANCED_MODEL_DIR,
    EXTRA_TREES_MODEL_DIR,
    ADVANCED_LGBM_MODEL_DIR,
    XGBOOST_MODEL_DIR,
    CATBOOST_MODEL_DIR,
    DEEP_LEARNING_MODEL_DIR,
    ENSEMBLE_MODEL_DIR,
    EXPLORATION_FIGURES_DIR,
    BASELINE_FIGURES_DIR,
    VISUAL_FIGURES_DIR,
    ADVANCED_FIGURES_DIR,
    DEEP_LEARNING_FIGURES_DIR,
    FINAL_EVALUATION_FIGURES_DIR

]


def create_directories():
    """
    Create all required project directories if they do not exist.
    """
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()

    print("Project directories created successfully.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Train image directory: {TRAIN_IMAGE_DIR}")
    print(f"Test image directory: {TEST_IMAGE_DIR}")