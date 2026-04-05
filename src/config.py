"""
Central configuration for the ML pipeline.
All paths, hyperparameters, and settings live here.
"""

import os
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"

MODELS_DIR.mkdir(exist_ok=True)

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classifier")

# ── Model Artifact ───────────────────────────────────────────────────────────
MODEL_NAME = "iris_classifier"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ── Training Hyperparameters ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

HYPERPARAMETER_GRID = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
}

CV_FOLDS = 5

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = "Iris Classifier API"
API_VERSION = "1.0.0"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
