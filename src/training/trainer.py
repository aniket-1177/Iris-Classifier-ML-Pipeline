"""
Core training logic with MLflow experiment tracking.
Runs GridSearchCV, logs params/metrics/artifacts, and saves the best model.
"""

import logging
import pickle
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV

from src.config import (
    CV_FOLDS,
    HYPERPARAMETER_GRID,
    LABEL_ENCODER_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
)
from src.data.loader import get_label_encoder, load_dataset, split_data
from src.evaluation.metrics import compute_metrics, log_metrics_table
from src.training.pipeline import build_pipeline

logger = logging.getLogger(__name__)


def _configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    logger.info("MLflow experiment : %s", MLFLOW_EXPERIMENT_NAME)


def _save_artifact(obj: Any, path) -> None:
    """Pickle-save an object to disk."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved artifact → %s", path)


def run_training() -> dict[str, Any]:
    """
    End-to-end training run:
      1. Load & split data
      2. Build sklearn Pipeline
      3. GridSearchCV over hyperparameter grid
      4. Evaluate best model on held-out test set
      5. Log everything to MLflow
      6. Persist best model + label encoder to disk

    Returns:
        dict with best_params, test metrics, and run_id
    """
    _configure_mlflow()

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    label_encoder = get_label_encoder(y)

    # ── Pipeline + Search ─────────────────────────────────────────────────────
    pipeline = build_pipeline()
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=HYPERPARAMETER_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started | run_id=%s", run_id)

        # ── Train ─────────────────────────────────────────────────────────
        logger.info("Starting GridSearchCV (%d folds)...", CV_FOLDS)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # ── Log hyperparameters ───────────────────────────────────────────
        best_params = grid_search.best_params_
        mlflow.log_params(
            {
                "cv_folds": CV_FOLDS,
                "test_size": 0.2,
                **best_params,
            }
        )
        logger.info("Best params: %s", best_params)

        # ── Evaluate ──────────────────────────────────────────────────────
        y_pred = best_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, label_encoder.classes_)
        log_metrics_table(metrics)

        # Log scalar metrics to MLflow
        mlflow.log_metrics(
            {
                "cv_best_score": grid_search.best_score_,
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
            }
        )

        # ── Log model to MLflow registry ──────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="IrisClassifier",
            input_example=X_test.head(3),
        )

        # ── Save to local disk (for FastAPI) ──────────────────────────────
        _save_artifact(best_model, MODEL_PATH)
        _save_artifact(label_encoder, LABEL_ENCODER_PATH)

        # Log local paths as MLflow artifacts too
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="local_artifacts")
        mlflow.log_artifact(str(LABEL_ENCODER_PATH), artifact_path="local_artifacts")

        logger.info(
            "Training complete | accuracy=%.4f | macro_f1=%.4f | run_id=%s",
            metrics["accuracy"],
            metrics["macro_f1"],
            run_id,
        )

    return {
        "run_id": run_id,
        "best_params": best_params,
        "cv_best_score": grid_search.best_score_,
        "metrics": metrics,
        "model_path": str(MODEL_PATH),
    }
