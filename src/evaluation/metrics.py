"""
Evaluation metrics computation and reporting.
Decoupled from training so metrics logic can be tested independently.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth labels
        y_pred: Model predictions
        class_names: Ordered list of class name strings

    Returns:
        Dictionary of scalar metric values
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=class_names)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
    }

    for cls, f1 in zip(class_names, per_class_f1):
        metrics[f"f1_{cls}"] = round(float(f1), 4)

    return metrics


def log_metrics_table(metrics: Dict[str, float]) -> None:
    """Pretty-print metrics to the logger."""
    logger.info("─" * 45)
    logger.info("%-30s %s", "Metric", "Value")
    logger.info("─" * 45)
    for name, value in metrics.items():
        logger.info("%-30s %.4f", name, value)
    logger.info("─" * 45)


def get_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """Return sklearn's full classification report as a string."""
    return classification_report(y_true, y_pred, target_names=class_names)


def get_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """Return confusion matrix as a labelled DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    return pd.DataFrame(cm, index=class_names, columns=class_names)
