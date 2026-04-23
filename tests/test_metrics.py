"""
Tests for src/evaluation/metrics.py

Run with:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    get_classification_report,
    get_confusion_matrix,
)

CLASSES = ["setosa", "versicolor", "virginica"]


def _make_perfect_labels():
    y_true = pd.Series(["setosa"] * 10 + ["versicolor"] * 10 + ["virginica"] * 10)
    y_pred = np.array(["setosa"] * 10 + ["versicolor"] * 10 + ["virginica"] * 10)
    return y_true, y_pred


def _make_imperfect_labels():
    y_true = pd.Series(["setosa"] * 10 + ["versicolor"] * 10 + ["virginica"] * 10)
    y_pred = np.array(
        ["setosa"] * 9 + ["versicolor"]           # 1 setosa misclassified
        + ["versicolor"] * 10
        + ["virginica"] * 10
    )
    return y_true, y_pred


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true, y_pred = _make_perfect_labels()
        metrics = compute_metrics(y_true, y_pred, CLASSES)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_returns_expected_keys(self):
        y_true, y_pred = _make_perfect_labels()
        metrics = compute_metrics(y_true, y_pred, CLASSES)
        expected_keys = {
            "accuracy", "macro_f1", "macro_precision", "macro_recall",
            "f1_setosa", "f1_versicolor", "f1_virginica",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_imperfect_accuracy(self):
        y_true, y_pred = _make_imperfect_labels()
        metrics = compute_metrics(y_true, y_pred, CLASSES)
        assert metrics["accuracy"] < 1.0
        assert metrics["accuracy"] == pytest.approx(29 / 30, abs=1e-3)

    def test_values_are_rounded(self):
        y_true, y_pred = _make_imperfect_labels()
        metrics = compute_metrics(y_true, y_pred, CLASSES)
        for key, val in metrics.items():
            assert val == round(val, 4), f"Metric '{key}' should be rounded to 4dp"

    def test_values_in_range(self):
        y_true, y_pred = _make_imperfect_labels()
        metrics = compute_metrics(y_true, y_pred, CLASSES)
        for val in metrics.values():
            assert 0.0 <= val <= 1.0


class TestConfusionMatrix:
    def test_shape(self):
        y_true, y_pred = _make_perfect_labels()
        cm = get_confusion_matrix(y_true, y_pred, CLASSES)
        assert cm.shape == (3, 3)

    def test_perfect_diagonal(self):
        y_true, y_pred = _make_perfect_labels()
        cm = get_confusion_matrix(y_true, y_pred, CLASSES)
        # assert (cm.values == np.diag(cm.values)).all()
        assert np.array_equal(cm.values, np.diag(np.diag(cm.values)))

    def test_labels(self):
        y_true, y_pred = _make_perfect_labels()
        cm = get_confusion_matrix(y_true, y_pred, CLASSES)
        assert list(cm.index) == CLASSES
        assert list(cm.columns) == CLASSES


class TestClassificationReport:
    def test_returns_string(self):
        y_true, y_pred = _make_perfect_labels()
        report = get_classification_report(y_true, y_pred, CLASSES)
        assert isinstance(report, str)
        assert "setosa" in report
        assert "precision" in report
