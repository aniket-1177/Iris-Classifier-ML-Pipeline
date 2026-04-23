"""
Tests for src/inference/predictor.py

These tests train a tiny model on the fly so they don't depend on a
pre-existing models/ artifact — making them safe to run in CI.

Run with:
    pytest tests/test_predictor.py -v
"""

import pickle
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.inference.predictor import ModelNotFoundError, Predictor

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """
    Train a minimal pipeline and write pkl files to a temp directory.
    Returns (model_path, label_encoder_path).
    """
    tmp = tmp_path_factory.mktemp("models")

    # Minimal training data (just enough to fit a pipeline)
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
        [4.9, 3.0, 1.4, 0.2],
        [6.4, 3.2, 4.5, 1.5],
        [5.8, 2.7, 5.1, 1.9],
    ])
    y = ["setosa", "versicolor", "virginica", "setosa", "versicolor", "virginica"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
    ])
    pipeline.fit(X, y)

    le = LabelEncoder()
    le.fit(y)

    model_path = tmp / "iris_classifier.pkl"
    le_path = tmp / "label_encoder.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    return model_path, le_path


@pytest.fixture
def predictor(trained_artifacts):
    model_path, le_path = trained_artifacts
    with (
        patch("src.inference.predictor.MODEL_PATH", model_path),
        patch("src.inference.predictor.LABEL_ENCODER_PATH", le_path),
    ):
        yield Predictor()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPredictorLoading:
    def test_raises_when_model_missing(self, tmp_path):
        with (
            patch("src.inference.predictor.MODEL_PATH", tmp_path / "missing.pkl"),
            patch("src.inference.predictor.LABEL_ENCODER_PATH", tmp_path / "le.pkl"),
        ):
            with pytest.raises(ModelNotFoundError):
                Predictor()

    def test_model_classes(self, predictor):
        assert set(predictor.model_classes) == {"setosa", "versicolor", "virginica"}


class TestPredict:
    SAMPLE = [5.1, 3.5, 1.4, 0.2]  # typical setosa measurements

    def test_returns_expected_keys(self, predictor):
        result = predictor.predict(self.SAMPLE)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "class_probabilities" in result

    def test_predicted_class_is_valid(self, predictor):
        result = predictor.predict(self.SAMPLE)
        assert result["predicted_class"] in {"setosa", "versicolor", "virginica"}

    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict(self.SAMPLE)
        total = sum(result["class_probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_confidence_matches_max_prob(self, predictor):
        result = predictor.predict(self.SAMPLE)
        max_prob = max(result["class_probabilities"].values())
        assert result["confidence"] == pytest.approx(max_prob, abs=1e-4)

    def test_raises_on_wrong_feature_count(self, predictor):
        with pytest.raises(ValueError, match="Expected 4 features"):
            predictor.predict([1.0, 2.0])  # only 2 features


class TestPredictBatch:
    def test_batch_length(self, predictor):
        samples = [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 6.0, 2.5]]
        results = predictor.predict_batch(samples)
        assert len(results) == 2

    def test_each_result_has_class(self, predictor):
        samples = [[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4]]
        results = predictor.predict_batch(samples)
        for r in results:
            assert "predicted_class" in r
