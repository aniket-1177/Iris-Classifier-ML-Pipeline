"""
Model inference module.
Loads the trained model from disk and provides a clean predict interface.
This module is imported by the FastAPI application.
"""

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import LABEL_ENCODER_PATH, MODEL_PATH

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact cannot be found on disk."""


class Predictor:
    """
    Wraps the trained sklearn Pipeline and LabelEncoder.

    Usage:
        predictor = Predictor()
        result = predictor.predict([5.1, 3.5, 1.4, 0.2])
    """

    FEATURE_NAMES = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    def __init__(self) -> None:
        self._model = None
        self._label_encoder = None
        self._load()

    def _load(self) -> None:
        """Load model and label encoder from disk."""
        if not Path(MODEL_PATH).exists():
            raise ModelNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Run the training pipeline first: `python scripts/train.py`"
            )
        if not Path(LABEL_ENCODER_PATH).exists():
            raise ModelNotFoundError(
                f"Label encoder not found at '{LABEL_ENCODER_PATH}'."
            )

        with open(MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self._label_encoder = pickle.load(f)

        logger.info("Model loaded from %s", MODEL_PATH)

    def predict(self, features: List[float]) -> Dict:
        """
        Run inference on a single sample.

        Args:
            features: List of 4 floats [sepal_l, sepal_w, petal_l, petal_w]

        Returns:
            dict with predicted_class and class_probabilities
        """
        if len(features) != 4:
            raise ValueError(
                f"Expected 4 features, got {len(features)}."
            )

        X = pd.DataFrame([features], columns=self.FEATURE_NAMES)

        predicted_label = self._model.predict(X)[0]
        probabilities = self._model.predict_proba(X)[0]
        classes = self._label_encoder.classes_.tolist()

        return {
            "predicted_class": predicted_label,
            "confidence": round(float(np.max(probabilities)), 4),
            "class_probabilities": {
                cls: round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
            },
        }

    def predict_batch(self, batch: List[List[float]]) -> List[Dict]:
        """Run inference on multiple samples."""
        return [self.predict(sample) for sample in batch]

    @property
    def model_classes(self) -> List[str]:
        return self._label_encoder.classes_.tolist()


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """
    Singleton factory — the model is loaded once and cached.
    Use this in FastAPI dependency injection.
    """
    return Predictor()
