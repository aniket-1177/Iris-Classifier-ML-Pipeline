"""
Builds the Scikit-learn training pipeline.
Separates pipeline construction from training logic for testability.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def build_pipeline() -> Pipeline:
    """
    Construct an sklearn Pipeline with:
      - StandardScaler: zero-mean, unit-variance scaling
      - RandomForestClassifier: ensemble tree classifier

    Named steps allow GridSearchCV to target hyperparameters via
    the 'classifier__<param>' syntax.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline
