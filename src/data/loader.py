"""
Data loading and preprocessing module.
Handles fetching the dataset and splitting into train/test sets.
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset and return as a DataFrame.

    Returns:
        X: Feature DataFrame
        y: Target Series with string class labels
    """
    logger.info("Loading Iris dataset...")
    iris = load_iris(as_frame=True)

    X: pd.DataFrame = iris.frame.drop(columns=["target"])
    # Map integer targets → human-readable class names
    y: pd.Series = iris.frame["target"].map(
        dict(enumerate(iris.target_names))
    )

    logger.info(
        "Dataset loaded | samples=%d | features=%d | classes=%s",
        len(X),
        X.shape[1],
        list(y.unique()),
    )
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Data split | train=%d | test=%d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


def get_label_encoder(y: pd.Series) -> LabelEncoder:
    """Fit and return a LabelEncoder for the target classes."""
    le = LabelEncoder()
    le.fit(y)
    return le
