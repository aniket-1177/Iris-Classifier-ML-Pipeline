"""
Tests for src/data/loader.py

Run with:
    pytest tests/test_data_loader.py -v
"""

import pandas as pd

from src.data.loader import get_label_encoder, load_dataset, split_data


class TestLoadDataset:
    def test_returns_dataframe_and_series(self):
        X, y = load_dataset()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_correct_shape(self):
        X, y = load_dataset()
        assert X.shape == (150, 4), "Iris should have 150 rows and 4 features"
        assert len(y) == 150

    def test_feature_columns(self):
        X, _ = load_dataset()
        expected = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        assert list(X.columns) == expected

    def test_target_classes(self):
        _, y = load_dataset()
        assert set(y.unique()) == {"setosa", "versicolor", "virginica"}

    def test_no_nulls(self):
        X, y = load_dataset()
        assert not X.isnull().any().any(), "Feature matrix should have no nulls"
        assert not y.isnull().any(), "Target should have no nulls"


class TestSplitData:
    def setup_method(self):
        self.X, self.y = load_dataset()

    def test_split_sizes(self):
        X_train, X_test, y_train, y_test = split_data(self.X, self.y)
        assert len(X_train) == 120  # 80% of 150
        assert len(X_test) == 30    # 20% of 150

    def test_no_overlap(self):
        X_train, X_test, _, _ = split_data(self.X, self.y)
        combined_indices = set(X_train.index) | set(X_test.index)
        assert len(combined_indices) == 150, "Train and test sets must not overlap"

    def test_stratification(self):
        """Each class should appear in both splits."""
        _, X_test, _, y_test = split_data(self.X, self.y)
        assert set(y_test.unique()) == {"setosa", "versicolor", "virginica"}


class TestLabelEncoder:
    def test_encoder_classes(self):
        _, y = load_dataset()
        le = get_label_encoder(y)
        assert list(le.classes_) == ["setosa", "versicolor", "virginica"]

    def test_encoder_transform(self):
        _, y = load_dataset()
        le = get_label_encoder(y)
        encoded = le.transform(["setosa", "virginica"])
        assert list(encoded) == [0, 2]
