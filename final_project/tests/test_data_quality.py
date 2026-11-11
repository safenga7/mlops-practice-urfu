"""
Data Quality Test Suite for Final Project ML Pipeline
Validates data integrity, schema conformism, completeness, value bounds, and distribution.
"""

import os
import pytest
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "iris_dataset.csv")

@pytest.fixture(scope="module")
def dataset():
    """Loads the production dataset and asserts physical existence."""
    assert os.path.exists(DATA_PATH), f"Production dataset not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Production dataset is empty"
    return df

def test_dataset_schema(dataset):
    """Verifies that all required feature and target columns exist with expected types."""
    expected_columns = {
        "sepal_length": np.floating,
        "sepal_width": np.floating,
        "petal_length": np.floating,
        "petal_width": np.floating,
        "target": np.integer,
        "target_name": object
    }
    
    for col, expected_type in expected_columns.items():
        assert col in dataset.columns, f"Missing required column: {col}"
        if issubclass(expected_type, np.floating):
            assert pd.api.types.is_float_dtype(dataset[col]), f"Column {col} must be float, got {dataset[col].dtype}"
        elif issubclass(expected_type, np.integer):
            assert pd.api.types.is_integer_dtype(dataset[col]), f"Column {col} must be integer, got {dataset[col].dtype}"
        else:
            assert pd.api.types.is_string_dtype(dataset[col]) or pd.api.types.is_object_dtype(dataset[col]), f"Column {col} must be string/object, got {dataset[col].dtype}"

def test_dataset_completeness(dataset):
    """Verifies that there are zero missing or null values in production data."""
    null_counts = dataset.isnull().sum()
    assert null_counts.sum() == 0, f"Found unexpected missing values:\n{null_counts[null_counts > 0]}"

def test_feature_value_ranges(dataset):
    """Verifies that all numerical features reside within physiologically valid physical bounds."""
    bounds = {
        "sepal_length": (4.0, 8.5),
        "sepal_width": (1.5, 5.0),
        "petal_length": (0.5, 7.5),
        "petal_width": (0.05, 3.0),
    }
    for feature, (min_val, max_val) in bounds.items():
        assert dataset[feature].min() >= min_val, f"{feature} min value {dataset[feature].min()} < {min_val}"
        assert dataset[feature].max() <= max_val, f"{feature} max value {dataset[feature].max()} > {max_val}"

def test_target_cardinality_and_balance(dataset):
    """Verifies that the target classes are known and the dataset is balanced across classes."""
    valid_classes = {"setosa", "versicolor", "virginica"}
    observed_classes = set(dataset["target_name"].unique())
    assert observed_classes == valid_classes, f"Unexpected target classes: {observed_classes.symmetric_difference(valid_classes)}"
    
    class_counts = dataset["target_name"].value_counts()
    for cls_name, count in class_counts.items():
        assert count == 50, f"Class {cls_name} has {count} samples, expected 50 (unbalanced)"

def test_statistical_distribution(dataset):
    """Checks feature means against reference population baselines to prevent data drift."""
    expected_means = {
        "sepal_length": 5.84,
        "sepal_width": 3.05,
        "petal_length": 3.76,
        "petal_width": 1.20,
    }
    for feature, ref_mean in expected_means.items():
        actual_mean = dataset[feature].mean()
        assert abs(actual_mean - ref_mean) < 0.15, f"Data drift detected in {feature}: mean {actual_mean:.2f} deviates from expected {ref_mean}"
