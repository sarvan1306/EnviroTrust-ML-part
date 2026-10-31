"""
Unit tests for data_validator.py

NOTE ON TEST FIXTURES:
These mock dataframes verify validation rules and error-handling branches.
They are software-verification mocks only and do not represent authentic data.
"""

import pytest
import pandas as pd
import numpy as np

from data_validator import DataValidator, DataValidationError, parse_label_value


@pytest.fixture
def base_config() -> dict:
    return {
        "column_mapping": {
            "production_rate": "prod_rate_col",
            "outlet_cod": "cod_col",
        },
        "features": {
            "numerical": ["production_rate", "outlet_cod"],
            "categorical": [],
            "mandatory_features": [],
        },
        "validation": {
            "min_rows": 3,
            "max_missing_ratio": 0.4,
        },
    }


def test_validation_passes_valid_data(base_config: dict):
    validator = DataValidator(base_config)
    df = pd.DataFrame({
        "prod_rate_col": [100.0, 200.0, 300.0, 400.0],
        "cod_col": [50.0, 60.0, 70.0, 80.0],
    })
    report = validator.validate_dataset(df)
    assert report.is_valid is True
    assert len(report.errors) == 0


def test_validation_fails_empty_data(base_config: dict):
    validator = DataValidator(base_config)
    df = pd.DataFrame(columns=["prod_rate_col", "cod_col"])
    report = validator.validate_dataset(df)
    assert report.is_valid is False
    assert any("0 rows" in err for err in report.errors)


def test_validation_fails_insufficient_rows(base_config: dict):
    validator = DataValidator(base_config)
    df = pd.DataFrame({
        "prod_rate_col": [100.0, 200.0],
        "cod_col": [50.0, 60.0],
    })
    report = validator.validate_dataset(df)
    assert report.is_valid is False
    assert any("fewer than required minimum" in err for err in report.errors)


def test_validation_passes_optional_missing_candidate_columns(base_config: dict):
    """Verify that candidate features are optional if at least one numerical feature is present."""
    validator = DataValidator(base_config)
    df = pd.DataFrame({
        "prod_rate_col": [100.0, 200.0, 300.0, 400.0],
        # 'cod_col' is an unmapped candidate feature, not mandatory
        "unrelated_col": [1, 2, 3, 4],
    })
    report = validator.validate_dataset(df)
    assert report.is_valid is True
    assert "outlet_cod" in report.missing_candidate_features


def test_validation_fails_missing_mandatory_feature():
    """Verify that specifically configured mandatory features must be present."""
    config = {
        "column_mapping": {
            "production_rate": "prod_rate_col",
            "outlet_cod": "cod_col",
        },
        "features": {
            "numerical": ["production_rate", "outlet_cod"],
            "categorical": [],
            "mandatory_features": ["outlet_cod"],  # Specifically mandatory
        },
        "validation": {"min_rows": 3, "max_missing_ratio": 0.5},
    }
    validator = DataValidator(config)
    df = pd.DataFrame({
        "prod_rate_col": [100.0, 200.0, 300.0, 400.0],
        # 'cod_col' is missing
    })
    report = validator.validate_dataset(df)
    assert report.is_valid is False
    assert "cod_col" in report.missing_columns
    assert any("mandatory feature 'outlet_cod'" in err for err in report.errors)


def test_validation_fails_zero_numerical_features():
    """Verify that validation fails if zero configured numerical features exist."""
    config = {
        "column_mapping": {
            "production_rate": "prod_rate_col",
            "outlet_cod": "cod_col",
        },
        "features": {
            "numerical": ["production_rate", "outlet_cod"],
            "categorical": [],
        },
        "validation": {"min_rows": 3, "max_missing_ratio": 0.5},
    }
    validator = DataValidator(config)
    df = pd.DataFrame({
        "completely_unrelated_col": [1.0, 2.0, 3.0, 4.0],
    })
    report = validator.validate_dataset(df)
    assert report.is_valid is False
    assert any("does not contain any configured numerical features" in err for err in report.errors)


def test_validation_fails_excessive_missing_values(base_config: dict):
    validator = DataValidator(base_config)
    df = pd.DataFrame({
        "prod_rate_col": [100.0, np.nan, np.nan, np.nan],  # 75% missing > 40% max
        "cod_col": [50.0, 60.0, 70.0, 80.0],
    })
    report = validator.validate_dataset(df)
    assert report.is_valid is False
    assert any("missing values" in err for err in report.errors)


def test_validate_feature_matrix():
    validator = DataValidator({})
    valid_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    validator.validate_feature_matrix(valid_mat)

    nan_mat = np.array([[1.0, np.nan], [3.0, 4.0]])
    with pytest.raises(DataValidationError) as exc_info:
        validator.validate_feature_matrix(nan_mat)
    assert "unhandled NaN" in str(exc_info.value)

    empty_mat = np.empty((0, 2))
    with pytest.raises(DataValidationError):
        validator.validate_feature_matrix(empty_mat)
