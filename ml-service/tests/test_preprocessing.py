"""
Unit tests for preprocessing.py

NOTE ON TEST FIXTURES:
These mock fixtures test numerical imputation, categorical encoding, and artifact
serialization. They are software-verification mocks only and do not represent authentic data.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from preprocessing import DataPreprocessor, PreprocessingError


@pytest.fixture
def prep_config() -> dict:
    return {
        "preprocessing": {
            "numerical_imputation_strategy": "median",
            "categorical_imputation_strategy": "most_frequent",
            "categorical_encoding": "onehot",
            "scale_features": False,
        },
        "features": {
            "numerical": ["production_rate", "outlet_cod"],
            "categorical": ["operating_mode"],
        },
    }


def test_fit_and_transform_imputation(prep_config: dict):
    preprocessor = DataPreprocessor(prep_config)

    # Train dataset: production_rate median is 500
    df_train = pd.DataFrame({
        "production_rate": [400.0, 500.0, 600.0],
        "outlet_cod": [100.0, 110.0, 120.0],
        "operating_mode": ["NORMAL", "NORMAL", "HIGH_LOAD"],
    })

    X_train = preprocessor.fit_transform(df_train)
    assert X_train.shape[0] == 3
    assert not np.isnan(X_train).any()

    # Test dataset with NaN in production_rate — should be imputed using train median (500.0)
    df_test = pd.DataFrame({
        "production_rate": [np.nan],
        "outlet_cod": [115.0],
        "operating_mode": ["NORMAL"],
    })

    X_test = preprocessor.transform(df_test)
    assert not np.isnan(X_test).any()
    # The first feature is production_rate; check imputed value equals train median (500.0)
    assert X_test[0, 0] == pytest.approx(500.0)


def test_unknown_category_handling_no_leakage(prep_config: dict):
    preprocessor = DataPreprocessor(prep_config)

    df_train = pd.DataFrame({
        "production_rate": [500.0, 600.0],
        "outlet_cod": [100.0, 120.0],
        "operating_mode": ["NORMAL", "HIGH_LOAD"],
    })
    preprocessor.fit(df_train)

    # New previously unseen category "UNKNOWN_MODE" should be safely handled with all zeros
    df_new = pd.DataFrame({
        "production_rate": [550.0],
        "outlet_cod": [110.0],
        "operating_mode": ["BRAND_NEW_MODE"],
    })

    X_new = preprocessor.transform(df_new)
    assert X_new.shape[0] == 1
    assert not np.isnan(X_new).any()


def test_save_and_load_preprocessor(prep_config: dict, tmp_path: Path):
    preprocessor = DataPreprocessor(prep_config)
    df_train = pd.DataFrame({
        "production_rate": [500.0, 600.0],
        "outlet_cod": [100.0, 120.0],
        "operating_mode": ["NORMAL", "HIGH_LOAD"],
    })
    preprocessor.fit(df_train)

    save_path = tmp_path / "test_preprocessor.joblib"
    preprocessor.save(save_path)
    assert save_path.exists()

    loaded = DataPreprocessor.load(save_path)
    assert loaded.is_fitted is True

    # Check that transform produces identical results
    test_sample = pd.DataFrame({
        "production_rate": [500.0],
        "outlet_cod": [100.0],
        "operating_mode": ["NORMAL"],
    })
    np.testing.assert_array_almost_equal(preprocessor.transform(test_sample), loaded.transform(test_sample))
