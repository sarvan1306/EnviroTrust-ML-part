"""
Unit and Integration Tests for Authentic Bangladesh Textile Industry Dataset

Verifies:
  1. bangladesh_textile_dataset.csv loads successfully via data_loader.
  2. Dataset structure: 15 factory rows, 9 columns.
  3. Feature engineering extracts 6 operational and environmental features:
     production, electricity_consumption, extracted_groundwater, water_flow, COD_mg_L, BOD_mg_L.
  4. Factory identifier is excluded from model numerical features.
  5. Constant string ranges (TDS '2000-3000*' and pH '6.0 - 8.5*') are excluded from numerical features.
  6. DataValidator passes with strategy='random' (N=15 >= min_rows=10).
  7. DataValidator detects and flags absent timestamp if strategy='chronological' is requested.
  8. Isolation Forest identifies outlier facilities (F12 mega-plant, F10 power outlier).
  9. File immutability: SHA-256 hash verification.
"""

import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import data_loader
from data_validator import DataValidator, DataValidationError
from feature_engineering import FeatureEngineeringPipeline
from anomaly_detection import IsolationForestDetector
from preprocessing import DataPreprocessor

DATASET_PATH = Path("data/raw/bangladesh_textile_dataset.csv")
EXPECTED_SHA256 = "daaa99510e68bcd97b2e8529b5d80706ade4c8da5221fb5d07bb8608280cdef3"


@pytest.fixture
def bangladesh_df() -> pd.DataFrame:
    """Load authentic Bangladesh textile dataset."""
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset missing at {DATASET_PATH.resolve()}")
    return data_loader.load_dataset(DATASET_PATH)


def test_bangladesh_dataset_loads_successfully(bangladesh_df: pd.DataFrame):
    """Verify clean ingestion and shape."""
    assert bangladesh_df is not None
    assert isinstance(bangladesh_df, pd.DataFrame)
    assert bangladesh_df.shape == (15, 9)
    assert "Factory" in bangladesh_df.columns
    assert "Total Production (kg)" in bangladesh_df.columns
    assert "COD (mg/L)" in bangladesh_df.columns
    assert "BOD (mg/L)" in bangladesh_df.columns


def test_bangladesh_dataset_profiling(bangladesh_df: pd.DataFrame):
    """Verify zero missing values and correct factory count."""
    profile = data_loader.inspect_dataset(bangladesh_df)
    assert profile["row_count"] == 15
    assert profile["column_count"] == 9
    assert profile["duplicate_rows"] == 0
    # Every factory has 100% complete data for all columns
    for col, missing_cnt in profile["missing_counts"].items():
        assert missing_cnt == 0, f"Column '{col}' unexpectedly has missing values"


def test_process_context_feature_extraction(bangladesh_df: pd.DataFrame):
    """Verify feature engineering extracts all 6 process-context and environmental features."""
    config = data_loader.load_config("config/data_config.yaml")
    pipeline = FeatureEngineeringPipeline(config)
    X = pipeline.process(bangladesh_df)

    expected_features = [
        "production",
        "electricity_consumption",
        "extracted_groundwater",
        "water_flow",
        "COD_mg_L",
        "BOD_mg_L",
    ]
    for feat in expected_features:
        assert feat in X.columns, f"Expected active feature '{feat}' not found in X"
        assert not X[feat].isna().any(), f"Feature '{feat}' has NaN values"

    # Identifier and constant ranges must be excluded
    assert "Factory" not in X.columns
    assert "TDS (mg/L)" not in X.columns
    assert "pH Range" not in X.columns


def test_validator_passes_under_random_split(bangladesh_df: pd.DataFrame):
    """Verify validator passes with strategy='random' since N=15 >= min_rows=10."""
    config = data_loader.load_config("config/data_config.yaml")
    config["splitting"]["strategy"] = "random"

    validator = DataValidator(config)
    report = validator.validate_dataset(bangladesh_df)
    assert report.is_valid, f"Validation failed unexpectedly: {report.errors}"
    assert len(report.errors) == 0


def test_validator_rejects_chronological_without_timestamps(bangladesh_df: pd.DataFrame):
    """Verify validator rejects chronological splitting because dataset lacks time column."""
    config = data_loader.load_config("config/data_config.yaml")
    config["splitting"]["strategy"] = "chronological"
    config["splitting"]["timestamp_column"] = "timestamp"

    validator = DataValidator(config)
    report = validator.validate_dataset(bangladesh_df)
    assert not report.is_valid
    assert any("timestamp" in err.lower() for err in report.errors)


def test_isolation_forest_outlier_detection_on_bangladesh_data(bangladesh_df: pd.DataFrame):
    """
    Verify Isolation Forest runs on the 15 records and flags F12 (mega-producer)
    and F10 (electricity outlier).
    """
    config = data_loader.load_config("config/data_config.yaml")
    pipeline = FeatureEngineeringPipeline(config)
    X = pipeline.process(bangladesh_df)

    preprocessor = DataPreprocessor(config)
    X_trans = preprocessor.fit_transform(X)

    iforest_config = {
        "isolation_forest": {
            "n_estimators": 300,
            "contamination": 0.13,
            "random_state": 42,
        }
    }
    detector = IsolationForestDetector(iforest_config)
    detector.fit(X_trans)

    scores = detector.compute_anomaly_scores(X_trans)
    assert len(scores) == 15

    # F12 is index 11 (highest production 41.5M kg, lowest COD 28 mg/L)
    # F10 is index 9 (highest electricity 65.5M kWh)
    f12_score = scores[11]
    f10_score = scores[9]

    # Both F12 and F10 should have positive anomaly scores (above decision boundary)
    assert f12_score > 0.0, f"Expected F12 to have high anomaly score, got {f12_score}"
    assert f10_score > 0.0, f"Expected F10 to have high anomaly score, got {f10_score}"


def test_bangladesh_csv_immutability():
    """Verify SHA-256 hash of bangladesh_textile_dataset.csv remains unchanged."""
    if not DATASET_PATH.exists():
        pytest.skip("Dataset file not present locally")
    sha256 = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    assert sha256 == EXPECTED_SHA256, "Bangladesh textile CSV hash mismatch!"
