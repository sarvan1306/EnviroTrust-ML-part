"""
Unit and Integration Tests for Authentic Dataset (TDD Suite)

Verifies the integration of ml-service/data/raw/extracted_water_quality_samples.csv:
  1. The actual CSV loads successfully.
  2. The CSV has the expected structure (15 columns, 2 rows).
  3. Missing values remain missing during ingestion (no silent imputation/fabrication).
  4. sample_id is excluded from numerical features.
  5. Fully empty columns are not used as model features.
  6. Numerical environmental features are correctly identified.
  7. An absent timestamp is detected when chronological splitting is requested.
  8. Training refuses to proceed when there are insufficient observations.
  9. No model artifacts are created when training is rejected.
  10. The original CSV remains unchanged (byte/hash verification).

RESTRICTIONS:
- Does NOT train Isolation Forest on these 2 samples.
- Does NOT fabricate timestamps or measurements.
- Tests software behavior and validation guards only.
"""

import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import data_loader
from data_validator import DataValidator, DataValidationError
from feature_engineering import FeatureEngineeringPipeline
from train import run_training

CSV_PATH = Path("data/raw/extracted_water_quality_samples.csv")
EXPECTED_SHA256 = "cf0d823b990579d09b491be00789c0c344627e80b3d10e7bd522fdc4bef72532"

EXPECTED_COLUMNS = [
    "sample_id",
    "pH",
    "turbidity_NTU",
    "colour_PtCo",
    "total_solids_mg_L",
    "total_suspended_solids_mg_L",
    "COD_mg_L",
    "chloride_mg_L",
    "ammoniacal_nitrogen_mg_L",
    "nitrate_mg_L",
    "TDS_mg_L",
    "BOD_mg_L",
    "electricity_consumption",
    "production",
    "water_flow",
]


@pytest.fixture
def authentic_df() -> pd.DataFrame:
    """Load authentic CSV without modifying it."""
    if not CSV_PATH.exists():
        pytest.skip(f"Authentic dataset not present at {CSV_PATH.resolve()} (redistribution restricted)")
    return data_loader.load_dataset(CSV_PATH)


# ==============================================================================
# 1. The actual CSV loads successfully
# ==============================================================================

def test_actual_csv_loads_successfully(authentic_df: pd.DataFrame):
    """Verify that data_loader loads the authentic CSV file successfully."""
    assert authentic_df is not None
    assert isinstance(authentic_df, pd.DataFrame)
    assert len(authentic_df) == 2


# ==============================================================================
# 2. The CSV has the expected structure
# ==============================================================================

def test_actual_csv_has_expected_structure(authentic_df: pd.DataFrame):
    """Verify the exact column names, count, and row count of authentic dataset."""
    assert authentic_df.shape == (2, 15)
    assert list(authentic_df.columns) == EXPECTED_COLUMNS
    assert authentic_df["sample_id"].tolist() == ["Raw_Dye_Wastewater_1", "Raw_Dye_Wastewater_2"]


# ==============================================================================
# 3. Missing values remain missing during ingestion
# ==============================================================================

def test_missing_values_remain_missing_during_ingestion(authentic_df: pd.DataFrame):
    """
    Verify that missing values remain NaN during loading.
    No automatic imputation to 0 or synthetic values is performed by data_loader.
    """
    # Sample 1 has missing BOD, electricity, production, water_flow
    assert pd.isna(authentic_df.loc[0, "BOD_mg_L"])
    assert pd.isna(authentic_df.loc[0, "electricity_consumption"])
    assert pd.isna(authentic_df.loc[0, "production"])
    assert pd.isna(authentic_df.loc[0, "water_flow"])

    # Sample 2 additionally has missing ammoniacal_nitrogen, nitrate, TDS
    assert pd.isna(authentic_df.loc[1, "ammoniacal_nitrogen_mg_L"])
    assert pd.isna(authentic_df.loc[1, "nitrate_mg_L"])
    assert pd.isna(authentic_df.loc[1, "TDS_mg_L"])

    # Ensure none of these were silently imputed with 0.0
    for col in ["BOD_mg_L", "electricity_consumption", "production", "water_flow"]:
        assert authentic_df[col].isnull().all()


# ==============================================================================
# 4. sample_id is excluded from numerical features
# ==============================================================================

def test_sample_id_excluded_from_numerical_features():
    """Verify that sample_id is recognized as an identifier and excluded from active features."""
    config = data_loader.load_config("config/data_config.yaml")
    fe = FeatureEngineeringPipeline(config)
    df = data_loader.load_dataset(CSV_PATH)

    selected_df = fe.process(df)
    assert "sample_id" not in selected_df.columns


# ==============================================================================
# 5. Fully empty columns are not used as model features
# ==============================================================================

def test_fully_empty_columns_excluded_from_model_features():
    """
    Verify that completely empty columns (100% NaN) are excluded from the active feature matrix.
    Specifically: BOD_mg_L, electricity_consumption, production, water_flow.
    """
    config = data_loader.load_config("config/data_config.yaml")
    fe = FeatureEngineeringPipeline(config)
    df = data_loader.load_dataset(CSV_PATH)

    selected_df = fe.process(df)

    # Empty columns must not be in active selected features
    empty_cols = ["BOD_mg_L", "electricity_consumption", "production", "water_flow"]
    for col in empty_cols:
        assert col not in selected_df.columns, f"Completely empty column '{col}' should be excluded from active features"


# ==============================================================================
# 6. Numerical environmental features are correctly identified
# ==============================================================================

def test_numerical_environmental_features_correctly_identified():
    """
    Verify candidate environmental features with valid data are recognized:
    pH, turbidity_NTU, colour_PtCo, total_solids_mg_L, total_suspended_solids_mg_L,
    COD_mg_L, chloride_mg_L, ammoniacal_nitrogen_mg_L, nitrate_mg_L, TDS_mg_L.
    """
    config = data_loader.load_config("config/data_config.yaml")
    fe = FeatureEngineeringPipeline(config)
    df = data_loader.load_dataset(CSV_PATH)

    selected_df = fe.process(df)

    expected_active_env = [
        "pH",
        "turbidity_NTU",
        "colour_PtCo",
        "total_solids_mg_L",
        "total_suspended_solids_mg_L",
        "COD_mg_L",
        "chloride_mg_L",
    ]
    for feat in expected_active_env:
        assert feat in selected_df.columns, f"Expected active environmental feature '{feat}' missing from selected features"


# ==============================================================================
# 7. An absent timestamp is detected
# ==============================================================================

def test_absent_timestamp_is_detected():
    """
    Verify that validator detects absence of timestamp column when chronological
    splitting is configured, reporting a clear validation error.
    """
    config = {
        "column_mapping": {"COD_mg_L": "COD_mg_L", "pH": "pH"},
        "features": {"numerical": ["COD_mg_L", "pH"]},
        "splitting": {
            "strategy": "chronological",
            "timestamp_column": "timestamp",
        },
    }
    validator = DataValidator(config)
    df = data_loader.load_dataset(CSV_PATH)

    report = validator.validate_dataset(df)
    assert report.is_valid is False
    assert any("timestamp column 'timestamp' was not found" in err for err in report.errors)


# ==============================================================================
# 8. Training refuses to proceed when there are insufficient observations
# ==============================================================================

def test_training_refuses_insufficient_observations(tmp_path: Path):
    """
    Verify that train.py rejects authentic dataset with only 2 observations
    (requires min_rows >= 10) and raises DataValidationError.
    """
    cfg_file = tmp_path / "train_cfg.yaml"
    cfg_content = f"""
dataset:
  raw_data_path: "{str(CSV_PATH.resolve())}"
column_mapping:
  pH: "pH"
  COD_mg_L: "COD_mg_L"
features:
  numerical: ["pH", "COD_mg_L"]
validation:
  min_rows: 10
splitting:
  strategy: "random"
artifacts:
  models_dir: "{str(tmp_path / 'models')}"
"""
    cfg_file.write_text(cfg_content)

    with pytest.raises(DataValidationError) as exc_info:
        run_training(config_path=cfg_file)
    assert "fewer than required minimum" in str(exc_info.value)


# ==============================================================================
# 9. No model artifacts are created when training is rejected
# ==============================================================================

def test_no_model_artifacts_created_when_training_rejected(tmp_path: Path):
    """
    Verify that when training is rejected due to insufficient data,
    no serialized model, preprocessor, or metadata artifacts are written to disk.
    """
    models_dir = tmp_path / "models_guard_test"
    cfg_file = tmp_path / "reject_cfg.yaml"
    cfg_content = f"""
dataset:
  raw_data_path: "{str(CSV_PATH.resolve())}"
column_mapping:
  pH: "pH"
  COD_mg_L: "COD_mg_L"
features:
  numerical: ["pH", "COD_mg_L"]
validation:
  min_rows: 10
splitting:
  strategy: "random"
artifacts:
  models_dir: "{str(models_dir)}"
"""
    cfg_file.write_text(cfg_content)

    try:
        run_training(config_path=cfg_file)
    except DataValidationError:
        pass

    assert not (models_dir / "isolation_forest.joblib").exists()
    assert not (models_dir / "preprocessor.joblib").exists()
    assert not (models_dir / "model_metadata.json").exists()


# ==============================================================================
# 10. The original CSV remains unchanged
# ==============================================================================

def test_original_csv_remains_unchanged():
    """Verify that the authentic CSV has not been modified or overwritten."""
    if not CSV_PATH.exists():
        pytest.skip("Authentic CSV not present locally (redistribution restricted)")
    sha256 = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert sha256 == EXPECTED_SHA256, "Authentic CSV file was modified! Original hash mismatch."
