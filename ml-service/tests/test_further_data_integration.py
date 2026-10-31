"""
Unit and Integration Tests for Further Authentic Data (Spreadsheet & Transposed V2)

Verifies the integration of:
  - ml-service/data/raw/data_spreadsheet_-_Sheet1.csv (Literature Parameter Table)
  - ml-service/data/raw/extracted_water_quality_samples_v2.csv (Integrated Observation Dataset)

Test Coverage:
  1. data_spreadsheet_-_Sheet1.csv loads successfully via data_loader.
  2. Dataset inspection profiles 13 parameter rows and 5 columns.
  3. transpose_parameter_table() transforms parameter rows to sample observation records.
  4. Testing methods metadata is accurately extracted from source table.
  5. Sample 1 (Textile Industrial Effluent) values match laboratory test results.
  6. Sample 2 (Dye-Intermediate Effluent) completes previously missing TDS, nitrogen, and nitrate.
  7. New environmental parameter sulphate_mg_L is verified for both samples.
  8. Unmeasured parameters (BOD, Electricity Consumption) remain NaN (no fabrication).
  9. extracted_water_quality_samples_v2.csv loads and validates correctly.
  10. Active environmental features now total 11 complete parameters (0% missing).
  11. Training continues to be safely halted (N=2 < min_rows=10).
  12. Immutability: SHA-256 hashes of authentic raw files remain intact.
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

RAW_SPREADSHEET_PATH = Path("data/raw/data_spreadsheet_-_Sheet1.csv")
RAW_V1_PATH = Path("data/raw/extracted_water_quality_samples.csv")
RAW_V2_PATH = Path("data/raw/extracted_water_quality_samples_v2.csv")

EXPECTED_SPREADSHEET_SHA256 = "d2df8ee865dd05476abf23b5acdce8a029d758c3867272181fe51ee4eca521a6"
EXPECTED_V1_SHA256 = "cf0d823b990579d09b491be00789c0c344627e80b3d10e7bd522fdc4bef72532"


@pytest.fixture
def raw_spreadsheet_df() -> pd.DataFrame:
    """Load authentic raw spreadsheet parameter table."""
    if not RAW_SPREADSHEET_PATH.exists():
        pytest.skip(f"Spreadsheet not present at {RAW_SPREADSHEET_PATH.resolve()} (redistribution restricted)")
    return data_loader.load_dataset(RAW_SPREADSHEET_PATH)


@pytest.fixture
def transposed_data(raw_spreadsheet_df: pd.DataFrame):
    """Transpose parameter table into sample observations and metadata."""
    return data_loader.transpose_parameter_table(raw_spreadsheet_df)


@pytest.fixture
def authentic_v2_df() -> pd.DataFrame:
    """Load integrated v2 observation dataset."""
    if not RAW_V2_PATH.exists():
        pytest.skip(f"V2 dataset not present at {RAW_V2_PATH.resolve()} (redistribution restricted)")
    return data_loader.load_dataset(RAW_V2_PATH)


# ==============================================================================
# 1. Raw Spreadsheet Ingestion
# ==============================================================================

def test_spreadsheet_loads_successfully(raw_spreadsheet_df: pd.DataFrame):
    """Verify that data_loader loads the authentic parameter spreadsheet."""
    assert raw_spreadsheet_df is not None
    assert isinstance(raw_spreadsheet_df, pd.DataFrame)
    assert raw_spreadsheet_df.shape == (13, 5)
    expected_cols = [
        "Parameter",
        "Textile Industrial Effluent",
        "Dye-Intermediate Effluent",
        "% Difference (Dye vs. Textile)",
        "Testing Method",
    ]
    assert list(raw_spreadsheet_df.columns) == expected_cols


# ==============================================================================
# 2. Spreadsheet Profiling
# ==============================================================================

def test_spreadsheet_inspection_profiling(raw_spreadsheet_df: pd.DataFrame):
    """Verify inspection report correctly profiles the 13 parameters."""
    profile = data_loader.inspect_dataset(raw_spreadsheet_df)
    assert profile["row_count"] == 13
    assert profile["column_count"] == 5
    assert profile["duplicate_rows"] == 0
    # BOD and Electricity Consumption have NaN in sample columns
    assert profile["missing_counts"]["Textile Industrial Effluent"] == 2
    assert profile["missing_counts"]["Dye-Intermediate Effluent"] == 2


# ==============================================================================
# 3. Parameter Transposition & Structure
# ==============================================================================

def test_transpose_parameter_table_structure(transposed_data):
    """Verify transpose_parameter_table transforms 13 parameter rows to 2 sample rows."""
    transposed_df, metadata = transposed_data
    assert isinstance(transposed_df, pd.DataFrame)
    assert len(transposed_df) == 2
    assert transposed_df["sample_id"].tolist() == [
        "Raw_Dye_Wastewater_1",
        "Raw_Dye_Wastewater_2",
    ]
    assert "sulphate_mg_L" in transposed_df.columns
    assert "COD_mg_L" in transposed_df.columns
    assert "TDS_mg_L" in transposed_df.columns


# ==============================================================================
# 4. Testing Method Metadata Extraction
# ==============================================================================

def test_testing_method_metadata_extracted(transposed_data):
    """Verify analytical testing methods are preserved in metadata."""
    _, metadata = transposed_data
    assert metadata["pH"]["testing_method"] == "Potentiometry"
    assert metadata["turbidity_NTU"]["testing_method"] == "Nephelometric"
    assert metadata["colour_PtCo"]["testing_method"] == "Platinum-Cobalt"
    assert metadata["total_solids_mg_L"]["testing_method"] == "Gravimetric"
    assert metadata["total_suspended_solids_mg_L"]["testing_method"] == "Gravimetric"
    assert metadata["COD_mg_L"]["testing_method"] == "Open Reflux"
    assert metadata["BOD_mg_L"]["testing_method"] == "Not Measured in Study"
    assert metadata["TDS_mg_L"]["testing_method"] == "Gravimetric"
    assert metadata["chloride_mg_L"]["testing_method"] == "Argentometric"
    assert metadata["ammoniacal_nitrogen_mg_L"]["testing_method"] == "Kjeldahl"
    assert metadata["nitrate_mg_L"]["testing_method"] == "Phenol Disulphonic Acid (PDA)"
    assert metadata["sulphate_mg_L"]["testing_method"] == "Turbidimetric"
    assert metadata["electricity_consumption"]["testing_method"] == "Raw Effluent Study (No Treatment Power Data)"


# ==============================================================================
# 5. Sample 1 (Textile Effluent) Quantitative Values
# ==============================================================================

def test_sample_1_textile_measurements(transposed_data):
    """Verify Sample 1 measurements match laboratory results."""
    transposed_df, _ = transposed_data
    s1 = transposed_df[transposed_df["sample_id"] == "Raw_Dye_Wastewater_1"].iloc[0]
    assert s1["pH"] == 6.0
    assert s1["turbidity_NTU"] == 554.0
    assert s1["colour_PtCo"] == 3462.0
    assert s1["total_solids_mg_L"] == 3042.0
    assert s1["total_suspended_solids_mg_L"] == 632.0
    assert s1["COD_mg_L"] == 4800.0
    assert s1["TDS_mg_L"] == 2410.0
    assert s1["chloride_mg_L"] == pytest.approx(299.91)
    assert s1["ammoniacal_nitrogen_mg_L"] == pytest.approx(8.44)
    assert s1["nitrate_mg_L"] == pytest.approx(101.8)
    assert s1["sulphate_mg_L"] == 295.0


# ==============================================================================
# 6. Sample 2 (Dye-Intermediate) Completion
# ==============================================================================

def test_sample_2_dye_intermediate_measurements_completed(transposed_data):
    """Verify Sample 2 newly completed TDS, nitrogen, nitrate, and sulphate values."""
    transposed_df, _ = transposed_data
    s2 = transposed_df[transposed_df["sample_id"] == "Raw_Dye_Wastewater_2"].iloc[0]
    assert s2["pH"] == 7.4
    assert s2["turbidity_NTU"] == 188.0
    assert s2["colour_PtCo"] == 1090.0
    assert s2["total_solids_mg_L"] == 3396.0
    assert s2["total_suspended_solids_mg_L"] == 146.0
    assert s2["COD_mg_L"] == 8800.0
    assert s2["chloride_mg_L"] == pytest.approx(499.85)

    # Completed values that were NaN in v1
    assert s2["TDS_mg_L"] == 3250.0
    assert s2["ammoniacal_nitrogen_mg_L"] == 34.0
    assert s2["nitrate_mg_L"] == pytest.approx(608.8)
    assert s2["sulphate_mg_L"] == 608.0


# ==============================================================================
# 7. Unmeasured Parameters Remain NaN (No Hallucination)
# ==============================================================================

def test_unmeasured_parameters_remain_nan(transposed_data):
    """Verify unmeasured BOD and electricity consumption remain NaN for both samples."""
    transposed_df, _ = transposed_data
    for _, row in transposed_df.iterrows():
        assert pd.isna(row["BOD_mg_L"]), f"BOD for {row['sample_id']} must be NaN"
        assert pd.isna(row["electricity_consumption"]), f"Electricity for {row['sample_id']} must be NaN"


# ==============================================================================
# 8. Integrated Observation Dataset (v2) Ingestion
# ==============================================================================

def test_v2_integrated_dataset_fidelity(authentic_v2_df: pd.DataFrame):
    """Verify extracted_water_quality_samples_v2.csv matches transposed parameters."""
    assert authentic_v2_df.shape == (2, 16)
    assert "sulphate_mg_L" in authentic_v2_df.columns
    assert authentic_v2_df.loc[0, "sulphate_mg_L"] == 295.0
    assert authentic_v2_df.loc[1, "sulphate_mg_L"] == 608.0
    assert authentic_v2_df.loc[1, "TDS_mg_L"] == 3250.0
    assert authentic_v2_df.loc[1, "ammoniacal_nitrogen_mg_L"] == 34.0
    assert authentic_v2_df.loc[1, "nitrate_mg_L"] == pytest.approx(608.8)


# ==============================================================================
# 9. 11 Complete Active Environmental Features
# ==============================================================================

def test_all_11_environmental_features_active_and_complete(authentic_v2_df: pd.DataFrame):
    """
    Verify that with the new data, 11 environmental features have 100% presence
    across both samples.
    """
    config = data_loader.load_config("config/data_config.yaml")
    pipeline = FeatureEngineeringPipeline(config)
    X = pipeline.process(authentic_v2_df)

    expected_active_features = [
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
        "sulphate_mg_L",
    ]
    for feat in expected_active_features:
        assert feat in X.columns, f"Feature '{feat}' should be active in feature matrix"
        assert not X[feat].isna().any(), f"Feature '{feat}' should have 0 missing values"

    # Completely empty columns must still be excluded
    assert "BOD_mg_L" not in X.columns
    assert "electricity_consumption" not in X.columns
    assert "sample_id" not in X.columns


# ==============================================================================
# 10. Training Refusal (Sample Size Guard N=2 < min_rows=10)
# ==============================================================================

def test_training_refusal_on_v2_due_to_sample_size(tmp_path: Path):
    """
    Verify that training on v2 is still safely rejected because N=2 < 10.
    Ensures model is NOT trained on 2 samples.
    """
    config = data_loader.load_config("config/data_config.yaml")
    config["dataset"]["raw_data_path"] = str(RAW_V2_PATH)
    config["splitting"]["strategy"] = "random"
    config["artifacts"]["models_dir"] = str(tmp_path / "models")

    with pytest.raises(DataValidationError, match="fewer than required minimum"):
        run_training(config)

    # Verify no artifacts were created
    models_dir = tmp_path / "models"
    assert not (models_dir / "isolation_forest.joblib").exists()
    assert not (models_dir / "preprocessor.joblib").exists()
    assert not (models_dir / "model_metadata.json").exists()


# ==============================================================================
# 11. File Immutability Verification
# ==============================================================================

def test_authentic_files_immutability():
    """Verify that both raw files have unchanged cryptographic hashes."""
    if not RAW_SPREADSHEET_PATH.exists() or not RAW_V1_PATH.exists():
        pytest.skip("Authentic raw research files not present locally (redistribution restricted)")
    def compute_sha(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    assert compute_sha(RAW_SPREADSHEET_PATH) == EXPECTED_SPREADSHEET_SHA256
    assert compute_sha(RAW_V1_PATH) == EXPECTED_V1_SHA256
