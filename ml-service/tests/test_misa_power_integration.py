"""
Unit and Integration Tests for Ontario MISA 2024 Electric Power Generation Dataset

Verifies:
  1. misa_electric_power_generation_2024.csv and Untitled spreadsheet - Sheet3.csv load successfully.
  2. Dataset profiling and structural validation (8 rows, 8 columns, 7 facilities).
  3. pivot_eav_effluent_table correctly pivots long-format regulatory EAV data into facility observations.
  4. Physical parameter conversion: calculates effluent BOD concentration (mg/L) from load (kg/d) and flow (m3/d).
  5. Facility parameter isolation (Darlington, Pickering, Atikokan, Lennox, Bruce Power).
  6. Training refusal guardrail: verifies standalone MISA dataset is rejected for model training (N=8 < min_rows=10).
  7. Cryptographic file immutability (SHA-256 verification).
"""

import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import data_loader
from data_validator import DataValidator, DataValidationError
from feature_engineering import FeatureEngineeringPipeline

MISA_CSV_PATH = Path("data/raw/misa_electric_power_generation_2024.csv")
SHEET3_CSV_PATH = Path("data/raw/Untitled spreadsheet - Sheet3.csv")
NOTEBOOK_PATH = Path("data/raw/sheet3.ipynb")

EXPECTED_CSV_SHA256 = "525e197c99429bf38c4d56dea09d085bba885cdc6f0ef104c256a97659353bac"
EXPECTED_NOTEBOOK_SHA256 = "ed78dca642701c85dfebd3bfbab4f0406bd8055501e06a14824caa060a6ec5b1"


@pytest.fixture
def misa_df() -> pd.DataFrame:
    """Load authentic MISA 2024 power generation effluent monitoring dataset."""
    if not MISA_CSV_PATH.exists():
        pytest.skip(f"MISA dataset missing at {MISA_CSV_PATH.resolve()}")
    return data_loader.load_dataset(MISA_CSV_PATH)


# ==============================================================================
# 1. Loading & Profiling
# ==============================================================================

def test_misa_dataset_loads_successfully(misa_df: pd.DataFrame):
    """Verify CSV loads cleanly into DataFrame."""
    assert isinstance(misa_df, pd.DataFrame)
    assert len(misa_df) == 8
    assert len(misa_df.columns) == 8


def test_sheet3_alias_file_identical_to_misa_file():
    """Verify Untitled spreadsheet - Sheet3.csv is identical to misa_electric_power_generation_2024.csv."""
    if not SHEET3_CSV_PATH.exists():
        pytest.skip(f"Sheet3 alias missing at {SHEET3_CSV_PATH.resolve()}")
    df_alias = data_loader.load_dataset(SHEET3_CSV_PATH)
    df_misa = data_loader.load_dataset(MISA_CSV_PATH)
    pd.testing.assert_frame_equal(df_alias, df_misa)


def test_misa_dataset_profiling(misa_df: pd.DataFrame):
    """Verify column names, data types, and non-null counts."""
    profile = data_loader.inspect_dataset(misa_df)

    assert profile["row_count"] == 8
    assert profile["column_count"] == 8
    assert profile["duplicate_rows"] == 0

    expected_cols = [
        "Sector", "Works Name", "Control Point Name", "Parameter Name",
        "Sample Date", "Component Type", "Value", "Unit of Measure"
    ]
    for col in expected_cols:
        assert col in profile["columns"]

    # All 8 values are non-null
    for col in expected_cols:
        assert profile["missing_counts"][col] == 0

    # Sector is strictly Electric Power Generation
    assert (misa_df["Sector"] == "ELECTRIC POWER GENERATION").all()
    # Control Point is strictly Plant - Process Effluent
    assert (misa_df["Control Point Name"] == "PLANT - PROCESS EFFLUENT").all()
    # Sample Date is 2024/01/31
    assert (misa_df["Sample Date"] == "2024/01/31").all()


# ==============================================================================
# 2. EAV Pivoting & Feature Extraction
# ==============================================================================

def test_pivot_eav_effluent_table(misa_df: pd.DataFrame):
    """Verify pivot_eav_effluent_table reshapes long-format EAV into facility observations."""
    pivoted, metadata = data_loader.pivot_eav_effluent_table(misa_df)

    # 7 unique facilities represented
    assert len(pivoted) == 7
    assert "Works Name" in pivoted.columns
    assert "Sample Date" in pivoted.columns

    # Canonical parameters present in pivoted matrix
    expected_params = [
        "BOD_kg_d", "water_flow_m3_d", "aluminium_kg_d",
        "iron_kg_d", "residue_particulate_kg_d", "solvent_extractables_kg_d"
    ]
    for p in expected_params:
        assert p in pivoted.columns, f"Expected parameter '{p}' in pivoted columns"

    # Verify extracted metadata
    assert "BOD_kg_d" in metadata
    assert metadata["BOD_kg_d"]["unit"] == "KG/D"
    assert metadata["water_flow_m3_d"]["unit"] == "M3/D"


def test_bruce_power_effluent_concentration_calculation(misa_df: pd.DataFrame):
    """
    Verify Bruce Power Inc. - Development Services reports both BOD (kg/d) and Flow (m3/d),
    and calculate authentic effluent concentration in mg/L.
    """
    pivoted, _ = data_loader.pivot_eav_effluent_table(misa_df)

    bruce_row = pivoted[pivoted["Works Name"].str.contains("DEVELOPMENT SERVICES")]
    assert len(bruce_row) == 1

    bod_kg_d = float(bruce_row["BOD_kg_d"].iloc[0])
    flow_m3_d = float(bruce_row["water_flow_m3_d"].iloc[0])

    assert np.isclose(bod_kg_d, 3.5335)
    assert np.isclose(flow_m3_d, 1161.0)

    # Mass balance: Concentration (mg/L) = (Load kg/d * 1000 g/kg * 1000 mg/g) / (Flow m3/d * 1000 L/m3)
    # Conc (mg/L) = (Load * 1000) / Flow
    bod_mg_l = (bod_kg_d * 1000.0) / flow_m3_d
    assert np.isclose(bod_mg_l, 3.043497, atol=1e-4)
    # Verify effluent BOD is within typical treated outfall range (1 - 10 mg/L)
    assert 1.0 < bod_mg_l < 10.0


def test_other_power_facilities_parameters(misa_df: pd.DataFrame):
    """Verify parameters for Ontario Power Generation nuclear and thermal facilities."""
    pivoted, _ = data_loader.pivot_eav_effluent_table(misa_df)

    # Darlington Nuclear: Flow 2400 m3/d
    darlington = pivoted[pivoted["Works Name"].str.contains("DARLINGTON")]
    assert float(darlington["water_flow_m3_d"].iloc[0]) == 2400.0

    # Pickering Nuclear: Flow 3100 m3/d
    pickering = pivoted[pivoted["Works Name"].str.contains("PICKERING")]
    assert float(pickering["water_flow_m3_d"].iloc[0]) == 3100.0

    # Atikokan Generating Station: Particulate residue 5.21 kg/d
    atikokan = pivoted[pivoted["Works Name"].str.contains("ATIKOKAN")]
    assert float(atikokan["residue_particulate_kg_d"].iloc[0]) == 5.21

    # Lennox G.S.: Solvent extractables 1.2 kg/d
    lennox = pivoted[pivoted["Works Name"].str.contains("LENNOX")]
    assert float(lennox["solvent_extractables_kg_d"].iloc[0]) == 1.2


# ==============================================================================
# 3. Training Refusal Guardrails
# ==============================================================================

def test_misa_standalone_training_refusal_due_to_sample_size(misa_df: pd.DataFrame):
    """
    Verify DataValidator strictly rejects standalone training on MISA dataset:
    N=8 in raw form (or N=7 in pivoted form) is below min_rows=10.
    """
    config = data_loader.load_config("config/data_config.yaml")
    validator = DataValidator(config)

    # 1. Raw long format
    report_raw = validator.validate_dataset(misa_df)
    assert not report_raw.is_valid
    assert any("fewer than required minimum" in err for err in report_raw.errors)

    # 2. Pivoted format (N=7 < 10)
    pivoted, _ = data_loader.pivot_eav_effluent_table(misa_df)
    report_pivoted = validator.validate_dataset(pivoted)
    assert not report_pivoted.is_valid
    assert any("fewer than required minimum" in err for err in report_pivoted.errors)


# ==============================================================================
# 4. Cryptographic Immutability
# ==============================================================================

def test_misa_files_cryptographic_immutability():
    """Verify SHA-256 hashes of MISA dataset CSV and companion notebook."""
    if MISA_CSV_PATH.exists():
        csv_hash = hashlib.sha256(MISA_CSV_PATH.read_bytes()).hexdigest()
        assert csv_hash == EXPECTED_CSV_SHA256, f"MISA CSV hash mismatch: got {csv_hash}"

    if NOTEBOOK_PATH.exists():
        nb_hash = hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()
        assert nb_hash == EXPECTED_NOTEBOOK_SHA256, f"MISA notebook hash mismatch: got {nb_hash}"

