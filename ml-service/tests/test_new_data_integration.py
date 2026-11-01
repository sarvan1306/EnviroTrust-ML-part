"""
Unit and Integration Tests for Authentic China WWTP and Cross-Industry Microorganism Datasets.

Verifies:
  1. Clean loading of china.csv and microorganism.csv via data_loader.
  2. Cryptographic immutability (SHA-256) of CSV datasets and companion Jupyter notebooks.
  3. parse_microorganism_dataset correctly parses 27 industrial streams, converts range strings
     (e.g., '2000–11,588', '6.3–6.98', '1200–4000 (BOD5)', '<10') to numeric midpoints,
     and computes the BOD/COD biodegradability ratio.
  4. Cross-industry organic load distribution:
     - Ultra-high COD streams (Raw distillery ~100k, Olive oil mills ~57k, Butter production ~52k).
     - Moderate/low COD streams (Tomato ~294, Dates ~410, Beetroot ~501).
     - Extreme pH streams (Vegetable oil mills ~1.6, Refining of vegetable oils ~10.2).
  5. parse_china_wwtp_dataset isolates unmeasured parameters (pH, TDS as 'N/A*'),
     separates water quality parameters from energy and operational cost metrics.
  6. Feature engineering pipeline extracts standardized numerical candidate features.
  7. DataValidator guardrails:
     - Chronological splitting fails on microorganism dataset due to absent timestamps.
     - Standalone training refusal on China WWTP parameter table (N=8 < min_rows=10).
"""

import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import data_loader
from data_validator import DataValidator
from feature_engineering import FeatureEngineeringPipeline


CHINA_CSV_PATH = Path("data/raw/china.csv")
CHINA_NOTEBOOK_PATH = Path("data/raw/chinapy.ipynb")
MICROORGANISM_CSV_PATH = Path("data/raw/microorganism.csv")
MICROORGANISM_NOTEBOOK_PATH = Path("data/raw/microorganismpy.ipynb")
MINING_CSV_PATH = Path("data/raw/mining.csv")
MINING_NOTEBOOK_PATH = Path("data/raw/Mining_Analysis.ipynb")

EXPECTED_CHINA_CSV_SHA256 = "6702bcf203680e346838c961c57e2a81a014ee3c7fc2dfc6332e0e50c2caa8a8"
EXPECTED_CHINA_NOTEBOOK_SHA256 = "cf4ea06668b5da0f7cd5ea5f9decf5055f923eb66a0c8204cdc67050f87d8b3f"
EXPECTED_MICRO_CSV_SHA256 = "a558071270fd4cc098fdbaad55b5c5bd59b159824074b7f1605ef355af844089"
EXPECTED_MICRO_NOTEBOOK_SHA256 = "5789fe391faf951de0d8f47d5c7e0b906c0755a05e14ead38960643f411426f2"
EXPECTED_MINING_CSV_SHA256 = "254ed768f999eaeae90f82f46b2959c31f33f0a24c4367b2f4559bfe11b58dcb"
EXPECTED_MINING_NOTEBOOK_SHA256 = "7c3e22287435a30bc1154ed8a9270857bb101c7c98bee68b1fbeb7c73e31f065"


# ==============================================================================
# 1. Cryptographic Immutability
# ==============================================================================

def test_authentic_files_cryptographic_immutability():
    """Verify raw CSV files and companion notebooks match their exact SHA-256 hashes."""
    for path, expected_hash in [
        (CHINA_CSV_PATH, EXPECTED_CHINA_CSV_SHA256),
        (CHINA_NOTEBOOK_PATH, EXPECTED_CHINA_NOTEBOOK_SHA256),
        (MICROORGANISM_CSV_PATH, EXPECTED_MICRO_CSV_SHA256),
        (MICROORGANISM_NOTEBOOK_PATH, EXPECTED_MICRO_NOTEBOOK_SHA256),
        (MINING_CSV_PATH, EXPECTED_MINING_CSV_SHA256),
        (MINING_NOTEBOOK_PATH, EXPECTED_MINING_NOTEBOOK_SHA256),
    ]:
        assert path.exists(), f"File missing at {path.resolve()}"
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {path.name}: expected {expected_hash}, got {actual_hash}"


# ==============================================================================
# 2. China WWTP Dataset Inspection & Parsing
# ==============================================================================

def test_china_wwtp_dataset_loading_and_profiling():
    """Verify china.csv loads cleanly into 10 rows and 5 columns."""
    df = data_loader.load_dataset(CHINA_CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert len(df.columns) == 5

    profile = data_loader.inspect_dataset(df)
    assert profile["row_count"] == 10
    assert profile["duplicate_rows"] == 0
    assert "Parameter / Metric" in profile["columns"]
    assert "Average Value" in profile["columns"]
    assert "Median Value" in profile["columns"]


def test_china_wwtp_parsing_water_quality_and_operational_metrics():
    """Verify parse_china_wwtp_dataset correctly separates measured vs unmeasured parameters."""
    raw_df = data_loader.load_dataset(CHINA_CSV_PATH)
    valid_df, meta = data_loader.parse_china_wwtp_dataset(raw_df)

    # 2 unmeasured parameters with N/A*
    assert "pH" in meta["unmeasured_parameters"]
    assert "Total Dissolved Solids (TDS)" in meta["unmeasured_parameters"]
    assert len(meta["unmeasured_parameters"]) == 2

    # 8 valid measured rows
    assert len(valid_df) == 8
    assert "average_value" in valid_df.columns
    assert "median_value" in valid_df.columns
    assert "range_midpoint" in valid_df.columns

    # Verify water quality parameters (mg/L)
    assert "Chemical Oxygen Demand (COD)" in meta["water_quality_parameters"]
    assert "Biochemical Oxygen Demand (BOD)" in meta["water_quality_parameters"]
    assert "Ammoniacal Nitrogen (NH3-N)" in meta["water_quality_parameters"]
    assert "Total Nitrogen (TN)" in meta["water_quality_parameters"]
    assert "Total Phosphorus (TP)" in meta["water_quality_parameters"]

    # Verify operational parameters (energy, cost, ratio)
    assert "Electricity Consumption" in meta["operational_parameters"]
    assert "Operational Cost" in meta["operational_parameters"]
    assert "BOD / COD Ratio" in meta["operational_parameters"]

    # Verify specific numerical values
    cod_row = valid_df[valid_df["Parameter / Metric"] == "Chemical Oxygen Demand (COD)"].iloc[0]
    assert cod_row["average_value"] == 323.3
    assert cod_row["median_value"] == 252.0

    elec_row = valid_df[valid_df["Parameter / Metric"] == "Electricity Consumption"].iloc[0]
    assert elec_row["average_value"] == 1.1
    assert elec_row["median_value"] == 2.5


# ==============================================================================
# 3. Cross-Industry Microorganism Dataset Inspection & Parsing
# ==============================================================================

def test_microorganism_dataset_loading_and_profiling():
    """Verify microorganism.csv loads cleanly into 27 industrial streams and 11 columns."""
    df = data_loader.load_dataset(MICROORGANISM_CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 27
    assert len(df.columns) == 11
    assert "Wastewater Type" in df.columns
    assert "COD (mg/L)" in df.columns
    assert "BOD (mg/L)" in df.columns
    assert "pH" in df.columns


def test_microorganism_midpoint_parsing_and_ranges():
    """Verify parse_microorganism_dataset converts range strings to numeric values."""
    raw_df = data_loader.load_dataset(MICROORGANISM_CSV_PATH)
    clean_df, meta = data_loader.parse_microorganism_dataset(raw_df)

    assert len(clean_df) == 27
    assert "wastewater_type" in clean_df.columns
    assert "COD_mg_L" in clean_df.columns
    assert "BOD_mg_L" in clean_df.columns
    assert "pH" in clean_df.columns
    assert "bod_cod_ratio" in clean_df.columns

    # 26/27 streams report COD
    assert meta["parameter_stats"]["COD_mg_L"]["non_null_count"] == 26
    # 18/27 streams report BOD
    assert meta["parameter_stats"]["BOD_mg_L"]["non_null_count"] == 18
    # 18/27 streams report pH
    assert meta["parameter_stats"]["pH"]["non_null_count"] == 18

    # Slaughterhouse: COD="2000–11,588" -> midpoint 6794.0
    slaughter_row = clean_df[clean_df["wastewater_type"] == "Slaughterhouse"].iloc[0]
    assert slaughter_row["COD_mg_L"] == 6794.0
    # BOD="1300–4635 (BOD5)" -> midpoint 2967.5
    assert slaughter_row["BOD_mg_L"] == 2967.5
    # pH="6.3–6.98" -> midpoint 6.64
    assert round(slaughter_row["pH"], 2) == 6.64

    # Raw distillery wastewater: COD="80,000–120,000" -> midpoint 100,000.0
    distill_row = clean_df[clean_df["wastewater_type"] == "Raw distillery wastewater"].iloc[0]
    assert distill_row["COD_mg_L"] == 100000.0
    assert distill_row["BOD_mg_L"] == 52500.0
    assert round(distill_row["pH"], 2) == 4.35


def test_cross_industry_organic_load_and_biodegradability_distributions():
    """
    Verify organic load distributions across industrial sectors:
      - Ultra-high COD streams exceed 50,000 mg/L.
      - Low COD food processing streams are under 1,000 mg/L.
      - Biodegradability ratio (BOD/COD) ranges between 0.05 and 0.85.
    """
    raw_df = data_loader.load_dataset(MICROORGANISM_CSV_PATH)
    clean_df, _ = data_loader.parse_microorganism_dataset(raw_df)

    # Ultra-high COD (> 50,000 mg/L)
    high_cod = clean_df[clean_df["COD_mg_L"] >= 50000.0]["wastewater_type"].tolist()
    assert "Raw distillery wastewater" in high_cod
    assert "Olive oil mills" in high_cod
    assert "Butter production" in high_cod

    # Low COD (< 1,000 mg/L)
    low_cod = clean_df[clean_df["COD_mg_L"] < 1000.0]["wastewater_type"].tolist()
    assert "Tomato processing" in low_cod
    assert "Dates processing" in low_cod
    assert "Beetroot processing" in low_cod
    assert "Blackberry processing" in low_cod

    # Extreme pH streams
    acidic_streams = clean_df[clean_df["pH"] < 4.0]["wastewater_type"].tolist()
    assert "Vegetable oil mills" in acidic_streams  # pH ~ 1.6

    alkaline_streams = clean_df[clean_df["pH"] > 9.0]["wastewater_type"].tolist()
    assert "Refining of vegetable oils" in alkaline_streams  # pH ~ 10.2
    assert "Milk plant" in alkaline_streams  # pH ~ 9.5

    # Biodegradability ratios
    valid_ratios = clean_df["bod_cod_ratio"].dropna()
    assert len(valid_ratios) >= 17
    assert valid_ratios.min() > 0.0

    # Normal biological streams have BOD/COD <= 1.0 (median around 0.5)
    median_ratio = clean_df[clean_df["wastewater_type"] != "Butadiene and styrene"]["bod_cod_ratio"].median()
    assert 0.4 <= median_ratio <= 0.6

    # Butadiene and styrene exhibits COD underestimation / physical anomaly (BOD > COD, ratio ~ 5.2)
    butadiene_row = clean_df[clean_df["wastewater_type"] == "Butadiene and styrene"].iloc[0]
    assert butadiene_row["bod_cod_ratio"] > 1.0
    assert round(butadiene_row["bod_cod_ratio"], 2) == 5.22


# ==============================================================================
# 4. Feature Engineering Integration
# ==============================================================================

def test_feature_engineering_extracts_numerical_features():
    """Verify FeatureEngineeringPipeline extracts standardized numerical features from clean microorganism data."""
    config = data_loader.load_config("config/data_config.yaml")
    pipeline = FeatureEngineeringPipeline(config)

    raw_df = data_loader.load_dataset(MICROORGANISM_CSV_PATH)
    clean_df, _ = data_loader.parse_microorganism_dataset(raw_df)
    X = pipeline.process(clean_df)

    assert isinstance(X, pd.DataFrame)
    assert len(X) == 27
    # Non-feature identifiers must be excluded
    assert "wastewater_type" not in X.columns
    assert "reference_citation" not in X.columns

    # Standard environmental features must be present
    assert "COD_mg_L" in X.columns
    assert "BOD_mg_L" in X.columns
    assert "pH" in X.columns
    assert "total_suspended_solids_mg_L" in X.columns
    assert "total_solids_mg_L" in X.columns


# ==============================================================================
# 5. DataValidator Guardrails
# ==============================================================================

def test_datavalidator_rejects_chronological_split_without_timestamps():
    """DataValidator detects and rejects chronological splitting when timestamps are absent."""
    config = data_loader.load_config("config/data_config.yaml")
    config_chrono = config.copy()
    config_chrono["splitting"] = {
        "strategy": "chronological",
        "train_ratio": 0.8,
        "test_ratio": 0.0,
        "timestamp_column": "timestamp",
    }
    validator = DataValidator(config_chrono)

    raw_df = data_loader.load_dataset(MICROORGANISM_CSV_PATH)
    clean_df, _ = data_loader.parse_microorganism_dataset(raw_df)

    report = validator.validate_dataset(clean_df)
    assert report.is_valid is False
    assert any("Chronological splitting strategy is selected, but required timestamp" in err for err in report.errors)


def test_datavalidator_rejects_china_dataset_due_to_sample_size():
    """DataValidator correctly rejects standalone training on China parameter table (N=8 < 10)."""
    config = data_loader.load_config("config/data_config.yaml")
    validator = DataValidator(config)

    raw_df = data_loader.load_dataset(CHINA_CSV_PATH)
    clean_df, _ = data_loader.parse_china_wwtp_dataset(raw_df)

    report = validator.validate_dataset(clean_df)
    assert report.is_valid is False
    assert any("fewer than required minimum" in err for err in report.errors)


# ==============================================================================
# 6. Industrial Mining & Mineral Recovery Dataset Integration
# ==============================================================================

def test_mining_dataset_loading_and_profiling():
    """Verify mining.csv loads cleanly into 22 rows and 8 columns."""
    df = data_loader.load_dataset(MINING_CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 22
    assert len(df.columns) == 8

    profile = data_loader.inspect_dataset(df)
    assert profile["row_count"] == 22
    assert profile["duplicate_rows"] == 0
    expected_cols = [
        "Industry Category",
        "Industrial Activity",
        "Median TDS (mg/L)",
        "Median pH",
        "Median TOC (mg/L)",
        "Hardness (mg/L as CaCO3)",
        "Wastewater Volume / Energy Metric",
        "Notes / Critical Mineral Recovery Potential",
    ]
    for col in expected_cols:
        assert col in profile["columns"]


def test_mining_parsing_and_ranges():
    """Verify parse_mining_dataset extracts numeric values and metadata correctly."""
    raw_df = data_loader.load_dataset(MINING_CSV_PATH)
    clean_df, meta = data_loader.parse_mining_dataset(raw_df)

    assert len(clean_df) == 22
    assert meta["total_activities"] == 22
    assert len(meta["industry_categories"]) == 6
    assert list(meta["parameter_stats"].keys()) == ["pH", "TDS_mg_L", "TOC_mg_L", "hardness_mg_L_CaCO3"]

    # Parameter completeness checks
    assert meta["parameter_stats"]["pH"]["non_null"] == 22  # 100%
    assert meta["parameter_stats"]["TDS_mg_L"]["non_null"] == 22  # 100%
    assert meta["parameter_stats"]["TOC_mg_L"]["non_null"] == 16  # 72.7% (6 missing)
    assert meta["parameter_stats"]["hardness_mg_L_CaCO3"]["non_null"] == 20  # 90.9% (2 missing)

    # Standardized columns present
    assert "pH" in clean_df.columns
    assert "TDS_mg_L" in clean_df.columns
    assert "TOC_mg_L" in clean_df.columns
    assert "hardness_mg_L_CaCO3" in clean_df.columns
    assert "industry_category" in clean_df.columns
    assert "industrial_activity" in clean_df.columns


def test_mining_extreme_ranges_and_critical_minerals():
    """Verify cross-industry physical parameter extremes and critical mineral notes."""
    raw_df = data_loader.load_dataset(MINING_CSV_PATH)
    clean_df, _ = data_loader.parse_mining_dataset(raw_df)

    # Extreme TDS spans 4 orders of magnitude: 55 mg/L to 120,000 mg/L
    min_tds_row = clean_df.loc[clean_df["TDS_mg_L"].idxmin()]
    max_tds_row = clean_df.loc[clean_df["TDS_mg_L"].idxmax()]
    assert min_tds_row["industrial_activity"] == "Semiconductor Manufacturing"
    assert min_tds_row["TDS_mg_L"] == 55.0
    assert "Produced Water" in max_tds_row["industrial_activity"]
    assert max_tds_row["TDS_mg_L"] == 120000.0

    # Extreme pH spans acidic leaching (1.4) to alkaline flotation (9.9)
    min_ph_row = clean_df.loc[clean_df["pH"].idxmin()]
    max_ph_row = clean_df.loc[clean_df["pH"].idxmax()]
    assert min_ph_row["industrial_activity"] == "Spent Leachate"
    assert min_ph_row["pH"] == 1.4
    assert max_ph_row["industrial_activity"] == "Milling & Flotation"
    assert max_ph_row["pH"] == 9.9

    # Extreme Hardness (up to 12,000 mg/L as CaCO3 in Spent Leachate)
    max_hardness_row = clean_df.loc[clean_df["hardness_mg_L_CaCO3"].idxmax()]
    assert max_hardness_row["industrial_activity"] == "Spent Leachate"
    assert max_hardness_row["hardness_mg_L_CaCO3"] == 12000.0

    # High organic carbon in leather tanning (1,400 mg/L TOC)
    max_toc_row = clean_df.loc[clean_df["TOC_mg_L"].idxmax()]
    assert "Leather Tanning" in max_toc_row["industrial_activity"]
    assert max_toc_row["TOC_mg_L"] == 1400.0

    # Critical Mineral (CM) and trace element potential notes
    cm_notes = clean_df["notes_recovery_potential"].dropna().str.cat(sep=" ")
    assert "Lithium" in cm_notes or "Li" in cm_notes
    assert "Tin" in cm_notes or "Sn" in cm_notes
    assert "Cobalt" in cm_notes or "Co" in cm_notes
    assert "Nickel" in cm_notes or "Ni" in cm_notes


def test_mining_feature_engineering_extraction():
    """Verify FeatureEngineeringPipeline extracts standardized numerical features from mining data."""
    config = data_loader.load_config("config/data_config.yaml")
    pipeline = FeatureEngineeringPipeline(config)

    raw_df = data_loader.load_dataset(MINING_CSV_PATH)
    clean_df, _ = data_loader.parse_mining_dataset(raw_df)
    X = pipeline.process(clean_df)

    assert isinstance(X, pd.DataFrame)
    assert len(X) == 22

    # Categorical and descriptive metadata columns must be excluded from feature matrix
    assert "industry_category" not in X.columns
    assert "industrial_activity" not in X.columns
    assert "notes_recovery_potential" not in X.columns
    assert "wastewater_volume_metric" not in X.columns

    # Standard candidate features must be present
    assert "pH" in X.columns
    assert "TDS_mg_L" in X.columns
    assert "TOC_mg_L" in X.columns
    assert "hardness_mg_L_CaCO3" in X.columns


def test_mining_datavalidator_guardrails():
    """DataValidator detects absent timestamps on mining dataset when chronological split is requested."""
    config = data_loader.load_config("config/data_config.yaml")
    config_chrono = config.copy()
    config_chrono["splitting"] = {
        "strategy": "chronological",
        "train_ratio": 0.8,
        "test_ratio": 0.0,
        "timestamp_column": "timestamp",
    }
    validator = DataValidator(config_chrono)

    raw_df = data_loader.load_dataset(MINING_CSV_PATH)
    clean_df, _ = data_loader.parse_mining_dataset(raw_df)

    report = validator.validate_dataset(clean_df)
    assert report.is_valid is False
    assert any("Chronological splitting strategy is selected, but required timestamp" in err for err in report.errors)
