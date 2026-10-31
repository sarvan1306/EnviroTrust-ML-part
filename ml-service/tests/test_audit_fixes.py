"""
Unit and Integration Tests for Audit Fixes

Covers the 11 required audit verification areas:
  1. Environmental-only training schema (COD, BOD, pH, TDS, temp, turbidity).
  2. Process-context training schema (environmental + operational features).
  3. Missing optional columns (clean candidate subset handling).
  4. Missing mandatory features (raises error).
  5. Chronological split ordering (time-sorted without leakage).
  6. Label-aware normal-only training (anomalies excluded from Isolation Forest fit).
  7. String label conversion (normal vs anomaly mapped; rejects ambiguous values).
  8. Threshold calibration on normal validation samples.
  9. Generalized telemetry API (POST /analyze/telemetry).
  10. Inference using saved training schema (rejects missing required features).
  11. HTTP 503 when no model is available.

NOTE: All fixtures in this file are software-verification mocks used purely to test
code paths and error boundaries. They do NOT represent authentic industrial data.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

import main
from main import app
from anomaly_detection import IsolationForestDetector, ModelNotTrainedError
from data_validator import DataValidator, DataValidationError, parse_label_value
from preprocessing import DataPreprocessor, MissingFeatureError
from predict import EnviroTrustPredictor
from train import run_training, split_dataset


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


# ==============================================================================
# 1. Environmental-Only Training Schema
# ==============================================================================

def test_environmental_only_training_schema(tmp_path: Path):
    """
    Verify the pipeline can train using environmental-only features
    (COD, BOD, pH, TDS, Temperature, Turbidity) without requiring operational metrics.
    """
    csv_file = tmp_path / "env_only_data.csv"
    models_dir = tmp_path / "models_env"
    cfg_file = tmp_path / "config_env.yaml"

    np.random.seed(42)
    n = 35
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "outlet_cod": np.random.uniform(50, 150, n),
        "bod": np.random.uniform(20, 80, n),
        "ph": np.random.uniform(6.5, 8.5, n),
        "tds": np.random.uniform(300, 800, n),
        "temperature": np.random.uniform(20, 35, n),
        "turbidity": np.random.uniform(5, 50, n),
    })
    df.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  outlet_cod: "outlet_cod"
  bod: "bod"
  ph: "ph"
  tds: "tds"
  temperature: "temperature"
  turbidity: "turbidity"
features:
  numerical:
    - "outlet_cod"
    - "bod"
    - "ph"
    - "tds"
    - "temperature"
    - "turbidity"
  categorical: []
splitting:
  strategy: "chronological"
  train_ratio: 0.80
  timestamp_column: "timestamp"
artifacts:
  models_dir: "{str(models_dir)}"
"""
    cfg_file.write_text(cfg_content)

    manifest = run_training(config_path=cfg_file)
    assert (models_dir / "isolation_forest.joblib").exists()
    assert (models_dir / "preprocessor.joblib").exists()
    assert manifest["feature_schema"]["numerical_features"] == [
        "outlet_cod", "bod", "ph", "tds", "temperature", "turbidity"
    ]
    assert manifest["train_samples_fitted"] > 0


# ==============================================================================
# 2. Process-Context Training Schema
# ==============================================================================

def test_process_context_training_schema(tmp_path: Path):
    """
    Verify the pipeline trains with both environmental and industrial operational context.
    """
    csv_file = tmp_path / "process_context_data.csv"
    models_dir = tmp_path / "models_proc"
    cfg_file = tmp_path / "config_proc.yaml"

    np.random.seed(42)
    n = 35
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "outlet_cod": np.random.uniform(50, 150, n),
        "production_rate": np.random.uniform(300, 900, n),
        "wastewater_flow": np.random.uniform(200, 800, n),
        "treatment_dose": np.random.uniform(30, 90, n),
        "energy_consumption": np.random.uniform(400, 950, n),
        "operating_mode": np.random.choice(["NORMAL", "HIGH_LOAD"], n),
    })
    df.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  outlet_cod: "outlet_cod"
  production_rate: "production_rate"
  wastewater_flow: "wastewater_flow"
  treatment_dose: "treatment_dose"
  operating_mode: "operating_mode"
  energy_consumption: "energy_consumption"
features:
  numerical:
    - "outlet_cod"
    - "production_rate"
    - "wastewater_flow"
    - "treatment_dose"
    - "energy_consumption"
  categorical:
    - "operating_mode"
splitting:
  strategy: "chronological"
  train_ratio: 0.80
  timestamp_column: "timestamp"
artifacts:
  models_dir: "{str(models_dir)}"
"""
    cfg_file.write_text(cfg_content)

    manifest = run_training(config_path=cfg_file)
    assert (models_dir / "isolation_forest.joblib").exists()
    assert "operating_mode" in manifest["feature_schema"]["categorical_features"]
    assert "production_rate" in manifest["feature_schema"]["numerical_features"]


# ==============================================================================
# 3. Missing Optional Columns
# ==============================================================================

def test_missing_optional_columns(tmp_path: Path):
    """
    Verify that candidate features configured in data_config.yaml are optional.
    The pipeline must succeed on any subset of candidate features if at least one numerical is present.
    """
    csv_file = tmp_path / "subset_data.csv"
    models_dir = tmp_path / "models_subset"
    cfg_file = tmp_path / "config_subset.yaml"

    np.random.seed(42)
    n = 25
    # Only provides outlet_cod and ph (omits bod, tds, turbidity, production_rate, etc.)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "outlet_cod": np.random.uniform(60, 120, n),
        "ph": np.random.uniform(6.8, 7.8, n),
    })
    df.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  outlet_cod: "outlet_cod"
  ph: "ph"
  bod: "bod"                        # Configured but absent in CSV
  production_rate: "production_rate" # Configured but absent in CSV
features:
  numerical:
    - "outlet_cod"
    - "ph"
    - "bod"
    - "production_rate"
  categorical: []
splitting:
  strategy: "random"
  train_ratio: 0.80
artifacts:
  models_dir: "{str(models_dir)}"
"""
    cfg_file.write_text(cfg_content)

    manifest = run_training(config_path=cfg_file)
    # Model should have learned features present in dataset without error
    assert manifest["feature_schema"]["numerical_features"] == ["outlet_cod", "ph"]


# ==============================================================================
# 4. Missing Mandatory Features
# ==============================================================================

def test_missing_mandatory_features(tmp_path: Path):
    """
    Verify that if a feature is explicitly configured under mandatory_features,
    its absence raises DataValidationError.
    """
    csv_file = tmp_path / "missing_mand.csv"
    cfg_file = tmp_path / "config_mand.yaml"

    df = pd.DataFrame({
        "ph": [7.0, 7.1, 7.2, 7.3, 7.4],
    })
    df.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  ph: "ph"
  outlet_cod: "cod_column"
features:
  numerical: ["ph", "outlet_cod"]
  mandatory_features: ["outlet_cod"] # Explicitly mandatory
splitting:
  strategy: "random"
"""
    cfg_file.write_text(cfg_content)

    with pytest.raises(DataValidationError) as exc_info:
        run_training(config_path=cfg_file)
    assert "mandatory feature 'outlet_cod'" in str(exc_info.value)


# ==============================================================================
# 5. Chronological Split Ordering
# ==============================================================================

def test_chronological_split_ordering():
    """
    Verify chronological splitting:
      - Records are sorted by timestamp.
      - Train set receives earlier records.
      - Val set receives later records.
      - No random shuffling occurs.
    """
    # Create unordered time-series dataframe
    df = pd.DataFrame({
        "ts": ["2026-01-05", "2026-01-01", "2026-01-03", "2026-01-02", "2026-01-04"],
        "val": [50.0, 10.0, 30.0, 20.0, 40.0],
    })

    config = {
        "splitting": {
            "strategy": "chronological",
            "train_ratio": 0.60, # 3 train, 2 val
            "timestamp_column": "ts",
        }
    }

    train_df, val_df = split_dataset(df, config)

    # Train set should contain earliest 3 records (Jan 1, Jan 2, Jan 3)
    assert list(train_df["ts"]) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert list(train_df["val"]) == [10.0, 20.0, 30.0]

    # Val set should contain latest 2 records (Jan 4, Jan 5)
    assert list(val_df["ts"]) == ["2026-01-04", "2026-01-05"]
    assert list(val_df["val"]) == [40.0, 50.0]


def test_chronological_split_fails_on_missing_timestamp():
    """Verify that chronological splitting fails if timestamp column is absent."""
    df = pd.DataFrame({"cod": [10.0, 20.0, 30.0]})
    config = {
        "splitting": {
            "strategy": "chronological",
            "timestamp_column": "non_existent_ts",
        }
    }
    with pytest.raises(ValueError) as exc:
        split_dataset(df, config)
    assert "timestamp column 'non_existent_ts' not found" in str(exc.value)


# ==============================================================================
# 6. Label-Aware Normal-Only Training
# ==============================================================================

def test_label_aware_normal_only_training(tmp_path: Path):
    """
    Verify that when ground-truth labels exist:
      - Isolation Forest fits strictly on normal records (0).
      - Known anomalies (1) are excluded from model training.
    """
    csv_file = tmp_path / "labeled_data.csv"
    models_dir = tmp_path / "models_lbl"
    cfg_file = tmp_path / "config_lbl.yaml"

    np.random.seed(42)
    # 30 normal samples (cod ~ 100) + 10 anomalous samples (cod ~ 20)
    normals = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="h"),
        "outlet_cod": np.random.normal(100.0, 5.0, 30),
        "status_label": ["normal"] * 30,
    })
    anomalies = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-02", periods=10, freq="h"),
        "outlet_cod": np.random.normal(20.0, 2.0, 10),
        "status_label": ["tampered"] * 10,
    })
    df_all = pd.concat([normals, anomalies]).reset_index(drop=True)
    df_all.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  outlet_cod: "outlet_cod"
features:
  numerical: ["outlet_cod"]
  categorical: []
splitting:
  strategy: "random"
  train_ratio: 0.80
labels:
  label_column: "status_label"
  normal_values: ["normal"]
  anomaly_values: ["tampered"]
artifacts:
  models_dir: "{str(models_dir)}"
"""
    cfg_file.write_text(cfg_content)

    manifest = run_training(config_path=cfg_file)
    assert manifest["has_ground_truth_labels"] is True
    # The fitted train samples should be fewer than total 80% split because anomalies were filtered
    assert manifest["train_samples_fitted"] < int(len(df_all) * 0.8)


# ==============================================================================
# 7. String Label Conversion
# ==============================================================================

def test_string_label_conversion():
    """Verify parse_label_value handles standard strings, booleans, and rejects ambiguous values."""
    # Normal inputs
    assert parse_label_value("normal") == 0
    assert parse_label_value("NOMINAL") == 0
    assert parse_label_value("0") == 0
    assert parse_label_value(0) == 0
    assert parse_label_value(False) == 0

    # Anomaly inputs
    assert parse_label_value("anomaly") == 1
    assert parse_label_value("TAMPERED") == 1
    assert parse_label_value("suspicious") == 1
    assert parse_label_value("1") == 1
    assert parse_label_value(1) == 1
    assert parse_label_value(True) == 1

    # Ambiguous inputs should raise ValueError
    with pytest.raises(ValueError) as exc:
        parse_label_value("unlabeled_maybe_faulty")
    assert "Unrecognized or ambiguous label value" in str(exc.value)


# ==============================================================================
# 8. Threshold Calibration on Normal Validation Samples
# ==============================================================================

def test_threshold_calibration_on_normal_validation_samples():
    """Verify threshold calibration records validation sample count and percentile in metadata."""
    detector = IsolationForestDetector({
        "calibration": {"threshold_percentile": 95.0}
    })
    np.random.seed(42)
    X_train = np.random.normal(50.0, 2.0, (60, 2))
    X_val_normal = np.random.normal(50.0, 2.0, (40, 2))

    detector.fit(X_train)
    th = detector.calibrate_threshold(X_val_normal)

    assert detector.calibration_metadata["validation_sample_count"] == 40
    assert detector.calibration_metadata["threshold_percentile"] == 95.0
    assert detector.calibration_metadata["threshold"] == round(th, 4)


# ==============================================================================
# 9. Generalized Telemetry API
# ==============================================================================

def test_generalized_telemetry_api(test_client: TestClient, tmp_path: Path):
    """Verify POST /analyze/telemetry endpoint with dictionary of measurements."""
    # 1. Train model on COD and pH
    csv_file = tmp_path / "api_data.csv"
    models_dir = tmp_path / "api_models"
    cfg_file = tmp_path / "api_config.yaml"

    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "cod": np.random.uniform(80, 120, n),
        "ph": np.random.uniform(6.5, 7.5, n),
    })
    df.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  outlet_cod: "cod"
  ph: "ph"
features:
  numerical: ["outlet_cod", "ph"]
  categorical: []
splitting:
  strategy: "random"
artifacts:
  models_dir: "{str(models_dir)}"
"""
    cfg_file.write_text(cfg_content)
    run_training(config_path=cfg_file)

    # 2. Reload predictor
    main.predictor = EnviroTrustPredictor(config_path=cfg_file, models_dir=str(models_dir))
    assert main.predictor.is_trained() is True

    # 3. Call POST /analyze/telemetry
    payload = {
        "readingId": "TEL-2026-001",
        "sensorId": "ENV-STAT-09",
        "timestamp": "2026-10-31T12:00:00Z",
        "measurements": {
            "cod": 105.0,
            "ph": 7.1,
        },
    }

    resp = test_client.post("/analyze/telemetry", json=payload)
    assert resp.status_code == 200
    res = resp.json()
    assert res["reading_id"] == "TEL-2026-001"
    assert "anomaly_score" in res
    assert "anomaly_threshold" in res
    assert res["status"] in ["CONSISTENT", "SUSPICIOUS"]
    assert "evaluated_features" in res


# ==============================================================================
# 10. Inference Using Saved Schema Rejects Missing Required Features
# ==============================================================================

def test_inference_rejects_missing_required_features(test_client: TestClient, tmp_path: Path):
    """
    Verify that inference does not silently fabricate missing required features.
    If a model was trained on COD and pH, omitting pH must return HTTP 422.
    """
    csv_file = tmp_path / "schema_train.csv"
    models_dir = tmp_path / "schema_models"
    cfg_file = tmp_path / "schema_cfg.yaml"

    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "cod": np.random.uniform(80, 120, n),
        "ph": np.random.uniform(6.5, 7.5, n),
    })
    df.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  outlet_cod: "cod"
  ph: "ph"
features:
  numerical: ["outlet_cod", "ph"]
  categorical: []
splitting:
  strategy: "random"
artifacts:
  models_dir: "{str(models_dir)}"
"""
    cfg_file.write_text(cfg_content)
    run_training(config_path=cfg_file)

    main.predictor = EnviroTrustPredictor(config_path=cfg_file, models_dir=str(models_dir))

    # Provide only COD, missing required feature 'ph'
    incomplete_payload = {
        "readingId": "INCOMPLETE-001",
        "measurements": {
            "cod": 100.0,
            # 'ph' is missing
        },
    }

    resp = test_client.post("/analyze/telemetry", json=incomplete_payload)
    assert resp.status_code == 422
    assert "Missing required model features" in resp.json()["detail"]


# ==============================================================================
# 11. HTTP 503 When Model is Unavailable
# ==============================================================================

def test_http_503_when_no_model_available(test_client: TestClient, tmp_path: Path):
    """Verify that both /analyze/telemetry and /analyze/cod return HTTP 503 when untrained."""
    main.predictor = EnviroTrustPredictor(models_dir=str(tmp_path / "empty_dir"))
    assert main.predictor.is_trained() is False

    # Check /analyze/telemetry
    resp1 = test_client.post("/analyze/telemetry", json={"measurements": {"cod": 50.0}})
    assert resp1.status_code == 503
    assert "Model is not trained" in resp1.json()["detail"]

    # Check /analyze/cod
    resp2 = test_client.post("/analyze/cod", json={"actualCod": 50.0})
    assert resp2.status_code == 503
    assert "Model is not trained" in resp2.json()["detail"]
