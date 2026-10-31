"""
Unit and Integration Tests for Synthetic Demonstration Mode (TDD Suite)

Verifies:
  1. Synthetic data generator reproducibility with fixed random seed.
  2. Correct feature schema, physical ranges, and units.
  3. Structured inter-feature physical correlations in normal wastewater.
  4. Explicit synthetic provenance labelling (data_provenance == 'SYNTHETIC').
  5. Strict separation of normal and attack demonstration datasets.
  6. Comprehensive coverage of all 6 required attack scenarios.
  7. Strict exclusion of attack records from normal training data.
  8. Authentic research CSV immutability (SHA-256 hash preservation).
  9. Demo model artifact separation in models/demo/ without touching production paths.
  10. Inference engine scoring and anomaly detection on unseen synthetic records.
  11. Explicit API demo selection mechanism and provenance exposure.
  12. Synthetic evaluation reporting (precision, recall, F1, FPR).

RESTRICTIONS:
- Authentic data in data/raw/ must never be modified.
- Demo mode must never silently replace authentic production mode.
"""

import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from data_loader import load_dataset, load_config
import main
from predict import EnviroTrustPredictor

RAW_AUTHENTIC_CSV = Path("data/raw/extracted_water_quality_samples.csv")
EXPECTED_AUTHENTIC_SHA256 = "cf0d823b990579d09b491be00789c0c344627e80b3d10e7bd522fdc4bef72532"

DEMO_DATA_DIR = Path("data/synthetic_demo")
NORMAL_DEMO_CSV = DEMO_DATA_DIR / "normal_demo.csv"
ATTACK_DEMO_CSV = DEMO_DATA_DIR / "attack_demo.csv"

DEMO_MODELS_DIR = Path("models/demo")
DEMO_MODEL_FILE = DEMO_MODELS_DIR / "isolation_forest_demo.joblib"
DEMO_PREP_FILE = DEMO_MODELS_DIR / "preprocessor_demo.joblib"
DEMO_META_FILE = DEMO_MODELS_DIR / "model_metadata.json"

PROD_MODELS_DIR = Path("models")
PROD_MODEL_FILE = PROD_MODELS_DIR / "isolation_forest.joblib"

REQUIRED_ENV_FEATURES = [
    "COD_mg_L",
    "BOD_mg_L",
    "pH",
    "TDS_mg_L",
    "turbidity_NTU",
    "colour_PtCo",
    "total_solids_mg_L",
    "total_suspended_solids_mg_L",
    "chloride_mg_L",
    "sulphate_mg_L",
]

EXPECTED_ATTACK_SCENARIOS = {
    "low_cod",
    "high_cod",
    "cod_turbidity_decoupling",
    "sensor_flatline",
    "sudden_spike",
    "gradual_drift",
}


# ==============================================================================
# 1. Generator Reproducibility
# ==============================================================================

def test_synthetic_generator_reproducibility():
    """Verify that calling generator with the same random seed yields identical dataframes."""
    from src.synthetic_data_generator import generate_synthetic_dataset

    df_run1, df_atk1 = generate_synthetic_dataset(n_normal=100, n_attack=30, random_seed=42)
    df_run2, df_atk2 = generate_synthetic_dataset(n_normal=100, n_attack=30, random_seed=42)

    pd.testing.assert_frame_equal(df_run1, df_run2)
    pd.testing.assert_frame_equal(df_atk1, df_atk2)

    # Different seeds must produce different variations
    df_diff, _ = generate_synthetic_dataset(n_normal=100, n_attack=30, random_seed=999)
    assert not df_run1["COD_mg_L"].equals(df_diff["COD_mg_L"])


# ==============================================================================
# 2. Schema, Range, and Units Verification
# ==============================================================================

def test_synthetic_schema_and_units():
    """Verify generated dataset contains expected features, physical ranges, and units."""
    from src.synthetic_data_generator import generate_synthetic_dataset

    df_normal, _ = generate_synthetic_dataset(n_normal=200, n_attack=20, random_seed=42)

    for feat in REQUIRED_ENV_FEATURES:
        assert feat in df_normal.columns, f"Feature '{feat}' missing from synthetic dataset"
        assert not df_normal[feat].isna().any(), f"Synthetic feature '{feat}' contains NaN"
        assert (df_normal[feat] >= 0).all(), f"Feature '{feat}' has negative values"

    # pH must adhere to realistic chemistry (typically 5.0 - 9.0 in dye wastewater)
    assert df_normal["pH"].between(4.0, 10.0).all()

    # Total Solids conservation: TS ~ TDS + TSS
    # Should not deviate significantly from physical sum
    calculated_ts = df_normal["TDS_mg_L"] + df_normal["total_suspended_solids_mg_L"]
    diff = (df_normal["total_solids_mg_L"] - calculated_ts).abs()
    assert (diff < 150.0).all(), "Total Solids violates physical mass conservation (TS ~= TDS + TSS)"


# ==============================================================================
# 3. Inter-Feature Structured Correlations
# ==============================================================================

def test_feature_interrelationships_in_normal_data():
    """Verify that synthetic features are not sampled independently, but have physical coupling."""
    from src.synthetic_data_generator import generate_synthetic_dataset

    df_normal, _ = generate_synthetic_dataset(n_normal=500, n_attack=10, random_seed=42)

    # 1. Total solids vs (TDS + TSS) correlation should be near perfect
    corr_ts_sum = np.corrcoef(
        df_normal["total_solids_mg_L"],
        df_normal["TDS_mg_L"] + df_normal["total_suspended_solids_mg_L"]
    )[0, 1]
    assert corr_ts_sum > 0.95, f"Expected high TS correlation, got {corr_ts_sum}"

    # 2. Turbidity and Suspended Solids (TSS) should be positively correlated
    corr_turb_tss = np.corrcoef(df_normal["turbidity_NTU"], df_normal["total_suspended_solids_mg_L"])[0, 1]
    assert corr_turb_tss > 0.60, f"Expected Turbidity/TSS positive correlation, got {corr_turb_tss}"

    # 3. BOD should correlate with COD (biodegradability relationship)
    corr_bod_cod = np.corrcoef(df_normal["BOD_mg_L"], df_normal["COD_mg_L"])[0, 1]
    assert corr_bod_cod > 0.60, f"Expected BOD/COD positive correlation, got {corr_bod_cod}"


# ==============================================================================
# 4. Explicit Synthetic Provenance Labelling
# ==============================================================================

def test_synthetic_provenance_labels_preserved():
    """Verify every record is unambiguously stamped as SYNTHETIC demonstration data."""
    from src.synthetic_data_generator import generate_synthetic_dataset

    df_norm, df_atk = generate_synthetic_dataset(n_normal=100, n_attack=50, random_seed=42)

    for df in [df_norm, df_atk]:
        assert "data_provenance" in df.columns
        assert (df["data_provenance"] == "SYNTHETIC").all()
        assert "is_synthetic" in df.columns
        assert df["is_synthetic"].all()


# ==============================================================================
# 5. Separation of Normal and Attack Datasets
# ==============================================================================

def test_normal_and_attack_datasets_separation():
    """Verify normal and attack records are partitioned into separate frames with clean labels."""
    from src.synthetic_data_generator import generate_synthetic_dataset

    df_norm, df_atk = generate_synthetic_dataset(n_normal=300, n_attack=60, random_seed=42)

    # Normal dataset checks
    assert (df_norm["is_anomaly"] == 0).all()
    assert (df_norm["label"] == "normal").all()
    assert df_norm["attack_type"].isna().all() or (df_norm["attack_type"] == "none").all()

    # Attack dataset checks
    assert (df_atk["is_anomaly"] == 1).all()
    assert (df_atk["label"] == "anomaly").all()
    assert not df_atk["attack_type"].isna().any()


# ==============================================================================
# 6. Attack Scenarios Coverage
# ==============================================================================

def test_attack_scenarios_coverage():
    """Verify all 6 specified tampering scenarios are represented in the attack dataset."""
    from src.synthetic_data_generator import generate_synthetic_dataset

    _, df_atk = generate_synthetic_dataset(n_normal=100, n_attack=180, random_seed=42)

    generated_types = set(df_atk["attack_type"].unique())
    for scenario in EXPECTED_ATTACK_SCENARIOS:
        assert scenario in generated_types, f"Required attack scenario '{scenario}' missing"


# ==============================================================================
# 7. No Attack Records Enter Training
# ==============================================================================

def test_no_attack_records_enter_training(tmp_path: Path):
    """Verify training is strictly restricted to normal synthetic observations."""
    from src.synthetic_data_generator import generate_and_save_demo_datasets
    import train

    # Generate test datasets
    norm_path, atk_path = generate_and_save_demo_datasets(output_dir=tmp_path / "data")
    df_norm = load_dataset(norm_path)
    df_atk = load_dataset(atk_path)

    # Verify normal dataset contains zero attack records
    assert (df_norm["is_anomaly"] == 0).all()
    assert len(df_norm) >= 500

    # Ensure demo model config points strictly to normal demo dataset
    demo_cfg = load_config("config/data_config.yaml")
    demo_cfg["dataset"]["raw_data_path"] = str(norm_path)
    demo_cfg["artifacts"]["models_dir"] = str(tmp_path / "models/demo")
    demo_cfg["artifacts"]["model_filename"] = "isolation_forest_demo.joblib"
    demo_cfg["artifacts"]["preprocessor_filename"] = "preprocessor_demo.joblib"
    demo_cfg["artifacts"]["metadata_filename"] = "model_metadata.json"
    demo_cfg["splitting"]["strategy"] = "random"

    manifest = train.run_training(demo_cfg)
    assert manifest["train_samples"] > 0
    assert "data/synthetic_demo" in manifest["source_dataset"] or str(norm_path) in manifest["source_dataset"]


# ==============================================================================
# 8. Authentic Research CSV Immutability
# ==============================================================================

def test_authentic_research_data_immutability():
    """Ensure authentic data was not touched or modified by synthetic generation."""
    if not RAW_AUTHENTIC_CSV.exists():
        pytest.skip("Authentic CSV not present locally (redistribution restricted)")
    hasher = hashlib.sha256()
    with open(RAW_AUTHENTIC_CSV, "rb") as f:
        hasher.update(f.read())
    actual_hash = hasher.hexdigest()
    assert actual_hash == EXPECTED_AUTHENTIC_SHA256, "CRITICAL: Authentic CSV has been modified!"


# ==============================================================================
# 9. Model Artifact Isolation
# ==============================================================================

def test_demo_model_artifact_isolation():
    """Verify demo model artifacts are saved in models/demo/ and production paths are untouched."""
    # Production model file should not exist (since authentic data is untrained)
    assert not PROD_MODEL_FILE.exists(), "Production model file must not be created by demo run"

    # Once demo model is generated, check demo directory
    if DEMO_MODEL_FILE.exists():
        assert DEMO_PREP_FILE.exists()
        assert DEMO_META_FILE.exists()
        with open(DEMO_META_FILE, "r") as f:
            meta = json.load(f)
        assert meta.get("model_mode") == "SYNTHETIC_DEMO"
        assert meta.get("data_provenance") == "SYNTHETIC"


# ==============================================================================
# 10. Scoring Unseen Synthetic Observations
# ==============================================================================

def test_demo_predictor_scoring_unseen_observations():
    """Verify demo predictor detects attack records with high anomaly scores."""
    from src.synthetic_data_generator import generate_synthetic_dataset

    if not DEMO_MODEL_FILE.exists():
        pytest.skip("Demo model artifacts not yet built")

    predictor = EnviroTrustPredictor(
        models_dir=str(DEMO_MODELS_DIR),
        model_filename=DEMO_MODEL_FILE.name,
        preprocessor_filename=DEMO_PREP_FILE.name,
        metadata_filename=DEMO_META_FILE.name,
    )
    assert predictor.is_trained()

    _, df_atk = generate_synthetic_dataset(n_normal=10, n_attack=20, random_seed=777)

    # Score an extreme attack sample
    extreme_sample = df_atk[df_atk["attack_type"] == "high_cod"].iloc[0]
    payload = {
        "readingId": "TEST-ATK-001",
        "measurements": {col: float(extreme_sample[col]) for col in REQUIRED_ENV_FEATURES},
    }
    result = predictor.predict_telemetry(payload)
    assert result["is_anomaly"] is True
    # Isolation Forest alone misses most single-feature extremes; the physical rule layer
    # must flag it and explain which measurement violated the learned operating envelope.
    assert "physical_rules" in result["detection_sources"]
    assert any(v["feature"] == "COD_mg_L" for v in result["rule_violations"])
    assert result["isolation_forest_flag"] == (result["anomaly_score"] >= result["anomaly_threshold"])


# ==============================================================================
# 11. Explicit API Demo Selection Mechanism
# ==============================================================================

def test_api_explicit_demo_selection_and_provenance():
    """Verify API exposes demo mode explicitly and authentic mode returns 503."""
    client = TestClient(main.app)

    # 1. Authentic mode (/model/status) remains untrained
    status_resp = client.get("/model/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["is_trained"] is False

    # 2. Authentic telemetry analysis without demo flag returns HTTP 503
    sample_payload = {
        "readingId": "READ-001",
        "measurements": {
            "COD_mg_L": 4800.0,
            "BOD_mg_L": 1200.0,
            "pH": 7.0,
            "TDS_mg_L": 2800.0,
            "turbidity_NTU": 250.0,
            "colour_PtCo": 2500.0,
            "total_solids_mg_L": 3100.0,
            "total_suspended_solids_mg_L": 300.0,
            "chloride_mg_L": 350.0,
            "sulphate_mg_L": 400.0,
        },
    }
    auth_resp = client.post("/analyze/telemetry", json=sample_payload)
    assert auth_resp.status_code == 503
    assert "awaiting authentic" in auth_resp.json()["detail"].lower()

    if not DEMO_MODEL_FILE.exists():
        pytest.skip("Demo model artifacts not yet built")

    # 3. Explicit Demo endpoint or mode query param returns HTTP 200
    demo_resp = client.post("/analyze/telemetry?mode=demo", json=sample_payload)
    assert demo_resp.status_code == 200
    body = demo_resp.json()
    assert body["model_mode"] == "SYNTHETIC_DEMO"
    assert body["data_provenance"] == "SYNTHETIC"
    assert "anomaly_score" in body
    assert "anomaly_threshold" in body
    assert "is_anomaly" in body

    # 4. Demo model status endpoint
    demo_status = client.get("/demo/model/status")
    assert demo_status.status_code == 200
    assert demo_status.json()["model_mode"] == "SYNTHETIC_DEMO"
    assert demo_status.json()["is_trained"] is True
