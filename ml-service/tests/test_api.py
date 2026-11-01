"""
Unit tests for main.py (FastAPI application)

NOTE ON TEST FIXTURES:
These mock fixtures test HTTP endpoints, input validation schemas, and unavailable
model error responses. They are software-verification mocks only.
"""

import copy
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

import main
from main import app
from predict import EnviroTrustPredictor
from train import run_training


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_model_status_untrained(client: TestClient, tmp_path: Path):
    """Verify GET /model/status indicates untrained state before data is provided."""
    # Ensure predictor points to an empty models directory
    main.predictor = EnviroTrustPredictor(models_dir=str(tmp_path / "empty_models"))
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_trained"] is False
    assert "Awaiting authentic" in data["message"]


def test_analyze_cod_returns_503_when_untrained(client: TestClient, tmp_path: Path):
    """Verify POST /analyze/cod returns 503 Service Unavailable when model is not trained."""
    main.predictor = EnviroTrustPredictor(models_dir=str(tmp_path / "empty_models"))

    payload = {
        "readingId": "READ-001",
        "sensorId": "COD-OUT-001",
        "actualCod": 15.0,
        "timestamp": "2026-10-31T10:00:00Z",
        "context": {
            "productionRate": 900.0,
            "wastewaterFlow": 800.0,
            "treatmentDose": 70.0,
            "operatingMode": "HIGH_LOAD",
            "energyConsumption": 950.0,
        },
    }

    response = client.post("/analyze/cod", json=payload)
    # CORE REQUIREMENT: HTTP 503 when model is not trained
    assert response.status_code == 503
    data = response.json()
    assert "Model is not trained" in data["detail"]


def test_analyze_cod_validation_error_on_missing_actual_cod(client: TestClient):
    """Verify POST /analyze/cod returns 422 if actualCod is missing."""
    malformed_payload = {
        "readingId": "READ-001",
        # 'actualCod' is missing
        "context": {
            "productionRate": 500.0,
        },
    }
    response = client.post("/analyze/cod", json=malformed_payload)
    assert response.status_code == 422


def test_analyze_cod_returns_200_when_trained(client: TestClient, tmp_path: Path):
    """
    Verify POST /analyze/cod returns 200 and structured predictions once a model
    is trained and loaded.
    """
    # 1. Generate functional test training data
    csv_file = tmp_path / "test_data.csv"
    models_dir = tmp_path / "models"
    cfg_file = tmp_path / "config.yaml"

    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "production_rate": np.random.uniform(400, 600, n),
        "wastewater_flow": np.random.uniform(400, 600, n),
        "treatment_dose": np.random.uniform(40, 60, n),
        "operating_mode": ["NORMAL"] * n,
        "energy_consumption": np.random.uniform(400, 600, n),
        "outlet_cod": np.random.uniform(100, 120, n),
    })
    df.to_csv(csv_file, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
column_mapping:
  production_rate: "production_rate"
  wastewater_flow: "wastewater_flow"
  treatment_dose: "treatment_dose"
  operating_mode: "operating_mode"
  energy_consumption: "energy_consumption"
  outlet_cod: "outlet_cod"
features:
  numerical:
    - "production_rate"
    - "wastewater_flow"
    - "treatment_dose"
    - "energy_consumption"
    - "outlet_cod"
  categorical:
    - "operating_mode"
preprocessing:
  numerical_imputation_strategy: "median"
  categorical_imputation_strategy: "most_frequent"
  categorical_encoding: "onehot"
  scale_features: false
isolation_forest:
  n_estimators: 20
  max_samples: 15
  random_state: 42
calibration:
  threshold_percentile: 90.0
  fallback_threshold: 0.5
artifacts:
  models_dir: "{str(models_dir)}"
  model_filename: "isolation_forest.joblib"
  preprocessor_filename: "preprocessor.joblib"
  metadata_filename: "model_metadata.json"
"""
    cfg_file.write_text(cfg_content)

    # 2. Run training to generate test model artifacts
    run_training(config_path=cfg_file)

    # 3. Reload predictor in API
    main.predictor = EnviroTrustPredictor(config_path=cfg_file, models_dir=str(models_dir))
    assert main.predictor.is_trained() is True

    # 4. Check model status endpoint
    status_resp = client.get("/model/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["is_trained"] is True

    # 5. Run inference request
    payload = {
        "readingId": "READ-TEST-001",
        "sensorId": "COD-OUT-001",
        "actualCod": 110.0,
        "timestamp": "2026-10-31T10:00:00Z",
        "context": {
            "productionRate": 500.0,
            "wastewaterFlow": 500.0,
            "treatmentDose": 50.0,
            "operatingMode": "NORMAL",
            "energyConsumption": 500.0,
        },
    }

    resp = client.post("/analyze/cod", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()

    assert res_data["reading_id"] == "READ-TEST-001"
    assert res_data["sensor_id"] == "COD-OUT-001"
    assert res_data["actual_cod"] == 110.0
    assert "anomaly_score" in res_data
    assert "anomaly_threshold" in res_data
    assert "is_anomaly" in res_data
    assert res_data["status"] in ["CONSISTENT", "SUSPICIOUS"]
    assert res_data["model_version"] == "iforest-cod-v1.0"


# ==============================================================================
# Demo Model & Exact Telemetry Schema Tests
# ==============================================================================

def test_demo_model_status_returns_200(client: TestClient):
    """Verify GET /demo/model/status returns 200 with complete demo model metadata."""
    resp = client.get("/demo/model/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_trained"] is True
    assert data["model_mode"] == "SYNTHETIC_DEMO"
    assert data["data_provenance"] == "SYNTHETIC"
    assert data["model_version"] == "iforest-demo-v1.0"
    assert "feature_schema" in data
    # 10 measurements plus one rolling-window (stuck-sensor) feature per measurement
    num_features = data["feature_schema"]["numerical_features"]
    assert len(num_features) == 20
    assert len([f for f in num_features if f.endswith("_roll_log_std")]) == 10
    assert "threshold" in data


def test_analyze_telemetry_exact_request_response_demo(client: TestClient):
    """
    Verify POST /demo/analyze/telemetry accepts the exact documented request schema
    and returns the exact documented response structure with all fields.
    """
    request_payload = {
        "readingId": "TEL-2026-001",
        "sensorId": "SENSOR-EFFLUENT-01",
        "timestamp": "2026-11-01T02:00:00Z",
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
        "context": {
            "productionRate": 900.0,
            "wastewaterFlow": 800.0,
            "operatingMode": "NORMAL",
            "energyConsumption": 950.0,
        },
    }

    resp = client.post("/demo/analyze/telemetry", json=request_payload)
    assert resp.status_code == 200
    data = resp.json()

    # Exact field assertions
    assert data["reading_id"] == "TEL-2026-001"
    assert data["sensor_id"] == "SENSOR-EFFLUENT-01"
    assert data["timestamp"] == "2026-11-01T02:00:00Z"
    assert isinstance(data["anomaly_score"], float)
    assert isinstance(data["anomaly_threshold"], float)
    assert isinstance(data["is_anomaly"], bool)
    assert isinstance(data["isolation_forest_flag"], bool)
    assert isinstance(data["rule_violations"], list)
    assert isinstance(data["detection_sources"], list)
    assert data["status"] in ["CONSISTENT", "SUSPICIOUS"]
    assert data["model_version"] == "iforest-demo-v1.0"
    assert data["model_mode"] == "SYNTHETIC_DEMO"
    assert data["data_provenance"] == "SYNTHETIC"
    assert isinstance(data["evaluated_features"], list)
    assert len(data["evaluated_features"]) == 20
    assert isinstance(data["rolling_window_filled"], bool)
    assert "evaluated_at_utc" in data


def test_analyze_telemetry_missing_features_returns_422(client: TestClient):
    """Verify POST /demo/analyze/telemetry returns 422 when required model features are omitted."""
    # Omit 'sulphate_mg_L' and 'chloride_mg_L'
    incomplete_payload = {
        "readingId": "TEL-INCOMPLETE-001",
        "measurements": {
            "COD_mg_L": 4800.0,
            "BOD_mg_L": 1200.0,
            "pH": 7.0,
            "TDS_mg_L": 2800.0,
            "turbidity_NTU": 250.0,
            "colour_PtCo": 2500.0,
            "total_solids_mg_L": 3100.0,
            "total_suspended_solids_mg_L": 300.0,
            # Missing chloride_mg_L, sulphate_mg_L
        },
    }

    resp = client.post("/demo/analyze/telemetry", json=incomplete_payload)
    assert resp.status_code == 422
    assert "Missing required model features" in resp.json()["detail"]


def test_analyze_telemetry_invalid_payload_returns_422(client: TestClient):
    """Verify POST /analyze/telemetry returns 422 for structurally malformed payloads."""
    # Missing 'measurements' dictionary
    malformed_payload = {
        "readingId": "TEL-BAD-001",
        "context": {"productionRate": 900.0},
    }
    resp = client.post("/analyze/telemetry", json=malformed_payload)
    assert resp.status_code == 422


def test_analyze_telemetry_authentic_untrained_returns_503(client: TestClient, tmp_path: Path):
    """Verify POST /analyze/telemetry in authentic mode returns 503 when model is untrained."""
    main.predictor = EnviroTrustPredictor(models_dir=str(tmp_path / "empty_dir"))
    payload = {
        "measurements": {"COD_mg_L": 5000.0, "pH": 7.0},
    }
    resp = client.post("/analyze/telemetry", json=payload)
    assert resp.status_code == 503
    assert "Model is not trained" in resp.json()["detail"]


def test_telemetry_input_immutability(client: TestClient):
    """Verify that incoming telemetry dictionary is never modified in-place by the ML service."""
    telemetry_input = {
        "readingId": "IMMUTABLE-001",
        "sensorId": "SENSOR-01",
        "timestamp": "2026-11-01T02:00:00Z",
        "measurements": {
            "COD_mg_L": 5000.0,
            "BOD_mg_L": 1250.0,
            "pH": 6.8,
            "TDS_mg_L": 2700.0,
            "turbidity_NTU": 220.0,
            "colour_PtCo": 2400.0,
            "total_solids_mg_L": 3000.0,
            "total_suspended_solids_mg_L": 280.0,
            "chloride_mg_L": 340.0,
            "sulphate_mg_L": 390.0,
        },
        "context": {
            "productionRate": 850.0,
            "operatingMode": "NORMAL",
        },
    }
    input_snapshot = copy.deepcopy(telemetry_input)

    resp = client.post("/demo/analyze/telemetry", json=telemetry_input)
    assert resp.status_code == 200

    # Ensure input_snapshot and telemetry_input remain strictly identical
    assert telemetry_input == input_snapshot


# ==============================================================================
# Backend Authentication Mechanism Tests
# ==============================================================================

def test_backend_authentication_open_when_env_not_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Verify service runs in open mode when no auth token is set in environment."""
    for var in ["ML_SERVICE_TOKEN", "ENVIROTRUST_ML_TOKEN", "ML_SERVICE_API_KEY", "X_ML_SERVICE_TOKEN"]:
        monkeypatch.delenv(var, raising=False)

    # Health check is 200 without auth
    assert client.get("/health").status_code == 200

    # Model status is 200 without auth
    assert client.get("/model/status").status_code == 200


def test_backend_authentication_enforced_when_token_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """
    Verify backend authentication when ML_SERVICE_TOKEN is set in environment.
    - Requests without token return 401 Unauthorized
    - Requests with invalid token return 401 Unauthorized
    - Requests with matching X-ML-Service-Token header succeed
    - Requests with matching Authorization: Bearer <token> succeed
    - Requests with matching X-API-Key succeed
    - /health endpoint remains unauthenticated (200 OK)
    """
    token_val = "envirotrust-backend-secret-xyz"
    monkeypatch.setenv("ML_SERVICE_TOKEN", token_val)

    # 1. Health check remains unauthenticated
    h_resp = client.get("/health")
    assert h_resp.status_code == 200

    # 2. Protected endpoint without token -> 401
    status_no_auth = client.get("/model/status")
    assert status_no_auth.status_code == 401
    assert "missing" in status_no_auth.json()["detail"].lower()

    # 3. Protected endpoint with invalid token -> 401
    status_bad_auth = client.get("/model/status", headers={"X-ML-Service-Token": "wrong-token"})
    assert status_bad_auth.status_code == 401
    assert "invalid" in status_bad_auth.json()["detail"].lower()

    # 4. Protected endpoint with correct X-ML-Service-Token -> 200
    status_good_auth = client.get("/model/status", headers={"X-ML-Service-Token": token_val})
    assert status_good_auth.status_code == 200

    # 5. Protected endpoint with Authorization: Bearer <token> -> 200
    status_bearer = client.get("/model/status", headers={"Authorization": f"Bearer {token_val}"})
    assert status_bearer.status_code == 200

    # 6. Protected endpoint with X-API-Key -> 200
    status_apikey = client.get("/model/status", headers={"X-API-Key": token_val})
    assert status_apikey.status_code == 200

    # 7. Telemetry endpoint with token
    payload = {
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
        }
    }
    # Without header -> 401
    tel_no_auth = client.post("/demo/analyze/telemetry", json=payload)
    assert tel_no_auth.status_code == 401

    # With header -> 200
    tel_with_auth = client.post("/demo/analyze/telemetry", json=payload, headers={"X-ML-Service-Token": token_val})
    assert tel_with_auth.status_code == 200
