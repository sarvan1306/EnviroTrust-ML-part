"""
Unit tests for main.py (FastAPI application)

NOTE ON TEST FIXTURES:
These mock fixtures test HTTP endpoints, input validation schemas, and unavailable
model error responses. They are software-verification mocks only.
"""

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
