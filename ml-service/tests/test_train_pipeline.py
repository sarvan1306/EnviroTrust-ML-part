"""
Unit tests for train.py pipeline

NOTE ON TEST FIXTURES:
The dataset generated in this test file is used solely to verify pipeline execution
and artifact creation end-to-end. It is a temporary mock and cannot be used
to claim model accuracy.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from train import run_training, TrainingHaltedError


def test_train_halts_without_authentic_dataset(tmp_path: Path):
    """Verify that training strictly halts if no authentic dataset path is supplied."""
    cfg_path = tmp_path / "mock_config.yaml"
    cfg_content = """
dataset:
  raw_data_path: null
column_mapping: {}
features:
  numerical: []
  categorical: []
"""
    cfg_path.write_text(cfg_content)

    with pytest.raises(TrainingHaltedError) as exc_info:
        run_training(config_path=cfg_path, data_path=None)

    assert "TRAINING HALTED" in str(exc_info.value)
    assert "No authentic industrial dataset provided" in str(exc_info.value)


def test_train_halts_with_missing_file():
    """Verify that training raises FileNotFoundError if configured path does not exist."""
    fake_path = "/tmp/does_not_exist_at_all_12345.csv"
    with pytest.raises(FileNotFoundError):
        run_training(data_path=fake_path)


def test_train_succeeds_with_functional_dataset(tmp_path: Path):
    """
    Verify end-to-end training execution when a dataset is supplied.
    Tests artifact creation, threshold calibration, and metadata saving.
    """
    # 1. Create temporary functional test dataset
    csv_file = tmp_path / "functional_test_data.csv"
    np.random.seed(42)
    n_samples = 40
    data = {
        "prod_col": np.random.uniform(200, 800, n_samples),
        "flow_col": np.random.uniform(200, 700, n_samples),
        "treat_col": np.random.uniform(30, 80, n_samples),
        "mode_col": np.random.choice(["NORMAL", "HIGH_LOAD"], n_samples),
        "energy_col": np.random.uniform(300, 900, n_samples),
        "cod_col": np.random.uniform(80, 160, n_samples),
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    # 2. Config pointing to this dataset and temporary models directory
    models_dir = tmp_path / "models"
    cfg_file = tmp_path / "test_config.yaml"
    cfg_content = f"""
dataset:
  raw_data_path: "{str(csv_file)}"
  validation_data_path: null
column_mapping:
  production_rate: "prod_col"
  wastewater_flow: "flow_col"
  treatment_dose: "treat_col"
  operating_mode: "mode_col"
  energy_consumption: "energy_col"
  outlet_cod: "cod_col"
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
  n_estimators: 30
  max_samples: 20
  contamination: "auto"
  random_state: 42
calibration:
  threshold_percentile: 95.0
  fallback_threshold: 0.65
artifacts:
  models_dir: "{str(models_dir)}"
  model_filename: "isolation_forest.joblib"
  preprocessor_filename: "preprocessor.joblib"
  metadata_filename: "model_metadata.json"
"""
    cfg_file.write_text(cfg_content)

    manifest = run_training(config_path=cfg_file)

    # 3. Assertions
    assert (models_dir / "isolation_forest.joblib").exists()
    assert (models_dir / "preprocessor.joblib").exists()
    assert (models_dir / "model_metadata.json").exists()

    assert manifest["model_version"] == "iforest-cod-v1.0"
    assert manifest["train_samples"] > 0
    assert "calibrated_threshold" in manifest
    assert "validation_score_distribution" in manifest
