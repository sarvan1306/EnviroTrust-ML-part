"""
Unit tests for evaluation.py

NOTE ON TEST FIXTURES:
These mock fixtures test evaluation metrics calculation (score distributions,
empirical FPR, precision, recall, ROC-AUC). They are software-verification
mocks only and do not represent authentic data.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from evaluation import evaluate_dataset
from train import run_training


def test_evaluate_dataset_supervised_and_unsupervised(tmp_path: Path):
    """
    Verify evaluation module computes score distributions and supervised
    metrics when a ground truth label column is present.
    """
    # 1. Generate functional test data
    train_csv = tmp_path / "eval_train.csv"
    eval_csv = tmp_path / "eval_test.csv"
    models_dir = tmp_path / "models"
    cfg_file = tmp_path / "eval_config.yaml"

    np.random.seed(42)
    n_train = 40
    df_train = pd.DataFrame({
        "production_rate": np.random.uniform(400, 600, n_train),
        "wastewater_flow": np.random.uniform(400, 600, n_train),
        "treatment_dose": np.random.uniform(40, 60, n_train),
        "operating_mode": ["NORMAL"] * n_train,
        "energy_consumption": np.random.uniform(400, 600, n_train),
        "outlet_cod": np.random.uniform(100, 120, n_train),
    })
    df_train.to_csv(train_csv, index=False)

    n_eval = 20
    df_eval = pd.DataFrame({
        "production_rate": np.random.uniform(400, 600, n_eval),
        "wastewater_flow": np.random.uniform(400, 600, n_eval),
        "treatment_dose": np.random.uniform(40, 60, n_eval),
        "operating_mode": ["NORMAL"] * n_eval,
        "energy_consumption": np.random.uniform(400, 600, n_eval),
        "outlet_cod": np.random.uniform(100, 120, n_eval),
        "ground_truth_anomaly": [0] * 18 + [1, 1],  # Labeled mock
    })
    df_eval.to_csv(eval_csv, index=False)

    cfg_content = f"""
dataset:
  raw_data_path: "{str(train_csv)}"
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

    # Train model
    run_training(config_path=cfg_file)

    # Run evaluation
    report = evaluate_dataset(
        data_path=eval_csv,
        config_path=cfg_file,
        models_dir=str(models_dir),
        label_column="ground_truth_anomaly",
    )

    assert "score_distribution" in report
    assert report["score_distribution"]["sample_count"] == n_eval
    assert "supervised_metrics" in report
    assert "precision" in report["supervised_metrics"]
    assert "recall" in report["supervised_metrics"]
    assert "f1_score" in report["supervised_metrics"]
    assert "confusion_matrix" in report["supervised_metrics"]
