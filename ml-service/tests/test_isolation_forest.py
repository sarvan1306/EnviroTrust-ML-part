"""
Unit tests for anomaly_detection.py

NOTE ON TEST FIXTURES:
These synthetic matrices verify mathematical and algorithmic behavior of the
Isolation Forest wrapper. They are software-verification mocks only and do not
represent authentic industrial data.
"""

from pathlib import Path
import pytest
import numpy as np

from anomaly_detection import IsolationForestDetector, ModelNotTrainedError


@pytest.fixture
def mock_detector_config() -> dict:
    return {
        "isolation_forest": {
            "n_estimators": 50,
            "max_samples": 30,
            "contamination": "auto",
            "random_state": 42,
        },
        "calibration": {
            "threshold_percentile": 95.0,
            "fallback_threshold": 0.5,
        },
    }


def test_model_not_trained_error():
    detector = IsolationForestDetector()
    dummy_input = np.array([[1.0, 2.0]])
    with pytest.raises(ModelNotTrainedError):
        detector.compute_anomaly_scores(dummy_input)


def test_fit_and_higher_score_for_outliers(mock_detector_config: dict):
    detector = IsolationForestDetector(mock_detector_config)

    # Synthetic normal cluster centered around [10.0, 10.0]
    np.random.seed(42)
    X_normal = np.random.normal(loc=10.0, scale=0.5, size=(60, 2))
    detector.fit(X_normal)

    # Inlier sample close to center
    inlier = np.array([[10.1, 9.9]])
    # Extreme outlier sample far away from center
    outlier = np.array([[100.0, -50.0]])

    inlier_score = float(detector.compute_anomaly_scores(inlier)[0])
    outlier_score = float(detector.compute_anomaly_scores(outlier)[0])

    # CORE REQUIREMENT VERIFICATION:
    # Higher anomaly scores MUST represent more suspicious observations!
    assert outlier_score > inlier_score, (
        f"Outlier score ({outlier_score}) should be strictly greater than inlier score ({inlier_score})"
    )


def test_threshold_calibration(mock_detector_config: dict):
    detector = IsolationForestDetector(mock_detector_config)

    np.random.seed(42)
    X_train = np.random.normal(loc=10.0, scale=0.5, size=(80, 2))
    X_val = np.random.normal(loc=10.0, scale=0.5, size=(40, 2))

    detector.fit(X_train)
    calibrated_th = detector.calibrate_threshold(X_val, percentile=95.0)

    # The calibrated threshold should be approximately at the 95th percentile of validation scores
    val_scores = detector.compute_anomaly_scores(X_val)
    expected_th = np.percentile(val_scores, 95.0)
    assert calibrated_th == pytest.approx(expected_th, rel=1e-4)

    # Predictions check
    predictions = detector.predict(X_val)
    assert len(predictions) == 40
    for p in predictions:
        assert "anomaly_score" in p
        assert "anomaly_threshold" in p
        assert "is_anomaly" in p
        assert p["status"] in ["CONSISTENT", "SUSPICIOUS"]


def test_save_and_load_detector(mock_detector_config: dict, tmp_path: Path):
    detector = IsolationForestDetector(mock_detector_config)
    X_train = np.random.normal(loc=5.0, scale=0.2, size=(50, 2))
    detector.fit(X_train)
    detector.calibrate_threshold(X_train, percentile=90.0)

    save_file = tmp_path / "test_iforest.joblib"
    detector.save(save_file)
    assert save_file.exists()

    loaded = IsolationForestDetector.load(save_file)
    assert loaded.is_fitted is True
    assert loaded.calibrated_threshold == pytest.approx(detector.calibrated_threshold)

    sample = np.array([[5.1, 4.9]])
    np.testing.assert_array_almost_equal(
        detector.compute_anomaly_scores(sample),
        loaded.compute_anomaly_scores(sample)
    )
