"""
EnviroTrust ML Service — Isolation Forest Anomaly Detection

Wraps scikit-learn's IsolationForest for industrial process consistency monitoring.
Ensures that higher anomaly scores strictly represent more suspicious observations.
Supports empirical threshold calibration over clean normal validation sets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("EnviroTrust.AnomalyDetection")


class ModelNotTrainedError(Exception):
    """Raised when inference is requested before authentic model training has occurred."""
    pass


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector calibrated for industrial process cross-checking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        iforest_cfg = self.config.get("isolation_forest", {})

        self.n_estimators = int(iforest_cfg.get("n_estimators", 150))
        self.max_samples = iforest_cfg.get("max_samples", "auto")
        self.contamination = iforest_cfg.get("contamination", "auto")
        self.random_state = iforest_cfg.get("random_state", 42)
        self.bootstrap = bool(iforest_cfg.get("bootstrap", False))
        self.n_jobs = int(iforest_cfg.get("n_jobs", -1))

        # Model instance
        self.model: Optional[IsolationForest] = None
        self.is_fitted: bool = False

        # Calibrated decision threshold (higher score = more suspicious)
        cal_cfg = self.config.get("calibration", {})
        self.threshold_percentile = float(cal_cfg.get("threshold_percentile", 99.0))
        self.manual_threshold = cal_cfg.get("manual_threshold")
        self.calibrated_threshold: float = (
            float(self.manual_threshold)
            if self.manual_threshold is not None
            else float(cal_cfg.get("fallback_threshold", 0.65))
        )

        # Calibration metadata for auditability
        self.calibration_metadata: Dict[str, Any] = {
            "strategy": "manual" if self.manual_threshold is not None else "empirical_percentile",
            "threshold_percentile": self.threshold_percentile if self.manual_threshold is None else None,
            "validation_sample_count": 0,
            "threshold": self.calibrated_threshold,
        }

        # Validation stats for score distribution
        self.score_min: float = 0.0
        self.score_max: float = 1.0

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit Isolation Forest model on validated numerical feature matrix.
        When labels are available, caller must ensure X contains normal records only.
        """
        if X is None or len(X) == 0:
            raise ValueError("Training matrix X must contain at least one sample.")

        logger.info(
            f"Training Isolation Forest (n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, contamination={self.contamination}, samples={X.shape[0]})"
        )

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X)
        self.is_fitted = True

        train_scores = self.compute_anomaly_scores(X)
        self.score_min = float(np.min(train_scores))
        self.score_max = float(np.max(train_scores)) if np.max(train_scores) > self.score_min else (self.score_min + 1.0)

        logger.info(f"Isolation Forest fitted successfully. Training score range: [{self.score_min:.4f}, {self.score_max:.4f}]")
        return self

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly scores.
        Scikit-learn convention:
          - decision_function(X) > 0 for inliers, < 0 for outliers.
        EnviroTrust convention:
          - Higher anomaly scores represent more suspicious observations.
        Therefore, we negate the decision_function:
          anomaly_score = -model.decision_function(X)
        """
        if not self.is_fitted or self.model is None:
            raise ModelNotTrainedError("Isolation Forest model is not trained yet.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        raw_scores = -self.model.decision_function(X)
        return raw_scores

    def calibrate_threshold(self, X_val_normal: np.ndarray, percentile: Optional[float] = None) -> float:
        """
        Calibrate the anomaly decision threshold using clean normal validation data.
        NOTE: The threshold defines an empirical cutoff on the validation sample distribution.
        It does NOT guarantee an invariant real-world false-positive rate due to concept drift.
        """
        if not self.is_fitted or self.model is None:
            raise ModelNotTrainedError("Cannot calibrate threshold before model is fitted.")

        if self.manual_threshold is not None:
            self.calibrated_threshold = float(self.manual_threshold)
            self.calibration_metadata = {
                "strategy": "manual_override",
                "threshold_percentile": None,
                "validation_sample_count": len(X_val_normal) if X_val_normal is not None else 0,
                "threshold": self.calibrated_threshold,
            }
            logger.info(f"Using manual anomaly threshold override: {self.calibrated_threshold:.4f}")
            return self.calibrated_threshold

        if X_val_normal is None or len(X_val_normal) == 0:
            logger.warning("Empty normal validation matrix provided. Retaining fallback threshold.")
            return self.calibrated_threshold

        pct = percentile if percentile is not None else self.threshold_percentile
        val_scores = self.compute_anomaly_scores(X_val_normal)

        self.calibrated_threshold = float(np.percentile(val_scores, pct))
        self.calibration_metadata = {
            "strategy": "empirical_percentile",
            "threshold_percentile": float(pct),
            "validation_sample_count": int(len(val_scores)),
            "threshold": round(self.calibrated_threshold, 4),
            "validation_score_mean": round(float(np.mean(val_scores)), 4),
            "validation_score_std": round(float(np.std(val_scores)), 4),
            "empirical_validation_cutoff_note": (
                f"Calibrated on {len(val_scores)} normal validation samples at {pct}th percentile. "
                "Real-world operational false-positive rate may differ due to environmental variance."
            ),
        }

        logger.info(
            f"Calibrated empirical threshold: {self.calibrated_threshold:.4f} "
            f"({pct}th percentile on {len(val_scores)} normal validation samples)"
        )
        return self.calibrated_threshold

    def predict(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate samples and return structured anomaly predictions.
        Consistent threshold applied between training evaluation and inference.
        """
        scores = self.compute_anomaly_scores(X)
        active_threshold = threshold if threshold is not None else self.calibrated_threshold

        results = []
        for score in scores:
            is_anomaly = bool(score >= active_threshold)
            status = "SUSPICIOUS" if is_anomaly else "CONSISTENT"
            results.append({
                "anomaly_score": round(float(score), 4),
                "anomaly_threshold": round(float(active_threshold), 4),
                "is_anomaly": is_anomaly,
                "status": status,
            })

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize trained model to disk."""
        if not self.is_fitted or self.model is None:
            raise ModelNotTrainedError("Cannot save an unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved Isolation Forest artifact to: {path.resolve()}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IsolationForestDetector":
        """Load serialized model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError(f"Loaded object is of type {type(detector)}, expected {cls}.")
        logger.info(f"Loaded Isolation Forest artifact from: {path.resolve()}")
        return detector
