"""
EnviroTrust ML Service — Prediction & Inference Interface

Provides inference interface for evaluating incoming environmental and operational
telemetry records. Formats inputs, validates against the saved training schema,
and invokes the Isolation Forest detector.

Strictly refuses to fabricate predictions if the authentic model has not been trained,
and refuses to silently invent missing features required by the trained model.

Rolling-window features need history a single record does not carry, so the predictor
keeps the last few readings per sensorId in memory. The buffer is per process: with
several workers, route each sensor to one worker or its windows will be incomplete.
The CUSUM statistics are kept per sensorId the same way.
"""

from collections import deque
import json
import logging
from pathlib import Path
import threading
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from anomaly_detection import IsolationForestDetector, ModelNotTrainedError
from consistency_rules import PhysicalConsistencyChecker
from cusum import CusumDetector
from data_loader import load_config
from feature_engineering import DEFAULT_STREAM_KEY, ROLLING_SUFFIX, FeatureEngineeringPipeline
from preprocessing import DataPreprocessor, MissingFeatureError

logger = logging.getLogger("EnviroTrust.Predict")


class EnviroTrustPredictor:
    """
    Inference engine for EnviroTrust process anomaly detection.
    Supports both authentic operational models and isolated synthetic demonstration models.
    """

    def __init__(
        self,
        config_path: Union[str, Path, Dict[str, Any]] = "config/data_config.yaml",
        models_dir: Optional[Union[str, Path]] = None,
        model_filename: Optional[str] = None,
        preprocessor_filename: Optional[str] = None,
        metadata_filename: Optional[str] = None,
        model_mode: Optional[str] = None,
        rules_filename: Optional[str] = None,
        cusum_filename: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        art_cfg = self.config.get("artifacts", {})

        self.models_dir = Path(models_dir or art_cfg.get("models_dir", "models"))
        self.model_filename = model_filename or art_cfg.get("model_filename", "isolation_forest.joblib")
        self.prep_filename = preprocessor_filename or art_cfg.get("preprocessor_filename", "preprocessor.joblib")
        self.meta_filename = metadata_filename or art_cfg.get("metadata_filename", "model_metadata.json")
        self.rules_filename = rules_filename or art_cfg.get("rules_filename", "consistency_rules.json")
        self.cusum_filename = cusum_filename or art_cfg.get("cusum_filename", "cusum.json")
        self.model_mode_override = model_mode

        self.fe_pipeline = FeatureEngineeringPipeline(self.config)
        self.preprocessor: Optional[DataPreprocessor] = None
        self.detector: Optional[IsolationForestDetector] = None
        self.rules: Optional[PhysicalConsistencyChecker] = None
        self.cusum: Optional[CusumDetector] = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self.model_mode: str = "AUTHENTIC"
        self.data_provenance: str = "AUTHENTIC"

        # Sequential state per sensor: rolling-window history and CUSUM statistics (one lock guards both)
        self.rolling_features: List[str] = []
        self._history: Dict[str, Deque[Dict[str, Any]]] = {}
        self._cusum_states: Dict[str, Dict[str, Any]] = {}
        self._history_lock = threading.Lock()

        # Attempt to load model on initialization
        self.reload()

    @classmethod
    def get_demo_predictor(cls, config_path: Union[str, Path] = "config/data_config.yaml") -> "EnviroTrustPredictor":
        """Factory for creating a demo predictor instance targeting models/demo/."""
        return cls(
            config_path=config_path,
            models_dir="models/demo",
            model_filename="isolation_forest_demo.joblib",
            preprocessor_filename="preprocessor_demo.joblib",
            metadata_filename="model_metadata.json",
            model_mode="SYNTHETIC_DEMO",
        )

    def reload(self) -> bool:
        """
        Attempt to load model artifacts from disk.
        Returns True if all required artifacts were successfully loaded, False otherwise.
        """
        model_path = self.models_dir / self.model_filename
        prep_path = self.models_dir / self.prep_filename
        meta_path = self.models_dir / self.meta_filename

        is_demo_path = "demo" in str(self.models_dir).lower()
        default_mode = "SYNTHETIC_DEMO" if is_demo_path else "AUTHENTIC"
        self.model_mode = self.model_mode_override or default_mode
        self.data_provenance = "SYNTHETIC" if self.model_mode == "SYNTHETIC_DEMO" else "AUTHENTIC"
        self.rolling_features = []
        with self._history_lock:
            self._history = {}
            self._cusum_states = {}

        if not (model_path.exists() and prep_path.exists() and meta_path.exists()):
            logger.info(f"Model artifacts not found in {self.models_dir}. Predictor in UNTRAINED state.")
            self.is_loaded = False
            self.detector = None
            self.preprocessor = None
            self.rules = None
            self.cusum = None
            self.metadata = {}
            return False

        try:
            self.preprocessor = DataPreprocessor.load(prep_path)
            self.detector = IsolationForestDetector.load(model_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            # Rules are optional: models trained before the rule layer existed run without them
            rules_path = self.models_dir / self.rules_filename
            self.rules = PhysicalConsistencyChecker.load(rules_path) if rules_path.exists() else None
            cusum_path = self.models_dir / self.cusum_filename
            self.cusum = CusumDetector.load(cusum_path) if cusum_path.exists() else None
            self._configure_rolling()
            self.is_loaded = True
            self.model_mode = self.model_mode_override or self.metadata.get("model_mode", default_mode)
            self.data_provenance = self.metadata.get("data_provenance", "SYNTHETIC" if self.model_mode == "SYNTHETIC_DEMO" else "AUTHENTIC")
            logger.info(f"Loaded {self.model_mode} model artifacts (version: {self.metadata.get('model_version')}).")
            return True
        except Exception as e:
            logger.error(f"Failed to load model artifacts from {self.models_dir}: {e}")
            self.is_loaded = False
            return False

    def _configure_rolling(self) -> None:
        """
        Match rolling features to the loaded model rather than the current config, so a
        config edit after training cannot silently change the window the model was fitted on.
        """
        self.rolling_features = [c for c in self.preprocessor.active_num_cols if c.endswith(ROLLING_SUFFIX)]
        if not self.rolling_features:
            self.fe_pipeline = FeatureEngineeringPipeline(self.config)
            return
        roll_meta = self.metadata.get("rolling_features") or {}
        cfg = dict(self.config)
        cfg["rolling_features"] = {
            **(self.config.get("rolling_features") or {}),
            **{k: roll_meta[k] for k in ("window", "group_column") if k in roll_meta},
            "enabled": True,
            "base_features": [c[: -len(ROLLING_SUFFIX)] for c in self.rolling_features],
        }
        self.fe_pipeline = FeatureEngineeringPipeline(cfg)

    def _apply_rolling(
        self, df_mapped: pd.DataFrame, source_record: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
        """
        Compute the record's rolling features from the sensor's buffered history.
        Returns (frame with rolling columns, buffer entry to remember, window filled).
        A retried reading (same readingId as the last buffered one) replaces it rather than
        counting as a repeat, otherwise retries would look like a stuck sensor.
        """
        key = str(source_record.get("sensorId") or DEFAULT_STREAM_KEY)
        ts_col = self.fe_pipeline.rolling_cfg["timestamp_column"]
        row = df_mapped.iloc[0]
        entry = {b: row.get(b, np.nan) for b in self.fe_pipeline.rolling_cfg["base_features"]}
        entry[ts_col] = source_record.get("timestamp")
        entry["readingId"] = source_record.get("readingId")

        with self._history_lock:
            history = list(self._history.get(key, ()))
        if history and entry["readingId"] is not None and history[-1]["readingId"] == entry["readingId"]:
            history = history[:-1]
        keep = self.fe_pipeline.rolling_cfg["window"] - 1
        history = history[-keep:] if keep > 0 else []

        rolled = self.fe_pipeline.add_rolling_features(pd.DataFrame(history + [entry]), require_timestamp=False)
        current = rolled[self.rolling_features].iloc[-1]

        out = df_mapped.copy()
        for col in self.rolling_features:
            out[col] = current[col]
        return out, entry, bool(current.notna().any())

    def _remember(self, sensor_id: Optional[str], entry: Dict[str, Any]) -> None:
        """
        Append a scored reading to its sensor's history buffer. Keeps one reading more than a
        window needs, so replacing a retried reading never shortens the next window.
        """
        key = str(sensor_id or DEFAULT_STREAM_KEY)
        maxlen = self.fe_pipeline.rolling_cfg["window"]
        with self._history_lock:
            buf = self._history.setdefault(key, deque(maxlen=maxlen))
            if buf and entry["readingId"] is not None and buf[-1]["readingId"] == entry["readingId"]:
                buf[-1] = entry
            else:
                buf.append(entry)

    def _apply_cusum(
        self, df_mapped: pd.DataFrame, source_record: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Advance the sensor's CUSUM statistics by this reading. Returns (alarms, pending update).
        A retried reading (same readingId as the last one) is applied to the state from before
        that reading, so retries do not accumulate the same deviation twice.
        """
        key = str(source_record.get("sensorId") or DEFAULT_STREAM_KEY)
        reading_id = source_record.get("readingId")
        with self._history_lock:
            prev = self._cusum_states.get(key)
        if prev is None:
            before = self.cusum.initial_state()
        elif reading_id is not None and prev["readingId"] == reading_id:
            before = prev["before"]
        else:
            before = prev["after"]
        after = self.cusum.step(before, df_mapped.iloc[0].to_dict())
        return self.cusum.alarms(after), {"key": key, "readingId": reading_id, "before": before, "after": after}

    def is_trained(self) -> bool:
        """Check if model and preprocessor are loaded and ready for inference."""
        return self.is_loaded and self.detector is not None and self.preprocessor is not None

    def get_status(self) -> Dict[str, Any]:
        """Return runtime status of the predictor and saved training schema."""
        if not self.is_trained():
            is_demo = self.model_mode == "SYNTHETIC_DEMO" or "demo" in str(self.models_dir).lower()
            return {
                "is_trained": False,
                "model_mode": "SYNTHETIC_DEMO" if is_demo else "AUTHENTIC",
                "data_provenance": "SYNTHETIC" if is_demo else "AUTHENTIC",
                "message": (
                    "Synthetic demonstration model is not trained."
                    if is_demo
                    else "Model is not trained. Awaiting authentic industrial operational dataset."
                ),
            }
        return {
            "is_trained": True,
            "model_mode": self.model_mode,
            "data_provenance": self.data_provenance,
            "model_version": self.metadata.get("model_version", "unknown"),
            "training_timestamp_utc": self.metadata.get("training_timestamp_utc"),
            "feature_schema": self.metadata.get("feature_schema", {}),
            "features": self.metadata.get("features", []),
            "threshold": self.detector.calibrated_threshold,
            "calibration_metadata": self.metadata.get("calibration_metadata", {}),
            "hyperparameters": self.metadata.get("hyperparameters", {}),
            "rolling_features": self.metadata.get("rolling_features"),
            "consistency_rules": self.rules.to_dict() if self.rules is not None else None,
            "cusum": self.cusum.to_dict() if self.cusum is not None else None,
        }

    def _format_cod_record(self, record: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert legacy COD analysis request dict into a DataFrame.
        Maintained for backward compatibility with /analyze/cod endpoint.
        """
        flat_dict: Dict[str, Any] = {}

        if "context" in record and isinstance(record["context"], dict):
            ctx = record["context"]
            for k, v in ctx.items():
                flat_dict[k] = v
            # Common aliases
            if "productionRate" in ctx:
                flat_dict["production_rate"] = ctx["productionRate"]
            if "wastewaterFlow" in ctx:
                flat_dict["wastewater_flow"] = ctx["wastewaterFlow"]
            if "treatmentDose" in ctx:
                flat_dict["treatment_dose"] = ctx["treatmentDose"]
            if "operatingMode" in ctx:
                flat_dict["operating_mode"] = ctx["operatingMode"]
            if "energyConsumption" in ctx:
                flat_dict["energy_consumption"] = ctx["energyConsumption"]

        # Map COD
        if "actualCod" in record:
            flat_dict["outlet_cod"] = record["actualCod"]
        elif "actual_cod" in record:
            flat_dict["outlet_cod"] = record["actual_cod"]

        for k, v in record.items():
            if k not in ["context", "recentCodReadings"]:
                flat_dict[k] = v

        return pd.DataFrame([flat_dict])

    def _format_telemetry_record(self, record: Dict[str, Any]) -> pd.DataFrame:
        """
        Format generalized telemetry record into a DataFrame row.
        Combines measurements dict and optional operational context.
        """
        flat_dict: Dict[str, Any] = {}

        # 1. Flatten measurements (e.g. {"cod": 15.0, "ph": 7.2, "bod": 45.0, ...})
        measurements = record.get("measurements", {})
        if isinstance(measurements, dict):
            for k, v in measurements.items():
                flat_dict[k] = v

        # 2. Flatten operational context if provided
        context = record.get("context", {})
        if isinstance(context, dict):
            for k, v in context.items():
                flat_dict[k] = v
            # Alias camelCase context fields
            aliases = {
                "productionRate": "production_rate",
                "wastewaterFlow": "wastewater_flow",
                "treatmentDose": "treatment_dose",
                "operatingMode": "operating_mode",
                "energyConsumption": "energy_consumption",
            }
            for camel, snake in aliases.items():
                if camel in context and snake not in context:
                    flat_dict[snake] = context[camel]

        # 3. Add identifiers
        for meta_k in ["readingId", "sensorId", "timestamp"]:
            if meta_k in record:
                flat_dict[meta_k] = record[meta_k]

        return pd.DataFrame([flat_dict])

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict anomaly score and status for a single incoming COD telemetry record.
        Maintained for backward compatibility.
        """
        if not self.is_trained():
            raise ModelNotTrainedError("Model is not trained. Awaiting authentic industrial operational dataset.")

        df_single = self._format_cod_record(record)
        return self._execute_inference(df_single, record)

    def predict_telemetry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict anomaly score for generalized telemetry.
        Validates against exact training feature schema.
        """
        if not self.is_trained():
            raise ModelNotTrainedError("Model is not trained. Awaiting authentic industrial operational dataset.")

        df_single = self._format_telemetry_record(record)
        return self._execute_inference(df_single, record)

    def _execute_inference(self, df_raw_record: pd.DataFrame, source_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering, preprocessing, and model scoring with strict schema validation."""
        # Apply column mapping
        df_mapped = self.fe_pipeline.apply_column_mapping(df_raw_record)

        # Physical rules run on raw mapped values (before any imputation)
        violations = self.rules.check(df_mapped)[0] if self.rules is not None else []

        # Rolling features from this sensor's recent readings (None when the model has none).
        # Until the window fills they are NaN and imputed, so stuck-sensor detection is inactive.
        rolling_window_filled: Optional[bool] = None
        history_entry: Optional[Dict[str, Any]] = None
        if self.rolling_features:
            df_mapped, history_entry, rolling_window_filled = self._apply_rolling(df_mapped, source_record)

        # CUSUM runs on raw mapped values, like the rules; committed only once the record is accepted
        cusum_alarms: List[Dict[str, Any]] = []
        cusum_update: Optional[Dict[str, Any]] = None
        if self.cusum is not None:
            cusum_alarms, cusum_update = self._apply_cusum(df_mapped, source_record)

        # Preprocessor transform checks required features and raises MissingFeatureError if absent
        # allow_missing_columns=False ensures no silent fabrication of missing features
        X = self.preprocessor.transform(df_mapped, allow_missing_columns=False)
        if history_entry is not None:
            self._remember(source_record.get("sensorId"), history_entry)
        if cusum_update is not None:
            with self._history_lock:
                self._cusum_states[cusum_update["key"]] = cusum_update

        scores = self.detector.compute_anomaly_scores(X)
        score = float(scores[0])
        threshold = float(self.detector.calibrated_threshold)
        iforest_flag = bool(score >= threshold)
        detection_sources = (
            (["isolation_forest"] if iforest_flag else [])
            + (["physical_rules"] if violations else [])
            + (["cusum"] if cusum_alarms else [])
        )
        is_anomaly = bool(detection_sources)
        status = "SUSPICIOUS" if is_anomaly else "CONSISTENT"

        return {
            "reading_id": source_record.get("readingId"),
            "sensor_id": source_record.get("sensorId"),
            "timestamp": source_record.get("timestamp"),
            "anomaly_score": round(score, 4),
            "anomaly_threshold": round(threshold, 4),
            "is_anomaly": is_anomaly,
            "isolation_forest_flag": iforest_flag,
            "rule_violations": violations,
            "cusum_alarms": cusum_alarms,
            "detection_sources": detection_sources,
            "rolling_window_filled": rolling_window_filled,
            "status": status,
            "model_version": self.metadata.get("model_version", "iforest-cod-v1.0"),
            "model_mode": self.model_mode,
            "data_provenance": self.data_provenance,
            "evaluated_features": self.preprocessor.get_feature_names_out(),
        }
