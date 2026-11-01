"""
EnviroTrust ML Service — Training Pipeline

Trains scikit-learn Isolation Forest model exclusively on authentic industrial datasets.
Supports:
  1. Flexible feature configurations (Environmental-only or Process-context)
  2. Chronological data splitting for time-series telemetry (preventing temporal leakage)
  3. Label-aware training (fitting strictly on verified normal baseline records)
  4. Empirical threshold calibration on clean validation sets
  5. Persistence of exact training feature schema alongside artifacts
  6. Held-out test reservation (splitting.test_ratio), never used for fitting or calibration
  7. Physical consistency rule envelopes learned on normal training records
  8. CUSUM sustained-shift baseline learned on normal training records, threshold calibrated
     on the normal validation stream (temporal datasets only)
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from anomaly_detection import IsolationForestDetector
from consistency_rules import PhysicalConsistencyChecker
from cusum import CusumDetector
from data_loader import load_config, load_dataset, inspect_dataset, format_inspection_report
from data_validator import DataValidator, DataValidationError, parse_label_value
from feature_engineering import FeatureEngineeringPipeline
from preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnviroTrust.Train")

MODEL_VERSION = "iforest-cod-v1.0"


class TrainingHaltedError(Exception):
    """Raised when training cannot proceed due to missing authentic datasets."""
    pass


def split_dataset(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and validation subsets based on configured strategy.

    Strategies:
      - 'chronological': Sorts by timestamp_column, trains on earlier records, validates on later records.
      - 'random': Shuffles randomly (only suitable for independent, non-time-series data).

    Raises:
        ValueError: If chronological splitting is requested but timestamp is missing/invalid.
    """
    split_cfg = config.get("splitting", {})
    strategy = split_cfg.get("strategy", "random").lower() if split_cfg else "random"
    train_ratio = float(split_cfg.get("train_ratio", 0.80))

    if strategy == "chronological":
        ts_col = split_cfg.get("timestamp_column", "timestamp")
        if ts_col not in df.columns:
            raise ValueError(
                f"Chronological splitting requested, but timestamp column '{ts_col}' not found in dataset. "
                "Specify a valid 'splitting.timestamp_column' or explicitly set strategy to 'random' "
                "if dataset is non-temporal. Do not silently claim chronological validation."
            )

        # Check for missing timestamps
        if df[ts_col].isnull().any():
            missing_count = int(df[ts_col].isnull().sum())
            raise ValueError(
                f"Timestamp column '{ts_col}' contains {missing_count} missing values. "
                "Chronological splitting requires complete timestamps."
            )

        logger.info(f"Applying chronological split on column '{ts_col}' (train: {train_ratio*100:.0f}%, val: {(1-train_ratio)*100:.0f}%)")
        # Parse datetimes and sort chronologically
        df_sorted = df.copy()
        df_sorted["_parsed_ts"] = pd.to_datetime(df_sorted[ts_col], errors="raise")
        df_sorted = df_sorted.sort_values(by="_parsed_ts", ascending=True).reset_index(drop=True)
        df_sorted = df_sorted.drop(columns=["_parsed_ts"])

        split_idx = int(len(df_sorted) * train_ratio)
        df_train = df_sorted.iloc[:split_idx].copy()
        df_val = df_sorted.iloc[split_idx:].copy()

        logger.info(f"Chronological split complete: {len(df_train)} earlier train rows, {len(df_val)} later val rows.")
        return df_train, df_val

    elif strategy == "random":
        logger.info(f"Applying random split (train: {train_ratio*100:.0f}%, val: {(1-train_ratio)*100:.0f}%)")
        rs = config.get("isolation_forest", {}).get("random_state", 42)
        df_train, df_val = train_test_split(df, test_size=(1.0 - train_ratio), random_state=rs, shuffle=True)
        return df_train.copy(), df_val.copy()

    else:
        raise ValueError(f"Unknown splitting strategy '{strategy}'. Supported: 'chronological', 'random'.")


def split_holdout_test(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reserve a held-out test subset that is never used for fitting or threshold calibration.

    Controlled by 'splitting.test_ratio' (0 disables the holdout). Uses the same strategy as
    split_dataset: 'chronological' reserves the latest records, 'random' a seeded random sample.
    Deterministic for a given dataset and config, so evaluation can reconstruct the exact subset.

    Returns:
        Tuple of (development subset for train/validation, held-out test subset).
    """
    split_cfg = config.get("splitting", {}) or {}
    test_ratio = float(split_cfg.get("test_ratio", 0.0) or 0.0)
    if test_ratio <= 0.0:
        return df.copy(), df.iloc[0:0].copy()
    if test_ratio >= 1.0:
        raise ValueError(f"splitting.test_ratio must be in [0, 1), got {test_ratio}.")

    strategy = split_cfg.get("strategy", "random").lower()
    n_test = int(round(len(df) * test_ratio))

    if strategy == "chronological":
        ts_col = split_cfg.get("timestamp_column", "timestamp")
        if ts_col not in df.columns:
            raise ValueError(
                f"Chronological holdout requested, but timestamp column '{ts_col}' not found in dataset."
            )
        order = pd.to_datetime(df[ts_col], errors="raise").sort_values(kind="mergesort").index
        df_sorted = df.loc[order]
        df_dev, df_test = df_sorted.iloc[: len(df) - n_test], df_sorted.iloc[len(df) - n_test:]
    elif strategy == "random":
        rs = config.get("isolation_forest", {}).get("random_state", 42)
        df_dev, df_test = train_test_split(df, test_size=n_test, random_state=rs, shuffle=True)
    else:
        raise ValueError(f"Unknown splitting strategy '{strategy}'. Supported: 'chronological', 'random'.")

    logger.info(f"Reserved {len(df_test)} held-out test rows ({strategy}); {len(df_dev)} rows remain for train/validation.")
    return df_dev.copy(), df_test.copy()


def run_training(
    config_path: Union[str, Path, Dict[str, Any]] = "config/data_config.yaml",
    data_path: Optional[Union[str, Path]] = None,
    validation_path: Optional[Union[str, Path]] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Isolation Forest training workflow using authentic datasets.

    Parameters:
        config_path: Path to configuration YAML, or in-memory dict.
        data_path: Optional override path for training dataset.
        validation_path: Optional override path for validation dataset.
        strategy: Optional override for splitting strategy ('chronological' or 'random').

    Returns:
        Manifest dictionary with training metadata.

    Raises:
        TrainingHaltedError: If authentic dataset is missing or unconfigured.
        DataValidationError: If dataset fails structural validation.
    """
    config = load_config(config_path) if not isinstance(config_path, dict) else config_path.copy()
    if strategy is not None:
        if "splitting" not in config:
            config["splitting"] = {}
        config["splitting"]["strategy"] = strategy

    # 1. Resolve dataset path
    resolved_data_path = data_path or config.get("dataset", {}).get("raw_data_path")

    if not resolved_data_path:
        halt_msg = (
            "======================================================================\n"
            "TRAINING HALTED: No authentic industrial dataset provided.\n"
            "EnviroTrust requires authentic environmental and operational datasets\n"
            "from the facility team to train the Isolation Forest model.\n"
            "In accordance with project policy, NO fabricated or synthetic datasets\n"
            "will be generated or used for training.\n\n"
            "To train the model once authentic data is received:\n"
            "  1. Place the dataset in ml-service/data/\n"
            "  2. Configure 'dataset.raw_data_path' in config/data_config.yaml, or\n"
            "  3. Run: python train.py --data-path <path_to_authentic_data.csv>\n"
            "======================================================================"
        )
        logger.error(halt_msg)
        raise TrainingHaltedError(halt_msg)

    data_file = Path(resolved_data_path)
    if not data_file.exists():
        halt_msg = f"Authentic dataset file not found at: {data_file.resolve()}"
        logger.error(halt_msg)
        raise FileNotFoundError(halt_msg)

    logger.info(f"Initiating training with authentic dataset: {data_file.resolve()}")

    # 2. Ingestion & Profiling
    df_raw = load_dataset(data_file)
    profile = inspect_dataset(df_raw)
    logger.info("\n" + format_inspection_report(profile))

    # 3. Data Validation
    validator = DataValidator(config)
    report = validator.validate_dataset(df_raw)
    if not report.is_valid:
        logger.error("Dataset validation failed:\n" + report.summary())
        raise DataValidationError(f"Dataset validation failed: {report.errors}")
    logger.info("Dataset validation passed successfully.")

    # Rolling-window features are computed on the complete chronological stream, before any
    # split or label filter, so trailing windows never span gaps left by removed rows
    fe_pipeline = FeatureEngineeringPipeline(config)
    df_raw = fe_pipeline.add_rolling_features(df_raw)

    # 4. Held-out test reservation, then Train/Validation splitting (Chronological or Random)
    df_dev, df_holdout = split_holdout_test(df_raw, config)

    resolved_val_path = validation_path or config.get("dataset", {}).get("validation_data_path")
    if resolved_val_path and Path(resolved_val_path).exists():
        logger.info(f"Loading separate validation dataset from: {resolved_val_path}")
        df_train_raw = df_dev.copy()
        df_val_raw = fe_pipeline.add_rolling_features(load_dataset(resolved_val_path))
    else:
        df_train_raw, df_val_raw = split_dataset(df_dev, config)

    # 5. Label-Aware Handling
    label_cfg = config.get("labels", {})
    label_col = label_cfg.get("label_column")
    has_labels = False

    if label_col and label_col in df_raw.columns:
        has_labels = True
        logger.info(f"Label-aware training active using column '{label_col}'.")
        norm_vals = set(label_cfg.get("normal_values", []))
        anom_vals = set(label_cfg.get("anomaly_values", []))
        cust_map = label_cfg.get("custom_mapping") or {}

        # Parse binary labels (0=normal, 1=anomaly)
        df_train_raw["_binary_label"] = df_train_raw[label_col].apply(
            lambda v: parse_label_value(v, norm_vals, anom_vals, cust_map)
        )
        df_val_raw["_binary_label"] = df_val_raw[label_col].apply(
            lambda v: parse_label_value(v, norm_vals, anom_vals, cust_map)
        )

        train_anom_cnt = int((df_train_raw["_binary_label"] == 1).sum())
        logger.info(f"Filtering training set: {train_anom_cnt} known anomalous records excluded from Isolation Forest fit.")
        # Fit Isolation Forest ONLY on normal training records
        df_train_fit = df_train_raw[df_train_raw["_binary_label"] == 0].copy()

        # Validation set for threshold calibration: verified normal records only
        df_val_calib = df_val_raw[df_val_raw["_binary_label"] == 0].copy()
        df_val_anomalies = df_val_raw[df_val_raw["_binary_label"] == 1].copy()
    else:
        logger.info("No label column configured. Operating in unsupervised mode under normal-baseline assumption.")
        df_train_fit = df_train_raw.copy()
        df_val_calib = df_val_raw.copy()
        df_val_anomalies = pd.DataFrame()

    # 6. Feature Engineering & Column Mapping
    df_train_features = fe_pipeline.process(df_train_fit)
    df_val_calib_features = fe_pipeline.process(df_val_calib)

    # 7. Fit Preprocessor (TRAINING SPLIT ONLY - Zero Data Leakage)
    preprocessor = DataPreprocessor(config)
    X_train = preprocessor.fit_transform(df_train_features)
    # Validation uses transform with allow_missing_columns=False to ensure strict schema consistency
    X_val_calib = preprocessor.transform(df_val_calib_features, allow_missing_columns=False)

    validator.validate_feature_matrix(X_train)
    validator.validate_feature_matrix(X_val_calib)

    # 8. Train Isolation Forest
    detector = IsolationForestDetector(config)
    detector.fit(X_train)

    # 9. Calibrate Anomaly Threshold on Normal Validation Data
    threshold = detector.calibrate_threshold(X_val_calib)
    val_normal_scores = detector.compute_anomaly_scores(X_val_calib)

    # 10. Fit Physical Consistency Rules on raw (unimputed) normal training features.
    # Rolling features belong to Isolation Forest; rules stay per-reading checks.
    rules = PhysicalConsistencyChecker(config).fit(
        df_train_features.drop(columns=fe_pipeline.rolling_features, errors="ignore")
    )
    val_rule_flags = rules.flags(df_val_calib_features)
    val_iforest_flags = val_normal_scores >= threshold

    # 10b. CUSUM: baseline from normal training values, threshold from the normal validation
    # stream in time order. Needs a timestamp: row order of non-temporal data is meaningless.
    cusum = CusumDetector(config)
    cusum_active = cusum.enabled and cusum.timestamp_column in df_raw.columns
    if cusum_active:
        df_val_calib_mapped = fe_pipeline.apply_column_mapping(df_val_calib)
        cusum.fit(df_train_features)
        cusum.calibrate(df_val_calib_mapped)
        val_cusum_flags = cusum.flags(df_val_calib_mapped)
    else:
        if cusum.enabled:
            logger.info(f"CUSUM skipped: timestamp column '{cusum.timestamp_column}' not present (non-temporal dataset).")
        val_cusum_flags = np.zeros(len(val_rule_flags), dtype=bool)
    val_combined_flags = val_iforest_flags | val_rule_flags | val_cusum_flags

    # 11. Save Artifacts
    art_cfg = config.get("artifacts", {})
    models_dir = Path(art_cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / art_cfg.get("model_filename", "isolation_forest.joblib")
    prep_path = models_dir / art_cfg.get("preprocessor_filename", "preprocessor.joblib")
    meta_path = models_dir / art_cfg.get("metadata_filename", "model_metadata.json")
    rules_path = models_dir / art_cfg.get("rules_filename", "consistency_rules.json")
    cusum_path = models_dir / art_cfg.get("cusum_filename", "cusum.json")

    detector.save(model_path)
    preprocessor.save(prep_path)
    rules.save(rules_path)
    if cusum_active:
        cusum.save(cusum_path)
    elif cusum_path.exists():
        # A stale baseline from an earlier model must not be paired with this one
        cusum_path.unlink()

    # Exact training schema recorded
    training_schema = preprocessor.get_training_schema()

    metadata: Dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "algorithm": "sklearn.ensemble.IsolationForest",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(data_file.resolve()),
        "has_ground_truth_labels": has_labels,
        "label_column": label_col if has_labels else None,
        "normal_baseline_assumption": (
            "Verified normal records filtered for training"
            if has_labels
            else "Unlabeled dataset: training assumes predominantly normal baseline operations."
        ),
        "split_strategy": config.get("splitting", {}).get("strategy", "chronological"),
        "train_samples": int(X_train.shape[0]),
        "train_samples_fitted": int(X_train.shape[0]),
        "validation_samples": int(X_val_calib.shape[0]),
        "validation_samples_calibrated": int(X_val_calib.shape[0]),
        "validation_anomalies_retained": int(len(df_val_anomalies)) if has_labels else 0,
        "feature_schema": training_schema,
        "features": preprocessor.get_feature_names_out(),
        "calibrated_threshold": round(float(threshold), 4),
        "calibration_metadata": detector.calibration_metadata,
        "validation_score_distribution": {
            "min": round(float(np.min(val_normal_scores)), 4),
            "mean": round(float(np.mean(val_normal_scores)), 4),
            "median": round(float(np.median(val_normal_scores)), 4),
            "max": round(float(np.max(val_normal_scores)), 4),
            "95th_percentile": round(float(np.percentile(val_normal_scores, 95.0)), 4),
            "99th_percentile": round(float(np.percentile(val_normal_scores, 99.0)), 4),
        },
        "validation_normal_score_distribution": {
            "min": round(float(np.min(val_normal_scores)), 4),
            "mean": round(float(np.mean(val_normal_scores)), 4),
            "median": round(float(np.median(val_normal_scores)), 4),
            "max": round(float(np.max(val_normal_scores)), 4),
            "95th_percentile": round(float(np.percentile(val_normal_scores, 95.0)), 4),
            "99th_percentile": round(float(np.percentile(val_normal_scores, 99.0)), 4),
        },
        "hyperparameters": {
            "n_estimators": detector.n_estimators,
            "max_samples": detector.max_samples,
            "contamination": detector.contamination,
            "random_state": detector.random_state,
        },
        "rolling_features": {
            "enabled": fe_pipeline.rolling_cfg["enabled"],
            "window": fe_pipeline.rolling_cfg["window"],
            "group_column": fe_pipeline.rolling_cfg["group_column"],
            "features": [f for f in fe_pipeline.rolling_features if f in training_schema["numerical_features"]],
        },
        "holdout_test": {
            "test_ratio": float(config.get("splitting", {}).get("test_ratio", 0.0) or 0.0),
            "rows": int(len(df_holdout)),
            "note": "Never used for fitting or calibration. Reconstruct with train.split_holdout_test.",
        },
        "consistency_rules": {
            "enabled": rules.enabled,
            "artifact": rules_path.name,
            "feature_envelopes": len(rules.envelopes),
            "ratio_bands": len(rules.ratio_bands),
        },
        "cusum": {
            "enabled": cusum_active,
            "artifact": cusum_path.name if cusum_active else None,
            "k": cusum.k,
            "h": cusum.h if cusum_active else None,
            "features": len(cusum.features),
        },
        "validation_flag_rates": {
            "isolation_forest": round(float(np.mean(val_iforest_flags)), 4) if len(val_iforest_flags) else None,
            "consistency_rules": round(float(np.mean(val_rule_flags)), 4) if len(val_rule_flags) else None,
            "cusum": round(float(np.mean(val_cusum_flags)), 4) if len(val_cusum_flags) else None,
            "combined": round(float(np.mean(val_combined_flags)), 4) if len(val_combined_flags) else None,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata manifest to: {meta_path.resolve()}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnviroTrust Isolation Forest Training")
    parser.add_argument("--config", default="config/data_config.yaml", help="Path to config YAML")
    parser.add_argument("--data-path", default=None, help="Path to authentic dataset (CSV or Excel)")
    parser.add_argument("--val-path", default=None, help="Path to authentic validation dataset")
    parser.add_argument("--strategy", choices=["chronological", "random"], default=None, help="Override splitting strategy ('chronological' or 'random')")
    args = parser.parse_args()

    try:
        manifest = run_training(
            config_path=args.config,
            data_path=args.data_path,
            validation_path=args.val_path,
            strategy=args.strategy,
        )
        print("\nTraining completed successfully:")
        print(json.dumps(manifest, indent=2))
    except (TrainingHaltedError, FileNotFoundError, DataValidationError, ValueError) as e:
        logger.error(f"Training did not proceed: {e}")
        exit(1)
