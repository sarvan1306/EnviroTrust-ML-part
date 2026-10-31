"""
EnviroTrust ML Service — Evaluation Module

Provides quantitative evaluation metrics for Isolation Forest anomaly detection.
Supports both unsupervised operational distributions and supervised benchmark metrics
(precision, recall, F1-score, false positive rate, detection rate, ROC-AUC)
when labeled test cases (string or numerical) are available.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

from anomaly_detection import IsolationForestDetector, ModelNotTrainedError
from data_loader import load_config, load_dataset
from data_validator import parse_label_value, DEFAULT_NORMAL_LABELS, DEFAULT_ANOMALY_LABELS
from feature_engineering import FeatureEngineeringPipeline
from preprocessing import DataPreprocessor

logger = logging.getLogger("EnviroTrust.Evaluation")


def evaluate_dataset(
    data_path: Union[str, Path, pd.DataFrame],
    config_path: Union[str, Path, Dict[str, Any]] = "config/data_config.yaml",
    models_dir: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate the trained Isolation Forest model on a given dataset.

    Parameters:
        data_path: Path to dataset (CSV or Excel) or pandas DataFrame.
        config_path: Path to configuration YAML or in-memory dict.
        models_dir: Directory containing trained model artifacts.
        label_column: Optional column name containing string or numerical ground-truth labels.

    Returns:
        Dict containing score distribution and (if labeled) supervised metrics.
    """
    config = load_config(config_path)
    art_cfg = config.get("artifacts", {})
    m_dir = Path(models_dir or art_cfg.get("models_dir", "models"))

    model_path = m_dir / art_cfg.get("model_filename", "isolation_forest.joblib")
    prep_path = m_dir / art_cfg.get("preprocessor_filename", "preprocessor.joblib")

    # Fallback to demo artifact filenames if standard names do not exist in the directory
    if not model_path.exists() and (m_dir / "isolation_forest_demo.joblib").exists():
        model_path = m_dir / "isolation_forest_demo.joblib"
    if not prep_path.exists() and (m_dir / "preprocessor_demo.joblib").exists():
        prep_path = m_dir / "preprocessor_demo.joblib"

    if not model_path.exists() or not prep_path.exists():
        raise ModelNotTrainedError(f"Model or preprocessor artifacts not found in {m_dir}. Cannot evaluate.")

    detector = IsolationForestDetector.load(model_path)
    preprocessor = DataPreprocessor.load(prep_path)

    if isinstance(data_path, pd.DataFrame):
        df_raw = data_path.copy()
        dataset_name = "in_memory_dataframe"
    else:
        df_raw = load_dataset(data_path)
        dataset_name = str(Path(data_path).resolve())

    # 1. Parse labels if present (support string and numerical labels)
    lbl_col = label_column or config.get("labels", {}).get("label_column")
    labels = None

    if lbl_col and lbl_col in df_raw.columns:
        norm_vals = set(config.get("labels", {}).get("normal_values", DEFAULT_NORMAL_LABELS))
        anom_vals = set(config.get("labels", {}).get("anomaly_values", DEFAULT_ANOMALY_LABELS))
        custom_map = config.get("labels", {}).get("custom_mapping") or {}

        try:
            parsed = df_raw[lbl_col].apply(lambda v: parse_label_value(v, norm_vals, anom_vals, custom_map))
            labels = parsed.to_numpy(dtype=int)
            logger.info(f"Loaded ground-truth label column '{lbl_col}': {int((labels == 1).sum())} anomalies, {int((labels == 0).sum())} normals.")
        except ValueError as e:
            logger.error(f"Failed parsing label column '{lbl_col}': {e}")
            raise

    # 2. Extract features using saved preprocessor schema
    fe_pipeline = FeatureEngineeringPipeline(config)
    df_features = fe_pipeline.process(df_raw)

    # Transform without fitting (allow_missing_columns=False ensures strict schema validation)
    X = preprocessor.transform(df_features, allow_missing_columns=False)

    # 3. Compute scores and threshold predictions
    scores = detector.compute_anomaly_scores(X)
    threshold = float(detector.calibrated_threshold)
    predictions = (scores >= threshold).astype(int)

    score_dist = {
        "sample_count": int(len(scores)),
        "threshold": round(threshold, 4),
        "flagged_anomalies_count": int(np.sum(predictions)),
        "flagged_percentage": round(float(np.mean(predictions)) * 100, 2),
        "score_min": round(float(np.min(scores)), 4),
        "score_mean": round(float(np.mean(scores)), 4),
        "score_median": round(float(np.median(scores)), 4),
        "score_max": round(float(np.max(scores)), 4),
        "percentile_90": round(float(np.percentile(scores, 90.0)), 4),
        "percentile_95": round(float(np.percentile(scores, 95.0)), 4),
        "percentile_99": round(float(np.percentile(scores, 99.0)), 4),
    }

    report: Dict[str, Any] = {
        "evaluation_dataset": dataset_name,
        "score_distribution": score_dist,
    }

    # 4. Supervised Metrics (ONLY calculated when verified ground-truth labels exist)
    if labels is not None:
        prec = float(precision_score(labels, predictions, zero_division=0))
        rec = float(recall_score(labels, predictions, zero_division=0))
        f1 = float(f1_score(labels, predictions, zero_division=0))
        cm = confusion_matrix(labels, predictions)

        # Confusion matrix elements: tn, fp, fn, tp
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            detection_rate = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        else:
            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
            fpr = 0.0
            detection_rate = rec

        try:
            auc = float(roc_auc_score(labels, scores))
        except Exception:
            auc = None

        report["supervised_metrics"] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "detection_rate": round(detection_rate, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }
        logger.info(
            f"Supervised Evaluation: Precision={prec:.4f}, Recall={rec:.4f}, "
            f"F1={f1:.4f}, FPR={fpr:.4f}, DetectionRate={detection_rate:.4f}"
        )
    else:
        logger.info(
            "Unlabeled evaluation: Reporting score distributions and flagged percentages. "
            "Precision, recall, and detection rate are not claimed without ground truth."
        )

    return report
