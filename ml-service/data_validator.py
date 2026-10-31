"""
EnviroTrust ML Service — Data Validation Module

Performs structural and quality validation on incoming datasets against configured
feature mappings. Validates candidate feature availability, splitting prerequisites,
and label hygiene without inventing or imputing missing measurements during validation checks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("EnviroTrust.DataValidator")

# Standard recognized label mappings
DEFAULT_NORMAL_LABELS = {"normal", "nominal", "0", "false", "clean", "ok", "safe", 0, False}
DEFAULT_ANOMALY_LABELS = {"anomaly", "abnormal", "tampered", "suspicious", "fraud", "1", "true", 1, True}


class DataValidationError(Exception):
    """Raised when dataset validation encounters fatal structural or quality issues."""
    pass


def parse_label_value(
    val: Any,
    normal_set: Optional[Set[Any]] = None,
    anomaly_set: Optional[Set[Any]] = None,
    custom_map: Optional[Dict[Any, int]] = None,
) -> int:
    """
    Parse a single ground-truth label value into 0 (normal) or 1 (anomaly).
    Rejects ambiguous or unknown label values rather than silently misclassifying them.

    Returns:
        0 for normal, 1 for anomaly.

    Raises:
        ValueError: If value cannot be unambiguously mapped.
    """
    if custom_map and val in custom_map:
        mapped = custom_map[val]
        if mapped in (0, 1):
            return mapped
        raise ValueError(f"Custom mapping for '{val}' must be 0 or 1, got: {mapped}")

    norm_set = normal_set if normal_set is not None else DEFAULT_NORMAL_LABELS
    anom_set = anomaly_set if anomaly_set is not None else DEFAULT_ANOMALY_LABELS

    # Exact match (bool, int, or direct string)
    if val in norm_set:
        return 0
    if val in anom_set:
        return 1

    # Case-insensitive string check
    if isinstance(val, str):
        val_lower = val.strip().lower()
        norm_str_set = {str(x).strip().lower() for x in norm_set}
        anom_str_set = {str(x).strip().lower() for x in anom_set}
        if val_lower in norm_str_set:
            return 0
        if val_lower in anom_str_set:
            return 1

    raise ValueError(f"Unrecognized or ambiguous label value: '{val}'. Expected normal or anomaly indicator.")


@dataclass
class ValidationReport:
    """Contains results of structural and statistical dataset validation."""
    is_valid: bool
    row_count: int
    column_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    available_candidate_features: List[str] = field(default_factory=list)
    missing_candidate_features: List[str] = field(default_factory=list)
    high_missingness_columns: Dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Validation Status: {status}",
            f"Rows: {self.row_count:,} | Columns: {self.column_count}",
            f"Duplicate Rows: {self.duplicate_rows}",
            f"Available Candidate Features: {self.available_candidate_features}",
        ]
        if self.missing_candidate_features:
            lines.append(f"Optional Unmapped Features: {self.missing_candidate_features}")
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - [ERROR] {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - [WARNING] {warn}")
        return "\n".join(lines)


class DataValidator:
    """
    Validates industrial datasets against flexible configuration rules.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.col_map: Dict[str, str] = config.get("column_mapping", {})
        self.num_features: List[str] = config.get("features", {}).get("numerical", [])
        self.cat_features: List[str] = config.get("features", {}).get("categorical", [])
        self.mandatory_features: List[str] = config.get("features", {}).get("mandatory_features", [])
        self.min_rows: int = int(config.get("validation", {}).get("min_rows", 10))
        self.max_missing_ratio: float = float(config.get("validation", {}).get("max_missing_ratio", 0.5))

    def validate_dataset(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate raw DataFrame against candidate feature mappings, mandatory constraints,
        and temporal splitting prerequisites.
        """
        if df is None:
            return ValidationReport(
                is_valid=False,
                row_count=0,
                column_count=0,
                errors=["Dataset is None."]
            )

        row_count, col_count = len(df), len(df.columns)
        errors: List[str] = []
        warnings: List[str] = []
        missing_cols: List[str] = []
        available_features: List[str] = []
        missing_features: List[str] = []

        # 1. Row count validation
        if row_count == 0:
            errors.append("Dataset contains 0 rows (empty dataset).")
        elif row_count < self.min_rows:
            errors.append(f"Dataset has {row_count} rows, which is fewer than required minimum ({self.min_rows}).")

        # 2. Duplicate rows check
        dup_count = int(df.duplicated().sum()) if row_count > 0 else 0
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows ({round(dup_count / row_count * 100, 2)}%).")

        raw_cols = set(df.columns)

        # 3. Candidate Feature Identification
        for k, v in self.col_map.items():
            if v in raw_cols:
                available_features.append(k)
            elif k in raw_cols:
                available_features.append(v)
            else:
                missing_features.append(k)

        # 4. Mandatory Features Check
        for req_feat in self.mandatory_features:
            raw_req_name = self.col_map.get(req_feat, req_feat)
            if raw_req_name not in raw_cols and req_feat not in raw_cols:
                missing_cols.append(raw_req_name)
                errors.append(
                    f"Configured mandatory feature '{req_feat}' (expected in column '{raw_req_name}') "
                    f"was not found in dataset columns: {list(raw_cols)}"
                )

        # 5. Core Feature Requirement: At least one configured numerical feature MUST exist
        available_num_features = []
        for feat in self.num_features:
            if feat in raw_cols:
                available_num_features.append(feat)
                continue
            for k, v in self.col_map.items():
                if (k == feat and v in raw_cols) or (v == feat and k in raw_cols):
                    available_num_features.append(feat)
                    break
        if not available_num_features:
            errors.append(
                f"Dataset does not contain any configured numerical features. "
                f"At least one valid numerical feature (e.g., COD, BOD, pH, TDS, temperature, "
                f"production_rate) is required to train the model. "
                f"Configured candidate numericals: {self.num_features}. "
                f"Found dataset columns: {list(raw_cols)}"
            )

        # 6. Check missing values rate per present mapped column
        high_missing: Dict[str, float] = {}
        for canonical_feat in available_features:
            raw_col = self.col_map.get(canonical_feat, canonical_feat)
            if raw_col in df.columns and row_count > 0:
                is_fully_empty = bool(df[raw_col].isnull().all())
                null_ratio = df[raw_col].isnull().sum() / row_count
                if null_ratio > self.max_missing_ratio:
                    high_missing[raw_col] = round(null_ratio * 100, 2)
                    # Fully empty optional candidate columns are treated as unmeasured (warning)
                    if is_fully_empty and canonical_feat not in self.mandatory_features and raw_col not in self.mandatory_features:
                        warnings.append(
                            f"Candidate column '{raw_col}' is 100% empty (unmeasured in dataset) and will be excluded from active features."
                        )
                    else:
                        errors.append(
                            f"Column '{raw_col}' (feature '{canonical_feat}') has {high_missing[raw_col]}% missing values, "
                            f"exceeding allowable maximum of {self.max_missing_ratio * 100}%."
                        )

        # 7. Type validation for available numerical features
        for feat in available_num_features:
            raw_col = self.col_map.get(feat, feat)
            if raw_col in df.columns:
                non_numeric_cnt = pd.to_numeric(df[raw_col], errors="coerce").isnull().sum() - df[raw_col].isnull().sum()
                if non_numeric_cnt > 0:
                    errors.append(
                        f"Numerical feature '{raw_col}' contains {non_numeric_cnt} non-numeric values."
                    )

        # 8. Chronological Splitting Validation Prerequisite
        # Only validate timestamp if chronological splitting is EXPLICITLY requested in config
        split_strategy = self.config.get("splitting", {}).get("strategy")
        if split_strategy == "chronological":
            ts_col = self.config.get("splitting", {}).get("timestamp_column", "timestamp")
            if ts_col not in raw_cols:
                missing_cols.append(ts_col)
                errors.append(
                    f"Chronological splitting strategy is selected, but required timestamp column "
                    f"'{ts_col}' was not found in dataset columns: {list(raw_cols)}. "
                    f"Configure 'splitting.timestamp_column' or set strategy to 'random' if data is independent."
                )
            else:
                null_ts = df[ts_col].isnull().sum()
                if null_ts > 0:
                    errors.append(
                        f"Timestamp column '{ts_col}' contains {null_ts} missing values. "
                        f"Chronological splitting requires complete timestamps."
                    )
                coerced_ts = pd.to_datetime(df[ts_col], errors="coerce")
                bad_ts_cnt = coerced_ts.isnull().sum() - null_ts
                if bad_ts_cnt > 0:
                    errors.append(
                        f"Timestamp column '{ts_col}' contains {bad_ts_cnt} unparseable datetime entries."
                    )

        # 9. Label Hygiene Validation
        label_col = self.config.get("labels", {}).get("label_column")
        if label_col:
            if label_col not in raw_cols:
                missing_cols.append(label_col)
                errors.append(
                    f"Configured label column '{label_col}' was not found in dataset columns: {list(raw_cols)}."
                )
            else:
                norm_vals = set(self.config.get("labels", {}).get("normal_values", DEFAULT_NORMAL_LABELS))
                anom_vals = set(self.config.get("labels", {}).get("anomaly_values", DEFAULT_ANOMALY_LABELS))
                custom_map = self.config.get("labels", {}).get("custom_mapping") or {}

                unrecognized: List[Any] = []
                for val in df[label_col].dropna().unique():
                    try:
                        parse_label_value(val, norm_vals, anom_vals, custom_map)
                    except ValueError:
                        unrecognized.append(val)

                if unrecognized:
                    errors.append(
                        f"Label column '{label_col}' contains unrecognized or ambiguous label values: {unrecognized}. "
                        f"Must be mapped to 0 (normal) or 1 (anomaly) via 'labels.custom_mapping' or standard labels."
                    )

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            row_count=row_count,
            column_count=col_count,
            errors=errors,
            warnings=warnings,
            missing_columns=missing_cols,
            available_candidate_features=available_features,
            missing_candidate_features=missing_features,
            high_missingness_columns=high_missing,
            duplicate_rows=dup_count
        )

    def validate_feature_matrix(self, X: np.ndarray, expected_cols: Optional[int] = None) -> None:
        """
        Validate final processed numerical feature matrix prior to Isolation Forest fitting or scoring.
        """
        if X is None:
            raise DataValidationError("Feature matrix X cannot be None.")

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.ndim != 2:
            raise DataValidationError(f"Expected 2D feature matrix, got shape {X.shape}.")

        if X.shape[0] == 0:
            raise DataValidationError("Feature matrix has 0 samples.")

        if expected_cols is not None and X.shape[1] != expected_cols:
            raise DataValidationError(
                f"Feature matrix dimension mismatch: expected {expected_cols} columns, got {X.shape[1]}."
            )

        if np.isnan(X).any():
            raise DataValidationError("Feature matrix contains unhandled NaN values.")

        if np.isinf(X).any():
            raise DataValidationError("Feature matrix contains Infinite values.")
