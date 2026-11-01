"""
EnviroTrust ML Service — Feature Engineering & Mapping Module

Translates raw authentic dataset columns into standardized conceptual features.
Supports both Environmental-Only (COD, BOD, pH, TDS, Temperature, Turbidity, etc.)
and Process-Context (incorporating flow, production rate, electricity consumption) configurations.
Excludes identifier columns and completely empty (100% NaN) columns from active feature matrices.

Rolling-window features (stuck-sensor detection):
  A single reading cannot reveal a frozen sensor: a stuck value can sit at a perfectly
  plausible level. For each base measurement, log1p of the trailing-window standard
  deviation is added per sensor stream. A stuck sensor drives it to exactly 0, far below
  the natural noise floor seen in training, which Isolation Forest isolates quickly.
  The log scale matters: raw std spans hundreds of mg/L, so a 0 sits close to the
  training range on a linear axis and is rarely isolated.
"""

from typing import Any, Dict, List, Optional, Set
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("EnviroTrust.FeatureEngineering")

ROLLING_SUFFIX = "_roll_log_std"
DEFAULT_STREAM_KEY = "__default__"


def rolling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolved rolling-window settings with defaults applied."""
    cfg = config.get("rolling_features") or {}
    base = cfg.get("base_features") or config.get("features", {}).get("numerical", [])
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "window": int(cfg.get("window", 5)),
        "base_features": [f for f in base if not f.endswith(ROLLING_SUFFIX)],
        "group_column": cfg.get("group_column", "sensorId"),
        "timestamp_column": cfg.get("timestamp_column", "timestamp"),
    }


def rolling_feature_names(config: Dict[str, Any]) -> List[str]:
    """Candidate rolling feature names (empty when rolling features are disabled)."""
    roll_cfg = rolling_config(config)
    if not roll_cfg["enabled"]:
        return []
    return [f"{f}{ROLLING_SUFFIX}" for f in roll_cfg["base_features"]]


class FeatureEngineeringPipeline:
    """
    Applies configurable column mappings and feature transformations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.col_map: Dict[str, str] = config.get("column_mapping", {})
        self.num_features: List[str] = list(config.get("features", {}).get("numerical", []))
        self.cat_features: List[str] = config.get("features", {}).get("categorical", [])
        self.id_columns: Set[str] = set(
            config.get("features", {}).get("id_columns", ["sample_id", "readingId", "sensorId"])
        )

        self.rolling_cfg: Dict[str, Any] = rolling_config(config)
        self.rolling_features: List[str] = rolling_feature_names(config)
        self.num_features += [f for f in self.rolling_features if f not in self.num_features]
        self.all_target_features: List[str] = self.num_features + self.cat_features

    def apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns in df from raw authentic names to standardized conceptual feature names.
        Only renames columns that exist in df and are present in column_mapping.
        Does not create duplicate target columns if target name already exists.
        """
        rename_dict = {}
        for a, b in self.col_map.items():
            canonical = a if a in self.all_target_features else (b if b in self.all_target_features else a)
            raw = b if canonical == a else a
            if raw in df.columns and canonical not in df.columns:
                rename_dict[raw] = canonical
        if rename_dict:
            logger.debug(f"Applying column mapping: {rename_dict}")
            mapped_df = df.rename(columns=rename_dict).copy()
        else:
            mapped_df = df.copy()

        return mapped_df

    def select_features(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """
        Extract configured active numerical and categorical candidate features.
        Strictly excludes identifier columns (e.g. sample_id).
        Excludes completely empty columns (100% NaN) from active modeling features.
        Requires at least one valid numerical feature with non-empty measurements.
        """
        mapped_df = self.apply_column_mapping(df)

        # Identify candidate features present in mapped DataFrame (excluding ID columns)
        candidate_cols = [
            col for col in self.all_target_features
            if col in mapped_df.columns and col not in self.id_columns
        ]

        # Exclude completely empty columns (100% NaN values)
        available_features = [
            col for col in candidate_cols
            if not mapped_df[col].isna().all()
        ]

        missing_features = [col for col in self.all_target_features if col not in mapped_df.columns]
        empty_features = [col for col in candidate_cols if mapped_df[col].isna().all()]

        if missing_features:
            logger.debug(f"Optional candidate features not present in dataset: {missing_features}")
        if empty_features:
            logger.info(f"Excluding completely empty columns from active feature matrix: {empty_features}")

        if not available_features:
            raise ValueError(
                f"No configured candidate features with measurements found in dataset. "
                f"Candidate features: {self.all_target_features}. "
                f"Present columns: {list(mapped_df.columns)}"
            )

        # Ensure at least one numerical feature is selected
        available_nums = [
            col for col in self.num_features
            if col in available_features and col not in self.id_columns
        ]
        if not available_nums:
            raise ValueError(
                f"No numerical features with non-empty data found in dataset. "
                f"At least one numerical feature is required. Configured numericals: {self.num_features}"
            )

        return mapped_df[available_features].copy()

    def add_domain_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally compute physical domain ratios if the underlying columns exist:
          - flow_to_production_ratio: wastewater_flow / production_rate
        """
        out_df = df.copy()

        flow_col = "water_flow" if "water_flow" in out_df.columns else "wastewater_flow"
        prod_col = "production" if "production" in out_df.columns else "production_rate"

        if flow_col in out_df.columns and prod_col in out_df.columns:
            prod = pd.to_numeric(out_df[prod_col], errors="coerce")
            flow = pd.to_numeric(out_df[flow_col], errors="coerce")
            valid_mask = (prod > 0) & prod.notnull() & flow.notnull()
            out_df["flow_to_production_ratio"] = np.nan
            out_df.loc[valid_mask, "flow_to_production_ratio"] = flow[valid_mask] / prod[valid_mask]

        return out_df

    def add_rolling_features(self, df: pd.DataFrame, require_timestamp: bool = True) -> pd.DataFrame:
        """
        Append '<feature>_roll_log_std' columns: log1p of the trailing-window standard
        deviation of each base measurement, computed per sensor stream (group_column) in
        timestamp order. Windows look backwards only, so no future values leak in.

        The first window-1 readings of a stream, and any window containing a missing value,
        get NaN (imputed downstream like any other missing value).

        Must run on the complete chronological stream, before label filtering or splitting:
        dropping rows first would make windows span gaps.

        Parameters:
            df: Input DataFrame (raw or mapped column names). Returned with original index.
            require_timestamp: If True (training/evaluation), skip when timestamp_column is
                               absent, since row order of non-temporal data is meaningless.
                               Inference passes False: buffered readings are in arrival order.
        """
        out_df = df.copy()
        if not self.rolling_features or df.empty:
            return out_df

        ts_col = self.rolling_cfg["timestamp_column"]
        if require_timestamp and ts_col not in df.columns:
            logger.info(f"Rolling features skipped: timestamp column '{ts_col}' not present (non-temporal dataset).")
            return out_df

        mapped = self.apply_column_mapping(df).reset_index(drop=True)
        bases = [f for f in self.rolling_cfg["base_features"] if f in mapped.columns]
        if not bases:
            return out_df

        if ts_col in mapped.columns:
            ts = pd.to_datetime(mapped[ts_col], errors="coerce", utc=True)
            mapped = mapped.loc[ts.sort_values(kind="mergesort").index]

        group_col = self.rolling_cfg["group_column"]
        if group_col in mapped.columns:
            keys = mapped[group_col].fillna(DEFAULT_STREAM_KEY).astype(str)
        else:
            keys = pd.Series(DEFAULT_STREAM_KEY, index=mapped.index)

        window = self.rolling_cfg["window"]
        values = mapped[bases].apply(pd.to_numeric, errors="coerce")
        rolled = values.groupby(keys).transform(
            lambda s: s.rolling(window, min_periods=window).std(ddof=0)
        )
        rolled = np.log1p(rolled).sort_index()

        for base in bases:
            out_df[f"{base}{ROLLING_SUFFIX}"] = rolled[base].to_numpy()

        return out_df

    def process(self, df: pd.DataFrame, include_ratios: bool = False) -> pd.DataFrame:
        """
        Full feature engineering pipeline execution.
        """
        mapped_df = self.apply_column_mapping(df)
        if include_ratios:
            mapped_df = self.add_domain_ratios(mapped_df)
        selected_df = self.select_features(mapped_df, strict=False)
        return selected_df
