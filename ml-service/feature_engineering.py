"""
EnviroTrust ML Service — Feature Engineering & Mapping Module

Translates raw authentic dataset columns into standardized conceptual features.
Supports both Environmental-Only (COD, BOD, pH, TDS, Temperature, Turbidity, etc.)
and Process-Context (incorporating flow, production rate, electricity consumption) configurations.
Excludes identifier columns and completely empty (100% NaN) columns from active feature matrices.
"""

from typing import Any, Dict, List, Optional, Set
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("EnviroTrust.FeatureEngineering")


class FeatureEngineeringPipeline:
    """
    Applies configurable column mappings and feature transformations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.col_map: Dict[str, str] = config.get("column_mapping", {})
        self.num_features: List[str] = config.get("features", {}).get("numerical", [])
        self.cat_features: List[str] = config.get("features", {}).get("categorical", [])
        self.id_columns: Set[str] = set(
            config.get("features", {}).get("id_columns", ["sample_id", "readingId", "sensorId"])
        )
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

    def process(self, df: pd.DataFrame, include_ratios: bool = False) -> pd.DataFrame:
        """
        Full feature engineering pipeline execution.
        """
        mapped_df = self.apply_column_mapping(df)
        if include_ratios:
            mapped_df = self.add_domain_ratios(mapped_df)
        selected_df = self.select_features(mapped_df, strict=False)
        return selected_df
