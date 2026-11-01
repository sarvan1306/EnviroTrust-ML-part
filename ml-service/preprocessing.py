"""
EnviroTrust ML Service — Preprocessing Pipeline

Provides reusable scikit-learn transformers for numerical imputation, categorical
one-hot encoding, and feature scaling. Strictly isolates fitting to training datasets
to prevent data leakage into validation or inference phases.
Preserves the exact training schema and prevents silent fabrication of missing features during inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import rolling_feature_names

logger = logging.getLogger("EnviroTrust.Preprocessing")


class PreprocessingError(Exception):
    """Raised during preprocessing transformation failures."""
    pass


class MissingFeatureError(PreprocessingError):
    """Raised when an inference sample is missing features present in the trained model schema."""
    pass


class DataPreprocessor:
    """
    Modular preprocessor for industrial tabular telemetry.
    Saves the exact training feature schema and strictly prevents silent imputation of missing columns.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        prep_cfg = config.get("preprocessing", {})
        self.num_strategy = prep_cfg.get("numerical_imputation_strategy", "median")
        self.cat_strategy = prep_cfg.get("categorical_imputation_strategy", "most_frequent")
        self.cat_encoding = prep_cfg.get("categorical_encoding", "onehot")
        self.scale_features = prep_cfg.get("scale_features", False)

        self.configured_num_cols: List[str] = list(config.get("features", {}).get("numerical", []))
        self.configured_num_cols += [f for f in rolling_feature_names(config) if f not in self.configured_num_cols]
        self.configured_cat_cols: List[str] = config.get("features", {}).get("categorical", [])

        # State tracked after fitting
        self.is_fitted: bool = False
        self.active_num_cols: List[str] = []
        self.active_cat_cols: List[str] = []
        self.feature_names_out: List[str] = []
        self.training_schema: Dict[str, Any] = {}
        self.transformer: Optional[ColumnTransformer] = None

    def _build_pipeline(self) -> ColumnTransformer:
        """Construct ColumnTransformer with scikit-learn standard components."""
        transformers = []

        if self.active_num_cols:
            if self.scale_features:
                from sklearn.pipeline import Pipeline
                num_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy=self.num_strategy)),
                    ("scaler", StandardScaler()),
                ])
            else:
                num_pipe = SimpleImputer(strategy=self.num_strategy)
            transformers.append(("num", num_pipe, self.active_num_cols))

        if self.active_cat_cols:
            cat_pipe = [
                ("imputer", SimpleImputer(strategy=self.cat_strategy, fill_value="UNKNOWN")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            from sklearn.pipeline import Pipeline
            transformers.append(("cat", Pipeline(cat_pipe), self.active_cat_cols))

        if not transformers:
            raise PreprocessingError("No numerical or categorical features identified for preprocessing.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessor exclusively on training data and preserve schema.
        """
        if df is None or df.empty:
            raise PreprocessingError("Cannot fit preprocessor on an empty DataFrame.")

        # Determine which configured columns are present in training dataset
        self.active_num_cols = [c for c in self.configured_num_cols if c in df.columns]
        self.active_cat_cols = [c for c in self.configured_cat_cols if c in df.columns]

        if not self.active_num_cols:
            raise PreprocessingError(
                f"No configured numerical features were found in training data. "
                f"Configured num: {self.configured_num_cols}. Present columns: {list(df.columns)}"
            )

        logger.info(
            f"Fitting preprocessor with {len(self.active_num_cols)} numerical features: {self.active_num_cols} "
            f"and {len(self.active_cat_cols)} categorical features: {self.active_cat_cols}"
        )

        self.transformer = self._build_pipeline()
        self.transformer.fit(df)
        self.is_fitted = True

        # Extract output feature names
        try:
            self.feature_names_out = list(self.transformer.get_feature_names_out())
        except Exception:
            self.feature_names_out = self.active_num_cols + self.active_cat_cols

        self.training_schema = {
            "numerical_features": list(self.active_num_cols),
            "categorical_features": list(self.active_cat_cols),
            "feature_names_out": list(self.feature_names_out),
            "feature_count": len(self.feature_names_out),
        }

        return self

    def transform(self, df: pd.DataFrame, allow_missing_columns: bool = False) -> np.ndarray:
        """
        Transform new data using parameters learned during fit.
        Guarantees that inference or validation data is never used to compute statistics.

        Parameters:
            df: Input DataFrame.
            allow_missing_columns: If False (default for inference), raises MissingFeatureError
                                   if any required training column is entirely absent.
        """
        if not self.is_fitted or self.transformer is None:
            raise PreprocessingError("Preprocessor is not fitted yet. Call fit() or load() first.")

        if df is None or df.empty:
            raise PreprocessingError("Cannot transform an empty DataFrame.")

        # Verify all required training features exist in incoming data
        if not allow_missing_columns:
            missing_num = [c for c in self.active_num_cols if c not in df.columns]
            missing_cat = [c for c in self.active_cat_cols if c not in df.columns]
            all_missing = missing_num + missing_cat
            if all_missing:
                raise MissingFeatureError(
                    f"Missing required model features in incoming record: {all_missing}. "
                    f"The active model requires all training features: {self.active_num_cols + self.active_cat_cols}. "
                    f"Features will not be silently fabricated."
                )

        df_copy = df.copy()
        if allow_missing_columns:
            for col in self.active_num_cols:
                if col not in df_copy.columns:
                    df_copy[col] = np.nan
            for col in self.active_cat_cols:
                if col not in df_copy.columns:
                    df_copy[col] = "UNKNOWN"

        try:
            matrix = self.transformer.transform(df_copy)
            return np.asarray(matrix, dtype=np.float64)
        except Exception as e:
            raise PreprocessingError(f"Error during preprocessing transformation: {e}") from e

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience method for training workflow."""
        return self.fit(df).transform(df, allow_missing_columns=False)

    def get_feature_names_out(self) -> List[str]:
        """Return the names of transformed features."""
        return self.feature_names_out

    def get_training_schema(self) -> Dict[str, Any]:
        """Return the exact schema recorded during training."""
        return dict(self.training_schema)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize preprocessor to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor artifact to: {path.resolve()}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DataPreprocessor":
        """Load serialized preprocessor from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {path.resolve()}")
        preprocessor = joblib.load(path)
        if not isinstance(preprocessor, cls):
            raise TypeError(f"Loaded object is of type {type(preprocessor)}, expected {cls}.")
        logger.info(f"Loaded preprocessor artifact from: {path.resolve()}")
        return preprocessor
