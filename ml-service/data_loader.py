"""
EnviroTrust ML Service — Data Ingestion Module

Provides robust loading and profiling for CSV and Excel industrial datasets.
Does not make assumptions about authentic dataset schema before inspection.
Handles missing files and corrupted tables gracefully without hallucinating
or inventing missing measurements.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnviroTrust.DataLoader")


class DataLoadingError(Exception):
    """Raised when a dataset cannot be loaded or parsed."""
    pass


def load_config(config_path: Union[str, Path, Dict[str, Any]] = "config/data_config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file or return dictionary if already loaded.

    Parameters:
        config_path: Path to YAML configuration, or already-parsed config dict.

    Returns:
        Dict containing configuration values.
    """
    if isinstance(config_path, dict):
        return config_path

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file.resolve()}")

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {cfg_file.name}")
        return config or {}
    except Exception as e:
        raise DataLoadingError(f"Error parsing YAML configuration at {cfg_file}: {e}") from e


def load_dataset(
    file_path: Union[str, Path],
    file_format: str = "auto",
    sheet_name: Optional[Union[str, int]] = 0,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load dataset from CSV or Excel file.

    Parameters:
        file_path: Path to the dataset file.
        file_format: 'csv', 'excel', or 'auto' (inferred from suffix).
        sheet_name: Excel sheet to read if Excel format (defaults to first sheet).
        **kwargs: Additional keyword arguments passed to pd.read_csv / pd.read_excel.

    Returns:
        pd.DataFrame containing raw data as read from the source file.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataLoadingError: If the file is unreadable, empty, or format is unsupported.
    """
    path = Path(file_path)

    if not path.exists():
        msg = f"Dataset file does not exist at path: {path.resolve()}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = f"Specified path is a directory, not a file: {path.resolve()}"
        logger.error(msg)
        raise DataLoadingError(msg)

    # Infer format if auto
    fmt = file_format.lower()
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in [".csv", ".txt"]:
            fmt = "csv"
        elif suffix in [".xlsx", ".xls", ".xlsm"]:
            fmt = "excel"
        else:
            raise DataLoadingError(
                f"Unsupported file extension '{suffix}'. Expected CSV (.csv) or Excel (.xlsx, .xls)."
            )

    try:
        if fmt == "csv":
            logger.info(f"Loading CSV dataset from: {path.resolve()}")
            df = pd.read_csv(path, **kwargs)
        elif fmt == "excel":
            logger.info(f"Loading Excel dataset from: {path.resolve()} (sheet: {sheet_name})")
            df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        else:
            raise DataLoadingError(f"Unknown format specification: {file_format}")

        if df.empty:
            logger.warning(f"Loaded dataset from {path.name} is empty (0 rows).")

        return df

    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        msg = f"Failed to parse dataset file '{path.name}': {e}"
        logger.error(msg)
        raise DataLoadingError(msg) from e
    except Exception as e:
        msg = f"Unexpected error reading dataset from '{path.name}': {e}"
        logger.error(msg)
        raise DataLoadingError(msg) from e


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect a loaded dataset without assuming any predetermined schema.
    Computes summary metrics, missing value rates, data types, and duplicate rows.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        Dictionary containing inspection metadata and profiling statistics.
    """
    if df is None:
        raise ValueError("DataFrame cannot be None for inspection.")

    row_count = len(df)
    col_count = len(df.columns)

    # Missing value analysis
    missing_counts = df.isnull().sum().to_dict()
    missing_percentages = {
        col: round((count / row_count) * 100, 2) if row_count > 0 else 0.0
        for col, count in missing_counts.items()
    }

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = round((duplicate_count / row_count) * 100, 2) if row_count > 0 else 0.0

    # Column classification
    numerical_cols: List[str] = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols: List[str] = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols: List[str] = df.select_dtypes(include=["datetime"]).columns.tolist()

    # Column dtypes map
    dtypes_map = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    # Memory usage
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)

    return {
        "row_count": row_count,
        "column_count": col_count,
        "columns": [str(c) for c in df.columns],
        "dtypes": dtypes_map,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "memory_usage_mb": memory_mb,
    }


def format_inspection_report(profile: Dict[str, Any]) -> str:
    """
    Format the inspection dictionary into a clean human-readable text table.
    """
    lines = [
        "=================================================================",
        "           ENVIROTRUST DATASET INSPECTION REPORT",
        "=================================================================",
        f"Total Records: {profile['row_count']:,} | Total Columns: {profile['column_count']}",
        f"Duplicate Rows: {profile['duplicate_rows']} ({profile['duplicate_percentage']}%)",
        f"Memory Footprint: {profile['memory_usage_mb']} MB",
        "-----------------------------------------------------------------",
        f"{'Column Name':<28} {'Type':<12} {'Missing':<10} {'Missing %':<10}",
        "-----------------------------------------------------------------"
    ]

    for col in profile["columns"]:
        col_type = profile["dtypes"].get(col, "unknown")
        miss_cnt = profile["missing_counts"].get(col, 0)
        miss_pct = profile["missing_percentages"].get(col, 0.0)
        lines.append(f"{col:<28} {col_type:<12} {miss_cnt:<10} {miss_pct:<10}%")

    lines.append("=================================================================")
    return "\n".join(lines)


DEFAULT_PARAMETER_NAME_MAP: Dict[str, str] = {
    "pH": "pH",
    "Turbidity (NTU)": "turbidity_NTU",
    "Colour (Pt. Co.)": "colour_PtCo",
    "Total Solids (mg/l)": "total_solids_mg_L",
    "Total Suspended Solids (mg/l)": "total_suspended_solids_mg_L",
    "Chemical Oxygen Demand (COD) (mg/l)": "COD_mg_L",
    "Biochemical Oxygen Demand (BOD)": "BOD_mg_L",
    "Total Dissolved Solids (TDS) (mg/l)": "TDS_mg_L",
    "Chloride (mg/l)": "chloride_mg_L",
    "Ammoniacal Nitrogen (mg/l)": "ammoniacal_nitrogen_mg_L",
    "Nitrate (mg/l)": "nitrate_mg_L",
    "Sulphate (mg/l)": "sulphate_mg_L",
    "Electricity Consumption": "electricity_consumption",
}

DEFAULT_SAMPLE_ID_MAP: Dict[str, str] = {
    "Textile Industrial Effluent": "Raw_Dye_Wastewater_1",
    "Dye-Intermediate Effluent": "Raw_Dye_Wastewater_2",
}


def transpose_parameter_table(
    df: pd.DataFrame,
    parameter_column: str = "Parameter",
    testing_method_column: Optional[str] = "Testing Method",
    parameter_name_map: Optional[Dict[str, str]] = None,
    sample_id_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Standardize and transpose a parameter-oriented table (where parameters are rows
    and samples are columns, common in environmental literature and laboratory test reports)
    into an observation-oriented DataFrame (where each row is a sample observation).

    Parameters:
        df: Input DataFrame containing parameters as rows.
        parameter_column: Name of the column containing parameter names.
        testing_method_column: Name of optional column containing laboratory testing methods.
        parameter_name_map: Mapping from raw parameter strings to canonical column names.
        sample_id_map: Mapping from sample column headers to standardized sample IDs.

    Returns:
        Tuple of:
          - pd.DataFrame: Transposed observation matrix with 'sample_id' and parameter columns.
          - Dict: Parameter-level metadata (testing method, raw parameter title, etc.).
    """
    if parameter_column not in df.columns:
        raise DataLoadingError(
            f"Expected parameter column '{parameter_column}' not found. Available: {list(df.columns)}"
        )

    p_map = parameter_name_map or DEFAULT_PARAMETER_NAME_MAP
    s_map = sample_id_map or DEFAULT_SAMPLE_ID_MAP

    # Identify metadata columns vs sample columns
    exclude_cols = {parameter_column}
    if testing_method_column and testing_method_column in df.columns:
        exclude_cols.add(testing_method_column)

    # Any difference/variance columns (e.g. '% Difference...') are analytical metadata
    for col in df.columns:
        if "%" in col or "difference" in col.lower() or "method" in col.lower():
            exclude_cols.add(col)

    sample_columns = [col for col in df.columns if col not in exclude_cols]
    if not sample_columns:
        raise DataLoadingError("No sample columns identified in parameter table.")

    # Extract parameter metadata (e.g., testing methods)
    metadata: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        raw_param = str(row[parameter_column]).strip()
        canonical_name = p_map.get(raw_param, raw_param)
        meta_entry: Dict[str, Any] = {"raw_name": raw_param}
        if testing_method_column and testing_method_column in df.columns:
            method_val = row[testing_method_column]
            meta_entry["testing_method"] = str(method_val).strip() if pd.notna(method_val) else None
        metadata[canonical_name] = meta_entry

    # Construct observations row-by-row per sample
    records: List[Dict[str, Any]] = []
    for sample_col in sample_columns:
        sample_id = s_map.get(sample_col, sample_col)
        record: Dict[str, Any] = {"sample_id": sample_id}

        for _, row in df.iterrows():
            raw_param = str(row[parameter_column]).strip()
            canonical_name = p_map.get(raw_param, raw_param)
            raw_val = row[sample_col]

            # Parse value safely without inventing data
            if pd.isna(raw_val) or str(raw_val).strip() in ("N/A", "NA", "-", ""):
                record[canonical_name] = np.nan
            else:
                try:
                    record[canonical_name] = float(raw_val)
                except (ValueError, TypeError):
                    record[canonical_name] = raw_val

        records.append(record)

    transposed_df = pd.DataFrame(records)
    logger.info(
        f"Transposed parameter table: {len(sample_columns)} samples, "
        f"{len(transposed_df.columns) - 1} parameters extracted."
    )
    return transposed_df, metadata
