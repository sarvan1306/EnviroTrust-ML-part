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
import re
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
    # Ontario MISA 2024 Electric Power Generation Mappings
    "BIOCHEMICAL OXYGEN DEMAND, 5 DAY, TOTAL DEMAND": "BOD_kg_d",
    "FLOW": "water_flow_m3_d",
    "ALUMINIUM, UNFILTERED TOTAL": "aluminium_kg_d",
    "IRON, UNFILTERED TOTAL": "iron_kg_d",
    "RESIDUE, PARTICULATE": "residue_particulate_kg_d",
    "SOLVENT EXTRACTABLES": "solvent_extractables_kg_d",
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


def pivot_eav_effluent_table(
    df: pd.DataFrame,
    entity_column: str = "Works Name",
    parameter_column: str = "Parameter Name",
    value_column: str = "Value",
    unit_column: Optional[str] = "Unit of Measure",
    date_column: Optional[str] = "Sample Date",
    parameter_name_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Pivot an Entity-Attribute-Value (EAV) / Long format effluent monitoring table
    (such as Ontario MISA regulatory reporting tables) into an observation-oriented
    DataFrame where each row is an entity observation and each parameter is a column.

    Parameters:
        df: Input DataFrame containing long-format parameter rows.
        entity_column: Column identifying the facility/works name.
        parameter_column: Column identifying the parameter name.
        value_column: Column containing the numerical measurement value.
        unit_column: Optional column containing units of measure.
        date_column: Optional column containing sample date/timestamp.
        parameter_name_map: Optional mapping from raw parameter strings to canonical names.

    Returns:
        Tuple of:
          - pd.DataFrame: Pivoted observation matrix.
          - Dict: Parameter-level metadata (units, sample dates, raw names).
    """
    if entity_column not in df.columns:
        raise DataLoadingError(f"Expected entity column '{entity_column}' not found. Available: {list(df.columns)}")
    if parameter_column not in df.columns:
        raise DataLoadingError(f"Expected parameter column '{parameter_column}' not found. Available: {list(df.columns)}")
    if value_column not in df.columns:
        raise DataLoadingError(f"Expected value column '{value_column}' not found. Available: {list(df.columns)}")

    p_map = parameter_name_map or DEFAULT_PARAMETER_NAME_MAP

    # Extract parameter metadata (units, dates, raw names)
    metadata: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        raw_p = str(row[parameter_column]).strip()
        canon_p = p_map.get(raw_p, raw_p)
        if canon_p not in metadata:
            metadata[canon_p] = {
                "raw_name": raw_p,
                "unit": str(row[unit_column]).strip() if unit_column and unit_column in df.columns and pd.notna(row[unit_column]) else None,
                "sample_date": str(row[date_column]).strip() if date_column and date_column in df.columns and pd.notna(row[date_column]) else None,
            }

    # Index columns to preserve during pivoting
    index_cols = [entity_column]
    if date_column and date_column in df.columns:
        index_cols.append(date_column)
    for extra_col in ["Sector", "Control Point Name", "Component Type"]:
        if extra_col in df.columns and extra_col not in index_cols:
            index_cols.append(extra_col)

    # Pivot table
    pivoted_df = df.pivot_table(
        index=index_cols,
        columns=parameter_column,
        values=value_column,
        aggfunc="first"
    ).reset_index()

    # Rename parameter columns using mapping
    rename_cols = {col: p_map.get(col, col) for col in pivoted_df.columns if col not in index_cols}
    pivoted_df = pivoted_df.rename(columns=rename_cols)

    logger.info(
        f"Pivoted EAV effluent table: {len(pivoted_df)} facility observations, "
        f"{len(pivoted_df.columns) - len(index_cols)} parameters extracted."
    )
    return pivoted_df, metadata


# ==============================================================================
# Cross-Industry & WWTP Dataset Ingestion Helpers
# ==============================================================================

def parse_midpoint_range(val: Any) -> float:
    """
    Convert string ranges (e.g., '2000–11,588', '6.3–6.98', '1200–4000 (BOD5)', '<10')
    into clean numeric midpoints for analytical and modelling purposes.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ["-", "<10", "", "N/A*", "N/A"]:
        return np.nan
    # Remove annotations like (BOD5), (KN), (Influent), (25-75%)
    val_clean = re.sub(r"\([^)]*\)", "", s).replace(",", "").strip()
    # Handle en-dash or hyphen ranges
    if "–" in val_clean or "-" in val_clean:
        parts = re.split(r"–|-", val_clean)
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            return float(np.mean(nums)) if nums else np.nan
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(val_clean)
    except (ValueError, TypeError):
        return np.nan


def parse_microorganism_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse the literature cross-industry wastewater dataset (microorganism.csv).
    Converts string parameter ranges into numeric midpoints and aligns them
    with canonical EnviroTrust feature nomenclature.

    Parameters:
        df: Raw DataFrame loaded from microorganism.csv

    Returns:
        Tuple of:
          - pd.DataFrame: Cleaned DataFrame with 27 industrial streams and numeric parameters
          - Dict[str, Any]: Metadata containing raw ranges, non-null counts, and sector classifications
    """
    required_col = "Wastewater Type"
    if required_col not in df.columns:
        raise DataLoadingError(f"Expected column '{required_col}' not found. Available: {list(df.columns)}")

    cleaned_df = pd.DataFrame()
    cleaned_df["wastewater_type"] = df["Wastewater Type"].astype(str).str.strip()

    col_mappings = {
        "COD (mg/L)": "COD_mg_L",
        "BOD (mg/L)": "BOD_mg_L",
        "TS (mg/L)": "total_solids_mg_L",
        "SS (mg/L)": "total_suspended_solids_mg_L",
        "VSS (mg/L)": "volatile_suspended_solids_mg_L",
        "TN (mg/L)": "total_nitrogen_mg_L",
        "TP (mg/L)": "total_phosphorus_mg_L",
        "N-NH4+ (mg/L)": "ammoniacal_nitrogen_mg_L",
        "pH": "pH",
    }

    metadata: Dict[str, Any] = {
        "total_streams": len(df),
        "raw_columns": list(df.columns),
        "parameter_stats": {},
    }

    for raw_col, canon_col in col_mappings.items():
        if raw_col in df.columns:
            cleaned_df[canon_col] = df[raw_col].apply(parse_midpoint_range)
            metadata["parameter_stats"][canon_col] = {
                "raw_column": raw_col,
                "non_null_count": int(cleaned_df[canon_col].notna().sum()),
                "min": float(cleaned_df[canon_col].min()) if cleaned_df[canon_col].notna().any() else None,
                "max": float(cleaned_df[canon_col].max()) if cleaned_df[canon_col].notna().any() else None,
                "mean": float(cleaned_df[canon_col].mean()) if cleaned_df[canon_col].notna().any() else None,
            }

    if "Reference Citation" in df.columns:
        cleaned_df["reference_citation"] = df["Reference Citation"].astype(str).str.strip()

    # Calculate BOD/COD ratio where both parameters are available
    cleaned_df["bod_cod_ratio"] = cleaned_df["BOD_mg_L"] / cleaned_df["COD_mg_L"]

    logger.info(
        f"Parsed cross-industry microorganism dataset: {len(cleaned_df)} wastewater streams, "
        f"{len(cleaned_df.columns)} standardized columns."
    )
    return cleaned_df, metadata


def parse_china_wwtp_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse the Yellow River Basin industrial WWTP dataset (china.csv).
    Extracts influent water quality concentrations, energy metrics, and operational costs.

    Parameters:
        df: Raw DataFrame loaded from china.csv

    Returns:
        Tuple of:
          - pd.DataFrame: Valid parameters with numeric Average, Median, and Range midpoints
          - Dict[str, Any]: Metadata covering unmeasured parameters (pH, TDS) and unit definitions
    """
    param_col = "Parameter / Metric"
    if param_col not in df.columns:
        raise DataLoadingError(f"Expected column '{param_col}' not found. Available: {list(df.columns)}")

    metadata: Dict[str, Any] = {
        "total_raw_rows": len(df),
        "unmeasured_parameters": [],
        "water_quality_parameters": [],
        "operational_parameters": [],
    }

    # Separate unmeasured parameters
    unmeasured_mask = df["Average Value"].astype(str).str.strip() == "N/A*"
    unmeasured_df = df[unmeasured_mask]
    for _, row in unmeasured_df.iterrows():
        metadata["unmeasured_parameters"].append(str(row[param_col]).strip())

    valid_df = df[~unmeasured_mask].copy()
    valid_df["average_value"] = pd.to_numeric(valid_df["Average Value"], errors="coerce")
    valid_df["median_value"] = pd.to_numeric(valid_df["Median Value"], errors="coerce")

    # Range parsing
    if "25%-75% / Min-Max Range" in valid_df.columns:
        valid_df["range_midpoint"] = valid_df["25%-75% / Min-Max Range"].apply(parse_midpoint_range)

    # Classify parameters
    if "Unit / Notes" in valid_df.columns:
        for _, row in valid_df.iterrows():
            p_name = str(row[param_col]).strip()
            unit_str = str(row["Unit / Notes"])
            if "mg/L" in unit_str:
                metadata["water_quality_parameters"].append(p_name)
            else:
                metadata["operational_parameters"].append(p_name)

    logger.info(
        f"Parsed China WWTP dataset: {len(valid_df)} measured parameters, "
        f"{len(metadata['unmeasured_parameters'])} unmeasured (N/A*)."
    )
    return valid_df, metadata


def parse_mining_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse the industrial wastewater and mining effluent dataset (mining.csv).
    Standardizes water quality parameters (pH, TDS, TOC, Hardness) and associates
    industry activities and critical mineral recovery potential.

    Parameters:
        df: Raw DataFrame loaded from mining.csv

    Returns:
        Tuple of:
          - pd.DataFrame: Standardized DataFrame with 22 industrial activities and numeric parameters
          - Dict[str, Any]: Metadata containing sector counts, parameter ranges, and critical mineral notes
    """
    required_cols = ["Industry Category", "Industrial Activity", "Median pH", "Median TDS (mg/L)"]
    for rc in required_cols:
        if rc not in df.columns:
            raise DataLoadingError(f"Expected column '{rc}' not found in mining dataset. Available: {list(df.columns)}")

    clean_df = pd.DataFrame()
    clean_df["industry_category"] = df["Industry Category"].astype(str).str.strip()
    clean_df["industrial_activity"] = df["Industrial Activity"].astype(str).str.strip()

    # Numeric conversion helper
    def _clean_num(val: Any) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).replace(",", "").strip()
        if s in ["N/A", "NaN", "-", "", "None"]:
            return np.nan
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    clean_df["pH"] = df["Median pH"].apply(_clean_num)
    clean_df["TDS_mg_L"] = df["Median TDS (mg/L)"].apply(_clean_num)

    if "Median TOC (mg/L)" in df.columns:
        clean_df["TOC_mg_L"] = df["Median TOC (mg/L)"].apply(_clean_num)

    if "Hardness (mg/L as CaCO3)" in df.columns:
        clean_df["hardness_mg_L_CaCO3"] = df["Hardness (mg/L as CaCO3)"].apply(_clean_num)

    if "Wastewater Volume / Energy Metric" in df.columns:
        clean_df["wastewater_volume_metric"] = df["Wastewater Volume / Energy Metric"].astype(str).str.strip()

    if "Notes / Critical Mineral Recovery Potential" in df.columns:
        clean_df["notes_recovery_potential"] = df["Notes / Critical Mineral Recovery Potential"].astype(str).str.strip()

    metadata: Dict[str, Any] = {
        "total_activities": len(clean_df),
        "industry_categories": clean_df["industry_category"].value_counts().to_dict(),
        "parameter_stats": {
            "pH": {"non_null": int(clean_df["pH"].notna().sum()), "min": float(clean_df["pH"].min()), "max": float(clean_df["pH"].max())},
            "TDS_mg_L": {"non_null": int(clean_df["TDS_mg_L"].notna().sum()), "min": float(clean_df["TDS_mg_L"].min()), "max": float(clean_df["TDS_mg_L"].max())},
        },
    }

    if "TOC_mg_L" in clean_df.columns:
        metadata["parameter_stats"]["TOC_mg_L"] = {
            "non_null": int(clean_df["TOC_mg_L"].notna().sum()),
            "min": float(clean_df["TOC_mg_L"].min()) if clean_df["TOC_mg_L"].notna().any() else None,
            "max": float(clean_df["TOC_mg_L"].max()) if clean_df["TOC_mg_L"].notna().any() else None,
        }

    if "hardness_mg_L_CaCO3" in clean_df.columns:
        metadata["parameter_stats"]["hardness_mg_L_CaCO3"] = {
            "non_null": int(clean_df["hardness_mg_L_CaCO3"].notna().sum()),
            "min": float(clean_df["hardness_mg_L_CaCO3"].min()) if clean_df["hardness_mg_L_CaCO3"].notna().any() else None,
            "max": float(clean_df["hardness_mg_L_CaCO3"].max()) if clean_df["hardness_mg_L_CaCO3"].notna().any() else None,
        }

    logger.info(
        f"Parsed industrial mining dataset: {len(clean_df)} activities across "
        f"{len(metadata['industry_categories'])} sectors."
    )
    return clean_df, metadata
