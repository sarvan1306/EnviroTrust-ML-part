"""
Unit tests for data_loader.py

NOTE ON TEST FIXTURES:
All fixtures in this file are temporary software-verification mocks generated
solely in memory/temporary directories to verify file loading and parsing logic.
They do NOT represent authentic industrial data and cannot be used to evaluate
model accuracy.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from data_loader import (
    load_dataset,
    inspect_dataset,
    format_inspection_report,
    load_config,
    DataLoadingError,
)


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Creates a small temporary CSV file for loader software testing only."""
    csv_path = tmp_path / "mock_software_test.csv"
    data = {
        "production_rate": [500.0, 600.0, 700.0, np.nan],
        "wastewater_flow": [450.0, 550.0, 650.0, 750.0],
        "operating_mode": ["NORMAL", "NORMAL", "HIGH_LOAD", "CLEANING"],
        "outlet_cod": [120.0, 130.0, 150.0, 80.0],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_excel_file(tmp_path: Path) -> Path:
    """Creates a small temporary Excel file for loader software testing only."""
    xlsx_path = tmp_path / "mock_software_test.xlsx"
    data = {
        "production_rate": [300.0, 400.0],
        "wastewater_flow": [280.0, 390.0],
        "outlet_cod": [95.0, 110.0],
    }
    pd.DataFrame(data).to_excel(xlsx_path, index=False, engine="openpyxl")
    return xlsx_path


def test_load_csv_successfully(temp_csv_file: Path):
    """Verify CSV dataset ingestion."""
    df = load_dataset(temp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "production_rate" in df.columns


def test_load_excel_successfully(temp_excel_file: Path):
    """Verify Excel dataset ingestion."""
    df = load_dataset(temp_excel_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "outlet_cod" in df.columns


def test_missing_file_handling():
    """Verify that a non-existent file path raises FileNotFoundError gracefully."""
    non_existent = Path("/tmp/definitely_not_a_real_dataset_envirotrust.csv")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(non_existent)
    assert "does not exist" in str(exc_info.value)


def test_unsupported_format_handling(tmp_path: Path):
    """Verify that an unsupported extension raises DataLoadingError."""
    bad_file = tmp_path / "test.json"
    bad_file.write_text('{"key": "val"}')
    with pytest.raises(DataLoadingError) as exc_info:
        load_dataset(bad_file, file_format="auto")
    assert "Unsupported file extension" in str(exc_info.value)


def test_inspect_dataset_reporting():
    """Verify dataset profiling functionality."""
    df = pd.DataFrame({
        "num_col": [1.0, 2.0, np.nan, 4.0, 4.0],
        "cat_col": ["A", "B", "C", "D", "D"],
    })
    profile = inspect_dataset(df)

    assert profile["row_count"] == 5
    assert profile["column_count"] == 2
    assert profile["missing_counts"]["num_col"] == 1
    assert profile["missing_percentages"]["num_col"] == 20.0
    assert profile["duplicate_rows"] == 1
    assert "num_col" in profile["numerical_columns"]
    assert "cat_col" in profile["categorical_columns"]

    report_str = format_inspection_report(profile)
    assert "ENVIROTRUST DATASET INSPECTION REPORT" in report_str
    assert "num_col" in report_str
