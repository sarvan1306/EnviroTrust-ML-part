"""
Unit tests for feature_engineering.py

NOTE ON TEST FIXTURES:
These mock dataframes verify column mapping and feature extraction logic.
They are software-verification mocks only and do not represent authentic data.
"""

import pytest
import pandas as pd
from feature_engineering import FeatureEngineeringPipeline


@pytest.fixture
def mock_config() -> dict:
    return {
        "column_mapping": {
            "production_rate": "Factory_Prod_Rate",
            "wastewater_flow": "Plant_Effluent_Flow",
            "outlet_cod": "Sensor_Reported_COD",
            "operating_mode": "Op_State",
        },
        "features": {
            "numerical": ["production_rate", "wastewater_flow", "outlet_cod"],
            "categorical": ["operating_mode"],
        },
    }


def test_column_mapping_renames_correctly(mock_config: dict):
    pipeline = FeatureEngineeringPipeline(mock_config)
    raw_df = pd.DataFrame({
        "Factory_Prod_Rate": [500.0, 600.0],
        "Plant_Effluent_Flow": [450.0, 550.0],
        "Sensor_Reported_COD": [120.0, 140.0],
        "Op_State": ["NORMAL", "HIGH_LOAD"],
        "Ignored_Column": [999, 888],
    })

    mapped_df = pipeline.apply_column_mapping(raw_df)
    assert "production_rate" in mapped_df.columns
    assert "wastewater_flow" in mapped_df.columns
    assert "outlet_cod" in mapped_df.columns
    assert "operating_mode" in mapped_df.columns
    assert "Factory_Prod_Rate" not in mapped_df.columns


def test_select_features_filters_unwanted_columns(mock_config: dict):
    pipeline = FeatureEngineeringPipeline(mock_config)
    raw_df = pd.DataFrame({
        "Factory_Prod_Rate": [500.0],
        "Plant_Effluent_Flow": [450.0],
        "Sensor_Reported_COD": [120.0],
        "Op_State": ["NORMAL"],
        "Extraneous_Data": ["DiscardMe"],
    })

    selected_df = pipeline.process(raw_df)
    assert set(selected_df.columns) == {"production_rate", "wastewater_flow", "outlet_cod", "operating_mode"}
    assert "Extraneous_Data" not in selected_df.columns


def test_domain_ratios_calculation(mock_config: dict):
    pipeline = FeatureEngineeringPipeline(mock_config)
    df = pd.DataFrame({
        "production_rate": [500.0, 1000.0],
        "wastewater_flow": [250.0, 800.0],
    })
    ratio_df = pipeline.add_domain_ratios(df)
    assert "flow_to_production_ratio" in ratio_df.columns
    assert ratio_df["flow_to_production_ratio"].iloc[0] == pytest.approx(0.5)
    assert ratio_df["flow_to_production_ratio"].iloc[1] == pytest.approx(0.8)
