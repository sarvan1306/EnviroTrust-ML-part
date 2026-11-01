"""
Unit and end-to-end tests for rolling-window (stuck-sensor) features.

Frames here are small hand-built software-verification mocks. The end-to-end tests use
the synthetic demonstration pipeline and do not establish real-world detection accuracy.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data_loader import load_config
from feature_engineering import ROLLING_SUFFIX, FeatureEngineeringPipeline, rolling_feature_names
from predict import EnviroTrustPredictor
from preprocessing import DataPreprocessor
from src.synthetic_data_generator import (
    DEMO_TEST_RATIO,
    SIMULATION_FEATURES,
    generate_and_save_demo_datasets,
    train_demo_model,
)

WINDOW = 3


def _config(**rolling) -> dict:
    return {
        "column_mapping": {"cod_raw": "COD_mg_L"},
        "features": {"numerical": ["COD_mg_L", "pH"], "categorical": [], "id_columns": ["sensorId"]},
        "rolling_features": {"enabled": True, "window": WINDOW, **rolling},
    }


def _stream(values, sensor="S-1", start="2026-11-01T00:00:00Z") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(values), freq="10min")
    return pd.DataFrame({
        "sensorId": sensor,
        "timestamp": ts.astype(str),
        "COD_mg_L": values,
        "pH": np.linspace(6.5, 7.5, len(values)),
    })


def test_stuck_values_give_zero_after_window_fills():
    fe = FeatureEngineeringPipeline(_config())
    out = fe.add_rolling_features(_stream([100.0, 140.0, 90.0, 120.0, 120.0, 120.0]))
    col = out[f"COD_mg_L{ROLLING_SUFFIX}"]

    assert col.iloc[: WINDOW - 1].isna().all()
    assert col.iloc[WINDOW - 1] > 0
    assert col.iloc[-1] == 0.0
    assert np.isclose(col.iloc[WINDOW - 1], np.log1p(np.std([100.0, 140.0, 90.0])))


def test_rolling_uses_mapped_raw_columns_and_keeps_input_index():
    fe = FeatureEngineeringPipeline(_config())
    df = _stream([1.0, 2.0, 3.0, 4.0]).rename(columns={"COD_mg_L": "cod_raw"})
    df.index = [40, 10, 30, 20]
    out = fe.add_rolling_features(df)

    assert list(out.index) == [40, 10, 30, 20]
    assert "cod_raw" in out.columns
    assert out[f"COD_mg_L{ROLLING_SUFFIX}"].notna().sum() == 2


def test_windows_follow_timestamps_not_row_order():
    fe = FeatureEngineeringPipeline(_config())
    ordered = _stream([10.0, 50.0, 20.0, 80.0])
    shuffled = ordered.iloc[[3, 1, 0, 2]]
    col = f"COD_mg_L{ROLLING_SUFFIX}"

    expected = fe.add_rolling_features(ordered)[col]
    actual = fe.add_rolling_features(shuffled)[col]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())


def test_sensors_form_separate_streams():
    """Interleaved sensors: one stuck, one noisy. Windows must not mix them."""
    fe = FeatureEngineeringPipeline(_config())
    stuck = _stream([200.0] * 5, sensor="STUCK")
    noisy = _stream([100.0, 300.0, 150.0, 250.0, 120.0], sensor="NOISY")
    out = fe.add_rolling_features(pd.concat([stuck, noisy]).sort_values("timestamp", kind="mergesort"))
    col = f"COD_mg_L{ROLLING_SUFFIX}"

    stuck_vals = out.loc[out["sensorId"] == "STUCK", col].dropna()
    noisy_vals = out.loc[out["sensorId"] == "NOISY", col].dropna()
    assert len(stuck_vals) == len(noisy_vals) == 5 - WINDOW + 1
    assert (stuck_vals == 0.0).all()
    assert (noisy_vals > 0).all()


def test_non_temporal_dataset_is_skipped():
    fe = FeatureEngineeringPipeline(_config())
    out = fe.add_rolling_features(_stream([1.0, 2.0, 3.0, 4.0]).drop(columns=["timestamp"]))
    assert not any(c.endswith(ROLLING_SUFFIX) for c in out.columns)

    # Inference passes require_timestamp=False and uses arrival order
    out = fe.add_rolling_features(_stream([5.0, 5.0, 5.0]).drop(columns=["timestamp"]), require_timestamp=False)
    assert out[f"COD_mg_L{ROLLING_SUFFIX}"].iloc[-1] == 0.0


def test_disabled_by_default_and_feature_names_extend_candidates():
    assert rolling_feature_names({"features": {"numerical": ["COD_mg_L"]}}) == []

    cfg = _config(base_features=["COD_mg_L"])
    names = rolling_feature_names(cfg)
    assert names == [f"COD_mg_L{ROLLING_SUFFIX}"]
    assert names[0] in FeatureEngineeringPipeline(cfg).num_features
    assert names[0] in DataPreprocessor(cfg).configured_num_cols


# ------------------------------------------------------------------------------
# End-to-end: synthetic demo model trained with rolling features
# ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("rolling_demo")
    data_dir, models_dir = root / "data", root / "models"
    generate_and_save_demo_datasets(output_dir=data_dir, random_seed=42)
    meta = train_demo_model(demo_data_dir=data_dir, demo_models_dir=models_dir, random_seed=42)
    df_norm = pd.read_csv(data_dir / "normal_demo.csv")
    holdout = df_norm.iloc[len(df_norm) - int(round(len(df_norm) * DEMO_TEST_RATIO)):]
    return models_dir, meta, holdout


def _predictor(models_dir: Path) -> EnviroTrustPredictor:
    return EnviroTrustPredictor(
        config_path=load_config("config/data_config.yaml"),
        models_dir=str(models_dir),
        model_filename="isolation_forest_demo.joblib",
        preprocessor_filename="preprocessor_demo.joblib",
        model_mode="SYNTHETIC_DEMO",
    )


def _payload(row: pd.Series, reading_id: str, sensor: str = "COD-OUT-001") -> dict:
    return {
        "readingId": reading_id,
        "sensorId": sensor,
        "timestamp": row["timestamp"],
        "measurements": {f: float(row[f]) for f in SIMULATION_FEATURES},
    }


def test_training_records_rolling_features_and_keeps_them_out_of_rules(demo_setup, tmp_path):
    models_dir, meta, _ = demo_setup
    expected = [f"{f}{ROLLING_SUFFIX}" for f in SIMULATION_FEATURES]

    assert meta["rolling_features"]["enabled"] is True
    assert meta["rolling_features"]["features"] == expected
    assert set(expected) <= set(meta["feature_schema"]["numerical_features"])

    predictor = _predictor(models_dir)
    assert predictor.rolling_features == expected
    assert not any(k.endswith(ROLLING_SUFFIX) for k in predictor.rules.envelopes)


def test_stuck_sensor_at_plausible_value_is_flagged_by_isolation_forest(demo_setup):
    """
    The case the rules cannot catch: a sensor frozen at a perfectly normal reading.
    Each reading passes every per-reading check; only the collapsed variance gives it away.
    """
    models_dir, _, holdout = demo_setup
    predictor = _predictor(models_dir)

    warmup = [predictor.predict_telemetry(_payload(row, f"N-{i}")) for i, (_, row) in enumerate(holdout.head(10).iterrows())]
    assert [r["rolling_window_filled"] for r in warmup[:4]] == [False] * 4
    assert all(r["rolling_window_filled"] for r in warmup[4:])
    assert sum(r["isolation_forest_flag"] for r in warmup) <= 1

    frozen = holdout.iloc[10]
    stuck = []
    for i in range(8):
        row = holdout.iloc[11 + i].copy()
        row[SIMULATION_FEATURES] = frozen[SIMULATION_FEATURES]
        stuck.append(predictor.predict_telemetry(_payload(row, f"S-{i}")))

    assert all(r["rule_violations"] == [] for r in stuck)
    # Once the window holds only frozen values, every reading is flagged by Isolation Forest
    assert all(r["isolation_forest_flag"] and r["status"] == "SUSPICIOUS" for r in stuck[4:])
    # CUSUM may also alarm when the frozen value sits off the normal mean; the rules never do
    assert all("physical_rules" not in r["detection_sources"] for r in stuck)


def test_retried_reading_is_not_mistaken_for_a_stuck_sensor(demo_setup):
    models_dir, _, holdout = demo_setup
    predictor = _predictor(models_dir)

    for i, (_, row) in enumerate(holdout.head(4).iterrows()):
        predictor.predict_telemetry(_payload(row, f"N-{i}"))
    retried = holdout.iloc[4]
    results = [predictor.predict_telemetry(_payload(retried, "N-RETRY")) for _ in range(6)]

    assert len({r["anomaly_score"] for r in results}) == 1
    assert [e["readingId"] for e in predictor._history["COD-OUT-001"]] == ["N-0", "N-1", "N-2", "N-3", "N-RETRY"]


def test_history_is_per_sensor_and_reset_on_reload(demo_setup):
    models_dir, _, holdout = demo_setup
    predictor = _predictor(models_dir)

    for i, (_, row) in enumerate(holdout.head(6).iterrows()):
        predictor.predict_telemetry(_payload(row, f"A-{i}", sensor="SENSOR-A"))
    first_b = predictor.predict_telemetry(_payload(holdout.iloc[6], "B-0", sensor="SENSOR-B"))
    assert first_b["rolling_window_filled"] is False

    predictor.reload()
    assert predictor._history == {}
    again = predictor.predict_telemetry(_payload(holdout.iloc[7], "A-7", sensor="SENSOR-A"))
    assert again["rolling_window_filled"] is False
