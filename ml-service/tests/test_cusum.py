"""
Unit and end-to-end tests for the CUSUM sustained-shift detector.

Frames here are small hand-built software-verification mocks. The end-to-end tests use
the synthetic demonstration pipeline and do not establish real-world detection accuracy.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from cusum import CusumDetector
from data_loader import load_config
from predict import EnviroTrustPredictor
from src.synthetic_data_generator import (
    DEMO_TEST_RATIO,
    SIMULATION_FEATURES,
    generate_and_save_demo_datasets,
    train_demo_model,
)
from train import run_training


def _config(**cusum) -> dict:
    return {
        "features": {"numerical": ["COD_mg_L", "pH", "flat"]},
        "cusum": {"enabled": True, "k": 0.5, **cusum},
    }


@pytest.fixture
def baseline() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "COD_mg_L": rng.normal(100.0, 10.0, 200),
        "pH": rng.normal(7.0, 0.2, 200),
        "flat": 5.0,
    })


def _stream(cod, sensor="S-1", start="2026-11-01T00:00:00Z") -> pd.DataFrame:
    return pd.DataFrame({
        "sensorId": sensor,
        "timestamp": pd.date_range(start, periods=len(cod), freq="10min").astype(str),
        "COD_mg_L": cod,
        "pH": 7.0,
    })


def test_fit_learns_baseline_and_skips_constant_features(baseline):
    det = CusumDetector(_config()).fit(baseline)
    assert det.features == ["COD_mg_L", "pH"]
    assert det.means["COD_mg_L"] == pytest.approx(baseline["COD_mg_L"].mean(), abs=1e-4)
    assert det.stds["COD_mg_L"] == pytest.approx(baseline["COD_mg_L"].std(ddof=0), abs=1e-4)


def test_step_matches_tabular_cusum_recursion():
    det = CusumDetector(_config(base_features=["COD_mg_L"]))
    det.features, det.means, det.stds = ["COD_mg_L"], {"COD_mg_L": 0.0}, {"COD_mg_L": 1.0}

    state, highs, lows = det.initial_state(), [], []
    for z in [1.5, 1.5, -0.2, 0.0, -3.0]:
        state = det.step(state, {"COD_mg_L": z})
        highs.append(state[0][0]); lows.append(state[1][0])

    # S+ = max(0, S+ + z - 0.5), S- = max(0, S- - z - 0.5)
    assert highs == pytest.approx([1.0, 2.0, 1.3, 0.8, 0.0])
    assert lows == pytest.approx([0.0, 0.0, 0.0, 0.0, 2.5])


def test_missing_value_leaves_statistic_unchanged(baseline):
    det = CusumDetector(_config()).fit(baseline)
    state = det.step(det.initial_state(), {"COD_mg_L": 150.0, "pH": 7.0})
    after = det.step(state, {"COD_mg_L": None, "pH": 7.0})
    assert after[0][0] == state[0][0]


def test_calibration_uses_validation_peak_with_margin_and_floor(baseline):
    det = CusumDetector(_config(h_margin=0.25, h_min=0.1)).fit(baseline)
    peak = det.max_statistic(baseline).max()
    assert det.calibrate(baseline) == pytest.approx(peak * 1.25, abs=1e-4)

    floored = CusumDetector(_config(h_min=1000.0)).fit(baseline)
    assert floored.calibrate(baseline) == 1000.0

    manual = CusumDetector(_config(h=7.5)).fit(baseline)
    assert manual.calibrate(baseline) == 7.5


def test_sustained_shift_alarms_with_direction_and_feature(baseline):
    det = CusumDetector(_config(h=5.0)).fit(baseline)
    # 1.5 std below the mean: individually plausible, sustained
    alarms = det.check(_stream([85.0] * 12))

    assert alarms[0] == []
    assert alarms[-1] and all(a["feature"] == "COD_mg_L" and a["rule"] == "cusum_downward_shift" for a in alarms[-1])
    # Once triggered, a sustained shift keeps alarming (no reset on alarm)
    first = next(i for i, a in enumerate(alarms) if a)
    assert all(alarms[i] for i in range(first, len(alarms)))


def test_streams_are_per_sensor_and_in_time_order(baseline):
    det = CusumDetector(_config(h=5.0)).fit(baseline)
    biased = _stream([85.0] * 12, sensor="BIASED")
    healthy = _stream([100.0] * 12, sensor="HEALTHY")
    df = pd.concat([biased, healthy]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    flags = det.flags(df)
    assert flags[(df["sensorId"] == "BIASED").to_numpy()].any()
    assert not flags[(df["sensorId"] == "HEALTHY").to_numpy()].any()

    shuffled = df.sample(frac=1.0, random_state=1)
    reordered = pd.Series(det.flags(shuffled), index=shuffled.index).loc[df.index].to_numpy()
    np.testing.assert_array_equal(reordered, flags)


def test_disabled_never_alarms(baseline):
    det = CusumDetector(_config(enabled=False, h=1.0)).fit(baseline)
    assert not det.flags(_stream([0.0] * 10)).any()


def test_json_roundtrip(baseline, tmp_path: Path):
    det = CusumDetector(_config()).fit(baseline)
    det.calibrate(baseline)
    det.save(tmp_path / "cusum.json")
    loaded = CusumDetector.load(tmp_path / "cusum.json")

    assert loaded.to_dict() == det.to_dict()
    stream = _stream([85.0] * 12)
    assert loaded.check(stream) == det.check(stream)


# ------------------------------------------------------------------------------
# Training & end-to-end inference
# ------------------------------------------------------------------------------

def test_training_skips_cusum_for_non_temporal_data(tmp_path: Path):
    rng = np.random.RandomState(1)
    csv = tmp_path / "non_temporal.csv"
    pd.DataFrame({"COD_mg_L": rng.normal(100, 10, 60), "pH": rng.normal(7, 0.2, 60)}).to_csv(csv, index=False)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "cusum.json").write_text("{}")  # stale artifact from an earlier model

    cfg = load_config("config/data_config.yaml")
    cfg["artifacts"]["models_dir"] = str(models_dir)
    cfg["splitting"].update({"strategy": "random", "test_ratio": 0.0})
    meta = run_training(cfg, data_path=csv)

    assert meta["cusum"]["enabled"] is False
    assert not (models_dir / "cusum.json").exists()


@pytest.fixture(scope="module")
def demo_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("cusum_demo")
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


def test_demo_training_writes_cusum_artifact(demo_setup):
    models_dir, meta, _ = demo_setup
    assert meta["cusum"]["enabled"] is True
    assert meta["cusum"]["features"] == len(SIMULATION_FEATURES)

    artifact = json.loads((models_dir / "cusum.json").read_text())
    assert artifact["h"] >= 4.0
    assert set(artifact["features"]) == set(SIMULATION_FEATURES)
    assert _predictor(models_dir).cusum is not None


def test_normal_held_out_stream_does_not_alarm(demo_setup):
    models_dir, _, holdout = demo_setup
    predictor = _predictor(models_dir)
    results = [predictor.predict_telemetry(_payload(row, f"N-{i}")) for i, (_, row) in enumerate(holdout.iterrows())]
    assert sum(bool(r["cusum_alarms"]) for r in results) == 0


def test_in_envelope_bias_is_caught_by_cusum_not_rules(demo_setup):
    """
    A chloride probe reading 1.5 std low. Chloride has no ratio partner, and most biased
    readings stay inside the learned envelope: the rules only catch the few that fall off
    its low tail. CUSUM accumulates the bias and flags every reading once it has built up.
    """
    models_dir, _, holdout = demo_setup
    predictor = _predictor(models_dir)
    std = predictor.cusum.stds["chloride_mg_L"]

    results = []
    for i, (_, row) in enumerate(holdout.head(60).iterrows()):
        row = row.copy()
        row["chloride_mg_L"] -= 1.5 * std
        results.append(predictor.predict_telemetry(_payload(row, f"B-{i}")))

    assert sum(bool(r["rule_violations"]) for r in results) <= 9
    flagged = [bool(r["cusum_alarms"]) for r in results]
    assert not any(flagged[:3])
    assert all(flagged[30:])
    alarm = results[-1]["cusum_alarms"][0]
    assert alarm["feature"] == "chloride_mg_L" and alarm["rule"] == "cusum_downward_shift"
    assert "cusum" in results[-1]["detection_sources"] and results[-1]["status"] == "SUSPICIOUS"


def test_retried_reading_does_not_accumulate_twice(demo_setup):
    models_dir, _, holdout = demo_setup
    predictor = _predictor(models_dir)
    row = holdout.iloc[0].copy()
    row["chloride_mg_L"] -= 3.0 * predictor.cusum.stds["chloride_mg_L"]

    first = predictor.predict_telemetry(_payload(row, "R-1"))
    state = predictor._cusum_states["COD-OUT-001"]["after"]
    for _ in range(5):
        retry = predictor.predict_telemetry(_payload(row, "R-1"))
    np.testing.assert_array_equal(predictor._cusum_states["COD-OUT-001"]["after"][1], state[1])
    assert retry["cusum_alarms"] == first["cusum_alarms"]


def test_cusum_state_resets_on_reload(demo_setup):
    models_dir, _, holdout = demo_setup
    predictor = _predictor(models_dir)
    predictor.predict_telemetry(_payload(holdout.iloc[0], "N-0"))
    assert predictor._cusum_states
    predictor.reload()
    assert predictor._cusum_states == {}
