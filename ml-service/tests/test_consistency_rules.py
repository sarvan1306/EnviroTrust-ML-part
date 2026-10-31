"""
Tests for the physical consistency rule layer and the held-out test split.

NOTE ON TEST FIXTURES:
Rule fixtures are small hand-built software-verification mocks. End-to-end tests
generate synthetic demonstration data into pytest tmp directories only; the
committed data/synthetic_demo/ and models/demo/ artifacts are never touched.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from consistency_rules import PhysicalConsistencyChecker
from data_loader import load_config
from predict import EnviroTrustPredictor
from train import split_holdout_test

FEATURES = [
    "COD_mg_L", "BOD_mg_L", "pH", "TDS_mg_L", "turbidity_NTU",
    "total_solids_mg_L", "total_suspended_solids_mg_L", "chloride_mg_L", "sulphate_mg_L",
]


@pytest.fixture
def normal_frame() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 200
    cod = rng.normal(5800, 400, n)
    tss = rng.normal(380, 40, n)
    tds = rng.normal(2850, 150, n)
    return pd.DataFrame({
        "COD_mg_L": cod,
        "BOD_mg_L": cod * rng.uniform(0.22, 0.27, n),
        "pH": rng.normal(6.9, 0.2, n),
        "TDS_mg_L": tds,
        "turbidity_NTU": 0.85 * tss + rng.normal(0, 10, n),
        "total_solids_mg_L": tds + tss,
        "total_suspended_solids_mg_L": tss,
        "chloride_mg_L": 0.12 * tds,
        "sulphate_mg_L": 0.15 * tds,
    })


def _reading(**overrides) -> pd.DataFrame:
    base = {
        "COD_mg_L": 5800.0, "BOD_mg_L": 1400.0, "pH": 6.9, "TDS_mg_L": 2850.0,
        "turbidity_NTU": 323.0, "total_solids_mg_L": 3230.0, "total_suspended_solids_mg_L": 380.0,
        "chloride_mg_L": 342.0, "sulphate_mg_L": 427.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _rules_hit(checker: PhysicalConsistencyChecker, df: pd.DataFrame) -> set:
    return {v["rule"] for v in checker.check(df)[0]}


def test_normal_reading_is_consistent(normal_frame):
    checker = PhysicalConsistencyChecker().fit(normal_frame)
    assert checker.check(_reading()) == [[]]
    # Low false-positive rate on the very distribution it was fitted on
    assert checker.flags(normal_frame).mean() <= 0.05


def test_physical_invariants_without_fitting():
    checker = PhysicalConsistencyChecker()  # no learned envelopes, invariants only
    assert "ph_physical_range" in _rules_hit(checker, _reading(pH=15.2))
    assert "negative_concentration" in _rules_hit(checker, _reading(COD_mg_L=-5.0, BOD_mg_L=-10.0))
    assert "bod_exceeds_cod" in _rules_hit(checker, _reading(BOD_mg_L=7000.0))
    assert "ions_exceed_tds" in _rules_hit(checker, _reading(chloride_mg_L=2000.0, sulphate_mg_L=1500.0))
    assert "solids_mass_balance" in _rules_hit(checker, _reading(total_solids_mg_L=5000.0))


def test_learned_envelope_catches_single_extreme_value(normal_frame):
    checker = PhysicalConsistencyChecker().fit(normal_frame)
    low = checker.check(_reading(COD_mg_L=40.0, BOD_mg_L=10.0))[0]
    assert any(v["rule"] == "operating_envelope" and v["feature"] == "COD_mg_L" for v in low)
    high = checker.check(_reading(COD_mg_L=40000.0, BOD_mg_L=10000.0))[0]
    assert any(v["rule"] == "operating_envelope" and v["feature"] == "COD_mg_L" for v in high)


def test_ratio_band_catches_decoupling(normal_frame):
    checker = PhysicalConsistencyChecker().fit(normal_frame)
    # Turbidity far too low for the suspended solids, both individually in range
    hits = checker.check(_reading(turbidity_NTU=200.0, total_suspended_solids_mg_L=420.0,
                                  total_solids_mg_L=3270.0))[0]
    assert any(v["rule"] == "ratio_band" and v["feature"] == "turbidity_NTU/total_suspended_solids_mg_L" for v in hits)


def test_missing_measurements_are_skipped_not_imputed(normal_frame):
    checker = PhysicalConsistencyChecker().fit(normal_frame)
    assert checker.check(_reading(BOD_mg_L=np.nan)) == [[]]


def test_disabled_rules_never_flag(normal_frame):
    checker = PhysicalConsistencyChecker({"consistency_rules": {"enabled": False}}).fit(normal_frame)
    assert checker.check(_reading(pH=20.0)) == [[]]


def test_rules_json_roundtrip(normal_frame, tmp_path: Path):
    checker = PhysicalConsistencyChecker().fit(normal_frame)
    path = tmp_path / "rules.json"
    checker.save(path)
    loaded = PhysicalConsistencyChecker.load(path)
    assert loaded.envelopes == checker.envelopes
    assert loaded.ratio_bands == checker.ratio_bands
    probe = _reading(COD_mg_L=40.0, BOD_mg_L=10.0)
    assert loaded.check(probe) == checker.check(probe)


def test_chronological_holdout_reserves_latest_rows():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="10min").astype(str)[::-1],
        "COD_mg_L": np.arange(100, dtype=float),
    })
    cfg = {"splitting": {"strategy": "chronological", "timestamp_column": "timestamp", "test_ratio": 0.2}}
    dev, test = split_holdout_test(df, cfg)
    assert len(dev) == 80 and len(test) == 20
    assert pd.to_datetime(test["timestamp"]).min() > pd.to_datetime(dev["timestamp"]).max()
    # Deterministic: same input gives the same holdout
    _, test_again = split_holdout_test(df, cfg)
    assert test.index.equals(test_again.index)


def test_holdout_disabled_returns_empty_test():
    df = pd.DataFrame({"COD_mg_L": np.arange(20, dtype=float)})
    dev, test = split_holdout_test(df, {"splitting": {"strategy": "random", "test_ratio": 0.0}})
    assert len(dev) == 20 and test.empty


def test_demo_pipeline_holdout_rules_and_per_scenario_eval(tmp_path: Path):
    """Train and evaluate a demo model in tmp: holdout excluded, rules saved, per-scenario report."""
    from src.synthetic_data_generator import DEMO_TEST_RATIO, evaluate_demo_model, train_demo_model

    data_dir, models_dir = tmp_path / "data", tmp_path / "models_demo"
    meta = train_demo_model(demo_data_dir=data_dir, demo_models_dir=models_dir)

    n_holdout = int(round(1000 * DEMO_TEST_RATIO))
    assert meta["holdout_test"]["rows"] == n_holdout
    assert meta["train_samples"] + meta["validation_samples"] == 1000 - n_holdout
    assert (models_dir / "consistency_rules.json").exists()

    report = evaluate_demo_model(demo_models_dir=models_dir, demo_data_dir=data_dir)
    assert report["normal_test_samples"] == n_holdout
    assert set(report["per_scenario"]) == {
        "none", "low_cod", "high_cod", "cod_turbidity_decoupling",
        "sensor_flatline", "sudden_spike", "gradual_drift",
    }
    # Rules must catch single-reading extremes that Isolation Forest alone misses
    for scenario in ("low_cod", "high_cod"):
        assert report["per_scenario"][scenario]["flag_rate_combined"] >= 0.95
    assert report["metrics"]["recall"] >= report["isolation_forest_only_metrics"]["recall"]

    predictor = EnviroTrustPredictor(
        config_path=load_config("config/data_config.yaml"),
        models_dir=str(models_dir),
        model_filename="isolation_forest_demo.joblib",
        preprocessor_filename="preprocessor_demo.joblib",
        model_mode="SYNTHETIC_DEMO",
    )
    measurements = {
        "COD_mg_L": 45.0, "BOD_mg_L": 12.0, "pH": 6.9, "TDS_mg_L": 2850.0, "turbidity_NTU": 323.0,
        "colour_PtCo": 2600.0, "total_solids_mg_L": 3230.0, "total_suspended_solids_mg_L": 380.0,
        "chloride_mg_L": 342.0, "sulphate_mg_L": 427.0,
    }
    result = predictor.predict_telemetry({"readingId": "R-1", "measurements": measurements})
    assert result["is_anomaly"] is True
    assert "physical_rules" in result["detection_sources"]
    assert any(v["feature"] == "COD_mg_L" for v in result["rule_violations"])
