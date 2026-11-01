"""
EnviroTrust ML Service — CUSUM Sustained-Shift Detector

Complements Isolation Forest and the physical rules with a sequential check. Both of
those score one reading at a time, so a sensor that slowly develops a bias (calibration
drift, probe fouling) keeps producing individually plausible readings and is never
flagged. The two-sided tabular CUSUM accumulates small, persistent deviations per
measurement until they exceed a decision threshold:

    z_t   = (x_t - mean) / std                (mean, std from NORMAL training records)
    S+_t  = max(0, S+_{t-1} + z_t - k)        upward shift
    S-_t  = max(0, S-_{t-1} - z_t - k)        downward shift
    alarm when S+_t >= h or S-_t >= h

  - k (allowance, in std units) absorbs normal variation; CUSUM is most sensitive to
    shifts of about 2k. Normal operation with a daily cycle needs k ~ 1.0: at the
    textbook 0.5 the cycle alone accumulates into false alarms.
  - h (decision threshold) is calibrated on the normal validation stream: its largest
    statistic, widened by h_margin, never below h_min.

Statistics run per sensor stream in timestamp order and never reset on alarm: a
sustained shift keeps alarming until the process returns to normal and the statistic
decays (about k + |z| per reading). Missing measurements leave their statistic unchanged.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("EnviroTrust.Cusum")

DEFAULT_STREAM_KEY = "__default__"

# Fewer non-null training values than this and a feature is not monitored
MIN_CUSUM_SAMPLES = 10

CusumState = Tuple[np.ndarray, np.ndarray]


class CusumDetector:
    """
    Two-sided per-feature CUSUM for sustained mean shifts, fitted on normal records.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        cfg = config.get("cusum") or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.k = float(cfg.get("k", 1.0))
        self.manual_h = cfg.get("h")
        self.h_margin = float(cfg.get("h_margin", 0.25))
        self.h_min = float(cfg.get("h_min", 4.0))
        self.base_features: List[str] = list(cfg.get("base_features") or config.get("features", {}).get("numerical", []))
        self.group_column = cfg.get("group_column", "sensorId")
        self.timestamp_column = cfg.get("timestamp_column", "timestamp")

        self.features: List[str] = []
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.h: float = float(self.manual_h) if self.manual_h is not None else self.h_min
        self.calibration: Dict[str, Any] = {}
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting & calibration
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "CusumDetector":
        """
        Learn per-feature mean and std from normal training records (raw, unimputed values).
        Caller must ensure df contains normal records only. Constant features are skipped.
        """
        self.features, self.means, self.stds = [], {}, {}
        for col in self.base_features:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            std = float(values.std(ddof=0)) if len(values) else 0.0
            if len(values) >= MIN_CUSUM_SAMPLES and std > 0:
                self.features.append(col)
                self.means[col] = round(float(values.mean()), 6)
                self.stds[col] = round(std, 6)

        self.is_fitted = True
        logger.info(f"Fitted CUSUM on {len(self.features)} features (k={self.k}).")
        return self

    def calibrate(self, df_val_normal: pd.DataFrame) -> float:
        """
        Set h from the normal validation stream: its largest statistic * (1 + h_margin),
        floored at h_min. A manual h in config overrides calibration.
        """
        if self.manual_h is not None:
            self.h = float(self.manual_h)
            self.calibration = {"strategy": "manual", "h": self.h}
            return self.h

        peak = float(self.max_statistic(df_val_normal).max()) if len(df_val_normal) and self.features else 0.0
        self.h = round(max(self.h_min, peak * (1.0 + self.h_margin)), 4)
        self.calibration = {
            "strategy": "validation_peak",
            "validation_samples": int(len(df_val_normal)),
            "validation_peak_statistic": round(peak, 4),
            "h_margin": self.h_margin,
            "h_min": self.h_min,
            "h": self.h,
        }
        logger.info(f"Calibrated CUSUM threshold h={self.h:.4f} (validation peak {peak:.4f}).")
        return self.h

    # ------------------------------------------------------------------
    # Online (per-reading) interface
    # ------------------------------------------------------------------

    def initial_state(self) -> CusumState:
        return np.zeros(len(self.features)), np.zeros(len(self.features))

    def _standardize(self, row: Dict[str, Any]) -> np.ndarray:
        z = np.full(len(self.features), np.nan)
        for i, col in enumerate(self.features):
            v = pd.to_numeric(row.get(col), errors="coerce")
            if v is not None and not pd.isna(v):
                z[i] = (float(v) - self.means[col]) / self.stds[col]
        return z

    def step(self, state: CusumState, row: Dict[str, Any]) -> CusumState:
        """Advance the statistics by one reading. Missing values leave their statistic unchanged."""
        hi, lo = state
        z = self._standardize(row)
        valid = ~np.isnan(z)
        z = np.where(valid, z, 0.0)
        return (
            np.where(valid, np.maximum(0.0, hi + z - self.k), hi),
            np.where(valid, np.maximum(0.0, lo - z - self.k), lo),
        )

    def alarms(self, state: CusumState) -> List[Dict[str, Any]]:
        """Explainable alarms for every feature whose statistic has reached h."""
        if not self.enabled:
            return []
        hi, lo = state
        out: List[Dict[str, Any]] = []
        for i, col in enumerate(self.features):
            for direction, stat in (("upward", hi[i]), ("downward", lo[i])):
                if stat >= self.h:
                    out.append({
                        "rule": f"cusum_{direction}_shift",
                        "feature": col,
                        "statistic": round(float(stat), 4),
                        "threshold": round(self.h, 4),
                        "baseline_mean": self.means[col],
                        "message": (
                            f"{col} shows a sustained {direction} shift from its normal mean "
                            f"{self.means[col]:.2f} (CUSUM {stat:.2f} >= {self.h:.2f})."
                        ),
                    })
        return out

    # ------------------------------------------------------------------
    # Offline (whole-stream) interface
    # ------------------------------------------------------------------

    def _stream_order(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Positional frame sorted by timestamp (stable), plus its stream keys."""
        frame = df.reset_index(drop=True)
        if self.timestamp_column in frame.columns:
            ts = pd.to_datetime(frame[self.timestamp_column], errors="coerce", utc=True)
            frame = frame.loc[ts.sort_values(kind="mergesort").index]
        if self.group_column in frame.columns:
            keys = frame[self.group_column].fillna(DEFAULT_STREAM_KEY).astype(str)
        else:
            keys = pd.Series(DEFAULT_STREAM_KEY, index=frame.index)
        return frame, keys

    def max_statistic(self, df: pd.DataFrame) -> np.ndarray:
        """Largest S+ or S- across features after each row, aligned to df's row order."""
        return np.array([max(float(h.max(initial=0.0)), float(l.max(initial=0.0))) for h, l in self._run(df)])

    def _run(self, df: pd.DataFrame) -> List[CusumState]:
        """State after each row of df, in df's row order, computed per stream in time order."""
        states: List[Optional[CusumState]] = [None] * len(df)
        if df.empty:
            return []
        frame, keys = self._stream_order(df)
        current: Dict[str, CusumState] = {}
        for pos, key, row in zip(frame.index, keys, frame.to_dict(orient="records")):
            current[key] = self.step(current.get(key, self.initial_state()), row)
            states[pos] = current[key]
        return states  # type: ignore[return-value]

    def check(self, df: pd.DataFrame) -> List[List[Dict[str, Any]]]:
        """Alarms for each row of df (empty list = no sustained shift)."""
        if not self.enabled or not self.features:
            return [[] for _ in range(len(df))]
        return [self.alarms(state) for state in self._run(df)]

    def flags(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean array: True where any feature's CUSUM has reached h."""
        return np.array([bool(a) for a in self.check(df)], dtype=bool)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "k": self.k,
            "h": self.h,
            "group_column": self.group_column,
            "timestamp_column": self.timestamp_column,
            "calibration": self.calibration,
            "features": {col: {"mean": self.means[col], "std": self.stds[col]} for col in self.features},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CusumDetector":
        features = data.get("features", {})
        detector = cls({"cusum": {
            "enabled": data.get("enabled", True),
            "k": data.get("k", 1.0),
            "h": data.get("h"),
            "group_column": data.get("group_column", "sensorId"),
            "timestamp_column": data.get("timestamp_column", "timestamp"),
            "base_features": list(features),
        }})
        detector.features = list(features)
        detector.means = {c: float(v["mean"]) for c, v in features.items()}
        detector.stds = {c: float(v["std"]) for c, v in features.items()}
        detector.calibration = dict(data.get("calibration", {}))
        detector.is_fitted = True
        return detector

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist the fitted baseline and threshold as human-auditable JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved CUSUM detector to: {path.resolve()}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CusumDetector":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CUSUM artifact not found at: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
