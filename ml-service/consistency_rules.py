"""
EnviroTrust ML Service — Physical Consistency Rules

Complements Isolation Forest with explicit, explainable per-reading checks.
Isolation Forest scores the joint feature space and can miss a single absurd
measurement (e.g. COD = 40 mg/L in industrial effluent) because only the few
trees that split on that feature isolate it quickly. These rules catch such
readings directly and report exactly which check failed.

Two rule families:
  1. Physical invariants (fixed, always true regardless of site):
       - pH within [0, 14]
       - concentrations (mg/L, NTU, Pt-Co) are non-negative
       - BOD <= COD
       - chloride + sulphate <= TDS
       - total solids ~= TDS + TSS (within a relative tolerance)
  2. Learned operating envelopes (fitted on normal training records only):
       - per-feature value range
       - ratio bands for configured feature pairs (e.g. BOD/COD, turbidity/TSS)

Rules evaluate the raw mapped measurements, never imputed values: a missing
measurement is skipped rather than checked against a fabricated median.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("EnviroTrust.ConsistencyRules")

CONCENTRATION_SUFFIXES = ("_mg_L", "_NTU", "_PtCo")

DEFAULT_RATIO_PAIRS = [
    ["BOD_mg_L", "COD_mg_L"],
    ["turbidity_NTU", "total_suspended_solids_mg_L"],
]

# Fewer non-null training values than this and an envelope is not learned
MIN_ENVELOPE_SAMPLES = 10


class PhysicalConsistencyChecker:
    """
    Explainable per-reading physical consistency checks with learned envelopes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("consistency_rules") or {}
        self.enabled = bool(cfg.get("enabled", True))
        quantiles = cfg.get("envelope_quantiles", [0.005, 0.995])
        self.q_low, self.q_high = float(quantiles[0]), float(quantiles[1])
        self.envelope_margin = float(cfg.get("envelope_margin", 0.10))
        self.ratio_pairs: List[List[str]] = [list(p) for p in cfg.get("ratio_pairs", DEFAULT_RATIO_PAIRS)]
        self.ratio_margin = float(cfg.get("ratio_margin", 0.10))
        self.solids_balance_tolerance = float(cfg.get("solids_balance_tolerance", 0.10))

        self.envelopes: Dict[str, List[float]] = {}
        self.ratio_bands: Dict[str, Dict[str, Any]] = {}
        self.is_fitted: bool = False

    def _band(self, values: pd.Series, margin: float) -> List[float]:
        """Quantile band widened on both sides by margin * band width."""
        lo = float(values.quantile(self.q_low))
        hi = float(values.quantile(self.q_high))
        pad = margin * (hi - lo)
        return [round(lo - pad, 4), round(hi + pad, 4)]

    def fit(self, df: pd.DataFrame) -> "PhysicalConsistencyChecker":
        """
        Learn operating envelopes from normal training records (raw, unimputed values).
        Caller must ensure df contains normal records only.
        """
        self.envelopes = {}
        self.ratio_bands = {}

        for col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) >= MIN_ENVELOPE_SAMPLES:
                self.envelopes[col] = self._band(values, self.envelope_margin)

        for num_col, den_col in self.ratio_pairs:
            if num_col not in df.columns or den_col not in df.columns:
                continue
            num = pd.to_numeric(df[num_col], errors="coerce")
            den = pd.to_numeric(df[den_col], errors="coerce")
            ratio = (num / den)[(den > 0) & num.notnull()].dropna()
            if len(ratio) >= MIN_ENVELOPE_SAMPLES:
                self.ratio_bands[f"{num_col}/{den_col}"] = {
                    "numerator": num_col,
                    "denominator": den_col,
                    "range": self._band(ratio, self.ratio_margin),
                }

        self.is_fitted = True
        logger.info(
            f"Fitted physical consistency rules: {len(self.envelopes)} feature envelopes, "
            f"{len(self.ratio_bands)} ratio bands."
        )
        return self

    def _check_row(self, row: Dict[str, float]) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []

        def get(name: str) -> Optional[float]:
            v = row.get(name)
            return None if v is None or pd.isna(v) else float(v)

        # 1. Physical invariants
        ph = get("pH")
        if ph is not None and not (0.0 <= ph <= 14.0):
            violations.append({
                "rule": "ph_physical_range", "feature": "pH", "value": round(ph, 4),
                "expected_range": [0.0, 14.0], "message": "pH outside the physical 0-14 scale.",
            })

        for name, value in row.items():
            if name.endswith(CONCENTRATION_SUFFIXES):
                v = get(name)
                if v is not None and v < 0:
                    violations.append({
                        "rule": "negative_concentration", "feature": name, "value": round(v, 4),
                        "message": f"{name} is negative, which is physically impossible.",
                    })

        cod, bod = get("COD_mg_L"), get("BOD_mg_L")
        if cod is not None and bod is not None and bod > cod:
            violations.append({
                "rule": "bod_exceeds_cod", "feature": "BOD_mg_L", "value": round(bod, 4),
                "message": f"BOD ({bod:.1f}) exceeds COD ({cod:.1f}); BOD is a subset of COD.",
            })

        tds, cl, so4 = get("TDS_mg_L"), get("chloride_mg_L"), get("sulphate_mg_L")
        if tds is not None and cl is not None and so4 is not None and (cl + so4) > tds:
            violations.append({
                "rule": "ions_exceed_tds", "feature": "TDS_mg_L", "value": round(tds, 4),
                "message": f"Chloride + sulphate ({cl + so4:.1f}) exceed TDS ({tds:.1f}).",
            })

        ts, tss = get("total_solids_mg_L"), get("total_suspended_solids_mg_L")
        if ts is not None and tds is not None and tss is not None and ts > 0:
            imbalance = abs(ts - (tds + tss)) / ts
            if imbalance > self.solids_balance_tolerance:
                violations.append({
                    "rule": "solids_mass_balance", "feature": "total_solids_mg_L", "value": round(ts, 4),
                    "message": (
                        f"Total solids ({ts:.1f}) differ from TDS + TSS ({tds + tss:.1f}) by "
                        f"{imbalance * 100:.1f}% (tolerance {self.solids_balance_tolerance * 100:.0f}%)."
                    ),
                })

        # 2. Learned operating envelopes
        for name, (lo, hi) in self.envelopes.items():
            v = get(name)
            if v is not None and not (lo <= v <= hi):
                violations.append({
                    "rule": "operating_envelope", "feature": name, "value": round(v, 4),
                    "expected_range": [lo, hi],
                    "message": f"{name} = {v:.2f} is outside the learned normal range [{lo}, {hi}].",
                })

        for key, band in self.ratio_bands.items():
            num, den = get(band["numerator"]), get(band["denominator"])
            if num is None or den is None or den <= 0:
                continue
            ratio = num / den
            lo, hi = band["range"]
            if not (lo <= ratio <= hi):
                violations.append({
                    "rule": "ratio_band", "feature": key, "value": round(ratio, 4),
                    "expected_range": [lo, hi],
                    "message": f"Ratio {key} = {ratio:.3f} is outside the learned normal band [{lo}, {hi}].",
                })

        return violations

    def check(self, df: pd.DataFrame) -> List[List[Dict[str, Any]]]:
        """Return a list of violations for each row of df (empty list = consistent)."""
        if not self.enabled:
            return [[] for _ in range(len(df))]
        numeric = df.apply(pd.to_numeric, errors="coerce")
        return [self._check_row(row) for row in numeric.to_dict(orient="records")]

    def flags(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean array: True where a row violates at least one rule."""
        return np.array([bool(v) for v in self.check(df)], dtype=bool)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "envelope_quantiles": [self.q_low, self.q_high],
            "envelope_margin": self.envelope_margin,
            "ratio_margin": self.ratio_margin,
            "solids_balance_tolerance": self.solids_balance_tolerance,
            "envelopes": self.envelopes,
            "ratio_bands": self.ratio_bands,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhysicalConsistencyChecker":
        checker = cls({"consistency_rules": {
            "enabled": data.get("enabled", True),
            "envelope_quantiles": data.get("envelope_quantiles", [0.005, 0.995]),
            "envelope_margin": data.get("envelope_margin", 0.10),
            "ratio_margin": data.get("ratio_margin", 0.10),
            "solids_balance_tolerance": data.get("solids_balance_tolerance", 0.10),
            "ratio_pairs": [[b["numerator"], b["denominator"]] for b in data.get("ratio_bands", {}).values()],
        }})
        checker.envelopes = {k: list(v) for k, v in data.get("envelopes", {}).items()}
        checker.ratio_bands = dict(data.get("ratio_bands", {}))
        checker.is_fitted = True
        return checker

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist learned rules as human-auditable JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved physical consistency rules to: {path.resolve()}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PhysicalConsistencyChecker":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Consistency rules artifact not found at: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
