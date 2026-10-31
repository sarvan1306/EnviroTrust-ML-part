# EnviroTrust — Machine Learning Anomaly Detection Service

Independent Python ML microservice for **EnviroTrust**, an intelligent environmental monitoring and sensor-tampering detection platform.

This service implements process-consistency cross-checking using **scikit-learn Isolation Forest** combined with an explainable **physical consistency rule layer**. It operates on the principle that even if a pollution measurement carries a valid cryptographic signature and unmodified ledger history, it can still be physically fraudulent, manipulated, or incongruent with operating reality. Cross-referencing environmental readings against independent industrial operational metrics enables automated detection of sensor tampering and unrecorded process excursions.

> [!IMPORTANT]
> **Authentic Data Policy & Synthetic Demonstration Mode:**
> This service strictly distinguishes between **authentic research data** and **synthetic demonstration data**:
> - **Authentic Production Mode:** Strictly awaits authentic factory operational datasets. When untrained, it returns `HTTP 503 Service Unavailable` rather than fabricating predictions.
> - **Synthetic Demonstration Mode:** Demonstrates the end-to-end anomaly detection architecture using physics-inspired synthetic wastewater telemetry and 6 injected tampering scenarios.
>
> > [!WARNING]
> > **Demonstration Disclaimer:**
> > *This model demonstrates the EnviroTrust anomaly-detection architecture using synthetic wastewater data. Its results do not establish real-world industrial accuracy or validate detection performance on actual factory telemetry.*

---

## Architecture Overview

```
EnviroTrust-ML-part/
└── ml-service/
    ├── config/
    │   └── data_config.yaml         # Configurable column mappings & hyperparameters
    ├── data/
    │   ├── raw/                     # Authentic research data (preserved and immutable)
    │   │   ├── extracted_water_quality_samples.csv
    │   │   ├── data_spreadsheet_-_Sheet1.csv
    │   │   ├── extracted_water_quality_samples_v2.csv
    │   │   └── industrial_water_pollution_ml.ipynb
    │   ├── synthetic_demo/          # Synthetic demonstration data (isolated)
    │   │   ├── normal_demo.csv      # 1,000 normal synthetic observations
    │   │   └── attack_demo.csv      # 200 labelled tampering attack observations
    │   └── processed/               # Cached intermediate representations
    ├── models/                      # Production model destination
    │   └── demo/                    # Isolated synthetic demonstration model artifacts
    │       ├── isolation_forest_demo.joblib
    │       ├── preprocessor_demo.joblib
    │       ├── consistency_rules.json   # Learned envelopes (human-auditable JSON)
    │       └── model_metadata.json
    ├── src/
    │   ├── __init__.py
    │   └── synthetic_data_generator.py # Synthetic dataset generator & demo trainer
    ├── tests/                       # 86 unit & integration tests
    │   ├── test_api.py
    │   ├── test_audit_fixes.py
    │   ├── test_authentic_data_integration.py
    │   ├── test_consistency_rules.py # Rule layer, held-out split, per-scenario eval
    │   ├── test_data_loader.py
    │   ├── test_data_validator.py
    │   ├── test_evaluation.py
    │   ├── test_feature_engineering.py
    │   ├── test_further_data_integration.py
    │   ├── test_isolation_forest.py
    │   ├── test_preprocessing.py
    │   ├── test_synthetic_demo.py   # Demonstration mode & attack scenario tests
    │   └── test_train_pipeline.py
    ├── anomaly_detection.py         # Isolation Forest wrapper & threshold calibration
    ├── consistency_rules.py         # Physical invariants & learned operating envelopes
    ├── data_loader.py               # Ingestion, profiling, & parameter table transposition
    ├── data_profile.md              # Authentic data profiling & readiness report
    ├── data_validator.py            # Candidate validation, label hygiene, and temporal checks
    ├── evaluation.py                # Supervised (precision/recall/FPR/F1) and unsupervised metrics
    ├── feature_engineering.py       # Dynamic column mapping & feature selector
    ├── main.py                      # FastAPI application with dual-mode support
    ├── predict.py                   # Real-time inference engine with schema verification
    ├── preprocessing.py             # Leak-free imputation, encoding, and schema persistence
    ├── pytest.ini                   # Pytest configuration
    ├── requirements.txt             # Production and development dependencies
    └── train.py                     # Training workflow (chronological split, label-aware)
```

---

## Supported Operating Modes

### 1. Authentic Production Mode (Default)
* **Goal:** High-assurance anomaly detection based exclusively on genuine industrial telemetry.
* **Integrity Guarantee:** Never trains on fabricated measurements.
* **Untrained Status:** If authentic datasets have not yet been trained, endpoints return `HTTP 503 Service Unavailable` with an informative message.
* **Model Storage:** Saved to `models/isolation_forest.joblib`.

### 2. Synthetic Demonstration Mode
* **Goal:** Full-pipeline demonstration and hackathon verification of sensor-tampering detection.
* **Data Origin:** Generated by `src/synthetic_data_generator.py` using physical mass-conservation and biodegradability relationships inspired by textile dye wastewater research.
* **Attack Injection:** Includes 6 separately labelled tampering attack scenarios.
* **Provenance Stamping:** Every record and API response is stamped with `model_mode: "SYNTHETIC_DEMO"` and `data_provenance: "SYNTHETIC"`.
* **Model Storage:** Saved strictly to `models/demo/isolation_forest_demo.joblib` without overwriting production paths.

---

## Synthetic Attack Scenarios

The synthetic generator simulates 6 realistic tampering and malfunction scenarios:

| Attack Scenario | Physical Description | Tampering Mechanism |
| :--- | :--- | :--- |
| **`low_cod`** | Unnaturally low COD ($15 - 75\,\text{mg/L}$) while background ionic solids reflect industrial effluent. | Dilution tampering (washing sensor probe in tap water). |
| **`high_cod`** | Acute pollution surge ($24,000 - 52,000\,\text{mg/L}$). | Illegal untreated chemical dump or spent dye bath slug discharge. |
| **`cod_turbidity_decoupling`** | Physical relationship between organics and suspended solids severed (e.g. COD $20,000\,\text{mg/L}$ with turbidity $12\,\text{NTU}$). | Sensor spoofing or bypass tampering. |
| **`sensor_flatline`** | Identical floating-point values reported over consecutive intervals with zero natural sensor noise ($\sigma \approx 0$). | Sensor freeze, hardware failure, or replay attack. |
| **`sudden_spike`** | Instantaneous 5x–8x step increase or 95% plunge between adjacent 10-minute intervals. | Sensor glitch or impossible instantaneous chemical transition. |
| **`gradual_drift`** | Monotonic bias ramp shifting readings downwards by $130\,\text{mg/L}$ per step over 30+ readings. | Diaphragm fouling or stealthy recalibration tampering. |

> [!NOTE]
> In accordance with EnviroTrust training safeguards, **zero attack records enter the training subset**. The demonstration model is trained exclusively on normal synthetic records and evaluates against separate attack records.
>
> The latest **15% of normal records (150 rows) are held out** and never used for fitting or threshold calibration, so the reported false-positive rate is measured on unseen normal data.

---

## Core Algorithm: Isolation Forest

Unlike supervised regressors that predict a scalar target from inputs, **Isolation Forest** (`sklearn.ensemble.IsolationForest`) isolates observations across a joint multi-dimensional feature space:

$$\mathbf{x} = [\text{COD\_mg\_L}, \text{BOD\_mg\_L}, \text{pH}, \text{TDS\_mg\_L}, \text{turbidity\_NTU}, \text{colour\_PtCo}, \text{total\_solids\_mg\_L}, \dots]$$

1. **Isolation Trees (iTrees):** Normal operational conditions form dense clusters. Manipulated sensor values occupy sparse regions and are isolated with short tree path lengths.
2. **Directional Anomaly Score:** Anomaly scores are standardized such that **higher values represent higher suspicion**:
   $$\text{anomaly\_score}(\mathbf{x}) = -\text{decision\_function}(\mathbf{x})$$
3. **Threshold Calibration:** The alert threshold $T$ is empirically calibrated on clean normal validation data:
   $$T = \text{Percentile}_{99}(\text{Normal Validation Scores})$$

---

## Physical Consistency Rules

Isolation Forest scores the joint feature space. A single absurd measurement (e.g. COD = 45 mg/L in raw dye effluent) only shortens the path in the few trees that happen to split on that feature, so it is often scored as normal. The rule layer (`consistency_rules.py`) checks each reading directly and reports **which** check failed. A reading is `SUSPICIOUS` if **either** detector flags it.

| Rule | Type | Check |
| :--- | :--- | :--- |
| `ph_physical_range` | Fixed invariant | $0 \le \text{pH} \le 14$ |
| `negative_concentration` | Fixed invariant | mg/L, NTU and Pt-Co values $\ge 0$ |
| `bod_exceeds_cod` | Fixed invariant | $\text{BOD} \le \text{COD}$ |
| `ions_exceed_tds` | Fixed invariant | $\text{Cl}^- + \text{SO}_4^{2-} \le \text{TDS}$ |
| `solids_mass_balance` | Fixed invariant | $\lvert \text{TS} - (\text{TDS} + \text{TSS}) \rvert / \text{TS} \le 10\%$ |
| `operating_envelope` | Learned | Per-feature 0.5–99.5% quantile band of normal training data, widened by 10% |
| `ratio_band` | Learned | BOD/COD and turbidity/TSS ratio bands from normal training data |

Envelopes are learned from **normal training records only** and saved as `consistency_rules.json` next to the model. Rules evaluate raw measurements, never imputed values: a missing measurement is skipped. Settings live under `consistency_rules` in `config/data_config.yaml`.

---

## REST API Specification

### 1. Service Health Check
* **Endpoint:** `GET /health`
* **Response:**
  ```json
  {
    "status": "healthy",
    "service": "EnviroTrust ML Anomaly Detection Service",
    "version": "1.1.0",
    "timestamp_utc": "2026-10-31T21:15:00Z"
  }
  ```

### 2. Model Status Inspection
* **Endpoint:** `GET /model/status` (Authentic) | `GET /model/status?mode=demo` (Demo) | `GET /demo/model/status`
* **Response (Demo Mode):**
  ```json
  {
    "is_trained": true,
    "model_mode": "SYNTHETIC_DEMO",
    "data_provenance": "SYNTHETIC",
    "model_version": "iforest-demo-v1.0",
    "threshold": 0.0694,
    "features": ["COD_mg_L", "BOD_mg_L", "pH", "TDS_mg_L", "turbidity_NTU", "colour_PtCo", "total_solids_mg_L", "total_suspended_solids_mg_L", "chloride_mg_L", "sulphate_mg_L"]
  }
  ```

### 3. Telemetry Anomaly Analysis
* **Authentic Endpoint:** `POST /analyze/telemetry` *(returns HTTP 503 while awaiting authentic data)*
* **Demo Endpoint:** `POST /analyze/telemetry?mode=demo` or `POST /demo/analyze/telemetry`
* **Request Payload:**
  ```json
  {
    "readingId": "TEL-2026-1001",
    "sensorId": "SENSOR-EFFLUENT-01",
    "timestamp": "2026-10-31T12:00:00Z",
    "measurements": {
      "COD_mg_L": 4800.0,
      "BOD_mg_L": 1200.0,
      "pH": 7.0,
      "TDS_mg_L": 2800.0,
      "turbidity_NTU": 250.0,
      "colour_PtCo": 2500.0,
      "total_solids_mg_L": 3100.0,
      "total_suspended_solids_mg_L": 300.0,
      "chloride_mg_L": 350.0,
      "sulphate_mg_L": 400.0
    }
  }
  ```
* **Response (Demo Mode):**
  ```json
  {
    "reading_id": "TEL-2026-1001",
    "sensor_id": "SENSOR-EFFLUENT-01",
    "timestamp": "2026-10-31T12:00:00Z",
    "anomaly_score": -0.0142,
    "anomaly_threshold": 0.0694,
    "is_anomaly": false,
    "isolation_forest_flag": false,
    "rule_violations": [],
    "detection_sources": [],
    "status": "CONSISTENT",
    "model_version": "iforest-demo-v1.0",
    "model_mode": "SYNTHETIC_DEMO",
    "data_provenance": "SYNTHETIC",
    "evaluated_features": ["COD_mg_L", "BOD_mg_L", "pH", "TDS_mg_L", "turbidity_NTU", "colour_PtCo", "total_solids_mg_L", "total_suspended_solids_mg_L", "chloride_mg_L", "sulphate_mg_L"],
    "evaluated_at_utc": "2026-10-31T21:15:30Z"
  }
  ```

* **Flagged reading (rule violation):** `rule_violations` explains the decision, e.g.
  ```json
  {
    "is_anomaly": true,
    "isolation_forest_flag": false,
    "detection_sources": ["physical_rules"],
    "rule_violations": [
      {
        "rule": "operating_envelope",
        "feature": "COD_mg_L",
        "value": 45.0,
        "expected_range": [3488.917, 8062.4819],
        "message": "COD_mg_L = 45.00 is outside the learned normal range [3488.917, 8062.4819]."
      }
    ],
    "status": "SUSPICIOUS"
  }
  ```

---

## Synthetic Demonstration Results

Measured on 150 **held-out** normal records and all 200 attack records (`python -m src.synthetic_data_generator --evaluate`).

| Detector | Recall | False-positive rate | Precision |
| :--- | :--- | :--- | :--- |
| Isolation Forest only | 10.5% | 0.7% (1/150) | 95.5% |
| Physical rules only | 93.0% | 0.0% (0/150) | 100% |
| **Combined** | **93.0%** | **0.7% (1/150)** | **99.5%** |

| Scenario | Isolation Forest | Combined |
| :--- | :--- | :--- |
| `low_cod` | 0% | 100% |
| `high_cod` | 12% | 100% |
| `cod_turbidity_decoupling` | 48% | 100% |
| `sudden_spike` | 0% | 100% |
| `gradual_drift` | 3% | 86% |
| `sensor_flatline` | 0% | 73% |

> [!WARNING]
> **How to read these numbers:**
> - The rules check relationships the generator itself encodes (e.g. turbidity ≈ 0.85 × TSS), so high synthetic recall is expected and does **not** transfer automatically to real telemetry.
> - `sensor_flatline` and `gradual_drift` are temporal attacks. They are caught here only because the generator's attack rows also break a single-reading relationship (e.g. the frozen turbidity value is inconsistent with TSS) or drift outside the envelope. A genuine flatline at plausible values would not be detected until window-based checks are added.
> - Precision depends on the 150:200 normal-to-attack mix in the test set. Real attack prevalence is far lower, so real-world precision would be lower at the same false-positive rate.
> - A 0.7% false-positive rate on 150 samples is a single reading; its confidence interval is wide.

---

## CLI & Pipeline Commands

### 1. Generate Synthetic Datasets
```bash
cd ml-service
python -m src.synthetic_data_generator --generate --seed 42
```
Generates:
* `data/synthetic_demo/normal_demo.csv` (1,000 normal records)
* `data/synthetic_demo/attack_demo.csv` (200 labelled tampering attack records across 6 scenarios)

### 2. Train Synthetic Demo Model
```bash
python -m src.synthetic_data_generator --train --seed 42
```
Holds out the latest 15% of `normal_demo.csv`, trains on the earliest records, calibrates the threshold and learns rule envelopes on normal data only, and persists artifacts to `models/demo/`.

### 3. Evaluate Demo Model Against Attacks
```bash
python -m src.synthetic_data_generator --evaluate
```
Evaluates against the 150 held-out normal records and all 200 attack records, reporting combined, Isolation-Forest-only and rules-only metrics plus a per-scenario breakdown.

### 4. Run Full Pytest Test Suite
```bash
pytest -v
```
**86 passing tests** verifying generator reproducibility, schema contracts, physical interrelationships, data isolation, artifact protection, the physical rule layer, held-out evaluation, and REST API endpoints.
