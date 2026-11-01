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
    │   │   ├── Mining_Analysis.ipynb    # Mining & critical mineral recovery notebook
    │   │   ├── bangladesh_textile_dataset.csv
    │   │   ├── bangladesh_textile_industry_anomaly_detection.ipynb
    │   │   ├── china.csv            # Yellow River Basin WWTP influent & cost metrics
    │   │   ├── chinapy.ipynb
    │   │   ├── data_spreadsheet_-_Sheet1.csv
    │   │   ├── extracted_water_quality_samples.csv
    │   │   ├── extracted_water_quality_samples_v2.csv
    │   │   ├── industrial_water_pollution_ml.ipynb
    │   │   ├── microorganism.csv    # 27 industrial streams anaerobic digestion literature
    │   │   ├── microorganismpy.ipynb
    │   │   ├── mining.csv           # 22 industrial activities mining & mineral recovery
    │   │   ├── misa_electric_power_generation_2024.csv
    │   │   ├── sheet3.ipynb
    │   │   └── Untitled spreadsheet - Sheet3.csv
    │   ├── synthetic_demo/          # Synthetic demonstration data (isolated)
    │   │   ├── normal_demo.csv      # 1,000 normal synthetic observations
    │   │   └── attack_demo.csv      # 200 labelled tampering attack observations
    │   └── processed/               # Cached intermediate representations
    ├── models/                      # Production model destination
    │   └── demo/                    # Isolated synthetic demonstration model artifacts
    │       ├── isolation_forest_demo.joblib
    │       ├── preprocessor_demo.joblib
    │       ├── consistency_rules.json   # Learned envelopes (human-auditable JSON)
    │       ├── cusum.json               # CUSUM baseline & threshold (human-auditable JSON)
    │       └── model_metadata.json
    ├── src/
    │   ├── __init__.py
    │   └── synthetic_data_generator.py # Synthetic dataset generator & demo trainer
    ├── tests/                       # 177 unit & integration tests
    │   ├── test_api.py              # Endpoint tests, auth, schema & error handling
    │   ├── test_audit_fixes.py
    │   ├── test_authentic_data_integration.py
    │   ├── test_bangladesh_textile_integration.py # 15-factory SCADA & effluent tests
    │   ├── test_consistency_rules.py # Rule layer, held-out split, per-scenario eval
    │   ├── test_cusum.py            # CUSUM recursion, calibration, per-sensor state, bias detection
    │   ├── test_data_loader.py
    │   ├── test_data_validator.py
    │   ├── test_evaluation.py
    │   ├── test_feature_engineering.py
    │   ├── test_further_data_integration.py
    │   ├── test_isolation_forest.py
    │   ├── test_misa_power_integration.py # Ontario MISA 2024 power generation tests
    │   ├── test_new_data_integration.py # China WWTP, 27-sector micro & 22-activity mining tests
    │   ├── test_preprocessing.py
    │   ├── test_store_and_forward.py # Edge gateway offline buffering & automatic retry
    │   ├── test_synthetic_demo.py   # Demonstration mode & attack scenario tests
    │   ├── test_tamper_alarm.py     # Physical tamper alarm & cryptographic audit ledger
    │   └── test_train_pipeline.py
    ├── anomaly_detection.py         # Isolation Forest wrapper & threshold calibration
    ├── consistency_rules.py         # Physical invariants & learned operating envelopes
    ├── cusum.py                     # CUSUM sustained-shift detector (slow sensor bias)
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
    ├── store_and_forward.py         # SQLite edge queue, priority transmitter & simulator
    ├── tamper_alarm.py              # Physical sensor tamper alarm manager & simulator
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

### Rolling-Window Features (Stuck Sensor Detection)

A single reading cannot show that a sensor is frozen: a stuck value can sit at a perfectly normal level. For each measurement, `feature_engineering.py` adds `<feature>_roll_log_std`, the log1p of the trailing-window standard deviation, computed per sensor stream (`sensorId`) in timestamp order (`rolling_features` in `config/data_config.yaml`, default window 5).

* A stuck sensor drives the feature to exactly 0, far below the noise floor learned from normal data, so Isolation Forest isolates it quickly. The log scale matters: on a raw scale the 0 sits close to the training range and is rarely isolated (6% vs 88% on the synthetic `sensor_flatline` scenario).
* Windows look backwards only and are computed on the full chronological stream before splitting or label filtering, so no future values leak in and windows never span removed rows.
* Skipped for datasets without a timestamp column (row order of non-temporal data is meaningless). Rolling features are not given to the rule layer, which stays a per-reading check.
* **Inference:** the predictor keeps the last few readings per `sensorId` in memory. The first `window - 1` readings of a sensor (and the first after a restart or model reload) have no full window; the feature is then imputed and `rolling_window_filled` is `false` in the response, meaning stuck-sensor detection is not yet active for that reading. A retried reading with the same `readingId` replaces the buffered one instead of counting as a repeat. The buffer is per process, so with several workers each sensor must be routed to one worker.

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

## CUSUM Sustained-Shift Detection

Isolation Forest and the rules judge one reading at a time. A sensor that slowly develops a bias (calibration drift, probe fouling) keeps producing individually plausible readings and is never flagged. `cusum.py` runs a two-sided tabular CUSUM per measurement and per sensor stream, accumulating small persistent deviations from the normal baseline:

$$z_t = \frac{x_t - \mu}{\sigma}, \quad S^+_t = \max(0,\ S^+_{t-1} + z_t - k), \quad S^-_t = \max(0,\ S^-_{t-1} - z_t - k), \quad \text{alarm if } S^\pm_t \ge h$$

* $\mu$, $\sigma$ are learned from **normal training records only**. $h$ is calibrated on the normal validation stream (peak statistic × 1.25, never below 4). Both are saved as `cusum.json` next to the model.
* **k = 1.0**, not the textbook 0.5. At 0.5 the normal daily COD cycle alone accumulates into alarms (13% of normal held-out readings in testing). The cost is delay: on the demo model a 1σ bias first alarms after 33–66 readings (5.5–11 hours at 10-minute readings) and a 1.5σ bias after 20–30, depending on the feature.
* Alarms are explainable (`cusum_alarms`: feature, `upward`/`downward`, statistic, threshold) and add `cusum` to `detection_sources`. The statistic does not reset on alarm: a sustained shift keeps alarming until the reading returns to normal and the statistic decays.
* Fitted only for datasets with a timestamp column. At inference the statistics are kept per `sensorId` in memory (reset on restart or model reload, and per process). A retried `readingId` is not counted twice.
* **Not a fix for everything:** a sensor stuck exactly at its normal mean produces $z \approx 0$ and never alarms (the rolling-window features cover that case), and the first readings of a slow ramp are inside normal noise for every detector.

---

## Service Startup & Backend Integration

### 1. Starting the Service
Start the FastAPI application using `uvicorn`:

```bash
cd ml-service
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

* Binding to `--host 0.0.0.0` ensures the service listens on all network interfaces, allowing host processes, Docker containers, backend APIs, and devices across the Local Area Network (LAN) to connect.
* For hot-reloading during development, append `--reload`:
  ```bash
  .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  ```

### 2. Local Area Network (LAN) Connection Guide
To connect backend services running on another machine or container on the same network:

1. **Find Host LAN IP Address:**
   * **macOS:**
     ```bash
     ipconfig getifaddr en0
     # Example output: 192.168.1.105
     ```
   * **Linux:**
     ```bash
     hostname -I | awk '{print $1}'
     ```
2. **Verify LAN Connectivity:**
   From any device on the same local network:
   ```bash
   curl -s http://192.168.1.105:8000/health
   ```
   Expected response:
   ```json
   {"status":"healthy","service":"EnviroTrust ML Anomaly Detection Service","version":"1.1.0",...}
   ```
3. **Firewall Note:** Ensure incoming traffic on port `8000` is permitted by your operating system or local network firewall.

---

## Backend Authentication

EnviroTrust ML service implements a configurable, secret-free token authentication mechanism.

### Configuration
Set the authentication token via environment variable before starting the service:

```bash
# Supported variable names (checked in precedence order):
export ML_SERVICE_TOKEN="envirotrust-backend-secret-xyz"
# or: export ENVIROTRUST_ML_TOKEN="envirotrust-backend-secret-xyz"
# or: export ML_SERVICE_API_KEY="envirotrust-backend-secret-xyz"

.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

### Authentication Behavior
* **Enforced Mode (Variable Set):**
  All protected analytical and status endpoints (`/model/status`, `/analyze/telemetry`, `/demo/analyze/telemetry`, etc.) require valid credentials.
  Clients must pass the token in any of the following supported headers:
  * `X-ML-Service-Token: envirotrust-backend-secret-xyz` *(Recommended)*
  * `Authorization: Bearer envirotrust-backend-secret-xyz`
  * `X-API-Key: envirotrust-backend-secret-xyz`

  Missing or invalid tokens return `HTTP 401 Unauthorized`. Constant-time comparison (`hmac.compare_digest`) is used to prevent timing attacks.
* **Open Development Mode (Variable Unset / Empty):**
  If no token variable is set, authentication is bypassed for local development and offline unit testing.
* **Public Health Probe:**
  `GET /health`, `/docs`, and `/openapi.json` are always public (unauthenticated) so orchestration healthchecks (e.g., Docker, Kubernetes, load balancers) operate seamlessly.

---

## REST API Specification & Exact Contracts

### 1. Service Health Check (Public)
* **Endpoint:** `GET /health`
* **Headers:** None required
* **Response (HTTP 200 OK):**
  ```json
  {
    "status": "healthy",
    "service": "EnviroTrust ML Anomaly Detection Service",
    "version": "1.1.0",
    "timestamp_utc": "2026-11-01T02:00:00.000000+00:00"
  }
  ```

### 2. Model Status Inspection
* **Endpoints:**
  * Authentic Model: `GET /model/status`
  * Demonstration Model: `GET /demo/model/status` or `GET /model/status?mode=demo`
* **Headers:** `X-ML-Service-Token: <token>` (if configured)
* **Response (HTTP 200 OK — Demo Mode Loaded):**
  ```json
  {
    "is_trained": true,
    "model_mode": "SYNTHETIC_DEMO",
    "data_provenance": "SYNTHETIC",
    "model_version": "iforest-demo-v1.0",
    "training_timestamp_utc": "2026-10-31T17:35:16.033376+00:00",
    "feature_schema": {
      "numerical_features": [
        "COD_mg_L", "BOD_mg_L", "pH", "TDS_mg_L", "turbidity_NTU",
        "colour_PtCo", "total_solids_mg_L", "total_suspended_solids_mg_L",
        "chloride_mg_L", "sulphate_mg_L"
      ],
      "categorical_features": [],
      "feature_names_out": [
        "num__COD_mg_L", "num__BOD_mg_L", "num__pH", "num__TDS_mg_L",
        "num__turbidity_NTU", "num__colour_PtCo", "num__total_solids_mg_L",
        "num__total_suspended_solids_mg_L", "num__chloride_mg_L", "num__sulphate_mg_L"
      ],
      "feature_count": 10
    },
    "features": [
      "num__COD_mg_L", "num__BOD_mg_L", "num__pH", "num__TDS_mg_L",
      "num__turbidity_NTU", "num__colour_PtCo", "num__total_solids_mg_L",
      "num__total_suspended_solids_mg_L", "num__chloride_mg_L", "num__sulphate_mg_L"
    ],
    "threshold": 0.0694,
    "calibration_metadata": {
      "strategy": "empirical_percentile",
      "threshold_percentile": 99.0,
      "validation_sample_count": 170,
      "threshold": 0.0694
    }
  }
  ```
* **Response (HTTP 200 OK — Authentic Mode Untrained):**
  ```json
  {
    "is_trained": false,
    "model_mode": "AUTHENTIC",
    "data_provenance": "AUTHENTIC",
    "message": "Model is not trained. Awaiting authentic industrial operational dataset."
  }
  ```

### 3. Telemetry Anomaly Analysis (`POST /analyze/telemetry`)
* **Endpoints:**
  * Demonstration: `POST /demo/analyze/telemetry` or `POST /analyze/telemetry?mode=demo`
  * Authentic: `POST /analyze/telemetry` *(returns HTTP 503 until authentic training is executed)*
* **Headers:**
  * `Content-Type: application/json`
  * `X-ML-Service-Token: <token>` (if configured)

#### Exact Request JSON
```json
{
  "readingId": "TEL-2026-001",
  "sensorId": "SENSOR-EFFLUENT-01",
  "timestamp": "2026-11-01T02:00:00Z",
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
  },
  "context": {
    "productionRate": 900.0,
    "wastewaterFlow": 800.0,
    "operatingMode": "NORMAL",
    "energyConsumption": 950.0
  }
}
```

#### Exact Response JSON (Nominal Reading — HTTP 200 OK)
```json
{
  "reading_id": "TEL-2026-001",
  "sensor_id": "SENSOR-EFFLUENT-01",
  "timestamp": "2026-11-01T02:00:00Z",
  "anomaly_score": 0.0125,
  "anomaly_threshold": 0.0694,
  "is_anomaly": false,
  "isolation_forest_flag": false,
  "rule_violations": [],
  "detection_sources": [],
  "status": "CONSISTENT",
  "model_version": "iforest-demo-v1.0",
  "model_mode": "SYNTHETIC_DEMO",
  "data_provenance": "SYNTHETIC",
  "evaluated_features": [
    "num__COD_mg_L",
    "num__BOD_mg_L",
    "num__pH",
    "num__TDS_mg_L",
    "num__turbidity_NTU",
    "num__colour_PtCo",
    "num__total_solids_mg_L",
    "num__total_suspended_solids_mg_L",
    "num__chloride_mg_L",
    "num__sulphate_mg_L"
  ],
  "evaluated_at_utc": "2026-11-01T02:30:00.000000+00:00"
}
```

#### Exact Response JSON (Flagged Suspicious Tampering / Outlier — HTTP 200 OK)
```json
{
  "reading_id": "TEL-ATK-001",
  "sensor_id": "SENSOR-EFFLUENT-01",
  "timestamp": "2026-11-01T02:00:00Z",
  "anomaly_score": 0.1450,
  "anomaly_threshold": 0.0694,
  "is_anomaly": true,
  "isolation_forest_flag": true,
  "rule_violations": [
    {
      "rule": "operating_envelope",
      "feature": "COD_mg_L",
      "value": 45.0,
      "expected_range": [3488.92, 8062.48],
      "message": "COD_mg_L = 45.00 is outside the learned normal range [3488.92, 8062.48]."
    }
  ],
  "detection_sources": [
    "isolation_forest",
    "physical_rules"
  ],
  "status": "SUSPICIOUS",
  "model_version": "iforest-demo-v1.0",
  "model_mode": "SYNTHETIC_DEMO",
  "data_provenance": "SYNTHETIC",
  "evaluated_features": [
    "num__COD_mg_L",
    "num__BOD_mg_L",
    "num__pH",
    "num__TDS_mg_L",
    "num__turbidity_NTU",
    "num__colour_PtCo",
    "num__total_solids_mg_L",
    "num__total_suspended_solids_mg_L",
    "num__chloride_mg_L",
    "num__sulphate_mg_L"
  ],
  "evaluated_at_utc": "2026-11-01T02:30:00.000000+00:00"
}
```

---

## Required Model Features by Active Model

The ML service enforces strict schema checking at inference time. Missing measurements are **never** silently imputed or fabricated.

### 1. Synthetic Demonstration Model (`models/demo/`)
Requires all **10 active environmental parameters**:
1. `COD_mg_L` (Chemical Oxygen Demand in mg/L)
2. `BOD_mg_L` (Biochemical Oxygen Demand in mg/L)
3. `pH` (pH value, dimensionless, 0–14)
4. `TDS_mg_L` (Total Dissolved Solids in mg/L)
5. `turbidity_NTU` (Nephelometric Turbidity Units)
6. `colour_PtCo` (Colour in Platinum-Cobalt units)
7. `total_solids_mg_L` (Total Solids in mg/L)
8. `total_suspended_solids_mg_L` (Total Suspended Solids in mg/L)
9. `chloride_mg_L` (Chloride ion concentration in mg/L)
10. `sulphate_mg_L` (Sulphate ion concentration in mg/L)

*Note: Column mapping aliases (`cod` $\to$ `COD_mg_L`, `bod` $\to$ `BOD_mg_L`, `ph` $\to$ `pH`, `tds` $\to$ `TDS_mg_L`, `turbidity` $\to$ `turbidity_NTU`) are automatically applied.*

### 2. Authentic Bangladesh Factory Model (`models/`)
Requires all **6 factory-level operational and effluent parameters**:
1. `COD_mg_L` (Chemical Oxygen Demand in mg/L)
2. `BOD_mg_L` (Biochemical Oxygen Demand in mg/L)
3. `production` (Factory production throughput in kg)
4. `electricity_consumption` (Factory power usage in kWh)
5. `water_flow` (Discharged wastewater in Million Liters)
6. `extracted_groundwater` (Groundwater extracted in Million Liters)

---

## HTTP Error Handling & Status Codes

| Status Code | Reason | Cause | Response Example |
| :---: | :--- | :--- | :--- |
| **`401`** | **Unauthorized** | Missing or incorrect `X-ML-Service-Token` when auth is configured. | `{"detail": "Missing authentication token. Pass 'X-ML-Service-Token' header."}` |
| **`422`** | **Unprocessable Entity** | Required model features are missing from payload, or payload is malformed. | `{"detail": "Missing required model features in incoming record: ['sulphate_mg_L']. The active model requires all training features: [...]. Features will not be silently fabricated."}` |
| **`503`** | **Service Unavailable** | Requested model mode is untrained or artifacts are missing. | `{"detail": "Model is not trained. Awaiting authentic industrial operational dataset."}` |
| **`500`** | **Internal Server Error** | Unexpected execution exception during inference. | `{"detail": "Inference execution failed: <error_message>"}` |

---

## Telemetry Immutability Guarantee
The EnviroTrust ML service is strictly read-only with respect to incoming telemetry. Incoming payloads are parsed into immutable data frames and never mutated in-place. Automated integration tests (`test_telemetry_input_immutability`) verify that caller telemetry dictionaries remain bit-for-bit identical before and after inference.

---

## Synthetic Demonstration Results

Measured on 150 **held-out** normal records and all 200 attack records (`python -m src.synthetic_data_generator --evaluate`).

| Detector | Recall | False-positive rate | Precision |
| :--- | :--- | :--- | :--- |
| Isolation Forest only | 64.5% | 1.3% (2/150) | 98.5% |
| Physical rules only | 93.0% | 0.0% (0/150) | 100% |
| CUSUM only | 87.0% | 0.0% (0/150) | 100% |
| **Combined** | **97.0%** | **1.3% (2/150)** | **99.0%** |

| Scenario | Isolation Forest | Rules | CUSUM | Combined |
| :--- | :--- | :--- | :--- | :--- |
| `low_cod` | 88% | 100% | 100% | 100% |
| `high_cod` | 76% | 100% | 100% | 100% |
| `cod_turbidity_decoupling` | 73% | 100% | 100% | 100% |
| `sudden_spike` | 58% | 100% | 100% | 100% |
| `gradual_drift` | 9% | 86% | 83% | 86% |
| `sensor_flatline` | 88% | 73% | 39% | 97% |

Each scenario is scored as its own sensor stream (`sensorId` per scenario). Scored as one continuous stream, sequential detectors carry state between scenarios: the large `low_cod` shift alone kept CUSUM alarming through every later scenario (100% everywhere), which says nothing about detection.

> [!WARNING]
> **How to read these numbers:**
> - The rules check relationships the generator itself encodes (e.g. turbidity ≈ 0.85 × TSS), so high synthetic recall is expected and does **not** transfer automatically to real telemetry.
> - `sensor_flatline` is caught by the rolling-window features (Isolation Forest 0% → 88%), including flatlines at plausible values that pass every rule (`tests/test_rolling_features.py`). The remaining 12% are the first 4 frozen readings, whose window still contains earlier non-frozen values; a stuck sensor is only detectable once a full window of identical readings exists.
> - The Isolation Forest gains on the other scenarios are partly a generator artefact: each scenario is a contiguous block of attack rows whose variance differs from normal operation. Do not expect the same lift on real, isolated attacks.
> - **CUSUM does not raise combined recall on these scenarios.** Every attack it catches here is already caught by the rules or Isolation Forest. On `gradual_drift`, the rule envelope alarms by the 4th reading and CUSUM by the 7th; the first 5 drift readings are inside normal noise for every detector. CUSUM's value is a bias that stays inside the envelope, which no scenario here simulates: for a chloride probe reading 1.5σ low, the rules catch the ~10% of readings that fall off the envelope tail, while CUSUM first alarms at the 20th reading and flags every reading from the 30th onward (`tests/test_cusum.py`).
> - Precision depends on the 150:200 normal-to-attack mix in the test set. Real attack prevalence is far lower, so real-world precision would be lower at the same false-positive rate.
> - A 1.3% false-positive rate on 150 samples is two readings; its confidence interval is wide. CUSUM's 0/150 is equally limited evidence for its long-run false-alarm rate.

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
**177 passing tests** verifying generator reproducibility, schema contracts, physical interrelationships, data isolation, artifact protection, the physical rule layer, held-out evaluation, authentic dataset integrations (Bangladesh, Ontario MISA, China WWTP, 27-sector Microorganism literature & 22-activity Mining study), backend token authentication, physical sensor tamper alarm engine, cryptographic audit ledger, edge gateway offline store-and-forward queue, and REST API endpoints.

---

## Physical Sensor Tamper Alarm (Hardware Security)

> [!IMPORTANT]
> **Strict Architectural Boundary: Security vs. ML Anomaly Detection**
> - The **Physical Sensor Tamper Alarm** is a **hardware security feature**, strictly independent of statistical Machine Learning anomaly detection (Isolation Forest).
> - While Isolation Forest detects subtle multivariate drift and unrecorded process excursions in continuous chemical measurements, physical tampering represents binary hardware circuit disruptions (e.g. broken tamper seals, open enclosure switches, severed probe loops).
> - Physical tamper alarms **never** alter Isolation Forest model parameters or weights.
> - An event received by this service indicates **possible physical hardware interference**, alerting government regulators to inspect field hardware; it is **not** an automated judicial declaration of pollution fraud.

### 1. Supported Hardware Event Types & Prioritization

The service defines 5 hardware event types with strict severity and categorization rules:

| Event Type | Category | Severity | Confirmed Tamper? | Physical Mechanism & Indicator |
| :--- | :--- | :--- | :---: | :--- |
| `ENCLOSURE_OPEN` | `PHYSICAL_TAMPER` | `CRITICAL` | **Yes** | Microswitch/reed switch on unit door indicates open casing. |
| `SEAL_BROKEN` | `PHYSICAL_TAMPER` | `CRITICAL` | **Yes** | Continuous electrical tamper-seal wire broken or open-circuit. |
| `SENSOR_REMOVED` | `PHYSICAL_TAMPER` | `CRITICAL` | **Yes** | Bracket interlock/proximity switch indicates probe extracted. |
| `SENSOR_DISCONNECTED` | `PHYSICAL_TAMPER` | `HIGH` | **Yes** | NAMUR NE43 breakdown (<3.6 mA) or zero-current circuit break. |
| `DEVICE_OFFLINE` | `CONNECTIVITY_WARNING` | `MEDIUM` | **No** | Watchdog heartbeat timeout. Connectivity warning (unconfirmed tampering). |

> [!NOTE]
> **Device Offline vs. Confirmed Physical Tamper:**
> `DEVICE_OFFLINE` triggers when unit heartbeats cease. Because network outages, power interruptions, or ISP drops occur without human interference, the system strictly tags offline events as `CONNECTIVITY_WARNING` (`is_confirmed_tamper=False`). If an offline device subsequently sends a stored physical tamper event or an active enclosure alert exists, the physical tamper condition takes precedence (`TAMPER_ACTIVE`).

### 2. Physical Tamper Detection Principles & Hardware Requirements

In real-world wastewater monitoring installations (e.g. textile dye basins, power generation effluent channels), detection requires dedicated field hardware:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   FIELD MONITORING UNIT (IP66/NEMA 4X)                 │
│                                                                        │
│  [Enclosure Microswitch] ──────────┐                                   │
│  [Tamper-Seal Loop Wire] ──────────┼──> [Microcontroller / IoT SoC]    │
│  [Mounting Interlock Switch] ──────┤    (ESP32-S3 / STM32 / Cortex-M)  │
│  [4-20mA Current Loop Sensing] ────┘    - Hardware GPIO Pull-ups       │
│                                         - Hardware Watchdog            │
│  [Cryptographic Secure Element] ─────── - Real-Time Clock (RTC / NTP)  │
│  (ATECC608A / Ed25519 Keypair)          - Monotonic Sequence Counter   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Signed Payload (TLS / MQTT / HTTPS)
                                    ▼
                     [Node.js / Express Backend]
                     - Digital Signature Verification
                     - Device Identity Authentication
                                    │ Verified Event Forwarding
                                    ▼
                   [FastAPI Service: tamper_alarm.py]
                   - Sequence Monotonicity & Deduplication
                   - Real-Time Device Status State Machine
                   - Cryptographic SHA-256 Chained Audit Ledger
                   - Immediate Regulator Alert Generation
                                    │
                                    ▼
                      [Regulator Security Dashboard]
```

#### Physical Hardware Requirements for Deployment
1. **Tamper-Evident Enclosure:** Ruggedized IP66/NEMA 4X housing fitted with normally-closed (NC) microswitches or optical interrupters that trigger immediately when the lid is unlatched.
2. **Electrical Tamper Seals:** Numbered wire seals woven through access bolts, terminating in an active low-current supervisory loop. Severing or unthreading the wire breaks circuit continuity.
3. **NAMUR NE43 Analog Fault Detection:** For 4–20 mA industrial probes (pH, COD, DO, turbidity):
   - Normal operating signal: $4.0 \text{ mA} \le I \le 20.0 \text{ mA}$.
   - Probe open-circuit or disconnected wire: $I < 3.6 \text{ mA}$ (typically $0.0 \text{ mA}$).
4. **Mechanical Mounting Interlocks:** Hall-effect or inductive proximity sensors embedded in the flow channel mount to detect physical withdrawal of the sensor tube.
5. **Cryptographic Secure Element (e.g. Microchip ATECC608A):** Stores the device private key in hardware. Every event is signed before transmission, ensuring tamper events cannot be forged or replayed over the wire.
6. **Battery-Backed Real-Time Clock (RTC):** Ensures chronological integrity even during power outages, preventing timestamp manipulation.

### 3. State Transitions & Historical Ledger Preservation

* **Active Tamper Transition:** When an active event arrives (`event_state="ACTIVE"`), the device tamper status transitions to `TAMPER_ACTIVE` (or `OFFLINE_WARNING`), and a high-priority `TamperAlert` is published.
* **Resolution Transition (`CLEARED`):** When authorized inspectors service and reseal the unit, a `CLEARED` event is transmitted.
  - The alert's status is updated to `"CLEARED"` and `cleared_at_utc` is recorded.
  - Device status re-evaluates to `NOMINAL` (if no other alerts remain active).
  - **CRUCIAL INTEGRITY GUARANTEE:** The historical alert and its originating event are **never deleted**. The complete incident history remains permanently queryable for regulatory compliance and audit inspections.
* **Deduplication:** Events with duplicate `event_id` or duplicate `(device_id, event_type, sequence_number)` are recognized, logged as `DUPLICATE_IGNORED` in the audit ledger, and discarded without generating redundant alerts.

### 4. Tamper-Evident Cryptographic Audit Ledger

Every physical security transaction is appended to a cryptographically chained audit ledger:
$$\text{Entry Hash}_k = \text{SHA-256}\Big(\text{Entry Hash}_{k-1} \;\|\; \text{Canonical JSON}(\text{Entry Payload}_k)\Big)$$

- The ledger begins with a known genesis hash (`0000000000000000000000000000000000000000000000000000000000000000`).
- Any attempt to retroactively modify, delete, or insert an audit entry causes subsequent hash chain verification (`GET /tamper/audit-log` or `verify_audit_log_integrity()`) to fail with a cryptographic mismatch error.

### 5. Implementation Status: Software Demonstration vs. Field Hardware

| Capability | Current Software Implementation | Field Hardware Required for Deployment |
| :--- | :--- | :--- |
| **Tamper Ingestion API** | Fully implemented (`POST /tamper/events`) | Verified HTTPS gateway from IoT field controller |
| **Event Validation** | Pydantic strict schema (sequence $\ge 0$, skew $< 5$ min) | Hardware RTC & NTP time synchronization |
| **Deduplication** | In-memory thread-safe key tracking | Shared Redis or distributed database for multi-node |
| **Audit Ledger** | SHA-256 cryptographic hash-chained ledger | Blockchain or append-only write-once cloud storage |
| **Demonstration Simulation** | 6 scenarios via `POST /demo/tamper/simulate` | No hardware required (pure software demonstration) |
| **Physical Detection** | Simulated via API & tests (stamped `SIMULATED`) | Physical microswitches, break-wire loops, 4-20mA ADCs |
| **Digital Signatures** | Architectural boundary documented | Microchip ATECC608A / Ed25519 signing coprocessor |

---

## Physical Tamper REST API Endpoints

### 1. Ingest Tamper Event (`POST /tamper/events`)
* **Headers:**
  * `Content-Type: application/json`
  * `X-ML-Service-Token: <token>` (if authentication configured)
* **Request JSON:**
```json
{
  "event_id": "EVT-TAMPER-2026-001",
  "device_id": "DEV-FAC-ONT-01",
  "sensor_id": "PROBE-COD-EFFLUENT-A",
  "event_type": "ENCLOSURE_OPEN",
  "event_timestamp": "2026-11-01T02:30:00Z",
  "sequence_number": 104,
  "event_state": "ACTIVE",
  "source": "HARDWARE",
  "metadata": {
    "location": "Basin 3 Effluent Weir",
    "switch_pin": "GPIO_17",
    "contact_state": "OPEN"
  }
}
```
* **Response (HTTP 200 OK):**
```json
{
  "status": "success",
  "message": "Physical tamper alert successfully generated and logged.",
  "is_new_alert": true,
  "event_id": "EVT-TAMPER-2026-001",
  "device_id": "DEV-FAC-ONT-01",
  "event_type": "ENCLOSURE_OPEN",
  "event_state": "ACTIVE",
  "alert": {
    "alert_id": "ALT-DEV-FAC-ONT-01-ENCLOSURE_OPEN-8A1C3E",
    "device_id": "DEV-FAC-ONT-01",
    "sensor_id": "PROBE-COD-EFFLUENT-A",
    "originating_event_id": "EVT-TAMPER-2026-001",
    "event_type": "ENCLOSURE_OPEN",
    "severity": "CRITICAL",
    "category": "PHYSICAL_TAMPER",
    "is_confirmed_tamper": true,
    "status": "ACTIVE",
    "summary": "Physical Tamper Alert: Protective sensor enclosure door or casing opened. on device DEV-FAC-ONT-01",
    "details": "Security Notice: Hardware switch or tamper circuit indicated potential physical interference. Notice to Regulator: Inspect unit mounting and physical seals. (Note: This alert denotes physical sensor interference, not automated fraud or discharge violation).",
    "created_at_utc": "2026-11-01T02:30:01.124560+00:00",
    "cleared_at_utc": null,
    "source": "HARDWARE",
    "is_simulated": false,
    "audit_hash": "3f98c4...dae1"
  }
}
```

### 2. Device Security Status (`GET /tamper/devices/{device_id}`)
* **Response (HTTP 200 OK):**
```json
{
  "device_id": "DEV-FAC-ONT-01",
  "tamper_status": "TAMPER_ACTIVE",
  "active_alerts_count": 1,
  "active_event_types": ["ENCLOSURE_OPEN"],
  "last_event_timestamp": "2026-11-01T02:30:00+00:00",
  "last_heartbeat_timestamp": "2026-11-01T02:30:00+00:00",
  "highest_sequence_seen": 104,
  "total_incidents_recorded": 1,
  "is_simulated": false,
  "updated_at_utc": "2026-11-01T02:30:01.124560+00:00"
}
```

### 3. Regulator Security Alerts Feed (`GET /tamper/alerts`)
* **Query Parameters:** `status` (`ACTIVE`/`CLEARED`), `severity` (`CRITICAL`/`HIGH`/`MEDIUM`), `device_id`, `source` (`HARDWARE`/`SIMULATED`).
* **Response (HTTP 200 OK):** Array of `TamperAlert` objects sorted newest first. Historical records are preserved.

### 4. Cryptographic Audit Log Query (`GET /tamper/audit-log`)
* **Query Parameters:** `device_id`, `limit` (default 100).
* **Response (HTTP 200 OK):**
```json
{
  "is_chain_valid": true,
  "integrity_error": null,
  "total_returned": 2,
  "entries": [
    {
      "entry_id": "AUDIT-3C89B1A041E2",
      "timestamp_utc": "2026-11-01T02:30:01.124560+00:00",
      "action": "ALERT_GENERATED",
      "event_id": "EVT-TAMPER-2026-001",
      "device_id": "DEV-FAC-ONT-01",
      "event_type": "ENCLOSURE_OPEN",
      "event_state": "ACTIVE",
      "source": "HARDWARE",
      "details": "Generated CRITICAL alert ALT-DEV-FAC-ONT-01-ENCLOSURE_OPEN-8A1C3E...",
      "prev_hash": "d820c7...b109",
      "entry_hash": "3f98c4...dae1"
    }
  ]
}
```

### 5. Simulate Tamper Scenario (`POST /demo/tamper/simulate`)
* **Request JSON:**
```json
{
  "scenario": "enclosure_open",
  "device_id": "DEV-SIM-001",
  "sensor_id": "PROBE-COD-01"
}
```
Supported scenarios: `"enclosure_open"`, `"seal_broken"`, `"sensor_disconnected"`, `"sensor_removed"`, `"device_offline"`, `"clear"`. All generated events and alerts are clearly stamped `source: "SIMULATED"` and `is_simulated: true`.

---

## Edge Gateway Offline Buffering & Automatic Retransmission (Store-and-Forward)

> [!IMPORTANT]
> **Production Architecture & Boundary Definition:**
> ```
> ┌────────────────────────┐
> │    Physical Sensors    │ (pH, COD, DO, Turbidity probes, Tamper switches)
> └───────────┬────────────┘
>             ▼
> ┌────────────────────────┐
> │    IoT Edge Gateway    │ (Edge hardware running store_and_forward.py)
> │  - Durable SQLite DB   │ - ACID local persistence (transactions commit before ack)
> │  - Separate Sequences  │ - Isolated counters for TELEMETRY and SECURITY
> │  - Priority & Backoff  │ - Tamper events prioritized; anti-starvation for telemetry
> └───────────┬────────────┘
>             │ HTTPS / TLS Mutual Auth (Retries with Exponential Backoff + Jitter)
>             ▼
> ┌────────────────────────┐
> │ Node.js / Express      │ (Central Cloud / Enterprise Backend)
> │ Backend Gateway        │ - Hardware digital signature verification (Ed25519)
> │                        │ - Durable server-side ledger & deduplication
> └───────────┬────────────┘
>             │ Internal Authenticated Dispatch (X-ML-Service-Token)
>             ▼
> ┌────────────────────────┐
> │ FastAPI ML Microservice│
> │  - Anomaly Predictor   │ - Process consistency cross-checking (Isolation Forest)
> │  - Tamper Alarm Manager│ - Real-time state machine & SHA-256 audit ledger
> └───────────┬────────────┘
>             ▼
> ┌────────────────────────┐
> │  Regulator Dashboard   │ (Real-time security alert feeds & compliance audit logs)
> └────────────────────────┘
> ```
> - **Authority Boundary:** The Python ML service is **not** the authority for physical-device identities or private keys. Device identity authentication and digital signature verification reside in the IoT gateway and Node.js backend.
> - **Durability Principle:** The edge gateway **never** considers a message enqueued until the SQLite transaction commits to disk.
> - **Acknowledgement Principle:** The gateway **never** deletes or marks a message delivered simply because an HTTP connection succeeded; it requires an explicit, structured `ServerAckResponse` with a cryptographic receipt token.

### 1. SQLite Persistent Queue Schema

The edge gateway persists messages in SQLite (default: `edge_gateway_queue.db`) using three relational tables:

```sql
-- 1. Main message queue
CREATE TABLE IF NOT EXISTS edge_messages (
    message_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    message_category TEXT NOT NULL,       -- 'SECURITY' or 'TELEMETRY'
    message_type TEXT NOT NULL,           -- 'ENCLOSURE_OPEN', 'COD_READING', etc.
    event_timestamp TEXT NOT NULL,        -- ISO 8601 UTC timestamp
    sequence_number INTEGER NOT NULL,     -- Monotonic sequence counter
    priority INTEGER NOT NULL,            -- 1=Tamper, 2=Health, 3=Telemetry
    payload_json TEXT NOT NULL,           -- Canonical JSON payload
    payload_checksum TEXT NOT NULL,       -- SHA-256 hash for disk integrity check
    delivery_status TEXT NOT NULL,        -- 'PENDING', 'IN_FLIGHT', 'ACKNOWLEDGED', 'FAILED'
    retry_count INTEGER DEFAULT 0,
    last_transmission_attempt TEXT,
    server_ack_timestamp TEXT,
    server_ack_receipt TEXT,
    created_at TEXT NOT NULL
);

-- Indexes for priority dispatch and sequential querying
CREATE INDEX idx_status_priority_seq ON edge_messages (delivery_status, priority, sequence_number);
CREATE INDEX idx_device_cat_seq ON edge_messages (device_id, message_category, sequence_number);

-- 2. Monotonic sequence counters (isolated per device and category)
CREATE TABLE IF NOT EXISTS device_sequences (
    device_id TEXT NOT NULL,
    message_category TEXT NOT NULL,       -- 'SECURITY' or 'TELEMETRY'
    last_sequence INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (device_id, message_category)
);

-- 3. Queue audit log for storage warnings, evictions, and disk corruption
CREATE TABLE IF NOT EXISTS queue_audit_log (
    log_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    action TEXT NOT NULL,
    message_id TEXT,
    device_id TEXT,
    details TEXT
);
```

### 2. Monotonic Sequence Isolation

To ensure that priority transmission does not introduce false sequence gaps or confuse regulatory audit verification:
- Telemetry messages increment a dedicated `TELEMETRY` sequence counter per device.
- Security events increment an independent `SECURITY` sequence counter per device.
- Both counters persist in `device_sequences`, ensuring sequence continuity across gateway restarts.

### 3. Priority Ordering & Anti-Starvation Mechanism

When connectivity is available, the gateway dequeues messages according to strict priority:

| Priority Level | Category | Event Types | Dispatch Rule |
| :---: | :--- | :--- | :--- |
| **1** | `SECURITY` | `ENCLOSURE_OPEN`, `SEAL_BROKEN`, `SENSOR_REMOVED`, `SENSOR_DISCONNECTED` | Dispatched first upon reconnection. |
| **2** | `SECURITY` | `DEVICE_OFFLINE`, watchdog warnings, battery health | Dispatched after physical tamper alerts. |
| **3** | `TELEMETRY` | Routine environmental measurements (COD, BOD, pH, TDS, flow) | Dispatched chronologically following security events. |

#### Anti-Starvation Policy
If an industrial facility experiences continuous security events (e.g. repeated sensor disconnects), routine telemetry could be starved indefinitely.
The gateway implements an **anti-starvation yield counter** (`max_consecutive_security = 5`):
- After transmitting 5 consecutive security messages, if pending telemetry is waiting in the queue, the gateway forces the oldest pending telemetry message into the next transmission slot before resuming security dispatch.
- This guarantees bounded latency for telemetry while maintaining high priority for security alerts.

### 4. Automatic Retry with Exponential Backoff & Jitter

When network connectivity is lost:
1. The gateway continues collecting, checksumming, and committing readings to SQLite.
2. Unacknowledged messages remain in `PENDING` (or `FAILED`) status; **none are deleted**.
3. Retries use exponential backoff with full randomized jitter to prevent thundering-herd synchronizations:
   $$t_{\text{wait}} = \min(t_{\text{max}}, t_{\text{base}} \times 2^{\text{retry\_count}}) \times \Big(1 \pm \text{jitter\_pct} \times \text{random}(-1, 1)\Big)$$
   - $t_{\text{base}} = 1.0 \text{ s}$, $t_{\text{max}} = 60.0 \text{ s}$, $\text{jitter\_pct} = 25\%$.
4. Upon network restoration, the gateway flushes buffered messages in priority order until the queue is clear.

### 5. Idempotent Ingestion & Lost Acknowledgement Handling

A major risk in industrial IoT networks is **lost acknowledgements** (the server successfully processes a message, but the connection drops before the ACK reaches the gateway).
1. When the gateway reconnects, it retransmits the unacknowledged message.
2. The server checks its durable receipt registry.
3. If the `message_id` has already been recorded:
   - The server **does not** trigger a second tamper alarm or duplicate ML anomaly score.
   - It returns `status: "DUPLICATE_ACKNOWLEDGED"` along with the original cryptographic receipt token.
4. The gateway records the server acknowledgement and updates message status to `ACKNOWLEDGED`.

### 6. Storage Capacity Monitoring & Retention Policies

Edge gateways have finite local flash/eMMC storage. The queue actively monitors storage utilization against `max_capacity_messages`:

```
0% ────────────────────────── 75% ────────────────── 90% ────────────── 100%
      [NOMINAL STATUS]             [WARNING_75]        [RESERVED FOR]    [STORAGE_EXHAUSTED]
                                                      [SECURITY ONLY]
```

- **75% Warning (`WARNING_75`):** Gateway logs warning and surfaces warning flag in health probes.
- **90% Critical Threshold (`CRITICAL_90_SECURITY_RESERVED`):**
  - The top 10% of queue capacity is **strictly reserved for Security events**.
  - New telemetry is handled according to configurable policy:
    * `DROP_OLDEST_TELEMETRY` *(Default)*: Safely drops the oldest pending telemetry record to free space, recording an immutable audit entry (`TELEMETRY_DROPPED_DUE_TO_STORAGE_LIMIT`).
    * `REJECT_NEW_TELEMETRY`: Refuses new routine telemetry with `StorageExhaustedError`.
  - **Security events are never rejected at this threshold.**
- **100% Storage Exhaustion (`STORAGE_EXHAUSTED`):**
  - If security events fill the entire buffer, the gateway refuses new inputs and alerts local diagnostics.
  - **Security events are NEVER silently discarded.**

### 7. Local Disk Checksum Validation

To protect against flash bit-rot or local file tampering:
- Every message payload is hashed with SHA-256 upon enqueue:
  $$\text{Checksum} = \text{SHA-256}\Big(\text{Canonical JSON}(\text{Payload})\Big)$$
- Whenever a message is loaded from SQLite for transmission, its payload is re-hashed.
- If a mismatch is detected, the gateway raises `CorruptedPayloadError` and writes a security incident to `queue_audit_log`.

---

## Edge Gateway REST API Contracts

### 1. Ingest Message (`POST /gateway/ingest`)
* **Headers:**
  * `Content-Type: application/json`
  * `X-ML-Service-Token: <token>` (if configured)
* **Request JSON:**
```json
{
  "message_id": "MSG-7B9A2F8E1C4D4A90",
  "device_id": "DEV-FAC-ONT-01",
  "message_category": "TELEMETRY",
  "message_type": "COD_READING",
  "event_timestamp": "2026-11-01T03:30:00Z",
  "sequence_number": 142,
  "priority": 3,
  "payload": {
    "COD_mg_L": 480.0,
    "BOD_mg_L": 120.0,
    "pH": 7.2,
    "turbidity_NTU": 140.0
  },
  "payload_checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "delivery_status": "PENDING",
  "retry_count": 0,
  "created_at": "2026-11-01T03:30:00.124560Z"
}
```
* **Response (HTTP 200 OK — First Transmission):**
```json
{
  "message_id": "MSG-7B9A2F8E1C4D4A90",
  "status": "ACKNOWLEDGED",
  "server_receipt_timestamp": "2026-11-01T03:30:01.054120+00:00",
  "receipt_token": "RCT-4F1A88B39C1D02E4",
  "message_category": "TELEMETRY",
  "details": "Telemetry cross-checked against ML model."
}
```
* **Response (HTTP 200 OK — Idempotent Duplicate Transmission):**
```json
{
  "message_id": "MSG-7B9A2F8E1C4D4A90",
  "status": "DUPLICATE_ACKNOWLEDGED",
  "server_receipt_timestamp": "2026-11-01T03:30:01.054120+00:00",
  "receipt_token": "RCT-4F1A88B39C1D02E4",
  "message_category": "TELEMETRY",
  "details": "Message was previously recorded; duplicate request acknowledged idempotently."
}
```

### 2. Batch Ingest Messages (`POST /gateway/ingest/batch`)
* **Request:** Array of `QueueMessage` objects.
* **Response:** Array of `ServerAckResponse` objects corresponding to each message.

### 3. Gateway Ingestion Status (`GET /gateway/status`)
* **Response (HTTP 200 OK):**
```json
{
  "service": "EnviroTrust Store-and-Forward Ingestion Service",
  "is_online": true,
  "total_recorded_messages": 142,
  "timestamp_utc": "2026-11-01T03:35:00.000000+00:00"
}
```

### 4. Interactive Gateway Simulator (`POST /demo/gateway/simulate`)
* **Query Parameter:** `scenario` (`A`, `B`, `C`, `D`, `E`, `F`, or `ALL`).
* **Scenarios Executed:**
  - **Scenario A:** Normal operation (immediate generation & delivery).
  - **Scenario B:** Internet outage (network disabled, messages accumulate in SQLite).
  - **Scenario C:** Tamper event during outage (sensor enclosure open buffered locally with Priority 1).
  - **Scenario D:** Network restoration (security dispatched first, older telemetry follows, acks verified).
  - **Scenario E:** Duplicate delivery (simulated lost ACK, idempotent backend resolution without duplicate alert).
  - **Scenario F:** Application restart recovery (unacknowledged messages safely restored from SQLite).

---

## Field Hardware Requirements & Deployment Architecture

For full field deployment on physical industrial monitoring units:

| Hardware Component | Specification | Deployment Role |
| :--- | :--- | :--- |
| **Edge Compute SoC** | ARM Cortex-A7/A53 (e.g. Raspberry Pi CM4, NXP i.MX6/8, BeagleBone Industrial) | Runs SQLite edge queue daemon and local telemetry acquisition. |
| **Storage Medium** | Industrial pSLC eMMC or High-Endurance MicroSD (wear-levelled) | Power-loss-protected persistent storage for SQLite WAL mode. |
| **Real-Time Clock** | Battery-backed RTC (DS3231 or PCF8563) with NTP synchronization | Prevents timestamp manipulation or time-drift during power outages. |
| **Hardware Secure Element** | Microchip ATECC608A / OPTIGA™ Trust M | Stores private key in hardware; computes Ed25519 digital signatures. |
| **Network Interface** | Dual-SIM 4G/LTE Cat-M1/NB-IoT modem + 10/100 Industrial Ethernet | Cellular store-and-forward link with automatic failover. |
| **Supercapacitor UPS** | Short-duration power-loss backup (5–10 seconds) | Ensures clean SQLite commit and WAL flush during sudden factory power cuts. |

---

## Known Limitations & Production Roadmap

1. **Backend Separation:** The Node.js/Express backend repository is maintained separately. In this ML microservice workspace, the edge queue interacts with the server ingestion contract (`GatewayIngestionService`) which feeds `tamper_manager` and `predict_telemetry`.
2. **Local Database Encryption:** SQLite WAL mode provides ACID transaction durability. In ultra-high security field environments, hardware-level encryption (e.g. SQLCipher or Linux LUKS block-device encryption) is recommended to prevent direct reading of flash chips.
3. **Simulation vs Physical Hardware:** The edge gateway simulator validates the complete state machine, backoff algorithms, SQLite transactions, and API contracts. Real-world physical hardware integration requires deploying the Python queue or compiled embedded equivalent to the on-site gateway controller.
