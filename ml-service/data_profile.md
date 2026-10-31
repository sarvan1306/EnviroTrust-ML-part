# EnviroTrust — Data Profile & Training Readiness Assessment

**Last Updated:** 2026-10-31  
**Target Component:** EnviroTrust ML Service (`ml-service/`)  
**Core Model:** Scikit-Learn Isolation Forest

---

## 1. Executive Summary & Current Condition Assessment

EnviroTrust's machine learning service has now integrated **two distinct authentic industrial datasets** alongside a dedicated, fully quarantined **Synthetic Demonstration Mode**.

### Current Condition Matrix

| Dimension | Condition / Status | Technical Evaluation |
| :--- | :--- | :--- |
| **Authentic Sample Size** | **$N = 15$ factories** (Bangladesh dataset) + **$N = 2$ raw effluent streams** (literature study) | Exceeds software validation minimum ($N \ge 10$), allowing unsupervised Isolation Forest to fit without runtime rejection. However, $N=15$ remains statistically small for high-dimensional tree depth without risk of small-sample variance. |
| **Process-Context Telemetry (Configuration B)** | **Fully Available (Factory-Level)** | Production (kg), Electricity (kWh), Groundwater Extraction (ML), and Discharged Wastewater (ML) are 100% complete for all 15 factories. |
| **Effluent Monitoring (Outfall vs Influent)** | **Treated Discharge Outfall (Authentic)** | Bangladesh dataset measures treated discharge effluent (COD: 28–216 mg/L, BOD: 4–44 mg/L), aligning directly with EnviroTrust's compliance and tampering detection outfall sensor mandate. |
| **Zero-Variance / Regulatory Columns** | **Identified & Safely Excluded** | `TDS (mg/L)` (`2000-3000*`) and `pH Range` (`6.0 - 8.5*`) are static regulatory threshold strings, not sensor telemetry. Correctly excluded from Isolation Forest feature matrices. |
| **Temporal Telemetry & Timestamps** | **Absent (Cross-Sectional Only)** | The 15 records represent annual/periodic aggregate factory operations, not continuous time-series. Splitting strategy must be `random`; `chronological` splitting correctly rejects execution due to missing timestamps. |
| **Ground-Truth Anomaly Labels** | **Absent (Unsupervised Only)** | No pre-labelled tampering incidents exist. Isolation Forest operates in pure unsupervised mode, flagging statistical outliers (F12 and F10). |
| **Synthetic Demonstration Pipeline** | **Operational & Quarantined** | Provides 1,000 baseline time-series records and 200 labelled tampering attack scenarios (dilution, flatlining, sensor drift) for frontend integration without contaminating authentic research data. |
| **Test Suite Health** | **93 Passing Tests (100%)** | Full end-to-end coverage across data loading, schema validation, feature engineering, preprocessing, anomaly detection, consistency rules, API endpoints, and authentic data immutability. |

---

## 2. Authentic Research Datasets Catalog

All raw authentic files are stored under `ml-service/data/raw/` and protected against modification by automated cryptographic immutability tests.

```
ml-service/data/raw/
├── bangladesh_textile_dataset.csv                       # SHA-256: 188c7ae123494380ddc2a019506f0fdbef8635f6d811aebcebf44ff9a132d09a
├── bangladesh_textile_industry_anomaly_detection.ipynb  # SHA-256: bc69d8dd7a5c65c5e5b61f2ee0c07a5ab1cc80ff09f3b0fbceab4cce509385c8
├── data_spreadsheet_-_Sheet1.csv                        # SHA-256: d2df8ee865dd05476abf23b5acdce8a029d758c3867272181fe51ee4eca521a6
├── extracted_water_quality_samples.csv                  # SHA-256: cf0d823b990579d09b491be00789c0c344627e80b3d10e7bd522fdc4bef72532
├── extracted_water_quality_samples_v2.csv               # SHA-256: 069a5fcb5a0977dd53e23468e21d98a3e8eef9a8255edd2f6d1d6393b8a354e9
└── industrial_water_pollution_ml.ipynb                  # SHA-256: e199dc8602ad319efaa0394e197de0da7b74613cb8731cfdc6c9dd23fc6b3f71
```

---

## 3. Bangladesh Textile Industry Dataset Profiling

* **File:** `data/raw/bangladesh_textile_dataset.csv`
* **Format:** Comma-Separated Values (CSV)
* **Shape:** 15 rows $\times$ 9 columns
* **Target Industry:** Export textile manufacturing and wet-processing dyeing mills in Bangladesh.
* **Scope:** Factory-level operational throughput, resource extraction, energy consumption, and effluent discharge concentrations.

### Column Inventory & Data Typing

| Column Name | Inferred Dtype | Non-Null Count | Missing Count | Unique Values | Role in EnviroTrust ML |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Factory` | `string` / `object` | 15 | 0 | 15 (`F1` – `F15`) | **Identifier** (Excluded from feature matrix) |
| `Total Production (kg)` | `int64` | 15 | 0 | 15 | **Active Process Feature** (`production`) |
| `Total Electricity Use (kWh)` | `int64` | 15 | 0 | 15 | **Active Process Feature** (`electricity_consumption`) |
| `Extracted Groundwater (Million Liters)` | `float64` | 15 | 0 | 15 | **Active Process Feature** (`extracted_groundwater`) |
| `Discharged Wastewater (Million Liters)` | `float64` | 15 | 0 | 15 | **Active Process Feature** (`water_flow`) |
| `COD (mg/L)` | `int64` | 15 | 0 | 15 | **Active Environmental Feature** (`COD_mg_L`) |
| `BOD (mg/L)` | `int64` | 15 | 0 | 14 | **Active Environmental Feature** (`BOD_mg_L`) |
| `TDS (mg/L)` | `string` / `object` | 15 | 0 | 1 (`2000-3000*`) | **Excluded** (Zero-variance regulatory standard text) |
| `pH Range` | `string` / `object` | 15 | 0 | 1 (`6.0 - 8.5*`) | **Excluded** (Zero-variance regulatory standard text) |

### Numerical Distributions & Descriptive Statistics ($N = 15$)

| Feature (Canonical Name) | Unit | Mean | Std Dev | Min | Median (50%) | Max | IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `production` | kg | $7.60 \times 10^6$ | $9.80 \times 10^6$ | $2.41 \times 10^6$ | $4.33 \times 10^6$ | $41.53 \times 10^6$ | $3.55 \times 10^6$ |
| `electricity_consumption` | kWh | $1.77 \times 10^7$ | $1.87 \times 10^7$ | $4.19 \times 10^6$ | $9.70 \times 10^6$ | $6.55 \times 10^7$ | $1.21 \times 10^7$ |
| `extracted_groundwater` | Million Liters | $961.26$ | $1049.74$ | $356.67$ | $652.66$ | $4526.32$ | $567.45$ |
| `water_flow` (Discharged) | Million Liters | $640.24$ | $643.37$ | $260.12$ | $372.53$ | $2815.04$ | $413.62$ |
| `COD_mg_L` | mg/L | $87.87$ | $53.53$ | $28.00$ | $72.00$ | $216.00$ | $40.00$ |
| `BOD_mg_L` | mg/L | $21.80$ | $12.27$ | $4.00$ | $20.00$ | $44.00$ | $15.50$ |

---

## 4. Anomaly Analysis & Factory Isolation

When the EnviroTrust Isolation Forest model is fitted to the 15 authentic factory profiles using standardized process-context features:

```
Scores (Higher = More Anomalous, Offset at Decision Boundary 0.0):
Factory F12 : +0.1981  --> Flagged Outlier (Extreme Scale & Unusually Low Effluent COD/BOD)
Factory F10 : +0.0251  --> Flagged Outlier (Extreme Electricity Intensity)
Factory F7  : -0.0055  --> Inlier (Near Boundary)
Factory F3  : -0.0504  --> Inlier (High COD 216 mg/L, but consistent scale)
...
Factory F4  : -0.1546  --> Nominal Cluster Core
```

### Physical Interpretations of Flagged Factories

1. **Factory F12 (Primary Anomaly, Score $+0.1981$):**
   - **Production:** $41,530,362$ kg ($>5\times$ sample mean).
   - **Groundwater Extraction:** $4,526.3$ ML ($4.7\times$ sample mean).
   - **Discharge Volume:** $2,815.0$ ML ($4.4\times$ sample mean).
   - **Discharge COD & BOD:** $28$ mg/L and $4$ mg/L (lowest in the entire dataset).
   - *Physical Implication:* In industrial wastewater monitoring, an enterprise with five times the typical production volume reporting near-drinking-water effluent COD ($28$ mg/L vs average $88$ mg/L) presents a classic tampering / dilution signature, or indicates a highly capitalized multi-stage modern ETP requiring on-site audit verification.

2. **Factory F10 (Secondary Anomaly, Score $+0.0251$):**
   - **Production:** $10,702,135$ kg.
   - **Electricity Consumption:** $65,546,454$ kWh ($3.7\times$ sample mean, absolute highest).
   - **Energy Intensity:** $6.12$ kWh/kg production (vs sample average $\approx 2.3$ kWh/kg).
   - *Physical Implication:* Significant energy consumption decoupling relative to output, indicating heavy on-site wastewater aeration/treatment equipment or off-spec plant machinery.

---

## 5. Dual-Dataset Architecture & Contrast

| Property | Dataset 1: Laboratory Analytical Study | Dataset 2: Bangladesh Textile Industrial Survey |
| :--- | :--- | :--- |
| **Source Files** | `extracted_water_quality_samples_v2.csv` | `bangladesh_textile_dataset.csv` |
| **Sample Size** | $N = 2$ | $N = 15$ |
| **Stream Monitored** | **Raw untreated influent** (dye bath) | **Treated discharge effluent** (ETP outfall) |
| **Typical COD Range** | $4,800 - 8,800$ mg/L | $28 - 216$ mg/L |
| **Typical BOD Range** | Unmeasured (`NaN`) | $4 - 44$ mg/L |
| **Operational Telemetry** | Completely Absent | Complete (Production, Power, Water, Discharge) |
| **EnviroTrust Config** | **Configuration A** (Environmental-Only) | **Configuration B** (Process-Context Cross-Checking) |
| **Splitting Support** | None ($N=2 < \text{min\_rows}$) | Random Split Supported ($N=15 \ge \text{min\_rows}$) |
| **Timestamps** | Absent | Absent |

---

## 6. Software Architecture & Ingestion Capabilities

To cleanly ingest both datasets without code regressions or data corruption, `ml-service/` incorporates:

1. **Idempotent Bidirectional Feature Mapping:**
   - Handles both raw column names (e.g. `"Total Production (kg)"` $\to$ `"production"`) and already-canonical features.
   - Prevents recursive or destructive renaming during repeated feature engineering steps.
2. **Dynamic Range Filtering:**
   - Static string range columns (e.g. `"2000-3000*"`, `"6.0 - 8.5*"`) are parsed during profiling and safely excluded from numeric feature matrices to prevent pipeline failure.
3. **Splitting Guardrails:**
   - Datasets without ISO 8601 timestamps are rejected under `strategy: chronological`, enforcing `strategy: random` to avoid erroneous temporal ordering.
4. **Separation of Authentic and Synthetic Data:**
   - Model weights trained on synthetic attack scenarios are strictly saved under `models/demo_isolation_forest.joblib`.
   - Authentic evaluation occurs strictly on untainted raw data.

---

## 7. Actionable Next Steps & Data Collection Recommendations

To move EnviroTrust to a fully certified production deployment:

1. **Obtain High-Frequency Time-Series Telemetry:**
   - Transition from cross-sectional annual snapshots ($N=15$) to hourly or 15-minute SCADA intervals ($N \ge 1,000$ timestamps).
   - This enables continuous EWMA drift monitoring, flatline detection, and chronological train/validation splitting.
2. **Collect Numeric Sensor Telemetry for pH and TDS:**
   - Replace regulatory specification text (`6.0 - 8.5*`, `2000-3000*`) with continuous streaming probe measurements.
3. **Benchmark Flagged Outliers (F12, F10):**
   - Use the unsupervised Isolation Forest scores to prioritize physical facility inspections and verify whether low COD at F12 is authentic advanced ETP efficiency or sensor dilution.
