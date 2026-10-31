# EnviroTrust — Data Profile & Training Readiness Assessment

**Source Files:**
1. `ml-service/data/raw/data_spreadsheet_-_Sheet1.csv` (Literature Analytical Parameter Table)
   - **SHA-256:** `d2df8ee865dd05476abf23b5acdce8a029d758c3867272181fe51ee4eca521a6`
2. `ml-service/data/raw/extracted_water_quality_samples.csv` (Initial Starter Samples v1)
   - **SHA-256:** `cf0d823b990579d09b491be00789c0c344627e80b3d10e7bd522fdc4bef72532`
3. `ml-service/data/raw/extracted_water_quality_samples_v2.csv` (Integrated Observation Dataset v2)
   - **SHA-256:** `8634cd0080a8177710839e36109296d5ff3a63d660d31cd50c6884a59827be66`
4. `ml-service/data/raw/industrial_water_pollution_ml.ipynb` (Companion Starter Notebook)

**Inspection Date:** 2026-10-31

---

## 1. Executive Decision: Training Readiness

> [!CAUTION]
> **FINAL DECISION: NOT READY FOR MODEL TRAINING**
>
> While the newly provided literature spreadsheet resolves previous missing values for Sample 2 and contributes a new parameter (`sulphate_mg_L`), the dataset remains fundamentally restricted to **$N = 2$ observations**.
>
> In accordance with EnviroTrust's strict engineering and trust guidelines:
> - **Isolation Forest has NOT been fitted.** Fitting Isolation Forest on 2 samples is statistically invalid; tree partitioning cannot isolate multidimensional anomalies or learn continuous density boundaries on two points.
> - **No synthetic samples or values have been fabricated.** Unmeasured parameters (`BOD_mg_L`, `electricity_consumption`, `production`, `water_flow`) are strictly preserved as `NaN`.
> - **Validation guards reject training execution** (`DataValidationError: Dataset has 2 rows, which is fewer than required minimum (10)`), preventing artifact emission.

---

## 2. Ingestion of Further Data (`data_spreadsheet_-_Sheet1.csv`)

The newly supplied spreadsheet provides a structured laboratory parameter comparison table between two authentic industrial effluent streams:
* **Sample 1:** Textile Industrial Effluent (`Raw_Dye_Wastewater_1`)
* **Sample 2:** Dye-Intermediate Effluent (`Raw_Dye_Wastewater_2`)

### Parameter Comparison & Laboratory Testing Methods

| Parameter | Textile Effluent (Sample 1) | Dye-Intermediate (Sample 2) | % Difference (Dye vs. Textile) | Testing Method | Canonical Feature Name |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **pH** | 6.0 | 7.4 | +23.3% | Potentiometry | `pH` |
| **Turbidity (NTU)** | 554.0 | 188.0 | -66.10% | Nephelometric | `turbidity_NTU` |
| **Colour (Pt. Co.)** | 3462.0 | 1090.0 | -68.50% | Platinum-Cobalt | `colour_PtCo` |
| **Total Solids (mg/l)** | 3042.0 | 3396.0 | +11.6% | Gravimetric | `total_solids_mg_L` |
| **Total Suspended Solids (mg/l)** | 632.0 | 146.0 | -76.90% | Gravimetric | `total_suspended_solids_mg_L` |
| **Chemical Oxygen Demand (COD) (mg/l)** | 4800.0 | 8800.0 | +83.3% | Open Reflux | `COD_mg_L` |
| **Biochemical Oxygen Demand (BOD)** | N/A | N/A | N/A | *Not Measured in Study* | `BOD_mg_L` (Unmeasured) |
| **Total Dissolved Solids (TDS) (mg/l)** | 2410.0 | 3250.0 | +34.9% | Gravimetric | `TDS_mg_L` |
| **Chloride (mg/l)** | 299.91 | 499.85 | +66.7% | Argentometric | `chloride_mg_L` |
| **Ammoniacal Nitrogen (mg/l)** | 8.44 | 34.0 | +302.8% | Kjeldahl | `ammoniacal_nitrogen_mg_L` |
| **Nitrate (mg/l)** | 101.8 | 608.8 | +498.0% | Phenol Disulphonic Acid (PDA) | `nitrate_mg_L` |
| **Sulphate (mg/l)** | 295.0 | 608.0 | +106.1% | Turbidimetric | `sulphate_mg_L` (New Feature) |
| **Electricity Consumption** | N/A | N/A | N/A | *Raw Effluent Study (No Treatment Power Data)* | `electricity_consumption` (Unmeasured) |

---

## 3. Key Analytical Findings & Progress

### A. Previously Missing Parameters Completed for Sample 2
In the initial extraction (`v1`), Sample 2 had missing values for three key environmental parameters. The new spreadsheet provides authentic laboratory measurements for all three:
1. **Total Dissolved Solids (`TDS_mg_L`):** **3,250.0 mg/L** (previously `NaN`; +34.9% vs. Textile).
2. **Ammoniacal Nitrogen (`ammoniacal_nitrogen_mg_L`):** **34.0 mg/L** (previously `NaN`; +302.8% vs. Textile).
3. **Nitrate (`nitrate_mg_L`):** **608.8 mg/L** (previously `NaN`; +498.0% vs. Textile).

### B. New Environmental Parameter Added
* **Sulphate (`sulphate_mg_L`):** Added for both streams (Textile: 295.0 mg/L, Dye-Intermediate: 608.0 mg/L, +106.1% difference). Testing method: Turbidimetric.
* Added to canonical configuration in `config/data_config.yaml` (`column_mapping` and `features.numerical`).

### C. Confirmed Negative Findings (Explicit Absence)
The testing method column provides explicit literature justification for unmeasured parameters:
* **BOD:** Documented as *"Not Measured in Study"*. Confirms 100% missingness without ambiguity.
* **Electricity Consumption:** Documented as *"Raw Effluent Study (No Treatment Power Data)"*. Confirms that the source study focuses exclusively on raw untreated effluent characteristics, without power monitoring for treatment processes.

---

## 4. Feature Profile & Classification Matrix (Integrated Dataset v2)

| Column Name | Inferred Dtype | Non-Null Count | Missing Count | Missing % | Values Range | Role in EnviroTrust ML |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sample_id` | `object` | 2 | 0 | 0.0% | `Raw_Dye_Wastewater_1`, `Raw_Dye_Wastewater_2` | **Identifier** (Excluded from ML) |
| `pH` | `float64` | 2 | 0 | 0.0% | 6.00 – 7.40 | **Active Environmental Feature** |
| `turbidity_NTU` | `float64` | 2 | 0 | 0.0% | 188.00 – 554.00 | **Active Environmental Feature** |
| `colour_PtCo` | `float64` | 2 | 0 | 0.0% | 1090.00 – 3462.00 | **Active Environmental Feature** |
| `total_solids_mg_L` | `float64` | 2 | 0 | 0.0% | 3042.00 – 3396.00 | **Active Environmental Feature** |
| `total_suspended_solids_mg_L` | `float64` | 2 | 0 | 0.0% | 146.00 – 632.00 | **Active Environmental Feature** |
| `COD_mg_L` | `float64` | 2 | 0 | 0.0% | 4800.00 – 8800.00 | **Active Environmental Feature** |
| `chloride_mg_L` | `float64` | 2 | 0 | 0.0% | 299.91 – 499.85 | **Active Environmental Feature** |
| `ammoniacal_nitrogen_mg_L` | `float64` | 2 | 0 | **0.0%** *(was 50%)* | 8.44 – 34.00 | **Active Environmental Feature** |
| `nitrate_mg_L` | `float64` | 2 | 0 | **0.0%** *(was 50%)* | 101.80 – 608.80 | **Active Environmental Feature** |
| `TDS_mg_L` | `float64` | 2 | 0 | **0.0%** *(was 50%)* | 2410.00 – 3250.00 | **Active Environmental Feature** |
| `sulphate_mg_L` | `float64` | 2 | 0 | **0.0%** *(new)* | 295.00 – 608.00 | **Active Environmental Feature** |
| `BOD_mg_L` | `float64` | 0 | 2 | 100.0% | NaN | **Excluded** (100% Empty) |
| `electricity_consumption` | `float64` | 0 | 2 | 100.0% | NaN | **Excluded** (100% Empty) |
| `production` | `float64` | 0 | 2 | 100.0% | NaN | **Excluded** (100% Empty) |
| `water_flow` | `float64` | 0 | 2 | 100.0% | NaN | **Excluded** (100% Empty) |

**Active Environmental Completeness:** **11 out of 11** configured environmental features now possess 100% data completeness across all observations.

---

## 5. Software Architecture & Ingestion Capabilities

To handle literature comparison spreadsheets seamlessly, the following enhancements have been integrated into `ml-service/`:

1. **Native Parameter Transposition (`data_loader.transpose_parameter_table`):**
   - Transposes parameter-oriented tables (parameters as rows, samples as columns) into canonical observation matrices (samples as rows, parameters as columns).
   - Extracts laboratory testing methods into a dedicated metadata dictionary.
   - Non-destructively handles `N/A`, `-`, and missing values, preserving them as `np.nan` without fabrication.
2. **Flexible Configuration (`data_config.yaml`):**
   - Added `sulphate_mg_L` to `column_mapping` and `features.numerical`.
   - Supports configuration passing as either a file path or in-memory dictionary.
3. **Cryptographic Immutability:**
   - Both original raw files (`extracted_water_quality_samples.csv` and `data_spreadsheet_-_Sheet1.csv`) are cryptographically verified and immutable.
4. **Comprehensive Test Suite:**
   - **65 passing tests across 11 test modules** (100% passing).

---

## 6. Remaining Data Gaps Prior to Training Readiness

To transition EnviroTrust from inspection readiness to active model training:

1. **Sample Volume:**
   - A minimum of **200–500+ authentic observations** is required to establish a continuous multidimensional operating manifold for Isolation Forest.
2. **Plant Operational Telemetry (Stream 2):**
   - Production throughput ($\text{units/hr}$ or $\text{kg/hr}$).
   - Wastewater flow rate ($\text{m}^3\text{/hr}$ or $\text{L/hr}$).
   - Treatment dosage ($\text{mg/L}$ coagulant / neutralizer).
   - Real-time electricity consumption ($\text{kWh}$).
3. **Temporal Metadata:**
   - ISO 8601 UTC timestamps for chronological data splitting to prevent temporal data leakage and verify sensor continuity.
4. **Effluent Stream Context:**
   - Clear distinction between **raw untreated influent** and **treated discharge outfall** (EnviroTrust tampering detection monitors the treated outfall).
5. **Baseline vs. Anomaly Labels (for evaluation):**
   - Labeled historical incidents (sensor drift, dilution tampering, bypass dumping) to benchmark detection accuracy.
