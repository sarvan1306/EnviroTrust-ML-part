# EnviroTrust — Data Profile & Training Readiness Assessment

**Last Updated:** 2026-11-01
**Target Component:** EnviroTrust ML Service (`ml-service/`)  
**Core Model:** Scikit-Learn Isolation Forest & Physical Consistency Rules

---

## 1. Executive Summary & Current Condition Assessment

EnviroTrust's machine learning service has now integrated **three distinct authentic industrial datasets** alongside a dedicated, fully quarantined **Synthetic Demonstration Mode**.

### Current Condition Matrix

| Dimension | Condition / Status | Technical Evaluation |
| :--- | :--- | :--- |
| **Authentic Datasets Available** | **3 Distinct Sources** (Textile Raw Influent, Bangladesh Textile Mills, Ontario MISA Electric Power Generation) | Captures raw influent dye chemistry, factory SCADA operational telemetry, and regulatory outfall discharge loads. |
| **Authentic Sample Counts** | **$N = 15$ factories** (Bangladesh) + **$N = 8$ records / 7 facilities** (Ontario MISA) + **$N = 2$ streams** (Literature study) | Bangladesh dataset exceeds the validation threshold ($N = 15 \ge 10$), enabling full training and calibration of Isolation Forest. MISA ($N=8$) and Literature ($N=2$) standalone datasets are correctly held back by `min_rows: 10` guards. |
| **Process-Context Telemetry (Configuration B)** | **Fully Available (Factory-Level)** | Production (kg), Electricity (kWh), Groundwater Extraction (ML), and Discharged Wastewater (ML) are 100% complete across all 15 Bangladesh factories. MISA provides daily hydraulic discharge flow ($1,161 - 3,100 \text{ m}^3\text{/d}$). |
| **Effluent Monitoring (Outfall vs Influent)** | **Treated Discharge Outfall (Authentic)** | Bangladesh measures treated outfall concentrations (COD: $28 - 216$ mg/L, BOD: $4 - 44$ mg/L). MISA measures treated process effluent discharge (BOD: $3.04$ mg/L, $3.53$ kg/d load). Both align directly with EnviroTrust's outfall monitoring mandate. |
| **Mass Load vs. Concentration Conversion** | **Supported via `pivot_eav_effluent_table`** | Long-format EAV regulatory data is dynamically pivoted to facility observations with mass-to-concentration physical derivation ($\text{mg/L} = \text{kg/d} \times 1,000 / \text{m}^3\text{/d}$). |
| **Temporal Telemetry & Timestamps** | **Present in MISA (`2024/01/31`); Absent in Bangladesh / Literature** | Cross-sectional factory aggregations require `strategy: random`. Datasets without timestamps correctly trigger `DataValidationError` when chronological splitting is configured. |
| **Ground-Truth Anomaly Labels** | **Absent in Authentic Data (Unsupervised Only)** | Outliers (`F12` scale/dilution, `F10` electricity intensity) are flagged by unsupervised Isolation Forest and physical rule envelopes. Label-aware training and supervised attack testing are demonstrated via quarantined synthetic mode. |
| **Synthetic Demonstration Pipeline** | **Operational & Quarantined** | Provides 1,000 baseline time-series records and 200 labelled tampering attack scenarios (dilution, flatlining, sensor drift) for frontend integration without contaminating authentic research data. |
| **Test Suite Health** | **101 Passing Tests (100%)** | Full end-to-end coverage across data loading, schema validation, feature engineering, preprocessing, anomaly detection, consistency rules, API endpoints, and authentic data immutability. |

---

## 2. Authentic Research Datasets Catalog

All raw authentic files are stored under `ml-service/data/raw/` and protected against modification by automated cryptographic immutability tests.

```
ml-service/data/raw/
├── bangladesh_textile_dataset.csv                       # SHA-256: daaa99510e68bcd97b2e8529b5d80706ade4c8da5221fb5d07bb8608280cdef3
├── bangladesh_textile_industry_anomaly_detection.ipynb  # SHA-256: bc69d8dd7a5c65c5e5b61f2ee0c07a5ab1cc80ff09f3b0fbceab4cce509385c8
├── data_spreadsheet_-_Sheet1.csv                        # SHA-256: d2df8ee865dd05476abf23b5acdce8a029d758c3867272181fe51ee4eca521a6
├── extracted_water_quality_samples.csv                  # SHA-256: cf0d823b990579d09b491be00789c0c344627e80b3d10e7bd522fdc4bef72532
├── extracted_water_quality_samples_v2.csv               # SHA-256: 069a5fcb5a0977dd53e23468e21d98a3e8eef9a8255edd2f6d1d6393b8a354e9
├── industrial_water_pollution_ml.ipynb                  # SHA-256: e199dc8602ad319efaa0394e197de0da7b74613cb8731cfdc6c9dd23fc6b3f71
├── misa_electric_power_generation_2024.csv              # SHA-256: 525e197c99429bf38c4d56dea09d085bba885cdc6f0ef104c256a97659353bac
├── sheet3.ipynb                                         # SHA-256: ed78dca642701c85dfebd3bfbab4f0406bd8055501e06a14824caa060a6ec5b1
└── Untitled spreadsheet - Sheet3.csv                    # SHA-256: 525e197c99429bf38c4d56dea09d085bba885cdc6f0ef104c256a97659353bac
```

---

## 3. Dataset 3 Profiling: Ontario MISA 2024 Electric Power Generation

* **File:** `data/raw/misa_electric_power_generation_2024.csv` (alias `Untitled spreadsheet - Sheet3.csv`)
* **Companion Notebook:** `data/raw/sheet3.ipynb`
* **Format:** Comma-Separated Values (CSV) in Long / Entity-Attribute-Value (EAV) regulatory format
* **Shape:** 8 rows $\times$ 8 columns
* **Source:** Ontario Ministry of the Environment, Conservation and Parks — Municipal/Industrial Strategy for Abatement (MISA) 2024.
* **Sector:** Electric Power Generation (Nuclear and Thermal Power Stations).
* **Control Point:** `PLANT - PROCESS EFFLUENT` (treated industrial process wastewater outfall).
* **Sample Date:** `2024/01/31`.

### Raw Long-Format Inventory

| Row | Facility (`Works Name`) | Parameter Name | Value | Unit of Measure |
| :---: | :--- | :--- | :---: | :---: |
| 1 | Bruce Power Inc. — Development Services A & B | BIOCHEMICAL OXYGEN DEMAND, 5 DAY, TOTAL DEMAND | 3.5335 | KG/D |
| 2 | Bruce Power Inc. — Development Services A & B | FLOW | 1161.0 | M3/D |
| 3 | Bruce Power Inc. — GS A | ALUMINIUM, UNFILTERED TOTAL | 0.0540 | KG/D |
| 4 | Bruce Power Inc. — GS B | IRON, UNFILTERED TOTAL | 0.1200 | KG/D |
| 5 | Ontario Power Generation Inc. — Atikokan | RESIDUE, PARTICULATE | 5.2100 | KG/D |
| 6 | Ontario Power Generation Inc. — Darlington Nuclear | FLOW | 2400.0 | M3/D |
| 7 | Ontario Power Generation Inc. — Lennox G.S. | SOLVENT EXTRACTABLES | 1.2000 | KG/D |
| 8 | Ontario Power Generation Inc. — Pickering Nuclear | FLOW | 3100.0 | M3/D |

### Pivoted Facility Observation Matrix (`data_loader.pivot_eav_effluent_table`)

When reshaped into facility-level observations across parameters:

| Facility Name | Flow ($\text{m}^3\text{/d}$) | BOD ($\text{kg/d}$) | Particulate ($\text{kg/d}$) | Solvents ($\text{kg/d}$) | Al ($\text{kg/d}$) | Fe ($\text{kg/d}$) | Derived BOD ($\text{mg/L}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bruce Power Inc. — Dev. Services** | **1,161.0** | **3.5335** | `NaN` | `NaN` | `NaN` | `NaN` | **3.04 mg/L** |
| **OPG — Darlington Nuclear** | **2,400.0** | `NaN` | `NaN` | `NaN` | `NaN` | `NaN` | `NaN` |
| **OPG — Pickering Nuclear** | **3,100.0** | `NaN` | `NaN` | `NaN` | `NaN` | `NaN` | `NaN` |
| **OPG — Atikokan Generating Station** | `NaN` | `NaN` | **5.21** | `NaN` | `NaN` | `NaN` | `NaN` |
| **OPG — Lennox G.S.** | `NaN` | `NaN` | `NaN` | **1.20** | `NaN` | `NaN` | `NaN` |
| **Bruce Power Inc. — GS A** | `NaN` | `NaN` | `NaN` | `NaN` | **0.054** | `NaN` | `NaN` |
| **Bruce Power Inc. — GS B** | `NaN` | `NaN` | `NaN` | `NaN` | `NaN` | **0.120** | `NaN` |

### Physical Concentration Derivation
At Bruce Power Development Services, both mass discharge load and hydraulic discharge rate are reported simultaneously:
$$\text{Concentration } (\text{mg/L}) = \frac{\text{Load } (\text{kg/d}) \times 1,000 \text{ g/kg} \times 1,000 \text{ mg/g}}{\text{Flow } (\text{m}^3\text{/d}) \times 1,000 \text{ L/m}^3} = \frac{3.5335 \times 1,000}{1,161} = \mathbf{3.0435 \text{ mg/L}}$$
This confirms that the treated effluent discharge is well within typical regulatory limits for thermal power plant outfalls (Ontario MISA standard: $< 15$–$25$ mg/L BOD).

---

## 4. Multi-Dataset Comparative Analysis

| Feature / Dimension | Dataset 1: Lab Study Table | Dataset 2: Bangladesh Textile Survey | Dataset 3: Ontario MISA Power |
| :--- | :--- | :--- | :--- |
| **Source Files** | `extracted_water_quality_samples_v2.csv` | `bangladesh_textile_dataset.csv` | `misa_electric_power_generation_2024.csv` |
| **Target Sector** | Chemical / Dye Synthesis | Export Textile & Dyeing Mills | Electric Power Generation (Nuclear/Thermal) |
| **Sample Size** | $N = 2$ streams | $N = 15$ factories | $N = 8$ records ($7$ facilities) |
| **Wastewater Stream** | **Raw untreated influent** | **Treated discharge effluent** | **Treated process outfall** |
| **Organic Load** | High COD ($4,800 - 8,800$ mg/L) | Moderate COD ($28 - 216$ mg/L) | Low BOD ($3.04$ mg/L, $3.53$ kg/d) |
| **Hydraulic Flow** | Unmeasured (`NaN`) | $260 - 2,815$ Million Liters | $1,161 - 3,100 \text{ m}^3\text{/d}$ |
| **Production Context** | None | Full ($2.4\text{M} - 41.5\text{M}$ kg) | Power generation MW (not reported) |
| **Power Telemetry** | None | Full ($4.2\text{M} - 65.5\text{M}$ kWh) | None (generating facility) |
| **Format** | Transposed parameter table | Standard observation table | Long / EAV regulatory report |
| **Training Feasibility** | **Halted** ($N=2 < 10$) | **Trainable** ($N=15 \ge 10$, `strategy: random`) | **Halted** ($N=8 < 10$, feature sparsity) |

---

## 5. Sector Heterogeneity & Model Training Principles

### Why Datasets Cannot Be Naively Merged Without Sector Conditioning
In industrial wastewater monitoring, combining disparate industrial sectors into a single unconditioned feature distribution introduces physical contradictions:
1. **Effluent Concentrations Differ by Orders of Magnitude:**
   - Raw dye manufacturing wastewater reaches COD $\sim 8,800$ mg/L.
   - Textile effluent reaches COD $\sim 88$ mg/L, BOD $\sim 22$ mg/L.
   - Nuclear power plant process water operates at BOD $\sim 3$ mg/L.
2. **Hydraulic vs. Organic Tradeoffs:**
   - Power stations discharge vast quantities of low-concentration cooling and boiler blowdown water ($>3,000 \text{ m}^3\text{/d}$).
   - Wet-processing textile mills discharge concentrated dye liquors with high dissolved solids and chemical oxygen demand.
3. **Engineering Guardrail:**
   EnviroTrust supports **sector-specific models** (e.g. `TEXTILE_WET_PROCESSING`, `POWER_GENERATION`) or **process-conditioned feature normalization**. Standalone training on MISA is currently halted by sample size safety checks ($N=8 < 10$).

---

## 6. Active Model Training on Authentic Industrial Data

EnviroTrust can be trained on the authentic Bangladesh industrial dataset using the following command:

```bash
cd ml-service
.venv/bin/python train.py --data-path data/raw/bangladesh_textile_dataset.csv --strategy random
```

### Training Pipeline Execution Results
- **Training Samples:** 10 factories (80% development split after 2-sample held-out reservation).
- **Validation Samples:** 3 normal factories (used strictly for empirical percentile threshold calibration).
- **Active Features (6):** `COD_mg_L`, `BOD_mg_L`, `electricity_consumption`, `production`, `water_flow`, `extracted_groundwater`.
- **Learned Envelopes:** 6 feature boundaries and 1 ratio band (`BOD_mg_L / COD_mg_L`).
- **Calibrated Threshold:** `0.0314` (99th percentile of normal validation scores).

### Authentic Model Detection on Known Facilities
- **Factory `F12` (Anomaly):** Score `+0.1308` ($> 0.0314$ threshold). Flagged as `SUSPICIOUS` by both Isolation Forest and physical envelope checks (extreme scale $+5\times$ mean, unusually low COD/BOD dilution signature).
- **Factory `F4` (Nominal Inlier):** Score `-0.0929` ($< 0.0314$ threshold). Classified as `CONSISTENT` with zero rule violations.
