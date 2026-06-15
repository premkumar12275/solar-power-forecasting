# Solar Power Generation Forecasting Using Machine Learning

> Research project · MS in Artificial Intelligence · University of Texas at Austin

Comparative study of machine learning models for predicting AC power output at two utility-scale solar photovoltaic plants, using real-world sensor data from the [Kaggle Solar Power Generation Dataset](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data).

---

## Overview

Accurate solar power forecasting is critical for grid balancing and energy market operations. Forecasting errors translate directly into balancing costs — estimated at hundreds of millions of euros annually across Europe. This project explores whether standard ML models trained on meteorological sensor data can reliably predict plant-level AC power output, and whether performance generalises across different plant locations.

Three models are compared across two utility-scale solar plants in India:
- **Linear Regression** — baseline
- **Random Forest** — ensemble method
- **XGBoost** — gradient boosting with SHAP explainability

---

## Key Results

| Plant | Model | RMSE (kW) | MAE (kW) | R² |
|-------|-------|-----------|----------|----|
| Plant 1 | Linear Regression | 1,022.4 | 788.7 | 0.9821 |
| Plant 1 | Random Forest | 656.6 | 424.1 | 0.9926 |
| Plant 1 | **XGBoost** | **614.7** | **417.5** | **0.9935** |
| Plant 2 | Linear Regression | 3,371.4 | 2,595.4 | 0.5824 |
| Plant 2 | Random Forest | 2,556.0 | 1,401.7 | 0.7600 |
| Plant 2 | **XGBoost** | **2,481.0** | **1,271.2** | **0.7738** |

**Key finding:** XGBoost achieves near-perfect prediction at Plant 1 (R² = 0.9935) but drops significantly at Plant 2 (R² = 0.7738) — revealing that model performance is highly site-dependent and cannot be assumed to generalise across locations.

---

## SHAP Feature Importance

SHAP (SHapley Additive Explanations) analysis was applied to the XGBoost models to interpret predictions at each plant.

| Feature | Plant 1 SHAP (kW) | Plant 2 SHAP (kW) |
|---------|-------------------|-------------------|
| IRRADIATION | 6,390.4 | 4,507.5 |
| TIME_OF_DAY | 217.9 | 287.6 |
| AMBIENT_TEMPERATURE | 127.4 | 311.7 |
| DAY_OF_MONTH | 62.5 | 621.2 |
| MODULE_TEMPERATURE | 57.6 | 230.5 |

**Notable:** `DAY_OF_MONTH` is nearly 10x more influential at Plant 2 than Plant 1 — likely due to panel soiling, monsoon-driven cloud patterns, or scheduled curtailment. This kind of operational signal would be invisible without explainability tooling.

---

## Visualisations

All figures are saved to the `output/` folder when the notebook is run.

| File | Description |
|------|-------------|
| `fig1_ac_power_distribution.png` | AC power output distribution — both plants |
| `fig2_time_series.png` | 34-day AC power time series |
| `fig3_irradiation_vs_acpower.png` | Irradiation vs AC power scatter |
| `fig4_hourly_pattern.png` | Average AC power by hour of day (diurnal pattern) |
| `fig5_correlation_heatmap.png` | Feature correlation heatmaps |
| `fig6_model_comparison.png` | RMSE / MAE / R² comparison across models |
| `fig7_predicted_vs_actual.png` | Predicted vs actual scatter — all models |
| `fig8_shap_importance.png` | SHAP feature importance — XGBoost both plants |
| `fig9_forecast_vs_actual_timeseries.png` | XGBoost forecast vs actual — test period |

---

## Dataset

- **Source:** [Solar Power Generation Data — Kaggle (Anikannal, 2020)](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Period:** May 15 – June 17, 2020 (34 days, 15-minute intervals)
- **Plants:** 2 utility-scale PV installations in India (~22 inverters each)
- **Files needed:** `Plant_1_Generation_Data.csv`, `Plant_2_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv`, `Plant_2_Weather_Sensor_Data.csv`

### Features Used

| Feature | Description |
|---------|-------------|
| IRRADIATION | Solar irradiation (W/m²) |
| AMBIENT_TEMPERATURE | Ambient air temperature (°C) |
| MODULE_TEMPERATURE | Solar panel surface temperature (°C) |
| TIME_OF_DAY | Continuous decimal time (e.g. 14:30 = 14.5) |
| HOUR | Hour of day (integer) |
| DAY_OF_WEEK | Day of week (0–6) |
| DAY_OF_MONTH | Day of month (1–31) |

**Target:** `AC_POWER` — aggregated plant-level AC output (kW)

**Excluded to prevent data leakage:** `DC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`

---

## Methodology

### Preprocessing
- Standardised inconsistent datetime formats across files (Plant 1 used DD-MM-YYYY; others used ISO)
- Aggregated per-inverter generation data to plant level (~68,000 → ~3,200 rows per plant)
- Merged generation and weather sensor data on `DATE_TIME` (inner join, no rows lost)
- Filtered to daytime only (irradiation > 0) to avoid trivially inflated metrics

### Validation Strategy
Temporal train/test split (80/20) — preserving chronological order to prevent future data leaking into training. Test period covers June 11–17, 2020.

### Models
- **Linear Regression** — least squares baseline
- **Random Forest** — 100 estimators, `random_state=42`
- **XGBoost** — 200 estimators, learning rate 0.05, max depth 6; SHAP via `TreeExplainer`

---

## Project Structure

```
solar-power-forecasting/
├── Solar_Power_Generation_Forecasting_Using_ML.ipynb   # Main notebook (Colab/Jupyter/VSCode)
├── solar_power_forecasting.py                          # Standalone Python script
├── Solar_Power_Generation_Forecasting_Using_ML.pdf     # Research paper
├── output/                                             # Generated figures (after running notebook)
│   ├── fig1_ac_power_distribution.png
│   ├── fig2_time_series.png
│   ├── fig3_irradiation_vs_acpower.png
│   ├── fig4_hourly_pattern.png
│   ├── fig5_correlation_heatmap.png
│   ├── fig6_model_comparison.png
│   ├── fig7_predicted_vs_actual.png
│   ├── fig8_shap_importance.png
│   └── fig9_forecast_vs_actual_timeseries.png
├── data/                                               # Dataset files (not included — download from Kaggle)
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
├── requirements.txt
└── README.md
```

---

## Setup & Usage

```bash
git clone https://github.com/premkumar12275/solar-power-forecasting.git
cd solar-power-forecasting
pip install -r requirements.txt
```

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
2. Place the 4 CSV files in a `data/` folder
3. Open `Solar_Power_Generation_Forecasting_Using_ML.ipynb` in Jupyter, VSCode, or Google Colab
4. Update the data file paths in **Step 2** of the notebook if needed (Colab vs local paths are both documented inline)
5. Run all cells — figures will be saved to `output/`

To run as a script instead:
```bash
python solar_power_forecasting.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Context

The domain knowledge behind this research draws on professional experience building real-time sensor data ingestion pipelines for renewable energy assets — including solar and wind farms — across Europe. The operational implications of forecasting errors (grid balancing costs, energy market penalties) were a direct motivation for the research questions explored here.

This project is part of an MS in Artificial Intelligence at the University of Texas at Austin.

---

## Author

**Prem Kumar Katta** 
