# Traffic Pattern Forecasting

Forecasts hourly vehicle counts at road junctions using two complementary approaches: an XGBoost gradient boosting model with hand-crafted lag and rolling features, and a PyTorch LSTM with autoregressive test inference. Built as a full end-to-end pipeline covering EDA, feature engineering, model training, and cross-model comparison.

---

## Problem

Given hourly vehicle count history across four road junctions (January 2015 to June 2017), predict vehicle counts for each junction across a 5-month test horizon (July to November 2017 — 2,952 hourly predictions per junction).

The challenge is not a single series forecast. The four junctions have distinct volume levels, daily patterns, and seasonal behaviour. Junction 1 is dominant at 3–4× the volume of others with a strong weekend drop. Junction 3 has a seasonal spike in months 4–6. Junction 4 has only 5 months of training data. A model must handle this heterogeneity without overfitting to any single junction's behaviour.

---

## Dataset

| Property | Value |
|---|---|
| Train rows | 48,120 |
| Junctions | 4 |
| Granularity | Hourly |
| Train period | Jan 2015 – Jun 2017 |
| Test period | Jul 2017 – Nov 2017 |
| Target | Vehicles per hour |
| Null values | Zero (train and test) |

---

## Pipeline

### 1. EDA
- Confirmed zero null values in both train and test
- 2-week time series plot: Junction 1 dominant, clear daily cycles
- Hourly pattern: Junction 1 bimodal (peaks ~10AM and ~8PM); all junctions trough at 4–6AM
- Day-of-week pattern: Junction 1 weekend median ~28 vs weekday median ~45; J2/J3/J4 nearly flat
- Monthly pattern: Junction 1 peaks in June then drops; Junction 3 spikes in months 4–6
- Weekend vs weekday boxplot: Junction 1 shows widest weekday variance with outliers above 150 vehicles/hr

### 2. Preprocessing
- Confirmed no imputation needed
- Sorted by `Junction` and `DateTime` to ensure correct temporal ordering per junction

### 3. Feature Engineering

**Time-based features:**
- `hour`, `dayofweek`, `day`, `month`, `year`, `quarter`

**Binary flags:**
- `is_weekend` — Saturday and Sunday
- `is_night` — 10PM to 6AM
- `is_rush` — 7–9AM and 5–8PM

**Cyclical encodings** (sin/cos on hour, day of week, month):
- Prevents the model from treating hour 23 and hour 0 as 23 units apart

**Lag features** (computed per junction using `groupby` to prevent cross-junction leakage):
- `lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_168`

**Rolling statistics** (`shift(1)` applied before rolling to prevent target leakage):
- `roll_mean_3`, `roll_mean_6`, `roll_mean_24`
- `roll_std_3`, `roll_std_6`, `roll_std_24`

Total features: **27**

### 4. Models

#### XGBoost

| Config | Value |
|---|---|
| Estimators | 1,000 (early stopping at 50 rounds) |
| Learning rate | 0.05 |
| Max depth | 6 |
| Subsample | 0.8 |
| Colsample by tree | 0.8 |
| Min child weight | 3 |
| Val split date | 2017-05-01 |

Training stopped early at **203 trees**. Validation RMSE dropped from 28.48 at round 0 to **4.982** at best round, with MAE of **2.926**.

Test inference: lag features approximated using per-junction hourly averages from the training set (static proxy approach).

#### LSTM (PyTorch)

| Config | Value |
|---|---|
| Layers | 2 |
| Hidden size | 128 |
| Dropout | 0.3 |
| Sequence length | 48 hours |
| Input features | 9 (scaled vehicle count + 8 time features) |
| Epochs | 100 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Gradient clipping | 1.0 |
| Scaler | Per-junction MinMaxScaler, saved with checkpoint |

Architecture: `LSTM → LayerNorm(128) → Linear(128, 1)`. ~270K trainable parameters.

Test inference: **autoregressive** — each prediction is fed back as the next step's input across 2,952 steps per junction.

---

## Results

### Validation
| Model | RMSE | MAE |
|---|---|---|
| XGBoost | 4.982 | 2.926 |
| LSTM | Per-junction (see below) | — |

### Test Set Forecast Comparison (Jul–Nov 2017)

| Junction | XGB avg (veh/hr) | LSTM avg (veh/hr) | XGB max | LSTM max |
|---|---|---|---|---|
| J1 | 43.9 | 57.4 | 68 | 89 |
| J2 | 15.4 | 15.3 | 28 | 25 |
| J3 | 14.8 | 62.0 | 30 | 100 |
| J4 | 7.0 | 13.1 | 12 | 20 |

**J2:** both models agree closely — strongest validation of both approaches.

**J3:** largest divergence. XGBoost's static proxy lags systematically underestimate J3's test-period traffic. The LSTM's autoregressive rollout maintains a realistic count trajectory by relying on its own predictions rather than a static average. This is the key limitation of the XGBoost approach at test time.

**J1:** both models track the daily cycle. XGBoost skews lower due to the proxy lag effect.

**Feature importance (XGBoost):** `lag_1`, `lag_24`, and `lag_168` are the top three features, confirming that recent history and weekly periodicity are the dominant signals. Cyclical hour encoding ranks above raw integer hour.

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
xgboost
torch
```

Install:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost torch
```

---

## Running the Project

```bash
# 1. EDA, preprocessing, feature engineering
jupyter notebook notebooks/eda_preprocessing_featureEngineering.ipynb

# 2. Train XGBoost and generate submission_xgb.csv
jupyter notebook notebooks/xgboost_model.ipynb

# 3. Train LSTM and save lstm_tuned.pt
jupyter notebook notebooks/LSTM_model.ipynb

# 4. Generate comparison plots and submission_lstm.csv
python compare.py
```

Outputs: `submission_xgb.csv`, `submission_lstm.csv`, `forecast_per_junction.png`, `xgb_vs_lstm_comparison.png`.

---

## Key Design Decisions

**Why groupby for lags?** Without `groupby('Junction')`, a `shift(1)` on the full dataframe bleeds the last row of one junction into the first row of the next. Silent leakage. Every lag and rolling operation is scoped per junction.

**Why cyclical encoding?** Raw integer hour encoding implies hour 23 and hour 0 are 23 units apart. Sin/cos encoding places them adjacent on a unit circle, giving the model correct distance information for periodic features.

**Why autoregressive LSTM inference?** The test set has no ground-truth vehicle counts, so lag features cannot be computed. The LSTM sidesteps this by feeding each predicted value back as the next input, generating 2,952 steps per junction without needing external lag approximations.

---

## Internship Context

Built during a 6-week industrial internship facilitated by upskill Campus and The IoT Academy, in collaboration with UniConverge Technologies Pvt Ltd (UCT).