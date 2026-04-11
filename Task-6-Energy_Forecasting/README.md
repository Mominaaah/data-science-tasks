# Task 3 — Energy Consumption Time Series Forecasting

Forecast short-term household energy usage using historical power consumption data. Three models are compared: ARIMA, Prophet, and XGBoost.

---

## Dataset

- **Source:** [UCI — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Resolution:** 1-minute recordings resampled to hourly
- **Period:** December 2006 to November 2010
- **Target:** `Global_active_power` (kilowatts)

---

## Project Structure

```
Task-3-Energy-Forecasting/
│
├── energy_forecasting.ipynb        # Main notebook
├── household_power_consumption.txt # Dataset (download separately)
└── README.md                       # This file
```

---

## How to Run

**Step 1 — Download the dataset manually**

Go to: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Download the ZIP, extract it, and place `household_power_consumption.txt` in the same folder as the notebook.

**Step 2 — Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels prophet xgboost
```

**Step 3 — Run all cells top to bottom**

---

## What the Notebook Does

### Step 0–1 · Setup
Install all libraries and import them.

### Step 2–4 · Load & Prepare Data
- Load the raw CSV with `;` separator
- Combine `Date` and `Time` columns into a single datetime index
- Resample from 1-minute to hourly averages
- Forward-fill any missing hours

### Step 5 · EDA
- Daily average power trend over the full dataset
- Average usage by hour of day
- Average usage by day of week
- Power consumption distribution histogram

### Step 6 · Stationarity Test
- Augmented Dickey-Fuller (ADF) test to check if the series is stationary
- ACF and PACF plots to identify lag patterns

### Step 7 · Feature Engineering
Features created for XGBoost:

| Feature | Description |
|---|---|
| hour, dayofweek, month | Calendar time features |
| is_weekend | Binary flag |
| lag_1, lag_24, lag_168 | Previous 1hr, 1 day, 1 week values |
| rolling_mean_24 | 24-hour rolling average |
| rolling_mean_168 | 7-day rolling average |

### Step 8 · Train / Test Split
- **Train:** Everything except the last 7 days
- **Test:** Last 168 hours (7 days)

### Step 9 · ARIMA
- ARIMA(2,1,2) fitted on the last 30 days of training data
- Forecasts 168 steps ahead

### Step 10 · Prophet
- Facebook Prophet with daily and weekly seasonality
- Automatically handles trends and seasonal patterns

### Step 11 · XGBoost
- Gradient boosted trees trained on engineered features
- StandardScaler applied before training

### Step 12–14 · Results
- Actual vs Forecasted plots for all 3 models
- XGBoost feature importance chart
- Final MAE and RMSE comparison table and bar chart

---

## Metrics Used

| Metric | What it measures |
|---|---|
| MAE | Average absolute error in kilowatts |
| RMSE | Penalises large errors more — sensitive to big mistakes |

Lower values = better predictions.

---

## Skills Demonstrated

- Time series parsing and resampling
- Stationarity testing with ADF
- ACF / PACF analysis
- Time-based feature engineering with lag and rolling features
- Classical forecasting with ARIMA
- Decomposition forecasting with Prophet
- Machine learning forecasting with XGBoost
- Model evaluation using MAE and RMSE

---

## Part of

[Data Science Portfolio](../) — Task 3 of ongoing project series.
